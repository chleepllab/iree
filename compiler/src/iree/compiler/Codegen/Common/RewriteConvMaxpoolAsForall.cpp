//===----------------------------------------------------------------------===//
// RewriteConvMaxpoolAsForall
//
// Fuses a conv -> elementwise -> (re-pad) -> maxpool chain into ONE scf.forall
// so the full-resolution conv output is produced per pooled-output tile, pooled
// immediately, and discarded — it is never materialized at conv resolution.
//
// WHAT IT DOES
//   1. matchConvMaxpoolFromRoot: starting from the op tagged with
//      `iree_dispatch.special_conv_chain` (== the maxpool, the root), walks back
//      through the maxpool's padded input (tensor.insert_slice) to its
//      elementwise epilogue and to the conv, recording conv/maxpool kernel,
//      stride, pad, and shapes.
//   2. rewriteConvMaxpoolAsForall: builds an scf.forall over the maxpool output
//      tiles [(N,)C,OH,OW] (workgroup tile bounded by kMaxpoolTileBudgetBytes =
//      16384). Inside each tile, the conv+epilogue is evaluated one CHANNEL
//      CHUNK at a time (pickConvEpilogueChunkSizes, kVectorTileBudgetBytes =
//      4096), each chunk pooled into the tile and dropped, so the resident
//      conv-resolution buffer stays tiny. The conv activation is first zero-
//      padded so every tile's window is a static, in-bounds read.
//   3. Unlike RewriteConvChainAsForall, this pass writes NO vector schedule.
//      It leaves vectorization to GenericVectorization and only bounds the size
//      of each generic (via the channel/height chunking above) and keeps the
//      maxpool a plain elementwise `maximumf` over a zero guard band, so the
//      backend vectorizes each op without any lowering_config from us.
//
// WORKED EXAMPLE
//   The rewritten dispatch is the ResNet-18 stem:
//     conv    : act [1,3,224,224], filter [64,3,7,7], pad 3, stride 2 -> [1,64,112,112]
//     relu    (epilogue)                                              -> [1,64,112,112]
//     maxpool : [1,64,112,112], kernel 3x3, pad 1, stride 2           -> [1,64,56,56]
//   After the pass: one scf.forall over the [64,56,56] pooled output (CHW). For
//   each output tile and each channel chunk, the conv reads its activation
//   window, produces the corresponding 112-scale rows, relu, then 3x3/stride-2
//   maxpool into the tile — the [1,64,112,112] conv output never exists whole.
//   The [1,3,224,224] activation is pre-padded to H=230 (224 + conv pad 3 on
//   each side already applied upstream) and trailing-zero-padded so the last
//   OH=56 tile reads in-bounds (see the inline neededActH=229 <= 230 example).
//===----------------------------------------------------------------------===//
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>

#define DEBUG_TYPE "rewrite-conv-maxpool-as-forall"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REWRITECONVMAXPOOLASFORALLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static constexpr StringLiteral kSpecialFusionAttr =
    "iree_dispatch.special_conv_chain";
static constexpr StringLiteral kConvMaxpoolRewrittenAttr =
    "iree_codegen.conv_maxpool_rewritten";

static bool getDimCoeff(AffineExpr expr, unsigned dimPos, int64_t &coeff) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    if (dim.getPosition() == dimPos) {
      coeff = 1;
      return true;
    }
    return false;
  }
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (bin.getKind() != AffineExprKind::Mul)
      return false;
    AffineExpr lhs = bin.getLHS(), rhs = bin.getRHS();
    if (auto d = dyn_cast<AffineDimExpr>(lhs)) {
      if (d.getPosition() == dimPos)
        if (auto c = dyn_cast<AffineConstantExpr>(rhs)) {
          coeff = c.getValue();
          return true;
        }
    }
    if (auto d = dyn_cast<AffineDimExpr>(rhs)) {
      if (d.getPosition() == dimPos)
        if (auto c = dyn_cast<AffineConstantExpr>(lhs)) {
          coeff = c.getValue();
          return true;
        }
    }
  }
  return false;
}

static FailureOr<std::pair<int64_t, int64_t>>
getTwoTermAffineCoeffs(AffineExpr expr, unsigned spatialDim,
                       unsigned kernelDim) {
  auto sum = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!sum || sum.getKind() != AffineExprKind::Add)
    return failure();
  int64_t sc = 0, kc = 0;
  bool lhsS = getDimCoeff(sum.getLHS(), spatialDim, sc);
  bool rhsK = getDimCoeff(sum.getRHS(), kernelDim, kc);
  if (lhsS && rhsK)
    return std::make_pair(sc, kc);
  bool rhsS = getDimCoeff(sum.getRHS(), spatialDim, sc);
  bool lhsK = getDimCoeff(sum.getLHS(), kernelDim, kc);
  if (rhsS && lhsK)
    return std::make_pair(sc, kc);
  return failure();
}

// ---------------------------------------------------------------------------
// getOperandSliceFromIndexingMap — projected-permutation maps only.
// ---------------------------------------------------------------------------
static LogicalResult getOperandSliceFromIndexingMap(
    AffineMap operandMap, ArrayRef<OpFoldResult> loopOffsets,
    ArrayRef<OpFoldResult> loopSizes,
    SmallVectorImpl<OpFoldResult> &operandOffsets,
    SmallVectorImpl<OpFoldResult> &operandSizes,
    SmallVectorImpl<OpFoldResult> &operandStrides) {
  operandOffsets.clear();
  operandSizes.clear();
  operandStrides.clear();
  for (AffineExpr expr : operandMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      LLVM_DEBUG({
        llvm::dbgs() << "[getOperandSlice] only projected permutation maps "
                        "are supported\n";
      });
      return failure();
    }
    unsigned dimPos = dimExpr.getPosition();
    if (dimPos >= loopOffsets.size()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[getOperandSlice] dim position out of range\n";
      });
      return failure();
    }
    operandOffsets.push_back(loopOffsets[dimPos]);
    operandSizes.push_back(loopSizes[dimPos]);
    operandStrides.push_back(
        IntegerAttr::get(IndexType::get(operandMap.getContext()), 1));
  }
  return success();
}

static SmallVector<OpFoldResult> withBatch(bool hasBatch, OpFoldResult batchVal,
                                           ArrayRef<OpFoldResult> rest) {
  SmallVector<OpFoldResult> out;
  if (hasBatch)
    out.push_back(batchVal);
  out.append(rest.begin(), rest.end());
  return out;
}

static SmallVector<int64_t> withBatchShape(bool hasBatch, int64_t batchVal,
                                           ArrayRef<int64_t> rest) {
  SmallVector<int64_t> out;
  if (hasBatch)
    out.push_back(batchVal);
  out.append(rest.begin(), rest.end());
  return out;
}

// ---------------------------------------------------------------------------
// Pattern predicates
// ---------------------------------------------------------------------------
static bool isElementwiseGeneric(linalg::GenericOp op) {
  return op && op.getNumReductionLoops() == 0;
}

static bool isMaxpoolLikeGeneric(linalg::GenericOp op) {
  if (!op)
    return false;
  if (op.getNumReductionLoops() != 2)
    return false;
  if (op.getNumDpsInputs() < 1 || op.getNumDpsInputs() > 2)
    return false;
  if (op.getNumResults() != 1)
    return false;
  bool hasMax = false;
  op.getBlock()->walk([&](Operation *inner) {
    if (isa<arith::MaximumFOp>(inner))
      hasMax = true;
  });
  return hasMax;
}

static bool isConvLikeGeneric(linalg::GenericOp op) {
  if (!op)
    return false;
  if (op.getNumDpsInputs() != 2 || op.getNumResults() != 1)
    return false;
  if (op.getNumReductionLoops() < 3)
    return false;
  auto act = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto flt = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
  if (!act || !flt || flt.getRank() != 4)
    return false;
  return act.getRank() == 3 || act.getRank() == 4;
}

// ---------------------------------------------------------------------------
// Parameter extraction from indexing maps
// ---------------------------------------------------------------------------

// Maxpool generic loop dims: (n?, c, oh, ow, kh, kw)
// Input map result[B+1] = oh*strideH + kh, result[B+2] = ow*strideW + kw.
// If no batch:
//    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2 + d3, d2 * 2 + d4)>
//    affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
//    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
static FailureOr<std::pair<int64_t, int64_t>>
getMaxpoolStridesFromMap(linalg::GenericOp mp, bool hasBatch) {
  SmallVector<AffineMap> maps = mp.getIndexingMapsArray();
  if (maps.empty())
    return failure();
  AffineMap inputMap = maps[0];
  int64_t B = hasBatch ? 1 : 0;
  if (inputMap.getNumResults() != 3 + B)
    return failure();

  // strideH
  auto hCoeffs = getTwoTermAffineCoeffs(inputMap.getResult(B + 1),
                                        /*oh=*/B + 1, /*kh=*/B + 3);
  // strideW
  auto wCoeffs = getTwoTermAffineCoeffs(inputMap.getResult(B + 2),
                                        /*ow=*/B + 2, /*kw=*/B + 4);
  if (failed(hCoeffs) || failed(wCoeffs))
    return failure();
  return std::make_pair(hCoeffs->first, wCoeffs->first);
}

// Conv generic loop dims: (n?, oc, oh, ow, ic, kh, kw)
// B = hasBatch ? 1 : 0.
// Input map result[B+1] = oh*strideH + kh*dilH, result[B+2] = ow*strideW +
// kw*dilW.
struct ConvParams {
  int64_t strideH, strideW, dilH, dilW;
};
static FailureOr<ConvParams> getConvParamsFromMap(linalg::GenericOp conv,
                                                   bool hasBatch) {
  SmallVector<AffineMap> maps = conv.getIndexingMapsArray();
  if (maps.empty())
    return failure();
  AffineMap inputMap = maps[0];
  int64_t B = hasBatch ? 1 : 0;
  if (inputMap.getNumResults() != 3 + B)
    return failure();
  auto hCoeffs = getTwoTermAffineCoeffs(inputMap.getResult(B + 1),
                                        /*oh=*/B + 1, /*kh=*/B + 4);
  auto wCoeffs = getTwoTermAffineCoeffs(inputMap.getResult(B + 2),
                                        /*ow=*/B + 2, /*kw=*/B + 5);
  if (failed(hCoeffs) || failed(wCoeffs))
    return failure();
  return ConvParams{hCoeffs->first, wCoeffs->first, hCoeffs->second,
                    wCoeffs->second};
}

// ---------------------------------------------------------------------------
// Match struct
// ---------------------------------------------------------------------------
struct ConvMaxpoolMatch {
  linalg::GenericOp genericConv;
  linalg::GenericOp epilogue;
  tensor::InsertSliceOp insertSlice;
  linalg::GenericOp genericMaxpool;

  bool hasBatch = false;
  int64_t N = 1;

  // Maxpool params.
  int64_t mpKH = 0, mpKW = 0;
  int64_t mpStrideH = 0, mpStrideW = 0;
  int64_t mpPadH = 0, mpPadW = 0;

  // Conv params.
  int64_t convKH = 0, convKW = 0;
  int64_t convStrideH = 0, convStrideW = 0;
  int64_t convDilH = 0, convDilW = 0;

  // Tensor shapes (excluding any batch dimension).
  int64_t CIn = 0, COut = 0;
  int64_t H_epi = 0, W_epi = 0; // conv output shape
  int64_t OH = 0, OW = 0;

  Operation *maxpoolOp() { return genericMaxpool.getOperation(); }
};

// ---------------------------------------------------------------------------
// Pattern matching
// ---------------------------------------------------------------------------
static FailureOr<ConvMaxpoolMatch>
matchConvMaxpoolFromRoot(Operation *rootOp) {
  ConvMaxpoolMatch match;

  match.genericMaxpool = dyn_cast<linalg::GenericOp>(rootOp);
  if (!isMaxpoolLikeGeneric(match.genericMaxpool)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[matchConvMaxpool] root is not a maxpool-like op\n";
    });
    return failure();
  }

  // Maxpool output shape: rank 3 (CHW) or rank 4 (NCHW).
  auto mpOutType =
      dyn_cast<RankedTensorType>(rootOp->getResult(0).getType());
  if (!mpOutType || (mpOutType.getRank() != 3 && mpOutType.getRank() != 4)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[matchConvMaxpool] unexpected maxpool result rank\n";
    });
    return failure();
  }
  match.hasBatch = mpOutType.getRank() == 4;
  int64_t B = match.hasBatch ? 1 : 0;
  match.N = match.hasBatch ? mpOutType.getShape()[0] : 1;
  match.OH = mpOutType.getShape()[B + 1];
  match.OW = mpOutType.getShape()[B + 2];

  {
    auto strides = getMaxpoolStridesFromMap(match.genericMaxpool, match.hasBatch);
    if (failed(strides))
      return failure();
    match.mpStrideH = strides->first;
    match.mpStrideW = strides->second;
    if (match.genericMaxpool.getNumDpsInputs() < 2)
      return failure();
    auto kType = dyn_cast<RankedTensorType>(
        match.genericMaxpool.getDpsInputs()[1].getType());
    if (!kType || kType.getRank() != 2)
      return failure();
    match.mpKH = kType.getShape()[0];
    match.mpKW = kType.getShape()[1];
  }

  // InsertSlice (the padded activation fed into maxpool input[0]).
  Value mpIn = match.genericMaxpool.getDpsInputs()[0];
  match.insertSlice = mpIn.getDefiningOp<tensor::InsertSliceOp>();
  if (!match.insertSlice) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "[matchConvMaxpool] maxpool input is not a tensor.insert_slice\n";
    });
    return failure();
  }

  // Padding = static H/W offsets of the insert_slice (offset[B], offset[B+1]).
  SmallVector<OpFoldResult> mixedOffsets = match.insertSlice.getMixedOffsets();
  if (static_cast<int64_t>(mixedOffsets.size()) != 3 + B)
    return failure();
  auto toStatic = [](OpFoldResult ofr, int64_t &out) -> bool {
    auto attr = dyn_cast<Attribute>(ofr);
    if (!attr)
      return false;
    auto iAttr = dyn_cast<IntegerAttr>(attr);
    if (!iAttr)
      return false;
    out = iAttr.getInt();
    return true;
  };
  if (!toStatic(mixedOffsets[B + 1], match.mpPadH) ||
      !toStatic(mixedOffsets[B + 2], match.mpPadW)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[matchConvMaxpool] dynamic maxpool padding not supported\n";
    });
    return failure();
  }

  // Epilogue (source of the insert_slice).
  match.epilogue =
      match.insertSlice.getSource().getDefiningOp<linalg::GenericOp>();
  if (!isElementwiseGeneric(match.epilogue)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[matchConvMaxpool] insert_slice source is not "
                      "elementwise generic\n";
    });
    return failure();
  }

  // Conv (one input to the epilogue), same rank family as the maxpool.
  for (Value inp : match.epilogue.getDpsInputs()) {
    if (auto gc = inp.getDefiningOp<linalg::GenericOp>()) {
      if (isConvLikeGeneric(gc)) {
        auto actType = cast<RankedTensorType>(gc.getDpsInputs()[0].getType());
        if ((actType.getRank() == 4) == match.hasBatch) {
          match.genericConv = gc;
          break;
        }
      }
    }
  }
  if (!match.genericConv) {
    LLVM_DEBUG({
      llvm::dbgs() << "[matchConvMaxpool] no matching conv found as input to "
                      "epilogue\n";
    });
    return failure();
  }

  // Extract conv params.
  {
    auto params = getConvParamsFromMap(match.genericConv, match.hasBatch);
    if (failed(params))
      return failure();
    match.convStrideH = params->strideH;
    match.convStrideW = params->strideW;
    match.convDilH = params->dilH;
    match.convDilW = params->dilW;
    auto fType = dyn_cast<RankedTensorType>(
        match.genericConv.getDpsInputs()[1].getType());
    if (!fType || fType.getRank() != 4)
      return failure();
    match.convKH = fType.getShape()[2];
    match.convKW = fType.getShape()[3];
    auto inType = dyn_cast<RankedTensorType>(
        match.genericConv.getDpsInputs()[0].getType());
    if (!inType || inType.getRank() != 3 + B)
      return failure();
    match.CIn = inType.getShape()[B];
  }

  // Epilogue output = conv output shape.
  auto epiType =
      dyn_cast<RankedTensorType>(match.epilogue.getResult(0).getType());
  if (!epiType || epiType.getRank() != 3 + B)
    return failure();
  match.COut = epiType.getShape()[B];
  match.H_epi = epiType.getShape()[B + 1];
  match.W_epi = epiType.getShape()[B + 2];

  return match;
}

// ---------------------------------------------------------------------------
// Read distribution tile sizes from lowering_config on the maxpool op.
// Expects at least 3 values: [c, oh, ow] (any batch dim is always processed
// in full, never distributed).
// ---------------------------------------------------------------------------
static FailureOr<SmallVector<int64_t>>
getMaxpoolDistributionTiles(Operation *mpOp) {
  // Use the backend-agnostic lowering-config interface so this works for any
  // config attr (CPU/GPU/...) that carries workgroup distribution tiles.
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getLoweringConfig(mpOp);
  if (!loweringConfig)
    return failure();

  // The distribution tile sizes are the workgroup tile sizes; the interface
  // hands them back as a plain SmallVector<int64_t> (empty when unset).
  SmallVector<int64_t> vals = loweringConfig.getWorkgroupTileSizes();
  if (vals.size() < 3)
    return failure();

  return vals;
}

// ---------------------------------------------------------------------------
// Inner vector-tile chunking for the fused conv+epilogue computation.
//
// GenericVectorizationPass vectorizes each linalg.generic at its full static
// (or masked-dynamic) extent. Without further subdivision, a single forall
// tile's conv+epilogue op can be arbitrarily large (bounded only by the
// maxpool's kernel/stride and the workgroup tile size), which can blow past
// the backend's stack-allocation limit for a single vectorized value. Rather
// than depend on the generic LLVMCPUTileAndFuseProducerConsumerPass to
// subdivide this hand-built IR (it does not: our cloned ops carry only
// vector-level lowering_config, never a workgroup-tiling-level entry, so it
// never finds an anchor here), pick channel/height chunk sizes ourselves and
// loop over them explicitly below.
// ---------------------------------------------------------------------------
// Conservative on purpose: several same-sized buffers (the epilogue
// accumulator, the conv/epilogue chunk scratch tensors, the maxpool output
// tile, ...) coexist per function once bufferized/hoisted to the entry
// block, and the backend's 32768-byte cap applies to their *sum*, not to
// any single one of them.
static constexpr int64_t kVectorTileBudgetBytes = 4096;

// Byte budget for the per-workgroup maxpool OUTPUT tile [(N,) TC, TOH, TOW].
// Because the maxpool is folded into the channel-chunk loop (each conv+epilogue
// channel chunk is pooled and discarded immediately, see rewriteConvMaxpool),
// this small output tile is what bounds the workgroup tile size.
static constexpr int64_t kMaxpoolTileBudgetBytes = 16384;

// kFullChannelEpiBudget
// bounds that resident conv-output window (kept < the 32768 stack cap with room
// for the coexisting conv scratch / maxpool tiles).
static constexpr bool kFullChannel = false;
static constexpr int64_t kFullChannelEpiBudget = 16384;

// Largest divisor of `total` that is <= `maxVal` (result is always >= 1).
static int64_t largestDivisorAtMost(int64_t total, int64_t maxVal) {
  maxVal = std::min(maxVal, total);
  for (int64_t d = maxVal; d >= 1; --d) {
    if (total % d == 0)
      return d;
  }
  return 1;
}

// Picks (chunkC, chunkH) such that the conv+epilogue output chunk
// [N, chunkC, chunkH, local_W] stays within the per-op vector budget.
//
// Channel is shrunk first in divisors of TC, then height in divisors of
// local_H, until the budget is met or both bottom out at 1.
static std::pair<int64_t, int64_t> pickConvEpilogueChunkSizes(
    int64_t N, int64_t TC, int64_t /*CIn*/, int64_t local_H, int64_t local_W,
    int64_t /*convInHFull*/, int64_t /*convStrideW*/, int64_t /*convKW*/,
    int64_t /*convDilW*/, int64_t elemBytes, bool forceFullChannel = false) {
  auto fits = [&](int64_t c, int64_t h) {
    int64_t outBytes = N * c * h * local_W * elemBytes;
    return outBytes <= kVectorTileBudgetBytes;
  };
  // Shrink channel first, to a DIVISOR of TC, until it fits (bottoms out at 1).
  int64_t chunkC = TC;
  if (!forceFullChannel)
    while (chunkC > 1 && !fits(chunkC, local_H))
      chunkC = largestDivisorAtMost(TC, chunkC / 2);
  int64_t chunkH = local_H;
  while (chunkH > 1 && !fits(chunkC, chunkH))
    chunkH = largestDivisorAtMost(local_H, chunkH / 2);
  return {chunkC, chunkH};
}

// ---------------------------------------------------------------------------
// Main rewrite: build the forall that fuses conv -> epilogue -> pad -> maxpool.
// ---------------------------------------------------------------------------
static LogicalResult rewriteConvMaxpoolAsForall(IRRewriter &rewriter,
                                                 ConvMaxpoolMatch &match) {
  Operation *mpOp = match.maxpoolOp();
  Location loc = mpOp->getLoc();
  const bool hasBatch = match.hasBatch;

  // Tile sizes (C, OH, OW)
  int64_t TC = match.COut, TOH = 8, TOW = 8;  // defaults
  {
    auto tiles = getMaxpoolDistributionTiles(mpOp);
    if (succeeded(tiles) && tiles->size() >= 3) {
      // The maxpool loop order is [(n,) c, oh, ow, kh, kw], and the
      // distribution tile lists ALL of them (reduction dims kh,kw are 0). So
      // the three parallel tiles we want (c, oh, ow) start at `base = hasBatch`
      // and the trailing reduction zeros are ignored.
      //   no batch: [c, oh, ow, kh, kw]      -> base 0 -> (c,oh,ow)
      //   batch:    [n, c, oh, ow, kh, kw]   -> base 1 -> (c,oh,ow)
      SmallVector<int64_t> &tv = *tiles;
      size_t base = hasBatch ? 1 : 0;
      int64_t tc = tv[base], toh = tv[base + 1], tow = tv[base + 2];
      if (tc > 0)
        TC = tc;
      if (toh > 0)
        TOH = toh;
      if (tow > 0)
        TOW = tow;
    }
  }

  // Too larger tile that may exceed stack allocation limit.
  if (TOH >= match.OH)
    TOH = 8;
  if (TOW >= match.OW)
    TOW = 8;

  if (match.OH % TOH != 0 || match.OW % TOW != 0) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteConvMaxpool] output dims not divisible by "
                      "tile sizes\n";
    });
    return failure();
  }

  // Element type from the maxpool result.
  auto mpOutType =
      dyn_cast<RankedTensorType>(mpOp->getResult(0).getType());
  Type elemType = mpOutType.getElementType();
  auto zeroAttr = cast<TypedAttr>(rewriter.getZeroAttr(elemType));
  Value zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
  int64_t elemBytes = std::max<int64_t>(1, elemType.getIntOrFloatBitWidth() / 8);

  // Local padded buffer spatial extent (static, same for every tile).
  //   maxpool output tile (TOH x TOW) reads from a padded window of size
  //   ((TOH-1)*sH + kH) x ((TOW-1)*sW + kW) within the padded space.
  //
  // EXAMPLE: TOH=28, TOW=28, mpStride=2, mpK=3:
  //   local_H = (28-1)*2 + 3 = 57
  //   local_W = (28-1)*2 + 3 = 57
  auto computeLocal = [&](int64_t toh, int64_t tow) {
    return std::make_pair((toh - 1) * match.mpStrideH + match.mpKH,
                          (tow - 1) * match.mpStrideW + match.mpKW);
  };
  int64_t local_H, local_W;
  std::tie(local_H, local_W) = computeLocal(TOH, TOW);

  // The maxpool is folded into the per-channel-chunk loop below: each
  // conv+epilogue channel chunk is immediately pooled into the workgroup's
  // maxpool OUTPUT tile and then discarded, so NO full-TC
  // [(N,) TC, local_H, local_W] conv accumulator ever lives across the whole
  // tile. The only TC-sized buffer that survives is the maxpool output tile
  // [(N,) TC, TOH, TOW] -- roughly 4x smaller than the conv window it used to
  // require.
  // The resident conv window is bounded independently, per chunk, by 
  // pickConvEpilogueChunkSizes. Shrink TC first (keeping the spatial blocks
  // large for cache reuse and width vectorization), then TOH, then TOW, always
  // in divisors of the output extents so the forall tiles the output evenly.
  // For the small-filter convs this fusion targets, keeping the whole batch in
  // one tile reuses the (tiny) conv weights across all N images, which is
  // measured faster than per-image distribution. 
  const bool distributeBatch = false;
  const int64_t nb = distributeBatch ? 1 : match.N;

  // Bytes of the per-workgroup maxpool OUTPUT tile [(N,)TC,TOH,TOW].
  // EXAMPLE: nb=1, TC=16, TOH=28, TOW=28, elemBytes=4 (f32):
  //   mpBytes = 1*16*28*28*4 = 50176 bytes  (> 16384 budget -> must shrink)
  auto mpBytes = [&](int64_t tc, int64_t toh, int64_t tow) {
    return nb * tc * toh * tow * elemBytes;
  };
  // Shrink to fit the budget, one dim at a time, always to a DIVISOR of the
  // full output extent (so the forall still tiles the output evenly). Channel
  // shrinks first (keeps the spatial blocks large for cache reuse / width
  // vectorization), then height, then width..
  // EXAMPLE (continuing, budget=16384, COut=64,OW=OH=56):
  //   TC: 16 -> largestDivisorAtMost(64,8)=8 (1*8*28*28*4=25088 still >16384)
  //          ->  largestDivisorAtMost(64,4)=4 (1*4*28*28*4=12544 <=16384) stop
  //   so TC=4,TOH=28,TOW=28 here.
  while (TC > 1 && mpBytes(TC, TOH, TOW) > kMaxpoolTileBudgetBytes)
    TC = largestDivisorAtMost(match.COut, TC / 2);
  while (TOW > 1 && mpBytes(TC, TOH, TOW) > kMaxpoolTileBudgetBytes)
    TOW = largestDivisorAtMost(match.OW, TOW / 2);
  while (TOH > 1 && mpBytes(TC, TOH, TOW) > kMaxpoolTileBudgetBytes)
    TOH = largestDivisorAtMost(match.OH, TOH / 2);
  std::tie(local_H, local_W) = computeLocal(TOH, TOW);

  if (match.OH % TOH != 0 || match.OW % TOW != 0) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteConvMaxpool] output dims not divisible by "
                      "tile sizes\n";
    });
    return failure();
  }
  if (match.COut % TC != 0) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteConvMaxpool] output channels not divisible by "
                      "tile size\n";
    });
    return failure();
  }

  // Static size of the conv activation window needed to produce a full
  // local_H x local_W epilogue tile.
  //
  // EXAMPLE: local_W=57, convStride=2, convK=7, convDil=1:
  //   convInWFull = (57-1)*2 + (7-1)*1 + 1 = 119
  int64_t convInHFull = (local_H - 1) * match.convStrideH +
                        (match.convKH - 1) * match.convDilH + 1;
  int64_t convInWFull = (local_W - 1) * match.convStrideW +
                        (match.convKW - 1) * match.convDilW + 1;

  Value convActivation = match.genericConv.getDpsInputs()[0];
  Value convFilter = match.genericConv.getDpsInputs()[1];

  OpFoldResult zeroOfr = rewriter.getIndexAttr(0);
  OpFoldResult nOfr = rewriter.getIndexAttr(nb);

  // Every per-tile (and per-chunk) conv+epilogue computation built below 
  // must be statically shaped.
  {
    auto actType = cast<RankedTensorType>(convActivation.getType());
    int64_t B = hasBatch ? 1 : 0;
    if (actType.isDynamicDim(B + 1) || actType.isDynamicDim(B + 2)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteConvMaxpool] dynamic activation spatial dims "
                        "not supported\n";
      });
      return failure();
    }
    int64_t actH = actType.getDimSize(B + 1), actW = actType.getDimSize(B + 2);
    // The last tile starts at output row (OH-TOH); its epilogue-space start is
    // (OH-TOH)*mpStride - mpPad, floored at 0.
    // EXAMPLE: OH=56, TOH=28, mpStride=2, mpPad=1:
    //   maxEpiIh0 = max(0, (56-28)*2 - 1) = max(0, 55) = 55
    int64_t maxEpiIh0 = std::max<int64_t>(
        0, (match.OH - TOH) * match.mpStrideH - match.mpPadH);
    int64_t maxEpiIw0 = std::max<int64_t>(
        0, (match.OW - TOW) * match.mpStrideW - match.mpPadW);
    // Highest activation index any tile can touch: the last epilogue start,
    // Pad the activation up to this so every tile's window is a safe static 
    // in-bounds read.
    // EXAMPLE: neededActH = 55*2 + convInHFull(119) = 110 + 119 = 229
    int64_t neededActH = maxEpiIh0 * match.convStrideH + convInHFull;
    int64_t neededActW = maxEpiIw0 * match.convStrideW + convInWFull;
    // Trailing zeros to add. EXAMPLE: original conv input is 224 (pre-conv-pad
    // it becomes 230 in the dump); with actH=230, padH = max(0, 229-230) = 0.
    int64_t padH = std::max<int64_t>(0, neededActH - actH);
    int64_t padW = std::max<int64_t>(0, neededActW - actW);
    if (padH > 0 || padW > 0) {
      SmallVector<int64_t> paddedShape = withBatchShape(
          hasBatch, actType.getDimSize(0),
          {actType.getDimSize(B), actH + padH, actW + padW});
      auto paddedType = RankedTensorType::get(paddedShape, elemType);
      SmallVector<OpFoldResult> low(3 + B, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> high = withBatch(
          hasBatch, rewriter.getIndexAttr(0),
          {rewriter.getIndexAttr(0), rewriter.getIndexAttr(padH),
           rewriter.getIndexAttr(padW)});
      convActivation =
          tensor::PadOp::create(rewriter, loc, paddedType, convActivation,
                                low, high, zero, /*nofold=*/false);
    }
  }

  // Output init tensor for the forall result.
  // Every element is written by some tile's tensor.parallel_insert_slice, 
  // so no fill is needed here. 
  // A fill would also force GenericVectorizationPass to vectorize the *entire*
  // output tensor as one op, which is exactly the kind of oversized vector
  // this pass otherwise goes out of its way to avoid.
  Value outInit =
      tensor::EmptyOp::create(rewriter, loc, mpOutType.getShape(), elemType);

  // ----- Build scf.forall over [(n,) C, OH, OW] -----
  // The batch dim (when distributed) is the outermost, mapped.
  SmallVector<OpFoldResult> lbs, ubs, steps;
  if (distributeBatch) {
    lbs.push_back(rewriter.getIndexAttr(0));
    ubs.push_back(rewriter.getIndexAttr(match.N));
    steps.push_back(rewriter.getIndexAttr(nb));
  }
  lbs.append({rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
              rewriter.getIndexAttr(0)});
  ubs.append({rewriter.getIndexAttr(match.COut),
              rewriter.getIndexAttr(match.OH), rewriter.getIndexAttr(match.OW)});
  steps.append({rewriter.getIndexAttr(TC), rewriter.getIndexAttr(TOH),
                rewriter.getIndexAttr(TOW)});
  SmallVector<Attribute> mapping;
  if (distributeBatch)
    mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
        rewriter.getContext(), IREE::Codegen::WorkgroupId::IdZ, /*idx=*/1));
  mapping.append({IREE::Codegen::WorkgroupMappingAttr::get(
                      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdZ),
                  IREE::Codegen::WorkgroupMappingAttr::get(
                      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdY),
                  IREE::Codegen::WorkgroupMappingAttr::get(
                      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdX)});
  std::optional<ArrayAttr> mappingAttr = rewriter.getArrayAttr(mapping);

  rewriter.setInsertionPoint(mpOp);
  auto forallOp = scf::ForallOp::create(rewriter, loc, lbs, ubs, steps,
                                         ValueRange{outInit}, mappingAttr);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forallOp.getBody());

    auto ivs = forallOp.getInductionVars();
    size_t sp = distributeBatch ? 1 : 0;
    // Batch offset for global (activation/output) slices: the batch IV when
    // distributed, else 0.
    OpFoldResult n0Ofr =
        distributeBatch ? OpFoldResult(ivs[0]) : rewriter.getIndexAttr(0);
    Value c0 = ivs[sp], oh0 = ivs[sp + 1], ow0 = ivs[sp + 2];
    Value shared = forallOp.getRegionOutArgs().front();

    // -----------------------------------------------------------------------
    // [A] Compute the valid epilogue (= conv output) spatial window for this
    //     maxpool output tile.
    //
    //     In the padded space the tile reads rows:
    //       [oh0*sH,  oh0*sH + local_H - 1]
    //     Back in the unpadded epilogue space (subtract padH):
    //       epi_ih_start = oh0*sH - padH   (can be negative)
    //       epi_ih_end   = oh0*sH - padH + local_H   (exclusive)
    //     Clamp to [0, H_epi).
    // -----------------------------------------------------------------------
    Value cMpSH = arith::ConstantIndexOp::create(rewriter, loc, match.mpStrideH);
    Value cMpSW = arith::ConstantIndexOp::create(rewriter, loc, match.mpStrideW);
    Value cMpPH = arith::ConstantIndexOp::create(rewriter, loc, match.mpPadH);
    Value cMpPW = arith::ConstantIndexOp::create(rewriter, loc, match.mpPadW);
    Value cLocalH = arith::ConstantIndexOp::create(rewriter, loc, local_H);
    Value cLocalW = arith::ConstantIndexOp::create(rewriter, loc, local_W);
    Value cHEpi = arith::ConstantIndexOp::create(rewriter, loc, match.H_epi);
    Value cWEpi = arith::ConstantIndexOp::create(rewriter, loc, match.W_epi);
    Value c0idx = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // ---- Worked examples for the block below (mpStride=2, mpPad=1, local_W=17,
    //      W_epi=112). Two width tiles, step TOW=8 so ow0 in {0,8,16,...,48}:
    //
    //   BOUNDARY tile ow0=0:
    //     epi_iw_start = 0*2 - 1 = -1        (window starts 1 col left of pad)
    //     epi_iw0      = max(0,-1) = 0       (clamped read start in epilogue)
    //     epi_iw1      = min(112, -1+17) = 16
    //     valid_W      = 16 - 0 = 16         (only 16 real cols; col -1 is pad)
    //     insert_w     = max(0,-(-1)) = 1    (real data sits at local offset 1)
    //
    //   INTERIOR tile ow0=8:
    //     epi_iw_start = 8*2 - 1 = 15
    //     epi_iw0      = max(0,15) = 15
    //     epi_iw1      = min(112, 15+17) = 32
    //     valid_W      = 32 - 15 = 17        (full window, all real)
    //     insert_w     = max(0,-15) = 0      (no left padding)

    // epi_ih_start = oh0 * mpStrideH - mpPadH
    //   The top of this tile's pooling window, in unpadded epilogue (=conv
    //   output) coordinates. Negative when the window pokes into the maxpool's
    //   top zero-pad. EXAMPLE oh0=0: 0*2-1 = -1.
    Value epi_ih_start = arith::SubIOp::create(
        rewriter, loc, arith::MulIOp::create(rewriter, loc, oh0, cMpSH),
        cMpPH);
    Value epi_iw_start = arith::SubIOp::create(
        rewriter, loc, arith::MulIOp::create(rewriter, loc, ow0, cMpSW),
        cMpPW);

    // epi_ih0 = max(0, epi_ih_start): first REAL (non-pad) epilogue row read.
    // EXAMPLE oh0=0: max(0,-1)=0.
    Value epi_ih0 =
        arith::MaxSIOp::create(rewriter, loc, epi_ih_start, c0idx);
    Value epi_iw0 =
        arith::MaxSIOp::create(rewriter, loc, epi_iw_start, c0idx);

    // epi_ih1 = min(H_epi, epi_ih_start + local_H): exclusive end of the real
    // region, clamped to the epilogue height. 
    // EXAMPLE oh0=0: min(112,-1+57)=56.
    Value epi_ih1 = arith::MinSIOp::create(
        rewriter, loc,
        arith::AddIOp::create(rewriter, loc, epi_ih_start, cLocalH), cHEpi);
    Value epi_iw1 = arith::MinSIOp::create(
        rewriter, loc,
        arith::AddIOp::create(rewriter, loc, epi_iw_start, cLocalW), cWEpi);

    // valid_H = epi_ih1 - epi_ih0: count of REAL rows in this tile's window
    // (the rest are maxpool zero-pad). 
    // This dynamic bound is what the maxpool body compares against; everything 
    // outside reads as 0. 
    // EXAMPLE oh0=0: 56 - 0 = 56.
    Value valid_H =
        arith::SubIOp::create(rewriter, loc, epi_ih1, epi_ih0);
    Value valid_W =
        arith::SubIOp::create(rewriter, loc, epi_iw1, epi_iw0);

    // insert_h = max(0, -epi_ih_start) = max(0, mpPadH - oh0*mpStrideH):
    // how many pad rows precede the real data, i.e. the local offset at which
    // the real region begins inside the [0,local_H) tile. 
    // EXAMPLE oh0=0: max(0,-(-1))=1 (one pad row on top). 
    // Interior oh0=8: max(0,-15)=0.
    Value insert_h = arith::MaxSIOp::create(
        rewriter, loc,
        arith::SubIOp::create(rewriter, loc, c0idx, epi_ih_start), c0idx);
    Value insert_w = arith::MaxSIOp::create(
        rewriter, loc,
        arith::SubIOp::create(rewriter, loc, c0idx, epi_iw_start), c0idx);

    // -----------------------------------------------------------------------
    // [B] Conv input window.
    //     conv_in_iw_start = epi_iw0 * convStrideW. 
    //     It's always in-bounds thanks to the activation padding above, so no 
    //     per-tile clamping is needed. 
    //     The HEIGHT window is chunked, so its start is computed per
    //     (channel, height) chunk below.
    // -----------------------------------------------------------------------
    Value cConvSH =
        arith::ConstantIndexOp::create(rewriter, loc, match.convStrideH);
    Value cConvSW =
        arith::ConstantIndexOp::create(rewriter, loc, match.convStrideW);

    // conv_in_iw_start = epi_iw0 * convStrideW: the first activation column the
    // conv must read to produce conv-output column epi_iw0.
    Value conv_in_iw_start =
        arith::MulIOp::create(rewriter, loc, epi_iw0, cConvSW);
    // The corresponding height-window computation is done per (channel,
    // height) chunk below, since the conv+epilogue computation is chunked
    // along height to bound vector size.

    // -----------------------------------------------------------------------
    // [C-E] Conv input/filter extraction, conv computation, and epilogue
    //     computation for this forall tile — chunked over (channel, height)
    //     so that no single vectorized op created here can grow unboundedly
    //     large (GenericVectorizationPass vectorizes each linalg.generic at
    //     its full extent, and IREE's own second-level tiling pass does not
    //     apply to this hand-built IR — see pickConvEpilogueChunkSizes).
    //     Every chunk — and the un-chunked width dimension — is *statically*
    //     sized (chunkH x local_W, not the dynamic valid_H/valid_W), both to
    //     bound vector size and because a dynamically-shaped intermediate
    //     here would bufferize to a stack allocation this backend rejects
    //     outright.
    // -----------------------------------------------------------------------
    SmallVector<Value> epiInputs(match.epilogue.getDpsInputs());
    SmallVector<Value> epiInits(match.epilogue.getDpsInits());
    SmallVector<AffineMap> epiMaps = match.epilogue.getIndexingMapsArray();

    unsigned nEpiIn = epiInputs.size();
    unsigned nEpiOut = epiInits.size();
    if (nEpiOut != 1 || match.epilogue.getNumResults() != 1)
      return failure();

    ArrayRef<AffineMap> epiInMaps =
        ArrayRef<AffineMap>(epiMaps).take_front(nEpiIn);
    AffineMap epiOutMap =
        ArrayRef<AffineMap>(epiMaps).drop_front(nEpiIn).front();

    Value oldConvResult = match.genericConv.getResult(0);

    SmallVector<OpFoldResult> strides1(3 + (hasBatch ? 1 : 0),
                                       rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> filtStrides(4, rewriter.getIndexAttr(1));

    // (No full [(N,) TC, local_H, local_W] epilogue accumulator is built: each
    // channel chunk's conv+epilogue result lives only long enough to be pooled
    // below, in a small per-chunk [(N,) chunkC, local_H, local_W] buffer.)

    int64_t chunkC = 0, chunkH = 0, convInHChunkFull = 0;

    // Computes conv+epilogue for a single (cAbs:cAbs+cSize, hAbs:hAbs+chunkH)
    // chunk of this forall tile — always chunkH rows and the full static
    // local_W columns (never the dynamic valid_H/valid_W), safe because of
    // the activation padding above. `cAbs`/`hAbs` are absolute coordinates in
    // the original (unfused) conv/epilogue loop space; returns a null Value
    // on failure.
    auto computeEpilogueChunk = [&](Value cAbs, int64_t cSize,
                                    Value hAbs) -> Value {
      // First activation ROW for this height chunk: hAbs (an absolute
      // conv-output row) times convStrideH. The slice then reads
      // convInHChunkFull rows from there and the full convInWFull columns.
      Value convInIhStartChunk =
          arith::MulIOp::create(rewriter, loc, hAbs, cConvSH);

      SmallVector<OpFoldResult> actOffsets = withBatch(
          hasBatch, n0Ofr,
          {rewriter.getIndexAttr(0), convInIhStartChunk, conv_in_iw_start});
      SmallVector<OpFoldResult> actSizes = withBatch(
          hasBatch, nOfr,
          {rewriter.getIndexAttr(match.CIn),
           rewriter.getIndexAttr(convInHChunkFull),
           rewriter.getIndexAttr(convInWFull)});
      auto actSliceType = RankedTensorType::get(
          withBatchShape(hasBatch, nb,
                        {match.CIn, convInHChunkFull, convInWFull}),
          elemType);
      Value actSlice = tensor::ExtractSliceOp::create(
          rewriter, loc, actSliceType, convActivation, actOffsets, actSizes,
          strides1);

      SmallVector<OpFoldResult> filtOffsets = {
          cAbs, rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> filtSizes = {
          rewriter.getIndexAttr(cSize), rewriter.getIndexAttr(match.CIn),
          rewriter.getIndexAttr(match.convKH),
          rewriter.getIndexAttr(match.convKW)};
      auto filtSliceType = RankedTensorType::get(
          {cSize, match.CIn, match.convKH, match.convKW}, elemType);
      Value filtSlice = tensor::ExtractSliceOp::create(
          rewriter, loc, filtSliceType, convFilter, filtOffsets, filtSizes,
          filtStrides);

      SmallVector<OpFoldResult> convOutSizes = withBatch(
          hasBatch, nOfr,
          {rewriter.getIndexAttr(cSize), rewriter.getIndexAttr(chunkH),
           rewriter.getIndexAttr(local_W)});
      Value convInitEmpty =
          tensor::EmptyOp::create(rewriter, loc, convOutSizes, elemType);
      Value convInitFill =
          linalg::FillOp::create(rewriter, loc, zero, convInitEmpty)
              .getResult(0);

      // Clone the generic conv with new operands and statically-sized
      // output.
      SmallVector<Value> convOps = {actSlice, filtSlice, convInitFill};
      SmallVector<Type> convTypes = {convInitFill.getType()};
      Operation *convTileOp = clone(
          rewriter, match.genericConv.getOperation(), convTypes, convOps);
      if (!convTileOp)
        return Value();
      // Offset the linalg.index values by the loop start of each dim
      // (n?, oc, oh, ow, ic, kh, kw).
      SmallVector<OpFoldResult> convLoopOffsets = withBatch(
          hasBatch, n0Ofr,
          {cAbs, hAbs, epi_iw0, rewriter.getIndexAttr(0),
           rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)});
      linalg::offsetIndices(rewriter, cast<linalg::LinalgOp>(convTileOp),
                            convLoopOffsets);
      Value convTile = convTileOp->getResult(0);

      // Loop offsets/sizes for this chunk in (N?,C,H,W) loop space, used both
      // to slice the epilogue's other inputs and to size its output.
      SmallVector<OpFoldResult> chunkLoopOffsets =
          withBatch(hasBatch, n0Ofr, {cAbs, hAbs, epi_iw0});
      SmallVector<OpFoldResult> chunkLoopSizes = withBatch(
          hasBatch, nOfr,
          {rewriter.getIndexAttr(cSize), rewriter.getIndexAttr(chunkH),
           rewriter.getIndexAttr(local_W)});

      SmallVector<Value> tiledEpiInputs;
      tiledEpiInputs.reserve(nEpiIn);
      for (auto [inp, inMap] : llvm::zip(epiInputs, epiInMaps)) {
        if (inp == oldConvResult) {
          tiledEpiInputs.push_back(convTile);
          continue;
        }
        auto inpType = dyn_cast<RankedTensorType>(inp.getType());
        if (!inpType)
          return Value();

        SmallVector<OpFoldResult> sliceOffsets, sliceSizes, sliceStrides;
        if (failed(getOperandSliceFromIndexingMap(
                inMap, chunkLoopOffsets, chunkLoopSizes, sliceOffsets,
                sliceSizes, sliceStrides)))
          return Value();

        // Build a static/dynamic shape for the slice result type.
        SmallVector<int64_t> sliceStaticShape;
        for (OpFoldResult ofr : sliceSizes) {
          if (auto attr = dyn_cast<Attribute>(ofr))
            sliceStaticShape.push_back(cast<IntegerAttr>(attr).getInt());
          else
            sliceStaticShape.push_back(ShapedType::kDynamic);
        }
        auto sliceType =
            RankedTensorType::get(sliceStaticShape, inpType.getElementType());
        Value sliced = tensor::ExtractSliceOp::create(
            rewriter, loc, sliceType, inp, sliceOffsets, sliceSizes,
            sliceStrides);
        tiledEpiInputs.push_back(sliced);
      }

      SmallVector<OpFoldResult> chunkOutOfr, chunkOutSizes, chunkOutStrides;
      if (failed(getOperandSliceFromIndexingMap(
              epiOutMap, chunkLoopOffsets, chunkLoopSizes, chunkOutOfr,
              chunkOutSizes, chunkOutStrides)))
        return Value();
      SmallVector<int64_t> chunkOutStaticShape;
      for (OpFoldResult ofr : chunkOutSizes) {
        if (auto attr = dyn_cast<Attribute>(ofr))
          chunkOutStaticShape.push_back(cast<IntegerAttr>(attr).getInt());
        else
          chunkOutStaticShape.push_back(ShapedType::kDynamic);
      }
      auto chunkTileType =
          RankedTensorType::get(chunkOutStaticShape, elemType);
      Value chunkInitEmpty =
          tensor::EmptyOp::create(rewriter, loc, chunkOutSizes, elemType);

      SmallVector<Value> tiledEpiOps = tiledEpiInputs;
      tiledEpiOps.push_back(chunkInitEmpty);

      Operation *epiChunkOp = clone(rewriter, match.epilogue.getOperation(),
                                    TypeRange{chunkTileType}, tiledEpiOps);
      if (!epiChunkOp)
        return Value();

      linalg::offsetIndices(rewriter, cast<linalg::LinalgOp>(epiChunkOp),
                            chunkLoopOffsets);
      return epiChunkOp->getResult(0);
    };

    std::tie(chunkC, chunkH) = pickConvEpilogueChunkSizes(
        nb, TC, match.CIn, local_H, local_W, convInHFull,
        match.convStrideW, match.convKW, match.convDilW, elemBytes,
        /*forceFullChannel=*/kFullChannel);
    convInHChunkFull = (chunkH - 1) * match.convStrideH +
                       (match.convKH - 1) * match.convDilH + 1;

    Value cChunkStep = arith::ConstantIndexOp::create(rewriter, loc, chunkC);
    Value cTileBound = arith::ConstantIndexOp::create(rewriter, loc, TC);
    Value hChunkStep = arith::ConstantIndexOp::create(rewriter, loc, chunkH);
    Value hTileBound = arith::ConstantIndexOp::create(rewriter, loc, local_H);

    int64_t B = hasBatch ? 1 : 0;
    MLIRContext *ctx = rewriter.getContext();

    // The workgroup's maxpool OUTPUT tile [(N,) TC, TOH, TOW], filled one
    // channel chunk at a time and written back once.
    SmallVector<int64_t> mpTileShape =
        withBatchShape(hasBatch, nb, {TC, TOH, TOW});
    Value mpTileInit =
        tensor::EmptyOp::create(rewriter, loc, mpTileShape, elemType);

    // ----- Channel-chunk loop: the maxpool is FUSED in here -----
    // Each iteration (a) materializes this chunk's conv+epilogue window
    // [(N,) chunkC, local_H, local_W] in a small local buffer, (b) maxpools it
    // into [(N,) chunkC, TOH, TOW], and (c) writes that into the output tile.
    // The conv window is thus never held for the whole TC-channel tile at once
    // -- which is exactly what lets TC/TOH/TOW be large blocks (few workgroups)
    auto cLoop = scf::ForOp::create(rewriter, loc, c0idx, cTileBound,
                                    cChunkStep, ValueRange{mpTileInit});
    {
      OpBuilder::InsertionGuard cGuard(rewriter);
      rewriter.setInsertionPointToStart(cLoop.getBody());
      Value cOff = cLoop.getInductionVar();
      Value cAbs = arith::AddIOp::create(rewriter, loc, c0, cOff);
      Value mpAcc = cLoop.getRegionIterArg(0);

      // (a) conv+epilogue window for this chunk's chunkC channels, assembled
      // over height chunks (chunkH divides local_H exactly, so no remainder).
      //
      // The window is assembled directly into a buffer carrying an mpPad-wide
      // ZERO GUARD BAND on every side, and each height chunk is masked to 0
      // outside the tile's dynamic valid region as it is inserted. Together
      // those two things make every maxpool tap below unconditionally in
      // bounds and correctly zero-padded, which is what lets the pool be a
      // plain vectorizable elementwise `maximumf` (see (b)).
      //
      // Masking and guard-banding here -- rather than in separate passes over
      // a full-size temporary -- keeps this to ONE local buffer. Two extra
      // local_H x local_W temporaries would blow the 32KB stack allocation
      // limit this backend enforces.
      int64_t padH = match.mpPadH, padW = match.mpPadW;
      int64_t B_ = B;
      SmallVector<int64_t> gShape = withBatchShape(
          hasBatch, nb, {chunkC, local_H + 2 * padH, local_W + 2 * padW});
      Value gEmpty = tensor::EmptyOp::create(rewriter, loc, gShape, elemType);
      Value gZero = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                           ValueRange{gEmpty})
                        .getResult(0);
      Value cPadHOfr = arith::ConstantIndexOp::create(rewriter, loc, padH);

      auto hLoop = scf::ForOp::create(rewriter, loc, c0idx, hTileBound,
                                      hChunkStep, ValueRange{gZero});
      {
        OpBuilder::InsertionGuard hGuard(rewriter);
        rewriter.setInsertionPointToStart(hLoop.getBody());
        Value hOff = hLoop.getInductionVar();
        Value hAbs = arith::AddIOp::create(rewriter, loc, epi_ih0, hOff);

        Value chunkResult = computeEpilogueChunk(cAbs, chunkC, hAbs);
        if (!chunkResult)
          return failure();

        // Mask this chunk to 0 at or beyond the dynamic valid window. Rows and
        // columns in [valid_H,local_H) / [valid_W,local_W) hold conv output
        // derived from the activation's trailing zero padding, which is NOT
        // zero (bias + ReLU), so it must be zeroed explicitly for the pool to
        // see true maxpool padding. Elementwise and statically shaped.
        SmallVector<int64_t> chunkShape =
            withBatchShape(hasBatch, nb, {chunkC, chunkH, local_W});
        auto chunkTy = RankedTensorType::get(chunkShape, elemType);
        AffineMap idChunk = AffineMap::getMultiDimIdentityMap(3 + B_, ctx);
        SmallVector<utils::IteratorType> chunkPar(
            3 + B_, utils::IteratorType::parallel);
        Value maskInit =
            tensor::EmptyOp::create(rewriter, loc, chunkShape, elemType);
        auto maskGeneric = linalg::GenericOp::create(
            rewriter, loc, TypeRange{chunkTy},
            /*inputs=*/ValueRange{chunkResult},
            /*outputs=*/ValueRange{maskInit},
            ArrayRef<AffineMap>{idChunk, idChunk}, chunkPar,
            [&](OpBuilder &b, Location l, ValueRange args) {
              // Row index is chunk-local; add hOff for the tile-local row.
              Value i = linalg::IndexOp::create(b, l, B_ + 1).getResult();
              Value j = linalg::IndexOp::create(b, l, B_ + 2).getResult();
              Value row = arith::AddIOp::create(b, l, hOff, i);
              Value hOk = arith::CmpIOp::create(b, l, arith::CmpIPredicate::slt,
                                                row, valid_H);
              Value wOk = arith::CmpIOp::create(b, l, arith::CmpIPredicate::slt,
                                                j, valid_W);
              Value ok = arith::AndIOp::create(b, l, hOk, wOk);
              Value v = arith::SelectOp::create(b, l, ok, args[0], zero);
              linalg::YieldOp::create(b, l, ValueRange{v});
            });

        // Insert at channel offset 0 (this buffer holds only this chunk's
        // channels) and height hOff, shifted by the guard band.
        Value dstH = arith::AddIOp::create(rewriter, loc, cPadHOfr, hOff);
        SmallVector<OpFoldResult> insOffsets = withBatch(
            hasBatch, zeroOfr,
            {zeroOfr, OpFoldResult(dstH), rewriter.getIndexAttr(padW)});
        SmallVector<OpFoldResult> insSizes = withBatch(
            hasBatch, nOfr,
            {rewriter.getIndexAttr(chunkC), rewriter.getIndexAttr(chunkH),
             rewriter.getIndexAttr(local_W)});
        Value updated = tensor::InsertSliceOp::create(
            rewriter, loc, maskGeneric.getResult(0),
            hLoop.getRegionIterArg(0), insOffsets, insSizes, strides1);
        scf::YieldOp::create(rewriter, loc, ValueRange{updated});
      }
      rewriter.setInsertionPointAfter(hLoop);
      Value epiGuarded = hLoop.getResult(0);

      // (b) Implicit-padding maxpool of this chunk -> [(N,) chunkC, TOH, TOW].
      //
      // The maxpool reads epiChunk at tile-local coordinates
      //   ih_e = oh*mpStrideH + kh - insert_h,   iw_e = ow*mpStrideW + kw - insert_w
      // and must see 0 wherever that falls outside the tile's dynamic valid
      // window [0,valid_H) x [0,valid_W) -- that is the maxpool's zero padding.
      //
      // Rather than test that per element inside the pool body (which forces a
      // scalar `tensor.extract` and blocks vectorization -- an all-parallel
      // linalg.generic whose body holds scf.for loops is neither elementwise
      // nor a reduction, so GenericVectorizationPass rejects it), the guard is
      // pushed into the *data* in two static, vectorizable steps:
      //
      //   (b1) mask epiChunk elementwise to 0 outside [0,valid_H)x[0,valid_W),
      //   (b2) copy that into a buffer carrying an mpPad-wide zero guard band,
      //
      // after which every pooling tap is unconditionally in bounds and reads
      // the correct value. The pool itself then becomes a plain elementwise
      // `maximumf` over a strided slice, with (kh,kw) as ordinary outer loops
      // -- the same shape the un-fused maxpool has, which vectorizes cleanly.
      // The pooling accumulator, seeded with the pool's identity (0, matching
      // the original body's initial value -- the epilogue is ReLU'd so all
      // real inputs are >= 0).
      SmallVector<int64_t> mpChunkShape =
          withBatchShape(hasBatch, nb, {chunkC, TOH, TOW});
      auto mpChunkType = RankedTensorType::get(mpChunkShape, elemType);
      Value mpChunkEmpty =
          tensor::EmptyOp::create(rewriter, loc, mpChunkShape, elemType);
      Value mpChunkInit =
          linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                 ValueRange{mpChunkEmpty})
              .getResult(0);

      // (b3) kh as an outer scf.for loop; kw stays a REDUCTION dimension of
      // the pooling generic, over a slice that is CONTIGUOUS in W.
      //
      // Making kw an outer loop too would be simpler, but then each read is a
      // stride-mpStrideW slice, and a strided vector.transfer_read on the
      // innermost dim lowers to gather-grade code -- measurably slower than
      // the scalar pool it replaces. Instead the generic reads a contiguous
      // window of width (TOW-1)*mpStrideW + mpKW with the affine map
      // (.., ow, kw) -> (.., ow*mpStrideW + kw), which is exactly the form the
      // un-fused maxpool has: the vectorizer turns it into contiguous loads
      // plus vector.extract_strided_slice.
      //
      // H is still strided (mpStrideH) but only as a slice stride between
      // rows, never within a vector, so it costs nothing.
      Value cKH = arith::ConstantIndexOp::create(rewriter, loc, match.mpKH);
      Value c1b = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value cPadH = arith::ConstantIndexOp::create(rewriter, loc, padH);
      Value cPadW = arith::ConstantIndexOp::create(rewriter, loc, padW);
      // base_h = mpPadH - insert_h (>= 0, since insert_h <= mpPadH): the
      // guard-band row that tile-local epilogue row 0 sits at.
      Value baseH = arith::SubIOp::create(rewriter, loc, cPadH, insert_h);
      Value baseW = arith::SubIOp::create(rewriter, loc, cPadW, insert_w);

      // Width of the contiguous W window the reduction reads.
      int64_t winW = (TOW - 1) * match.mpStrideW + match.mpKW;
      SmallVector<int64_t> winShape =
          withBatchShape(hasBatch, nb, {chunkC, TOH, winW});
      auto winType = RankedTensorType::get(winShape, elemType);

      // Pool maps over loops (n?, c, oh, ow, kw); kw is the reduction.
      unsigned nPool = 4 + B_;
      SmallVector<AffineExpr> inExprs, outExprs;
      for (unsigned d = 0; d < 3u + B_; ++d) {
        inExprs.push_back(rewriter.getAffineDimExpr(d));
        outExprs.push_back(rewriter.getAffineDimExpr(d));
      }
      // Replace the trailing ow expr on the input with ow*mpStrideW + kw.
      inExprs.back() = rewriter.getAffineDimExpr(2 + B_) * match.mpStrideW +
                       rewriter.getAffineDimExpr(3 + B_);
      AffineMap poolInMap = AffineMap::get(nPool, 0, inExprs, ctx);
      AffineMap poolOutMap = AffineMap::get(nPool, 0, outExprs, ctx);
      // linalg requires the CONCATENATED indexing map to be invertible, and
      // (.., ow*mpStrideW + kw) entangles ow with kw. A dummy kernel-shape
      // operand mapped to (kw) alone recovers invertibility -- this is exactly
      // why linalg pooling ops carry an otherwise unused window operand. It is
      // never read by the body and folds away.
      AffineMap poolKMap = AffineMap::get(
          nPool, 0, {rewriter.getAffineDimExpr(3 + B_)}, ctx);
      Value poolWindowDummy = tensor::EmptyOp::create(
          rewriter, loc, ArrayRef<int64_t>{match.mpKW}, elemType);
      SmallVector<utils::IteratorType> poolIters(3 + B_,
                                                 utils::IteratorType::parallel);
      poolIters.push_back(utils::IteratorType::reduction);

      auto khLoop = scf::ForOp::create(rewriter, loc, c0idx, cKH, c1b,
                                       ValueRange{mpChunkInit});
      {
        OpBuilder::InsertionGuard khGuard(rewriter);
        rewriter.setInsertionPointToStart(khLoop.getBody());
        Value kh = khLoop.getInductionVar();
        Value acc = khLoop.getRegionIterArg(0);

        Value offH = arith::AddIOp::create(rewriter, loc, baseH, kh);
        SmallVector<OpFoldResult> wOffsets =
            withBatch(hasBatch, zeroOfr,
                      {zeroOfr, OpFoldResult(offH), OpFoldResult(baseW)});
        SmallVector<OpFoldResult> wSizes = withBatch(
            hasBatch, nOfr,
            {rewriter.getIndexAttr(chunkC), rewriter.getIndexAttr(TOH),
             rewriter.getIndexAttr(winW)});
        SmallVector<OpFoldResult> wStrides = withBatch(
            hasBatch, rewriter.getIndexAttr(1),
            {rewriter.getIndexAttr(1),
             rewriter.getIndexAttr(match.mpStrideH), rewriter.getIndexAttr(1)});
        Value window = tensor::ExtractSliceOp::create(
            rewriter, loc, winType, epiGuarded, wOffsets, wSizes, wStrides);

        auto poolGeneric = linalg::GenericOp::create(
            rewriter, loc, TypeRange{mpChunkType},
            /*inputs=*/ValueRange{window, poolWindowDummy},
            /*outputs=*/ValueRange{acc},
            ArrayRef<AffineMap>{poolInMap, poolKMap, poolOutMap}, poolIters,
            [&](OpBuilder &b, Location l, ValueRange args) {
              // args[1] is the dummy window operand; unused.
              Value m = arith::MaximumFOp::create(b, l, args[2], args[0]);
              linalg::YieldOp::create(b, l, ValueRange{m});
            });
        scf::YieldOp::create(rewriter, loc,
                             ValueRange{poolGeneric.getResult(0)});
      }
      rewriter.setInsertionPointAfter(khLoop);
      Value mpChunk = khLoop.getResult(0);

      // (c) place this chunk's pooled channels into the workgroup output tile.
      SmallVector<OpFoldResult> mpInsOffsets =
          withBatch(hasBatch, zeroOfr, {cOff, zeroOfr, zeroOfr});
      SmallVector<OpFoldResult> mpInsSizes = withBatch(
          hasBatch, nOfr,
          {rewriter.getIndexAttr(chunkC), rewriter.getIndexAttr(TOH),
           rewriter.getIndexAttr(TOW)});
      Value mpUpdated = tensor::InsertSliceOp::create(
          rewriter, loc, mpChunk, mpAcc, mpInsOffsets, mpInsSizes, strides1);
      scf::YieldOp::create(rewriter, loc, ValueRange{mpUpdated});
    }
    Value mpTile = cLoop.getResult(0);

    // -----------------------------------------------------------------------
    // [H] Write back via tensor.parallel_insert_slice.
    //     Covers output positions [(0:N,) c0:c0+TC, oh0:oh0+TOH, ow0:ow0+TOW].
    // -----------------------------------------------------------------------
    auto *parallelBlock = &forallOp.getTerminator().getRegion().front();
    rewriter.setInsertionPointToStart(parallelBlock);

    SmallVector<OpFoldResult> outOffsets =
        withBatch(hasBatch, n0Ofr, {c0, oh0, ow0});
    SmallVector<OpFoldResult> outSizes = withBatch(
        hasBatch, nOfr,
        {rewriter.getIndexAttr(TC), rewriter.getIndexAttr(TOH),
         rewriter.getIndexAttr(TOW)});
    tensor::ParallelInsertSliceOp::create(rewriter, loc, mpTile, shared,
                                           outOffsets, outSizes, strides1);
  }

  // Replace uses of the original maxpool result with the forall result.
  Value newResult = forallOp.getResult(0);
  mpOp->getResult(0).replaceAllUsesWith(newResult);
  rewriter.eraseOp(mpOp);

  return success();
}

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------
struct RewriteConvMaxpoolAsForallPass final
    : impl::RewriteConvMaxpoolAsForallPassBase<
          RewriteConvMaxpoolAsForallPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    if (funcOp->hasAttr(kConvMaxpoolRewrittenAttr)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[RewriteConvMaxpoolAsForall] skip: already rewritten\n";
      });
      return;
    }

    Operation *markedRoot = nullptr;
    funcOp.walk([&](Operation *op) {
      if (op->hasAttr(kSpecialFusionAttr)) {
        markedRoot = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!markedRoot)
      return;

    // a maxpool has exactly 2 reduction loop (kh, kw only).
    if (auto rootGeneric = dyn_cast<linalg::GenericOp>(markedRoot);
        rootGeneric && rootGeneric.getNumReductionLoops() != 2) {
      return;
    }

    FailureOr<ConvMaxpoolMatch> match = matchConvMaxpoolFromRoot(markedRoot);
    if (failed(match)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[RewriteConvMaxpoolAsForall] pattern match failed\n";
      });
      return;
    }

    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(match->maxpoolOp());

    if (failed(rewriteConvMaxpoolAsForall(rewriter, *match))) {
      LLVM_DEBUG({
        llvm::dbgs() << "[RewriteConvMaxpoolAsForall] rewrite failed\n";
      });
      signalPassFailure();
      return;
    }

    funcOp->setAttr(kConvMaxpoolRewrittenAttr,
                    UnitAttr::get(funcOp.getContext()));
  }
};

} // namespace
} // namespace mlir::iree_compiler
