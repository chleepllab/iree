//===----------------------------------------------------------------------===//
// RewriteConvChainAsForall
//
// Fuses a conv -> elementwise -> (re-pad) -> conv chain into ONE scf.forall so
// the intermediate activation between the two convs is produced per output
// tile and never fully materialized, then hands each cloned conv an explicit
// vector schedule so it lowers to vector.contract instead of a scalar loop.
//
// WHAT IT DOES
//   1. matchConvChainFromRoot: starting from the op the dispatch-creation stage
//      tagged with `iree_dispatch.special_conv_chain` (== conv2, the root),
//      walks back through conv2's padded input (tensor.insert_slice) to its
//      elementwise epilogue1 and to conv1. epilogue2 (conv2's elementwise
//      consumer, e.g. the residual add + ReLU) is matched if present, optional.
//   2. rewriteMatchAsForall: builds an scf.forall over conv2's output tiles.
//      Each iteration slices exactly the intermediate window conv1 must produce
//      for that tile, runs conv1 -> epilogue1 -> conv2 (-> epilogue2) locally,
//      and parallel-inserts the tile. A channel-chunk sub-loop keeps the
//      per-chunk producer buffers within the backend stack cap
//      (kStackLimitBytes = 32768; modelled budget kStackModelBudgetBytes).
//   3. setConvVectorTiles: overwrites each cloned conv's vector tiling levels
//      with an [oc=8, ow=8] output block over cin=8 (kh=kw=1) — mirroring
//      IREE's ConvTileAndDecomposeExpert — so GenericVectorization emits a
//      vector<1x8x8> x vector<8x8> contract rather than a scalar nest. (The
//      convs are batch-squeezed CHW generics; expandConvWithUnitBatch re-adds a
//      unit batch dim so the decomposed 1-D conv is NCW, which the vectorizer
//      accepts.)
//
// WORKED EXAMPLE
//   The rewritten dispatch is a layer1 residual block's two 3x3 convs (all such
//   pairs in conv-conv.mlir share these exact shapes):
//     conv1 : act [1,64,56,56], filter [64,64,3,3], pad 1, stride 1 -> [1,64,56,56]
//     relu  (epilogue1)                                             -> [1,64,56,56]
//     conv2 : act [1,64,56,56], filter [64,64,3,3], pad 1, stride 1 -> [1,64,56,56]
//   Seen here as CHW generics ([64,56,56]). After the pass: one scf.forall over
//   conv2's [64,56,56] output. Per tile, only the small [64, tileH+2, tileW+2]
//   slice of the intermediate is produced by conv1 (padded via insert_slice,
//   relu'd) and consumed by conv2 — the full [1,64,56,56] intermediate is never
//   allocated. Both cloned convs carry the oc=8/ow=8/cin=8 vector schedule.
//===----------------------------------------------------------------------===//
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "rewrite-conv-chain-as-forall"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REWRITECONVCHAINASFORALLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static constexpr StringLiteral kSpecialFusionAttr =
    "iree_dispatch.special_conv_chain";
static constexpr StringLiteral kConvChainRewrittenAttr =
    "iree_codegen.conv_chain_rewritten";

// Per-workgroup stack allocation budget (bytes). The channel chunk size Cb is
// derived from this so the per-chunk producer tile(s) stay within the limit,
// instead of materialising the full-channel producer at once.
static constexpr int64_t kStackLimitBytes = 32768;

// Register-blocking shape for the cloned convs' vector tiling (see
// setConvVectorTiles). Matches what IREE's ConvTileAndDecomposeExpert picks for
// the unfused f32 convs of this network: an [oc x ow] output block accumulated
// over `cin` input channels per (kh, kw) step.
static constexpr int64_t kConvVecOcTile = 8;
static constexpr int64_t kConvVecOwTile = 8;
static constexpr int64_t kConvVecCinTile = 8;
// Upper bound on the width tile when it has to be snapped to a divisor of the
// op's output width (see setConvVectorTiles); one AVX-512 f32 register.
static constexpr int64_t kConvVecOwMax = 16;

// Modelled stack budget shared by the producer buffers and the ocTile-sized
// tiles (see [BUDGET] in rewriteMatchAsForall). CALIBRATED, not derived:
// measured stack usage runs ~1.7x this estimate, so it sits well below the
// real kStackLimitBytes cap. Raising it overflows the backend's limit and
// fails the compile outright -- re-measure on the full network before touching.
static constexpr int64_t kStackModelBudgetBytes = 22528;


struct ConvChainMatch {
  linalg::GenericOp conv1;
  linalg::GenericOp epilogue1;
  tensor::InsertSliceOp insertSlice;
  linalg::GenericOp conv2;
  linalg::GenericOp epilogue2;
};

// Prepends `batchVal` to `rest` iff `hasBatch`. Used throughout to build
// offset/size lists for ops whose rank is 3 (CHW) or 4 (NCHW) depending on
// whether an explicit batch dimension is present.
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

struct ConvSpatialParams {
  int64_t strideH;
  int64_t strideW;
  int64_t dilationH;
  int64_t dilationW;
};

// Re-expresses a batch-less (CHW) convolution generic as the equivalent NCHW
// one with a leading unit batch dim, returning the new op (its result is
// collapsed back to the original CHW type, so callers see no type change).
//
// This is required for vectorization, not cosmetic. Once the window dims are
// tiled to 1 and DecomposeConvolutionToLowerDimOps folds the unit H dims away,
// a CHW conv becomes a rank-2 `(c, w + kw)` op. MLIR's Conv1DGenerator only
// recognises the non-channeled (`w`), `nwc` and `ncw` layouts, so it rejects
// that shape, linalg vectorization bails, and the conv survives as a scalar
// loop nest. With the unit batch dim the decomposed form is `(n, c, w + kw)` =
// NCW, which lowers to a vector.contract -- exactly what IREE's own unfused
// NCHW convs get.
//
// Returns {nullptr, nullptr} (leaving `op` untouched) if the op is not the
// expected 3-operand CHW conv generic.
struct ExpandedConv {
  linalg::GenericOp op;   // the NCHW conv; nullptr if nothing was done
  Value chwResult;        // its result, collapsed back to CHW
};
static ExpandedConv expandConvWithUnitBatch(RewriterBase &rewriter,
                                            Location loc,
                                            linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
    return {};

  Value act = op.getDpsInputs()[0];
  Value filter = op.getDpsInputs()[1];
  Value init = op.getDpsInits()[0];
  auto actType = dyn_cast<RankedTensorType>(act.getType());
  auto initType = dyn_cast<RankedTensorType>(init.getType());
  if (!actType || !initType || actType.getRank() != 3 || initType.getRank() != 3)
    return {};

  // rank 3 -> rank 4, folding the new leading unit dim into the first group.
  SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
  auto expandType = [&](RankedTensorType t) {
    SmallVector<int64_t> shape = {1};
    shape.append(t.getShape().begin(), t.getShape().end());
    return RankedTensorType::get(shape, t.getElementType());
  };
  auto actExpType = expandType(actType);
  auto initExpType = expandType(initType);
  Value actExp = tensor::ExpandShapeOp::create(rewriter, loc, actExpType, act,
                                               reassoc);
  Value initExp = tensor::ExpandShapeOp::create(rewriter, loc, initExpType,
                                                init, reassoc);

  // Shift every existing iteration dim up by one and give the activation and
  // the output (but not the filter) the new dim 0 as their leading result.
  SmallVector<AffineMap> oldMaps = op.getIndexingMapsArray();
  unsigned numDims = op.getNumLoops() + 1;
  AffineExpr batchExpr = rewriter.getAffineDimExpr(0);
  auto withLeadingBatch = [&](AffineMap m) {
    AffineMap shifted = m.shiftDims(1);
    SmallVector<AffineExpr> results = {batchExpr};
    results.append(shifted.getResults().begin(), shifted.getResults().end());
    return AffineMap::get(numDims, 0, results, rewriter.getContext());
  };
  SmallVector<AffineMap> newMaps = {withLeadingBatch(oldMaps[0]),
                                    oldMaps[1].shiftDims(1),
                                    withLeadingBatch(oldMaps[2])};

  SmallVector<utils::IteratorType> newIters = {utils::IteratorType::parallel};
  llvm::append_range(newIters, op.getIteratorTypesArray());

  auto newOp = linalg::GenericOp::create(rewriter, loc, TypeRange{initExpType},
                                         ValueRange{actExp, filter},
                                         ValueRange{initExp}, newMaps,
                                         newIters);
  rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                             newOp.getRegion().end());
  // Discardable attrs only -- copying the whole dictionary would put the old
  // op's (now wrong-rank) indexing_maps and iterator_types back.
  newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());

  Value collapsed = tensor::CollapseShapeOp::create(
      rewriter, loc, initType, newOp.getResult(0), reassoc);
  rewriter.replaceOp(op, collapsed);
  return {newOp, collapsed};
}

// Overwrites `op`'s vector tiling levels with a convolution-friendly schedule.
//
// Both convs are cloned from the dispatch's original ops, so they inherit that
// op's lowering_config. In the batch-squeezed CHW form those ops are plain
// linalg.generics and KernelDispatch hands them a *generic* schedule
// (vector_common_parallel = [1, 1, 8], vector_reduction = [0, 0, 0, 1, 1, 16]),
// which is fatal here for two reasons:
//
//   1. The window reduction dims (kh/kw) are not tiled to 1. A conv generic's
//      activation map (`d1 + d4`, `d2 + d5`) is not a projected permutation,
//      so linalg vectorization bails on it outright -- the conv survives
//      GenericVectorization as a scalar linalg.generic nest. Tiling kh/kw to 1
//      collapses the window offsets into the extract_slice, leaving a
//      contraction the vectorizer turns into a vector.contract.
//   2. The output-channel and input-channel dims are tiled to 1, so even once
//      vectorized there is no register blocking -- one FMA column per contract.
//
// The schedule mirrors what IREE's own ConvTileAndDecomposeExpert picks for the
// unfused convs of this network (oc=8, oh=1, ow=8 parallel; cin=8, kh=kw=1
// reduction), which lowers to a vector<1x8x8> x vector<8x8> contract. Any
// existing distribution level is preserved (it marks conv2 as the dispatch
// root); only the vector levels are replaced.
static int64_t largestDivisorAtMost(int64_t total, int64_t maxVal);

static void setConvVectorTiles(Operation *op, bool hasBatch, int64_t ocTile,
                               int64_t owTile, int64_t cinTile) {
  MLIRContext *ctx = op->getContext();
  int64_t B = hasBatch ? 1 : 0;

  // The width tile must *divide* the op's own output width. A tile that only
  // partially covers the last iteration (e.g. tile 8 over a 9-wide producer
  // window) leaves an `affine.min`-sized, dynamically shaped conv slice, and
  // MLIR's Conv1DGenerator only vectorizes statically shaped 1-D convs -- so
  // the conv would decompose correctly and still fail to vectorize. Snapping
  // to a divisor keeps every tile static.
  if (auto initType = dyn_cast<RankedTensorType>(
          cast<linalg::LinalgOp>(op).getDpsInits()[0].getType())) {
    int64_t owExtent = initType.getShape().back();
    if (!ShapedType::isDynamic(owExtent)) {
      owTile = owExtent <= kConvVecOwMax
                   ? owExtent
                   : largestDivisorAtMost(owExtent, kConvVecOwMax);
    }
  }

  // Iteration space is [(n,) oc, oh, ow, cin, kh, kw].
  SmallVector<int64_t> parallelTiles(6 + B, 0);
  SmallVector<int64_t> reductionTiles(6 + B, 0);
  if (hasBatch)
    parallelTiles[0] = 1;
  parallelTiles[B + 0] = ocTile;
  parallelTiles[B + 1] = 1;
  parallelTiles[B + 2] = owTile;
  reductionTiles[B + 3] = cinTile;
  reductionTiles[B + 4] = 1;
  reductionTiles[B + 5] = 1;

  SmallVector<NamedAttribute> items;
  StringRef distName =
      IREE::CPU::getTilingLevelName(IREE::CPU::TilingLevel::DistributionTiles);
  if (auto oldConfig = dyn_cast_or_null<IREE::CPU::LoweringConfigAttr>(
          op->getAttr("lowering_config"))) {
    for (NamedAttribute item : oldConfig.getConfig())
      if (item.getName() == distName)
        items.push_back(item);
  }
  items.emplace_back(
      IREE::CPU::getTilingLevelName(
          IREE::CPU::TilingLevel::VectorCommonParallelTiles),
      IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(ctx, parallelTiles));
  items.emplace_back(
      IREE::CPU::getTilingLevelName(IREE::CPU::TilingLevel::VectorReductionTiles),
      IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(ctx, reductionTiles));
  op->setAttr("lowering_config", IREE::CPU::LoweringConfigAttr::get(ctx, items));
}

// Largest divisor of `total` that is <= `maxVal` (result is always >= 1).
// Used to snap a desired tile size down to one that evenly divides the output
// extent, so the outer forall's tiles partition the output with no remainder.
static int64_t largestDivisorAtMost(int64_t total, int64_t maxVal) {
  if (total <= 0) return 1;
  maxVal = std::min(maxVal, total);
  if (maxVal < 1) maxVal = 1;
  for (int64_t d = maxVal; d >= 1; --d)
    if (total % d == 0)
      return d;
  return 1;
}

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

    AffineExpr lhs = bin.getLHS();
    AffineExpr rhs = bin.getRHS();

    if (auto dim = dyn_cast<AffineDimExpr>(lhs)) {
      if (dim.getPosition() == dimPos) {
        if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
          coeff = cst.getValue();
          return true;
        }
      }
    }
    if (auto dim = dyn_cast<AffineDimExpr>(rhs)) {
      if (dim.getPosition() == dimPos) {
        if (auto cst = dyn_cast<AffineConstantExpr>(lhs)) {
          coeff = cst.getValue();
          return true;
        }
      }
    }
  }

  return false;
}

static FailureOr<std::pair<int64_t, int64_t>>
getTwoTermAffineCoeffs(AffineExpr expr, unsigned spatialDim, unsigned kernelDim) {
  // Support:
  //   spatial + kernel
  //   spatial*c1 + kernel
  //   spatial + kernel*c2
  //   spatial*c1 + kernel*c2
  //
  // i.e. expr = a * spatialDim + b * kernelDim

  int64_t spatialCoeff = 0;
  int64_t kernelCoeff = 0;

  if (auto sum = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (sum.getKind() != AffineExprKind::Add) {
      return failure();
    }

    AffineExpr lhs = sum.getLHS();
    AffineExpr rhs = sum.getRHS();

    bool lhsIsSpatial = getDimCoeff(lhs, spatialDim, spatialCoeff);
    bool rhsIsKernel = getDimCoeff(rhs, kernelDim, kernelCoeff);
    if (lhsIsSpatial && rhsIsKernel)
      return std::make_pair(spatialCoeff, kernelCoeff);

    bool rhsIsSpatial = getDimCoeff(rhs, spatialDim, spatialCoeff);
    bool lhsIsKernel = getDimCoeff(lhs, kernelDim, kernelCoeff);
    if (rhsIsSpatial && lhsIsKernel)
      return std::make_pair(spatialCoeff, kernelCoeff);

    return failure();
  }

  // Degenerate cases like just "oh" or just "kh" are not enough for conv.
  return failure();
}

static FailureOr<ConvSpatialParams>
getConvSpatialParamsFromInputMap(linalg::GenericOp conv, bool hasBatch) {
  SmallVector<AffineMap> maps = conv.getIndexingMapsArray();
  if (maps.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "[getConvSpatialParamsFromInputMap] missing indexing maps\n";
    });
    return failure();
  }

  // Assume input 0 is the activation tensor.
  // Expected input map shape for CHW conv (B=0) or NCHW conv (B=1):
  //   (n?, oc, oh, ow, ic, kh, kw) -> (n?, ic, oh*strideH + kh*dilationH,
  //                                          ow*strideW + kw*dilationW)
  AffineMap inputMap = maps[0];
  int64_t B = hasBatch ? 1 : 0;
  if (inputMap.getNumResults() != 3 + B) {
    LLVM_DEBUG({
      llvm::dbgs() << "[getConvSpatialParamsFromInputMap] expected rank-"
                   << (3 + B) << " input map\n";
    });
    return failure();
  }

  // Current loop dim convention in this rewrite:
  // (n?=0,) oc=B, oh=B+1, ow=B+2, ic=B+3, kh=B+4, kw=B+5
  FailureOr<std::pair<int64_t, int64_t>> hParams = getTwoTermAffineCoeffs(
      inputMap.getResult(B + 1), /*oh=*/B + 1, /*kh=*/B + 4);
  FailureOr<std::pair<int64_t, int64_t>> wParams = getTwoTermAffineCoeffs(
      inputMap.getResult(B + 2), /*ow=*/B + 2, /*kw=*/B + 5);

  if (failed(hParams) || failed(wParams)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "[getConvSpatialParamsFromInputMap] unsupported input indexing map pattern\n";
    });
    return failure();
  }

  ConvSpatialParams params;
  params.strideH = hParams->first;
  params.dilationH = hParams->second;
  params.strideW = wParams->first;
  params.dilationW = wParams->second;

  if (params.strideH <= 0 || params.strideW <= 0 ||
      params.dilationH <= 0 || params.dilationW <= 0) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "[getConvSpatialParamsFromInputMap] stride/dilation must be positive\n";
    });
    return failure();
  }

  return params;
}

// Conv-like generic, 
// either CHW (rank-3 activation, no batch) 
// or NCHW (rank-4 activation, explicit batch). 
// Filter is always rank-4 (oc, ic, kh, kw).
static bool isConvLikeGeneric(linalg::GenericOp op) {
  if (!op) return false;
  if (op.getNumDpsInputs() != 2) return false;
  if (op.getNumResults() != 1) return false;
  if (op.getNumReductionLoops() < 3) return false;

  auto input0Type = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
  auto input1Type = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
  if (!input0Type || !input1Type) return false;
  if (input1Type.getRank() != 4) return false;
  if (input0Type.getRank() != 3 && input0Type.getRank() != 4) return false;

  return true;
}

static bool isElementwiseGeneric(linalg::GenericOp op) {
  if (!op) return false;
  return op.getNumReductionLoops() == 0;
}

static FailureOr<ConvChainMatch> matchConvChainFromRoot(Operation *rootOp) {
  auto conv2 = dyn_cast<linalg::GenericOp>(rootOp);
  if (!isConvLikeGeneric(conv2)) {
    return failure();
  }

  Operation *insertSliceOp = nullptr;
  for (Value input : conv2.getDpsInputs()) {
    if (auto def = input.getDefiningOp<tensor::InsertSliceOp>()) {
      insertSliceOp = def;
      break;
    }
  }
  if (!insertSliceOp) {
    return failure();
  }

  auto insertSlice = cast<tensor::InsertSliceOp>(insertSliceOp);

  auto epilogue1 =
      insertSlice.getSource().getDefiningOp<linalg::GenericOp>();
  if (!isElementwiseGeneric(epilogue1)) {
    return failure();
  }

  linalg::GenericOp conv1;
  for (Value input : epilogue1.getDpsInputs()) {
    if (auto def = input.getDefiningOp<linalg::GenericOp>()) {
      if (isConvLikeGeneric(def)) {
        conv1 = def;
        break;
      }
    }
  }
  if (!conv1) {
    return failure();
  }

  // conv1 and conv2 must belong to the same rank family (both CHW or both
  // NCHW) -- the rewrite below assumes a single hasBatch/B applies to the
  // whole chain.
  auto conv1ActType =
      cast<RankedTensorType>(conv1.getDpsInputs()[0].getType());
  auto conv2ActType =
      cast<RankedTensorType>(conv2.getDpsInputs()[0].getType());
  if (conv1ActType.getRank() != conv2ActType.getRank()) {
    LLVM_DEBUG({
      llvm::dbgs() << "[matchConvChainFromRoot] conv1/conv2 rank mismatch\n";
    });
    return failure();
  }

  ConvChainMatch match;
  match.conv1 = conv1;
  match.epilogue1 = epilogue1;
  match.insertSlice = insertSlice;
  match.conv2 = conv2;
  match.epilogue2 = nullptr;

  Value conv2Result = conv2.getResult(0);
  auto conv2ResultType = dyn_cast<RankedTensorType>(conv2Result.getType());

  for (Operation *user : conv2Result.getUsers()) {
    auto genericUser = dyn_cast<linalg::GenericOp>(user);
    if (!genericUser) continue;

    if (!isElementwiseGeneric(genericUser)) continue;
    if (genericUser->getNumResults() != 1) continue;

    auto genericResultType =
        dyn_cast<RankedTensorType>(genericUser.getResult(0).getType());
    if (!genericResultType || !conv2ResultType) continue;

    if (genericResultType != conv2ResultType) continue;

    bool usesConv2Result = false;
    for (Value input : genericUser.getDpsInputs()) {
      if (input == conv2Result) {
        usesConv2Result = true;
        break;
      }
    }
    if (!usesConv2Result) continue;

    match.epilogue2 = genericUser;
    break;
  }

  // epilogue2 is OPTIONAL. In a standalone conv->epi->pad->conv chain (e.g. a
  // ResNet block whose residual add + ReLU is split into a *separate* dispatch)
  // conv2's result leaves this dispatch directly, with no in-dispatch
  // elementwise consumer. rewriteMatchAsForall already handles a null
  // epilogue2 (it writes conv2's tile straight to the output), so requiring one
  // here only declined otherwise-fusable chains. Leave match.epilogue2 null and
  // proceed.

  return match;
}

static FailureOr<SmallVector<int64_t>>
getConv2DistributionTiles(linalg::GenericOp conv2) {
  Attribute loweringConfigAttr = conv2->getAttr("lowering_config");
  if (!loweringConfigAttr) return failure();

  std::string loweringConfigStr;
  llvm::raw_string_ostream os(loweringConfigStr);
  loweringConfigAttr.print(os);
  os.flush();

  size_t distPos = loweringConfigStr.find("distribution = [");
  if (distPos == std::string::npos) return failure();
  distPos += std::string("distribution = [").size();

  size_t endPos = loweringConfigStr.find(']', distPos);
  if (endPos == std::string::npos) return failure();

  std::string distList = loweringConfigStr.substr(distPos, endPos - distPos);
  SmallVector<int64_t> vals;
  SmallVector<StringRef> parts;
  StringRef(distList).split(parts, ',', -1, false);
  for (StringRef p : parts) {
    int64_t v = 0;
    if (p.trim().getAsInteger(10, v)) return failure();
    vals.push_back(v);
  }

  if (vals.size() < 3) return failure();
  return SmallVector<int64_t>{vals[0], vals[1], vals[2]};
}

static LogicalResult getOperandSliceFromIndexingMap(
    AffineMap operandMap,
    ArrayRef<OpFoldResult> loopOffsets,
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
        llvm::dbgs()
            << "[getOperandSliceFromIndexingMap] only projected permutation maps are supported\n";
      });
      return failure();
    }

    unsigned dimPos = dimExpr.getPosition();
    if (dimPos >= loopOffsets.size() || dimPos >= loopSizes.size()) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "[getOperandSliceFromIndexingMap] dim position out of range\n";
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

// ---------------------------------------------------------------------------
// Main rewrite.
//
// conv2's original input was a tensor.insert_slice of epilogue1's output into
// a zero-padded [C, H1+2*padH, W1+2*padW] tensor. Rather than reconstructing
// an equivalent padded tensor per forall tile (which -- even built locally --
// costs two same-sized stack buffers per channel chunk: one to zero-fill,
// one to hold the final result), this rewrite uses *implicit* padding, the
// same technique RewriteConvMaxpoolAsForall.cpp uses for the maxpool's own
// halo:
//
//   1. conv1's activation input is padded once, up front (a real
//      tensor.pad, not a per-tile op), with just enough trailing zeros that
//      every tile's full, unclamped [CIn, producerH, producerW] conv1+
//      epilogue1 window can always be extracted at a statically-sized,
//      in-bounds offset -- no per-tile clamping of the *size* is ever
//      needed, only of the low-side *start*.
//   2. The handful of positions this window "over-reads" past conv1's true
//      (H1, W1) extent get some well-defined-but-meaningless value from
//      conv1+epilogue1 (e.g. BN bias survives a zero input). A select
//      appended directly into epilogue1's cloned body -- using the tile's
//      analytically-known valid sub-region -- zeroes exactly those
//      positions before conv2 ever reads them.
//
// The result is fed straight into conv2 with no separate padded-buffer
// materialisation at all, cutting the per-channel-chunk stack footprint
// (conv1's zero-fill init + epilogue1's result) in half versus building a
// second, separately zero-filled "producer" tile.
// ---------------------------------------------------------------------------
static LogicalResult rewriteMatchAsForall(
    IRRewriter &rewriter,
    ConvChainMatch &match) {
  Location loc = match.conv2.getLoc();

  auto conv2 = match.conv2;
  auto epilogue2 = match.epilogue2;

  Value conv2Weight = conv2.getDpsInputs()[1];

  auto conv2WeightType = dyn_cast<RankedTensorType>(conv2Weight.getType());
  auto conv2ResultType = dyn_cast<RankedTensorType>(conv2.getResult(0).getType());
  // epilogue2 is optional (see matchConvChainFromRoot). When absent, conv2's
  // own result is the dispatch output, so the "final" type is conv2's type.
  auto epilogue2ResultType =
      epilogue2 ? dyn_cast<RankedTensorType>(epilogue2.getResult(0).getType())
                : conv2ResultType;

  if (!conv2WeightType || !conv2ResultType || !epilogue2ResultType) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] expected ranked tensor types\n";
    });
    return failure();
  }

  if (conv2WeightType.getRank() != 4 ||
      (conv2ResultType.getRank() != 3 && conv2ResultType.getRank() != 4) ||
      epilogue2ResultType.getRank() != conv2ResultType.getRank()) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] expected rank-3 (CHW) or rank-4 "
                      "(NCHW) conv2+epilogue2\n";
    });
    return failure();
  }
  const bool hasBatch = conv2ResultType.getRank() == 4;
  const int64_t B = hasBatch ? 1 : 0;
  const int64_t N = hasBatch ? conv2ResultType.getShape()[0] : 1;

  FailureOr<SmallVector<int64_t>> tiles = getConv2DistributionTiles(conv2);
  if (failed(tiles) || tiles->size() != 3) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] failed to get conv2 distribution tiles\n";
    });
    return failure();
  }

  int64_t COut = conv2ResultType.getShape()[B];
  int64_t OH = conv2ResultType.getShape()[B + 1];
  int64_t OW = conv2ResultType.getShape()[B + 2];

  int64_t CIn = conv2WeightType.getShape()[1];
  int64_t kH = conv2WeightType.getShape()[2];
  int64_t kW = conv2WeightType.getShape()[3];

  // Desired distribution tiles (the /4 subdivides H/W as in the original
  // heuristic), then SNAPPED down to a divisor of the corresponding output
  // extent so the forall always partitions the output evenly. The raw
  // distribution values need not divide OH/OW (e.g. for the 7x7, 14x14, 28x28
  // layers of ResNet), and forcing a hard failure there needlessly declined
  // the fusion; snapping keeps it firing for every layer.
  int64_t ocTile = largestDivisorAtMost(COut, std::max<int64_t>(1, (*tiles)[0]));
  int64_t tileH = largestDivisorAtMost(OH, std::max<int64_t>(1, (*tiles)[1] / 4));
  int64_t tileW = largestDivisorAtMost(OW, std::max<int64_t>(1, (*tiles)[2] / 4));

  if (COut % ocTile != 0 || OH % tileH != 0 || OW % tileW != 0) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] output shape not divisible by tile sizes\n";
    });
    return failure();
  }

  FailureOr<ConvSpatialParams> conv2SpatialParams =
      getConvSpatialParamsFromInputMap(conv2, hasBatch);
  if (failed(conv2SpatialParams)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] failed to infer conv2 stride/dilation\n";
    });
    return failure();
  }
  int64_t conv2StrideH = conv2SpatialParams->strideH;
  int64_t conv2StrideW = conv2SpatialParams->strideW;
  int64_t conv2DilH = conv2SpatialParams->dilationH;
  int64_t conv2DilW = conv2SpatialParams->dilationW;

  // conv2's own (now implicit) padding, recovered from the offsets of the
  // insert_slice that used to build its padded input tensor.
  int64_t conv2PadH = 0, conv2PadW = 0;
  {
    SmallVector<OpFoldResult> mixedOffsets = match.insertSlice.getMixedOffsets();
    if (static_cast<int64_t>(mixedOffsets.size()) != 3 + B) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] expected rank-" << (3 + B)
                     << " insert_slice offsets\n";
      });
      return failure();
    }
    auto toStatic = [](OpFoldResult ofr, int64_t &out) -> bool {
      auto intAttr = dyn_cast_or_null<IntegerAttr>(dyn_cast<Attribute>(ofr));
      if (!intAttr) return false;
      out = intAttr.getInt();
      return true;
    };
    if (!toStatic(mixedOffsets[B + 1], conv2PadH) ||
        !toStatic(mixedOffsets[B + 2], conv2PadW)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] dynamic conv2 padding not supported\n";
      });
      return failure();
    }
  }

  // Producer window: a conv2 output tile [ocTile,tileH,tileW] needs an
  // unpadded-epilogue1-space window of [CIn, producerH, producerW].
  // ==========================================================================
  // [TILE] Pick the spatial tile to minimise redundant conv1 work.
  //
  // The distribution hint (/4 above) is a poor guide once the output is small:
  // for a 7x7 layer it yields a 1x1 tile, whose 3x3 producer window makes the
  // chain recompute conv1 NINE times over. Measured on ResNet-18 batch 1, the
  // two 512x7x7 chains alone cost 14.3ms of a 28.8ms run for that reason.
  //
  // The real objective is the total conv1 multiplier
  //
  //     redundancy = (COut / ocTile) * (producer area / tile area)
  //
  // and the two factors pull against each other: the conv2 accumulator is
  // [ocTile, tileH, tileW], so a bigger spatial tile forces a smaller ocTile
  // under the same stack budget, and vice versa. Neither term can be minimised
  // alone -- 1x1 minimises the accumulator but maximises halo overlap, while a
  // full-output tile minimises halo but collapses ocTile. So search the (small)
  // space of divisor pairs and evaluate the product directly, applying the same
  // budget rule the ocTile choice uses later so the two agree.
  // ==========================================================================
  {
    const int64_t elemBytesTile =
        (conv2ResultType.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    auto producerExtent = [&](int64_t tile, int64_t stride, int64_t k,
                              int64_t dil) {
      return (tile - 1) * stride + (k - 1) * dil + 1;
    };
    // nb is not known yet (it depends on the conv1 window, hence on the tile),
    // but it scales every candidate's buffers equally, so the ranking is
    // unaffected; N==1 is exact for the batch-1 case and a stand-in otherwise.
    const int64_t nbGuess = hasBatch ? 1 : N;
    double bestScore = 0.0;
    int64_t bestH = tileH, bestW = tileW, bestArea = 0;
    for (int64_t th = 1; th <= OH; ++th) {
      if (OH % th != 0) continue;
      for (int64_t tw = 1; tw <= OW; ++tw) {
        if (OW % tw != 0) continue;
        int64_t ph = producerExtent(th, conv2StrideH, kH, conv2DilH);
        int64_t pw = producerExtent(tw, conv2StrideW, kW, conv2DilW);
        // Producer-side buffers: conv1 accumulator + epilogue1 result, plus the
        // shift buffer (approximated by conv2's padding, the common case).
        int64_t perChan =
            2 * nbGuess * ph * pw * elemBytesTile +
            nbGuess * (ph + conv2PadH) * (pw + conv2PadW) * elemBytesTile;
        int64_t ocBudget = kStackModelBudgetBytes - perChan;
        if (ocBudget <= 0) continue;
        int64_t denom = std::max<int64_t>(1, nbGuess * th * tw * elemBytesTile);
        int64_t oc = largestDivisorAtMost(
            COut, std::min(COut, std::max<int64_t>(1, ocBudget / denom)));
        double redundancy = (double(COut) / oc) *
                            (double(ph) * pw / (double(th) * tw));
        int64_t area = th * tw;
        // Lower redundancy wins; on a tie prefer the larger tile (fewer
        // workgroups, and more contiguous width for the vectorizer).
        if (bestArea == 0 || redundancy < bestScore - 1e-9 ||
            (redundancy < bestScore + 1e-9 && area > bestArea)) {
          bestScore = redundancy;
          bestH = th;
          bestW = tw;
          bestArea = area;
        }
      }
    }
    tileH = bestH;
    tileW = bestW;

  }

  int64_t producerH = (tileH - 1) * conv2StrideH + (kH - 1) * conv2DilH + 1;
  int64_t producerW = (tileW - 1) * conv2StrideW + (kW - 1) * conv2DilW + 1;

  // Extract lowering_config from the original fills so the vectorizer knows to
  // tile them at [1,1,8] rather than materializing the full tile as one vector.
  // Without this, linalg.fill on tensor<64x30x30xf32> becomes
  // vector.transfer_write (vector<64x30x30xf32>) which is 230 KB >> 32768.
  Attribute fillLoweringConfig;
  {
    Value conv1InitVal = match.conv1.getDpsInits()[0];
    if (auto *defOp = conv1InitVal.getDefiningOp())
      fillLoweringConfig = defOp->getAttr("lowering_config");
    if (!fillLoweringConfig) {
      Value conv2InitVal = conv2.getDpsInits()[0];
      if (auto *defOp = conv2InitVal.getDefiningOp())
        fillLoweringConfig = defOp->getAttr("lowering_config");
    }
  }

  auto zeroAttr =
      cast<TypedAttr>(rewriter.getZeroAttr(epilogue2ResultType.getElementType()));
  Value zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

  // ==========================================================================
  // Gather conv1 / epilogue1 static info, and implicitly pad conv1's own
  // activation (once, outside the forall) -- see the big comment above this
  // function for the overall scheme.
  // ==========================================================================
  auto conv1 = match.conv1;
  auto epilogue1 = match.epilogue1;

  Value conv1Input = conv1.getDpsInputs()[0];
  Value conv1Weight = conv1.getDpsInputs()[1];

  auto conv1InputType = cast<RankedTensorType>(conv1Input.getType());
  auto conv1WeightType = cast<RankedTensorType>(conv1Weight.getType());
  auto conv1ResultType = cast<RankedTensorType>(conv1.getResult(0).getType());

  if (conv1InputType.getRank() != 3 + B || conv1ResultType.getRank() != 3 + B) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] conv1 rank does not match conv2's "
                      "rank family\n";
    });
    return failure();
  }

  int64_t C1In = conv1InputType.getShape()[B];
  int64_t C1Out = conv1ResultType.getShape()[B];
  int64_t H1 = conv1ResultType.getShape()[B + 1];
  int64_t W1 = conv1ResultType.getShape()[B + 2];

  // conv1's output channels feed conv2's reduction, so they must match CIn.
  if (C1Out != CIn) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] conv1 out-channels (" << C1Out
                   << ") != conv2 reduction channels (" << CIn << ")\n";
    });
    return failure();
  }

  int64_t k1h = conv1WeightType.getShape()[2];
  int64_t k1w = conv1WeightType.getShape()[3];

  FailureOr<ConvSpatialParams> conv1SpatialParams =
      getConvSpatialParamsFromInputMap(conv1, hasBatch);
  if (failed(conv1SpatialParams)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] failed to infer conv1 stride/dilation\n";
    });
    return failure();
  }
  int64_t conv1StrideH = conv1SpatialParams->strideH;
  int64_t conv1StrideW = conv1SpatialParams->strideW;
  int64_t conv1DilH = conv1SpatialParams->dilationH;
  int64_t conv1DilW = conv1SpatialParams->dilationW;

  // conv1 input window needed to produce a full, unclamped producerH x
  // producerW epilogue1 tile.
  int64_t conv1InputTileH =
      (producerH - 1) * conv1StrideH + (k1h - 1) * conv1DilH + 1;
  int64_t conv1InputTileW =
      (producerW - 1) * conv1StrideW + (k1w - 1) * conv1DilW + 1;

  // ==========================================================================
  // Batch tiling. The forall distributes the batch dimension (NCHW only) so
  // every per-workgroup buffer holds only `nb` images, not the full N. Without
  // this, the un-chunked conv1 input window [N, C1In, conv1InputTileH,
  // conv1InputTileW] alone is N x larger than the batch-1 case and blows past
  // the backend's per-op vector-size / stack limit for any N > 1 (e.g. the
  // whole-ResNet batch of 10) -- the channel chunking only bounds conv2's
  // reduction tile, not this input window. nb is the largest divisor of N whose
  // full-channel input window fits kStackLimitBytes (so it evenly tiles the
  // batch); it collapses to 1 for the deep, many-channel layers and is a no-op
  // for the batch-squeezed CHW form (N == 1).
  int64_t elemBytesEarly =
      (conv1ResultType.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  int64_t nb = N;
  if (hasBatch) {
    int64_t inWindowPerImg =
        C1In * conv1InputTileH * conv1InputTileW * elemBytesEarly;
    int64_t maxNb =
        inWindowPerImg > 0 ? kStackLimitBytes / inWindowPerImg : N;
    nb = largestDivisorAtMost(N, std::max<int64_t>(1, maxNb));
  }

  // Bound ocTile so the ocTile-sized conv2 accumulator (and the epilogue2 /
  // write-back tiles that share its shape) fit their share of the stack
  // budget. Without this, a large output-channel tile (e.g. 512 for the deep
  // layers) alone would blow the per-workgroup stack sum even after the
  // producer/input windows are chunked. Re-snapped to a divisor of COut so the
  // forall still tiles the output channels evenly.
  //
  // ocTile is deliberately grown *past* the distribution hint, up to the full
  // COut, rather than clamped to it. The conv1 half of the chain is recomputed
  // once per output-channel tile -- the producer window depends on the spatial
  // tile only, not on which conv2 output channels are being produced -- so a
  // COut/ocTile factor of redundant conv1 work is paid for nothing. At
  // ocTile = COut the chain computes each conv1 window exactly once and only
  // the (much smaller) spatial halo is recomputed.
  // (ocTile is actually chosen below, once the producer chunk sizes it has
  // to share the stack with are known.)

  if (conv1InputType.isDynamicDim(B + 1) || conv1InputType.isDynamicDim(B + 2)) {
    LLVM_DEBUG({
      llvm::dbgs() << "[rewriteMatchAsForall] dynamic conv1 activation spatial dims not supported\n";
    });
    return failure();
  }

  // ==========================================================================
  // Producer-window anchoring: two-sided clamp, no padded activation.
  //
  // Every tile computes a *full*, statically-sized producerH x producerW
  // epilogue1 window anchored at epiH0/epiW0. The naive anchor
  // (rawStart = oh0*conv2StrideH - conv2PadH, clamped only on the low side)
  // runs past conv1's true H1/W1 extent on the *last* tile, which used to be
  // absorbed by padding conv1's activation with trailing zeros up front.
  //
  // That padding is what destroyed vectorization: folding an extract_slice at
  // a dynamic offset through a tensor.pad yields a *dynamically shaped*
  // in-bounds slice plus a residual pad, so GenericVectorization can no longer
  // emit one bulk transfer_read for the conv1 activation window. It degrades
  // to a per-element `scf.if` + vector<1xf32> read + insert_strided_slice
  // gather -- ~100 branchy scalar loads per (input channel, kh) iteration --
  // which measured ~60x slower than the unfused convolutions.
  //
  // Instead, clamp epiH0/epiW0 on the *high* side too, to H1 - producerH /
  // W1 - producerW. Then:
  //   * the conv1 activation window [epiH0*conv1StrideH, + conv1InputTileH)
  //     is exactly in bounds for every tile -- extract_slice stays fully
  //     static, so the window vectorizes as a single contiguous read;
  //   * the window is always fully inside conv1's real output, so *every*
  //     value in it is genuine -- the epilogue1 validH/validW mask that used
  //     to zero the over-read positions is no longer needed at all;
  //   * the price is that the last tile's window is shifted `tailMax`
  //     positions *earlier* than conv2 expects. That is corrected by the same
  //     shift buffer that already handles the low-side clamp, now reading back
  //     out at a dynamic offset instead of a fixed 0 (see [SHIFT] below).
  //
  // A tile larger than conv1's own output (H1 < producerH) leaves no room to
  // clamp high; that case keeps the old padded-activation scheme (and its
  // mask), which is correct, just slow.
  // ==========================================================================
  const int64_t maxRawStartH = (OH - tileH) * conv2StrideH - conv2PadH;
  const int64_t maxRawStartW = (OW - tileW) * conv2StrideW - conv2PadW;
  const bool clampHigh = H1 >= producerH && W1 >= producerW;
  // How far past conv1's extent the last tile's raw anchor lands -- i.e. how
  // many positions the high clamp pulls its window back by, and therefore the
  // largest offset the shift buffer must be read back out at.
  const int64_t tailMaxH =
      clampHigh ? std::max<int64_t>(0, maxRawStartH - (H1 - producerH)) : 0;
  const int64_t tailMaxW =
      clampHigh ? std::max<int64_t>(0, maxRawStartW - (W1 - producerW)) : 0;
  // Headroom the shift buffer needs: conv2's padding on the low side, the
  // high-clamp pull-back on the high side (see [SHIFT] below). Zero on both
  // axes means no shift buffer is ever built.
  const int64_t shiftExtraH = std::max<int64_t>(conv2PadH, tailMaxH);
  const int64_t shiftExtraW = std::max<int64_t>(conv2PadW, tailMaxW);

  // ==========================================================================
  // [BUDGET] Per-workgroup stack sizing: channel chunks first, then ocTile.
  //
  // The backend caps the *sum* of a function's stack allocations, so these
  // cannot be picked from independent per-buffer budgets -- an earlier version
  // did that and overflowed the cap on every ResNet layer once ocTile was
  // allowed to grow. Instead: size the producer-side chunks (which the spatial
  // tile fixes anyway), then spend whatever budget is left on ocTile, which is
  // the one knob that buys back redundant conv1 work.
  //
  // The estimate below counts only buffers that are really allocated: the
  // conv1 accumulator, epilogue1's result, the optional shift buffer, and the
  // ocTile-sized conv2 accumulator / epilogue2 tile. Weight and activation
  // windows are extract_slices of dispatch buffers -- subviews, not allocas --
  // so budgeting for them is what made the old model both wrong and tight.
  //
  // kStackModelBudgetBytes is *calibrated*, not derived: measured stack usage
  // runs ~1.7x this estimate (loop iter_arg/result pairs double several of
  // these buffers), so the estimate is held well under the real 32768 cap.
  // Re-measure before raising it -- the failure mode is a hard compile error.
  // ==========================================================================
  const int64_t perChanBytes = [&] {
    int64_t bytes = 2 * nb * producerH * producerW * elemBytesEarly;
    if (shiftExtraH != 0 || shiftExtraW != 0) {
      bytes += nb * (producerH + shiftExtraH) * (producerW + shiftExtraW) *
               elemBytesEarly;
    }
    return bytes;
  }();

  // ocTile is claimed FIRST, and is deliberately allowed to grow past the
  // distribution hint up to the full COut. conv1's producer window depends only
  // on the spatial tile, not on which conv2 output channels are being produced,
  // so the chain recomputes all of conv1 COut/ocTile times -- pure waste. At
  // ocTile == COut each conv1 window is computed exactly once and only the
  // (much smaller) spatial halo is redone.
  //
  // Order matters: sizing the channel chunk first and giving ocTile the
  // leftover measurably backfires. A larger budget then just grows channelChunk
  // (which only changes how the reduction is split, not how much work is done),
  // starving ocTile and making the dispatch SLOWER -- raising the budget from
  // 17408 to 22528 that way took conv-conv from 1.48ms to 2.07ms. channelChunk
  // therefore gets what is left, floored at one channel.
  {
    int64_t denom = std::max<int64_t>(1, nb * tileH * tileW * elemBytesEarly);
    // Always leave room for at least a single-channel producer chunk.
    int64_t ocBudget = kStackModelBudgetBytes - perChanBytes;
    int64_t maxOc = std::max<int64_t>(1, ocBudget / denom);
    ocTile = largestDivisorAtMost(COut, std::min(COut, maxOc));
  }

  // conv2's input-channel (= conv1's output-channel) chunk gets the remainder.
  const int64_t channelChunk = [&] {
    int64_t used = nb * ocTile * tileH * tileW * elemBytesEarly;
    int64_t maxCb = perChanBytes > 0
                        ? (kStackModelBudgetBytes - used) / perChanBytes
                        : CIn;
    return largestDivisorAtMost(CIn, std::max<int64_t>(1, maxCb));
  }();

  // conv1's input-channel (reduction) chunk. This window is a subview, so it
  // costs no stack; the chunking only keeps any single conv1 read vector from
  // exceeding the backend's per-vector limit.
  const int64_t conv1ChannelChunk = [&] {
    int64_t perCh = nb * conv1InputTileH * conv1InputTileW * elemBytesEarly;
    int64_t maxC1b = perCh > 0 ? (kStackLimitBytes / 4) / perCh : C1In;
    return largestDivisorAtMost(C1In, std::max<int64_t>(1, maxC1b));
  }();


  if (!clampHigh) {
    int64_t actH = conv1InputType.getDimSize(B + 1);
    int64_t actW = conv1InputType.getDimSize(B + 2);
    int64_t maxEpiH0 = std::max<int64_t>(0, maxRawStartH);
    int64_t maxEpiW0 = std::max<int64_t>(0, maxRawStartW);
    int64_t neededActH = maxEpiH0 * conv1StrideH + conv1InputTileH;
    int64_t neededActW = maxEpiW0 * conv1StrideW + conv1InputTileW;
    int64_t padH = std::max<int64_t>(0, neededActH - actH);
    int64_t padW = std::max<int64_t>(0, neededActW - actW);
    if (padH > 0 || padW > 0) {
      auto paddedType = RankedTensorType::get(
          withBatchShape(hasBatch, conv1InputType.getDimSize(0),
                        {conv1InputType.getDimSize(B), actH + padH,
                         actW + padW}),
          conv1InputType.getElementType());
      SmallVector<OpFoldResult> low(3 + B, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> high = withBatch(
          hasBatch, rewriter.getIndexAttr(0),
          {rewriter.getIndexAttr(0), rewriter.getIndexAttr(padH),
           rewriter.getIndexAttr(padW)});
      conv1Input = tensor::PadOp::create(rewriter, loc, paddedType, conv1Input,
                                         low, high, zero, /*nofold=*/false);
    }
  }

  // Final output init tensor for the new forall.
  Value outEmpty = tensor::EmptyOp::create(
      rewriter, loc,
      epilogue2ResultType.getShape(),
      epilogue2ResultType.getElementType());

  // Build outer forall over [(n,) oc, oh, ow]. When there is a batch dim
  // (NCHW), it is distributed too, with step nb (see the batch-tiling comment
  // above). A 4-dim workgroup forall is mapped the same way IREE's own
  // TileDispatchUsingForall does: the innermost 3 dims get x/y/z and any
  // further (outer) dim gets a delinearized IdZ mapping.
  const bool distributeBatch = hasBatch && nb < N;
  SmallVector<OpFoldResult> lbs, ubs, steps;
  if (distributeBatch) {
    lbs.push_back(rewriter.getIndexAttr(0));
    ubs.push_back(rewriter.getIndexAttr(N));
    steps.push_back(rewriter.getIndexAttr(nb));
  }
  lbs.append({rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
              rewriter.getIndexAttr(0)});
  ubs.append({rewriter.getIndexAttr(COut), rewriter.getIndexAttr(OH),
              rewriter.getIndexAttr(OW)});
  steps.append({rewriter.getIndexAttr(ocTile), rewriter.getIndexAttr(tileH),
                rewriter.getIndexAttr(tileW)});

  SmallVector<Attribute> mapping;
  if (distributeBatch) {
    // Outermost (batch) dim -> delinearized IdZ (relative index 1).
    mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
        rewriter.getContext(), IREE::Codegen::WorkgroupId::IdZ, /*idx=*/1));
  }
  mapping.append({IREE::Codegen::WorkgroupMappingAttr::get(
                      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdZ),
                  IREE::Codegen::WorkgroupMappingAttr::get(
                      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdY),
                  IREE::Codegen::WorkgroupMappingAttr::get(
                      rewriter.getContext(), IREE::Codegen::WorkgroupId::IdX)});

  std::optional<ArrayAttr> mappingAttr = rewriter.getArrayAttr(mapping);

  rewriter.setInsertionPoint(conv2);
  auto forallOp = scf::ForallOp::create(
      rewriter, loc, lbs, ubs, steps, ValueRange{outEmpty}, mappingAttr);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forallOp.getBody());

    auto ivs = forallOp.getInductionVars();
    size_t expectedIvs = distributeBatch ? 4 : 3;
    if (ivs.size() != expectedIvs) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] unexpected forall IV count\n";
      });
      return failure();
    }

    // Batch offset for global (activation/output) slices: the batch IV when
    // distributed, else 0 (whole batch processed per tile, or no batch dim).
    size_t sp = distributeBatch ? 1 : 0;
    OpFoldResult n0Ofr =
        distributeBatch ? OpFoldResult(ivs[0]) : rewriter.getIndexAttr(0);
    Value oc0 = ivs[sp];
    Value oh0 = ivs[sp + 1];
    Value ow0 = ivs[sp + 2];
    Value shared = forallOp.getRegionOutArgs().front();

    // ==========================================================================
    // [A] Valid-window bookkeeping for this conv2 output tile, mirroring
    // RewriteConvMaxpoolAsForall's implicit-padding scheme: a clamped-low
    // start (epiH0/epiW0) used for the actual extraction, plus
    // (insertH/insertW, validH/validW) recording where -- within the always-
    // full, statically-sized [producerH, producerW] local window -- the true,
    // non-garbage data lives. Positions outside
    // [insertH, insertH+validH) x [insertW, insertW+validW) are explicitly
    // zeroed by the mask appended to epilogue1's clone below, standing in for
    // conv2's own zero padding without ever materialising a padded tensor.
    // ==========================================================================
    Value cConv2SH = arith::ConstantIndexOp::create(rewriter, loc, conv2StrideH);
    Value cConv2SW = arith::ConstantIndexOp::create(rewriter, loc, conv2StrideW);
    Value cConv2PH = arith::ConstantIndexOp::create(rewriter, loc, conv2PadH);
    Value cConv2PW = arith::ConstantIndexOp::create(rewriter, loc, conv2PadW);
    Value cProducerH = arith::ConstantIndexOp::create(rewriter, loc, producerH);
    Value cProducerW = arith::ConstantIndexOp::create(rewriter, loc, producerW);
    Value cH1 = arith::ConstantIndexOp::create(rewriter, loc, H1);
    Value cW1 = arith::ConstantIndexOp::create(rewriter, loc, W1);
    Value c0idx = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // rawStart = oh0 * conv2StrideH - conv2PadH  (can be negative, or run
    // past H1 - producerH for the last tile).
    Value rawStartH = arith::SubIOp::create(
        rewriter, loc, arith::MulIOp::create(rewriter, loc, oh0, cConv2SH),
        cConv2PH);
    Value rawStartW = arith::SubIOp::create(
        rewriter, loc, arith::MulIOp::create(rewriter, loc, ow0, cConv2SW),
        cConv2PW);

    Value epiH0 = arith::MaxSIOp::create(rewriter, loc, rawStartH, c0idx);
    Value epiW0 = arith::MaxSIOp::create(rewriter, loc, rawStartW, c0idx);
    if (clampHigh) {
      // Pull the window back so it never runs past conv1's true extent; the
      // resulting misalignment is undone by the shift buffer's read offset.
      Value hiH = arith::ConstantIndexOp::create(rewriter, loc, H1 - producerH);
      Value hiW = arith::ConstantIndexOp::create(rewriter, loc, W1 - producerW);
      epiH0 = arith::MinSIOp::create(rewriter, loc, epiH0, hiH);
      epiW0 = arith::MinSIOp::create(rewriter, loc, epiW0, hiW);
    }

    // Where, inside the producerH x producerW window, the data conv2 expects
    // at its own local row/col 0 lives. `insert*` is how far the window had to
    // be clamped up at the low edge (conv2's leading zero padding); `read*` is
    // how far it had to be pulled back at the high edge. Exactly one of the
    // two is ever non-zero for a given tile.
    Value offH = arith::SubIOp::create(rewriter, loc, rawStartH, epiH0);
    Value offW = arith::SubIOp::create(rewriter, loc, rawStartW, epiW0);
    Value insertH = arith::MaxSIOp::create(
        rewriter, loc, arith::SubIOp::create(rewriter, loc, c0idx, offH),
        c0idx);
    Value insertW = arith::MaxSIOp::create(
        rewriter, loc, arith::SubIOp::create(rewriter, loc, c0idx, offW),
        c0idx);
    Value readH = arith::MaxSIOp::create(rewriter, loc, offH, c0idx);
    Value readW = arith::MaxSIOp::create(rewriter, loc, offW, c0idx);

    // Only meaningful for the !clampHigh (padded-activation) path, where the
    // window can over-read past conv1's (H1, W1) and those positions must be
    // masked out of epilogue1's result.
    Value validH, validW;
    if (!clampHigh) {
      Value epiH1 = arith::MinSIOp::create(
          rewriter, loc,
          arith::AddIOp::create(rewriter, loc, rawStartH, cProducerH), cH1);
      Value epiW1 = arith::MinSIOp::create(
          rewriter, loc,
          arith::AddIOp::create(rewriter, loc, rawStartW, cProducerW), cW1);
      validH = arith::SubIOp::create(rewriter, loc, epiH1, epiH0);
      validW = arith::SubIOp::create(rewriter, loc, epiW1, epiW0);
    }

    // ==========================================================================
    // [B] conv1 input window position: always safe/static thanks to the
    // upfront padding above; only the low side needs runtime clamping (via
    // epiH0/epiW0 above). stage1's resulting misalignment relative to what
    // conv2 naturally expects (see below) is corrected after epilogue1, via
    // an explicit shift into a small zero-initialised buffer -- not here.
    // ==========================================================================
    Value cConv1SH = arith::ConstantIndexOp::create(rewriter, loc, conv1StrideH);
    Value cConv1SW = arith::ConstantIndexOp::create(rewriter, loc, conv1StrideW);
    Value convInIh0 = arith::MulIOp::create(rewriter, loc, epiH0, cConv1SH);
    Value convInIw0 = arith::MulIOp::create(rewriter, loc, epiW0, cConv1SW);

    OpFoldResult zeroOfr = rewriter.getIndexAttr(0);
    // Batch-tile size for every tile-local slice/shape (= nb, the batch step).
    OpFoldResult nOfr = rewriter.getIndexAttr(nb);

    // The conv1 input window [nb, C1In, conv1InputTileH, conv1InputTileW] is NOT
    // extracted whole -- for the deeper layers its full-input-channel extent
    // (e.g. C1In=512) exceeds the backend's per-vector limit on its own,
    // independent of any spatial or batch tiling. It is instead sliced per
    // conv1 input-channel chunk inside the reduction loop below (see [D]). Only
    // the shared strides and window origin (convInIh0/convInIw0) are needed
    // here.
    SmallVector<OpFoldResult> conv1InputStrides(3 + B,
                                                rewriter.getIndexAttr(1));

    // ==========================================================================
    // [CHUNK] Channel-reduction chunking.
    //
    // conv2 reduces over its input channels (= conv1 output channels, CIn).
    // Materialising the full [CIn, producerH, producerW] tile overflows the
    // stack, so we split the reduction into chunks of Cb channels. Each chunk
    // produces only conv1 output channels [c, c+Cb) -> a [Cb, ...] tile, and
    // conv2 accumulates that chunk's partial reduction into the loop iter_arg.
    //
    // Cb is derived (not hard-coded) as the largest divisor of CIn whose
    // per-chunk resident buffers fit kStackLimitBytes. Two same-sized
    // [Cb, producerH, producerW] buffers are live at once per chunk: conv1's
    // zero-fill init and epilogue1's (masked) result. When conv2 has actual
    // padding, a third, slightly larger [Cb, producerH+conv2PadH,
    // producerW+conv2PadW] buffer is also live briefly (the shift-into-
    // zero-buffer step below, needed to realign epilogue1's output with what
    // conv2's unshifted read expects at boundary tiles) -- included here so
    // Cb still shrinks enough to keep the *sum* under budget.
    // ==========================================================================
    // The backend's kStackLimitBytes cap applies to the *sum* of every live
    // per-workgroup allocation, so the budget is split across the buffer
    // classes that coexist inside the chunk loops rather than handed whole to
    // each. The producer-sized buffers (conv1 accumulator, epilogue1 result,
    // and the optional shift buffer -- all ~[nb, Cb, producerH, producerW])
    // dominate and get the largest share; the conv1 input window [nb, C1b,
    // ...] and the conv2 accumulator / weight tiles get the rest. These are
    // deliberately conservative fractions (they need not be tight -- only sum
    // to < kStackLimitBytes with headroom for the smaller weight tiles).
    // channelChunk / conv1ChannelChunk / ocTile were all sized together in
    // [BUDGET] above, against the shared stack cap.

    // conv2 accumulator (zero-initialised), carried across channel chunks.
    auto conv2AccType = RankedTensorType::get(
        withBatchShape(hasBatch, nb, {ocTile, tileH, tileW}),
        conv2ResultType.getElementType());
    Value conv2AccEmpty = tensor::EmptyOp::create(
        rewriter, loc, conv2AccType.getShape(), conv2AccType.getElementType());
    auto conv2AccFillOp =
        linalg::FillOp::create(rewriter, loc, zero, conv2AccEmpty);
    if (fillLoweringConfig)
      conv2AccFillOp->setAttr("lowering_config", fillLoweringConfig);
    Value conv2AccInit = conv2AccFillOp.getResult(0);

    Value cChunkLb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value cChunkUb = arith::ConstantIndexOp::create(rewriter, loc, CIn);
    Value cChunkStep =
        arith::ConstantIndexOp::create(rewriter, loc, channelChunk);
    auto chunkLoop = scf::ForOp::create(
        rewriter, loc, cChunkLb, cChunkUb, cChunkStep,
        ValueRange{conv2AccInit});

    // All tile-local conv1/epilogue1/conv2 work happens inside the chunk
    // loop body; insertion point moves there now and is restored after the
    // loop (before epilogue2 / write-back).
    rewriter.setInsertionPointToStart(chunkLoop.getBody());
    Value c = chunkLoop.getInductionVar();
    Value conv2Acc = chunkLoop.getRegionIterArgs().front();

    auto stage1TileType = RankedTensorType::get(
        withBatchShape(hasBatch, nb, {channelChunk, producerH, producerW}),
        conv1ResultType.getElementType());

    Value stage1InitEmpty = tensor::EmptyOp::create(
        rewriter, loc,
        stage1TileType.getShape(),
        stage1TileType.getElementType());

    auto stage1FillOp = linalg::FillOp::create(rewriter, loc, zero, stage1InitEmpty);
    if (fillLoweringConfig)
      stage1FillOp->setAttr("lowering_config", fillLoweringConfig);
    Value stage1ConvInit = stage1FillOp.getResult(0);

    // ==========================================================================
    // [D] conv1 tile-local op. conv1's reduction over its input channels (C1In)
    // is CHUNKED (step conv1ChannelChunk) so no single read/compute vector is
    // oversized for the deep, many-channel layers. Each chunk clones conv1 on
    // an input window [nb, C1b, conv1InputTileH, conv1InputTileW] and weight
    // [Cb, C1b, k1h, k1w], accumulating its partial products into a running
    // conv1 accumulator (seeded by the zero-fill stage1ConvInit). The epilogue
    // below runs only after this loop, on the fully-reduced conv1 output --
    // exactly as it would on the un-chunked conv1.
    Value cc1Lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value cc1Ub = arith::ConstantIndexOp::create(rewriter, loc, C1In);
    Value cc1Step =
        arith::ConstantIndexOp::create(rewriter, loc, conv1ChannelChunk);
    auto conv1RedLoop = scf::ForOp::create(rewriter, loc, cc1Lb, cc1Ub, cc1Step,
                                           ValueRange{stage1ConvInit});
    {
      OpBuilder::InsertionGuard g1(rewriter);
      rewriter.setInsertionPointToStart(conv1RedLoop.getBody());
      Value cc = conv1RedLoop.getInductionVar();
      Value conv1Acc = conv1RedLoop.getRegionIterArgs().front();

      // conv1 input window for input-channel chunk [cc, cc+C1b).
      SmallVector<OpFoldResult> conv1InOffsets = withBatch(
          hasBatch, n0Ofr, {cc, convInIh0, convInIw0});
      SmallVector<OpFoldResult> conv1InSizes = withBatch(
          hasBatch, nOfr,
          {rewriter.getIndexAttr(conv1ChannelChunk),
           rewriter.getIndexAttr(conv1InputTileH),
           rewriter.getIndexAttr(conv1InputTileW)});
      auto conv1InChunkType = RankedTensorType::get(
          withBatchShape(hasBatch, nb,
                         {conv1ChannelChunk, conv1InputTileH, conv1InputTileW}),
          conv1InputType.getElementType());
      Value conv1InChunk = tensor::ExtractSliceOp::create(
          rewriter, loc, conv1InChunkType, conv1Input, conv1InOffsets,
          conv1InSizes, conv1InputStrides);

      // conv1 weight for out-channels [c, c+Cb) x in-channels [cc, cc+C1b).
      SmallVector<OpFoldResult> conv1WeightOffsets = {
          c, cc, rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
      SmallVector<OpFoldResult> conv1WeightSizes = {
          rewriter.getIndexAttr(channelChunk),
          rewriter.getIndexAttr(conv1ChannelChunk),
          rewriter.getIndexAttr(k1h), rewriter.getIndexAttr(k1w)};
      SmallVector<OpFoldResult> conv1WeightStrides(4, rewriter.getIndexAttr(1));
      auto conv1WeightChunkType = RankedTensorType::get(
          {channelChunk, conv1ChannelChunk, k1h, k1w},
          conv1WeightType.getElementType());
      Value conv1WeightChunk = tensor::ExtractSliceOp::create(
          rewriter, loc, conv1WeightChunkType, conv1Weight, conv1WeightOffsets,
          conv1WeightSizes, conv1WeightStrides);

      SmallVector<Value> conv1TiledOperands = {conv1InChunk, conv1WeightChunk,
                                               conv1Acc};
      Operation *conv1TileOp =
          clone(rewriter, conv1.getOperation(),
                TypeRange{stage1TileType}, conv1TiledOperands);
      if (!conv1TileOp) {
        LLVM_DEBUG({
          llvm::dbgs() << "[rewriteMatchAsForall] failed to clone tiled conv1\n";
        });
        return failure();
      }
      SmallVector<OpFoldResult> conv1LoopOffsets = withBatch(
          hasBatch, n0Ofr,
          {c, epiH0, epiW0, cc, rewriter.getIndexAttr(0),
           rewriter.getIndexAttr(0)});
      linalg::offsetIndices(rewriter, cast<linalg::LinalgOp>(conv1TileOp),
                            conv1LoopOffsets);
      Value conv1Result = conv1TileOp->getResult(0);
      bool conv1HasBatch = hasBatch;
      if (!hasBatch) {
        rewriter.setInsertionPointAfter(conv1TileOp);
        ExpandedConv expanded = expandConvWithUnitBatch(
            rewriter, loc, cast<linalg::GenericOp>(conv1TileOp));
        if (expanded.op) {
          conv1TileOp = expanded.op;
          conv1Result = expanded.chwResult;
          conv1HasBatch = true;
        }
        rewriter.setInsertionPointAfterValue(conv1Result);
      }
      setConvVectorTiles(conv1TileOp, conv1HasBatch,
                         /*ocTile=*/std::min<int64_t>(kConvVecOcTile,
                                                      channelChunk),
                         /*owTile=*/kConvVecOwTile,
                         /*cinTile=*/std::min<int64_t>(kConvVecCinTile,
                                                       conv1ChannelChunk));
      scf::YieldOp::create(rewriter, loc, ValueRange{conv1Result});
    }
    Value conv1Tile = conv1RedLoop.getResult(0);

    // ==========================================================================
    // [E] epilogue1 tile-local op starts here.
    // Replace old conv1 result operand with conv1Tile, tile the others by map,
    // then clone original epilogue1, then append a boundary mask that zeroes
    // any position outside the true valid window -- implicit padding for
    // conv2's own halo, folded directly into epilogue1's output.
    // ==========================================================================
    SmallVector<Value> epi1Inputs(epilogue1.getDpsInputs());
    SmallVector<Value> epi1Inits(epilogue1.getDpsInits());
    SmallVector<AffineMap> epi1Maps = epilogue1.getIndexingMapsArray();

    unsigned epi1NumInputs = epi1Inputs.size();
    unsigned epi1NumInits = epi1Inits.size();

    if (epi1Maps.size() != epi1NumInputs + epi1NumInits) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] epilogue1 indexing_maps count mismatch\n";
      });
      return failure();
    }

    if (epi1NumInits != 1 || epilogue1->getNumResults() != 1) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] only single-output epilogue1 is supported\n";
      });
      return failure();
    }

    ArrayRef<AffineMap> epi1InputMaps =
        ArrayRef<AffineMap>(epi1Maps).take_front(epi1NumInputs);
    ArrayRef<AffineMap> epi1InitMaps =
        ArrayRef<AffineMap>(epi1Maps).drop_front(epi1NumInputs);

    SmallVector<OpFoldResult> stage1LoopOffsets =
        withBatch(hasBatch, n0Ofr, {c, epiH0, epiW0});
    SmallVector<OpFoldResult> stage1LoopSizes = withBatch(
        hasBatch, nOfr,
        {rewriter.getIndexAttr(channelChunk), rewriter.getIndexAttr(producerH),
         rewriter.getIndexAttr(producerW)});

    SmallVector<Value> tiledEpi1Inputs;
    tiledEpi1Inputs.reserve(epi1NumInputs);

    Value oldConv1Result = conv1.getResult(0);

    for (auto [operand, operandMap] : llvm::zip(epi1Inputs, epi1InputMaps)) {
      if (operand == oldConv1Result) {
        tiledEpi1Inputs.push_back(conv1Tile);
        continue;
      }

      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType) {
        LLVM_DEBUG({
          llvm::dbgs() << "[rewriteMatchAsForall] epilogue1 input is not ranked tensor\n";
        });
        return failure();
      }

      SmallVector<OpFoldResult> operandOffsets;
      SmallVector<OpFoldResult> operandSizes;
      SmallVector<OpFoldResult> operandStrides;

      if (failed(getOperandSliceFromIndexingMap(
              operandMap, stage1LoopOffsets, stage1LoopSizes,
              operandOffsets, operandSizes, operandStrides))) {
        LLVM_DEBUG({
          llvm::dbgs() << "[rewriteMatchAsForall] failed to infer tiled slice for epilogue1 input\n";
        });
        return failure();
      }

      SmallVector<int64_t> staticShape;
      staticShape.reserve(operandSizes.size());
      for (OpFoldResult ofr : operandSizes) {
        auto attr = dyn_cast<Attribute>(ofr);
        auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
        if (!intAttr) {
          LLVM_DEBUG({
            llvm::dbgs() << "[rewriteMatchAsForall] expected static tile size for epilogue1 input\n";
          });
          return failure();
        }
        staticShape.push_back(intAttr.getInt());
      }

      auto tiledOperandType =
          RankedTensorType::get(staticShape, operandType.getElementType());

      Value tiledOperand = tensor::ExtractSliceOp::create(
          rewriter, loc, tiledOperandType, operand,
          operandOffsets, operandSizes, operandStrides);

      tiledEpi1Inputs.push_back(tiledOperand);
    }

    AffineMap epi1InitMap = epi1InitMaps.front();

    SmallVector<OpFoldResult> epi1InitOffsets;
    SmallVector<OpFoldResult> epi1InitSizes;
    SmallVector<OpFoldResult> epi1InitStrides;

    if (failed(getOperandSliceFromIndexingMap(
            epi1InitMap, stage1LoopOffsets, stage1LoopSizes,
            epi1InitOffsets, epi1InitSizes, epi1InitStrides))) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] failed to infer tiled slice for epilogue1 init/output\n";
      });
      return failure();
    }

    auto epi1InitType = dyn_cast<RankedTensorType>(epi1Inits.front().getType());
    if (!epi1InitType) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] epilogue1 init is not ranked tensor\n";
      });
      return failure();
    }

    SmallVector<int64_t> epi1InitStaticShape;
    epi1InitStaticShape.reserve(epi1InitSizes.size());
    for (OpFoldResult ofr : epi1InitSizes) {
      auto attr = dyn_cast<Attribute>(ofr);
      auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
      if (!intAttr) {
        LLVM_DEBUG({
          llvm::dbgs() << "[rewriteMatchAsForall] expected static tile size for epilogue1 init/output\n";
        });
        return failure();
      }
      epi1InitStaticShape.push_back(intAttr.getInt());
    }

    auto epi1TileType =
        RankedTensorType::get(epi1InitStaticShape, epi1InitType.getElementType());

    Value epi1Init = tensor::EmptyOp::create(
        rewriter, loc,
        epi1TileType.getShape(),
        epi1TileType.getElementType());

    SmallVector<Value> epi1TiledOperands = tiledEpi1Inputs;
    epi1TiledOperands.push_back(epi1Init);

    SmallVector<Type> epi1ResultTensorTypes = {epi1TileType};

    Operation *epi1TileOp = clone(
        rewriter, epilogue1.getOperation(),
        epi1ResultTensorTypes, epi1TiledOperands);

    if (!epi1TileOp) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] failed to clone tiled epilogue1\n";
      });
      return failure();
    }

    SmallVector<OpFoldResult> epi1LoopOffsets =
        withBatch(hasBatch, n0Ofr, {c, epiH0, epiW0});
    linalg::offsetIndices(
        rewriter, cast<linalg::LinalgOp>(epi1TileOp), epi1LoopOffsets);

    // Append a mask directly into epilogue1's cloned body: replace its
    // yielded value with a select against [0, validH) x [0, validW), the
    // sub-range (of this epiH0/epiW0-anchored, always-full producerH x
    // producerW window) that actually falls within conv1's true (H1, W1)
    // extent -- rows/cols at or beyond validH/validW hold a well-defined but
    // meaningless value (the window over-read past H1/W1 thanks to the
    // padding above) and must not leak into conv2's reduction.
    //
    // Under the two-sided clamp there is no over-read: every position in the
    // window is a genuine conv1 output, so the mask is skipped entirely.
    if (!clampHigh) {
      Block &epi1Block = epi1TileOp->getRegion(0).front();
      auto epi1Yield = cast<linalg::YieldOp>(epi1Block.getTerminator());
      OpBuilder::InsertionGuard maskGuard(rewriter);
      rewriter.setInsertionPoint(epi1Yield);
      Value lh = linalg::IndexOp::create(rewriter, loc, B + 1);
      Value lw = linalg::IndexOp::create(rewriter, loc, B + 2);
      Value hOk = arith::CmpIOp::create(rewriter, loc,
                                        arith::CmpIPredicate::slt, lh, validH);
      Value wOk = arith::CmpIOp::create(rewriter, loc,
                                        arith::CmpIPredicate::slt, lw, validW);
      Value inBnd = arith::AndIOp::create(rewriter, loc, hOk, wOk);
      Value orig = epi1Yield.getOperand(0);
      Value masked = arith::SelectOp::create(rewriter, loc, inBnd, orig, zero);
      epi1Yield.setOperand(0, masked);
    }

    // epi1TileOp's (masked) result holds r1[epiH0 + j] at local row j (zero
    // for j >= validH, i.e. past conv1's true extent) -- but conv2 below is
    // cloned as an ordinary, unmodified linalg op, reading its activation
    // with plain 0-based indexing (local output row r reads local rows
    // [r*strideH, r*strideH+kH)), exactly as if local row 0 *were*
    // rawStartH. For interior tiles epiH0 == rawStartH so this is already
    // correct; at a boundary tile (epiH0 clamped to 0, rawStartH < 0) it
    // isn't -- conv2 would read insertH rows further into real data than it
    // should, silently corrupting exactly the boundary tiles. Shift the
    // (masked) data into a small zero-initialised buffer at offset
    // insertH/insertW to correct this: the buffer is sized with `conv2PadH`/
    // `conv2PadW` extra headroom (a compile-time constant, and a tight
    // bound: insertH/insertW can never exceed it), so the insert -- at a
    // *dynamic* offset, but always a static size -- is always in bounds, and
    // extracting back out at a fixed offset 0 gives a stage1Tile whose local
    // row j now equals r1[rawStartH + j] (zero if out of bounds), matching
    // what conv2's unshifted read expects.
    //
    // [SHIFT] Under the two-sided clamp the same buffer also absorbs the
    // high-edge pull-back: the window is inserted at insertH/insertW (low
    // clamp) and read back out at readH/readW (high clamp), so local row j of
    // stage1Tile is r1[rawStartH + j] either way, zero where that falls
    // outside conv1. The buffer therefore needs headroom for whichever clamp
    // is larger -- conv2Pad on the low side, tailMax on the high side.
    Value stage1Tile;
    if (shiftExtraH == 0 && shiftExtraW == 0) {
      // Neither clamp can ever fire (rawStart >= 0 always, and the window
      // never runs past H1/W1), so insert/read offsets are statically zero --
      // skip the extra buffer entirely.
      stage1Tile = epi1TileOp->getResult(0);
    } else {
      SmallVector<OpFoldResult> strides1(3 + B, rewriter.getIndexAttr(1));
      SmallVector<int64_t> shiftBufShape = withBatchShape(
          hasBatch, nb,
          {channelChunk, producerH + shiftExtraH, producerW + shiftExtraW});
      Value shiftBufEmpty = tensor::EmptyOp::create(rewriter, loc,
                                                     shiftBufShape,
                                                     conv1ResultType.getElementType());
      Value shiftBufZero =
          linalg::FillOp::create(rewriter, loc, zero, shiftBufEmpty)
              .getResult(0);
      SmallVector<OpFoldResult> shiftInOffsets =
          withBatch(hasBatch, zeroOfr, {zeroOfr, insertH, insertW});
      SmallVector<OpFoldResult> shiftInSizes = withBatch(
          hasBatch, nOfr,
          {rewriter.getIndexAttr(channelChunk), rewriter.getIndexAttr(producerH),
           rewriter.getIndexAttr(producerW)});
      Value shifted = tensor::InsertSliceOp::create(
          rewriter, loc, epi1TileOp->getResult(0), shiftBufZero,
          shiftInOffsets, shiftInSizes, strides1);
      SmallVector<OpFoldResult> shiftOutOffsets =
          withBatch(hasBatch, zeroOfr, {zeroOfr, readH, readW});
      stage1Tile = tensor::ExtractSliceOp::create(
          rewriter, loc,
          RankedTensorType::get(
              withBatchShape(hasBatch, nb, {channelChunk, producerH, producerW}),
              conv1ResultType.getElementType()),
          shifted, shiftOutOffsets, shiftInSizes, strides1);
    }

    // ------------------------------------------------------------------
    // conv2 weight tile: [ocTile, Cb, kH, kW] — input-channel chunk [c, c+Cb)
    // ------------------------------------------------------------------
    SmallVector<OpFoldResult> weightOffsets = {
        oc0,
        c,
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> weightSizes = {
        rewriter.getIndexAttr(ocTile),
        rewriter.getIndexAttr(channelChunk),
        rewriter.getIndexAttr(kH),
        rewriter.getIndexAttr(kW)};
    SmallVector<OpFoldResult> weightStrides = {
        rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};

    auto conv2WeightTileType = RankedTensorType::get(
        {ocTile, channelChunk, kH, kW},
        conv2WeightType.getElementType());

    Value conv2WeightTile = tensor::ExtractSliceOp::create(
        rewriter, loc, conv2WeightTileType, conv2Weight,
        weightOffsets, weightSizes, weightStrides);

    // ------------------------------------------------------------------
    // conv2 partial tile: [ocTile, tileH, tileW]
    // Accumulate this channel chunk's partial reduction into the loop iter_arg
    // (conv2Acc). No per-chunk fill: the running accumulator carries the sum.
    // ------------------------------------------------------------------
    auto conv2TileType = RankedTensorType::get(
        withBatchShape(hasBatch, nb, {ocTile, tileH, tileW}),
        conv2ResultType.getElementType());

    SmallVector<Value> conv2TiledOperands = {
        stage1Tile, conv2WeightTile, conv2Acc};

    SmallVector<Type> conv2ResultTensorTypes = {conv2TileType};

    Operation *conv2TileOp = clone(
        rewriter, cast<linalg::LinalgOp>(conv2.getOperation()),
        conv2ResultTensorTypes, conv2TiledOperands);

    if (!conv2TileOp) {
      LLVM_DEBUG({
        llvm::dbgs() << "[rewriteMatchAsForall] failed to clone tiled conv2\n";
      });
      return failure();
    }

    SmallVector<OpFoldResult> conv2LoopOffsets = withBatch(
        hasBatch, n0Ofr,
        {oc0, oh0, ow0, rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
         rewriter.getIndexAttr(0)});
    linalg::offsetIndices(
        rewriter, cast<linalg::LinalgOp>(conv2TileOp), conv2LoopOffsets);

    Value conv2Result = conv2TileOp->getResult(0);
    bool conv2HasBatch = hasBatch;
    if (!hasBatch) {
      rewriter.setInsertionPointAfter(conv2TileOp);
      ExpandedConv expanded = expandConvWithUnitBatch(
          rewriter, loc, cast<linalg::GenericOp>(conv2TileOp));
      if (expanded.op) {
        conv2TileOp = expanded.op;
        conv2Result = expanded.chwResult;
        conv2HasBatch = true;
      }
      rewriter.setInsertionPointAfterValue(conv2Result);
    }
    setConvVectorTiles(conv2TileOp, conv2HasBatch,
                       /*ocTile=*/std::min<int64_t>(kConvVecOcTile, ocTile),
                       /*owTile=*/kConvVecOwTile,
                       /*cinTile=*/std::min<int64_t>(kConvVecCinTile,
                                                     channelChunk));

    // Yield the accumulated partial result and close the channel-chunk loop.
    scf::YieldOp::create(rewriter, loc, ValueRange{conv2Result});
    rewriter.setInsertionPointAfter(chunkLoop);

    Value conv2PartialTile = chunkLoop.getResult(0);

    Value finalTile;

    if (epilogue2) {
      // ------------------------------------------------------------------
      // epilogue2 tile-local inputs
      // Do not assume operand semantics.
      // Only special-case the operand that is exactly the old conv2 result:
      // replace it with the newly created conv2PartialTile.
      // ------------------------------------------------------------------
      SmallVector<Value> epi2Inputs(epilogue2.getDpsInputs());
      SmallVector<Value> epi2Inits(epilogue2.getDpsInits());
      SmallVector<AffineMap> epi2Maps = epilogue2.getIndexingMapsArray();
      SmallVector<utils::IteratorType> epi2IterTypes =
          epilogue2.getIteratorTypesArray();

      unsigned numInputs = epi2Inputs.size();
      unsigned numInits = epi2Inits.size();

      if (epi2Maps.size() != numInputs + numInits) {
        LLVM_DEBUG({
          llvm::dbgs()
              << "[rewriteMatchAsForall] epilogue2 indexing_maps count mismatch\n";
        });
        return failure();
      }

      if (numInits != 1 || epilogue2->getNumResults() != 1) {
        LLVM_DEBUG({
          llvm::dbgs()
              << "[rewriteMatchAsForall] only single-output epilogue2 is supported\n";
        });
        return failure();
      }

      ArrayRef<AffineMap> epi2InputMaps =
          ArrayRef<AffineMap>(epi2Maps).take_front(numInputs);
      ArrayRef<AffineMap> epi2InitMaps =
          ArrayRef<AffineMap>(epi2Maps).drop_front(numInputs);

      SmallVector<OpFoldResult> loopOffsets =
          withBatch(hasBatch, n0Ofr, {oc0, oh0, ow0});
      SmallVector<OpFoldResult> loopSizes = withBatch(
          hasBatch, nOfr,
          {rewriter.getIndexAttr(ocTile), rewriter.getIndexAttr(tileH),
           rewriter.getIndexAttr(tileW)});

      SmallVector<Value> tiledEpi2Inputs;
      tiledEpi2Inputs.reserve(numInputs);

      Value oldConv2Result = conv2.getResult(0);

      for (auto [operand, operandMap] : llvm::zip(epi2Inputs, epi2InputMaps)) {
        if (operand == oldConv2Result) {
          tiledEpi2Inputs.push_back(conv2PartialTile);
          continue;
        }

        auto operandType = dyn_cast<RankedTensorType>(operand.getType());
        if (!operandType) {
          LLVM_DEBUG({
            llvm::dbgs()
                << "[rewriteMatchAsForall] epilogue2 input is not ranked tensor\n";
          });
          return failure();
        }

        SmallVector<OpFoldResult> operandOffsets;
        SmallVector<OpFoldResult> operandSizes;
        SmallVector<OpFoldResult> operandStrides;

        if (failed(getOperandSliceFromIndexingMap(
                operandMap, loopOffsets, loopSizes, operandOffsets,
                operandSizes, operandStrides))) {
          LLVM_DEBUG({
            llvm::dbgs()
                << "[rewriteMatchAsForall] failed to infer tiled slice for epilogue2 input\n";
          });
          return failure();
        }

        SmallVector<int64_t> staticShape;
        staticShape.reserve(operandSizes.size());
        for (OpFoldResult ofr : operandSizes) {
          auto attr = dyn_cast<Attribute>(ofr);
          auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
          if (!intAttr) {
            LLVM_DEBUG({
              llvm::dbgs()
                  << "[rewriteMatchAsForall] expected static tile size for epilogue2 input\n";
            });
            return failure();
          }
          staticShape.push_back(intAttr.getInt());
        }

        auto tiledOperandType =
            RankedTensorType::get(staticShape, operandType.getElementType());

        Value tiledOperand = tensor::ExtractSliceOp::create(
            rewriter, loc, tiledOperandType, operand, operandOffsets,
            operandSizes, operandStrides);

        tiledEpi2Inputs.push_back(tiledOperand);
      }

      // ------------------------------------------------------------------
      // epilogue2 output/init tile
      // ------------------------------------------------------------------
      AffineMap epi2InitMap = epi2InitMaps.front();

      SmallVector<OpFoldResult> initOffsets;
      SmallVector<OpFoldResult> initSizes;
      SmallVector<OpFoldResult> initStrides;

      if (failed(getOperandSliceFromIndexingMap(
              epi2InitMap, loopOffsets, loopSizes, initOffsets, initSizes,
              initStrides))) {
        LLVM_DEBUG({
          llvm::dbgs()
              << "[rewriteMatchAsForall] failed to infer tiled slice for epilogue2 init/output\n";
        });
        return failure();
      }

      auto initType = dyn_cast<RankedTensorType>(epi2Inits.front().getType());
      if (!initType) {
        LLVM_DEBUG({
          llvm::dbgs()
              << "[rewriteMatchAsForall] epilogue2 init is not ranked tensor\n";
        });
        return failure();
      }

      SmallVector<int64_t> initStaticShape;
      initStaticShape.reserve(initSizes.size());
      for (OpFoldResult ofr : initSizes) {
        auto attr = dyn_cast<Attribute>(ofr);
        auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
        if (!intAttr) {
          LLVM_DEBUG({
            llvm::dbgs()
                << "[rewriteMatchAsForall] expected static tile size for epilogue2 init/output\n";
          });
          return failure();
        }
        initStaticShape.push_back(intAttr.getInt());
      }

      auto epi2TileType =
          RankedTensorType::get(initStaticShape, initType.getElementType());

      // elementwise epilogue: no fill needed
      Value epi2Init = tensor::EmptyOp::create(
          rewriter, loc, epi2TileType.getShape(), epi2TileType.getElementType());

      SmallVector<Value> epi2TiledOperands = tiledEpi2Inputs;
      epi2TiledOperands.push_back(epi2Init);

      SmallVector<Type> epi2ResultTensorTypes = {epi2TileType};

      Operation *epi2Op = clone(
          rewriter, epilogue2.getOperation(), epi2ResultTensorTypes,
          epi2TiledOperands);

      if (!epi2Op) {
        LLVM_DEBUG({
          llvm::dbgs()
              << "[rewriteMatchAsForall] failed to clone tiled epilogue2\n";
        });
        return failure();
      }

      SmallVector<OpFoldResult> epi2LoopOffsets =
          withBatch(hasBatch, n0Ofr, {oc0, oh0, ow0});
      linalg::offsetIndices(
          rewriter, cast<linalg::LinalgOp>(epi2Op), epi2LoopOffsets);

      finalTile = epi2Op->getResult(0);
    } else {
      // No epilogue2: conv2 tile is already the final tile.
      finalTile = conv2PartialTile;
    }

    // ------------------------------------------------------------------
    // write back
    // ------------------------------------------------------------------
    auto *inParallelBlock = &forallOp.getTerminator().getRegion().front();
    rewriter.setInsertionPointToStart(inParallelBlock);

    SmallVector<OpFoldResult> finalOffsets =
        withBatch(hasBatch, n0Ofr, {oc0, oh0, ow0});
    SmallVector<OpFoldResult> finalSizes = withBatch(
        hasBatch, nOfr,
        {rewriter.getIndexAttr(ocTile), rewriter.getIndexAttr(tileH),
         rewriter.getIndexAttr(tileW)});
    SmallVector<OpFoldResult> finalStrides(3 + B, rewriter.getIndexAttr(1));

    tensor::ParallelInsertSliceOp::create(
        rewriter, loc, finalTile, shared,
        finalOffsets, finalSizes, finalStrides);
  }

  // Replace old final result with new forall result.
  Value newResult = forallOp.getResult(0);
  Value oldFinalResult =
      epilogue2 ? epilogue2.getResult(0) : conv2.getResult(0);
  oldFinalResult.replaceAllUsesWith(newResult);

  // Then erase old conv2 and optional epilogue2.
  if (epilogue2)
    rewriter.eraseOp(epilogue2.getOperation());
  rewriter.eraseOp(conv2.getOperation());

  return success();
}

struct RewriteConvChainAsForallPass final
    : impl::RewriteConvChainAsForallPassBase<
          RewriteConvChainAsForallPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    if (funcOp->hasAttr(kConvChainRewrittenAttr)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[RewriteConvChainAsForallPass] skip: already has attr "
                     << kConvChainRewrittenAttr << "\n";
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

    if (!markedRoot) {
      return;
    }

    // Root with exactly 2 reduction loops is a maxpool (kh, kw only).
    // Handle by iree-codegen-rewrite-conv-maxpool-as-forall pass.
    if (auto rootGeneric = dyn_cast<linalg::GenericOp>(markedRoot);
        rootGeneric && rootGeneric.getNumReductionLoops() == 2) {
      return;
    }

    FailureOr<ConvChainMatch> match = matchConvChainFromRoot(markedRoot);
    if (failed(match)) {
      return;
    }

    // The outer forall tiling should follow conv2, not conv1.
    auto tilableRoot = dyn_cast<TilingInterface>(match->conv2.getOperation());
    if (!tilableRoot) {
      LLVM_DEBUG({
        llvm::dbgs() << "[RewriteConvChainAsForallPass] conv2 is not a TilingInterface op\n";
      });
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(match->conv2);

    if (failed(rewriteMatchAsForall(rewriter, *match))) {
      LLVM_DEBUG({
        llvm::dbgs() << "[RewriteConvChainAsForallPass] rewrite failed\n";
      });
      signalPassFailure();
      return;
    }

    funcOp->setAttr(kConvChainRewrittenAttr,
                    UnitAttr::get(funcOp.getContext()));
  }
};

} // namespace
} // namespace mlir::iree_compiler
