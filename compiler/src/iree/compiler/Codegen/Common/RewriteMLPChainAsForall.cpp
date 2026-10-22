//===----------------------------------------------------------------------===//
// RewriteMLPChainAsForall
//
// Lowers the gated-MLP dispatch that DispatchCreation forms under
// --iree-dispatch-creation-fuse-mlp (root tagged `iree_dispatch.special_mlp`):
//
//   gate = x * Wg^T            [M, F] f32      (F = ffn width, 8192 for llama)
//   up   = x * Wu^T            [M, F] f32
//   h    = silu(gate) * up     [M, F] f16      (truncs folded into this generic)
//   down = h * Wd^T            [M, N] f32      (root)
//   out  = epilogue(down, ...) [M, N]          (trunc + residual add)
//
// into one scf.forall over M row tiles in which `h` only ever exists one
// F-chunk at a time -- the same "tile the consumer's reduction and fuse the
// producer into it" idea as HexagonMatmulFusionPass:
//
//   scf.forall m in [0, M) step TM                      (workgroups)
//     acc = 0                                    [TM, N] f32
//     scf.for f in [0, F) step TF iter_args(acc)
//       g, u = x[m] * Wg/Wu[f:f+TF]^T            [TM, TF] f32
//       h    = silu(g) * u                       [TM, TF] f16
//       acc += h * Wd[:, f:f+TF]^T
//     out[m] = epilogue(acc, ...)
//
// The forall tiles only M, so each workgroup computes gate/up exactly once for
// its rows (tiling N as well would recompute them once per N tile). The price
// is the [TM, N] f32 accumulator, which lives on the stack across the F loop:
// TM is picked as the largest power-of-two row tile, at most the root's
// distribution M size, whose accumulator and chunk buffers fit the target's
// `max_stack_allocation_size` (32 KiB unless raised with
// --iree-llvmcpu-stack-allocation-limit). TF is the root's cache_reduction size
// when it has one, else 128.
//
// Every op is then tiled on its own to the vector sizes in its lowering_config
// (falling back to the root's) and its lowering_config is dropped. The vector-level tile-and-fuse passes that
// follow would otherwise anchor on `down` and fuse `h` (and gate/up behind it)
// into each of down's N tiles -- recomputing the whole chunk N / 16 times. With
// no config left they find no anchor, and GenericVectorization infers the
// (bounded) vector sizes from the IR.
//
// The function is tagged `iree_codegen.mlp_chain_rewritten` so
// TileAndDistributeToWorkgroupsUsingForall leaves it alone.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include <numeric>

#define DEBUG_TYPE "rewrite-mlp-chain-as-forall"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REWRITEMLPCHAINASFORALLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static constexpr StringLiteral kSpecialMLPAttr = "iree_dispatch.special_mlp";
static constexpr StringLiteral kMLPChainRewrittenAttr =
    "iree_codegen.mlp_chain_rewritten";

// Fallback vector tile of every matmul and elementwise op, used for the loops
// an op's lowering_config does not size (and for ops without one): [kVecM,
// kVecN] outputs over a kVecK reduction step. These are the sizes
// KernelDispatch picks for these f16 matmuls on AVX-512.
static constexpr int64_t kVecM = 8;
static constexpr int64_t kVecN = 16;
static constexpr int64_t kVecK = 16;
// F-chunk the down projection is accumulated over, unless the root's
// lowering_config has a cache_reduction size for it.
static constexpr int64_t kFfnChunk = 128;
// Used when the target does not say (same default as the CPU backend).
static constexpr int64_t kDefaultStackLimitBytes = 32 * 1024;

struct MLPChainMatch {
  linalg::LinalgOp gate;
  linalg::LinalgOp up;
  linalg::GenericOp act;
  linalg::LinalgOp down;
  linalg::GenericOp epilogue;
};

static bool isAllParallelIdentity(linalg::GenericOp op) {
  return op.getNumLoops() == op.getNumParallelLoops() &&
         llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap m) { return m.isIdentity(); });
}

/// A plain 2-D matmul in the `ins(A[M,K], B[N,K]) outs(C[M,N])` form all the
/// llama projections use.
static bool isTransposedBMatmul(linalg::LinalgOp op) {
  if (!op || !linalg::isaContractionOpInterface(op) ||
      op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op.getNumLoops() != 3) {
    return false;
  }
  MLIRContext *ctx = op.getContext();
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  SmallVector<AffineMap> expected = {
      AffineMap::get(3, 0, {m, k}, ctx), AffineMap::get(3, 0, {n, k}, ctx),
      AffineMap::get(3, 0, {m, n}, ctx)};
  return op.getIndexingMapsArray() == expected;
}

/// The single user of `op` that is not a `tensor.dim` (tiling adds those), or
/// null.
static Operation *getSingleNonDimUser(Operation *op) {
  Operation *user = nullptr;
  for (Operation *u : op->getUsers()) {
    if (isa<tensor::DimOp>(u)) {
      continue;
    }
    if (user && user != u) {
      return nullptr;
    }
    user = u;
  }
  return user;
}

/// `requireSharedInput`: gate and up read the very same `x` value. Only checked
/// on the original ops -- tile-and-fuse clones the load and slice of `x`
/// separately for each of them.
static FailureOr<MLPChainMatch> matchMLPChain(Operation *root,
                                              bool requireSharedInput) {
  MLPChainMatch match;
  match.down = dyn_cast<linalg::LinalgOp>(root);
  if (!isTransposedBMatmul(match.down)) {
    return failure();
  }
  match.act = match.down.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!match.act || match.act.getNumDpsInputs() != 2 ||
      match.act.getNumDpsInits() != 1 || !isAllParallelIdentity(match.act) ||
      getSingleNonDimUser(match.act) != root) {
    return failure();
  }
  match.gate = match.act.getDpsInputs()[0].getDefiningOp<linalg::LinalgOp>();
  match.up = match.act.getDpsInputs()[1].getDefiningOp<linalg::LinalgOp>();
  if (!isTransposedBMatmul(match.gate) || !isTransposedBMatmul(match.up) ||
      match.gate == match.up ||
      getSingleNonDimUser(match.gate) != match.act ||
      getSingleNonDimUser(match.up) != match.act ||
      (requireSharedInput &&
       match.gate.getDpsInputs()[0] != match.up.getDpsInputs()[0])) {
    return failure();
  }
  match.epilogue =
      dyn_cast_or_null<linalg::GenericOp>(getSingleNonDimUser(root));
  if (!match.epilogue || match.epilogue.getNumDpsInits() != 1 ||
      !isAllParallelIdentity(match.epilogue) ||
      match.epilogue.getNumLoops() != 2) {
    return failure();
  }
  return match;
}

/// Size of `op`'s lowering_config at `level` for each loop (0 = not tiled), or
/// an empty list if the config does not have that level.
static SmallVector<int64_t> getConfigLevel(Operation *op,
                                           IREE::CPU::TilingLevel level) {
  auto config = getLoweringConfig<IREE::CPU::LoweringConfigAttr>(op);
  if (!config || !config.hasTilingLevel(llvm::to_underlying(level))) {
    return {};
  }
  return config.getStaticTilingLevelSizes(llvm::to_underlying(level), op);
}

/// Vector tile size of each of `op`'s loops from its lowering_config: the
/// vector_common_parallel size, else vector_inner_parallel, else
/// vector_reduction. Loops none of them size get `fallback`.
static SmallVector<int64_t> getVectorTileSizes(Operation *op,
                                               ArrayRef<int64_t> fallback) {
  SmallVector<int64_t> sizes(fallback);
  SmallVector<bool> fromConfig(sizes.size(), false);
  for (IREE::CPU::TilingLevel level :
       {IREE::CPU::TilingLevel::VectorCommonParallelTiles,
        IREE::CPU::TilingLevel::VectorInnerParallelTiles,
        IREE::CPU::TilingLevel::VectorReductionTiles}) {
    for (auto [i, size] : llvm::enumerate(getConfigLevel(op, level))) {
      if (i < sizes.size() && !fromConfig[i] && size > 0 &&
          !ShapedType::isDynamic(size)) {
        sizes[i] = size;
        fromConfig[i] = true;
      }
    }
  }
  return sizes;
}

/// Size `level` of `op`'s lowering_config gives loop `dim`, or 0.
static int64_t getConfigSize(Operation *op, IREE::CPU::TilingLevel level,
                             unsigned dim) {
  SmallVector<int64_t> sizes = getConfigLevel(op, level);
  if (dim >= sizes.size() || ShapedType::isDynamic(sizes[dim])) {
    return 0;
  }
  return sizes[dim];
}

/// Largest power-of-two row tile, at most `maxRows` (the root's distribution
/// M size), whose stack buffers fit in 3/4 of the limit, leaving the rest for
/// spills and the callee frames. Per row: the [., N] f32 accumulator plus the
/// [., ffnChunk] gate/up (f32) and h (f16) chunks -- which is what
/// bufferization allocates (9.25 KiB per row for llama's N = 2048 and a 128
/// chunk). 32 KiB gives 2 rows; 128 KiB gives 8.
static int64_t pickRowTile(FunctionOpInterface funcOp, int64_t n,
                           int64_t maxRows, int64_t ffnChunk) {
  int64_t limit = kDefaultStackLimitBytes;
  if (auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp)) {
    if (DictionaryAttr config = target.getConfiguration()) {
      if (auto attr =
              config.getAs<IntegerAttr>("max_stack_allocation_size")) {
        limit = attr.getInt();
      }
    }
  }
  int64_t bytesPerRow = n * 4 + ffnChunk * (4 + 4 + 2);
  int64_t rows = llvm::PowerOf2Ceil(maxRows);
  while (rows > 1 && rows * bytesPerRow > limit / 4 * 3) {
    rows /= 2;
  }
  return rows;
}

/// Tiles `op` (no fusion) with `tileSizes`; reductions are tiled as a
/// sequential scf.for carrying the full-precision accumulator.
static LogicalResult tileAlone(RewriterBase &rewriter, TilingInterface op,
                               ArrayRef<int64_t> tileSizes) {
  scf::SCFTilingOptions options;
  options.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(),
                                              tileSizes));
  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTilingResult> tiled =
      scf::tileUsingSCF(rewriter, op, options);
  if (failed(tiled)) {
    return failure();
  }
  rewriter.replaceOp(op, tiled->replacements);
  return success();
}

/// If `v` is `tensor.insert_slice %src into %dest` covering all of `%dest`,
/// where `%dest` is a `tensor.empty` (or a slice of one), returns `%src`.
/// Tiling a reduction hands its result back in that form, and with dynamic
/// sizes nothing folds it, so bufferization would give the [TM, N]
/// accumulator a second stack buffer just to copy it into.
static Value peelFullInsertIntoEmpty(Value v) {
  auto insert = v.getDefiningOp<tensor::InsertSliceOp>();
  if (!insert || insert.getSourceType() != insert.getDestType() ||
      !llvm::all_of(insert.getMixedOffsets(), isZeroInteger) ||
      !llvm::all_of(insert.getMixedStrides(), isOneInteger)) {
    return v;
  }
  SmallVector<OpFoldResult> destSizes;
  Operation *dest = insert.getDest().getDefiningOp();
  if (auto slice = dyn_cast_or_null<tensor::ExtractSliceOp>(dest)) {
    if (!slice.getSource().getDefiningOp<tensor::EmptyOp>()) {
      return v;
    }
    destSizes = slice.getMixedSizes();
  } else if (auto empty = dyn_cast_or_null<tensor::EmptyOp>(dest)) {
    destSizes = empty.getMixedSizes();
  } else {
    return v;
  }
  if (!isEqualConstantIntOrValueArray(destSizes, insert.getMixedSizes())) {
    return v;
  }
  return insert.getSource();
}

static Value lookupReplacement(const scf::SCFTileAndFuseResult &result,
                               Value v) {
  auto it = result.replacements.find(v);
  return it == result.replacements.end() ? Value()
                                         : peelFullInsertIntoEmpty(it->second);
}

static LogicalResult rewriteMLPChain(RewriterBase &rewriter,
                                     FunctionOpInterface funcOp,
                                     MLPChainMatch &match) {
  MLIRContext *ctx = rewriter.getContext();
  int64_t n = cast<ShapedType>(match.down->getResult(0).getType()).getDimSize(1);
  if (ShapedType::isDynamic(n)) {
    return failure();
  }
  // Tile sizes come from the root's lowering_config where it has them.
  SmallVector<int64_t> rootVec =
      getVectorTileSizes(match.down, {kVecM, kVecN, kVecK});
  int64_t vecK = rootVec[2];
  SmallVector<int64_t> gateVec =
      getVectorTileSizes(match.gate, {kVecM, kVecN, vecK});
  int64_t maxRows = getConfigSize(
      match.down, IREE::CPU::TilingLevel::DistributionTiles, /*dim=*/0);
  if (maxRows <= 0) {
    maxRows = kVecM;
  }
  int64_t ffnChunk = getConfigSize(
      match.down, IREE::CPU::TilingLevel::CacheReductionTiles, /*dim=*/2);
  if (ffnChunk <= 0) {
    ffnChunk = kFfnChunk;
  }
  // Whole vector tiles of both down's reduction and gate/up's F (= N) dim.
  int64_t chunkMultiple = std::lcm(vecK, gateVec[1]);
  ffnChunk = llvm::divideCeil(ffnChunk, chunkMultiple) * chunkMultiple;
  int64_t rowTile = pickRowTile(funcOp, n, maxRows, ffnChunk);
  LLVM_DEBUG(llvm::dbgs() << "[mlp] row tile " << rowTile << ", ffn chunk "
                          << ffnChunk << ", vector tile " << rootVec[0] << "x"
                          << rootVec[1] << "x" << vecK << "\n");

  // 1. Workgroups: tile the epilogue over M only and fuse the whole chain in.
  {
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(
        getAsIndexOpFoldResult(ctx, ArrayRef<int64_t>{rowTile, 0}));
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    tilingOptions.setMapping({IREE::Codegen::WorkgroupMappingAttr::get(
        ctx, IREE::Codegen::WorkgroupId::IdX)});
    scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(tilingOptions);
    rewriter.setInsertionPoint(match.epilogue);
    FailureOr<scf::SCFTileAndFuseResult> result =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(match.epilogue.getOperation()),
            options);
    if (failed(result)) {
      return failure();
    }
    Value repl = lookupReplacement(*result, match.epilogue->getResult(0));
    if (!repl) {
      return failure();
    }
    rewriter.replaceAllUsesWith(match.epilogue->getResult(0), repl);
  }

  // Re-find the clones inside the forall: the original ops are now dead.
  scf::ForallOp forallOp;
  funcOp.walk([&](scf::ForallOp op) {
    if (op.getMapping() && !forallOp) {
      forallOp = op;
    }
  });
  if (!forallOp) {
    return failure();
  }
  MLPChainMatch tiled;
  forallOp.walk([&](linalg::LinalgOp op) {
    if (op->hasAttr(kSpecialMLPAttr)) {
      tiled.down = op;
    }
  });
  if (!tiled.down) {
    return failure();
  }
  FailureOr<MLPChainMatch> inner = matchMLPChain(tiled.down, /*requireSharedInput=*/false);
  if (failed(inner)) {
    LLVM_DEBUG(llvm::dbgs() << "[mlp] chain not found inside forall\n");
    return failure();
  }
  tiled = *inner;

  // 2. F chunks: tile down's reduction and fuse h, gate and up into the loop,
  // so they are computed [rowTile, kFfnChunk] at a time. down's init stays
  // outside as the loop-carried accumulator.
  {
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(
        getAsIndexOpFoldResult(ctx, ArrayRef<int64_t>{0, 0, ffnChunk}));
    scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(tilingOptions);
    Operation *gate = tiled.gate, *up = tiled.up, *act = tiled.act;
    // down's own init (the zero fill) must stay outside: it is the loop's
    // initial accumulator, and fusing it as a destination would leave the
    // loop starting from the unfilled tensor.empty.
    Operation *accInit = tiled.down.getDpsInits()[0].getDefiningOp();
    options.setFusionControlFn(
        [&](tensor::ExtractSliceOp, OpResult producer, bool isDestination)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          Operation *owner = producer.getOwner();
          if (owner == accInit ||
              (isDestination && !isa<linalg::FillOp>(owner))) {
            return std::nullopt;
          }
          // gate/up's zero inits are fused so they are per-chunk, not [TM, F].
          if (owner == gate || owner == up || owner == act ||
              isa<linalg::FillOp>(owner)) {
            return scf::SCFTileAndFuseOptions::ControlFnResult{
                /*yieldProducerReplacement=*/false};
          }
          return std::nullopt;
        });
    rewriter.setInsertionPoint(tiled.down);
    FailureOr<scf::SCFTileAndFuseResult> result =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(tiled.down.getOperation()),
            options);
    if (failed(result)) {
      return failure();
    }
    Value repl = lookupReplacement(*result, tiled.down->getResult(0));
    if (!repl) {
      return failure();
    }
    rewriter.replaceAllUsesWith(tiled.down->getResult(0), repl);
  }

  // 3. Tile every op left in the forall to vector size on its own, and drop
  // its lowering_config (see the file comment). The sizes are read from that
  // config first; the clones keep the original op's.
  SmallVector<linalg::LinalgOp> ops;
  forallOp.walk([&](linalg::LinalgOp op) {
    if (!op->use_empty()) {
      ops.push_back(op);
    }
  });
  for (linalg::LinalgOp op : ops) {
    SmallVector<int64_t> sizes;
    if (op.getNumLoops() == 3) {
      sizes = getVectorTileSizes(op, {rootVec[0], rootVec[1], vecK});
    } else if (op.getNumLoops() == 2) {
      sizes = getVectorTileSizes(op, {rootVec[0], rootVec[1]});
    }
    op->removeAttr("lowering_config");
    op->removeAttr(kSpecialMLPAttr);
    if (sizes.empty()) {
      continue;
    }
    if (failed(tileAlone(rewriter, cast<TilingInterface>(op.getOperation()),
                         sizes))) {
      return failure();
    }
  }
  return success();
}

struct RewriteMLPChainAsForallPass final
    : impl::RewriteMLPChainAsForallPassBase<RewriteMLPChainAsForallPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp->hasAttr(kMLPChainRewrittenAttr)) {
      return;
    }
    Operation *markedRoot = nullptr;
    funcOp.walk([&](Operation *op) {
      if (op->hasAttr(kSpecialMLPAttr)) {
        markedRoot = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!markedRoot) {
      return;
    }
    FailureOr<MLPChainMatch> match = matchMLPChain(markedRoot, /*requireSharedInput=*/true);
    if (failed(match)) {
      LLVM_DEBUG(llvm::dbgs() << "[mlp] tagged root does not match\n");
      return;
    }
    IRRewriter rewriter(&getContext());
    if (failed(rewriteMLPChain(rewriter, funcOp, *match))) {
      funcOp.emitError("RewriteMLPChainAsForall: rewrite failed");
      return signalPassFailure();
    }
    funcOp->setAttr(kMLPChainRewrittenAttr, UnitAttr::get(&getContext()));
  }
};

} // namespace
} // namespace mlir::iree_compiler
