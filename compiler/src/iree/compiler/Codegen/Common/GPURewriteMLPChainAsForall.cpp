//===----------------------------------------------------------------------===//
// GPURewriteMLPChainAsForall
//
// GPU (LLVMGPU TileAndFuse) variant of RewriteMLPChainAsForall. Same chain,
// same schedule:
//
//   gate = x * Wg^T            [M, F] f32
//   up   = x * Wu^T            [M, F] f32
//   h    = silu(gate) * up     [M, F] f16
//   down = h * Wd^T            [M, N] f32
//   out  = epilogue(down, ...) [M, N]
//
//   scf.forall m in [0, M) step TM                      (workgroups)
//     acc = 0                                    [TM, N] f32
//     scf.for f in [0, F) step TF iter_args(acc)
//       g, u = x[m] * Wg/Wu[f:f+TF]^T            [TM, TF] f32
//       h    = silu(g) * u                       [TM, TF] f16
//       acc += h * Wd[:, f:f+TF]^T
//     out[m] = epilogue(acc, ...)
//
// What differs from the CPU pass:
//
// - The chain is matched by structure. The `iree_dispatch.special_mlp` tag
//   does not survive to GPU codegen (the ops are rebuilt as linalg.generic on
//   the way), and the GPU dispatch has M split into [D, 32], so ops are
//   matched in any rank: M is every output dim but the last.
//
// - Inside the workgroup forall, every op is distributed to threads on its own
//   (an scf.forall with gpu.thread linear mapping) and its reduction is tiled
//   to a serial loop of kVecK. Nothing is fused at thread level: fusing `h`
//   into down's thread tiles would recompute the whole chunk once per tile,
//   which is what the default TileAndFuse config does (64x gate/up recompute,
//   ~200k lines of PTX). GPUGreedilyDistributeToThreads skips ops already in a
//   thread forall, and GPUDistributeForall puts barriers around each one.
//
// - The [TM, N] accumulator and the gate/up/h chunks are the destinations of
//   thread foralls, so GPUInferMemorySpace puts them in shared memory. TM is
//   the largest power of two whose buffers fit in half of the target's
//   `max_workgroup_memory_bytes`.
//
// - Every lowering_config in the function is dropped, so the matmul-specific
//   TileAndFuse steps (operand promotion, packing to MMA intrinsics, the
//   reduction/thread/subgroup tiling levels) find nothing to act on.
//
// The function is tagged `iree_codegen.mlp_chain_rewritten` so
// TileAndDistributeToWorkgroupsUsingForall leaves it alone.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "gpu-rewrite-mlp-chain-as-forall"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUREWRITEMLPCHAINASFORALLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static constexpr StringLiteral kMLPChainRewrittenAttr =
    "iree_codegen.mlp_chain_rewritten";
// Marks `down` so its clone can be found again after workgroup tiling.
static constexpr StringLiteral kMLPDownAttr = "iree_codegen.mlp_down";

// Largest per-thread tile along the innermost parallel dim.
static constexpr int64_t kMaxThreadVec = 16;
// Serial step every reduction is tiled to inside a thread.
static constexpr int64_t kVecK = 16;
// F-chunk the down projection is accumulated over.
static constexpr int64_t kFfnChunk = 128;
// Largest row tile tried.
static constexpr int64_t kMaxRowTile = 8;
// Used when the target does not say (CUDA's static shared memory limit).
static constexpr int64_t kDefaultWorkgroupMemoryBytes = 48 * 1024;
// Threads per workgroup. Every op is distributed over all of them, so this
// replaces whatever size config selection picked for the root op.
static constexpr int64_t kWorkgroupThreads = 256;

struct MLPChainMatch {
  linalg::LinalgOp gate;
  linalg::LinalgOp up;
  linalg::GenericOp act;
  linalg::LinalgOp down;
  linalg::GenericOp epilogue;
};

/// All-parallel elementwise op over `rank` loops with identity maps.
static bool isElementwise(linalg::GenericOp op, int64_t rank) {
  return op && op.getNumLoops() == rank &&
         op.getNumLoops() == op.getNumParallelLoops() &&
         llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap m) { return m.isIdentity(); });
}

/// A projection `ins(A[m..., K], B[N, K]) outs(C[m..., N])`: the llama
/// projections in any number of M dims (1 on CPU, [D, 32] on GPU), as a named
/// matmul or a linalg.generic.
static bool isProjection(linalg::LinalgOp op) {
  if (!op || !linalg::isaContractionOpInterface(op) ||
      op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
    return false;
  }
  int64_t numLoops = op.getNumLoops();
  if (numLoops < 3 || op.getNumReductionLoops() != 1 ||
      !linalg::isReductionIterator(op.getIteratorTypesArray().back())) {
    return false;
  }
  MLIRContext *ctx = op.getContext();
  int64_t outRank = numLoops - 1;
  SmallVector<AffineExpr> mDims, lhs, out;
  for (int64_t i = 0; i < outRank - 1; ++i) {
    mDims.push_back(getAffineDimExpr(i, ctx));
  }
  AffineExpr n = getAffineDimExpr(outRank - 1, ctx);
  AffineExpr k = getAffineDimExpr(outRank, ctx);
  lhs = mDims;
  lhs.push_back(k);
  out = mDims;
  out.push_back(n);
  SmallVector<AffineMap> expected = {
      AffineMap::get(numLoops, 0, lhs, ctx),
      AffineMap::get(numLoops, 0, {n, k}, ctx),
      AffineMap::get(numLoops, 0, out, ctx)};
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

/// Matches the chain ending in `downCandidate`. `requireSharedInput`: gate and
/// up read the very same `x` value; only checked on the original ops, since
/// tile-and-fuse clones the slice of `x` separately for each.
static FailureOr<MLPChainMatch> matchMLPChain(Operation *downCandidate,
                                              bool requireSharedInput) {
  MLPChainMatch match;
  match.down = dyn_cast<linalg::LinalgOp>(downCandidate);
  if (!isProjection(match.down)) {
    return failure();
  }
  int64_t rank = match.down.getNumLoops() - 1;
  match.act = match.down.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!isElementwise(match.act, rank) || match.act.getNumDpsInputs() != 2 ||
      match.act.getNumDpsInits() != 1 ||
      getSingleNonDimUser(match.act) != downCandidate) {
    return failure();
  }
  match.gate = match.act.getDpsInputs()[0].getDefiningOp<linalg::LinalgOp>();
  match.up = match.act.getDpsInputs()[1].getDefiningOp<linalg::LinalgOp>();
  if (!isProjection(match.gate) || !isProjection(match.up) ||
      match.gate == match.up ||
      match.gate.getNumLoops() != match.down.getNumLoops() ||
      match.up.getNumLoops() != match.down.getNumLoops() ||
      getSingleNonDimUser(match.gate) != match.act ||
      getSingleNonDimUser(match.up) != match.act ||
      (requireSharedInput &&
       match.gate.getDpsInputs()[0] != match.up.getDpsInputs()[0])) {
    return failure();
  }
  match.epilogue =
      dyn_cast_or_null<linalg::GenericOp>(getSingleNonDimUser(downCandidate));
  if (!isElementwise(match.epilogue, rank) ||
      match.epilogue.getNumDpsInits() != 1) {
    return failure();
  }
  return match;
}

/// Largest power-of-two row tile (at most kMaxRowTile, dividing `mInner` when
/// it is static) whose shared memory fits in half of the target's workgroup
/// memory, leaving the rest for what bufferization adds. Per row: the [., N]
/// f32 accumulator plus the [., kFfnChunk] gate/up (f32) and h (f16) chunks --
/// 9.25 KiB for llama's N = 2048, so sm_121's 99 KiB gives 4 rows.
static int64_t pickRowTile(FunctionOpInterface funcOp, int64_t n,
                           int64_t mInner) {
  int64_t limit = kDefaultWorkgroupMemoryBytes;
  if (IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp)) {
    limit = target.getWgp().getMaxWorkgroupMemoryBytes();
  }
  int64_t bytesPerRow = n * 4 + kFfnChunk * (4 + 4 + 2);
  int64_t rows = kMaxRowTile;
  while (rows > 1 &&
         (rows * bytesPerRow > limit / 2 ||
          (!ShapedType::isDynamic(mInner) && mInner % rows != 0))) {
    rows /= 2;
  }
  return rows;
}

/// Workgroup mapping for the tiled dims of `tileSizes`, innermost tiled dim
/// on x (the same order TileAndDistributeToWorkgroupsUsingForall uses).
static SmallVector<Attribute> getWorkgroupMapping(MLIRContext *ctx,
                                                  ArrayRef<int64_t> tileSizes) {
  SmallVector<Attribute> mapping;
  for (int64_t size : llvm::reverse(tileSizes)) {
    if (size == 0) {
      continue;
    }
    int64_t idx = mapping.size();
    if (idx < 3) {
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          ctx, *IREE::Codegen::symbolizeWorkgroupId(idx)));
    } else {
      mapping.push_back(IREE::Codegen::WorkgroupMappingAttr::get(
          ctx, IREE::Codegen::WorkgroupId::IdZ, idx - 2));
    }
  }
  return llvm::to_vector(llvm::reverse(mapping));
}

/// If `v` is `tensor.insert_slice %src into %dest` covering all of `%dest`,
/// where `%dest` is a `tensor.empty` (or a slice of one), returns `%src`.
/// Tiling a reduction hands its result back in that form, and nothing folds it
/// with dynamic sizes, so bufferization would copy the accumulator into a
/// second buffer.
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

/// Distributes `op` to the threads of the workgroup on its own (no fusion):
/// an scf.forall with descending gpu.thread linear mapping over its parallel
/// dims, every outer parallel dim tiled to 1 and the innermost one to the
/// largest power of two (at most kMaxThreadVec) that still leaves a tile for
/// every thread. Inside the forall the reduction dims are tiled to a serial
/// loop of kVecK so each thread works on bounded vectors.
static LogicalResult distributeToThreads(RewriterBase &rewriter,
                                         linalg::LinalgOp op,
                                         int64_t numThreads) {
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  if (llvm::any_of(ranges, ShapedType::isDynamic)) {
    LLVM_DEBUG(llvm::dbgs() << "[mlp] dynamic loop range: " << op << "\n");
    return failure();
  }
  int64_t innerParallel = -1;
  int64_t parallelCount = 1;
  for (auto [i, it] : llvm::enumerate(iterators)) {
    if (linalg::isParallelIterator(it)) {
      innerParallel = i;
      parallelCount *= ranges[i];
    }
  }
  if (innerParallel < 0) {
    return failure();
  }
  int64_t vec = kMaxThreadVec;
  while (vec > 1 && (ranges[innerParallel] % vec != 0 ||
                     parallelCount / vec < numThreads)) {
    vec /= 2;
  }

  SmallVector<int64_t> threadTiles(ranges.size(), 0);
  for (auto [i, it] : llvm::enumerate(iterators)) {
    if (!linalg::isParallelIterator(it) || ranges[i] == 1) {
      continue;
    }
    threadTiles[i] = (int64_t)i == innerParallel ? vec : 1;
  }
  if (llvm::all_of(threadTiles, [](int64_t s) { return s == 0; })) {
    // A single-element op: give it one thread so it still sits in a thread
    // forall (and its destination in shared memory).
    threadTiles[innerParallel] = 1;
  }
  SmallVector<Attribute> mapping;
  for (int64_t size : threadTiles) {
    if (size != 0) {
      mapping.push_back(gpu::GPUThreadMappingAttr::get(
          ctx, static_cast<gpu::MappingId>(
                   static_cast<unsigned>(gpu::MappingId::LinearDim0) +
                   mapping.size())));
    }
  }
  std::reverse(mapping.begin(), mapping.end());

  scf::SCFTilingOptions threadOptions;
  threadOptions.setTileSizes(getAsIndexOpFoldResult(ctx, threadTiles));
  threadOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  threadOptions.setMapping(mapping);
  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTilingResult> threadTiled = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), threadOptions);
  if (failed(threadTiled) || threadTiled->tiledOps.size() != 1) {
    return failure();
  }
  rewriter.replaceOp(op, threadTiled->replacements);

  SmallVector<int64_t> reductionTiles(ranges.size(), 0);
  bool hasReduction = false;
  for (auto [i, it] : llvm::enumerate(iterators)) {
    if (linalg::isReductionIterator(it) && ranges[i] > kVecK) {
      reductionTiles[i] = kVecK;
      hasReduction = true;
    }
  }
  if (!hasReduction) {
    return success();
  }
  auto perThread = cast<TilingInterface>(threadTiled->tiledOps.front());
  scf::SCFTilingOptions reductionOptions;
  reductionOptions.setTileSizes(getAsIndexOpFoldResult(ctx, reductionTiles));
  rewriter.setInsertionPoint(perThread);
  FailureOr<scf::SCFTilingResult> reductionTiled =
      scf::tileUsingSCF(rewriter, perThread, reductionOptions);
  if (failed(reductionTiled)) {
    return failure();
  }
  rewriter.replaceOp(perThread, reductionTiled->replacements);
  return success();
}

static LogicalResult rewriteMLPChain(RewriterBase &rewriter,
                                     FunctionOpInterface funcOp,
                                     MLPChainMatch &match) {
  MLIRContext *ctx = rewriter.getContext();
  auto outType = cast<ShapedType>(match.down->getResult(0).getType());
  int64_t rank = outType.getRank();
  int64_t n = outType.getDimSize(rank - 1);
  int64_t f = cast<ShapedType>(match.act->getResult(0).getType())
                  .getDimSize(rank - 1);
  if (ShapedType::isDynamic(n) || ShapedType::isDynamic(f) ||
      f % kFfnChunk != 0) {
    return failure();
  }
  IREE::Codegen::TranslationInfoAttr info = getTranslationInfo(funcOp);
  if (!info) {
    LLVM_DEBUG(llvm::dbgs() << "[mlp] no translation info\n");
    return failure();
  }
  int64_t numThreads = kWorkgroupThreads;
  if (IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp)) {
    numThreads = std::min<int64_t>(
        numThreads, target.getWgp().getMaxThreadCountPerWorkgroup());
  }
  (void)setTranslationInfo(
      funcOp, IREE::Codegen::TranslationInfoAttr::get(
                  ctx, info.getPassPipeline(), info.getCodegenSpec(),
                  ArrayRef<int64_t>{numThreads, 1, 1}, info.getSubgroupSize(),
                  info.getConfiguration()));
  int64_t rowTile = pickRowTile(funcOp, n, outType.getDimSize(rank - 2));
  LLVM_DEBUG(llvm::dbgs() << "[mlp] row tile " << rowTile << ", threads "
                          << numThreads << "\n");

  match.down->setAttr(kMLPDownAttr, UnitAttr::get(ctx));

  // 1. Workgroups: tile the epilogue over the M dims only (the outer ones to
  // 1, the innermost to the row tile) and fuse the whole chain in.
  {
    SmallVector<int64_t> tileSizes(rank, 1);
    tileSizes[rank - 2] = rowTile;
    tileSizes[rank - 1] = 0;
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(getAsIndexOpFoldResult(ctx, tileSizes));
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    tilingOptions.setMapping(getWorkgroupMapping(ctx, tileSizes));
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
  Operation *tiledDown = nullptr;
  forallOp.walk([&](linalg::LinalgOp op) {
    if (op->hasAttr(kMLPDownAttr)) {
      tiledDown = op;
    }
  });
  if (!tiledDown) {
    return failure();
  }
  FailureOr<MLPChainMatch> inner =
      matchMLPChain(tiledDown, /*requireSharedInput=*/false);
  if (failed(inner)) {
    LLVM_DEBUG(llvm::dbgs() << "[mlp] chain not found inside forall\n");
    return failure();
  }
  MLPChainMatch tiled = *inner;

  // 2. F chunks: tile down's reduction and fuse h, gate and up into the loop,
  // so they are computed [rowTile, kFfnChunk] at a time. down's init stays
  // outside as the loop-carried accumulator.
  {
    SmallVector<int64_t> tileSizes(rank + 1, 0);
    tileSizes[rank] = kFfnChunk;
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(getAsIndexOpFoldResult(ctx, tileSizes));
    scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(tilingOptions);
    Operation *gate = tiled.gate, *up = tiled.up, *act = tiled.act;
    // down's own init (the zero fill) must stay outside: it is the loop's
    // initial accumulator.
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

  // 3. Distribute every op left in the workgroup forall to threads on its own.
  SmallVector<linalg::LinalgOp> ops;
  forallOp.walk([&](linalg::LinalgOp op) {
    if (!op->use_empty()) {
      ops.push_back(op);
    }
  });
  for (linalg::LinalgOp op : ops) {
    op->removeAttr(kMLPDownAttr);
    if (failed(distributeToThreads(rewriter, op, numThreads))) {
      LLVM_DEBUG(llvm::dbgs() << "[mlp] thread distribution failed: " << op
                              << "\n");
      return failure();
    }
  }
  return success();
}

struct GPURewriteMLPChainAsForallPass final
    : impl::GPURewriteMLPChainAsForallPassBase<GPURewriteMLPChainAsForallPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp->hasAttr(kMLPChainRewrittenAttr)) {
      return;
    }
    std::optional<MLPChainMatch> match;
    funcOp.walk([&](linalg::LinalgOp op) {
      FailureOr<MLPChainMatch> m =
          matchMLPChain(op, /*requireSharedInput=*/true);
      if (succeeded(m)) {
        match = *m;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!match) {
      return;
    }
    // The chain is codegen'd here; the configs picked for its ops would steer
    // the matmul-specific TileAndFuse steps (see the file comment).
    funcOp.walk([](Operation *op) { op->removeAttr("lowering_config"); });
    IRRewriter rewriter(&getContext());
    if (failed(rewriteMLPChain(rewriter, funcOp, *match))) {
      funcOp.emitError("GPURewriteMLPChainAsForall: rewrite failed");
      return signalPassFailure();
    }
    funcOp->setAttr(kMLPChainRewrittenAttr, UnitAttr::get(&getContext()));
  }
};

} // namespace

bool isGatedMLPChainDown(Operation *op) {
  return succeeded(matchMLPChain(op, /*requireSharedInput=*/true));
}

} // namespace mlir::iree_compiler
