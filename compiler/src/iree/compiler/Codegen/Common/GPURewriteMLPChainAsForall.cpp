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
//   fixed at 8 (1 for decode); down's thread tile is [TM, 16] with K in
//   serial steps of --iree-codegen-gpu-mlp-down-k-tile (16).
//
// - Every lowering_config in the function is dropped, so the matmul-specific
//   TileAndFuse steps (operand promotion, packing to MMA intrinsics, the
//   reduction/thread/subgroup tiling levels) find nothing to act on.
//
// MMA path (--iree-codegen-gpu-mlp-mma, the default; needs the inner M dim to
// be a multiple of 16, i.e. prefill -- decode keeps the scalar path): TM is
// one intrinsic (16 rows) and N is also split across workgroups
// (kMMANTile = 1024), so the [16, nTile] accumulator fits in shared memory;
// gate/up are recomputed once per N tile. Inside each F chunk, down walks the
// accumulator in serial N steps of kMMADownNTile (256), and each warp reduces
// its acc tile in serial k2 steps of kMMADownKTile (64) on its own, after
// subgroup tiling (GPUTileMLPDownPerWarp). Instead of the thread distribution
// above, gate, up and down get MMA lowering configs and h a thread config, and
// the TileAndFuse levels lower them: each op has its own config, so none is
// fused into another (h reaches down through an iree_gpu.value_barrier).
// After GPUFuseAndHoistParallelLoops, gate's and up's k loops are fused into
// one (GPUFuseMLPGateUpLoops), so each x slice is loaded once for both.
//
// The function is tagged `iree_codegen.mlp_chain_rewritten` so
// TileAndDistributeToWorkgroupsUsingForall leaves it alone.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "gpu-rewrite-mlp-chain-as-forall"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUREWRITEMLPCHAINASFORALLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static llvm::cl::opt<bool> clMLPUseMMA(
    "iree-codegen-gpu-mlp-mma",
    llvm::cl::desc("Lower the projections of the rewritten MLP chain with MMA "
                   "intrinsics (TM = 16, N split across workgroups) instead of "
                   "per-thread FMAs."),
    llvm::cl::init(true), llvm::cl::Hidden);

static llvm::cl::opt<int64_t> clMLPDownKTile(
    "iree-codegen-gpu-mlp-down-k-tile",
    llvm::cl::desc("Scalar path: serial K step of down inside each thread "
                   "(the `scf.for k2` loop); 0 reduces the whole F chunk in "
                   "one op."),
    llvm::cl::init(16), llvm::cl::Hidden);

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
// Row tile TM (fixed; smaller only when the inner M dim is not a multiple).
static constexpr int64_t kMaxRowTile = 8;
// N strip of `down` each thread owns (all TM rows of it): the inner
// `scf.for N (16)` of the schedule, distributed over the threads.
static constexpr int64_t kDownThreadN = 16;
// Used when the target does not say (CUDA's static shared memory limit).
static constexpr int64_t kDefaultWorkgroupMemoryBytes = 48 * 1024;
// Threads per workgroup. Every op is distributed over all of them, so this
// replaces whatever size config selection picked for the root op.
static constexpr int64_t kWorkgroupThreads = 256;
// MMA path: rows per workgroup (one intrinsic M tile).
static constexpr int64_t kMMARowTile = 16;
// MMA path: workgroup tile along N; gate/up are recomputed once per N tile
// (2x for llama's N = 2048). [16, 1024] f32 accumulator = 64 KiB.
static constexpr int64_t kMMANTile = 1024;
// MMA path: down walks the [16, nTile] accumulator in a serial loop of N
// steps this wide (8 warps x 32 columns = 4 MMA tiles per warp per k2 slice).
// Fewer live accumulator tiles per warp than covering all of nTile at once:
// GX10 MLP kernel time -7.5% at prefill S = 2048 (TN = 1024 / 512 / 256:
// 938.5 / 875.8 / 868.2 ms over 16 layers).
static constexpr int64_t kMMADownNTile = 256;
// MMA path: down's serial k2 step over the F chunk (128 -> 2 steps of 64,
// 4 K slices unrolled per step), inside each warp, inside the N loop. GX10
// MLP kernel time, prefill S = 512 / 1024 / 2048 (16 layers): all 8 K slices
// unrolled 306.5 / 457.9 / 866.7 ms; k2 outside N 223.5 / 353.0 / 680.7 ms;
// k2 inside N, workgroup loop 211.8 / 324.0 / 626.9 ms; k2 inside N and
// inside the warp 192.2 / 295.5 / 573.5 ms.
static constexpr int64_t kMMADownKTile = 64;
// MMA path: K tile, in intrinsics, gate/up are reduced in (4 x 16 = 64).
static constexpr int64_t kMMAReductionK = 4;
// MMA path: promote gate/up's operands (x and the weight tile) to shared
// memory. Off: with the 64 KiB accumulator, promotion at K = 32 needs
// 104,448 B (over sm_121's 101,376 B); at K = 16 it fits but is slower than
// reading the operands from global memory at K = 64 (GX10, prefill S = 512 /
// 1024 / 2048: 1.83 / 2.39 / 4.23 s vs 1.65 / 2.09 / 3.65 s).
static constexpr bool kMMAPromote = false;

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

/// Row tile TM: kMaxRowTile, halved only to divide a static inner M dim
/// (decode: M = 1). [TM, N] f32 + chunks = 74 KiB for llama at TM = 8, under
/// sm_121's 99 KiB of workgroup memory.
static int64_t pickRowTile(int64_t mInner) {
  int64_t rows = kMaxRowTile;
  while (rows > 1 && !ShapedType::isDynamic(mInner) && mInner % rows != 0) {
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
                                         int64_t numThreads,
                                         ArrayRef<int64_t> forcedTiles = {},
                                         int64_t reductionStep = kVecK) {
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
  if (!forcedTiles.empty()) {
    threadTiles.assign(forcedTiles.begin(), forcedTiles.end());
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
    if (linalg::isReductionIterator(it) && reductionStep > 0 &&
        ranges[i] > reductionStep) {
      reductionTiles[i] = reductionStep;
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

/// Lowering config for a projection inside the F loop: packed to `kind`, each
/// subgroup taking one intrinsic along M and `subgroupN` along N, and the K dim
/// tiled to `reductionK` intrinsics (0: left whole).
static IREE::GPU::LoweringConfigAttr
getProjectionMMAConfig(MLIRContext *ctx, IREE::GPU::MMAAttr kind,
                       int64_t numLoops, int64_t subgroupN, int64_t reductionK,
                       bool promote) {
  Builder b(ctx);
  SmallVector<NamedAttribute> attrs;
  IREE::GPU::setMmaKind(ctx, attrs, kind);
  if (promote) {
    IREE::GPU::appendPromotedOperandsList(ctx, attrs, {0, 1});
  }
  SmallVector<int64_t> subgroup(numLoops, 0);
  subgroup[numLoops - 3] = 1;
  subgroup[numLoops - 2] = subgroupN;
  attrs.emplace_back("subgroup", b.getI64ArrayAttr(subgroup));
  if (reductionK > 0) {
    SmallVector<int64_t> reduction(numLoops, 0);
    reduction[numLoops - 1] = reductionK;
    attrs.emplace_back("reduction", b.getI64ArrayAttr(reduction));
  }
  return IREE::GPU::LoweringConfigAttr::get(ctx,
                                            DictionaryAttr::get(ctx, attrs));
}

// Key in down's lowering config: the K tile, in intrinsics, of the serial k2
// loop GPUTileMLPDownPerWarp makes inside each warp's tile.
static constexpr StringLiteral kMLPDownWarpKKey = "mlp_down_warp_k";

/// MMA path for step 3: instead of distributing every op to threads here,
/// give gate, up and down MMA configs and h a thread config, and let the
/// TileAndFuse levels lower them. Each op has its own config at the level it
/// is tiled at, so none is fused into another (GPUApplyTilingLevel only fuses
/// producers without one): h is computed once per chunk, not once per down
/// subgroup tile. The fills and the epilogue are left to
/// GPUGreedilyDistributeToThreads.
static LogicalResult setMMAConfigs(scf::ForallOp forallOp,
                                   IREE::GPU::MMAAttr kind, int64_t nTile,
                                   int64_t numSubgroups) {
  MLIRContext *ctx = forallOp.getContext();
  linalg::LinalgOp down;
  forallOp.walk([&](linalg::LinalgOp op) {
    if (op->hasAttr(kMLPDownAttr) && op->getParentOfType<scf::ForOp>()) {
      down = op;
    }
  });
  if (!down) {
    return failure();
  }
  auto act = down.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!act || act.getNumDpsInputs() != 2) {
    return failure();
  }
  auto gate = act.getDpsInputs()[0].getDefiningOp<linalg::LinalgOp>();
  auto up = act.getDpsInputs()[1].getDefiningOp<linalg::LinalgOp>();
  if (!gate || !up) {
    return failure();
  }
  auto [intrinsicM, intrinsicN, intrinsicK] = kind.getMNKShape();
  int64_t chunkTilesN = kFfnChunk / intrinsicN;
  int64_t downTilesN = nTile / intrinsicN;
  if (intrinsicM != kMMARowTile || chunkTilesN % numSubgroups != 0 ||
      downTilesN % numSubgroups != 0) {
    return failure();
  }
  int64_t numLoops = down.getNumLoops();
  for (linalg::LinalgOp proj : {gate, up}) {
    setLoweringConfig(proj, getProjectionMMAConfig(
                                ctx, kind, numLoops, chunkTilesN / numSubgroups,
                                kMMAReductionK, kMMAPromote));
  }
  int64_t tn = std::min(kMMADownNTile, nTile);
  int64_t tk = std::min(kMMADownKTile, kFfnChunk);
  if (nTile % tn != 0 || (tn / intrinsicN) % numSubgroups != 0 ||
      kFfnChunk % tk != 0 || tk % intrinsicK != 0) {
    return failure();
  }
  IREE::GPU::LoweringConfigAttr downCfg = getProjectionMMAConfig(
      ctx, kind, numLoops, (tn / intrinsicN) / numSubgroups,
      /*reductionK=*/0, /*promote=*/false);
  if (tn < nTile || tk < kFfnChunk) {
    // Serial N loop (steps of tn) at the reduction level, over the whole
    // workgroup (in intrinsics): it is tiled after packing, so each step works
    // on a slice of the packed accumulator in place. Tiling N before the
    // TileAndFuse levels instead copies every [16, tn] slice to its own shared
    // buffer. The k2 loop (steps of tk) is made inside each warp's tile after
    // subgroup tiling (GPUTileMLPDownPerWarp), so a warp keeps its acc tile in
    // registers across k2 with no workgroup barrier between the steps.
    SmallVector<NamedAttribute> attrs(downCfg.getAttributes().getValue());
    SmallVector<int64_t> red(numLoops, 0);
    red[numLoops - 2] = tn / intrinsicN;
    if (tk < kFfnChunk) {
      attrs.emplace_back(StringAttr::get(ctx, kMLPDownWarpKKey),
                         Builder(ctx).getI64IntegerAttr(tk / intrinsicK));
    }
    attrs.emplace_back(StringAttr::get(ctx, "reduction"),
                       Builder(ctx).getI64ArrayAttr(red));
    downCfg = IREE::GPU::LoweringConfigAttr::get(
        ctx, DictionaryAttr::get(ctx, attrs));
  }
  setLoweringConfig(down, downCfg);
  down->removeAttr(kMLPDownAttr);
  {
    // h -> value_barrier -> down: the reduction level would otherwise fuse h
    // into down's N loop and recompute it once per N step; it does not
    // fuse through the barrier, so h is computed once per chunk. (An all-zero
    // reduction tile on h does not work: tiling clones h, and the clone is not
    // a root of the level any more.)
    OpBuilder vb(down);
    OpOperand &hOperand = down->getOpOperand(0);
    auto barrier = IREE::GPU::ValueBarrierOp::create(
        vb, down.getLoc(), ValueRange{hOperand.get()});
    hOperand.set(barrier.getResult(0));
  }

  // h: [., kMMARowTile, kFfnChunk] over the workgroup's threads, 8 wide.
  Builder b(ctx);
  SmallVector<int64_t> thread(act.getNumLoops(), 0);
  thread[act.getNumLoops() - 2] = 1;
  thread[act.getNumLoops() - 1] = 8;
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back("thread", b.getI64ArrayAttr(thread));
  setLoweringConfig(act, IREE::GPU::LoweringConfigAttr::get(
                             ctx, DictionaryAttr::get(ctx, attrs)));
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
  int64_t mInner = outType.getDimSize(rank - 2);
  int64_t rowTile = pickRowTile(mInner);
  // MMA path: one intrinsic of rows, and N split so the [TM, nTile] f32
  // accumulator plus the chunk buffers fit in workgroup memory.
  auto mmaKind = IREE::GPU::MMAAttr::get(
      ctx, IREE::GPU::MMAIntrinsic::NV_MMA_SYNC_F32_16x8x16_F16);
  int64_t nTile = kMMANTile;
  int64_t subgroupSize = info.getSubgroupSize() ? info.getSubgroupSize() : 32;
  bool useMMA = clMLPUseMMA && !ShapedType::isDynamic(mInner) &&
                mInner % kMMARowTile == 0 && nTile > 0 && n % nTile == 0 &&
                numThreads % subgroupSize == 0;
  if (useMMA) {
    int64_t limit = kDefaultWorkgroupMemoryBytes;
    if (IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp)) {
      limit = target.getWgp().getMaxWorkgroupMemoryBytes();
    }
    // acc (f32) + h (f16); gate/up stay in registers (GPUFuseMLPGateUpLoops).
    useMMA = kMMARowTile * (nTile * 4 + kFfnChunk * 2) <= limit;
  }
  if (useMMA) {
    rowTile = kMMARowTile;
  }
  LLVM_DEBUG(llvm::dbgs() << "[mlp] row tile " << rowTile << ", threads "
                          << numThreads << (useMMA ? ", mma" : "") << "\n");

  match.down->setAttr(kMLPDownAttr, UnitAttr::get(ctx));

  // 1. Workgroups: tile the epilogue over the M dims (the outer ones to 1, the
  // innermost to the row tile), and on the MMA path over N, and fuse the whole
  // chain in.
  {
    SmallVector<int64_t> tileSizes(rank, 1);
    tileSizes[rank - 2] = rowTile;
    tileSizes[rank - 1] = useMMA ? nTile : 0;
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

  // 3. MMA path: configs for the TileAndFuse levels.
  if (useMMA) {
    if (failed(setMMAConfigs(forallOp, mmaKind, nTile,
                             numThreads / subgroupSize))) {
      LLVM_DEBUG(llvm::dbgs() << "[mlp] setting MMA configs failed\n");
      return failure();
    }
    return success();
  }

  // 3. Otherwise distribute every op left in the workgroup forall to threads
  // on its own.
  SmallVector<linalg::LinalgOp> ops;
  forallOp.walk([&](linalg::LinalgOp op) {
    if (!op->use_empty()) {
      ops.push_back(op);
    }
  });
  for (linalg::LinalgOp op : ops) {
    // down: each thread owns acc[0:TM, n:n+kDownThreadN] (every M dim whole,
    // N tiled to kDownThreadN), K reduced kVecK at a time.
    SmallVector<int64_t> forced;
    int64_t reductionStep = kVecK;
    if (op->hasAttr(kMLPDownAttr) && op->getParentOfType<scf::ForOp>()) {
      forced.assign(op.getNumLoops(), 0);
      forced[op.getNumLoops() - 2] = kDownThreadN;
      reductionStep = clMLPDownKTile;
    }
    op->removeAttr(kMLPDownAttr);
    if (failed(distributeToThreads(rewriter, op, numThreads, forced,
                                   reductionStep))) {
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

/// After subgroup tiling: tiles each warp's down (an inner_tiled op whose
/// config carries kMLPDownWarpKKey) into a serial loop over its outer K dim,
/// inside the warp: the acc tile stays in registers across the k2 steps and
/// there is no workgroup barrier between them.
struct GPUTileMLPDownPerWarpPass final
    : PassWrapper<GPUTileMLPDownPerWarpPass,
                  InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUTileMLPDownPerWarpPass)
  StringRef getArgument() const override {
    return "iree-codegen-gpu-tile-mlp-down-per-warp";
  }
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (!funcOp->hasAttr(kMLPChainRewrittenAttr)) {
      return;
    }
    SmallVector<IREE::Codegen::InnerTiledOp> targets;
    funcOp.walk([&](IREE::Codegen::InnerTiledOp op) {
      auto cfg = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      if (cfg && cfg.getAttributes().get(kMLPDownWarpKKey)) {
        targets.push_back(op);
      }
    });
    IRRewriter rewriter(&getContext());
    for (IREE::Codegen::InnerTiledOp op : targets) {
      auto cfg = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      int64_t kTile =
          cast<IntegerAttr>(cfg.getAttributes().get(kMLPDownWarpKKey))
              .getInt();
      int64_t numLoops = op.getIndexingMapsArray()[0].getNumDims();
      SmallVector<int64_t> tiles(numLoops, 0);
      tiles[numLoops - 1] = kTile;
      scf::SCFTilingOptions opts;
      opts.setTileSizes(getAsIndexOpFoldResult(&getContext(), tiles));
      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTilingResult> t = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(op.getOperation()), opts);
      if (failed(t)) {
        op.emitError("GPUTileMLPDownPerWarp: tiling failed");
        return signalPassFailure();
      }
      rewriter.replaceOp(op, t->replacements);
    }
  }
};

/// Moves the ops `op` depends on that sit in its block between `anchor` and
/// `op` to just before `anchor` (order kept), so `op` can be moved next to
/// `anchor`.
static void moveDepsBefore(Operation *op, Operation *anchor) {
  Block *block = anchor->getBlock();
  llvm::SetVector<Operation *> deps;
  SmallVector<Operation *> worklist{op};
  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    cur->walk([&](Operation *nested) {
      for (Value v : nested->getOperands()) {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != block || !anchor->isBeforeInBlock(def) ||
            def == op || deps.contains(def)) {
          continue;
        }
        deps.insert(def);
        worklist.push_back(def);
      }
    });
  }
  SmallVector<Operation *> sorted(deps.begin(), deps.end());
  llvm::sort(sorted, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });
  for (Operation *d : sorted) {
    d->moveBefore(anchor);
  }
}

// Unroll factor of the fused gate/up k loop: lets ptxas issue the next step's
// loads ahead of this step's MMAs. GX10 MLP kernel time, prefill S = 512 /
// 2048: 176.9 / 559.6 -> 169.0 / 523.3 ms (x4: 169.9 / 530.3 ms).
static constexpr int64_t kGateUpUnroll = 2;

/// After GPUFuseAndHoistParallelLoops: gate and up are two sibling thread
/// foralls, each with its own serial k loop. Fuses the foralls, then the two k
/// loops, so one loop issues both projections' MMAs and (after CSE) x is
/// loaded once per step. GX10 MLP kernel time, prefill S = 512 / 1024 / 2048
/// (16 layers): 188.0 / 296.2 / 570.8 -> 174.3 / 284.6 / 561.8 ms.
struct GPUFuseMLPGateUpLoopsPass final
    : PassWrapper<GPUFuseMLPGateUpLoopsPass,
                  InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUFuseMLPGateUpLoopsPass)
  StringRef getArgument() const override {
    return "iree-codegen-gpu-fuse-mlp-gate-up-loops";
  }
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (!funcOp->hasAttr(kMLPChainRewrittenAttr)) {
      return;
    }
    auto isProjForall = [](scf::ForallOp forall) {
      if (!forall.getMapping()) {
        return false;
      }
      bool found = false;
      forall.getBody()->walk([&](scf::ForOp loop) {
        loop.getBody()->walk(
            [&](IREE::Codegen::InnerTiledOp) { found = true; });
      });
      return found;
    };
    IRRewriter rewriter(&getContext());
    SmallVector<scf::ForOp> chunkLoops;
    funcOp.walk([&](scf::ForOp loop) { chunkLoops.push_back(loop); });
    for (scf::ForOp chunk : chunkLoops) {
      SmallVector<scf::ForallOp> projs;
      for (Operation &op : *chunk.getBody()) {
        if (auto forall = dyn_cast<scf::ForallOp>(&op)) {
          if (isProjForall(forall)) {
            projs.push_back(forall);
          }
        }
      }
      if (projs.size() < 2) {
        continue;
      }
      scf::ForallOp gate = projs[0], up = projs[1];
      if (gate.getMixedUpperBound() != up.getMixedUpperBound() ||
          gate.getMapping() != up.getMapping()) {
        continue;
      }
      moveDepsBefore(up, gate);
      up->moveAfter(gate);
      scf::ForallOp fused =
          fuseIndependentSiblingForallLoops(gate, up, rewriter);
      // Inside the fused forall: the two k loops.
      SmallVector<scf::ForOp> kLoops;
      for (Operation &op : *fused.getBody()) {
        if (auto loop = dyn_cast<scf::ForOp>(&op)) {
          kLoops.push_back(loop);
        }
      }
      if (kLoops.size() != 2) {
        fused.emitError("GPUFuseMLPGateUpLoops: expected two k loops");
        return signalPassFailure();
      }
      moveDepsBefore(kLoops[1], kLoops[0]);
      kLoops[1]->moveAfter(kLoops[0]);
      scf::ForOp fusedK =
          fuseIndependentSiblingForLoops(kLoops[0], kLoops[1], rewriter);
      if (kGateUpUnroll > 1) {
        (void)loopUnrollByFactor(fusedK, kGateUpUnroll);
      }
      if (failed(computeHInRegisters(rewriter, fused))) {
        fused.emitError("GPUFuseMLPGateUpLoops: h in registers failed");
        return signalPassFailure();
      }
      return;
    }
  }

  /// `fused` yields the gate and up tiles (each thread's MMA accumulators)
  /// into two shared buffers, and a separate thread forall reads them back to
  /// compute h. Instead apply h's body to each thread's own gate/up tiles and
  /// yield only h (f16), so the f32 gate/up buffers, the h forall and the
  /// barrier between them go away.
  static LogicalResult computeHInRegisters(RewriterBase &rewriter,
                                           scf::ForallOp fused) {
    if (fused.getNumResults() != 2) {
      return failure();
    }
    // The h forall: reads collapse_shape(fused result) through extract_slice.
    linalg::GenericOp hGeneric;
    scf::ForallOp hForall;
    for (Operation *user : fused.getResult(0).getUsers()) {
      auto collapse = dyn_cast<tensor::CollapseShapeOp>(user);
      if (!collapse) {
        continue;
      }
      for (Operation *u : collapse->getUsers()) {
        if (auto forall = u->getParentOfType<scf::ForallOp>()) {
          forall.getBody()->walk([&](linalg::GenericOp g) { hGeneric = g; });
          hForall = forall;
        }
      }
    }
    if (!hGeneric || !hForall || hForall.getNumResults() != 1 ||
        hGeneric.getNumDpsInputs() != 2) {
      return failure();
    }
    auto collapseOfH = [&](Value in) -> std::optional<unsigned> {
      auto slice = in.getDefiningOp<tensor::ExtractSliceOp>();
      if (!slice) {
        return std::nullopt;
      }
      auto collapse =
          slice.getSource().getDefiningOp<tensor::CollapseShapeOp>();
      if (!collapse) {
        return std::nullopt;
      }
      auto res = dyn_cast<OpResult>(collapse.getSrc());
      if (!res || res.getOwner() != fused.getOperation()) {
        return std::nullopt;
      }
      return res.getResultNumber();
    };
    SmallVector<unsigned> inputResult;
    for (Value in : hGeneric.getDpsInputs()) {
      std::optional<unsigned> r = collapseOfH(in);
      if (!r) {
        return failure();
      }
      inputResult.push_back(*r);
    }
    // Each thread's tiles: the sources of the parallel_insert_slices (through
    // the linalg.copy into the shared buffer slice, if any).
    scf::InParallelOp term = fused.getTerminator();
    SmallVector<tensor::ParallelInsertSliceOp> inserts;
    for (Operation &op : term.getYieldingOps()) {
      inserts.push_back(cast<tensor::ParallelInsertSliceOp>(&op));
    }
    if (inserts.size() != 2) {
      return failure();
    }
    SmallVector<Value> tiles;
    for (auto ins : inserts) {
      Value src = ins.getSource();
      if (auto copy = src.getDefiningOp<linalg::CopyOp>()) {
        src = copy.getInputs()[0];
      }
      tiles.push_back(src);
    }
    auto tileType = cast<RankedTensorType>(tiles[0].getType());
    Type hElem = cast<ShapedType>(hGeneric.getDpsInits()[0].getType())
                     .getElementType();
    auto hTileType = RankedTensorType::get(tileType.getShape(), hElem);

    // New forall: same bounds, one f16 output in the gate buffer's layout.
    auto gateBufType = cast<RankedTensorType>(fused.getResult(0).getType());
    auto hBufType = RankedTensorType::get(gateBufType.getShape(), hElem);
    rewriter.setInsertionPoint(fused);
    Value hInit = tensor::EmptyOp::create(rewriter, fused.getLoc(),
                                          hBufType.getShape(), hElem);
    auto newForall = scf::ForallOp::create(
        rewriter, fused.getLoc(), fused.getMixedLowerBound(),
        fused.getMixedUpperBound(), fused.getMixedStep(), ValueRange{hInit},
        fused.getMapping(), /*bodyBuilderFn=*/nullptr);
    Block *newBody = newForall.getBody();
    // Drop the default terminator; move the old body over.
    rewriter.eraseOp(newBody->getTerminator());
    Block *oldBody = fused.getBody();
    SmallVector<Value> argRepl(newBody->getArguments().begin(),
                               newBody->getArguments().begin() +
                                   fused.getRank());
    // Old output block args: replaced below once their uses are gone.
    BlockArgument oldGateArg = oldBody->getArgument(fused.getRank());
    BlockArgument oldUpArg = oldBody->getArgument(fused.getRank() + 1);
    // Build h from the tiles, before the old terminator.
    rewriter.setInsertionPoint(term);
    Value hEmpty = tensor::EmptyOp::create(rewriter, fused.getLoc(),
                                           hTileType.getShape(), hElem);
    SmallVector<Value> hIns;
    for (unsigned r : inputResult) {
      hIns.push_back(tiles[r]);
    }
    int64_t rank = tileType.getRank();
    SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank,
                                           utils::IteratorType::parallel);
    auto hTile = linalg::GenericOp::create(rewriter, fused.getLoc(),
                                           TypeRange{hTileType}, hIns,
                                           ValueRange{hEmpty}, maps, iters);
    rewriter.cloneRegionBefore(hGeneric.getRegion(), hTile.getRegion(),
                               hTile.getRegion().end());
    // New terminator: insert h at the gate tile's offsets.
    tensor::ParallelInsertSliceOp gateIns = inserts[0];
    SmallVector<OpFoldResult> offs = gateIns.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = gateIns.getMixedSizes();
    SmallVector<OpFoldResult> strides = gateIns.getMixedStrides();
    // Erase the old terminator (and with it the inserts).
    rewriter.eraseOp(term);
    // The copies into the old shared buffers are dead now.
    SmallVector<Operation *> deadCopies;
    oldBody->walk([&](linalg::CopyOp c) {
      if (c->use_empty()) {
        deadCopies.push_back(c);
      }
    });
    for (Operation *c : deadCopies) {
      rewriter.eraseOp(c);
    }
    // Slices of the old output args used by the copies.
    SmallVector<Operation *> deadSlices;
    for (BlockArgument a : {oldGateArg, oldUpArg}) {
      for (Operation *u : a.getUsers()) {
        if (u->use_empty()) {
          deadSlices.push_back(u);
        }
      }
    }
    for (Operation *s : deadSlices) {
      rewriter.eraseOp(s);
    }
    if (!oldGateArg.use_empty() || !oldUpArg.use_empty()) {
      return failure();
    }
    // Move the body ops (no terminator now) into the new forall.
    rewriter.setInsertionPointToEnd(newBody);
    for (Operation &op : llvm::make_early_inc_range(*oldBody)) {
      op.moveBefore(newBody, newBody->end());
    }
    for (auto [oldArg, newArg] :
         llvm::zip(oldBody->getArguments().take_front(fused.getRank()),
                   argRepl)) {
      rewriter.replaceAllUsesWith(oldArg, newArg);
    }
    rewriter.setInsertionPointToEnd(newBody);
    auto newTerm = scf::InParallelOp::create(rewriter, fused.getLoc());
    rewriter.setInsertionPointToStart(newTerm.getBody());
    tensor::ParallelInsertSliceOp::create(
        rewriter, fused.getLoc(), hTile.getResult(0),
        newBody->getArguments().back(), offs, sizes, strides);
    // h result in [1, M, F] form: collapse like the gate buffer was.
    rewriter.setInsertionPointAfter(newForall);
    tensor::CollapseShapeOp gateCollapse;
    for (Operation *user : fused.getResult(0).getUsers()) {
      if (auto c = dyn_cast<tensor::CollapseShapeOp>(user)) {
        gateCollapse = c;
      }
    }
    auto hCollapsed = tensor::CollapseShapeOp::create(
        rewriter, fused.getLoc(), newForall.getResult(0),
        gateCollapse.getReassociationIndices());
    if (hCollapsed.getType() != hForall.getResult(0).getType()) {
      return failure();
    }
    rewriter.replaceAllUsesWith(hForall.getResult(0), hCollapsed.getResult());
    rewriter.eraseOp(hForall);
    // fused's results are now only used by dead collapse_shapes.
    for (OpResult r : fused->getResults()) {
      for (Operation *u : llvm::make_early_inc_range(r.getUsers())) {
        if (u->use_empty()) {
          rewriter.eraseOp(u);
        }
      }
    }
    if (!fused->use_empty()) {
      return failure();
    }
    rewriter.eraseOp(fused);
    return success();
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGPUFuseMLPGateUpLoopsPass() {
  return std::make_unique<GPUFuseMLPGateUpLoopsPass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGPUTileMLPDownPerWarpPass() {
  return std::make_unique<GPUTileMLPDownPerWarpPass>();
}

bool isGatedMLPChainDown(Operation *op) {
  return succeeded(matchMLPChain(op, /*requireSharedInput=*/true));
}

} // namespace mlir::iree_compiler
