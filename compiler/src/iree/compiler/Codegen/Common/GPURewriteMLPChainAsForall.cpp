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
//   serial steps of kDownKTile (16).
//
// - Every lowering_config in the function is dropped, so the matmul-specific
//   TileAndFuse steps (operand promotion, packing to MMA intrinsics, the
//   reduction/thread/subgroup tiling levels) find nothing to act on.
//
// MMA path (used whenever the inner M dim is a multiple of 16, i.e. prefill
// -- decode keeps the scalar path), when the direct path does not apply: TM is
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
// Direct path (always taken when it applies: prefill-shaped chains whose
// sizes fit, see emitDirectMLPKernel): the chain is
// emitted as one already-distributed SIMT kernel instead -- 16 rows x all N
// columns per workgroup (no N split, gate/up computed once), acc in registers
// across the F loop, x staged in shared memory (ldmatrix), 16-byte weight
// loads with K relabeled per 32-block, h in registers -> shared. GX10 MLP
// kernel time, prefill S = 512 / 1024 / 2048: 37.1 / 76.9 / 123.2 ms (baseline
// matmuls 82.5 / 172.5 / 333.8 ms; MMA path below 111.9 / 178.4 / 358.4 ms).
// The MMA / scalar paths below remain the fallback.
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
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
// Row tile TM (fixed; smaller only when the inner M dim is not a multiple).
static constexpr int64_t kMaxRowTile = 8;
// N strip of `down` each thread owns (all TM rows of it): the inner
// `scf.for N (16)` of the schedule, distributed over the threads.
static constexpr int64_t kDownThreadN = 16;
// Scalar path: serial K step of down inside each thread (the `scf.for k2`
// loop over the 128-wide F chunk).
static constexpr int64_t kDownKTile = 16;
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
getProjectionMMAConfig(MLIRContext *ctx,
                       IREE::Codegen::InnerTileDescAttrInterface kind,
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
static LogicalResult
setMMAConfigs(scf::ForallOp forallOp,
              IREE::Codegen::InnerTileDescAttrInterface kind, int64_t nTile,
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
  int64_t intrinsicM, intrinsicN, intrinsicK;
  if (auto v = dyn_cast<IREE::GPU::VirtualMMAAttr>(kind)) {
    std::tie(intrinsicM, intrinsicN, intrinsicK) = v.getMNKShape();
  } else {
    std::tie(intrinsicM, intrinsicN, intrinsicK) =
        cast<IREE::GPU::MMAAttr>(kind).getMNKShape();
  }
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
                                kMMAReductionK * 16 / intrinsicK,
                                kMMAPromote));
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

//===----------------------------------------------------------------------===//
// Direct path: the whole chain as one SIMT kernel (no TileAndFuse levels).
//===----------------------------------------------------------------------===//

// Direct path parameters. One workgroup: kDirectRows rows x all N columns.
static constexpr int64_t kDirectRows = 16;   // one mma.m16n8k16 M tile
static constexpr int64_t kDirectWarps = 16;  // 512 threads
static constexpr int64_t kDirectChunk = 128; // F chunk (h tile width)
static constexpr int64_t kDirectGroup = 4;   // down n-tiles per load group
static constexpr int64_t kDirectPrefetch = 2; // gate/up k32 blocks in flight
static constexpr int64_t kDirectUnroll = 16;  // gate/up k loop unroll

/// The 2-D buffer a chain operand was loaded from: through tensor/memref
/// expand_shapes back to the binding subspan.
static Value getChainBuffer(Value v) {
  while (auto e = v.getDefiningOp<tensor::ExpandShapeOp>()) {
    v = e.getSrc();
  }
  auto load = v.getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
  if (!load) {
    return {};
  }
  Value m = load.getBuffer();
  while (auto e = m.getDefiningOp<memref::ExpandShapeOp>()) {
    m = e.getSrc();
  }
  return m;
}

/// Sets the byte alignment of a vector.load / vector.store (all accesses of
/// the direct kernel are aligned by construction; without it LLVM splits the
/// unaligned-looking ones into 2-byte accesses).
static Value alignedLoad(Value loadResult, int64_t bytes) {
  loadResult.getDefiningOp()->setAttr(
      "alignment",
      IntegerAttr::get(IntegerType::get(loadResult.getContext(), 64), bytes));
  return loadResult;
}
static void alignedStore(Operation *op, int64_t bytes) {
  op->setAttr("alignment",
              IntegerAttr::get(IntegerType::get(op->getContext(), 64), bytes));
}

/// Applies the elementwise body of `generic` (all inputs and the result
/// elementwise, identity maps) to vector values `ins`: every body op is
/// recreated on vector types; values defined outside the body are broadcast.
static Value applyElementwiseBody(OpBuilder &b, Location loc,
                                  linalg::GenericOp generic, ValueRange ins,
                                  int64_t width) {
  IRMapping map;
  Block *body = generic.getBody();
  for (auto [arg, v] : llvm::zip(body->getArguments().take_front(ins.size()),
                                 ins)) {
    map.map(arg, v);
  }
  auto vecOf = [&](Type t) -> Type { return VectorType::get({width}, t); };
  for (Operation &op : body->without_terminator()) {
    SmallVector<Value> operands;
    for (Value o : op.getOperands()) {
      if (map.contains(o)) {
        operands.push_back(map.lookup(o));
      } else {
        operands.push_back(vector::BroadcastOp::create(
            b, loc, cast<VectorType>(vecOf(o.getType())), o));
      }
    }
    OperationState state(loc, op.getName());
    state.addOperands(operands);
    for (Type t : op.getResultTypes()) {
      state.addTypes(vecOf(t));
    }
    state.addAttributes(op.getAttrs());
    Operation *newOp = b.create(state);
    map.map(op.getResults(), newOp->getResults());
  }
  return map.lookup(body->getTerminator()->getOperand(0));
}

/// Emits the chain as one kernel, modeled on a hand-written CUDA kernel:
///   forall workgroups (16 rows of M):
///     x[16, H] -> shared (K relabeled in 32-blocks so each lane's B fragment
///                 of an mma.m16n8k16 pair is one 16-byte load)
///     acc[16, N] in registers (warp w owns N / 16 columns)
///     for each F chunk of 128:
///       gate/up: warp w computes 8 columns, Wg/Wu streamed from global with
///                kDirectPrefetch k32 blocks in flight, x via ldmatrix
///       h = act(gate, up) in registers -> shared (double buffered)
///       barrier
///       down: acc += h @ Wd[:, chunk]^T, Wd in groups of kDirectGroup tiles,
///             the next group's loads issued before this group's MMAs
///     out = epilogue(residual, acc)
/// The act / epilogue bodies are applied as they are (same f16 arithmetic).
static LogicalResult emitDirectMLPKernel(RewriterBase &rewriter,
                                         FunctionOpInterface funcOp,
                                         MLPChainMatch &match) {
  MLIRContext *ctx = rewriter.getContext();
  // Rows: the inner M dim of the (split) chain must hold whole 16-row tiles.
  auto outType = cast<ShapedType>(match.down->getResult(0).getType());
  if (outType.getRank() < 2) {
    return failure();
  }
  int64_t mInner = outType.getDimSize(outType.getRank() - 2);
  if (ShapedType::isDynamic(mInner) || mInner % kDirectRows != 0) {
    return failure();
  }
  Value xBuf = getChainBuffer(match.gate.getDpsInputs()[0]);
  Value wgBuf = getChainBuffer(match.gate.getDpsInputs()[1]);
  Value wuBuf = getChainBuffer(match.up.getDpsInputs()[1]);
  Value wdBuf = getChainBuffer(match.down.getDpsInputs()[1]);
  Value resBuf;
  int downInput = -1;
  for (auto [i, in] : llvm::enumerate(match.epilogue.getDpsInputs())) {
    if (in == match.down->getResult(0)) {
      downInput = i;
    } else {
      resBuf = getChainBuffer(in);
    }
  }
  IREE::Codegen::StoreToBufferOp store;
  for (Operation *user : match.epilogue->getResult(0).getUsers()) {
    store = dyn_cast<IREE::Codegen::StoreToBufferOp>(user);
  }
  if (!xBuf || !wgBuf || !wuBuf || !wdBuf || !resBuf || !store ||
      downInput < 0 || match.epilogue.getNumDpsInputs() != 2 ||
      match.act.getNumDpsInputs() != 2) {
    return failure();
  }
  Value outBuf = store.getBuffer();
  while (auto e = outBuf.getDefiningOp<memref::ExpandShapeOp>()) {
    outBuf = e.getSrc();
  }
  auto memType = [](Value v) { return cast<MemRefType>(v.getType()); };
  for (Value v : {xBuf, wgBuf, wuBuf, wdBuf, resBuf, outBuf}) {
    if (memType(v).getRank() != 2 || !memType(v).getElementType().isF16()) {
      return failure();
    }
  }
  int64_t h = memType(xBuf).getDimSize(1);
  int64_t f = memType(wgBuf).getDimSize(0);
  int64_t n = memType(wdBuf).getDimSize(0);
  if (ShapedType::isDynamic(h) || ShapedType::isDynamic(f) ||
      ShapedType::isDynamic(n) || memType(wgBuf).getDimSize(1) != h ||
      memType(wuBuf).getShape() != memType(wgBuf).getShape() ||
      memType(wdBuf).getDimSize(1) != f || h % 32 != 0 ||
      f % kDirectChunk != 0 || kDirectChunk / kDirectWarps != 8 ||
      n % (kDirectWarps * 8 * kDirectGroup) != 0 ||
      memType(resBuf).getDimSize(1) != n ||
      memType(outBuf).getDimSize(1) != n) {
    return failure();
  }
  const int64_t xsStride = h + 8, hsStride = kDirectChunk + 8;
  int64_t sharedBytes = 2 * (kDirectRows * xsStride +
                             2 * kDirectRows * hsStride);
  int64_t limit = kDefaultWorkgroupMemoryBytes;
  if (IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp)) {
    limit = target.getWgp().getMaxWorkgroupMemoryBytes();
    if (kDirectWarps * 32 > target.getWgp().getMaxThreadCountPerWorkgroup()) {
      return failure();
    }
  }
  if (sharedBytes > limit) {
    return failure();
  }
  const int64_t wcols = n / kDirectWarps;  // down columns per warp
  const int64_t ntDown = wcols / 8;
  // Which act input is gate (the order of the body's arguments).
  SmallVector<int> actFromGate;
  for (Value in : match.act.getDpsInputs()) {
    Operation *def = in.getDefiningOp();
    if (def == match.gate.getOperation()) {
      actFromGate.push_back(1);
    } else if (def == match.up.getOperation()) {
      actFromGate.push_back(0);
    } else {
      return failure();
    }
  }

  Location loc = store.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(store);
  auto cst = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(rewriter, loc, v);
  };
  Type f16 = rewriter.getF16Type(), f32 = rewriter.getF32Type();
  auto vA = VectorType::get({4, 2}, f16);
  auto vB = VectorType::get({2, 2}, f16);
  auto vC = VectorType::get({2, 2}, f32);
  auto v8 = VectorType::get({8}, f16);
  Value m = memref::DimOp::create(rewriter, loc, xBuf, 0);
  Value numWG = arith::DivUIOp::create(rewriter, loc, m, cst(kDirectRows));
  auto mapping = rewriter.getArrayAttr(
      {IREE::Codegen::WorkgroupMappingAttr::get(
          ctx, IREE::Codegen::WorkgroupId::IdX)});
  auto forall = scf::ForallOp::create(
      rewriter, loc, ArrayRef<OpFoldResult>{numWG}, ValueRange{}, mapping,
      /*bodyBuilderFn=*/nullptr);
  OpBuilder b = OpBuilder::atBlockBegin(forall.getBody());
  // `cst` uses rewriter; inside the forall use b.
  auto c = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  auto add = [&](Value x, Value y) -> Value {
    return arith::AddIOp::create(b, loc, x, y);
  };
  auto mul = [&](Value x, int64_t y) -> Value {
    return arith::MulIOp::create(b, loc, x, c(y));
  };
  Value m0 = mul(forall.getInductionVar(0), kDirectRows);
  Value tid = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value lane = arith::RemUIOp::create(b, loc, tid, c(32));
  Value warp = arith::DivUIOp::create(b, loc, tid, c(32));
  Value g = arith::DivUIOp::create(b, loc, lane, c(4));
  Value t = arith::RemUIOp::create(b, loc, lane, c(4));
  Value t8 = mul(t, 8);
  Value aRow = arith::RemUIOp::create(b, loc, lane, c(16));
  Value aCol = mul(arith::DivUIOp::create(b, loc, lane, c(16)), 8);

  auto wgSpace =
      gpu::AddressSpaceAttr::get(ctx,
                                 gpu::GPUDialect::getWorkgroupAddressSpace());
  auto xsType = MemRefType::get({kDirectRows, xsStride}, f16,
                                MemRefLayoutAttrInterface{}, wgSpace);
  auto hsType = MemRefType::get({2, kDirectRows, hsStride}, f16,
                                MemRefLayoutAttrInterface{}, wgSpace);
  Value xs = memref::AllocOp::create(b, loc, xsType);
  Value hs = memref::AllocOp::create(b, loc, hsType);

  // ---- x tile -> shared, K relabeled: logical word 4q + t <- physical 4t + q.
  {
    int64_t blocks = kDirectRows * (h / 32);
    auto loop = scf::ForOp::create(b, loc, tid, c(blocks),
                                   c(kDirectWarps * 32));
    OpBuilder lb = OpBuilder::atBlockBegin(loop.getBody());
    Value blk = loop.getInductionVar();
    Value nkb = arith::ConstantIndexOp::create(lb, loc, h / 32);
    Value row = arith::DivUIOp::create(lb, loc, blk, nkb);
    Value kb = arith::RemUIOp::create(lb, loc, blk, nkb);
    Value col = arith::MulIOp::create(
        lb, loc, kb, arith::ConstantIndexOp::create(lb, loc, 32));
    Value grow = arith::AddIOp::create(lb, loc, m0, row);
    Value v = alignedLoad(vector::LoadOp::create(lb, loc,
                                             VectorType::get({32}, f16), xBuf,
                                             ValueRange{grow, col}),
                      16);
    auto i32x16 = VectorType::get({16}, lb.getI32Type());
    Value words = vector::BitCastOp::create(lb, loc, i32x16, v);
    SmallVector<int64_t> mask(16);
    for (int64_t q = 0; q < 4; ++q) {
      for (int64_t tt = 0; tt < 4; ++tt) {
        mask[4 * q + tt] = 4 * tt + q;
      }
    }
    Value shuf = vector::ShuffleOp::create(lb, loc, words, words, mask);
    Value back =
        vector::BitCastOp::create(lb, loc, VectorType::get({32}, f16), shuf);
    alignedStore(
        vector::StoreOp::create(lb, loc, back, xs, ValueRange{row, col}), 16);
  }
  gpu::BarrierOp::create(b, loc);

  auto ldsm = [&](OpBuilder &ob, Value mem, ValueRange idx) -> Value {
    return nvgpu::LdMatrixOp::create(ob, loc, vA, mem, idx,
                                     /*transpose=*/false, /*numTiles=*/4);
  };
  auto mma = [&](OpBuilder &ob, Value a, Value b8, int64_t s,
                 Value acc) -> Value {
    Value bs = vector::ExtractStridedSliceOp::create(
        ob, loc, b8, ArrayRef<int64_t>{4 * s}, ArrayRef<int64_t>{4},
        ArrayRef<int64_t>{1});
    bs = vector::ShapeCastOp::create(ob, loc, vB, bs);
    return nvgpu::MmaSyncOp::create(ob, loc, a, bs, acc,
                                    ArrayRef<int64_t>{16, 8, 16});
  };
  Value zeroC = arith::ConstantOp::create(
      b, loc, vC, DenseElementsAttr::get(vC, b.getF32FloatAttr(0.f)));

  // ---- F chunk loop, acc tiles as iter_args.
  SmallVector<Value> accInit(ntDown, zeroC);
  auto chunkLoop = scf::ForOp::create(b, loc, c(0), c(f), c(kDirectChunk),
                                      accInit);
  {
    OpBuilder cb = OpBuilder::atBlockBegin(chunkLoop.getBody());
    auto cc = [&](int64_t v) -> Value {
      return arith::ConstantIndexOp::create(cb, loc, v);
    };
    Value k2 = chunkLoop.getInductionVar();
    SmallVector<Value> acc(chunkLoop.getRegionIterArgs().begin(),
                           chunkLoop.getRegionIterArgs().end());
    // gate / up rows of this lane: k2 + 8 * warp + g.
    Value wrow = arith::AddIOp::create(
        cb, loc, k2,
        arith::AddIOp::create(cb, loc,
                              arith::MulIOp::create(cb, loc, warp, cc(8)), g));
    auto ldW = [&](OpBuilder &ob, Value mem, Value row, Value col) -> Value {
      return alignedLoad(
          vector::LoadOp::create(ob, loc, v8, mem, ValueRange{row, col}), 16);
    };
    auto kcol = [&](OpBuilder &ob, Value kb) -> Value {
      return arith::AddIOp::create(
          ob, loc,
          arith::MulIOp::create(ob, loc, kb,
                                arith::ConstantIndexOp::create(ob, loc, 32)),
          t8);
    };
    SmallVector<Value> kbInit = {zeroC, zeroC};
    for (int64_t p = 0; p < kDirectPrefetch; ++p) {
      kbInit.push_back(ldW(cb, wgBuf, wrow, kcol(cb, cc(p))));
      kbInit.push_back(ldW(cb, wuBuf, wrow, kcol(cb, cc(p))));
    }
    auto kbLoop =
        scf::ForOp::create(cb, loc, cc(0), cc(h / 32), cc(1), kbInit);
    {
      OpBuilder kb_ = OpBuilder::atBlockBegin(kbLoop.getBody());
      Value kb = kbLoop.getInductionVar();
      auto args = kbLoop.getRegionIterArgs();
      Value ga = args[0], ua = args[1];
      Value nk = arith::MinUIOp::create(
          kb_, loc,
          arith::AddIOp::create(
              kb_, loc, kb,
              arith::ConstantIndexOp::create(kb_, loc, kDirectPrefetch)),
          arith::ConstantIndexOp::create(kb_, loc, h / 32 - 1));
      Value ng = ldW(kb_, wgBuf, wrow, kcol(kb_, nk));
      Value nu = ldW(kb_, wuBuf, wrow, kcol(kb_, nk));
      for (int64_t s = 0; s < 2; ++s) {
        Value colA = arith::AddIOp::create(
            kb_, loc,
            arith::MulIOp::create(kb_, loc, kb,
                                  arith::ConstantIndexOp::create(kb_, loc, 32)),
            arith::AddIOp::create(
                kb_, loc, arith::ConstantIndexOp::create(kb_, loc, 16 * s),
                aCol));
        Value a = ldsm(kb_, xs, ValueRange{aRow, colA});
        ga = mma(kb_, a, args[2], s, ga);
        ua = mma(kb_, a, args[3], s, ua);
      }
      SmallVector<Value> yields = {ga, ua};
      for (int64_t p = 1; p < kDirectPrefetch; ++p) {
        yields.push_back(args[2 + 2 * p]);
        yields.push_back(args[3 + 2 * p]);
      }
      yields.push_back(ng);
      yields.push_back(nu);
      scf::YieldOp::create(kb_, loc, yields);
    }
    Value ga = kbLoop.getResult(0), ua = kbLoop.getResult(1);
    // ---- h = act(gate, up) -> hs[chunk % 2], K relabeled like x.
    Value buf = arith::RemUIOp::create(
        cb, loc, arith::DivUIOp::create(cb, loc, k2, cc(kDirectChunk)), cc(2));
    // Physical f (within the chunk) of this lane's pair: 8 * warp + 2t;
    // logical position: (f & ~31) + 8 * ((f >> 1) & 3) + 2 * (f >> 3 & 3).
    Value fpos = arith::AddIOp::create(
        cb, loc, arith::MulIOp::create(cb, loc, warp, cc(8)),
        arith::MulIOp::create(cb, loc, t, cc(2)));
    Value fblk = arith::MulIOp::create(
        cb, loc, arith::DivUIOp::create(cb, loc, fpos, cc(32)), cc(32));
    Value fin = arith::RemUIOp::create(cb, loc, fpos, cc(32));
    Value tt = arith::DivUIOp::create(cb, loc, fin, cc(8));
    Value q = arith::RemUIOp::create(
        cb, loc, arith::DivUIOp::create(cb, loc, fin, cc(2)), cc(4));
    Value lpos = arith::AddIOp::create(
        cb, loc, fblk,
        arith::AddIOp::create(cb, loc,
                              arith::MulIOp::create(cb, loc, q, cc(8)),
                              arith::MulIOp::create(cb, loc, tt, cc(2))));
    for (int64_t r = 0; r < 2; ++r) {
      Value gr = vector::ExtractOp::create(cb, loc, ga, ArrayRef<int64_t>{r});
      Value ur = vector::ExtractOp::create(cb, loc, ua, ArrayRef<int64_t>{r});
      SmallVector<Value> ins;
      for (int fromGate : actFromGate) {
        ins.push_back(fromGate ? gr : ur);
      }
      Value hv = applyElementwiseBody(cb, loc, match.act, ins, 2);
      Value hrow = arith::AddIOp::create(cb, loc, g, cc(8 * r));
      alignedStore(
          vector::StoreOp::create(cb, loc, hv, hs, ValueRange{buf, hrow, lpos}),
          4);
    }
    gpu::BarrierOp::create(cb, loc);
    // ---- down.
    Value wdRow0 = arith::AddIOp::create(
        cb, loc, arith::MulIOp::create(cb, loc, warp, cc(wcols)), g);
    const int64_t ng = ntDown / kDirectGroup;
    const int64_t steps = (kDirectChunk / 32) * ng;
    auto loadGroup = [&](int64_t st) {
      int64_t kbi = st / ng, gi = st % ng;
      Value col = arith::AddIOp::create(
          cb, loc, k2, arith::AddIOp::create(cb, loc, cc(kbi * 32), t8));
      SmallVector<Value> vs;
      for (int64_t i = 0; i < kDirectGroup; ++i) {
        Value row = arith::AddIOp::create(
            cb, loc, wdRow0, cc((gi * kDirectGroup + i) * 8));
        vs.push_back(ldW(cb, wdBuf, row, col));
      }
      return vs;
    };
    SmallVector<Value> cur = loadGroup(0);
    for (int64_t st = 0; st < steps; ++st) {
      SmallVector<Value> next;
      if (st + 1 < steps) {
        next = loadGroup(st + 1);
      }
      int64_t kbi = st / ng, gi = st % ng;
      Value a0 = ldsm(cb, hs,
                      ValueRange{buf, aRow,
                                 arith::AddIOp::create(cb, loc, cc(kbi * 32),
                                                       aCol)});
      Value a1 = ldsm(cb, hs,
                      ValueRange{buf, aRow,
                                 arith::AddIOp::create(
                                     cb, loc, cc(kbi * 32 + 16), aCol)});
      for (int64_t i = 0; i < kDirectGroup; ++i) {
        int64_t tile = gi * kDirectGroup + i;
        acc[tile] = mma(cb, a0, cur[i], 0, acc[tile]);
        acc[tile] = mma(cb, a1, cur[i], 1, acc[tile]);
      }
      cur = next;
    }
    scf::YieldOp::create(cb, loc, acc);
    if (kDirectUnroll > 1) {
      (void)loopUnrollByFactor(kbLoop, kDirectUnroll);
    }
  }

  // ---- out = epilogue(residual, acc).
  for (int64_t i = 0; i < ntDown; ++i) {
    Value col = add(add(mul(warp, wcols), c(i * 8)), mul(t, 2));
    for (int64_t r = 0; r < 2; ++r) {
      Value row = add(m0, add(g, c(8 * r)));
      Value accRow = vector::ExtractOp::create(b, loc, chunkLoop.getResult(i),
                                               ArrayRef<int64_t>{r});
      Value resv =
          alignedLoad(vector::LoadOp::create(b, loc, VectorType::get({2}, f16),
                                         resBuf, ValueRange{row, col}),
                  4);
      SmallVector<Value> ins(2);
      ins[downInput] = accRow;
      ins[1 - downInput] = resv;
      Value o = applyElementwiseBody(b, loc, match.epilogue, ins, 2);
      alignedStore(
          vector::StoreOp::create(b, loc, o, outBuf, ValueRange{row, col}), 4);
    }
  }

  rewriter.eraseOp(store);
  funcOp->setAttr("iree_codegen.mlp_direct", UnitAttr::get(ctx));
  IREE::Codegen::TranslationInfoAttr info = getTranslationInfo(funcOp);
  // (GPUReduceBankConflicts skips iree_codegen.mlp_direct functions: the
  // shared buffers are already padded, and ldmatrix needs 16-byte rows.)
  (void)setTranslationInfo(
      funcOp, IREE::Codegen::TranslationInfoAttr::get(
                  ctx, info.getPassPipeline(), info.getCodegenSpec(),
                  ArrayRef<int64_t>{kDirectWarps * 32, 1, 1}, 32,
                  info.getConfiguration()));
  return success();
}

static LogicalResult rewriteMLPChain(RewriterBase &rewriter,
                                     FunctionOpInterface funcOp,
                                     MLPChainMatch &match) {
  MLIRContext *ctx = rewriter.getContext();
  if (succeeded(emitDirectMLPKernel(rewriter, funcOp, match))) {
    return success();
  }
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
  // 16x8x32 virtual mma.sync: two m16n8k16 with K (and M) permuted so each
  // lane's A/B operands are 8 contiguous K elements, i.e. 16-byte loads.
  // Same matmul up to f32 summation order (not bit-identical to the native
  // 16x8x16 order). GX10 MLP kernel time, prefill S = 512 / 2048, with the
  // unchecked x loads below: 163.7 / 504.2 -> 108.7 / 358.9 ms.
  IREE::Codegen::InnerTileDescAttrInterface mmaKind =
      IREE::GPU::VirtualMMAAttr::get(
          ctx, IREE::GPU::VirtualMMAIntrinsic::VNV_MMA_SYNC_F32_16x8x32_F16);
  int64_t nTile = kMMANTile;
  int64_t subgroupSize = info.getSubgroupSize() ? info.getSubgroupSize() : 32;
  bool useMMA = !ShapedType::isDynamic(mInner) &&
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
      reductionStep = kDownKTile;
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
    // The chain's map_loads only relayout in-bounds tensors: let their
    // lowering skip the per-element bounds check (MapLoadOp scalar
    // implementation), so x's loads become vector loads.
    funcOp->setAttr("iree_codegen.mlp_map_load_unchecked",
                    UnitAttr::get(&getContext()));
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
