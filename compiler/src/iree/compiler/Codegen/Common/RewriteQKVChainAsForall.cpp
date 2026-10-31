// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// RewriteQKVChainAsForall
//
// Lowers the attention-prologue dispatch that DispatchCreation forms under
// --iree-dispatch-creation-fuse-qkv (RoPE on Q tagged
// `iree_dispatch.special_qkv`):
//
//   ss    = sum(x^2)                        [S] f32      (reduction)
//   xn    = rmsnorm(x, ss) * w              [S, D] f16   (one elementwise op)
//   q/k/v = trunc(xn * Wq/Wk/Wv^T)          [S, Nq/Nk/Nv] f16
//   qr    = rope(q[s, h*64 + 2j], q[s, h*64 + 2j + 1], cos, sin)   (result)
//   kr    = rope(k[...], ...)                                        (result)
//   v                                                                 (result)
//
// into one scf.forall over S row tiles in which nothing larger than the
// [TM, D] normalized rows and one [TM, 64] head is ever live:
//
//   scf.forall m in [0, S) step TM                         (workgroups)
//     ss = sum(x[m]^2) ; xn = rmsnorm(x[m], ss) * w        [TM, D] f16
//     scf.for h in heads(Q):                               (32 for llama 1B)
//       acc = xn * Wq[h*64 : h*64+64]^T                    [TM, 64] f32
//       qr[.., h, m, ..] = rope(trunc(acc), cos[m], sin[m])
//     scf.for h in heads(K):  (same, 8 heads)
//     scf.for c in chunks(V): v[m, c*64 : c*64+64] = trunc(xn * Wv[c]^T)
//
// A RoPE pair (2j, 2j + 1) always lies inside one 64-wide head, so tiling the
// projections per head keeps the RoPE a pure epilogue. The norm is computed
// once per row tile and shared by all three projections.
//
// The ops are tiled by hand with TilingInterface: each op is tiled at the
// iteration offsets of the current (row tile, head), and every operand slice
// it then takes of a chain value is re-pointed at the in-loop tile of that
// value. TM is the largest power of two, at most the Q matmul's distribution
// M size, whose stack buffers ([TM, D] f16 xn plus one head's f32 accumulator
// and f16 trunc) fit 3/4 of `max_stack_allocation_size`.
//
// Every op is then tiled on its own to vector size and its lowering_config is
// dropped, as RewriteMLPChainAsForall does. The RoPE ops read their input at
// 2j / 2j + 1, which the linalg vectorizer does not take; they stay scalar
// loops (a few elementwise ops per output element, small next to the
// matmuls).
//
// The function is tagged `iree_codegen.qkv_chain_rewritten` so
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "rewrite-qkv-chain-as-forall"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REWRITEQKVCHAINASFORALLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static constexpr StringLiteral kSpecialQKVAttr = "iree_dispatch.special_qkv";
static constexpr StringLiteral kQKVChainRewrittenAttr =
    "iree_codegen.qkv_chain_rewritten";

// Fallback vector tile ([M, N] outputs over a K step), as in
// RewriteMLPChainAsForall.
static constexpr int64_t kVecM = 8;
static constexpr int64_t kVecN = 16;
static constexpr int64_t kVecK = 16;
static constexpr int64_t kDefaultStackLimitBytes = 32 * 1024;

//===----------------------------------------------------------------------===//
// Matching
//===----------------------------------------------------------------------===//

static bool isTransposedBMatmul(Operation *op) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp) ||
      linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1 ||
      linalgOp.getNumLoops() != 3) {
    return false;
  }
  MLIRContext *ctx = op->getContext();
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  SmallVector<AffineMap> expected = {AffineMap::get(3, 0, {m, k}, ctx),
                                     AffineMap::get(3, 0, {n, k}, ctx),
                                     AffineMap::get(3, 0, {m, n}, ctx)};
  return linalgOp.getIndexingMapsArray() == expected;
}

static bool isTruncOf(Operation *op, Operation *producer) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 ||
      generic.getNumLoops() != generic.getNumParallelLoops() ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); })) {
    return false;
  }
  Block *body = generic.getBlock();
  return llvm::hasNItems(*body, 2) && isa<arith::TruncFOp>(body->front()) &&
         generic.getDpsInputs()[0].getDefiningOp() == producer;
}

static SmallVector<Operation *> getNonDimUsers(Value v) {
  SmallVector<Operation *> users;
  for (Operation *user : v.getUsers()) {
    if (!isa<tensor::DimOp>(user) && !llvm::is_contained(users, user)) {
      users.push_back(user);
    }
  }
  return users;
}

/// A RoPE op and where it reads its projection: `t[row_dim, col]` with
/// `col = sum(coeff_i * d_i) + c`. `jDim` is the loop with the smallest
/// coefficient (the pair index); every other loop in `col` is a head loop.
struct RopeInfo {
  linalg::GenericOp op;
  unsigned rowDim = 0;
  unsigned jDim = 0;
  SmallVector<unsigned> headDims;       // by decreasing coefficient
  SmallVector<int64_t> headCoeffs;      // their coefficients
  int64_t numHeads = 1;
  int64_t headWidth = 0;                // columns per head
};

/// Coefficients of a linear `sum(c_i * d_i) + c` expression, or nullopt.
static std::optional<SmallVector<int64_t>> getLinearCoeffs(AffineExpr e,
                                                           unsigned numDims) {
  SmallVector<int64_t> coeffs(numDims, 0);
  std::function<bool(AffineExpr, int64_t)> visit = [&](AffineExpr e,
                                                       int64_t scale) {
    if (auto dim = dyn_cast<AffineDimExpr>(e)) {
      coeffs[dim.getPosition()] += scale;
      return true;
    }
    if (isa<AffineConstantExpr>(e)) {
      return true;
    }
    auto bin = dyn_cast<AffineBinaryOpExpr>(e);
    if (!bin) {
      return false;
    }
    if (bin.getKind() == AffineExprKind::Add) {
      return visit(bin.getLHS(), scale) && visit(bin.getRHS(), scale);
    }
    if (bin.getKind() == AffineExprKind::Mul) {
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
        return visit(bin.getLHS(), scale * c.getValue());
      }
      if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
        return visit(bin.getRHS(), scale * c.getValue());
      }
    }
    return false;
  };
  if (!visit(e, 1)) {
    return std::nullopt;
  }
  return coeffs;
}

static FailureOr<RopeInfo> analyzeRope(linalg::GenericOp rope, Value t) {
  RopeInfo info;
  info.op = rope;
  unsigned numDims = rope.getNumLoops();
  if (rope.getNumLoops() != rope.getNumParallelLoops() ||
      rope.getNumDpsInits() != 1) {
    return failure();
  }
  AffineMap readMap;
  for (OpOperand *input : rope.getDpsInputOperands()) {
    if (input->get() == t) {
      readMap = rope.getMatchingIndexingMap(input);
      break;
    }
  }
  if (!readMap || readMap.getNumResults() != 2) {
    return failure();
  }
  auto row = dyn_cast<AffineDimExpr>(readMap.getResult(0));
  if (!row) {
    return failure();
  }
  info.rowDim = row.getPosition();
  std::optional<SmallVector<int64_t>> coeffs =
      getLinearCoeffs(readMap.getResult(1), numDims);
  if (!coeffs) {
    return failure();
  }
  SmallVector<int64_t> ranges = rope.getStaticLoopRanges();
  SmallVector<unsigned> colDims;
  for (unsigned d = 0; d < numDims; ++d) {
    if ((*coeffs)[d] != 0) {
      if (d == info.rowDim || ShapedType::isDynamic(ranges[d])) {
        return failure();
      }
      colDims.push_back(d);
    }
  }
  if (colDims.size() < 2) {
    return failure();
  }
  llvm::sort(colDims, [&](unsigned a, unsigned b) {
    return (*coeffs)[a] > (*coeffs)[b];
  });
  info.jDim = colDims.back();
  for (unsigned d : ArrayRef<unsigned>(colDims).drop_back()) {
    info.headDims.push_back(d);
    info.headCoeffs.push_back((*coeffs)[d]);
    info.numHeads *= ranges[d];
  }
  info.headWidth = info.headCoeffs.back();
  // Heads must tile the projection's columns exactly, row-major.
  int64_t n = cast<ShapedType>(t.getType()).getDimSize(1);
  if (ShapedType::isDynamic(n) || info.numHeads * info.headWidth != n ||
      (*coeffs)[info.jDim] * ranges[info.jDim] != info.headWidth) {
    return failure();
  }
  for (unsigned i = 0; i + 1 < info.headDims.size(); ++i) {
    if (info.headCoeffs[i] !=
        info.headCoeffs[i + 1] * ranges[info.headDims[i + 1]]) {
      return failure();
    }
  }
  return info;
}

struct QKVChain {
  RopeInfo ropeQ, ropeK;
  // truncQ / truncK are null when folded into the RoPE.
  linalg::GenericOp truncQ, truncK, truncV;
  linalg::LinalgOp mmQ, mmK, mmV;
  linalg::GenericOp norm;
  linalg::GenericOp sumsq;
};

/// The input of `rope` that is a matmul result, or the trunc of one (after
/// dispatch formation, elementwise fusion usually folds the trunc into the
/// RoPE), with that matmul.
static Value getRopeProjection(linalg::GenericOp rope, Operation *&mm) {
  for (Value input : rope.getDpsInputs()) {
    if (isTransposedBMatmul(input.getDefiningOp())) {
      mm = input.getDefiningOp();
      return input;
    }
    auto trunc = input.getDefiningOp<linalg::GenericOp>();
    if (!trunc || trunc.getNumDpsInputs() != 1) {
      continue;
    }
    Operation *producer = trunc.getDpsInputs()[0].getDefiningOp();
    if (isTransposedBMatmul(producer) && isTruncOf(trunc, producer)) {
      mm = producer;
      return input;
    }
  }
  return Value();
}

static FailureOr<QKVChain> matchQKVChain(Operation *root) {
  QKVChain chain;
  auto ropeQ = dyn_cast<linalg::GenericOp>(root);
  if (!ropeQ) {
    return failure();
  }
  Operation *mmQ = nullptr;
  Value tq = getRopeProjection(ropeQ, mmQ);
  if (!tq) {
    return failure();
  }
  FailureOr<RopeInfo> qInfo = analyzeRope(ropeQ, tq);
  if (failed(qInfo)) {
    return failure();
  }
  chain.ropeQ = *qInfo;
  chain.mmQ = cast<linalg::LinalgOp>(mmQ);
  if (tq.getDefiningOp() != mmQ) {
    chain.truncQ = tq.getDefiningOp<linalg::GenericOp>();
  }

  Value xn = chain.mmQ.getDpsInputs()[0];
  chain.norm = xn.getDefiningOp<linalg::GenericOp>();
  if (!chain.norm || chain.norm->getNumResults() != 1 ||
      chain.norm.getNumLoops() != chain.norm.getNumParallelLoops() ||
      chain.norm.getNumLoops() != 2) {
    return failure();
  }
  for (Value input : chain.norm.getDpsInputs()) {
    auto generic = input.getDefiningOp<linalg::GenericOp>();
    if (generic && generic.getNumReductionLoops() == 1 &&
        generic.getNumLoops() == 2 && generic.getNumDpsInits() == 1) {
      chain.sumsq = generic;
    }
  }
  if (!chain.sumsq) {
    return failure();
  }

  SmallVector<Operation *> matmuls = getNonDimUsers(xn);
  if (matmuls.size() != 3 || !llvm::all_of(matmuls, isTransposedBMatmul)) {
    return failure();
  }
  for (Operation *mm : matmuls) {
    if (mm == mmQ) {
      continue;
    }
    SmallVector<Operation *> users = getNonDimUsers(mm->getResult(0));
    if (users.size() != 1) {
      return failure();
    }
    // K: the RoPE reads the matmul directly or through a trunc.
    auto direct = dyn_cast<linalg::GenericOp>(users[0]);
    Operation *ropeMM = nullptr;
    if (direct && !isTruncOf(direct, mm) &&
        getRopeProjection(direct, ropeMM) == mm->getResult(0)) {
      FailureOr<RopeInfo> kInfo = analyzeRope(direct, mm->getResult(0));
      if (chain.mmK || failed(kInfo)) {
        return failure();
      }
      chain.ropeK = *kInfo;
      chain.mmK = cast<linalg::LinalgOp>(mm);
      continue;
    }
    if (!isTruncOf(users[0], mm)) {
      return failure();
    }
    auto trunc = cast<linalg::GenericOp>(users[0]);
    linalg::GenericOp rope;
    for (Operation *user : getNonDimUsers(trunc->getResult(0))) {
      auto generic = dyn_cast<linalg::GenericOp>(user);
      if (generic && generic != ropeQ &&
          getRopeProjection(generic, ropeMM) == trunc->getResult(0)) {
        rope = generic;
      }
    }
    if (rope) {
      FailureOr<RopeInfo> kInfo = analyzeRope(rope, trunc->getResult(0));
      if (chain.mmK || failed(kInfo)) {
        return failure();
      }
      chain.ropeK = *kInfo;
      chain.mmK = cast<linalg::LinalgOp>(mm);
      chain.truncK = trunc;
    } else {
      if (chain.mmV) {
        return failure();
      }
      chain.mmV = cast<linalg::LinalgOp>(mm);
      chain.truncV = trunc;
    }
  }
  if (!chain.mmK || !chain.mmV) {
    return failure();
  }
  // V is cut into chunks of K's head width.
  int64_t nv = cast<ShapedType>(chain.truncV->getResult(0).getType())
                   .getDimSize(1);
  if (ShapedType::isDynamic(nv) || nv % chain.ropeK.headWidth != 0) {
    return failure();
  }
  return chain;
}

//===----------------------------------------------------------------------===//
// Tiling helpers
//===----------------------------------------------------------------------===//

/// In-loop tile of a full-size chain value, with the tile's offsets in it.
struct Tile {
  Value value;
  SmallVector<OpFoldResult> offsets;
};
using TileMap = llvm::DenseMap<Value, Tile>;

static OpFoldResult subOfr(OpBuilder &b, Location loc, OpFoldResult a,
                           OpFoldResult c) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  return affine::makeComposedFoldedAffineApply(b, loc, d0 - d1, {a, c});
}

/// Tiles `op` at iteration `offsets` / `sizes` and re-points each operand
/// slice of a value in `tiles` at that value's tile; operand slices of a
/// linalg.fill result get a tiled fill instead. `init`, if set, replaces the
/// tiled op's init. Records the result tile in `tiles` and returns it.
static FailureOr<Value> tileOp(OpBuilder &b, Operation *op,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes, TileMap &tiles,
                               Value init = Value()) {
  Location loc = op->getLoc();
  auto tilingOp = cast<TilingInterface>(op);
  FailureOr<TilingResult> tiled =
      tilingOp.getTiledImplementation(b, offsets, sizes);
  if (failed(tiled) || tiled->tiledOps.size() != 1) {
    return failure();
  }
  Operation *tiledOp = tiled->tiledOps.front();
  // Replacement operands must dominate the tiled op.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(tiledOp);
  for (OpOperand &operand : tiledOp->getOpOperands()) {
    auto slice = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice) {
      continue;
    }
    Value source = slice.getSource();
    auto it = tiles.find(source);
    if (it != tiles.end()) {
      SmallVector<OpFoldResult> rebased;
      for (auto [off, base] :
           llvm::zip_equal(slice.getMixedOffsets(), it->second.offsets)) {
        rebased.push_back(subOfr(b, loc, off, base));
      }
      Value newSlice = tensor::ExtractSliceOp::create(
          b, loc, slice.getType(), it->second.value, rebased,
          slice.getMixedSizes(), slice.getMixedStrides());
      operand.set(newSlice);
      continue;
    }
    if (auto fill = source.getDefiningOp<linalg::FillOp>()) {
      FailureOr<TilingResult> tiledFill =
          cast<TilingInterface>(fill.getOperation())
              .getTiledImplementation(b, slice.getMixedOffsets(),
                                      slice.getMixedSizes());
      if (failed(tiledFill)) {
        return failure();
      }
      operand.set(tiledFill->tiledValues.front());
    }
  }
  if (init) {
    cast<DestinationStyleOpInterface>(tiledOp).getDpsInitsMutable()[0].set(
        init);
  }
  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  if (failed(tilingOp.getResultTilePosition(b, 0, offsets, sizes,
                                            resultOffsets, resultSizes))) {
    return failure();
  }
  Value result = tiledOp->getResult(0);
  tiles[op->getResult(0)] = Tile{result, resultOffsets};
  return result;
}

/// Iteration offsets / sizes of `op` covering rows [m, m + tm) and every
/// other loop in full.
static void getRowTile(OpBuilder &b, Location loc, linalg::LinalgOp op,
                       unsigned rowDim, OpFoldResult m, OpFoldResult tm,
                       SmallVector<OpFoldResult> &offsets,
                       SmallVector<OpFoldResult> &sizes) {
  SmallVector<Range> ranges =
      cast<TilingInterface>(op.getOperation()).getIterationDomain(b);
  offsets.clear();
  sizes.clear();
  for (auto [d, range] : llvm::enumerate(ranges)) {
    offsets.push_back(d == rowDim ? m : range.offset);
    sizes.push_back(d == rowDim ? tm : range.size);
  }
}

/// Offsets / sizes of the result tile of `op` (result 0) for the iteration
/// tile.
static LogicalResult getResultTile(OpBuilder &b, Operation *op,
                                   ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes,
                                   SmallVector<OpFoldResult> &resultOffsets,
                                   SmallVector<OpFoldResult> &resultSizes) {
  return cast<TilingInterface>(op).getResultTilePosition(
      b, 0, offsets, sizes, resultOffsets, resultSizes);
}

/// Creates `scf.for i in [0, n)` carrying `init`, attached at `b`'s insertion
/// point, and builds its body with `body` (which returns the value to yield).
/// The body is built after the loop exists: the affine folding helpers walk
/// up parent regions, which a body-builder callback does not have yet.
static FailureOr<Value> buildLoop(
    OpBuilder &b, Location loc, int64_t n, Value init,
    function_ref<FailureOr<Value>(OpBuilder &, Location, Value, Value)> body) {
  Value lb = arith::ConstantIndexOp::create(b, loc, 0);
  Value ub = arith::ConstantIndexOp::create(b, loc, n);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  auto loop = scf::ForOp::create(
      b, loc, lb, ub, one, ValueRange{init},
      [](OpBuilder &nb, Location nloc, Value, ValueRange iterArgs) {
        scf::YieldOp::create(nb, nloc, iterArgs);
      });
  OpBuilder::InsertionGuard g(b);
  Operation *yield = loop.getBody()->getTerminator();
  b.setInsertionPoint(yield);
  FailureOr<Value> result =
      body(b, loc, loop.getInductionVar(), loop.getRegionIterArgs()[0]);
  if (failed(result)) {
    return failure();
  }
  yield->setOperand(0, *result);
  return loop.getResult(0);
}

/// Builds, at `b`'s insertion point, the loop over the heads of `rope`:
/// per head the projection matmul, its trunc and the RoPE, written into
/// `slab` (the row tile of the RoPE's result, at `slabOffsets`). Returns the
/// final slab.
static FailureOr<Value> buildRopeHeadLoop(OpBuilder &b, Location loc,
                                          const RopeInfo &rope,
                                          linalg::LinalgOp mm,
                                          linalg::GenericOp trunc,
                                          OpFoldResult m, OpFoldResult tm,
                                          Value slab,
                                          ArrayRef<OpFoldResult> slabOffsets,
                                          TileMap &tiles) {
  MLIRContext *ctx = b.getContext();
  SmallVector<int64_t> ranges =
      linalg::GenericOp(rope.op).getStaticLoopRanges();
  return buildLoop(
      b, loc, rope.numHeads, slab,
      [&](OpBuilder &nb, Location nloc, Value h,
          Value iterArg) -> FailureOr<Value> {
        TileMap local = tiles;
        // Head loop indices, row-major over the head dims, and the column
        // offset n0 = sum(idx_i * coeff_i) = h * headWidth.
        SmallVector<OpFoldResult> headIdx;
        int64_t inner = rope.numHeads;
        for (unsigned d : rope.headDims) {
          inner /= ranges[d];
          AffineExpr s0 = getAffineSymbolExpr(0, ctx);
          headIdx.push_back(affine::makeComposedFoldedAffineApply(
              nb, nloc, (s0.floorDiv(inner)) % ranges[d], {OpFoldResult(h)}));
        }
        AffineExpr s0 = getAffineSymbolExpr(0, ctx);
        OpFoldResult n0 = affine::makeComposedFoldedAffineApply(
            nb, nloc, s0 * rope.headWidth, {OpFoldResult(h)});
        OpFoldResult width = nb.getIndexAttr(rope.headWidth);
        OpFoldResult zero = nb.getIndexAttr(0);

        // acc = xn[m] * W[n0 : n0 + width]^T
        SmallVector<Range> mmRanges =
            cast<TilingInterface>(mm.getOperation()).getIterationDomain(nb);
        if (failed(tileOp(nb, mm, {m, n0, zero}, {tm, width, mmRanges[2].size},
                          local)) ||
            (trunc &&
             failed(tileOp(nb, trunc, {m, n0}, {tm, width}, local)))) {
          return failure();
        }

        // RoPE of this head, written into the slab.
        SmallVector<OpFoldResult> offsets, sizes;
        getRowTile(nb, nloc, rope.op, rope.rowDim, m, tm, offsets, sizes);
        for (auto [d, idx] : llvm::zip_equal(rope.headDims, headIdx)) {
          offsets[d] = idx;
          sizes[d] = nb.getIndexAttr(1);
        }
        SmallVector<OpFoldResult> resOffsets, resSizes;
        if (failed(getResultTile(nb, rope.op, offsets, sizes, resOffsets,
                                 resSizes))) {
          return failure();
        }
        SmallVector<OpFoldResult> rel, strides;
        for (auto [off, base] : llvm::zip_equal(resOffsets, slabOffsets)) {
          rel.push_back(subOfr(nb, nloc, off, base));
          strides.push_back(nb.getIndexAttr(1));
        }
        Value dest = tensor::ExtractSliceOp::create(nb, nloc, iterArg, rel,
                                                    resSizes, strides);
        FailureOr<Value> out = tileOp(nb, rope.op, offsets, sizes, local, dest);
        if (failed(out)) {
          return failure();
        }
        Value updated = tensor::InsertSliceOp::create(nb, nloc, *out,
                                                      iterArg, rel,
                                                      resSizes, strides);
        return updated;
      });
}

/// The V loop: chunks of `width` columns of trunc(xn * Wv^T) written into
/// `slab` ([tm, Nv]).
static FailureOr<Value> buildChunkLoop(OpBuilder &b, Location loc,
                                       linalg::LinalgOp mm,
                                       linalg::GenericOp trunc, int64_t width,
                                       OpFoldResult m, OpFoldResult tm,
                                       Value slab, TileMap &tiles) {
  MLIRContext *ctx = b.getContext();
  int64_t n = cast<ShapedType>(trunc->getResult(0).getType()).getDimSize(1);
  return buildLoop(
      b, loc, n / width, slab,
      [&](OpBuilder &nb, Location nloc, Value c,
          Value iterArg) -> FailureOr<Value> {
        TileMap local = tiles;
        AffineExpr s0 = getAffineSymbolExpr(0, ctx);
        OpFoldResult n0 = affine::makeComposedFoldedAffineApply(
            nb, nloc, s0 * width, {OpFoldResult(c)});
        OpFoldResult w = nb.getIndexAttr(width);
        OpFoldResult zero = nb.getIndexAttr(0);
        SmallVector<Range> mmRanges =
            cast<TilingInterface>(mm.getOperation()).getIterationDomain(nb);
        SmallVector<OpFoldResult> rel = {zero, n0};
        SmallVector<OpFoldResult> relSizes = {tm, w};
        SmallVector<OpFoldResult> strides(2, nb.getIndexAttr(1));
        Value dest = tensor::ExtractSliceOp::create(nb, nloc, iterArg, rel,
                                                    relSizes, strides);
        FailureOr<Value> out;
        if (failed(tileOp(nb, mm, {m, n0, zero}, {tm, w, mmRanges[2].size},
                          local)) ||
            failed(out = tileOp(nb, trunc, {m, n0}, {tm, w}, local, dest))) {
          return failure();
        }
        Value updated = tensor::InsertSliceOp::create(
            nb, nloc, *out, iterArg, rel, relSizes, strides);
        return updated;
      });
}

//===----------------------------------------------------------------------===//
// Tile sizes
//===----------------------------------------------------------------------===//

static SmallVector<int64_t> getConfigLevel(Operation *op,
                                           IREE::CPU::TilingLevel level) {
  auto config = getLoweringConfig<IREE::CPU::LoweringConfigAttr>(op);
  if (!config || !config.hasTilingLevel(llvm::to_underlying(level))) {
    return {};
  }
  return config.getStaticTilingLevelSizes(llvm::to_underlying(level), op);
}

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

static int64_t getConfigSize(Operation *op, IREE::CPU::TilingLevel level,
                             unsigned dim) {
  SmallVector<int64_t> sizes = getConfigLevel(op, level);
  if (dim >= sizes.size() || ShapedType::isDynamic(sizes[dim])) {
    return 0;
  }
  return sizes[dim];
}

/// Largest power-of-two row tile, at most `maxRows`, whose per-row stack
/// buffers fit 3/4 of the limit: the [., D] f16 normalized rows, the [.] f32
/// sum of squares and one head's [., width] f32 accumulator and f16 trunc.
static int64_t pickRowTile(FunctionOpInterface funcOp, int64_t d,
                           int64_t width, int64_t maxRows) {
  int64_t limit = kDefaultStackLimitBytes;
  if (auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp)) {
    if (DictionaryAttr config = target.getConfiguration()) {
      if (auto attr = config.getAs<IntegerAttr>("max_stack_allocation_size")) {
        limit = attr.getInt();
      }
    }
  }
  int64_t bytesPerRow = d * 2 + 4 + width * (4 + 2);
  int64_t rows = llvm::PowerOf2Ceil(maxRows);
  while (rows > 1 && rows * bytesPerRow > limit / 4 * 3) {
    rows /= 2;
  }
  return rows;
}

static LogicalResult tileAlone(RewriterBase &rewriter, TilingInterface op,
                               ArrayRef<int64_t> tileSizes) {
  scf::SCFTilingOptions options;
  options.setTileSizes(
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes));
  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTilingResult> tiled =
      scf::tileUsingSCF(rewriter, op, options);
  if (failed(tiled)) {
    return failure();
  }
  rewriter.replaceOp(op, tiled->replacements);
  return success();
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteQKVChain(RewriterBase &rewriter,
                                     FunctionOpInterface funcOp,
                                     QKVChain &c) {
  MLIRContext *ctx = rewriter.getContext();
  Location loc = c.ropeQ.op.getLoc();

  int64_t d = cast<ShapedType>(c.norm->getResult(0).getType()).getDimSize(1);
  if (ShapedType::isDynamic(d)) {
    return failure();
  }
  SmallVector<int64_t> rootVec =
      getVectorTileSizes(c.mmQ, {kVecM, kVecN, kVecK});
  int64_t maxRows = getConfigSize(
      c.mmQ, IREE::CPU::TilingLevel::DistributionTiles, /*dim=*/0);
  if (maxRows <= 0) {
    maxRows = kVecM;
  }
  int64_t width = std::max(c.ropeQ.headWidth, c.ropeK.headWidth);
  int64_t rowTile = pickRowTile(funcOp, d, width, maxRows);
  LLVM_DEBUG(llvm::dbgs() << "[qkv] row tile " << rowTile << ", heads "
                          << c.ropeQ.numHeads << "/" << c.ropeK.numHeads
                          << " x " << c.ropeQ.headWidth << "\n");

  // The forall goes after the last chain op, before the stores.
  SmallVector<Operation *> chainOps = {c.sumsq, c.norm,     c.mmQ,
                                       c.mmK,   c.mmV,      c.truncV,
                                       c.ropeK.op, c.ropeQ.op};
  for (Operation *trunc : {c.truncQ, c.truncK}) {
    if (trunc) {
      chainOps.push_back(trunc);
    }
  }
  Operation *last = chainOps.front();
  for (Operation *op : chainOps) {
    if (op->getBlock() != last->getBlock()) {
      return failure();
    }
    if (last->isBeforeInBlock(op)) {
      last = op;
    }
  }
  rewriter.setInsertionPointAfter(last);

  Value x = c.sumsq.getDpsInputs()[0];
  OpFoldResult rows = tensor::getMixedSize(rewriter, loc, x, 0);
  SmallVector<Value> inits = {c.ropeQ.op.getDpsInits()[0],
                              c.ropeK.op.getDpsInits()[0],
                              c.truncV.getDpsInits()[0]};
  auto forallOp = scf::ForallOp::create(
      rewriter, loc, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{rows},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(rowTile)}, inits,
      rewriter.getArrayAttr({IREE::Codegen::WorkgroupMappingAttr::get(
          ctx, IREE::Codegen::WorkgroupId::IdX)}));

  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forallOp.getTerminator());
    Value m = forallOp.getInductionVar(0);
    AffineExpr d0, s0;
    bindDims(ctx, d0);
    bindSymbols(ctx, s0);
    OpFoldResult tm = affine::makeComposedFoldedAffineMin(
        rewriter, loc,
        AffineMap::get(1, 1, {getAffineConstantExpr(rowTile, ctx), s0 - d0},
                       ctx),
        {OpFoldResult(m), rows});
    OpFoldResult mOfr = m;

    TileMap tiles;
    SmallVector<OpFoldResult> offsets, sizes;
    // Norm, once per row tile.
    getRowTile(rewriter, loc, c.sumsq, 0, mOfr, tm, offsets, sizes);
    if (failed(tileOp(rewriter, c.sumsq, offsets, sizes, tiles))) {
      return failure();
    }
    getRowTile(rewriter, loc, c.norm, 0, mOfr, tm, offsets, sizes);
    if (failed(tileOp(rewriter, c.norm, offsets, sizes, tiles))) {
      return failure();
    }

    // Row slabs of the three results.
    ValueRange outs = forallOp.getRegionIterArgs();
    SmallVector<SmallVector<OpFoldResult>> slabOffsets(3), slabSizes(3);
    SmallVector<Value> slabs;
    Operation *resultOps[3] = {c.ropeQ.op, c.ropeK.op, c.truncV};
    unsigned rowDims[3] = {c.ropeQ.rowDim, c.ropeK.rowDim, 0};
    for (unsigned i = 0; i < 3; ++i) {
      getRowTile(rewriter, loc, cast<linalg::LinalgOp>(resultOps[i]),
                 rowDims[i], mOfr, tm, offsets, sizes);
      if (failed(getResultTile(rewriter, resultOps[i], offsets, sizes,
                               slabOffsets[i], slabSizes[i]))) {
        return failure();
      }
      SmallVector<OpFoldResult> strides(slabOffsets[i].size(),
                                        rewriter.getIndexAttr(1));
      slabs.push_back(tensor::ExtractSliceOp::create(
          rewriter, loc, outs[i], slabOffsets[i], slabSizes[i], strides));
    }

    FailureOr<Value> q =
        buildRopeHeadLoop(rewriter, loc, c.ropeQ, c.mmQ, c.truncQ, mOfr, tm,
                          slabs[0], slabOffsets[0], tiles);
    FailureOr<Value> k =
        buildRopeHeadLoop(rewriter, loc, c.ropeK, c.mmK, c.truncK, mOfr, tm,
                          slabs[1], slabOffsets[1], tiles);
    FailureOr<Value> v =
        buildChunkLoop(rewriter, loc, c.mmV, c.truncV, c.ropeK.headWidth,
                       mOfr, tm, slabs[2], tiles);
    if (failed(q) || failed(k) || failed(v)) {
      return failure();
    }

    rewriter.setInsertionPointToStart(forallOp.getTerminator().getBody());
    Value results[3] = {*q, *k, *v};
    for (unsigned i = 0; i < 3; ++i) {
      SmallVector<OpFoldResult> strides(slabOffsets[i].size(),
                                        rewriter.getIndexAttr(1));
      tensor::ParallelInsertSliceOp::create(rewriter, loc, results[i], outs[i],
                                            slabOffsets[i], slabSizes[i],
                                            strides);
    }
  }

  rewriter.replaceAllUsesWith(c.ropeQ.op->getResult(0), forallOp.getResult(0));
  rewriter.replaceAllUsesWith(c.ropeK.op->getResult(0), forallOp.getResult(1));
  rewriter.replaceAllUsesExcept(c.truncV->getResult(0), forallOp.getResult(2),
                                forallOp);

  // Vector tiles: every op in the forall on its own, config dropped.
  SmallVector<linalg::LinalgOp> ops;
  forallOp.walk([&](linalg::LinalgOp op) {
    if (!op->use_empty()) {
      ops.push_back(op);
    }
  });
  for (linalg::LinalgOp op : ops) {
    SmallVector<int64_t> tileSizes;
    if (isa<linalg::FillOp>(op) && op.getNumLoops() == 2) {
      tileSizes = {rootVec[0], rootVec[1]};
    } else if (op.getNumLoops() == 3 && linalg::isaContractionOpInterface(op)) {
      tileSizes = getVectorTileSizes(op, rootVec);
    } else if (op.getNumLoops() == 2) {
      tileSizes = getVectorTileSizes(op, {rootVec[0], rootVec[1]});
    }
    op->removeAttr("lowering_config");
    op->removeAttr(kSpecialQKVAttr);
    if (tileSizes.empty()) {
      continue;
    }
    if (failed(tileAlone(rewriter, cast<TilingInterface>(op.getOperation()),
                         tileSizes))) {
      return failure();
    }
  }
  return success();
}

struct RewriteQKVChainAsForallPass final
    : impl::RewriteQKVChainAsForallPassBase<RewriteQKVChainAsForallPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp->hasAttr(kQKVChainRewrittenAttr)) {
      return;
    }
    Operation *markedRoot = nullptr;
    funcOp.walk([&](Operation *op) {
      if (op->hasAttr(kSpecialQKVAttr)) {
        markedRoot = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!markedRoot) {
      return;
    }
    FailureOr<QKVChain> chain = matchQKVChain(markedRoot);
    if (failed(chain)) {
      LLVM_DEBUG(llvm::dbgs() << "[qkv] tagged root does not match\n");
      return;
    }
    IRRewriter rewriter(&getContext());
    if (failed(rewriteQKVChain(rewriter, funcOp, *chain))) {
      funcOp.emitError("RewriteQKVChainAsForall: rewrite failed");
      return signalPassFailure();
    }
    funcOp->setAttr(kQKVChainRewrittenAttr, UnitAttr::get(&getContext()));
  }
};

} // namespace
} // namespace mlir::iree_compiler
