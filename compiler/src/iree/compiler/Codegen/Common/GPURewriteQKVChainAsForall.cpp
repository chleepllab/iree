// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// GPURewriteQKVChainAsForall
//
// GPU (LLVMGPU TileAndFuse) variant of RewriteQKVChainAsForall. Same chain,
// formed under --iree-dispatch-creation-fuse-qkv:
//
//   ss    = sum(x^2)                        [S] f32
//   xn    = rmsnorm(x, ss) * w              [S, D] f16
//   q/k/v = trunc(xn * Wq/Wk/Wv^T)          [S, Nq/Nk/Nv] f16
//   qr    = rope(q[s, h*64 + 2j], q[s, h*64 + 2j + 1], cos, sin)   (result)
//   kr    = rope(k[...], ...)                                        (result)
//   v                                                                 (result)
//
// and the same schedule:
//
//   scf.forall m in [0, S) step TM                         (workgroups)
//     ss = sum(x[m]^2) ; xn = rmsnorm(x[m], ss) * w        [TM, D] f16
//     scf.for h in heads(Q):                               (32 for llama 1B)
//       acc = xn * Wq[h*64 : h*64+64]^T                    [TM, 64] f32
//       qr[.., h, m, ..] = rope(trunc(acc), cos[m], sin[m])
//     scf.for h in heads(K):  (same, 8 heads)
//     scf.for c in chunks(V): v[m, c*64 : c*64+64] = trunc(xn * Wv[c]^T)
//
// What differs from the CPU pass:
//
// - On GPU the dispatch has S split into [D, 32] (BlockDynamicDimensions)
//   for every op but the two RoPEs, whose strided reads it cannot expand:
//   they read `collapse_shape` of the [D, 32, N] projections. The RoPEs are
//   first rewritten over [D, 32] rows too (expandRopeRows), so the whole
//   chain has the same two row dims and the row tile is static: TM = 16
//   (one MMA intrinsic) of the inner 32, i.e. S / 16 workgroups.
//
// - Instead of tiling every op to vectors, each op gets a lowering config for
//   the TileAndFuse levels, as in GPURewriteMLPChainAsForall's MMA path: the
//   projections are packed to mma.sync, kHeadsPerStep (4) heads -- [16, 256]
//   -- per step of the head loops, each of the 8 warps owning 4 16x8 tiles,
//   K in steps of 64; the truncs and the RoPEs get thread configs; the sum of
//   squares is distributed to threads here (one thread per row, K in serial
//   steps of 16). The norm has no config, so the reduction level fuses it
//   into each projection's K loop (kFuseNormIntoK): xn is recomputed per
//   [16, 64] K slice instead of living in shared memory ([16, D] f16 = 64 KiB),
//   which is what makes 4 heads per step fit.
//
// GX10, S = 2048, 16 layers: this kernel takes 68.0 ms against ~18 ms for the
// unfused kernels it replaces (Q / K / V matmuls, RMSNorm, RoPEs, even/odd
// copies), mostly because every 16-row workgroup streams all of Wq/Wk/Wv
// from global memory (1 head per step: 113 ms; 2 heads: 76.5 ms; xn in
// shared memory with 4 heads: 131 KiB, does not fit).
//
// The function is tagged `iree_codegen.qkv_chain_rewritten` so
// TileAndDistributeToWorkgroupsUsingForall leaves it alone.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

#define DEBUG_TYPE "gpu-rewrite-qkv-chain-as-forall"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUREWRITEQKVCHAINASFORALLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static constexpr StringLiteral kQKVChainRewrittenAttr =
    "iree_codegen.qkv_chain_rewritten";
// Rows per workgroup: one mma.sync M tile. (32 rows with 2 heads per step:
// QKV kernel 77.6 ms vs 68.0 ms at S = 2048 on the GX10; 32 rows with 4
// heads needs 128 KiB of shared memory.)
static constexpr int64_t kRowTile = 16;
static constexpr int64_t kWorkgroupThreads = 256;
// K step of the projections, in elements (4 x k16 intrinsics).
static constexpr int64_t kReductionK = 64;
// Serial K step of the sum of squares inside each thread.
static constexpr int64_t kVecK = 16;
// Q/K heads (and V columns / head width) per step of the head loops: more
// MMA tiles per warp per x fragment.
static constexpr int64_t kHeadsPerStep = 4;
// Fuse the norm (xn = x * rsqrt(ss) * w) into each projection's K loop
// instead of materializing the [16, D] f16 xn tile in shared memory (64 KiB
// for D = 2048): a few multiplies per element, against an MMA per 16x8x16.
static constexpr bool kFuseNormIntoK = true;

//===----------------------------------------------------------------------===//
// RoPE row expansion
//===----------------------------------------------------------------------===//

/// Rewrites `op`, an all-parallel generic that reads `collapse_shape(t)` of a
/// [A, B, N] value `t` at `(.., d_r, ..)`, over (d_r -> dA, dB): every operand
/// indexed by d_r is read through its [A, B] expansion (the collapse's source
/// directly), the result is produced expanded and collapsed back. linalg.index
/// of d_r becomes dA * B + dB. Returns the new op, or failure if `op` does not
/// have that form (d_r used other than as a plain map result).
static FailureOr<linalg::GenericOp> expandRopeRows(RewriterBase &rewriter,
                                                   linalg::GenericOp op) {
  if (op.getNumLoops() != op.getNumParallelLoops() ||
      op.getNumDpsInits() != 1) {
    return failure();
  }
  tensor::CollapseShapeOp collapse;
  unsigned rowDim = 0;
  for (OpOperand *in : op.getDpsInputOperands()) {
    auto c = in->get().getDefiningOp<tensor::CollapseShapeOp>();
    if (!c || c.getSrcType().getRank() != 3) {
      continue;
    }
    SmallVector<ReassociationIndices> reass = c.getReassociationIndices();
    if (reass.size() != 2 || reass[0].size() != 2 || reass[1].size() != 1) {
      continue;
    }
    auto d = dyn_cast<AffineDimExpr>(op.getMatchingIndexingMap(in).getResult(0));
    if (!d) {
      continue;
    }
    collapse = c;
    rowDim = d.getPosition();
    break;
  }
  if (!collapse) {
    return failure();
  }
  int64_t blockB = collapse.getSrcType().getDimSize(1);
  if (ShapedType::isDynamic(blockB)) {
    return failure();
  }
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  OpFoldResult blockA =
      tensor::getMixedSize(rewriter, loc, collapse.getSrc(), 0);

  unsigned numLoops = op.getNumLoops();
  SmallVector<AffineExpr> dimRepl;
  for (unsigned d = 0; d < numLoops; ++d) {
    dimRepl.push_back(getAffineDimExpr(d <= rowDim ? d : d + 1, ctx));
  }
  AffineExpr dA = getAffineDimExpr(rowDim, ctx);
  AffineExpr dB = getAffineDimExpr(rowDim + 1, ctx);

  // Expands `v` at dim `pos` into [blockA, blockB].
  auto expandValue = [&](Value v, int64_t pos) -> Value {
    auto type = cast<RankedTensorType>(v.getType());
    if (auto c = v.getDefiningOp<tensor::CollapseShapeOp>()) {
      SmallVector<ReassociationIndices> reass = c.getReassociationIndices();
      if (c.getSrcType().getRank() == type.getRank() + 1 &&
          reass[pos].size() == 2 && c.getSrcType().getDimSize(pos + 1) ==
                                        blockB) {
        bool rest = true;
        for (auto [i, r] : llvm::enumerate(reass)) {
          rest &= (int64_t)i == pos || r.size() == 1;
        }
        if (rest) {
          return c.getSrc();
        }
      }
    }
    SmallVector<ReassociationIndices> reass;
    SmallVector<OpFoldResult> shape;
    int64_t next = 0;
    for (int64_t i = 0; i < type.getRank(); ++i) {
      if (i == pos) {
        reass.push_back({next, next + 1});
        next += 2;
        shape.push_back(blockA);
        shape.push_back(rewriter.getIndexAttr(blockB));
        continue;
      }
      reass.push_back({next++});
      shape.push_back(tensor::getMixedSize(rewriter, loc, v, i));
    }
    SmallVector<int64_t> staticShape;
    SmallVector<Value> dynamic;
    dispatchIndexOpFoldResults(shape, dynamic, staticShape);
    auto newType = RankedTensorType::get(staticShape, type.getElementType());
    return tensor::ExpandShapeOp::create(rewriter, loc, newType, v, reass,
                                         shape);
  };

  SmallVector<AffineMap> newMaps;
  SmallVector<Value> newOperands;
  int64_t outRowPos = -1;
  for (OpOperand &operand : op->getOpOperands()) {
    AffineMap map = op.getMatchingIndexingMap(&operand);
    SmallVector<AffineExpr> results;
    int64_t rowPos = -1;
    for (auto [i, e] : llvm::enumerate(map.getResults())) {
      auto d = dyn_cast<AffineDimExpr>(e);
      if (d && d.getPosition() == rowDim) {
        rowPos = i;
        results.push_back(dA);
        results.push_back(dB);
        continue;
      }
      if (e.isFunctionOfDim(rowDim)) {
        return failure();
      }
      results.push_back(e.replaceDimsAndSymbols(dimRepl, {}));
    }
    newMaps.push_back(AffineMap::get(numLoops + 1, 0, results, ctx));
    Value v = operand.get();
    if (rowPos >= 0) {
      v = expandValue(v, rowPos);
    }
    if (op.isDpsInit(&operand)) {
      if (rowPos < 0) {
        return failure();
      }
      outRowPos = rowPos;
    }
    newOperands.push_back(v);
  }
  SmallVector<utils::IteratorType> iterators(numLoops + 1,
                                             utils::IteratorType::parallel);
  ValueRange ins = ValueRange(newOperands).drop_back();
  Value init = newOperands.back();
  auto newOp = linalg::GenericOp::create(rewriter, loc, init.getType(), ins,
                                         ValueRange{init}, newMaps, iterators);
  rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                             newOp.getRegion().end());
  SmallVector<linalg::IndexOp> indices;
  newOp.getRegion().walk([&](linalg::IndexOp idx) { indices.push_back(idx); });
  for (linalg::IndexOp idx : indices) {
    uint64_t d = idx.getDim();
    rewriter.setInsertionPoint(idx);
    if (d < rowDim) {
      continue;
    }
    if (d > rowDim) {
      rewriter.replaceOpWithNewOp<linalg::IndexOp>(idx, d + 1);
      continue;
    }
    Value a = linalg::IndexOp::create(rewriter, idx.getLoc(), rowDim);
    Value b = linalg::IndexOp::create(rewriter, idx.getLoc(), rowDim + 1);
    Value c = arith::ConstantIndexOp::create(rewriter, idx.getLoc(), blockB);
    Value row = arith::AddIOp::create(
        rewriter, idx.getLoc(),
        arith::MulIOp::create(rewriter, idx.getLoc(), a, c), b);
    rewriter.replaceOp(idx, row);
  }
  for (NamedAttribute attr : op->getDiscardableAttrs()) {
    newOp->setAttr(attr.getName(), attr.getValue());
  }
  // Collapse back for the original users.
  rewriter.setInsertionPointAfter(newOp);
  SmallVector<ReassociationIndices> reass;
  auto outType = cast<RankedTensorType>(op->getResult(0).getType());
  int64_t next = 0;
  for (int64_t i = 0; i < outType.getRank(); ++i) {
    if (i == outRowPos) {
      reass.push_back({next, next + 1});
      next += 2;
    } else {
      reass.push_back({next++});
    }
  }
  Value collapsed = tensor::CollapseShapeOp::create(
      rewriter, loc, outType, newOp->getResult(0), reass);
  rewriter.replaceOp(op, collapsed);
  return newOp;
}

//===----------------------------------------------------------------------===//
// Matching (the chain over two row dims [A, B])
//===----------------------------------------------------------------------===//

/// `ins(A[m0, m1, K], B[N, K]) outs(C[m0, m1, N])`.
static bool isProjection(Operation *op) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp) ||
      linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1 ||
      linalgOp.getNumLoops() != 4) {
    return false;
  }
  MLIRContext *ctx = op->getContext();
  AffineExpr m0, m1, n, k;
  bindDims(ctx, m0, m1, n, k);
  SmallVector<AffineMap> expected = {AffineMap::get(4, 0, {m0, m1, k}, ctx),
                                     AffineMap::get(4, 0, {n, k}, ctx),
                                     AffineMap::get(4, 0, {m0, m1, n}, ctx)};
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

/// A RoPE op reading its projection `t[rowA, rowB, col]` with
/// `col = sum(coeff_i * d_i) + c`; `jDim` is the loop with the smallest
/// coefficient (the pair index), every other loop in `col` a head loop.
struct RopeInfo {
  linalg::GenericOp op;
  unsigned rowA = 0, rowB = 0;
  unsigned jDim = 0;
  SmallVector<unsigned> headDims;   // by decreasing coefficient
  SmallVector<int64_t> headCoeffs;
  int64_t numHeads = 1;
  int64_t headWidth = 0;
};

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
  if (!readMap || readMap.getNumResults() != 3) {
    return failure();
  }
  auto rowA = dyn_cast<AffineDimExpr>(readMap.getResult(0));
  auto rowB = dyn_cast<AffineDimExpr>(readMap.getResult(1));
  if (!rowA || !rowB) {
    return failure();
  }
  info.rowA = rowA.getPosition();
  info.rowB = rowB.getPosition();
  std::optional<SmallVector<int64_t>> coeffs =
      getLinearCoeffs(readMap.getResult(2), numDims);
  if (!coeffs) {
    return failure();
  }
  SmallVector<int64_t> ranges = rope.getStaticLoopRanges();
  SmallVector<unsigned> colDims;
  for (unsigned d = 0; d < numDims; ++d) {
    if ((*coeffs)[d] != 0) {
      if (d == info.rowA || d == info.rowB ||
          ShapedType::isDynamic(ranges[d])) {
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
  int64_t n = cast<ShapedType>(t.getType()).getDimSize(2);
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
  linalg::GenericOp truncQ, truncK, truncV;  // truncQ/K null if in the RoPE
  linalg::LinalgOp mmQ, mmK, mmV;
  linalg::GenericOp norm;
  linalg::GenericOp sumsq;
};

static Value getRopeProjection(linalg::GenericOp rope, Operation *&mm) {
  for (Value input : rope.getDpsInputs()) {
    if (isProjection(input.getDefiningOp())) {
      mm = input.getDefiningOp();
      return input;
    }
    auto trunc = input.getDefiningOp<linalg::GenericOp>();
    if (!trunc || trunc.getNumDpsInputs() != 1) {
      continue;
    }
    Operation *producer = trunc.getDpsInputs()[0].getDefiningOp();
    if (isProjection(producer) && isTruncOf(trunc, producer)) {
      mm = producer;
      return input;
    }
  }
  return Value();
}

/// The chain whose Q RoPE is `root` (the RoPE with the most heads).
static FailureOr<QKVChain> matchQKVChain(linalg::GenericOp ropeQ) {
  QKVChain chain;
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
      chain.norm.getNumLoops() != 3) {
    return failure();
  }
  for (Value input : chain.norm.getDpsInputs()) {
    auto generic = input.getDefiningOp<linalg::GenericOp>();
    if (generic && generic.getNumReductionLoops() == 1 &&
        generic.getNumLoops() == 3 && generic.getNumDpsInits() == 1) {
      chain.sumsq = generic;
    }
  }
  if (!chain.sumsq) {
    return failure();
  }

  SmallVector<Operation *> matmuls = getNonDimUsers(xn);
  if (matmuls.size() != 3 || !llvm::all_of(matmuls, isProjection)) {
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
    Operation *ropeMM = nullptr;
    auto direct = dyn_cast<linalg::GenericOp>(users[0]);
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
  int64_t nv = cast<ShapedType>(chain.truncV->getResult(0).getType())
                   .getDimSize(2);
  if (ShapedType::isDynamic(nv) || nv % chain.ropeK.headWidth != 0) {
    return failure();
  }
  return chain;
}

//===----------------------------------------------------------------------===//
// Tiling helpers (as in RewriteQKVChainAsForall, two row dims)
//===----------------------------------------------------------------------===//

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

/// Tiles `op` at iteration `offsets` / `sizes`, re-pointing each operand slice
/// of a value in `tiles` at that value's tile (operand slices of a fill get a
/// tiled fill). `init` replaces the tiled op's init. Returns the tiled op.
static FailureOr<Operation *> tileOp(OpBuilder &b, Operation *op,
                                     ArrayRef<OpFoldResult> offsets,
                                     ArrayRef<OpFoldResult> sizes,
                                     TileMap &tiles, Value init = Value()) {
  Location loc = op->getLoc();
  auto tilingOp = cast<TilingInterface>(op);
  FailureOr<TilingResult> tiled =
      tilingOp.getTiledImplementation(b, offsets, sizes);
  if (failed(tiled) || tiled->tiledOps.size() != 1) {
    return failure();
  }
  Operation *tiledOp = tiled->tiledOps.front();
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
  tiles[op->getResult(0)] = Tile{tiledOp->getResult(0), resultOffsets};
  return tiledOp;
}

/// Iteration offsets / sizes of `op` covering rows [mA, mA + 1) x
/// [mB, mB + kRowTile) of its row dims and every other loop in full.
static void getRowTile(OpBuilder &b, linalg::LinalgOp op, unsigned rowA,
                       unsigned rowB, OpFoldResult mA, OpFoldResult mB,
                       SmallVector<OpFoldResult> &offsets,
                       SmallVector<OpFoldResult> &sizes) {
  SmallVector<Range> ranges =
      cast<TilingInterface>(op.getOperation()).getIterationDomain(b);
  offsets.clear();
  sizes.clear();
  for (auto [d, range] : llvm::enumerate(ranges)) {
    if (d == rowA) {
      offsets.push_back(mA);
      sizes.push_back(b.getIndexAttr(1));
    } else if (d == rowB) {
      offsets.push_back(mB);
      sizes.push_back(b.getIndexAttr(kRowTile));
    } else {
      offsets.push_back(range.offset);
      sizes.push_back(range.size);
    }
  }
}

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

/// The ops created per row tile, for the configs.
struct TiledOps {
  SmallVector<Operation *> projections;  // per head / chunk
  SmallVector<Operation *> elementwise;  // truncs (thread config [.., 4])
  SmallVector<std::pair<Operation *, unsigned>> ropes;  // with jDim
};

/// Per head of `rope`: the projection, its trunc and the RoPE, written into
/// `slab` (the row tile of the RoPE's result, at `slabOffsets`).
static FailureOr<Value> buildRopeHeadLoop(OpBuilder &b, Location loc,
                                          const RopeInfo &rope,
                                          linalg::LinalgOp mm,
                                          linalg::GenericOp trunc,
                                          OpFoldResult mA, OpFoldResult mB,
                                          Value slab,
                                          ArrayRef<OpFoldResult> slabOffsets,
                                          TileMap &tiles, TiledOps &made) {
  MLIRContext *ctx = b.getContext();
  SmallVector<int64_t> ranges =
      linalg::GenericOp(rope.op).getStaticLoopRanges();
  // Heads per step: kHeadsPerStep consecutive heads of the innermost head dim.
  int64_t hg = kHeadsPerStep;
  if (ranges[rope.headDims.back()] % hg != 0) {
    hg = 1;
  }
  return buildLoop(
      b, loc, rope.numHeads / hg, slab,
      [&](OpBuilder &nb, Location nloc, Value g,
          Value iterArg) -> FailureOr<Value> {
        TileMap local = tiles;
        // First head of the group: h = g * hg.
        AffineExpr s0 = getAffineSymbolExpr(0, ctx);
        SmallVector<OpFoldResult> headIdx;
        int64_t inner = rope.numHeads;
        for (unsigned d : rope.headDims) {
          inner /= ranges[d];
          headIdx.push_back(affine::makeComposedFoldedAffineApply(
              nb, nloc, ((s0 * hg).floorDiv(inner)) % ranges[d],
              {OpFoldResult(g)}));
        }
        OpFoldResult n0 = affine::makeComposedFoldedAffineApply(
            nb, nloc, s0 * (hg * rope.headWidth), {OpFoldResult(g)});
        OpFoldResult width = nb.getIndexAttr(hg * rope.headWidth);
        OpFoldResult zero = nb.getIndexAttr(0);
        OpFoldResult one = nb.getIndexAttr(1);
        OpFoldResult tm = nb.getIndexAttr(kRowTile);

        SmallVector<Range> mmRanges =
            cast<TilingInterface>(mm.getOperation()).getIterationDomain(nb);
        FailureOr<Operation *> mmT = tileOp(
            nb, mm, {mA, mB, n0, zero}, {one, tm, width, mmRanges[3].size},
            local);
        if (failed(mmT)) {
          return failure();
        }
        made.projections.push_back(*mmT);
        if (trunc) {
          FailureOr<Operation *> tr =
              tileOp(nb, trunc, {mA, mB, n0}, {one, tm, width}, local);
          if (failed(tr)) {
            return failure();
          }
          made.elementwise.push_back(*tr);
        }

        SmallVector<OpFoldResult> offsets, sizes;
        getRowTile(nb, rope.op, rope.rowA, rope.rowB, mA, mB, offsets, sizes);
        for (auto [d, idx] : llvm::zip_equal(rope.headDims, headIdx)) {
          offsets[d] = idx;
          sizes[d] = nb.getIndexAttr(d == rope.headDims.back() ? hg : 1);
        }
        SmallVector<OpFoldResult> resOffsets, resSizes;
        if (failed(cast<TilingInterface>(static_cast<Operation *>(rope.op))
                       .getResultTilePosition(nb, 0, offsets, sizes,
                                              resOffsets, resSizes))) {
          return failure();
        }
        SmallVector<OpFoldResult> rel, strides;
        for (auto [off, base] : llvm::zip_equal(resOffsets, slabOffsets)) {
          rel.push_back(subOfr(nb, nloc, off, base));
          strides.push_back(nb.getIndexAttr(1));
        }
        Value dest = tensor::ExtractSliceOp::create(nb, nloc, iterArg, rel,
                                                    resSizes, strides);
        FailureOr<Operation *> out =
            tileOp(nb, rope.op, offsets, sizes, local, dest);
        if (failed(out)) {
          return failure();
        }
        made.ropes.push_back({*out, rope.jDim});
        return Value(tensor::InsertSliceOp::create(
            nb, nloc, (*out)->getResult(0), iterArg, rel, resSizes, strides));
      });
}

/// The V loop: chunks of `width` columns of trunc(xn * Wv^T) into `slab`.
static FailureOr<Value> buildChunkLoop(OpBuilder &b, Location loc,
                                       linalg::LinalgOp mm,
                                       linalg::GenericOp trunc, int64_t width,
                                       OpFoldResult mA, OpFoldResult mB,
                                       Value slab, TileMap &tiles,
                                       TiledOps &made) {
  MLIRContext *ctx = b.getContext();
  int64_t n = cast<ShapedType>(trunc->getResult(0).getType()).getDimSize(2);
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
        OpFoldResult one = nb.getIndexAttr(1);
        OpFoldResult tm = nb.getIndexAttr(kRowTile);
        SmallVector<Range> mmRanges =
            cast<TilingInterface>(mm.getOperation()).getIterationDomain(nb);
        SmallVector<OpFoldResult> rel = {zero, zero, n0};
        SmallVector<OpFoldResult> relSizes = {one, tm, w};
        SmallVector<OpFoldResult> strides(3, nb.getIndexAttr(1));
        Value dest = tensor::ExtractSliceOp::create(nb, nloc, iterArg, rel,
                                                    relSizes, strides);
        FailureOr<Operation *> mmT = tileOp(
            nb, mm, {mA, mB, n0, zero}, {one, tm, w, mmRanges[3].size}, local);
        if (failed(mmT)) {
          return failure();
        }
        made.projections.push_back(*mmT);
        FailureOr<Operation *> out =
            tileOp(nb, trunc, {mA, mB, n0}, {one, tm, w}, local, dest);
        if (failed(out)) {
          return failure();
        }
        made.elementwise.push_back(*out);
        return Value(tensor::InsertSliceOp::create(
            nb, nloc, (*out)->getResult(0), iterArg, rel, relSizes, strides));
      });
}

//===----------------------------------------------------------------------===//
// Configs
//===----------------------------------------------------------------------===//

static IREE::GPU::LoweringConfigAttr getThreadConfig(MLIRContext *ctx,
                                                     ArrayRef<int64_t> tile) {
  Builder b(ctx);
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back("thread", b.getI64ArrayAttr(tile));
  return IREE::GPU::LoweringConfigAttr::get(ctx,
                                            DictionaryAttr::get(ctx, attrs));
}

/// Projection [1, TM, width, K]: packed to `kind`, each subgroup one intrinsic
/// along M and `subgroupN` along N, K in steps of `reductionK` intrinsics.
static IREE::GPU::LoweringConfigAttr
getProjectionConfig(MLIRContext *ctx, IREE::GPU::MMAAttr kind,
                    int64_t subgroupN, int64_t reductionK) {
  Builder b(ctx);
  SmallVector<NamedAttribute> attrs;
  IREE::GPU::setMmaKind(ctx, attrs, kind);
  attrs.emplace_back("subgroup", b.getI64ArrayAttr({0, 1, subgroupN, 0}));
  attrs.emplace_back("reduction", b.getI64ArrayAttr({0, 0, 0, reductionK}));
  return IREE::GPU::LoweringConfigAttr::get(ctx,
                                            DictionaryAttr::get(ctx, attrs));
}

/// One thread per row of the [1, TM, D] sum of squares, K in serial steps of
/// kVecK (its destination becomes shared memory).
static LogicalResult distributeSumSq(RewriterBase &rewriter,
                                     linalg::LinalgOp op) {
  MLIRContext *ctx = rewriter.getContext();
  scf::SCFTilingOptions threadOptions;
  threadOptions.setTileSizes(getAsIndexOpFoldResult(ctx, {0, 1, 0}));
  threadOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  threadOptions.setMapping(
      {gpu::GPUThreadMappingAttr::get(ctx, gpu::MappingId::LinearDim0)});
  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFTilingResult> threadTiled = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), threadOptions);
  if (failed(threadTiled) || threadTiled->tiledOps.size() != 1) {
    return failure();
  }
  rewriter.replaceOp(op, threadTiled->replacements);
  auto perThread = cast<TilingInterface>(threadTiled->tiledOps.front());
  scf::SCFTilingOptions reductionOptions;
  reductionOptions.setTileSizes(getAsIndexOpFoldResult(ctx, {0, 0, kVecK}));
  rewriter.setInsertionPoint(perThread);
  FailureOr<scf::SCFTilingResult> reductionTiled =
      scf::tileUsingSCF(rewriter, perThread, reductionOptions);
  if (failed(reductionTiled)) {
    return failure();
  }
  rewriter.replaceOp(perThread, reductionTiled->replacements);
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

  Value x = c.sumsq.getDpsInputs()[0];
  auto xType = cast<ShapedType>(x.getType());
  int64_t d = xType.getDimSize(2);
  int64_t inner = xType.getDimSize(1);
  if (ShapedType::isDynamic(d) || ShapedType::isDynamic(inner) ||
      inner % kRowTile != 0 || d % kReductionK != 0) {
    LLVM_DEBUG(llvm::dbgs() << "[qkv-gpu] unsupported shape " << xType << "\n");
    return failure();
  }
  int64_t width = c.ropeK.headWidth;
  if (c.ropeQ.headWidth != width) {
    return failure();
  }

  IREE::Codegen::TranslationInfoAttr info = getTranslationInfo(funcOp);
  if (!info) {
    return failure();
  }
  int64_t numThreads = kWorkgroupThreads;
  int64_t subgroupSize = info.getSubgroupSize() ? info.getSubgroupSize() : 32;
  int64_t numSubgroups = numThreads / subgroupSize;
  auto kind = IREE::GPU::MMAAttr::get(
      ctx, IREE::GPU::MMAIntrinsic::NV_MMA_SYNC_F32_16x8x16_F16);
  auto [intrinsicM, intrinsicN, intrinsicK] = kind.getMNKShape();
  // Columns per head-loop step; the head-group size must divide every head
  // loop (llama 1B: Q [8, 4] heads, K 8 heads, V 512 columns).
  int64_t stepWidth = kHeadsPerStep * width;
  auto innerHeads = [](const RopeInfo &r) {
    return linalg::GenericOp(r.op).getStaticLoopRanges()[r.headDims.back()];
  };
  int64_t nv = cast<ShapedType>(c.truncV->getResult(0).getType()).getDimSize(2);
  if (innerHeads(c.ropeQ) % kHeadsPerStep != 0 ||
      innerHeads(c.ropeK) % kHeadsPerStep != 0 || nv % stepWidth != 0) {
    return failure();
  }
  // Warps: kRowTile / 16 along M, the rest along N.
  int64_t warpsM = kRowTile / intrinsicM;
  if (kRowTile % intrinsicM != 0 || numSubgroups % warpsM != 0 ||
      (stepWidth / intrinsicN) % (numSubgroups / warpsM) != 0) {
    return failure();
  }
  int64_t warpsN = numSubgroups / warpsM;
  (void)setTranslationInfo(
      funcOp, IREE::Codegen::TranslationInfoAttr::get(
                  ctx, info.getPassPipeline(), info.getCodegenSpec(),
                  ArrayRef<int64_t>{numThreads, 1, 1}, subgroupSize,
                  info.getConfiguration()));

  // The forall goes after the last chain op.
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

  OpFoldResult outer = tensor::getMixedSize(rewriter, loc, x, 0);
  SmallVector<Value> inits = {c.ropeQ.op.getDpsInits()[0],
                              c.ropeK.op.getDpsInits()[0],
                              c.truncV.getDpsInits()[0]};
  SmallVector<Attribute> mapping = {
      IREE::Codegen::WorkgroupMappingAttr::get(ctx,
                                               IREE::Codegen::WorkgroupId::IdY),
      IREE::Codegen::WorkgroupMappingAttr::get(
          ctx, IREE::Codegen::WorkgroupId::IdX)};
  auto forallOp = scf::ForallOp::create(
      rewriter, loc,
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0),
                             rewriter.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{outer, rewriter.getIndexAttr(inner)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                             rewriter.getIndexAttr(kRowTile)},
      inits, rewriter.getArrayAttr(mapping));

  TiledOps made;
  Operation *sumsqT = nullptr;
  Operation *normT = nullptr;
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forallOp.getTerminator());
    OpFoldResult mA = forallOp.getInductionVar(0);
    OpFoldResult mB = forallOp.getInductionVar(1);

    TileMap tiles;
    SmallVector<OpFoldResult> offsets, sizes;
    getRowTile(rewriter, c.sumsq, 0, 1, mA, mB, offsets, sizes);
    FailureOr<Operation *> ss = tileOp(rewriter, c.sumsq, offsets, sizes, tiles);
    if (failed(ss)) {
      return failure();
    }
    sumsqT = *ss;
    getRowTile(rewriter, c.norm, 0, 1, mA, mB, offsets, sizes);
    FailureOr<Operation *> nm = tileOp(rewriter, c.norm, offsets, sizes, tiles);
    if (failed(nm)) {
      return failure();
    }
    normT = *nm;

    ValueRange outs = forallOp.getRegionIterArgs();
    SmallVector<SmallVector<OpFoldResult>> slabOffsets(3), slabSizes(3);
    SmallVector<Value> slabs;
    Operation *resultOps[3] = {c.ropeQ.op, c.ropeK.op, c.truncV};
    unsigned rowA[3] = {c.ropeQ.rowA, c.ropeK.rowA, 0};
    unsigned rowB[3] = {c.ropeQ.rowB, c.ropeK.rowB, 1};
    for (unsigned i = 0; i < 3; ++i) {
      getRowTile(rewriter, cast<linalg::LinalgOp>(resultOps[i]), rowA[i],
                 rowB[i], mA, mB, offsets, sizes);
      if (failed(cast<TilingInterface>(resultOps[i])
                     .getResultTilePosition(rewriter, 0, offsets, sizes,
                                            slabOffsets[i], slabSizes[i]))) {
        return failure();
      }
      SmallVector<OpFoldResult> strides(slabOffsets[i].size(),
                                        rewriter.getIndexAttr(1));
      slabs.push_back(tensor::ExtractSliceOp::create(
          rewriter, loc, outs[i], slabOffsets[i], slabSizes[i], strides));
    }

    FailureOr<Value> q =
        buildRopeHeadLoop(rewriter, loc, c.ropeQ, c.mmQ, c.truncQ, mA, mB,
                          slabs[0], slabOffsets[0], tiles, made);
    FailureOr<Value> k =
        buildRopeHeadLoop(rewriter, loc, c.ropeK, c.mmK, c.truncK, mA, mB,
                          slabs[1], slabOffsets[1], tiles, made);
    FailureOr<Value> v = buildChunkLoop(rewriter, loc, c.mmV, c.truncV,
                                        stepWidth,
                                        mA, mB, slabs[2], tiles, made);
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
  // The collapse_shape that expandRopeRows put after the K RoPE now reads the
  // forall, which comes after it: move such users behind the forall.
  for (Value result : forallOp.getResults()) {
    for (Operation *user : llvm::to_vector(result.getUsers())) {
      if (user->getBlock() == forallOp->getBlock() &&
          user->isBeforeInBlock(forallOp)) {
        rewriter.moveOpAfter(user, forallOp);
      }
    }
  }

  // Configs for the TileAndFuse levels.
  IREE::GPU::LoweringConfigAttr projCfg = getProjectionConfig(
      ctx, kind, (stepWidth / intrinsicN) / warpsN,
      kReductionK / intrinsicK);
  for (Operation *mm : made.projections) {
    setLoweringConfig(mm, projCfg);
    if (kFuseNormIntoK) {
      continue;
    }
    // xn -> value_barrier -> projection: the reduction level would otherwise
    // fuse the norm into the projection's K loop and recompute it per step.
    OpBuilder vb(mm);
    OpOperand &lhs = mm->getOpOperand(0);
    auto barrier = IREE::GPU::ValueBarrierOp::create(vb, mm->getLoc(),
                                                     ValueRange{lhs.get()});
    lhs.set(barrier.getResult(0));
  }
  if (!kFuseNormIntoK) {
    setLoweringConfig(normT, getThreadConfig(ctx, {0, 1, 8}));
  }
  for (Operation *op : made.elementwise) {
    setLoweringConfig(op, getThreadConfig(ctx, {0, 1, 4}));
  }
  for (auto [op, jDim] : made.ropes) {
    SmallVector<int64_t> tile(cast<linalg::LinalgOp>(op).getNumLoops(), 1);
    tile[jDim] = 4;
    setLoweringConfig(op, getThreadConfig(ctx, tile));
  }
  if (failed(distributeSumSq(rewriter, cast<linalg::LinalgOp>(sumsqT)))) {
    LLVM_DEBUG(llvm::dbgs() << "[qkv-gpu] sum of squares distribution failed\n");
    return failure();
  }
  return success();
}

struct GPURewriteQKVChainAsForallPass final
    : impl::GPURewriteQKVChainAsForallPassBase<GPURewriteQKVChainAsForallPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    if (funcOp->hasAttr(kQKVChainRewrittenAttr)) {
      return;
    }
    IRRewriter rewriter(&getContext());
    // RoPEs read collapse_shape of the [D, 32, N] projections: bring them to
    // the same [D, 32] rows first.
    SmallVector<linalg::GenericOp> candidates;
    funcOp.walk([&](linalg::GenericOp op) {
      for (Value in : op.getDpsInputs()) {
        if (in.getDefiningOp<tensor::CollapseShapeOp>()) {
          candidates.push_back(op);
          return;
        }
      }
    });
    SmallVector<linalg::GenericOp> ropes;
    for (linalg::GenericOp op : candidates) {
      Operation *mm = nullptr;
      // Only RoPE-like consumers of a (truncated) projection.
      bool readsProjection = false;
      for (Value in : op.getDpsInputs()) {
        auto c = in.getDefiningOp<tensor::CollapseShapeOp>();
        if (!c) {
          continue;
        }
        Operation *src = c.getSrc().getDefiningOp();
        readsProjection |= isProjection(src) ||
                           (src && src->getNumOperands() > 0 &&
                            isTruncOf(src, src->getOperand(0).getDefiningOp()) &&
                            isProjection(src->getOperand(0).getDefiningOp()));
      }
      (void)mm;
      if (!readsProjection) {
        continue;
      }
      FailureOr<linalg::GenericOp> expanded = expandRopeRows(rewriter, op);
      if (succeeded(expanded)) {
        ropes.push_back(*expanded);
      }
    }
    if (ropes.size() < 2) {
      return;
    }
    // Q is the RoPE with the most heads.
    std::optional<QKVChain> chain;
    for (linalg::GenericOp rope : ropes) {
      FailureOr<QKVChain> m = matchQKVChain(rope);
      if (succeeded(m) && m->ropeQ.numHeads > m->ropeK.numHeads) {
        chain = *m;
        break;
      }
    }
    if (!chain) {
      LLVM_DEBUG(llvm::dbgs() << "[qkv-gpu] chain does not match\n");
      return;
    }
    funcOp.walk([](Operation *op) { op->removeAttr("lowering_config"); });
    if (failed(rewriteQKVChain(rewriter, funcOp, *chain))) {
      funcOp.emitError("GPURewriteQKVChainAsForall: rewrite failed");
      return signalPassFailure();
    }
    funcOp->setAttr(kQKVChainRewrittenAttr, UnitAttr::get(&getContext()));
  }
};

} // namespace
} // namespace mlir::iree_compiler
