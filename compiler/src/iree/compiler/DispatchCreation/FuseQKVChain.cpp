// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// RMSNorm + Q/K/V projections + RoPE: preparation for one dispatch
//
// See FuseQKVChain.h. In llama prefill the Q and K RoPE ops read the
// projections through
//
//   %q4  = expand_shape %q [[0, 1], [2, 3]] : [S, 2048] -> [1, S, 32, 64]
//   %ev  = extract_slice %q4[0, 0, 0, 0] [1, S, 32, 32] [1, 1, 1, 2]
//   %od  = extract_slice %q4[0, 0, 0, 1] [1, S, 32, 32] [1, 1, 1, 2]
//   %ev5 = expand_shape %ev ... -> [S / 32, 32, 8, 4, 32]
//
// which stock dispatch formation can only turn into four `slow_memcpy`
// dispatches. Here the whole reshape / slice chain is folded into the RoPE
// op's indexing maps instead.
//===----------------------------------------------------------------------===//

#include "iree/compiler/DispatchCreation/FuseQKVChain.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-dispatch-creation-fuse-qkv"

namespace mlir::iree_compiler::DispatchCreation {

namespace {

//===----------------------------------------------------------------------===//
// Small matchers
//===----------------------------------------------------------------------===//

/// `ins(A[M, K], B[N, K]) outs(C[M, N])`, the form of every llama projection.
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

static bool isAllParallel(linalg::GenericOp op) {
  return op.getNumLoops() == op.getNumParallelLoops();
}

/// A one-in one-out all-parallel identity generic whose body is a single
/// arith cast of type `CastOp`.
template <typename CastOp>
static bool isPureCast(Operation *op) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || !isAllParallel(generic) ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); })) {
    return false;
  }
  Block *body = generic.getBlock();
  return llvm::hasNItems(*body, 2) && isa<CastOp>(body->front());
}

/// A generic with exactly one loop, a reduction.
static bool isRowReduction(Operation *op) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(op);
  return generic && generic.getNumReductionLoops() == 1 &&
         generic.getNumDpsInits() == 1;
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

//===----------------------------------------------------------------------===//
// Old-form RoPE: fold expand / strided slice / expand into indexing maps
//===----------------------------------------------------------------------===//

/// Where a RoPE input reads the 2-D projection `t`: `t[row, col]`, both as
/// expressions over the RoPE's loops.
struct ProjectionRead {
  Value t;
  AffineExpr row, col;
};

/// Walks `operand` of `rope` back through expand_shape / static
/// extract_slice ops to the 2-D result of a trunc generic, composing the
/// index expressions on the way.
static std::optional<ProjectionRead> traceProjectionRead(linalg::GenericOp rope,
                                                         OpOperand &operand) {
  unsigned numDims = rope.getNumLoops();
  SmallVector<AffineExpr> exprs =
      llvm::to_vector(rope.getMatchingIndexingMap(&operand).getResults());
  Value v = operand.get();
  MLIRContext *ctx = rope.getContext();
  while (true) {
    Operation *def = v.getDefiningOp();
    if (auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(def)) {
      ArrayRef<int64_t> shape = expand.getResultType().getShape();
      SmallVector<AffineExpr> srcExprs;
      for (ReassociationIndices group : expand.getReassociationIndices()) {
        AffineExpr acc = exprs[group.front()];
        for (int64_t d : ArrayRef<int64_t>(group).drop_front()) {
          // A leading index that is the constant 0 (the unit dim of
          // [S, N] -> [1, S, H, 64]) contributes nothing, whatever the size.
          auto c = dyn_cast<AffineConstantExpr>(acc);
          if (c && c.getValue() == 0) {
            acc = exprs[d];
            continue;
          }
          if (ShapedType::isDynamic(shape[d])) {
            return std::nullopt;
          }
          acc = acc * shape[d] + exprs[d];
        }
        srcExprs.push_back(acc);
      }
      exprs = std::move(srcExprs);
      v = expand.getSrc();
      continue;
    }
    if (auto slice = dyn_cast_or_null<tensor::ExtractSliceOp>(def)) {
      if (ShapedType::isDynamicShape(slice.getStaticOffsets()) ||
          ShapedType::isDynamicShape(slice.getStaticStrides())) {
        return std::nullopt;
      }
      llvm::SmallBitVector dropped = slice.getDroppedDims();
      SmallVector<AffineExpr> srcExprs;
      unsigned next = 0;
      for (auto [i, offset, stride] :
           llvm::enumerate(slice.getStaticOffsets(), slice.getStaticStrides())) {
        if (dropped.test(i)) {
          srcExprs.push_back(getAffineConstantExpr(offset, ctx));
          continue;
        }
        if (next >= exprs.size()) {
          return std::nullopt;
        }
        srcExprs.push_back(exprs[next++] * stride + offset);
      }
      if (next != exprs.size()) {
        return std::nullopt;
      }
      exprs = std::move(srcExprs);
      v = slice.getSource();
      continue;
    }
    break;
  }
  if (exprs.size() != 2 || !isPureCast<arith::TruncFOp>(v.getDefiningOp())) {
    return std::nullopt;
  }
  return ProjectionRead{v, simplifyAffineExpr(exprs[0], numDims, 0),
                        simplifyAffineExpr(exprs[1], numDims, 0)};
}

/// The RoPE generics this works on: all-parallel, one init with an identity
/// map, exactly two inputs reading the same projection at columns `c` and
/// `c + 1`. Returns the even / odd input numbers and the read.
struct OldRope {
  linalg::GenericOp op;
  unsigned even, odd;
  ProjectionRead read;
};

static std::optional<OldRope> matchOldRope(linalg::GenericOp op) {
  if (!isAllParallel(op) || op.getNumDpsInits() != 1 ||
      !op.getMatchingIndexingMap(op.getDpsInitOperand(0)).isIdentity() ||
      !op.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>()) {
    return std::nullopt;
  }
  SmallVector<std::pair<unsigned, ProjectionRead>> reads;
  for (OpOperand *input : op.getDpsInputOperands()) {
    if (std::optional<ProjectionRead> r = traceProjectionRead(op, *input)) {
      reads.push_back({input->getOperandNumber(), *r});
    }
  }
  LLVM_DEBUG(if (!reads.empty()) llvm::dbgs()
             << "[qkv] " << reads.size() << " projection reads in " << op
             << "\n");
  if (reads.size() != 2 || reads[0].second.t != reads[1].second.t ||
      reads[0].second.row != reads[1].second.row) {
    return std::nullopt;
  }
  unsigned numDims = op.getNumLoops();
  auto isPlusOne = [&](AffineExpr a, AffineExpr b) {
    return simplifyAffineExpr(a + 1, numDims, 0) == b;
  };
  if (isPlusOne(reads[0].second.col, reads[1].second.col)) {
    return OldRope{op, reads[0].first, reads[1].first, reads[0].second};
  }
  if (isPlusOne(reads[1].second.col, reads[0].second.col)) {
    return OldRope{op, reads[1].first, reads[0].first, reads[1].second};
  }
  return std::nullopt;
}

/// Finds loops (a, a + 1) with `row == d_a * extent(a + 1) + d_{a + 1}`.
static std::optional<unsigned> findSplitRowDims(linalg::GenericOp op,
                                                AffineExpr row) {
  unsigned numDims = op.getNumLoops();
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  MLIRContext *ctx = op.getContext();
  for (unsigned a = 0; a + 1 < numDims; ++a) {
    if (ShapedType::isDynamic(ranges[a + 1])) {
      continue;
    }
    AffineExpr candidate = getAffineDimExpr(a, ctx) * ranges[a + 1] +
                           getAffineDimExpr(a + 1, ctx);
    if (simplifyAffineExpr(candidate, numDims, 0) == row) {
      return a;
    }
  }
  return std::nullopt;
}

/// Rewrites one old-form RoPE (see the file comment). With `merge` set the
/// loops (merge, merge + 1) are folded into one.
static LogicalResult rewriteRope(RewriterBase &rewriter, OldRope &rope,
                                 std::optional<unsigned> merge) {
  linalg::GenericOp op = rope.op;
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  unsigned numDims = op.getNumLoops();
  unsigned newNumDims = merge ? numDims - 1 : numDims;

  // Old loop -> new loop. With a merge, a and a + 1 both go to a.
  auto remapDim = [&](unsigned d) -> unsigned {
    if (!merge || d <= *merge) {
      return d;
    }
    return d - 1;
  };
  auto isMerged = [&](AffineExpr e) {
    return merge && (e.isFunctionOfDim(*merge) || e.isFunctionOfDim(*merge + 1));
  };
  auto remapExpr = [&](AffineExpr e) {
    SmallVector<AffineExpr> repl;
    for (unsigned d = 0; d < numDims; ++d) {
      repl.push_back(getAffineDimExpr(remapDim(d), ctx));
    }
    return e.replaceDimsAndSymbols(repl, {});
  };

  // linalg.index on a merged loop has no single replacement.
  bool usesMergedIndex = false;
  op.getBlock()->walk([&](linalg::IndexOp index) {
    if (merge && (index.getDim() == *merge || index.getDim() == *merge + 1)) {
      usesMergedIndex = true;
    }
  });
  if (usesMergedIndex || isMerged(rope.read.col)) {
    return failure();
  }

  // For an operand map other than the two projection reads: drop the merged
  // pair to one result and, if the operand has that pair, collapse it.
  // Returns the new map and the reassociation (empty if no collapse).
  auto remapOperandMap =
      [&](AffineMap map,
          SmallVector<ReassociationIndices> &reassoc) -> FailureOr<AffineMap> {
    SmallVector<AffineExpr> results;
    reassoc.clear();
    ArrayRef<AffineExpr> old = map.getResults();
    bool collapsed = false;
    for (unsigned i = 0; i < old.size(); ++i) {
      if (merge && i + 1 < old.size() &&
          old[i] == getAffineDimExpr(*merge, ctx) &&
          old[i + 1] == getAffineDimExpr(*merge + 1, ctx)) {
        results.push_back(getAffineDimExpr(*merge, ctx));
        reassoc.push_back({static_cast<int64_t>(i), static_cast<int64_t>(i + 1)});
        collapsed = true;
        ++i;
        continue;
      }
      if (isMerged(old[i])) {
        return failure();
      }
      results.push_back(remapExpr(old[i]));
      reassoc.push_back({static_cast<int64_t>(i)});
    }
    if (!collapsed) {
      reassoc.clear();
    }
    return AffineMap::get(newNumDims, 0, results, ctx);
  };

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputs;
  SmallVector<AffineMap> maps;
  AffineExpr row = getAffineDimExpr(merge ? *merge : 0, ctx);
  if (!merge) {
    auto dim = dyn_cast<AffineDimExpr>(rope.read.row);
    if (!dim) {
      return failure();
    }
    row = dim;
  }
  for (OpOperand *input : op.getDpsInputOperands()) {
    unsigned number = input->getOperandNumber();
    if (number == rope.even || number == rope.odd) {
      AffineExpr col = rope.read.col;
      if (number == rope.odd) {
        col = col + 1;
      }
      col = simplifyAffineExpr(remapExpr(col), newNumDims, 0);
      inputs.push_back(rope.read.t);
      maps.push_back(AffineMap::get(newNumDims, 0, {row, col}, ctx));
      continue;
    }
    SmallVector<ReassociationIndices> reassoc;
    FailureOr<AffineMap> map =
        remapOperandMap(op.getMatchingIndexingMap(input), reassoc);
    if (failed(map)) {
      return failure();
    }
    Value value = input->get();
    if (!reassoc.empty()) {
      value = tensor::CollapseShapeOp::create(rewriter, loc, value, reassoc);
    }
    inputs.push_back(value);
    maps.push_back(*map);
  }

  // Output: identity over the new loops, on a fresh empty of the merged shape.
  auto oldEmpty = op.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
  SmallVector<OpFoldResult> oldSizes = oldEmpty.getMixedSizes();
  SmallVector<OpFoldResult> newSizes;
  SmallVector<ReassociationIndices> outReassoc;
  for (unsigned d = 0; d < numDims; ++d) {
    if (merge && d == *merge) {
      AffineExpr s0, s1;
      bindSymbols(ctx, s0, s1);
      newSizes.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, s0 * s1, {oldSizes[d], oldSizes[d + 1]}));
      outReassoc.push_back({d, d + 1});
      ++d;
      continue;
    }
    newSizes.push_back(oldSizes[d]);
    outReassoc.push_back({d});
  }
  Type elemType = oldEmpty.getType().getElementType();
  Value newInit = tensor::EmptyOp::create(rewriter, loc, newSizes, elemType);
  maps.push_back(AffineMap::getMultiDimIdentityMap(newNumDims, ctx));

  SmallVector<utils::IteratorType> iterators(newNumDims,
                                             utils::IteratorType::parallel);
  auto newOp = linalg::GenericOp::create(rewriter, loc, newInit.getType(),
                                         inputs, ValueRange{newInit}, maps,
                                         iterators);
  rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                             newOp.getRegion().end());
  newOp.getBlock()->walk([&](linalg::IndexOp index) {
    index.setDim(remapDim(index.getDim()));
  });

  Value result = newOp.getResult(0);
  if (merge) {
    result = tensor::ExpandShapeOp::create(rewriter, loc, op.getResultTypes()[0],
                                           result, outReassoc, oldSizes);
  }
  rewriter.replaceOp(op, result);
  rope.op = newOp;
  return success();
}

//===----------------------------------------------------------------------===//
// Norm: fuse the casts / scale into the ops that read them
//===----------------------------------------------------------------------===//

/// Fuses into `consumer` (repeatedly) every input producer that is either a
/// pure extf or, with `allowNormScale`, an elementwise op reading a row
/// reduction (the rsqrt scale of the RMSNorm). A producer with other users is
/// cloned for this one first so the fused op does not also yield it.
static linalg::GenericOp fuseNormProducers(RewriterBase &rewriter,
                                           linalg::GenericOp consumer,
                                           bool allowNormScale) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (OpOperand *input : consumer.getDpsInputOperands()) {
      auto producer = input->get().getDefiningOp<linalg::GenericOp>();
      if (!producer || producer->getNumResults() != 1) {
        continue;
      }
      bool isScale =
          allowNormScale && isAllParallel(producer) &&
          llvm::any_of(producer.getDpsInputs(), [](Value v) {
            return isRowReduction(v.getDefiningOp());
          });
      if (!isPureCast<arith::ExtFOp>(producer) && !isScale) {
        continue;
      }
      if (!linalg::areElementwiseOpsFusable(input)) {
        continue;
      }
      if (getNonDimUsers(producer->getResult(0)).size() > 1) {
        rewriter.setInsertionPoint(producer);
        Operation *clone = rewriter.clone(*producer);
        input->set(clone->getResult(0));
        producer = cast<linalg::GenericOp>(clone);
      }
      rewriter.setInsertionPoint(consumer);
      FailureOr<linalg::ElementwiseOpFusionResult> fused =
          linalg::fuseElementwiseOps(rewriter, input);
      if (failed(fused)) {
        continue;
      }
      Operation *fusedOp = fused->fusedOp;
      for (auto [oldValue, newValue] : fused->replacements) {
        rewriter.replaceUsesWithIf(oldValue, newValue, [&](OpOperand &use) {
          return use.getOwner() != fusedOp;
        });
      }
      rewriter.eraseOp(consumer);
      if (producer->use_empty()) {
        rewriter.eraseOp(producer);
      }
      consumer = cast<linalg::GenericOp>(fused->fusedOp);
      changed = true;
      break;
    }
  }
  return consumer;
}

} // namespace

//===----------------------------------------------------------------------===//
// New-form chain matcher (shared by preparation and dispatch formation)
//===----------------------------------------------------------------------===//

namespace {
struct QKVChain {
  linalg::GenericOp ropeQ, ropeK;
  Operation *truncQ = nullptr, *truncK = nullptr, *truncV = nullptr;
  Operation *mmQ = nullptr, *mmK = nullptr, *mmV = nullptr;
  linalg::GenericOp norm;
  Operation *sumsq = nullptr;
};
} // namespace

/// The projection a new-form RoPE reads: its only input that is the trunc of
/// a matmul.
static Value getRopeProjection(linalg::GenericOp rope) {
  Value t;
  for (Value input : rope.getDpsInputs()) {
    Operation *def = input.getDefiningOp();
    if (!isPureCast<arith::TruncFOp>(def) ||
        !isTransposedBMatmul(
            cast<linalg::GenericOp>(def).getDpsInputs()[0].getDefiningOp())) {
      continue;
    }
    if (t && t != input) {
      return Value();
    }
    t = input;
  }
  return t;
}

static FailureOr<QKVChain> matchQKVChain(linalg::GenericOp ropeQ) {
  QKVChain chain;
  chain.ropeQ = ropeQ;
  Value tq = getRopeProjection(ropeQ);
  if (!tq) {
    return failure();
  }
  chain.truncQ = tq.getDefiningOp();
  chain.mmQ =
      cast<linalg::GenericOp>(chain.truncQ).getDpsInputs()[0].getDefiningOp();
  if (!isTransposedBMatmul(chain.mmQ)) {
    return failure();
  }
  Value xn = cast<linalg::LinalgOp>(chain.mmQ).getDpsInputs()[0];
  chain.norm = xn.getDefiningOp<linalg::GenericOp>();
  if (!chain.norm || !isAllParallel(chain.norm) ||
      chain.norm->getNumResults() != 1) {
    return failure();
  }
  for (Value input : chain.norm.getDpsInputs()) {
    if (isRowReduction(input.getDefiningOp())) {
      chain.sumsq = input.getDefiningOp();
    }
  }
  if (!chain.sumsq) {
    return failure();
  }

  SmallVector<Operation *> matmuls = getNonDimUsers(xn);
  if (matmuls.size() != 3 || !llvm::all_of(matmuls, isTransposedBMatmul)) {
    return failure();
  }
  chain.mmK = chain.mmV = nullptr;
  for (Operation *mm : matmuls) {
    if (mm == chain.mmQ) {
      continue;
    }
    SmallVector<Operation *> mmUsers = getNonDimUsers(mm->getResult(0));
    if (mmUsers.size() != 1 || !isPureCast<arith::TruncFOp>(mmUsers[0])) {
      return failure();
    }
    Operation *trunc = mmUsers[0];
    linalg::GenericOp rope;
    for (Operation *user : getNonDimUsers(trunc->getResult(0))) {
      auto generic = dyn_cast<linalg::GenericOp>(user);
      if (generic && generic != ropeQ &&
          getRopeProjection(generic) == trunc->getResult(0)) {
        rope = generic;
      }
    }
    if (rope) {
      if (chain.mmK) {
        return failure();
      }
      chain.mmK = mm;
      chain.truncK = trunc;
      chain.ropeK = rope;
    } else {
      if (chain.mmV) {
        return failure();
      }
      chain.mmV = mm;
      chain.truncV = trunc;
    }
  }
  if (!chain.mmK || !chain.mmV) {
    return failure();
  }
  // Q's projection and its trunc are read by the RoPE only.
  if (getNonDimUsers(chain.mmQ->getResult(0)) !=
          SmallVector<Operation *>{chain.truncQ} ||
      getNonDimUsers(tq) != SmallVector<Operation *>{ropeQ.getOperation()} ||
      getNonDimUsers(chain.truncK->getResult(0)) !=
          SmallVector<Operation *>{chain.ropeK.getOperation()}) {
    return failure();
  }
  return chain;
}

LogicalResult collectQKVChain(Operation *root,
                              SmallVectorImpl<Operation *> &ops) {
  auto ropeQ = dyn_cast<linalg::GenericOp>(root);
  if (!ropeQ) {
    return failure();
  }
  FailureOr<QKVChain> chain = matchQKVChain(ropeQ);
  if (failed(chain)) {
    return failure();
  }
  // Outward from the root: FusionGroup needs every op appended to be a
  // producer or a consumer (not both) of ops already in the group.
  ops.append({chain->truncQ, chain->mmQ, chain->norm, chain->sumsq,
              chain->mmK, chain->truncK, chain->ropeK.getOperation(),
              chain->mmV, chain->truncV});
  return success();
}

//===----------------------------------------------------------------------===//
// Preparation
//===----------------------------------------------------------------------===//

void prepareQKVChains(FunctionOpInterface funcOp) {
  IRRewriter rewriter(funcOp.getContext());

  // 1. Old-form RoPEs, grouped by the normalized activation their projection
  // reads.
  SmallVector<OldRope> ropes;
  funcOp.walk([&](linalg::GenericOp op) {
    if (std::optional<OldRope> rope = matchOldRope(op)) {
      ropes.push_back(*rope);
    }
  });

  llvm::MapVector<Value, SmallVector<unsigned>> byActivation;
  for (auto [i, rope] : llvm::enumerate(ropes)) {
    Operation *mm =
        cast<linalg::GenericOp>(rope.read.t.getDefiningOp()).getDpsInputs()[0]
            .getDefiningOp();
    if (!isTransposedBMatmul(mm)) {
      continue;
    }
    byActivation[cast<linalg::LinalgOp>(mm).getDpsInputs()[0]].push_back(i);
  }

  SmallVector<linalg::GenericOp> candidates;
  for (auto &[xn, indices] : byActivation) {
    if (indices.size() != 2) {
      continue;
    }
    // Q is the RoPE over the wider projection.
    auto width = [&](unsigned i) {
      return cast<ShapedType>(ropes[i].read.t.getType()).getDimSize(1);
    };
    unsigned qi = indices[0], ki = indices[1];
    if (width(ki) > width(qi)) {
      std::swap(qi, ki);
    }
    if (width(qi) == width(ki)) {
      continue;
    }

    bool ok = true;
    for (unsigned i : {ki, qi}) {
      OldRope &rope = ropes[i];
      std::optional<unsigned> merge;
      if (!isa<AffineDimExpr>(rope.read.row)) {
        merge = findSplitRowDims(rope.op, rope.read.row);
        if (!merge) {
          ok = false;
          break;
        }
      }
      if (failed(rewriteRope(rewriter, rope, merge))) {
        LLVM_DEBUG(llvm::dbgs() << "[qkv] could not rewrite rope "
                                << rope.op << "\n");
        ok = false;
        break;
      }
    }
    if (!ok) {
      continue;
    }

    // 2. The norm: fuse its casts and the rsqrt scale into the op producing
    // xn, and the extf into the sum of squares.
    auto normOp = xn.getDefiningOp<linalg::GenericOp>();
    if (!normOp) {
      continue;
    }
    normOp = fuseNormProducers(rewriter, normOp, /*allowNormScale=*/true);
    for (Value input : normOp.getDpsInputs()) {
      if (auto reduction = input.getDefiningOp<linalg::GenericOp>();
          reduction && isRowReduction(reduction)) {
        fuseNormProducers(rewriter, reduction, /*allowNormScale=*/false);
        break;
      }
    }

    candidates.push_back(ropes[qi].op);
  }

  // The slices / reshapes the old RoPEs read are dead now.
  bool erased = true;
  while (erased) {
    erased = false;
    funcOp.walk([&](Operation *op) {
      if (isa<tensor::ExtractSliceOp, tensor::ExpandShapeOp,
              tensor::CollapseShapeOp, tensor::EmptyOp>(op) &&
          op->use_empty()) {
        rewriter.eraseOp(op);
        erased = true;
      }
    });
  }

  // 3. Tag the RoPEs on Q whose chain now matches (the dead reshapes / slices
  // above would otherwise count as extra users of the projections).
  for (linalg::GenericOp ropeQ : candidates) {
    if (succeeded(matchQKVChain(ropeQ))) {
      ropeQ->setAttr(kSpecialQKVAttr, rewriter.getUnitAttr());
      LLVM_DEBUG(llvm::dbgs() << "[qkv] tagged " << ropeQ << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[qkv] chain does not match after prep\n");
    }
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
