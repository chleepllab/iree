// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SkipCausalMaskedAttentionTiles
//
// Dispatch creation (FuseQKVChain.cpp, under
// --iree-dispatch-creation-fuse-attention)
// tags an attention op whose mask is provably -inf wherever the key position
// exceeds the query position:
//
//   iree_linalg_ext.causal_mask = {delta = d}
//
// meaning the mask is -inf at every iteration point where
// lin(K2) - lin(M) + d > 0, with lin the row-major position over the K2
// (resp. M) dims that the mask's map reads, in increasing order. Collapsing or expanding dims inside
// those groups does not change lin, so the tag stays valid through dispatch
// creation and codegen's re-blocking.
//
// After the attention is tiled into a serial scf.for over K2, the tiled
// online_attention covers one tile per iteration. Its key positions start at
// k_min(iv) = k0 + a * iv (from the K slice's offsets) and its query positions
// end at q_max (from the Q slice's offsets and sizes), all traced back to the
// full tensors. When k_min(iv) - q_max + d > 0 the whole tile is -inf: its
// probabilities are exp2(-inf) = 0, the running max is unchanged (it starts at
// a finite value), and so the rescale factor is exp2(0) = 1. Such an
// iteration changes none of the loop-carried values, and since k_min grows
// with iv, every iteration after the first fully masked one is fully masked
// too. The loop's upper bound is lowered to
//
//   ub' = min(ub, max(lb, floordiv(q_max - d - k0, a) + 1))
//
// which drops exactly those iterations (about half of them for a causal mask).
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-skip-causal-masked-attention-tiles"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SKIPCAUSALMASKEDATTENTIONTILESPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

static llvm::cl::opt<bool> clSkipCausalTiles(
    "iree-codegen-causal-attention-tile-skip",
    llvm::cl::desc("Skip the K2 tiles of a causal attention whose mask is all "
                   "-inf (attentions tagged under "
                   "--iree-dispatch-creation-fuse-attention). For A/B "
                   "comparisons."),
    llvm::cl::init(true), llvm::cl::Hidden);

namespace {

/// Set in dispatch creation (FuseQKVChain.h).
static constexpr StringLiteral kCausalMaskAttr = "iree_linalg_ext.causal_mask";

static std::optional<int64_t> getCausalDelta(Operation *op) {
  auto dict = op->getAttrOfType<DictionaryAttr>(kCausalMaskAttr);
  if (!dict) {
    return std::nullopt;
  }
  auto delta = dict.getAs<IntegerAttr>("delta");
  if (!delta) {
    return std::nullopt;
  }
  return delta.getInt();
}

/// One iteration dim of a tiled op: the terms adding up to its offset in the
/// full tensor, its tile size, and its full size.
struct DimTile {
  SmallVector<OpFoldResult> offsets;
  OpFoldResult size;
  int64_t fullSize = ShapedType::kDynamic;
};

/// The tiles of `dims` read off `operand`, a chain of unit-stride,
/// non-rank-reducing extract_slices of a tensor loaded whole from a binding,
/// through `map` (whose results must be plain dims).
static FailureOr<SmallVector<DimTile>>
getDimTiles(Value operand, AffineMap map, ArrayRef<int64_t> dims) {
  auto slice = operand.getDefiningOp<tensor::ExtractSliceOp>();
  if (!slice) {
    return failure();
  }
  SmallVector<OpFoldResult> sizes = slice.getMixedSizes();
  SmallVector<SmallVector<OpFoldResult>> offsets(sizes.size());
  Value root = operand;
  while (slice) {
    if (slice.getSourceType().getRank() != slice.getResultType().getRank() ||
        !slice.hasUnitStride()) {
      return failure();
    }
    for (auto [i, offset] : llvm::enumerate(slice.getMixedOffsets())) {
      offsets[i].push_back(offset);
    }
    root = slice.getSource();
    slice = root.getDefiningOp<tensor::ExtractSliceOp>();
  }
  if (!root.getDefiningOp<IREE::Codegen::LoadFromBufferOp>() &&
      !root.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>()) {
    return failure();
  }
  auto rootType = cast<ShapedType>(root.getType());
  SmallVector<DimTile> tiles;
  for (int64_t dim : dims) {
    std::optional<unsigned> pos = map.getResultPosition(
        getAffineDimExpr(dim, map.getContext()));
    if (!pos) {
      return failure();
    }
    tiles.push_back({offsets[*pos], sizes[*pos], rootType.getDimSize(*pos)});
  }
  return tiles;
}

/// Row-major strides of a dim group from its full sizes; the outermost may be
/// dynamic.
static std::optional<SmallVector<int64_t>>
getStrides(ArrayRef<DimTile> tiles) {
  SmallVector<int64_t> strides(tiles.size(), 1);
  for (int64_t i = static_cast<int64_t>(tiles.size()) - 2; i >= 0; --i) {
    if (ShapedType::isDynamic(tiles[i + 1].fullSize)) {
      return std::nullopt;
    }
    strides[i] = strides[i + 1] * tiles[i + 1].fullSize;
  }
  return strides;
}

static bool isZero(OpFoldResult ofr) {
  std::optional<int64_t> c = getConstantIntValue(ofr);
  return c && *c == 0;
}

/// Lowers the bound of the scf.for over K2 around `attn`, if the loop carries
/// only the attention's state. Returns true if the loop was changed.
static bool skipMaskedTiles(IREE::LinalgExt::OnlineAttentionOp attn,
                            int64_t delta) {
  auto forOp = dyn_cast<scf::ForOp>(attn->getParentOp());
  if (!forOp || !attn.getMask()) {
    return false;
  }
  // Each loop-carried value is an attention result, either yielded directly
  // or inserted into its own iter_arg (the CPU tiling).
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (auto [i, yielded] : llvm::enumerate(yield.getOperands())) {
    if (auto insert = yielded.getDefiningOp<tensor::InsertSliceOp>()) {
      if (insert.getDest() != forOp.getRegionIterArgs()[i]) {
        return false;
      }
      yielded = insert.getSource();
    }
    if (yielded.getDefiningOp() != attn.getOperation()) {
      return false;
    }
  }
  FailureOr<IREE::LinalgExt::AttentionOpDetail> detail =
      IREE::LinalgExt::AttentionOpDetail::get(
          attn.getQueryMap(), attn.getKeyMap(), attn.getValueMap(),
          attn.getOutputMap());
  if (failed(detail)) {
    return false;
  }
  // Only the M / K2 dims the mask reads make up the positions (M also holds
  // the query heads).
  std::optional<AffineMap> maskMap = attn.getMaskMap();
  if (!maskMap) {
    return false;
  }
  auto inMask = [&](ArrayRef<int64_t> dims) {
    SmallVector<int64_t> used;
    for (int64_t dim : dims) {
      if (maskMap->getResultPosition(getAffineDimExpr(dim, attn.getContext()))) {
        used.push_back(dim);
      }
    }
    return used;
  };
  SmallVector<int64_t> mDims = inMask(detail->getMDims());
  SmallVector<int64_t> k2Dims = inMask(detail->getK2Dims());
  llvm::sort(mDims);
  llvm::sort(k2Dims);
  FailureOr<SmallVector<DimTile>> mTiles =
      getDimTiles(attn.getQuery(), attn.getQueryMap(), mDims);
  FailureOr<SmallVector<DimTile>> kTiles =
      getDimTiles(attn.getKey(), attn.getKeyMap(), k2Dims);
  if (failed(mTiles) || failed(kTiles)) {
    return false;
  }
  std::optional<SmallVector<int64_t>> mStrides = getStrides(*mTiles);
  std::optional<SmallVector<int64_t>> kStrides = getStrides(*kTiles);
  if (!mStrides || !kStrides) {
    return false;
  }

  Value iv = forOp.getInductionVar();
  // Loop invariant: defined outside the loop, or computed in its body by
  // pure ops from such values (e.g. the CPU tiling's affine.apply of the
  // workgroup offsets); those are cloned in front of the loop when used.
  std::function<bool(Value)> isInvariantValue = [&](Value v) {
    if (forOp.isDefinedOutsideOfLoop(v)) {
      return true;
    }
    Operation *def = v.getDefiningOp();
    return def && def->getBlock() == forOp.getBody() &&
           def->getNumRegions() == 0 && def->getNumResults() == 1 &&
           isMemoryEffectFree(def) &&
           llvm::all_of(def->getOperands(), isInvariantValue);
  };
  auto isInvariant = [&](OpFoldResult ofr) {
    auto v = dyn_cast<Value>(ofr);
    return !v || isInvariantValue(v);
  };
  // Everything on the query side must be loop invariant.
  for (const DimTile &tile : *mTiles) {
    if (!llvm::all_of(tile.offsets, isInvariant) || !isInvariant(tile.size)) {
      return false;
    }
  }
  for (const DimTile &tile : *kTiles) {
    for (OpFoldResult term : tile.offsets) {
      if (!isInvariant(term) && dyn_cast<Value>(term) != iv) {
        return false;
      }
    }
  }

  OpBuilder b(forOp);
  Location loc = forOp.getLoc();
  IRMapping hoisted;
  std::function<Value(Value)> hoist = [&](Value v) -> Value {
    if (forOp.isDefinedOutsideOfLoop(v)) {
      return v;
    }
    if (Value mapped = hoisted.lookupOrNull(v)) {
      return mapped;
    }
    Operation *def = v.getDefiningOp();
    for (Value operand : def->getOperands()) {
      hoisted.map(operand, hoist(operand));
    }
    return b.clone(*def, hoisted)->getResult(0);
  };
  auto materialize = [&](OpBuilder &, Location, OpFoldResult ofr) -> Value {
    if (auto v = dyn_cast<Value>(ofr)) {
      return hoist(v);
    }
    return arith::ConstantIndexOp::create(
        b, loc, cast<IntegerAttr>(cast<Attribute>(ofr)).getInt());
  };
  auto c = [&](int64_t v) -> Value {
    return arith::ConstantIndexOp::create(b, loc, v);
  };
  auto mulC = [&](Value v, int64_t m) -> Value {
    return m == 1 ? v : arith::MulIOp::create(b, loc, v, c(m)).getResult();
  };
  // q_max = sum(stride * (offset + size - 1)) over M.
  Value qMax = c(0);
  for (auto [tile, stride] : llvm::zip_equal(*mTiles, *mStrides)) {
    for (OpFoldResult term : tile.offsets) {
      if (!isZero(term)) {
        qMax = arith::AddIOp::create(b, loc, qMax,
                                     mulC(materialize(b, loc, term), stride));
      }
    }
    Value last =
        arith::SubIOp::create(b, loc, materialize(b, loc, tile.size), c(1));
    qMax = arith::AddIOp::create(b, loc, qMax, mulC(last, stride));
  }
  // k_min = k0 + a * iv over K2.
  int64_t a = 0;
  Value k0 = c(0);
  for (auto [tile, stride] : llvm::zip_equal(*kTiles, *kStrides)) {
    for (OpFoldResult term : tile.offsets) {
      if (isZero(term)) {
        continue;
      }
      if (dyn_cast<Value>(term) == iv) {
        a += stride;
        continue;
      }
      k0 = arith::AddIOp::create(b, loc, k0,
                                 mulC(materialize(b, loc, term), stride));
    }
  }
  if (a <= 0) {
    return false;
  }

  // Iterations with iv <= floordiv(q_max - delta - k0, a) hold an unmasked
  // element; every later one is fully masked.
  Value num = arith::SubIOp::create(
      b, loc, arith::SubIOp::create(b, loc, qMax, c(delta)), k0);
  Value last = arith::FloorDivSIOp::create(b, loc, num, c(a));
  Value newUb = arith::AddIOp::create(b, loc, last, c(1));
  newUb = arith::MaxSIOp::create(b, loc, newUb, forOp.getLowerBound());
  newUb = arith::MinSIOp::create(b, loc, newUb, forOp.getUpperBound());
  forOp.getUpperBoundMutable().assign(newUb);
  LLVM_DEBUG(llvm::dbgs() << "[causal] lowered K2 loop bound of " << forOp
                          << "\n");
  return true;
}

struct SkipCausalMaskedAttentionTilesPass final
    : impl::SkipCausalMaskedAttentionTilesPassBase<
          SkipCausalMaskedAttentionTilesPass> {
  void runOnOperation() override {
    if (!clSkipCausalTiles) {
      return;
    }
    getOperation()->walk([&](IREE::LinalgExt::OnlineAttentionOp attn) {
      std::optional<int64_t> delta = getCausalDelta(attn);
      if (!delta) {
        return;
      }
      if (!skipMaskedTiles(attn, *delta)) {
        LLVM_DEBUG(llvm::dbgs() << "[causal] no K2 loop to shorten for "
                                << attn << "\n");
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler
