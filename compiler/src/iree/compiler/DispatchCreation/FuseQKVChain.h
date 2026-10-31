// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DISPATCHCREATION_FUSEQKVCHAIN_H_
#define IREE_COMPILER_DISPATCHCREATION_FUSEQKVCHAIN_H_

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::DispatchCreation {

/// Marks the RoPE-on-Q op that roots a fused RMSNorm + QKV + RoPE dispatch.
inline constexpr llvm::StringLiteral kSpecialQKVAttr =
    "iree_dispatch.special_qkv";

/// Under --iree-dispatch-creation-fuse-qkv, rewrites every prefill attention
/// prologue of the form
///
///   xn = rmsnorm(x) * w                         (extf, sum(x^2), scale, *w)
///   q/k/v = trunc(xn * Wq/Wk/Wv^T)              [S, Nq/Nk/Nv]
///   rope(expand(strided_slice(expand(q))), cos, sin)   (same for k)
///
/// into a form one dispatch can hold and RewriteQKVChainAsForall can tile by
/// rows:
///  - each RoPE generic reads `q` / `k` directly through affine maps
///    (s, ..., 2j) and (s, ..., 2j + 1), with the two sequence dims of the old
///    op (S / 32, 32) merged into one; its result is expanded back to the old
///    type outside, which is a free reshape at the dispatch boundary.
///  - the extf / scale / weight generics of the norm are fused into one
///    elementwise op that writes `xn`, and the extf into the sum-of-squares
///    reduction, so the norm has no [S, D] f32 intermediate.
/// The RoPE on Q is tagged with `kSpecialQKVAttr`.
void prepareQKVChains(FunctionOpInterface funcOp);

/// Every op of the chain rooted at the tagged RoPE-on-Q `root`, in the order
/// they can be appended to its fusion group (outward from the root), except the
/// root itself and the fills / empties that get cloned in anyway: the norm
/// reduction and elementwise op, the three matmuls and truncs, and the RoPE on
/// K. Fails if `root` is not such a chain.
LogicalResult collectQKVChain(Operation *root,
                              SmallVectorImpl<Operation *> &chain);

} // namespace mlir::iree_compiler::DispatchCreation

#endif // IREE_COMPILER_DISPATCHCREATION_FUSEQKVCHAIN_H_
