// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_CUSTOMFUSION_H_
#define IREE_COMPILER_UTILS_CUSTOMFUSION_H_

namespace mlir::iree_compiler {

// Runtime switches for the custom fusion work. Both default to off, which
// leaves the compiler behaving as upstream IREE does.

// --iree-dispatch-creation-fuse-conv-chain: conv -> conv / conv -> maxpool
// chain fusion, the pad-into-producer fusion it relies on, the matching
// codegen rewrites and the custom static slice packing.
bool isConvChainFusionEnabled();

// --iree-stream-custom-packing: the custom static slice packer alone. Also on
// whenever conv chain or MLP fusion is.
bool isCustomPackingEnabled();

// --iree-dispatch-creation-fuse-mlp: gated MLP fusion, its codegen rewrite and
// the custom static slice packing. Independent of the conv chain work: pass
// both flags to get both.
bool isMLPFusionEnabled();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_CUSTOMFUSION_H_
