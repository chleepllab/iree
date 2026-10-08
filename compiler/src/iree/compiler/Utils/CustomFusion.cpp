// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/CustomFusion.h"

#include "llvm/Support/CommandLine.h"

namespace mlir::iree_compiler {

static llvm::cl::opt<bool> clFuseConvChain(
    "iree-dispatch-creation-fuse-conv-chain",
    llvm::cl::desc("Fuse conv -> (elementwise / pad) -> conv or maxpool chains "
                   "into single dispatches, fuse conv padding into the "
                   "dispatch that produces it, and pack static slices with "
                   "the custom packer."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clCustomPacking(
    "iree-stream-custom-packing",
    llvm::cl::desc("Pack static transient slices with the custom packer "
                   "instead of the upstream greedy one. Implied by "
                   "--iree-dispatch-creation-fuse-conv-chain and "
                   "--iree-dispatch-creation-fuse-mlp."),
    llvm::cl::init(false));

// The fused dispatch needs RewriteMLPChainAsForall in codegen (LLVMCPU only)
// to avoid materializing the [S, ffn] intermediates in it.
static llvm::cl::opt<bool> clFuseMLP(
    "iree-dispatch-creation-fuse-mlp",
    llvm::cl::desc("Fuse a gated MLP (gate/up matmuls, silu(gate) * up, down "
                   "matmul and its epilogue) into a single dispatch, and "
                   "pack static slices with the custom packer."),
    llvm::cl::init(false));

bool isConvChainFusionEnabled() { return clFuseConvChain; }

bool isCustomPackingEnabled() { return clCustomPacking || clFuseConvChain || clFuseMLP; }

bool isMLPFusionEnabled() { return clFuseMLP; }

} // namespace mlir::iree_compiler
