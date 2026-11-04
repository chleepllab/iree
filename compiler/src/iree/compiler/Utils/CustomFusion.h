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

// --iree-dispatch-creation-fuse-qkv: RMSNorm + Q/K/V projections + RoPE fusion
// (prefill), its codegen rewrite, the K / V cache transposes fused into the
// cache scatters, the custom slice packing, and the attention mask broadcast
// fold (also under fuse-attention): a broadcast producing an attention mask folds into the mask's
// indexing map even when it broadcasts away whole dim groups (upstream refuses
// that for every operand; the dim groups are inferred from Q/K/V/O only, so
// the mask is safe), so the [heads, S, S] mask is never materialized.
bool isQKVFusionEnabled();

// --iree-dispatch-creation-fuse-attention: attention on IREE's own kernels.
//  - TensorDynamicDimAnalysis reads the dynamic dims of
//    `iree_codegen.load_from_buffer` results from the buffer (upstream only
//    handles dispatch.tensor.load and tensor.empty), so their divisibility
//    reaches BlockDynamicDimensions, which blocks K / V / the mask's K2 dim to
//    [?, 32] and lets the GPU MMA attention config apply.
//  - Causal attentions are tagged in dispatch creation and their fully masked
//    K2 tiles skipped in codegen (SkipCausalMaskedAttentionTiles).
//  - The attention mask broadcast fold (also on under fuse-qkv).
bool isAttentionFusionEnabled();

// The attention mask broadcast fold: under fuse-qkv or fuse-attention.
bool isAttentionMaskBroadcastFoldEnabled();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_CUSTOMFUSION_H_
