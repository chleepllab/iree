# QKV fusion on GPU (`GPURewriteQKVChainAsForall`)

`--iree-dispatch-creation-fuse-qkv` (RMSNorm + Q/K/V projections + RoPE in
one dispatch, see [DocFuseQKV.md](DocFuseQKV.md)) had only the CPU codegen
side; on CUDA the fused dispatch failed to compile (`failed to apply fusion +
hoisting patterns`). `Codegen/Common/GPURewriteQKVChainAsForall.cpp` is the
GPU side, added to the TileAndFuse pipeline after the MLP rewrite when the
flag is on (`LLVMGPU/Passes.cpp`; pass in `Common/Passes.td`, `CMakeLists.txt`).

```
scf.forall (dA, dB) step (1, 16)                     // S / 16 workgroups, 256 threads
  ss = sum(x[rows]^2)                    [16] f32    // one thread per row, K in steps of 16
  scf.for g in Q heads / 4:                          // 8 steps
    acc = norm(x[rows], ss) @ Wq[256g : +256]^T      // [16, 256], mma.sync 16x8x16, K step 64;
                                                     //   norm recomputed per K slice
    qr[.., 4 heads, rows, ..] = rope(trunc(acc))
  scf.for g in K heads / 4:  (same, 2 steps)
  scf.for c in V cols / 256: v[rows, 256c : +256] = trunc(norm(x) @ Wv[c]^T)   // 2 steps
```

- On GPU the dispatch has S split into `[D, 32]` for every op but the two
  RoPEs, which read `collapse_shape` of the projections. The pass first
  rewrites the RoPEs over the same `[D, 32]` rows, so the 16-row tile is
  static (needed for MMA). The CPU pass's matcher and head loops are reused
  over these two row dims.
- The projections get MMA configs; truncs and RoPEs thread configs. The norm
  has none, so it is fused into each projection's K loop instead of a
  `[16, 2048]` f16 tile in shared memory (64 KiB).

**K/V transpose into the cache scatters** (DocFuseQKV.md, item A) needs
nothing on GPU: the fusion is in dispatch creation, which does not depend on
the target, and the default GPU config compiles the fused scatters (the CPU
pass's `retileCacheScatters` only exists because the CPU tile put a 256 KiB
transposed tile on the stack). At S = 2048 the two scatters with the
transpose inside take 0.29 + 0.26 ms, against 0.52 ms of transposes plus
0.28 + 0.28 ms of scatters without the flag.

GX10, baseline (no flags) vs `--iree-dispatch-creation-fuse-qkv` alone (with A), 1 run each. Builds: `gq_base.vmfb` / `gq_fuse.vmfb` (`~/workspace/tmp/gx10-cuda/`); runs: `bench_s_1run.sh` / `bench_decode_s_1run.sh` (logs `gq_prefill.log` / `gq_decode.log` in `~/llama-mem-cuda`). Decode: one token at position S, KV cache zero-filled. Cells read baseline → fused; "(both)" means equal.

| LLaMA-3.2-1B f16 on GX10 (CUDA) | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 MiB → 39.75 MiB (−3.5 MiB) | 119.50 MiB → 112.50 MiB (−7 MiB) | 371.01 MiB → 357.01 MiB (−14 MiB) |
| Decode transient buffer | 132.33 MiB → 132.32 MiB (−10 KiB) | 136.37 MiB → 136.36 MiB (−10 KiB) | 144.43 MiB → 144.42 MiB (−10 KiB) |
| Prefill `DEVICE_LOCAL` peak | 2781.76 MiB → 2778.26 MiB (−3.5 MiB) | 2983.27 MiB → 2976.27 MiB (−7 MiB) | 3485.28 MiB → 3471.28 MiB (−14 MiB) |
| Decode `DEVICE_LOCAL` peak | 2617.84 MiB → 2617.83 MiB (−10 KiB) | 2621.87 MiB → 2621.86 MiB (−10 KiB) | 2629.93 MiB → 2629.92 MiB (−10 KiB) |
| Prefill `HOST_LOCAL` peak | 253.25 MiB (both) | 378.50 MiB (both) | 629.00 MiB (both) |
| Decode `HOST_LOCAL` peak | 74.01 MiB → 84.01 MiB | 74.01 MiB → 72.01 MiB | 72.01 MiB → 74.01 MiB |
| Prefill wall time (1 run) | 1.72 s → 1.50 s | 2.00 s → 1.83 s | 3.03 s → 3.01 s |
| Decode wall time (1 run) | 1.08 s → 1.09 s | 1.17 s → 1.09 s | 1.14 s → 1.09 s |

- Against the baseline, `fuse-qkv` on its own makes prefill 0.02–0.22 s faster. It is slower only in the comparison below, where the MLP and attention already use hand kernels.
- The transient sizes match x86 byte for byte. On CUDA the transient arena is part of `DEVICE_LOCAL`, so `DEVICE_LOCAL` drops by the same amount.
- Decode `HOST_LOCAL` moves by 2–10 MiB in both directions, and the baseline itself varies across S. This looks like run-to-run variation, not an effect of the flag.
- Decode on its own doesn't change; the −10 KiB comes from the custom slice packer, which the flag also turns on.

GX10, baseline (no flags) vs fuse-mlp + fuse-qkv (with A, B, C) + hand attention, 1 run each. Fused build: `--iree-dispatch-creation-fuse-mlp=true --iree-dispatch-creation-fuse-mlp-min-rows=2 --iree-dispatch-creation-fuse-qkv` (the `min-rows` flag has since been removed; 2 is now fixed) plus `--iree-hal-substitute-executable-source=prefill_bs1$async_dispatch_13=~/workspace/tmp/gx10-hand/attn/attn_d13_qkv.mlir`. `dispatch_13` was checked to still be `attention_8x4xDx64x64xD` in this tree, and the MLP is IREE's own fused kernel, not the hand MLP. Builds: `gq_base.vmfb` / `gx_fused.vmfb` (`~/workspace/tmp/gx10-cuda/`); logs `gx_prefill.log` / `gx_decode.log` in `~/llama-mem-cuda`. Same scripts and inputs as the table above.

| LLaMA-3.2-1B f16 on GX10 (CUDA) | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 MiB → 7.57 MiB (−35.69 MiB) | 119.50 MiB → 16.13 MiB (−103.37 MiB) | 371.01 MiB → 36.26 MiB (−334.75 MiB) |
| Decode transient buffer | 132.33 MiB → 132.29 MiB (−44 KiB) | 136.37 MiB → 136.29 MiB (−76 KiB) | 144.43 MiB → 144.29 MiB (−140 KiB) |
| Prefill `DEVICE_LOCAL` peak | 2781.76 MiB → 2746.08 MiB (−35.69 MiB) | 2983.27 MiB → 2879.90 MiB (−103.37 MiB) | 3485.28 MiB → 3150.54 MiB (−334.75 MiB) |
| Decode `DEVICE_LOCAL` peak | 2617.84 MiB → 2617.79 MiB (−44 KiB) | 2621.87 MiB → 2621.79 MiB (−76 KiB) | 2629.93 MiB → 2629.80 MiB (−140 KiB) |
| Prefill `HOST_LOCAL` peak | 253.25 MiB (both) | 378.50 MiB (both) | 629.00 MiB (both) |
| Decode `HOST_LOCAL` peak | 96.00 MiB → 74.01 MiB | 74.01 MiB (both) | 84.02 MiB → 84.01 MiB |
| Prefill wall time (1 run) | 1.40 s → 1.26 s | 1.80 s → 1.40 s | 2.94 s → 1.58 s |
| Decode wall time (1 run) | 1.10 s → 1.07 s | 1.11 s → 1.14 s | 1.14 s → 1.11 s |

- Prefill transient is fuse-mlp's gate/up removal plus A, B and C together. With B, fuse-mlp's leftover buffers share space with the rest, so the saving grows faster than linearly in S.
- Decode transient: decode sizes depend on the sequence length too, so B packs them, and the saving grows with S.
- Decode `HOST_LOCAL` varies between runs; the 96.00 MiB baseline at S=512 is one such run. It is not an effect of the flags.

GX10 prefill, previous best (hand MLP + mask fold + hand attention, see
[DocGPUFuseAttn.md](DocGPUFuseAttn.md#attention-mask-broadcast-fold-and-hand-written-kernel))
vs the same plus `fuse-qkv`, 1 run each, one session:

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Wall, previous best | 1.16 s | 1.18 s | 1.41 s |
| Wall, + fuse-qkv | 1.19 s | 1.31 s | 1.49 s |
| Transient, previous best | 11.25 MiB | 23.50 MiB | 51.01 MiB |
| Transient, + fuse-qkv | 7.75 MiB (−3.5) | 16.50 MiB (−7) | 37.01 MiB (−14) |
| Total GPU (nsys), previous best | 81.8 ms | 170.0 ms | 310.6 ms |
| Total GPU (nsys), + fuse-qkv | 98.9 ms | 200.6 ms | 360.8 ms |
| Fused QKV kernel | 21.6 ms | 39.2 ms | 68.0 ms |
| Kernels it replaces, unfused | ~5.3 ms | — | ~18.3 ms |
| Logits vs baseline, top-1 | 100% | 99.90% | 99.76% |

So the GPU QKV fusion is correct and saves 3.5 / 7 / 14 MiB more, but it is
~0.03–0.13 s slower: the fused kernel is ~3.7× the separate kernels, because
each 16-row workgroup streams all of Wq / Wk / Wv (12 MiB) from global
memory, where the separate IREE matmuls use taller tiles. Variants tried at
S = 2048 (QKV kernel): 1 head per step 113 ms; 2 heads 76.5 ms; 4 heads with
the norm fused into K **68.0 ms** (kept); 32 rows × 2 heads 77.6 ms; 4 heads
with xn in shared memory, or 32 rows × 4 heads, exceed 99 KiB. The fastest
configuration therefore leaves `fuse-qkv` off.

With `fuse-qkv` the dispatches renumber again: attention is
`prefill_bs1$async_dispatch_13` and the MLP `dispatch_16`. The attention
dispatch also gets 8 constants (K and V offsets become 64-bit:
Q `c0`, K `c1 | c2 << 32`, V `c3 | c4 << 32`, mask `c5`, out `c6`, S `c7`);
`attn_fa.cu` builds for it with `-DATTN_C8
-DATTN_NAME=prefill_bs1_async_dispatch_13_attention_8x4xDx64x64xD`
(`attn_fa_c8_d13.ptx`, `attn_d13_qkv.mlir`; MLP: `h2p3f_d16.mlir`). Build:
`~/workspace/tmp/gx10-cuda/qkv_final.vmfb` (= `qkv_all4f`).

## Mask broadcast fold

The model's mask is `[1, 1, S, S]` f16 (0 or −inf: causal plus key padding).
Lowering SDPA broadcasts it to every head, so two dispatches build it: one
computes `[S·S]` from indices, the next broadcasts it to `[32, S·S]` f16 (256
MiB at S = 2048), which the attention dispatch reads as `tensor<8x4x?x?xf16>`.
That broadcast is most of the transient buffer: baseline transient fits
0.052 MiB·S + 66 B·S², and 32 heads × 2 B = 64 B per (i, j).

`ElementwiseOpFusion` already folds broadcasts into attention's indexing maps
(`FuseTransposeWithAttentionLikeOp`), but upstream #24127
(`wouldBroadcastAwayAttentionDimGroup`) refuses any fusion that drops a whole
attention dim group, which the head (batch) dims of the mask are. The guard
protects the Q/K/V/O maps, from which the dim groups are inferred; the mask
map is not used for that. With the new flag the mask operand is exempt:

- `Utils/CustomFusion.{h,cpp}`: `--iree-dispatch-creation-fuse-attention-mask-broadcast` (now removed; the fold is part of `--iree-dispatch-creation-fuse-attention` and `-fuse-qkv`)
  (default off, upstream behavior).
- `DispatchCreation/ElementwiseOpFusion.cpp`: `wouldBroadcastAwayAttentionDimGroup`
  returns false for the attention op's mask operand when the flag is on.

The attention dispatch then reads the mask as `tensor<?x?xf16>` with map
`(d2, d5)`. Output is byte-identical to the build without the flag at S =
512 / 1024 / 2048. Wall time barely changes: the IREE attention kernel goes from 1,234.8 to
1,222.8 ms at S = 2048 (the 256 MiB mask was read once per layer).

The fold removes one dispatch, so the fused MLP becomes
`prefill_bs1$async_dispatch_27` (was 28) and attention is
`prefill_bs1$async_dispatch_24` (was 25). The MLP substitution file is renamed
to match (`h2p3f_d27.mlir`, `v2p3f_d27.ptx`: `dispatch_28` → `dispatch_27`;
the dispatch is otherwise identical).
