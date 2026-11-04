# Fusion overview (LLaMA-3.2-1B f16)

What to turn on, and where the details are. The fusion flags are shared: each
`--iree-dispatch-creation-fuse-*` flag fuses at dispatch creation (common to
all backends) and turns on that fusion's codegen on CPU (LLVMCPU) and GPU
(CUDA). Per-backend details and measurements are in the per-feature docs below.

## Flags

| Flag | What it fuses | CPU | GPU |
|---|---|---|---|
| `--iree-dispatch-creation-fuse-mlp` | gate/up matmuls, `silu(gate) * up`, down matmul + epilogue into one dispatch; never fuses decode (M = 1) | [DocFuseMlp.md](DocFuseMlp.md) | [DocGPUFuseMlp.md](DocGPUFuseMlp.md) |
| `--iree-dispatch-creation-fuse-qkv` | RMSNorm + Q/K/V projections + RoPE (prefill only), cache transposes into the scatters, mask broadcast fold | [DocFuseQKV.md](DocFuseQKV.md) | [DocGPUFuseQKV.md](DocGPUFuseQKV.md) |
| `--iree-dispatch-creation-fuse-attention` | attention on IREE's own kernels: K / V / mask K2 dim blocking (MMA config on GPU), causal tile skipping, mask broadcast fold | [DocFuseAttention.md](DocFuseAttention.md) | [DocGPUFuseAttention.md](DocGPUFuseAttention.md) |

All three default to off. `fuse-mlp` and `fuse-qkv` also turn on the custom
slice packer (`--iree-stream-custom-packing`).

## What each flag turns on

### 1. FuseQKV (RMSNorm + QKV + RoPE), `--iree-dispatch-creation-fuse-qkv`

- **Chain preparation** (`DispatchCreation/FuseQKVChain.cpp`, `prepareQKVChains`), before dispatch formation:
  - RoPE rewrite: each RoPE reads q / k directly at columns 2j / 2j + 1, which removes the 4 `slow_memcpy` dispatches per layer.
  - Norm fusion: the norm's extf, scale and weight multiply become one op, and the extf goes into the sum of squares, so there is no `[S, 2048]` f32 temporary.
- **Dispatch formation** (`FormDispatchRegions.cpp`): the norm, the 3 matmuls with their truncs, and both RoPEs go into one dispatch rooted at RoPE-Q, tagged `iree_dispatch.special_qkv`.
- **CPU codegen** (`Codegen/Common/RewriteQKVChainAsForall.cpp`): a forall over row tiles that normalizes the rows once, then loops over Q heads, K heads and V chunks, with RoPE as each head's epilogue.
- **GPU codegen** (`Codegen/Common/GPURewriteQKVChainAsForall.cpp`): RoPEs re-expressed over `[D, 32]` rows, a static 16-row tile, `mma.sync` for the projections, and the norm recomputed inside each projection's K loop.
- **A. K/V transpose into the cache scatters**: the transpose is fused into the KV-cache scatter's dispatch. CPU also needs `retileCacheScatters` (tiles each scatter to one page of one head, 4 KiB); GPU needs nothing extra.
- **B. Size-aware dynamic packer** (`Dialect/Stream/Transforms/LayoutSlices.cpp`, `packDynamicSlicesByUnit`): dynamic slices that are constant multiples of the same size (e.g. of S / 32) share space. Runs only with the custom packer on.
- **C. Attention mask broadcast fold**: same change as 2A; either flag turns it on (`isAttentionMaskBroadcastFoldEnabled()`).
- Turns on the custom slice packer, which B needs.

### 2. FuseAttention, `--iree-dispatch-creation-fuse-attention`

- **A. Attention mask broadcast fold** (`DispatchCreation/ElementwiseOpFusion.cpp`): the `[S, S]` mask is no longer materialized as `[32, S, S]`.
- **B. Dynamic dim analysis** (`Codegen/Common/TensorDynamicDimAnalysis.cpp`): reads dynamic dims through `load_from_buffer`, so K / V / the mask's K2 dim are blocked to `[?, 32]` and attention gets the MMA config on GPU. No effect on CPU.
- **C. Causal tile skipping**:
  - Tagging (`DispatchCreation/FuseQKVChain.cpp`, `tagCausalAttentionMasks`): proves the mask is causal and sets `iree_linalg_ext.causal_mask`.
  - The tag is kept through dim collapsing (`Dialect/LinalgExt/Transforms/ReshapeFusion.cpp`).
  - The pass (`Codegen/Common/SkipCausalMaskedAttentionTiles.cpp`) lowers the K2 loop bound, in both the LLVMGPU and LLVMCPU pipelines. Hidden `--iree-codegen-causal-attention-tile-skip=false` switches it off for A/B runs.
- Does not turn on the custom slice packer.
- Outside the compiler: hand-written attention kernel `attn_fa.cu`, plugged in with `--iree-hal-substitute-executable-source`.

### 3. FuseMLP (gate/up + down), `--iree-dispatch-creation-fuse-mlp`

- **A. Dispatch formation** (`FormDispatchRegions.cpp`, `tryFuseMLPChain`): tags the root `iree_dispatch.special_mlp`. Needs at least 2 static rows (`kFuseMLPMinRows`), so decode is never fused.
- **B. Custom static slice packer** (`Dialect/Stream/Transforms/LayoutSlices.cpp`).
- **C. CPU codegen** (`Codegen/Common/RewriteMLPChainAsForall.cpp`): M → F → N loops with fixed tile sizes (TM ≤ 8, FC = 128, 8×16×16 vector tile).
- **D. GPU codegen** (`Codegen/Common/GPURewriteMLPChainAsForall.cpp`):
  - D1. Routing (`LLVMGPU/KernelConfig.cpp`): sends the chain to TileAndFuse. The chain is matched by structure, since `special_mlp` does not reach GPU codegen.
  - D2. Direct path (`emitDirectMLPKernel`, the default for prefill): 16 rows × 16 warps, `ldmatrix`, 16 B aligned loads, prefetch 2, gate/up k loop unrolled ×16.
  - D3. TileAndFuse MMA path (fallback when the direct path does not apply): virtual intrinsic `VNV_MMA_SYNC_F32_16x8x32_F16` (`IREEGPUEnums.td`, `IREEGPUAttrs.cpp`), unchecked `MapLoadOp` lowering (`TilingInterfaceImpl.cpp`), perfect-nest loop coalescing (`TileAndFuseUtils.cpp`), and the helper passes `GPUTileMLPDownPerWarp` and `GPUFuseMLPGateUpLoops`.
  - D4. Scalar path: used for decode (M = 1).
  - D5. `GPUVerifyDistribution` and `GPUReduceBankConflicts` skip functions tagged `iree_codegen.mlp_direct`.
- Outside the compiler: hand-written MLP kernel `mlp_fused*.cu`; the direct path now matches it.

### Shared

- Flag plumbing: `Utils/CustomFusion.{h,cpp}`. With no flags the compiler behaves as upstream IREE.
- `Codegen/Common/TileDispatchUsingForall.cpp` skips functions tagged `iree_codegen.mlp_chain_rewritten` / `iree_codegen.qkv_chain_rewritten`.
- `--iree-stream-custom-packing`: the custom slice packer alone.
- `--iree-dispatch-creation-fuse-conv-chain` (conv-chain fusion) is separate work and is not turned on by any of the flags above.

## CPU (LLVMCPU, x86)

- MLP fusion: [DocFuseMlp.md](DocFuseMlp.md).
- QKV fusion: [DocFuseQKV.md](DocFuseQKV.md).
- Attention: [DocFuseAttention.md](DocFuseAttention.md).

## GPU (CUDA, GX10 / GB10 sm_121)

- MLP fusion: [DocGPUFuseMlp.md](DocGPUFuseMlp.md) (direct-path kernel, prefill
  faster than baseline with the 16 / 32 / 64 MiB transient saving).
- QKV fusion: [DocGPUFuseQKV.md](DocGPUFuseQKV.md).
- Attention: [DocGPUFuseAttention.md](DocGPUFuseAttention.md).

### Prefill

GX10, `prefill_bs1` with S tokens, 1 run each, IREE-generated kernels only
(no substituted hand kernels). GPU time is the sum of all prefill kernels
(nsys); transient is the transient buffer vs baseline.

| Build | Prefill GPU time S=512 | S=2048 | Prefill transient vs baseline (S = 512 / 1024 / 2048) | Wall S=512 / 1024 / 2048 |
|---|---:|---:|---|---|
| Baseline | 224.2 ms | 1,741.0 ms | — (43.25 / 119.50 / 371.01 MiB) | 1.41 / 1.77 / 3.00 s |
| fuse-mlp | 176.7 ms | 1,528.7 ms | −17.19 / −34.38 / −68.76 MiB | 1.34 / 1.68 / 2.82 s |
| fuse-qkv | 238.0 ms | 1,785.8 ms | −22.69 / −77.38 / −282.76 MiB | 1.45 / 1.81 / 3.07 s |
| fuse-attention | 133.0 ms | 528.1 ms | −16.00 / −64.00 / −256.00 MiB | 1.26 / 1.43 / 1.79 s |
| all three | **99.7 ms** | **369.1 ms** | **−35.68 / −103.37 / −334.75 MiB** | **1.29 / 1.42 / 1.59 s** |

- All three together is the best on both goals: prefill GPU time −56% / −79%
  at S = 512 / 2048, wall −0.12 / −0.35 / −1.41 s, and transient down to
  7.57 / 16.13 / 36.26 MiB (−83% / −87% / −90%).
- fuse-attention gives most of the time: IREE's attention kernel gets the MMA
  config and the causal tile skip (attention alone was ~1.24 s of the 1.74 s at
  S = 2048). Its transient saving is the mask broadcast fold (the `[S, S]` mask
  is no longer materialized per head).
- fuse-mlp: the direct-path MLP kernel (prefill faster than baseline), −16 /
  −32 / −64 MiB for the gate/up intermediates; the extra ~1 / 2 / 4.8 MiB comes
  from the custom slice packer.
- fuse-qkv saves the most memory alone but is slower on GPU (+6% / +3% GPU
  time, see [DocGPUFuseQKV.md](DocGPUFuseQKV.md)); inside "all three" its
  memory saving overlaps with the mask fold that fuse-attention also does.
- Builds: `~/workspace/tmp/gx10-cuda/dec_{base,mlp2,qkv,attn,all2}.vmfb` (same
  builds as the decode table); GPU time: `~/workspace/tmp/gx10-hand/nsys_mlp.sh`;
  memory and wall: `bench_s_1run.sh` on the GX10 (`bench_s_flags.log`).

### Decode

GX10, decode of one token at position S, 1 run each. GPU time is the sum of all
decode kernels (nsys); transient is the transient buffer vs baseline.

| Build | Decode GPU time S=512 | S=2048 | Decode transient vs baseline (S = 512 / 1024 / 2048) | Wall S=512 / 1024 / 2048 |
|---|---:|---:|---|---|
| Baseline | 16.0 ms | 23.8 ms | — | 1.11 / 1.13 / 1.17 s |
| fuse-mlp | 14.8 ms | 23.3 ms | −10 / −10 / −10 KiB | 1.14 / 1.13 / 1.11 s |
| fuse-qkv | 14.9 ms | 23.8 ms | −44 / −76 / −140 KiB | 1.14 / 1.15 / 1.12 s |
| fuse-attention | 14.9 ms | 24.0 ms | −34 / −66 / −130 KiB | 1.13 / 1.15 / 1.15 s |
| all three | 14.9 ms | 24.4 ms | −44 / −76 / −140 KiB | 1.14 / 1.13 / 1.10 s |

- All three flags are safe for decode: GPU time is within noise of the
  baseline, and decode transient drops by up to 140 KiB. That is about 0.1% of
  the 132–144 MiB decode transient; the real savings are in prefill.
- fuse-mlp leaves decode's MLP unfused (fewer than 2 static rows,
  `kFuseMLPMinRows` in `FormDispatchRegions.cpp`). Fusing it ran on the
  per-thread path in one workgroup per layer: decode GPU time 160.9 / 172.0 ms
  at S = 512 / 2048 (10x) for 22–32 KiB more saving, so it was removed. With
  fuse-mlp alone, the −10 KiB comes from the custom slice packer.
- Decode wall time is ~1.1 s of fixed overhead in every build; differences of
  a few hundredths are single-run noise.
- Builds: `~/workspace/tmp/gx10-cuda/dec_*.vmfb` (`dec_base`, `dec_mlp2`,
  `dec_qkv`, `dec_attn`, `dec_all2`); decode GPU time:
  `~/workspace/tmp/gx10-hand/nsys_dec.sh`; memory and wall:
  `bench_decode_s_1run.sh` on the GX10 (`~/llama-mem-cuda/`).
