# Attention on GPU: fusion vs. MMA (`--iree-dispatch-creation-fuse-attention`)

`--iree-dispatch-creation-fuse-attention` turns on all three steps of [the plan](#how-iree-lowers-attention-compared-with-the-hand-kernel):
- the `load_from_buffer` dim analysis, which blocks K / V / the mask's K2 dim so that attention gets the MMA config (steps 1–2);
- causal tile skipping (step 3);
- the attention mask broadcast fold (also on under `--iree-dispatch-creation-fuse-qkv`).

It replaces two earlier flags: `--iree-codegen-dim-analysis-load-from-buffer` (the dim analysis) and `--iree-dispatch-creation-fuse-attention-mask-broadcast` (the mask fold). Both are removed. Commands below use the new flag. The results in the sections before [Causal tile skipping](#causal-tile-skipping) were measured with the old flags, i.e. without the causal skip. The new flag adds the skip, which makes IREE's attention kernel faster (that section has the numbers); the hand attention kernel replaces that dispatch, so it is not affected.

The dim analysis part of the flag (then `--iree-codegen-dim-analysis-load-from-buffer`) does not change any fusion. It
lets the attention kernel IREE already fuses use tensor cores.

## Results: baseline vs `--iree-dispatch-creation-fuse-attention` (GX10)

GX10 (GB10, CUDA), baseline (no flags) vs `--iree-dispatch-creation-fuse-attention` alone (IREE's own attention kernel, no hand kernels), 1 run each. Builds: `ga_base.vmfb` / `ga_fuse.vmfb` in `~/workspace/tmp/gx10-cuda/` (`ga_base` is byte-identical to the earlier `gq_base`); runs: `bench_s_1run.sh` / `bench_decode_s_1run.sh`, logs `ga_prefill.log` / `ga_decode.log` in `~/llama-mem-cuda`. Prefill: `prefill_bs1` with S tokens. Decode: `decode_bs1`, one token at position S, KV cache zero-filled. Cells read baseline → fused; "(both)" means equal.

| LLaMA-3.2-1B f16 on GX10 (CUDA) | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 MiB → 27.25 MiB (−16 MiB) | 119.50 MiB → 55.50 MiB (−64 MiB) | 371.01 MiB → 115.01 MiB (−256 MiB) |
| Decode transient buffer | 132.33 MiB → 132.30 MiB (−34 KiB) | 136.37 MiB → 136.30 MiB (−66 KiB) | 144.43 MiB → 144.30 MiB (−130 KiB) |
| Prefill `DEVICE_LOCAL` peak | 2781.76 MiB → 2765.76 MiB (−16 MiB) | 2983.27 MiB → 2919.27 MiB (−64 MiB) | 3485.28 MiB → 3229.28 MiB (−256 MiB) |
| Decode `DEVICE_LOCAL` peak | 2617.84 MiB → 2617.80 MiB (−34 KiB) | 2621.87 MiB → 2621.80 MiB (−66 KiB) | 2629.93 MiB → 2629.81 MiB (−130 KiB) |
| Prefill `HOST_LOCAL` peak | 253.25 MiB (both) | 378.50 MiB (both) | 629.00 MiB (both) |
| Decode `HOST_LOCAL` peak | 74.01 MiB (both) | 74.01 MiB (both) | 74.01 MiB (both) |
| Prefill wall time (1 run) | 1.35 s → 1.26 s | 1.75 s → 1.43 s | 2.95 s → 1.78 s |
| Decode wall time (1 run) | 1.15 s → 1.12 s | 1.08 s → 1.08 s | 1.09 s → 1.09 s |

- Transient sizes are byte-identical to x86 ([DocFuseAttention.md](DocFuseAttention.md)). The prefill saving is the `[32, S, S]` f16 broadcast mask (32 × S² × 2 B), the decode saving decode's `[32, 1, P]` mask (P = 544 / 1056 / 2080 key positions).
- On CUDA the transient arena is part of `DEVICE_LOCAL`, so `DEVICE_LOCAL` drops by the same amount.
- Prefill wall time: −40% at S=2048 (2.95 → 1.78 s). Most of it is attention moving from scalar FMAs to MMA (the dim-analysis part): IREE's attention kernel was 1,236 ms of GPU time at S=2048 without the flag. The mask fold and causal tile skipping add to that ([Causal tile skipping](#causal-tile-skipping) has the kernel times).
- Decode is unchanged within noise. Its one query sits at position S, so no key tile lies past it and causal skipping drops nothing; only the small mask fold applies.

### IREE-generated vs hand attention kernel

GX10 prefill, 1 run each, all three in one session (log `gah_prefill.log` in `~/llama-mem-cuda`; builds `ga_base` / `ga_fuse` / `ga_hand` in `~/workspace/tmp/gx10-cuda/`). "Hand kernel" is `--iree-dispatch-creation-fuse-attention` with the attention dispatch (`dispatch_24`) replaced by the hand flash-attention kernel (`--iree-hal-substitute-executable-source=prefill_bs1$async_dispatch_24=~/workspace/tmp/gx10-hand/attn/attn_d24.mlir`; same 6-constant interface).

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Baseline | 1.35 s | 1.77 s | 2.94 s |
| IREE-generated (fuse-attention) | 1.27 s | 1.44 s | 1.69 s |
| Hand kernel (fuse-attention + hand attention kernel) | 1.32 s | 1.43 s | 1.73 s |
| Transient buffer vs baseline | −16 MiB | −64 MiB | −256 MiB |

- With fuse-attention, IREE's own attention kernel (MMA + causal tile skipping + mask fold) is as fast as the hand kernel in wall time: within ±0.05 s at every S. The hand kernel is still faster in isolation (8.0 vs 14.3 ms of GPU time at S=2048), but the difference is too small to show in wall time.
- The transient saving is the same with either kernel: it comes from the mask fold in dispatch creation.

## Fusion (already there without the flag)

IREE has always built attention as one dispatch: `online_attention` is tiled
along K2 into one `scf.for`, with the ~13 generics (QK^T, scale + mask, row
max, exp2, row sum, rescale, PV) decomposed inside that loop. That is flash
attention: scores and P never leave the K2 tile, with or without the flag.

```
scf.forall (head, q rows)                         // workgroups
  scf.for k2 = 0 .. S step 256                    // K2 loop (partial_reduction tile)
    s    = Q · K[k2]^T              (generic: matmul)
    s    = s * scale + mask[k2]     (generics)
    mnew = max(m, rowmax(s))        (generics: reduction, max)
    p    = exp2(s - mnew)           (generic)
    l    = l * exp2(m - mnew) + rowsum(p)
    o    = o * exp2(m - mnew) + p · V[k2]   (generic: matmul)
  out = o / l
```

## What the flag fixes

Inside that fused loop, the two matmuls (QK^T and PV) ran as scalar FMAs:
IREE's MMA attention config (`setAttentionIntrinsicBasedVectorDistributionConfig`
in `LLVMGPU/KernelConfig.cpp`) bails when the innermost M or K2 dim is
dynamic, and prefill's sequence length S is dynamic.

S is known to be a multiple of 32 (`util.assume.int ... udiv = 32`), and
`BlockDynamicDimensions` would split it into `[S/32, 32]`, making the inner
dims static. But `TensorDynamicDimAnalysis` recorded dims only for
`dispatch.tensor.load` and `tensor.empty` results, not for
`iree_codegen.load_from_buffer`, which is how the GPU pipeline loads tensors
by then. So Q, K, V and the mask never got divisibility info and were never
split.

With the flag, the analysis takes a `load_from_buffer` result's dynamic dims
from its buffer (the `hal.interface.binding.subspan`'s `{%d}`). Q, K, V and
the mask are then split into `[S/32, 32]` (K / V `[?, 32, 8, 64]`, mask
`[?, 32, ?, 32]`), the inner dims are static, and the attention takes the MMA
config (`NV_MMA_SYNC_F32_16x8x16_F16`). The loop structure is the same; only
how its body is compiled changes:

| | Loop structure (13 generics in one K2 loop) | QK^T / PV matmuls | Attention kernel, S=2048 (16 layers) |
|---|---|---|---|
| Without the flag | same | scalar FMAs | 1,236 ms |
| With the flag | same | `mma.sync` | 39.4 ms (23.7 ms with the mask broadcast fold) |
| Hand-written kernel (`attn_fa.cu`), for comparison | same idea | `mma.sync`, fully masked tiles skipped | 8.0 ms |

## Not specific to attention

The flag is an analysis fix that applies to every dispatch whose tensors are
loaded straight from buffers. On its own (no fusion flags) it takes the plain
baseline prefill from 3.02 s to 1.83 s at S = 2048 on the GX10.

The change is in `Codegen/Common/TensorDynamicDimAnalysis.cpp`; the flag
(default off, upstream behavior) is in `Utils/CustomFusion.{h,cpp}`. Without
the flag, builds are byte-identical to before.

Measurements, the mask broadcast fold and the hand-written attention kernel:
[Attention: mask broadcast fold and hand-written kernel](#attention-mask-broadcast-fold-and-hand-written-kernel)
and [IREE's own attention kernel on MMA](#irees-own-attention-kernel-on-mma---iree-dispatch-creation-fuse-attention) below.

## Attention: mask broadcast fold and hand-written kernel

After the MLP, attention was most of prefill: at S = 2048, 1,234.8 ms of the
1,751.9 ms baseline GPU time. Two changes, both measured on the GX10:

1. **Mask broadcast fold** (compiler, then `--iree-dispatch-creation-fuse-attention-mask-broadcast`, now part of `--iree-dispatch-creation-fuse-attention` and `-fuse-qkv`):
   the `[S, S]` causal mask is no longer materialized per head. Transient
   buffer −16 / −64 / −256 MiB more at S = 512 / 1024 / 2048, output
   bit-identical.
2. **Hand-written flash attention kernel** on `mma.sync`, swapped in like the
   MLP kernel: attention 1,234.8 → 8.0 ms at S = 2048.

GX10 prefill, 1 run each:

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Baseline (no fusion), wall | 1.34 s | 1.75 s | 2.98 s |
| Hand MLP (previous best), wall | 1.34 s | 1.66 s | 2.77 s |
| + mask broadcast fold, wall | 1.35 s | 1.59 s | 2.68 s |
| + hand attention, wall | **1.22 s** | **1.33 s** | **1.51 s** |
| Transient buffer: baseline | 43.25 MiB | 119.50 MiB | 371.01 MiB |
| Transient buffer: hand MLP | 27.25 MiB (−16) | 87.50 MiB (−32) | 307.01 MiB (−64) |
| Transient buffer: + mask fold (and hand attention) | **11.25 MiB (−32)** | **23.50 MiB (−96)** | **51.01 MiB (−320)** |
| `DEVICE_LOCAL` peak: baseline → final | 2,916.9 → 2,883.3 MB | 3,128.2 → 3,027.5 MB | 3,654.6 → 3,319.0 MB |

GPU kernel time, nsys, 16 layers, one session:

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Attention, baseline | 95.0 ms | 316.9 ms | 1,234.8 ms |
| Attention, mask fold (IREE kernel) | 93.6 ms | 313.3 ms | 1,222.8 ms |
| Attention, hand kernel | **0.7 ms** | **2.3 ms** | **8.0 ms** |
| Fused MLP (hand kernel) | 35.2 ms | 77.7 ms | 123.6 ms |
| Total GPU, baseline | 225.4 ms | 576.8 ms | 1,751.9 ms |
| Total GPU, final | **81.8 ms** | **170.0 ms** | **310.6 ms** |

Logits vs baseline (final build): top-1 100% / 99.90% / 99.76%, mean |Δ|
0.0024 / 0.0020 / 0.0019 at S = 512 / 1024 / 2048 (hand MLP alone: 100% /
100% / 99.71%, mean |Δ| 0.0019 / 0.0017 / 0.0017). See
[Numerics](#numerics) for the one larger difference.

The remaining ~1.2 s of wall time at every S is outside the GPU kernels
(311 ms of GPU time at S = 2048).

### Why the IREE attention kernel was slow

The attention dispatch (`attention_8x4xDx64x64xD`) goes to `VectorDistribute`
but not to its MMA config: `setAttentionIntrinsicBasedVectorDistributionConfig`
(`LLVMGPU/KernelConfig.cpp`) bails when the innermost M or K2 dim is dynamic
("TODO: Add dynamic shape support for inner most dims"), and prefill's S is
dynamic. The fallback `setAttentionReductionConfig` uses no intrinsic: the
kernel has 0 `mma` instructions, all scalar FMAs. At S = 2048 that is about
0.45 TFLOP/s (550 GFLOP in 1.23 s, full S × S). `NV_MMA_SYNC_F32_16x8x16_F16`
itself is supported by the MMA config (it has a lane distribution mapping);
only the dynamic-shape bail blocks it.


### Mask broadcast fold

Moved to [DocGPUFuseQKV.md](DocGPUFuseQKV.md#mask-broadcast-fold):
The mask broadcast fold (then `--iree-dispatch-creation-fuse-attention-mask-broadcast`) stops the `[S, S]`
causal mask from being materialized once per head (transient −16 / −64 /
−256 MiB at S = 512 / 1024 / 2048, output bit-identical). It also renumbers
the dispatches: attention becomes `dispatch_24` and the fused MLP
`dispatch_27`.

### Hand-written attention kernel

`~/workspace/tmp/gx10-hand/attn/attn_fa.cu`, FlashAttention-2 style, for this
dispatch's ABI (constants = 6, bindings = 2): Q `[32, S, 64]`, K/V
`[S, 8, 64]` (GQA, 4 query heads per KV head), mask `[S, S]`, out
`[S, 32, 64]` f16; scale 0.125; S a multiple of 32.

```
grid (ceil(S / 64), 32 heads), 128 threads = 4 warps        // 46,080 B shared, 136 regs
  Q[64 rows] -> shared -> A fragments in registers           // warp w: rows 16w : +16
  for each key tile j (64 keys), K/V double-buffered via cp.async:
    skip the tile if its mask is -inf for every row of the CTA   // causal: ~half the tiles
    s = Q K^T                     [16, 64] f32 per warp       // 16 mma.sync
    s = s * scale * log2 e + mask * log2 e; online softmax (exp2, f32)
    o = o * alpha + P V           [16, 64] f32 per warp       // P from s registers as f16, 16 mma.sync
  out = o / l -> f16
```

- **Tile skip.** The decision reads the mask (CTA-wide `__syncthreads_or`), so
  it is exact for any mask: a tile that is −inf for every row adds exactly 0
  (column 0 is never masked, so each row's running max is finite from the
  first tile on).
- **Standalone check** (`test_attn.cu`, random Q/K/V, causal + padding mask,
  double-precision CPU reference): max |err| 3.4–3.5e-4 at S = 32, 96 (90
  valid keys), 256, 512 (500 valid keys). Standalone time 0.042 / 0.135 /
  0.508 ms per call at S = 512 / 1024 / 2048, ~25–34 TFLOP/s on the causal
  FLOPs.

### Numerics

Every logits row matches the baseline about as closely as the hand-MLP build
does, except row 1 (second token): max |Δ| 0.51 there, 0.06 elsewhere. IREE's
attention decomposition multiplies f16 Q by scale · log2 e and rounds it to
f16 before QK^T (`AggregatedOpInterfaceImpl.cpp`); the hand kernel keeps Q
exact and scales in f32. A variant that copies IREE's rounding
(`-DIREE_QSCALE`) cuts row 1 to 0.21 and changes nothing else, so the
difference is rounding in the baseline at a position with large attention
scores, not an error in the kernel (which matches the double-precision
reference to 3.5e-4).

### Building the fastest configuration

```
cd ~/workspace/tmp/gx10-cuda
./compile.sh <name> --iree-dispatch-creation-fuse-mlp=true \
  --iree-dispatch-creation-fuse-attention=true \
  "--iree-hal-substitute-executable-source=prefill_bs1\$async_dispatch_27=$HOME/workspace/tmp/gx10-hand/h2p3f_d27.mlir" \
  "--iree-hal-substitute-executable-source=prefill_bs1\$async_dispatch_24=$HOME/workspace/tmp/gx10-hand/attn/attn_d24.mlir"
```

`attn_d24.mlir` points at `attn_fa.ptx`
(`nvcc -O3 -arch=sm_121 -ptx attn_fa.cu`, CUDA 13.0) with grid
`(ceil(S / 64), 32, 1)`, 128 threads and 46,080 B. The substitutions are keyed
by dispatch name, so any change to dispatch formation can renumber them; check
the `NOTE: ... substituted` lines in the compile log.

Files: x86 `~/workspace/tmp/gx10-hand/attn/` (`attn_fa.cu`, `test_attn.cu`,
`attn_fa.ptx`, `attn_d24.mlir`, `attn_fa_q.ptx` / `attn_d24_q.mlir` = IREE
rounding variant); `~/workspace/tmp/gx10-cuda/` (`best`, `mask1`, `attn1`,
`attn1q` `.vmfb`s); GX10 `~/attn/` (`test_attn`, `npycmp` logits compare),
`~/llama-mem-cuda/nsys_attn.sh` (attention / MLP / total kernel time).

## IREE's own attention kernel on MMA (`--iree-dispatch-creation-fuse-attention`)

### How IREE lowers attention, compared with the hand kernel

The generics have to sit inside a loop over K2, and IREE already builds that loop, so fusion isn't what's missing. What's missing is that neither matmul inside the loop uses MMA.

`attention` becomes `online_attention`, and its lowering config tiles it like this:

```
workgroup         = [1, 1, 1, 1, 16, 0, 0]   // 16 query rows of one head per workgroup
partial_reduction = [0, 0, 0, 0, 0, 0, 256]  // K2 (key) dim tiled to a serial scf.for of 256
```

After tiling, the op is decomposed into generics inside that loop body:

```
scf.forall (head, 16 q rows)                      // workgroups
  scf.for k2 = 0 .. S step 256                    // the K2 loop
    s    = Q · K[k2]^T              (generic: matmul)
    s    = s * scale + mask[k2]     (generics)
    mnew = max(m, rowmax(s))        (generics: reduction, max)
    p    = exp2(s - mnew)           (generic)
    l    = l * exp2(m - mnew) + rowsum(p)
    o    = o * exp2(m - mnew) + p · V[k2]   (generic: matmul)
  out = o / l
```

That is the same flash-attention structure as the hand kernel. Scores, max, sum and P exist only per K2 tile, in registers or shared memory, never as an [S, S] buffer.

| | IREE (before the flag) | Hand kernel |
|---|---|---|
| Loop over K2 with online softmax | yes (tile 256) | yes (tile 64) |
| QK^T and PV matmuls | scalar FMAs, 0 `mma` | `mma.sync` |
| P passed straight from QK accumulators to PV operands | no | yes |
| Skip fully masked K2 tiles | no | yes (about half for causal) |
| K/V tiles double-buffered via `cp.async` | no | yes |

The ~150× gap was almost entirely the second row. The MMA attention config (`setAttentionIntrinsicBasedVectorDistributionConfig`) bails out because K2's size is dynamic. M is fine: `BlockDynamicDimensions` already splits it into `[?, 32]`. That pass is meant to block both Q and K, but the dump showed K and V still `[?, 8, 64]`, with K2 a single dynamic dim. Once K2 is also split into `[?, 32]`, with its inner part static like M's, the existing MMA config applies, and VectorDistribute puts both matmuls on `mma.sync`, keeping P in registers between them.

Plan:

1. Find why K isn't blocked, and make K, V and the mask's K2 dim block to `[?, 32]`. **Done:** `TensorDynamicDimAnalysis` didn't see the dims of `iree_codegen.load_from_buffer` results. It was a gap in the analysis, not the 64-bit offsets suspected at first. See the fix below.
2. Confirm the attention gets the MMA config, check correctness, and measure against the hand kernel's 8 ms at S=2048. **Done:** 39.4 ms, or 23.7 ms with the mask fold (table below).
3. Causal tile skipping, the remaining difference, isn't something IREE did. **Done:** see [Causal tile skipping](#causal-tile-skipping).

### The fix

IREE already builds attention as flash attention: `online_attention` is
tiled along K2 into a serial `scf.for` (partial_reduction tile 256) and
decomposed inside it into the ~13 generics (QK^T, scale + mask, row max, exp2,
row sum, rescale, PV), so scores, max, sum and P exist per K2 tile only. What
was missing is MMA for the two matmuls: `setAttentionIntrinsicBasedVectorDistributionConfig`
(`LLVMGPU/KernelConfig.cpp`) bails when the innermost M or K2 dim is dynamic,
and the fallback reduction config is scalar FMAs (0 `mma`, see
[Why the IREE attention kernel was slow](#why-the-iree-attention-kernel-was-slow)).

`BlockDynamicDimensions` is meant to split S into `[S/32, 32]` for Q and K
(S has `udiv = 32`), which makes the inner M / K2 dims static. Q's M got
blocked only indirectly (through the elementwise ops after attention); K, V
and the mask's K2 dim never did: `TensorDynamicDimAnalysis` records dims for
`dispatch.tensor.load` and `tensor.empty` results but not for
`iree_codegen.load_from_buffer`, which is what the GPU pipeline has by then,
so no tensor loaded from a binding had divisibility info.

The fix (`Codegen/Common/TensorDynamicDimAnalysis.cpp`, behind
then `--iree-codegen-dim-analysis-load-from-buffer`, now `--iree-dispatch-creation-fuse-attention`, default off; flag in
`Utils/CustomFusion.{h,cpp}`) takes a `load_from_buffer` result's dynamic dims
from its buffer (`IREE::Util::findDynamicDims`, i.e. the subspan's `{%d}`).
Then K / V become `[?, 32, 8, 64]`, the mask `[?, 32, ?, 32]`, and the
attention gets the MMA config (`NV_MMA_SYNC_F32_16x8x16_F16`, VectorDistribute).
Without the flag the builds are byte-identical to before.

GX10, 1 run each, one session (nsys: 16 layers):

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Attention kernel, IREE (no flag) | 95.0 ms | 316.9 ms | 1,236.3 ms |
| Attention kernel, IREE + flag | 2.6 ms | — | 39.4 ms |
| Attention kernel, IREE + flag + mask fold | **1.7 ms** | — | **23.7 ms** |
| Attention kernel, hand (`attn_fa.cu`) | 0.7 ms | 2.3 ms | 8.0 ms |
| Wall, baseline (no flags) | 1.37 s | 1.78 s | 3.02 s |
| Wall, baseline + flag only | 1.29 s | 1.45 s | 1.83 s |
| Wall, hand MLP + mask fold + **IREE attention** + flag | **1.21 s** | **1.37 s** | **1.55 s** |
| Wall, hand MLP + mask fold + hand attention | 1.26 s | 1.37 s | 1.51 s |
| Logits vs baseline, IREE attention build: top-1 | 100% | 99.90% | 99.76% |
| Logits vs baseline, baseline + flag: top-1 | 100% | 100% | 99.71% |

- IREE's attention goes from ~150x to ~3x the hand kernel. The remaining gap
  is likely the hand kernel's skipping of fully masked K2 tiles (about half
  for the causal mask; IREE computes every tile) plus tile sizes and K/V
  prefetch. Wall time with IREE's attention is within 0.04 s of the hand
  kernel.
- The flag alone (no fusion) takes the plain baseline from 3.02 to 1.83 s at
  S = 2048.
- With the flag, the dispatch numbering of the mask-fold build is unchanged
  (MLP still `dispatch_27`).

Build (hand MLP + mask fold + IREE attention):

```
./compile.sh <name> --iree-dispatch-creation-fuse-mlp=true \
  --iree-dispatch-creation-fuse-attention=true \
  "--iree-hal-substitute-executable-source=prefill_bs1\$async_dispatch_27=$HOME/workspace/tmp/gx10-hand/h2p3f_d27.mlir"
```

(`~/workspace/tmp/gx10-cuda/iree_att.vmfb`; baseline + flag: `base_new.vmfb`.)

### Causal tile skipping

Under `--iree-dispatch-creation-fuse-attention`, the K2 loop of a causal attention stops at the last key tile that still has an unmasked element.

- **Dispatch creation** (`DispatchCreation/FuseQKVChain.cpp`, `tagCausalAttentionMasks`, called from `FormDispatchRegions`). The mask is a generic computing `select(cond, -inf, 0)`. The pass checks that `cond` has, in its `ori` tree, a `cmpi` that reduces to `lin(K2) - lin(M) + delta > 0`. Here `lin` is the row-major position over the M (resp. K2) dims the mask's map reads, in increasing order; the query-head dim is in M but not in the mask, so it is left out. Index values are read as linear forms over `linalg.index`, looking through `index_cast`/`ext`, `addi`/`subi` and `muli` by a constant. If the check passes, the attention gets `iree_linalg_ext.causal_mask = {delta}`. llama's mask, `(k > q) || (k >= seq_len)`, gives delta = 0.
- The tag doesn't depend on the dims' shape. Collapsing or expanding dims within the M or K2 groups doesn't change `lin`. `collapseOpIterationDims` (`LinalgExt/Transforms/ReshapeFusion.cpp`), which builds a new attention op, copies this one attribute; the expansion path clones the op, which keeps it.
- **Codegen** (`Codegen/Common/SkipCausalMaskedAttentionTiles.cpp`, in the GPU VectorDistribute pipeline and the CPU LinalgExtTileAndVectorize pipeline, right before `DecomposeAttention`). It works on the tiled `online_attention` inside the scf.for over K2, and only when the loop carries nothing but the attention's state. q_max is the last query position of the tile, from the Q slice; k_min = k0 + a·iv is the first key position, from the K slice. Both are traced through the extract_slice chain back to the tensors loaded from bindings, and the strides come from those tensors' full sizes. The loop bound becomes `min(ub, max(lb, floordiv(q_max - delta - k0, a) + 1))`. For llama that is `%q_block + 1` of `S / 32` key blocks.
- **Why the result is unchanged:** a skipped tile is entirely `-inf`. Its probabilities are `exp2(-inf) = 0`. The running max is unchanged, because it starts at `-FLT_MAX`, a finite value. So the rescale factor is `exp2(0) = 1`. Skipping the tile therefore leaves every loop-carried value exactly as it was.
- `--iree-codegen-causal-attention-tile-skip=false` (hidden, default true) turns the codegen part off for A/B comparisons.

GX10, fuse-mlp + fuse-qkv + fuse-attention (then `--iree-codegen-dim-analysis-load-from-buffer`; IREE's own attention), skip off vs on, 1 run each (`cs_off.vmfb` / `cs_on.vmfb` in `~/workspace/tmp/gx10-cuda/`; nsys via `nsys_attn.sh`, log `cs_prefill.log` in `~/llama-mem-cuda`):

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Attention kernel (nsys, 16 layers), off → on | 1.5 → 1.3 ms | 5.4 → 4.0 ms | 20.9 → 14.3 ms |
| Total GPU (nsys), off → on | 100.4 → 99.8 ms | 206.2 → 204.6 ms | 374.2 → 369.8 ms |
| Prefill wall time, off → on | 1.24 → 1.18 s | 1.38 → 1.39 s | 1.62 → 1.59 s |
| Prefill logits, on vs off | bit-identical | bit-identical | bit-identical |
| Hand attention kernel (for reference) | 0.7 ms | 2.3 ms | 8.0 ms |

- The attention kernel takes 32% less time at S=2048, even though about half the K2 iterations are dropped. Each query block still runs its diagonal tile, and the per-workgroup setup and epilogue don't shrink.
- IREE's attention is now 1.8× the hand kernel at S=2048, down from about 2.6×. The remaining gap is tile sizes and K/V double-buffering (the comparison table above).
- Wall time changes little because attention is now a small part of prefill: 14 ms of about 370 ms of GPU time at S=2048.

#### On CPU (x86)

The pass also runs in the LLVMCPU `LinalgExtTileAndVectorize` pipeline, right before `DecomposeAttention`. The tagging and the mask fold are target-independent, so `--iree-dispatch-creation-fuse-attention` turns on the same steps there. The dim-analysis part has no effect on CPU, since that pipeline doesn't use `load_from_buffer` at that stage. The CPU K2 loop differs from the GPU one in two ways, and the pass accepts both:
- it yields the attention results through `tensor.insert_slice` into its own iter_args;
- the Q offset is an `affine.apply` of the workgroup offsets inside the loop. Such pure, loop-invariant values are cloned in front of the loop.

For llama on CPU, the dims aren't blocked: one 64-row workgroup tile has 4-row vector tiles, and K2 runs in steps of 16. The bound becomes `min(S, q_max + 1)`.

x86 (Ryzen 9 9950X, `--device=local-task`), 1 run each: baseline (no flags) vs `--iree-dispatch-creation-fuse-attention` with the skip off (`--iree-codegen-causal-attention-tile-skip=false`) and on. Attention time is the dispatch's `host_execute` total over the 16 layers from `--device_profiling_mode=host-execution` + `iree-profile statistics`; `tile_sum` is the summed time of all its workgroup tiles.

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Attention dispatch, baseline | 176.6 ms | 735.5 ms | 2,934.3 ms |
| Attention dispatch, skip off → on | 175.3 → 108.5 ms | 718.4 → 545.8 ms | 2,889.2 → 2,219.5 ms |
| Attention `tile_sum`, skip off → on | 2,697 → 1,337 ms | 10,899 → 5,485 ms | 44,318 → 23,020 ms |
| Prefill wall time, baseline | 2.72 s | 4.54 s | 8.13 s |
| Prefill wall time, skip off → on | 2.07 → 2.01 s | 3.71 → 3.53 s | 8.03 → 7.36 s |
| Prefill logits | bit-identical to baseline (S=512, off and on) | | |

- The total work (`tile_sum`) halves, as expected. The dispatch's wall time drops less (−38% at S=512, −24% at S=1024 and S=2048) because the work is now uneven across workgroups. The last query tiles still do all S keys, and they decide when the dispatch ends.
- The logits stay bit-identical even with CPU's polynomial `exp2`: it gives exactly 0 for `-inf` and exactly 1 for 0.
- The baseline S=512 wall time (2.72 s) is probably a first-run warmup outlier; earlier baseline runs were 2.0–2.4 s.
