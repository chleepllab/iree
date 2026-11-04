# RMSNorm + QKV + RoPE fusion (`--iree-dispatch-creation-fuse-qkv`)

Fused RMSNorm + QKV + RoPE for prefill, behind the `--iree-dispatch-creation-fuse-qkv` flag (off by default; it also turns on the custom slice packer, like the other fusion flags). Only the LLVMCPU backend has the codegen side.

## Dispatches per layer (prefill, without the flag)

| # | Dispatch | What it does |
|---|---|---|
| 1 | `reduction_Dx2048` | RMSNorm |
| 2 | `matmul_Dx2048x2048` | Q projection |
| 3, 3 | `matmul_Dx512x2048` ×2 | K and V projections (separate dispatches) |
| 13–16 | `slow_memcpy` ×4 | Split Q/K into even and odd halves (stride-2 copies) for RoPE |
| 17 | `elementwise_broadcast_Dx8x2x32` | RoPE on K |
| 18, 21 | `elementwise_D_i64` | KV-cache index math |
| 19, 20, 19, 23 | transpose + scatter | Write K and V into the paged KV cache |
| 24 | `elementwise_broadcast_32xDx2x32` | RoPE on Q |
| 25 | `attention_8x4xDx64x64xD` | Whole attention: one `iree_linalg_ext.attention` op plus the output transpose |
| 26 | `matmul` | O projection |
| 1, 28–30 | | RMSNorm, then the MLP (the fuse-mlp target) |

RMSNorm + QKV + RoPE is spread over about 10 dispatches. This works like the MLP rewrite. The structure would be:

```
forall m (TM rows):
  r = rsqrt(mean(x[m]^2))                  // prologue: the row is needed whole anyway (K=2048)
  xn = x[m] * r * w_norm
  for n-tile over [Q|K|V] (3072 cols, tile = whole heads, multiple of 64):
    acc = xn @ Wqkv^T[n]
    if Q/K: rope(acc) in the epilogue      // even/odd pairs sit inside one 64-wide head
    store (or scatter K/V straight into the cache)
```

That would replace dispatches 1, 2, 3, 3, 13–16, 17 and 24 with one dispatch. At S=512 it would stop these intermediates from being written out:

| Tensor | Size |
|---|---|
| Normalized x | [512,2048] f16 = 2 MiB |
| Q | 2 MiB |
| K and V | 0.5 MiB each |
| Four even/odd half copies | about 2.5 MiB together |

## Implementation

**DispatchCreation** (`DispatchCreation/FuseQKVChain.{h,cpp}`, plus a hook in `FormDispatchRegions.cpp`):
- **RoPE rewrite:** it rewrites each prefill RoPE op so it reads `q`/`k` straight from the projection at columns `2j` and `2j+1`. That removes the four `slow_memcpy` dispatches. The two sequence dims (S/32, 32) become one, and outside the dispatch the result is reshaped back to its old shape at no cost.
- **Norm fusion:** it merges the norm's `extf`, rsqrt scale and weight multiply into one elementwise op, and the `extf` into the sum-of-squares reduction. So the norm no longer needs an [S, 2048] f32 temporary.
- **Grouping:** it puts the norm, the three matmuls with their truncs, and both RoPEs into one dispatch, rooted at the RoPE on Q. It takes no other producer or consumer fusion.
- **KV-cache transpose:** the transposes that turn K (the RoPE on K) and V (the V trunc) into the cache's per-page `[heads, 32, 64]` order, reached through reshapes only and feeding only a scatter, are tagged `iree_dispatch.qkv_cache_transpose`. Each one goes into its scatter's dispatch, so the transposed K/V copy (`[S/32,8,32,64]` f16, 512 KiB each) is never materialized and the separate transpose dispatch is gone.

**Codegen** (`Codegen/Common/RewriteQKVChainAsForall.cpp`, run in the LLVMCPU pipeline when the flag is on):
- One `scf.forall` runs over row tiles (TM). Each tile computes the norm once into [TM, 2048] f16.
- Q and K are computed one 64-wide head at a time, with RoPE applied as each head's epilogue. V is computed in 64-column chunks.
- TM is chosen so everything fits the stack limit: TM=4 at the default 32 KiB.
- After that, each op is tiled to vector size on its own, the same way the MLP rewrite does it.
- After dispatch formation, IREE's elementwise fusion usually folds the Q/K `trunc` into the RoPE, so the codegen matcher accepts the RoPE reading the matmul result either directly or through a `trunc`.
- In a KV-cache scatter dispatch with the tagged transpose fused in, the scatter's default tile (64 pages × one head × [32, 64], 256 KiB) has no smaller vector level, so the transposed tile would go on the stack whole and fail the 32 KiB limit. The pass sets the scatter's `distribution` and `vector_common_parallel` to one page of one head, `[1, …, 1, 32, 64]` (a 4 KiB stack tile).

The fused dispatch per layer (named `matmul_Dx2048x2048_f16xf16xf32`) has 7 loads (x, norm weight, Wq, Wk, Wv, cos, sin), 3 matmuls, 7 generics and 3 results: Q `[8,4,S,2,32]`, K `[S,8,2,32]`, V `[S,512]`.

Known limits:
- **Prefill only.** Decode lays out RoPE differently (S=1), so the matcher skips it and decode compiles as before.
- **RoPE stays scalar.** The linalg vectorizer doesn't accept the strided `2j`/`2j+1` reads, so RoPE becomes plain loops. That cost should be small next to the matmuls.
- **K and V still leave the QKV dispatch in [S, ...] order.** The transpose into cache order happens in the scatter (one 4 KiB tile at a time), not in the QKV dispatch, because attention reads K/V in [S, ...] order.

## Results

Host: AMD Ryzen 9 9950X (32 threads), `--device=local-task`, default 32 KiB stack (TM=4). Model: llama-3.2-1b-instruct f16, `prefill_bs1`, S=512, KV cache 128x524288 f16. Output written to `/dev/null`; base and fused runs alternated, 3 each. Transient buffer size is captured at runtime with the gdb breakpoint on `iree_hal_device_queue_alloca` described in [DocFuseMlp.md](DocFuseMlp.md#reproducing).

| LLaMA-3.2-1B f16, prefill S=512 | Baseline (no flags) | Custom fusion (`--iree-dispatch-creation-fuse-qkv`), TM=4 |
|---|---:|---:|
| Transient buffer (activations) | 45,353,344 B | 42,731,904 B (−2.5 MiB) |
| Max RSS (3 runs) | 2,872,064–2,872,832 KB | 2,868,736–2,870,016 KB (−2.9 MiB) |
| `HOST_LOCAL` peak | 265,551,872 B | 265,551,872 B |
| `DEVICE_LOCAL` peak | 134,222,088 B | 134,222,088 B |
| Minor page faults | 766,931–768,015 | 760,728–761,817 |
| Dispatch ops (prefill / whole module) | 363 / 822 | 219 / 678 |
| Number of executables (prefill / whole module) | 45 / 122 | 36 / 113 |
| Wall time (3 runs) | 1.99–2.04 s | 1.98–2.01 s |
| Output (1×512×128256 logits) | — | bit-identical |

- Transient buffer: −2,621,440 B (exactly 2.5 MiB, −5.8%). The intermediates this removes (normalized x, Q, K, the four even/odd half copies) are a few MiB per layer and are freed between layers; the arena is sized by other live ranges, so only part of them shows up.
- `HOST_LOCAL` peak is the 1×512×128256 f32 logits output (256.5 MiB); `DEVICE_LOCAL` is the caller's KV cache. Neither includes the transient arena, so both are unchanged.
- Wall time: no measurable change. 144 fewer dispatches per prefill do not show up at this size. An earlier comparison with `run_llama.sh` (3.38 s vs 2.72 s) was not a real speedup: those runs also time writing the 262 MB logits `.npy` file, which dominated the difference.
- Decode: unchanged (the matcher skips it); logits bit-identical.

## Why only 2.5 MiB

Prefill buffer sizes depend on S, so `LayoutSlices` packs the transient slices with `packDynamicSlicesConservatively`. That packer only lets two slices share space if their size is the same SSA value. Each size class gets its own region, and the arena is the sum over classes of (bins × size). A model of this packing, run on the IR before `iree-stream-layout-slices`, reproduces both measured arenas to the byte (45,353,344 B and 42,731,904 B).

Arena by size class with the flag on (S=512, 42,731,904 B = 40.75 MiB):

| Class | Contents | Arena |
|---|---|---:|
| [32,S,S] f16, 1 slice, live the whole prefill | Causal mask broadcast to 32 heads, once, by dispatch 2 | 16 MiB |
| [S,8192] f16 × 2 bins | MLP gate/up | 16 MiB |
| [S,2048] f16 × 3 bins | Residual, Q, attention output | 6 MiB |
| [S,512] f16 × 2 bins | K and V from the QKV dispatch | 1 MiB |
| [S/32,8,32,64] f16 × 2 bins | K and V transposed by dispatch 12, then copied into the cache by scatters 13/15 | 1 MiB |
| Rest | | 0.75 MiB |

The real peak of live bytes is 36.06 MiB, and it falls in the MLP (gate/up, mask, two hidden states), not in QKV. Shrinking QKV further only shows up if it removes a whole size class or bin.

Candidate changes, simulated with the same packing model (flag on, before A: 40.75 MiB). All three are implemented and turned on by `--iree-dispatch-creation-fuse-qkv` (see [Fusing the KV-cache transpose](#fusing-the-kv-cache-transpose-a), [Size-aware dynamic packer](#size-aware-dynamic-packer-b) and [Attention mask broadcast fold](#attention-mask-broadcast-fold-c)):

| Change | Arena | vs baseline | QKV-specific? |
|---|---:|---:|---|
| A. Fuse the K/V transpose (dispatch 12) into the cache scatters (13/15) — **done, measured 41,683,328 B** | 39.75 MiB | −3.5 MiB | yes |
| B. Size-aware dynamic packer: slices whose size is a multiple of S can share space (custom packer only) — **done, with A measured 38,339,584 B** | 36.56 MiB | −6.7 MiB | no |
| C. Don't materialize the 32-head mask: attention reads the [S,S] mask through a broadcast indexing map — **done** | 24.75 MiB | −18.5 MiB | no |
| A + C | 23.75 MiB | −19.5 MiB | |
| B + C (A + B + C) | 20.56 MiB (the lower bound) — **measured 21,562,368 B** | −22.7 MiB | |

B only runs inside the custom slice packer, which this flag turns on, so the baseline (no flags) is unchanged at 43.25 MiB. With B, the arena is set by the MLP peak (36.06 MiB of live bytes), so further QKV changes only save memory when combined with fuse-mlp.

## Fusing the KV-cache transpose (A)

Same host and method as Results; "before A" and "after A" are both with `--iree-dispatch-creation-fuse-qkv`, built from the same tree with and without this change. Runs alternated baseline / before / after, 3 rounds.

| LLaMA-3.2-1B f16, prefill S=512 | Baseline (no flags) | fuse-qkv, before A | fuse-qkv, after A |
|---|---:|---:|---:|
| Transient buffer (activations) | 45,353,344 B | 42,731,904 B (−2.5 MiB) | 41,683,328 B (−3.5 MiB) |
| Max RSS (3 runs) | 2,872,064–2,872,320 KB | 2,868,992–2,870,272 KB | 2,868,736–2,869,504 KB |
| Dispatch ops (prefill) | 363 | 219 | 187 |
| Wall time (3 runs) | 1.99–2.42 s | 1.96–2.05 s | 1.95–2.05 s |
| Prefill logits | — | bit-identical | bit-identical |
| Decode logits after this prefill (same KV cache) | — | bit-identical | bit-identical |

- Transient buffer: −1,048,576 B compared with before A. That is the `[S/32,8,32,64]` size class (2 bins × 512 KiB) disappearing, exactly as the packing model predicted.
- Wall time: no change. 32 fewer dispatches per prefill (one transpose for K and one for V per layer).
- Checking the cache: prefill logits don't depend on the cache writes, because attention reads K/V straight from the QKV dispatch. So the check runs prefill (S=512) and then decode at position 512 on the same cache in one process, through a small wrapper VM module (`--module=llama.vmfb --module=wrap.vmfb`), and compares the decode logits. Decode on an empty cache gives different logits, so this check does catch bad cache writes.

### Prefill length S and decode (x86)

Baseline (no flags) vs `--iree-dispatch-creation-fuse-qkv` (with A), default 32 KiB stack (TM=4). Prefill: `prefill_bs1` with S tokens. Decode: `decode_bs1`, one token at position S (seq_len S+1; KV cache zero-filled). Each cell is one gdb run for the transient size (the last `iree_hal_device_queue_alloca` of the call), `HOST_LOCAL` / `DEVICE_LOCAL` from `--print_statistics`, and 1 timed run. Cells read baseline → fused; "(both)" means equal.

| LLaMA-3.2-1B f16 on x86 | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 MiB → 39.75 MiB (−3.5 MiB) | 119.50 MiB → 112.50 MiB (−7 MiB) | 371.01 MiB → 357.01 MiB (−14 MiB) |
| Decode transient buffer | 132.33 MiB → 132.32 MiB (−10 KiB) | 136.37 MiB → 136.36 MiB (−10 KiB) | 144.43 MiB → 144.42 MiB (−10 KiB) |
| Prefill `DEVICE_LOCAL` peak | 128.00 MiB (both) | 128.01 MiB (both) | 128.02 MiB (both) |
| Decode `DEVICE_LOCAL` peak | 128.00 MiB (both) | 128.00 MiB (both) | 128.00 MiB (both) |
| Prefill `HOST_LOCAL` peak | 253.25 MiB (both) | 378.50 MiB (both) | 629.00 MiB (both) |
| Decode `HOST_LOCAL` peak | 0.24 MiB (both) | 0.24 MiB (both) | 0.24 MiB (both) |
| Prefill wall time (1 run) | 2.35 s → 2.13 s | 3.72 s → 3.70 s | 8.43 s → 8.05 s |
| Decode wall time (1 run) | 0.89 s → 0.87 s | 0.88 s → 0.88 s | 0.91 s → 0.91 s |

- Prefill saving is linear in S: 7 × S × 1 KiB (S=512: 3.5 MiB). Of that, S × 5 KiB is the fusion itself (2.5 MiB at S=512), and S × 2 KiB is A's transposed K/V class (2 bins × [S/32,8,32,64] f16).
- Decode: the QKV matcher skips decode, so its dispatches don't change. The −10,304 B comes from the custom slice packer, which the flag also turns on; decode's slice sizes are static, so they go through that packer.
- `HOST_LOCAL` / `DEVICE_LOCAL` don't include the transient arena on CPU, so they are unchanged.
- Wall time: no slowdown. The differences (−0.4 to 0 s) are within run-to-run noise for single runs.

## Size-aware dynamic packer (B)

`Dialect/Stream/Transforms/LayoutSlices.cpp`, `packDynamicSlicesByUnit`, inside `MY_PACKING` and run only when `isCustomPackingEnabled()` (the fusion flags turn it on). Before the conservative dynamic packer:

- Each dynamic slice size is written as `coeff × product(leaves)`. The analysis looks through constants, `arith.muli`, `arith.divui` by a constant that divides the coefficient exactly, and `util.assume.int`. So `S × 4096` and `(S / 32) × 32768` both become constants times the page count `S / 32`.
- Slices with the same leaves form a group. Each group is packed with the existing static packer (`chooseBestStaticSlicePacking`, best of three heuristics), using the coefficients as sizes, in units of the leaf product. The offsets become `offset_in_units × unit` at runtime. Scaling every size and offset by the same unit keeps disjoint slices disjoint for every S, and they stay aligned because the unit offsets and sizes are.
- Groups don't share space with each other: linear in S, quadratic in S (the mask) and static stay separate. Slices whose size can't be written this way still go to `packDynamicSlicesConservatively`.

Baseline (no flags) vs `--iree-dispatch-creation-fuse-qkv` with A and B, x86, default 32 KiB stack, same method as [Prefill length S and decode](#prefill-length-s-and-decode-x86), 1 run. Cells read baseline → fused; "(both)" means equal.

| LLaMA-3.2-1B f16 on x86 | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 MiB → 36.56 MiB (−6.69 MiB) | 119.50 MiB → 106.13 MiB (−13.38 MiB) | 371.01 MiB → 344.25 MiB (−26.76 MiB) |
| Decode transient buffer | 132.33 MiB → 132.32 MiB (−10 KiB) | 136.37 MiB → 136.36 MiB (−10 KiB) | 144.43 MiB → 144.42 MiB (−10 KiB) |
| Prefill `DEVICE_LOCAL` peak | 128.00 MiB (both) | 128.01 MiB (both) | 128.02 MiB (both) |
| Decode `DEVICE_LOCAL` peak | 128.00 MiB (both) | 128.00 MiB (both) | 128.00 MiB (both) |
| Prefill `HOST_LOCAL` peak | 253.25 MiB (both) | 378.50 MiB (both) | 629.00 MiB (both) |
| Decode `HOST_LOCAL` peak | 0.24 MiB (both) | 0.24 MiB (both) | 0.24 MiB (both) |
| Prefill wall time (1 run) | 2.03 s → 2.03 s | 4.60 s → 3.73 s | 8.10 s → 8.19 s |
| Decode wall time (1 run) | 0.89 s → 0.89 s | 0.88 s → 0.89 s | 0.88 s → 0.89 s |

- Transient at S=512: 38,339,584 B, which is what the packing model predicted for A + B (36.56 MiB; the lower bound is 36.06 MiB). B alone takes another 3.19 MiB off A's 39.75 MiB. The saving stays linear in S.
- Decode is unchanged by B: its slices are static and already go through the custom static packer.
- Prefill and decode logits after a prefill on the same cache (S=512), and prefill logits at S=2048, are bit-identical to baseline. The baseline vmfb is byte-identical with and without the change, since B only runs under the custom packer.
- Wall time: no slowdown. Other sessions loaded the machine during these runs (load average 5–8). Baseline at S=1024 measured 4.38–4.60 s in three runs, against 3.72 s on a quieter machine earlier, and A + B measured 3.57–3.73 s, so the S=1024 gap is mostly noise.

## Attention mask broadcast fold (C)

`DispatchCreation/ElementwiseOpFusion.cpp`. Under `--iree-dispatch-creation-fuse-qkv` (and also `--iree-dispatch-creation-fuse-attention`), a broadcast that produces an attention op's mask may fold into the mask's indexing map even when it broadcasts away whole dim groups. Upstream refuses that for every operand, but the attention dim groups are inferred from Q/K/V/O only, so it is safe for the mask. The `[32, S, S]` f16 mask (16 MiB at S=512, live for the whole prefill) is no longer materialized; attention reads the `[S, S]` mask directly. This was first written for the GPU attention work as the separate flag `--iree-dispatch-creation-fuse-attention-mask-broadcast`, which is now removed: the fold is part of fuse-qkv.

x86, S=512, `--iree-dispatch-creation-fuse-qkv` (A + B + C): transient 45,353,344 B → 21,562,368 B (43.25 → 20.56 MiB, −22.69 MiB), the packing model's lower bound. Prefill logits, and decode logits after a prefill on the same cache, are bit-identical to baseline. The baseline vmfb is unchanged.

## Reproducing

`QKV=0 ~/workspace/run_llama.sh 512` / `QKV=1 ~/workspace/run_llama.sh 512` (the flag is on by default). For clean timings, run `iree-run-module` with `--output=@/dev/null` under `/usr/bin/time -f "%e %M %R"`.


## Differences from the plan

Three parts don't match what was built:
- The pseudocode shows one combined Wqkv loop over 3,072 columns with K/V scattered straight into the cache. The implementation keeps Wq, Wk and Wv separate: it loops over Q heads, then K heads, then V chunks, and leaves the KV-cache scatters as their own dispatches (with the transpose into cache order fused into them).
- "Stops these intermediates from being written out": Q, K and V still leave the fused dispatch as its three outputs. Q and K now leave with RoPE already applied.
- The table's sizes add up to about 7.5 MiB, but the measured transient saving is 2.5 MiB, as the Results section shows.
