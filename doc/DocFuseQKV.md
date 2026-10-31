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

**Codegen** (`Codegen/Common/RewriteQKVChainAsForall.cpp`, run in the LLVMCPU pipeline when the flag is on):
- One `scf.forall` runs over row tiles (TM). Each tile computes the norm once into [TM, 2048] f16.
- Q and K are computed one 64-wide head at a time, with RoPE applied as each head's epilogue. V is computed in 64-column chunks.
- TM is chosen so everything fits the stack limit: TM=4 at the default 32 KiB.
- After that, each op is tiled to vector size on its own, the same way the MLP rewrite does it.
- After dispatch formation, IREE's elementwise fusion usually folds the Q/K `trunc` into the RoPE, so the codegen matcher accepts the RoPE reading the matmul result either directly or through a `trunc`.

The fused dispatch per layer (named `matmul_Dx2048x2048_f16xf16xf32`) has 7 loads (x, norm weight, Wq, Wk, Wv, cos, sin), 3 matmuls, 7 generics and 3 results: Q `[8,4,S,2,32]`, K `[S,8,2,32]`, V `[S,512]`.

Known limits:
- **Prefill only.** Decode lays out RoPE differently (S=1), so the matcher skips it and decode compiles as before.
- **RoPE stays scalar.** The linalg vectorizer doesn't accept the strided `2j`/`2j+1` reads, so RoPE becomes plain loops. That cost should be small next to the matmuls.
- **The KV-cache transpose and scatter stay separate dispatches.**

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

## Reproducing

`QKV=0 ~/workspace/run_llama.sh 512` / `QKV=1 ~/workspace/run_llama.sh 512` (the flag is on by default). For clean timings, run `iree-run-module` with `--output=@/dev/null` under `/usr/bin/time -f "%e %M %R"`.


## Differences from the plan

Three parts don't match what was built:
- The pseudocode shows one combined Wqkv loop over 3,072 columns with K/V scattered straight into the cache. The implementation keeps Wq, Wk and Wv separate: it loops over Q heads, then K heads, then V chunks, and leaves the KV-cache transpose and scatter as their own dispatches.
- "Stops these intermediates from being written out": Q, K and V still leave the fused dispatch as its three outputs. Q and K now leave with RoPE already applied.
- The table's sizes add up to about 7.5 MiB, but the measured transient saving is 2.5 MiB, as the Results section shows.
