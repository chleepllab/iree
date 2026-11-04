# Attention fusion on CPU (`--iree-dispatch-creation-fuse-attention`)

`--iree-dispatch-creation-fuse-attention` (off by default) turns on three changes to attention. The GPU side, and how each part was found, is in [DocGPUFuseAttn.md](DocGPUFuseAttn.md).

| Part | What it does | On CPU |
|---|---|---|
| Mask broadcast fold (`DispatchCreation/ElementwiseOpFusion.cpp`) | The `[S, S]` causal mask is no longer broadcast to `[32, S, S]` and stored; attention reads `[S, S]` through a broadcast indexing map. Also on under `--iree-dispatch-creation-fuse-qkv`. | yes (dispatch creation, target independent) |
| Causal tile skipping (`DispatchCreation/FuseQKVChain.cpp` tags the attention, `Codegen/Common/SkipCausalMaskedAttentionTiles.cpp` lowers the K2 loop bound) | The K2 loop stops at the last key tile that still has an unmasked element. Skipped tiles are all `-inf`, so the result is bit-identical. | yes (pass in the LLVMCPU `LinalgExtTileAndVectorize` pipeline, before `DecomposeAttention`) |
| `load_from_buffer` dim analysis (`Codegen/Common/TensorDynamicDimAnalysis.cpp`) | Blocks K / V / the mask's K2 dim to `[?, 32]` so the GPU MMA attention config applies. | no effect: the CPU pipeline doesn't use `load_from_buffer` at that stage, and CPU has no MMA attention config |

The custom slice packer is not turned on by this flag (only by fuse-conv / fuse-mlp / fuse-qkv), so prefill slices are still packed by the conservative dynamic packer.

## Results (x86)

Host: AMD Ryzen 9 9950X (32 threads), `--device=local-task`, default 32 KiB stack. Model: llama-3.2-1b-instruct f16, KV cache 128x524288 f16. Baseline (no flags) vs `--iree-dispatch-creation-fuse-attention`, 1 run each. Prefill: `prefill_bs1` with S tokens. Decode: `decode_bs1`, one token at position S (seq_len S+1; KV cache zero-filled). Each cell is one gdb run for the transient size (the last `iree_hal_device_queue_alloca` of the call; see [DocFuseMlp.md](DocFuseMlp.md#reproducing)), `HOST_LOCAL` / `DEVICE_LOCAL` from `--print_statistics`, and 1 timed run with output to `/dev/null`. Cells read baseline → fused; "(both)" means equal.

| LLaMA-3.2-1B f16 on x86 | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 MiB → 27.25 MiB (−16 MiB) | 119.50 MiB → 55.50 MiB (−64 MiB) | 371.01 MiB → 115.01 MiB (−256 MiB) |
| Decode transient buffer | 132.33 MiB → 132.30 MiB (−34 KiB) | 136.37 MiB → 136.30 MiB (−66 KiB) | 144.43 MiB → 144.30 MiB (−130 KiB) |
| Prefill `DEVICE_LOCAL` peak | 128.00 MiB (both) | 128.01 MiB (both) | 128.02 MiB (both) |
| Decode `DEVICE_LOCAL` peak | 128.00 MiB (both) | 128.00 MiB (both) | 128.00 MiB (both) |
| Prefill `HOST_LOCAL` peak | 253.25 MiB (both) | 378.50 MiB (both) | 629.00 MiB (both) |
| Decode `HOST_LOCAL` peak | 0.24 MiB (both) | 0.24 MiB (both) | 0.24 MiB (both) |
| Prefill wall time (1 run) | 2.07 s → 2.01 s | 3.69 s → 3.54 s | 8.10 s → 7.31 s |
| Decode wall time (1 run) | 0.87 s → 0.88 s | 0.88 s → 0.89 s | 0.90 s → 0.88 s |

- **Prefill transient:** the saving is exactly the `[32, S, S]` f16 broadcast mask, 32 × S² × 2 B: 16 / 64 / 256 MiB. It grows with S², so it is the largest memory saving of the fusion flags at long S.
- **Decode transient:** decode's mask broadcast is folded too. Its `[32, 1, P]` f16 mask is 32 × P × 2 B with P = 544 / 1056 / 2080 key positions (whole 32-token pages covering S + 1): 34 / 66 / 130 KiB.
- **Prefill wall time:** at S=2048, 8.10 → 7.31 s (−10%), mostly from causal tile skipping. The breakdown of the attention dispatch is below.
- `HOST_LOCAL` / `DEVICE_LOCAL` don't include the transient arena on CPU, so they are unchanged.
- Prefill logits (S=512) are bit-identical to baseline.

### Attention dispatch, causal tile skipping off vs on

`--iree-dispatch-creation-fuse-attention` with `--iree-codegen-causal-attention-tile-skip=false` (hidden; turns the codegen part off) vs on, same host, 1 run each. Attention time is the dispatch's `host_execute` total over 16 layers from `--device_profiling_mode=host-execution,executable-metadata` + `iree-profile statistics`; `tile_sum` is the summed time of all its workgroup tiles.

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Attention dispatch, baseline | 176.6 ms | 735.5 ms | 2,934.3 ms |
| Attention dispatch, skip off → on | 175.3 → 108.5 ms | 718.4 → 545.8 ms | 2,889.2 → 2,219.5 ms |
| Attention `tile_sum`, skip off → on | 2,697 → 1,337 ms | 10,899 → 5,485 ms | 44,318 → 23,020 ms |

- The total work halves, as expected for a causal mask. The dispatch's wall time drops less (−38% at S=512, −24% at S=1024 and S=2048): the workgroups for the last query tiles still process all S keys and decide when the dispatch ends.
- Skip off vs baseline (2,889 vs 2,934 ms at S=2048) is the mask fold alone: attention reads `[S, S]` instead of `[32, S, S]`.
- With skip on and off, the logits are bit-identical to each other and to baseline. CPU's polynomial `exp2` still gives exactly 0 for `-inf` and exactly 1 for 0, which is what makes a skipped tile a no-op.

## How the CPU K2 loop is shortened

For llama the CPU attention dispatch tiles 64 query rows per workgroup and 4 rows per vector tile, with the K2 (key) loop in steps of 16. The dims aren't blocked on CPU, so query and key positions are single dims. The pass reads the query tile's last position `q_max` from the Q slice and the key tile's first position `iv` from the K slice, and sets the loop bound to `min(S, q_max + 1)`. Two shapes differ from the GPU loop and are handled:

- the loop yields the attention results through `tensor.insert_slice` into its own iter_args;
- the Q offset is an `affine.apply` of the workgroup offsets inside the loop; such pure, loop-invariant values are cloned in front of the loop.

## Reproducing

Compile with `--iree-dispatch-creation-fuse-attention` added to the `iree-compile` line of `~/workspace/run_llama.sh` (or the host flags in [DocFuseQKV.md](DocFuseQKV.md#reproducing)). Add `--iree-codegen-causal-attention-tile-skip=false` for the skip-off build. Transient sizes: the gdb `iree_hal_device_queue_alloca` breakpoint in [DocFuseMlp.md](DocFuseMlp.md#reproducing).
