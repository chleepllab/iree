# Gated-MLP fusion on GPU (CUDA)

GPU side of `--iree-dispatch-creation-fuse-mlp` (see [DocFuseMlp.md](DocFuseMlp.md)
for the chain and the CPU side). Dispatch creation is target independent, so
the fused dispatch, and the transient saving, are the same as on CPU; this doc
covers the LLVMGPU codegen and the GX10 CUDA results.

## Compiling

```
iree-compile llama-3.2-1b-instruct-f16.mlir -o llama.vmfb \
    --iree-hal-target-device=cuda \
    --iree-cuda-target=sm_121 \
    --iree-cuda-target-features=+ptx88 \
    --iree-dispatch-creation-fuse-mlp=true
```

- The compiler build needs `IREE_TARGET_BACKEND_CUDA=ON`.
- sm_121 (GB10) needs PTX 8.8; the default `+ptx76` fails with
  `PTX version 7.6 does not support target 'sm_121'`, also without fusion.
- `--iree-cuda-use-ptxas` has no effect in this tree (`produceGpuImage` in
  `CUDATarget.cpp` is never called), so the vmfb carries PTX and the driver
  JIT-compiles it at load.

## Files changed (1 new file, 7 modified)

| File | Change |
|---|---|
| `Codegen/Common/GPURewriteMLPChainAsForall.cpp` (new, 558 lines) | The GPU pass |
| `Codegen/Common/Passes.td`, `Common/CMakeLists.txt` | Pass definition and source entry |
| `Codegen/Common/Passes.h` | Exports `isGatedMLPChainDown()` so config selection can use the matcher |
| `Codegen/LLVMGPU/Passes.cpp` | Adds the pass before workgroup tiling in TileAndFuse, under `#ifdef MY_PASS` / `isMLPFusionEnabled()` like the CPU side |
| `Codegen/LLVMGPU/KernelConfig.cpp` | Sends the MLP chain to TileAndFuse. Without this, decode (M=1) goes to VectorDistribute and the pass never runs |
| `Codegen/LLVMGPU/CMakeLists.txt` | `-DMY_PASS`, as in `LLVMCPU/CMakeLists.txt` |
| `Codegen/Common/TileAndFuseUtils.cpp` | Reduction tiling only coalesces the new loop with an enclosing `scf.for` when the two form a perfect nest (needed by the [MMA path](#mma-path)) |

## Differences from the CPU rewrite

Same M -> F schedule as `RewriteMLPChainAsForall`:

```
scf.forall m (TM rows)                               // step 1: M, workgroups
  acc = 0                       [TM, N] f32          // shared memory
  scf.for f (kFfnChunk = 128)                        // step 2: F, gate/up/h fused in
    gate, up = x[m] @ Wg/Wu[f]^T  [TM, 128] f32      // shared memory
    h = silu(gate) * up           [TM, 128] f16      // shared memory
    acc += h @ Wd[:, f]^T
  out[m] = epilogue(acc)
// step 3: every op above distributed to threads on its own
```

| | CPU (`RewriteMLPChainAsForall`) | GPU (`GPURewriteMLPChainAsForall`) |
|---|---|---|
| Finding the chain | `iree_dispatch.special_mlp` tag | Structural match; the tag does not reach GPU codegen |
| M dims | one (`[M, N]`) | any number; the GPU dispatch has M split into `[D, 32]` |
| Step 3 | each op tiled alone to 8x16x16 vectors | each op distributed alone to threads (`scf.forall`, `gpu.thread` linear mapping), innermost parallel dim up to 16 per thread, reduction tiled to a serial loop of 16 |
| acc, gate/up/h | stack | shared memory (destinations of thread foralls, see `GPUInferMemorySpace`) |
| TM budget | 3/4 of `max_stack_allocation_size` | 1/2 of the target's `max_workgroup_memory_bytes` |
| Workgroup size | — | set to 256 threads (capped at the target maximum) |
| `lowering_config` | dropped from the chain's ops | dropped from every op in the function |

Nothing is fused at thread level. The default TileAndFuse config for this
dispatch takes `down` as the root with `workgroup = [1, 32, 32]` and fuses
gate/up/silu into every tile, recomputing `[32, 8192]` of gate/up 2048/32 = 64
times: the MLP kernel was ~196,500 lines of PTX, and both the driver JIT
(121 GB RSS on the GX10) and offline `ptxas` (out of memory at 79 GB) failed
on it. With the rewrite it is ~1,000 lines, 88 registers, no spills.
`GPUDistributeForall` puts a barrier before and after each thread forall.

## Row tile TM

- bytes per row = N*4 (f32 acc) + 128*(4+4+2) (gate, up f32; h f16) = 9,472 B
- sm_121: `max_workgroup_memory_bytes = 101376` -> budget 50,688 B -> **TM = 4**
  for prefill (TM=8 needs 75,776 B)
- decode (M=1) -> **TM = 1**

## Results

The results in this section and in [Prefill length S](#prefill-length-s) are
from the per-thread FMA (scalar) path, before the [MMA path](#mma-path) became
the default for prefill. Memory numbers are the same on both paths.

Host: ASUS GX10, NVIDIA GB10 (sm_121), CUDA 13.0, driver 580.159.03,
`--device=cuda`. Compiled on x86 with the flags above; runtime built on the
GX10 with `IREE_HAL_DRIVER_CUDA=ON` (plus the runtime fix below).
Model: llama-3.2-1b-instruct f16, full module (prefill + decode). Prefill:
`prefill_bs1`, S=512, KV cache 128x524288 f16. Decode: `decode_bs1`, one token.
3 runs each.

The DEVICE_LOCAL peak drops by exactly 16 MiB on CUDA, since the transient arena there is allocated from device memory and shows up in HAL stats (unlike on CPU).

The fusion stops the two [512, 8192] f16 tensors (gate and up) from being
  written to memory, so the arena shrinks by 2 × 512 × 8192 × 2 B = 16,777,216 B. That is the 45,353,344 → 28,576,128 B row.

| LLaMA-3.2-1B f16 on GB10 | Baseline | Custom fusion (`--iree-dispatch-creation-fuse-mlp=true`) |
|---|---:|---:|
| Prefill transient buffer (activations)* | 45,353,344 B | 28,576,128 B (−16 MiB) |
| Prefill `DEVICE_LOCAL` peak | 2,916,891,272 B | 2,900,114,056 B (−16,777,216 B) |
| Decode `DEVICE_LOCAL` peak | 2,740,771,872 B | 2,740,739,104 B (−32 KiB) |
| `HOST_LOCAL` peak (prefill) | 265,551,872 B | 265,551,872 B |
| Max RSS, prefill (runs 2–3) | 377,528–377,704 KB | 377,484–377,508 KB |
| Prefill wall time (runs 2–3) | 1.38–1.45 s | 2.26–2.29 s |
| Decode wall time (3 runs) | 1.04–1.11 s | 1.19–1.27 s |
| Prefill logits vs baseline | — | max \|Δ\| 0.047, mean \|Δ\| 0.0019, top-1 100% |
| Decode logits vs baseline | — | max \|Δ\| 0.0039, mean \|Δ\| 0.0003, top-1 100% |
| vmfb size | 677,389 B | 669,629 B |

\* Captured with gdb on a prefill-only build of the module (`decode_bs1`
removed); the full module's `DEVICE_LOCAL` peak drops by the same 16 MiB.

- Transient buffer: −16,777,216 B (exactly 16 MiB, −37.0%), the same as on
  CPU. On CUDA the transient arena is allocated from device memory, so the
  saving also shows in the `DEVICE_LOCAL` peak (−16,777,216 B prefill,
  −32,768 B decode: the two `[1, 8192]` f16 intermediates).
- `HOST_LOCAL` is the logits output; max RSS is host-side only (the weights
  live in device memory) and does not change.
- Outputs are close but not bit-identical: the fused `down` accumulates
  128-wide F chunks in f32 instead of the baseline's tensor-core order. The
  difference is smaller than between the CUDA and CPU baselines (prefill
  max |Δ| 0.22).
- Wall time includes ~1 s of process start-up and weight upload (first runs
  are 1.9 s / 2.75 s and are left out). Fused prefill compute is ~0.9 s
  slower: the rewrite uses per-thread FMAs, the baseline matmuls use
  `NV_MMA_SYNC_F32_16x8x16_F16`.

### Prefill length S

Same setup, `prefill_bs1` at S=1024 and S=2048 (64 of the 128 KV pages). The
S=512 column repeats the table above. Transient sizes at S=1024/2048 are from
gdb on the full module; times and RSS are all 3 runs.

| LLaMA-3.2-1B f16 prefill on GB10 | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Transient buffer, baseline | 45,353,344 B | 125,309,696 B | 389,031,424 B |
| Transient buffer, fused | 28,576,128 B (−16 MiB, −37.0%) | 91,755,264 B (−32 MiB, −26.8%) | 321,922,560 B (−64 MiB, −17.3%) |
| `DEVICE_LOCAL` peak, baseline | 2,916,891,272 B | 3,128,185,992 B | 3,654,584,456 B |
| `DEVICE_LOCAL` peak, fused | 2,900,114,056 B (−16,777,216 B) | 3,094,631,560 B (−33,554,432 B) | 3,587,475,592 B (−67,108,864 B) |
| `HOST_LOCAL` peak (both) | 265,551,872 B | 396,886,016 B | 659,554,304 B |
| Max RSS, baseline | 377,528–377,704 KB | 505,872–506,064 KB | 762,404–762,676 KB |
| Max RSS, fused | 377,484–377,508 KB | 505,784–506,020 KB | 762,196–762,280 KB |
| Wall time, baseline | 1.38–1.45 s | 1.79 s | 3.07–3.11 s |
| Wall time, fused | 2.26–2.29 s | 3.33–3.35 s | 6.02–6.10 s |
| Logits vs baseline | max \|Δ\| 0.047, mean 0.0019, top-1 100% | max \|Δ\| 0.047, mean 0.0017, top-1 100% | max \|Δ\| 0.051, mean 0.0017, top-1 99.66% |

- The saving is 2 × S × 8192 × 2 B (`gate` and `up` in f16), linear in S, and
  the `DEVICE_LOCAL` peak drops by exactly that much at every S.
- The baseline transient grows faster than S (2.8x from 512 to 1024, 3.1x
  from 1024 to 2048), so the relative saving shrinks: −37.0%, −26.8%, −17.3%.
- S=2048 top-1: 7 of 2048 positions pick a different token. All 7 are ties in
  the baseline logits (top-1/top-2 gap 0 or 0.0039, one f16 step at ~6;
  median gap over all positions 0.25).
- Fused prefill wall time is 1.6x, 1.9x and 2.0x the baseline at S=512, 1024
  and 2048 (per-thread FMAs vs MMA; the ~1 s start-up is included).

## MMA path

### Why the scalar path is slow

`nsys` per-kernel GPU time for one S=512 prefill (`cuda_gpu_kern_sum`):

| | Baseline | Fused, scalar (TM=4) |
|---|---:|---:|
| MLP kernels, 16 layers | 92.0 ms (gate 39.4 + up 36.7 + down 15.9) | 905.7 ms (one kernel) |
| Rest of the model | 155.3 ms | 153.0 ms |
| Total GPU time | 247.3 ms | 1,058.7 ms |
| MLP throughput (51.5 GFLOP per layer) | ~9.0 TFLOP/s | ~0.9 TFLOP/s |

Row tile sweep (`--iree-codegen-gpu-mlp-row-tile`, all top-1 100% vs
baseline): TM=1 1,048.7 ms, TM=2 1,005.0 ms, TM=4 905.7 ms, TM=8 1,138.0 ms.
Cutting weight re-reads 8x leaves the time within ±15%, so weight reuse is not
the bottleneck. The PTX has no `mma.sync`: every product is an
`fma.rn.f32.f16` in chains of 16 dependent FMAs per output, and 88 registers x
256 threads plus 37.9 KB of shared memory allow 2 workgroups (16 of 48 warps)
per SM. At ~0.9 TFLOP/s the kernel reaches ~3% of the GB10's plain-FP32 peak
(48 SMs at 2.42 GHz). TM=8 is slowest because 75.8 KB of shared memory leaves
one workgroup per SM. Nsight Compute could not confirm the stall reasons: GPU
performance counters are admin-only on the GX10 (`RmProfilingAdminOnly: 1`).

### What it does

Default for prefill (inner M dim a multiple of 16); decode (M=1) keeps the
scalar path.

```
scf.forall (m, n) (16 rows, nTile = 512 cols)        // step 1: M and N, workgroups
  acc = 0                       [16, 512] f32        // shared memory, 32 KB
  scf.for f (kFfnChunk = 128)                        // step 2: F, gate/up/h fused in
    gate, up = x[m] @ Wg/Wu[f]^T  [16, 128] f32      // MMA, K tiled to 32
    h = silu(gate) * up           [16, 128] f16      // thread config
    acc += h @ Wd[n, f]^T                            // MMA
  out[m, n] = epilogue(acc)
```

- TM is one intrinsic (`NV_MMA_SYNC_F32_16x8x16_F16`, 16 rows). A [16, 2048]
  f32 accumulator (128 KB) does not fit in shared memory, so N is split too
  and gate/up are recomputed once per N tile (4x at nTile = 512, ~3x the
  MLP's FLOPs in total).
- Step 3 sets lowering configs instead of distributing to threads: gate and
  up get `mma_kind`, `promote_operands = [0, 1]`, `reduction` (K) and
  `subgroup` (N) tiles; down gets `mma_kind` and `subgroup`; h gets `thread`.
  The TileAndFuse levels then lower them. `GPUApplyTilingLevel` does not fuse
  a producer that has its own config at that level, so each op is tiled
  alone and h is not recomputed per down subgroup tile. Thread tiling runs
  before subgroup tiling, so h is already a thread forall when down is tiled.
- Reduction tiling of gate/up happens inside the F loop.
  `applyTileAndFuseToEachRoot` coalesced the new K loop with its parent
  `scf.for` and asserted in `coalesceLoops`, since the F loop is not a perfect
  nest; `TileAndFuseUtils.cpp` now only coalesces perfect nests.

Flags (hidden):

| Flag | Default | |
|---|---:|---|
| `--iree-codegen-gpu-mlp-mma` | true | MMA path (TM = 16, N split into 1024-column tiles); false gives the scalar TM = 8 path |
| `--iree-codegen-gpu-mlp-down-k-tile` | 16 | scalar path: down's serial K step per thread; 0 = whole 128 chunk in one op |

The MMA path's N tile, gate/up K tile and operand promotion are now fixed
constants, set to 1024, 4 intrinsics (K = 64) and off (Option 1: promotion
with the 64 KiB accumulator needs 104,448 B at K = 32, over the 101,376 B
limit; GX10 prefill S = 512 / 1024 / 2048, 1 run: 1.65 / 2.09 / 3.65 s vs
baseline 1.30 / 1.73 / 2.93 s) (`kMMANTile`, `kMMAReductionK`,
`kMMAPromote`); the `-mma-n-tile`, `-mma-reduction-k`, `-mma-promote` and
`-row-tile` flags used for the tuning below were removed.

Tuning at S=512 (fused MLP GPU time, 16 layers; every variant bit-identical
to baseline):

| nTile | K tile | Promote | MLP time | ms/layer |
|---:|---:|:-:|---:|---:|
| 1024 | 64 | no | 331.1 ms | 20.70 |
| 1024 | 32 | no | 417.0 ms | 26.06 |
| 1024 | 128 | no | 443.5 ms | 27.72 |
| 512 | 64 | no | 360.7 ms | 22.55 |
| 256 | 64 | no | 597.5 ms | 37.35 |
| **512** | **32** | **yes** | **303.2 ms** | **18.95** |
| 512 | 64 | yes | 441.2 ms | 27.57 |

### Results

Same host and setup as above; all three builds measured in one session, 3 runs
each. MLP GPU time from `nsys`.

| LLaMA-3.2-1B f16 prefill on GB10 | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| MLP GPU time, baseline (3 matmuls) | 92.0 ms | 191.7 ms | 368.7 ms |
| MLP GPU time, fused scalar | 905.7 ms | 1,754.7 ms | 3,370.0 ms |
| MLP GPU time, fused MMA | 303.5 ms | 619.0 ms | 1,163.8 ms |
| Total GPU time, baseline | 247.3 ms | 630.6 ms | 1,916.1 ms |
| Total GPU time, fused scalar | 1,058.7 ms | 2,197.0 ms | 4,917.1 ms |
| Total GPU time, fused MMA | 457.1 ms | 1,062.3 ms | 2,721.1 ms |
| Wall time, baseline | 1.34–1.36 s | 1.76–1.77 s | 3.15–3.18 s |
| Wall time, fused scalar | 2.16–2.23 s | 3.33–3.38 s | 6.06–6.17 s |
| Wall time, fused MMA | 1.61–1.66 s | 2.27–2.33 s | 3.97–4.01 s |
| `DEVICE_LOCAL` peak, fused (both paths) | 2,900,114,056 B (−16 MiB) | 3,094,631,560 B (−32 MiB) | 3,587,475,592 B (−64 MiB) |
| Logits, fused MMA vs baseline | bit-identical | bit-identical | bit-identical |

- The fused MLP kernel is ~3.0x faster than the scalar one at every S and
  ~3.2x slower than the baseline's three matmuls. Prefill wall time over
  baseline drops from +0.84 / +1.58 / +2.96 s to +0.28 / +0.52 / +0.83 s.
- Logits are bit-identical to the baseline: same intrinsic, and down
  accumulates the same 128-wide K chunks.
- Decode is unchanged (scalar path): 1.25–1.27 s vs 1.07–1.10 s baseline,
  logits max |Δ| 0.0039.
- The remaining gap is likely the 4x gate/up recompute (~3x the baseline's
  MLP FLOPs) and low occupancy from the shared memory each workgroup holds
  (32 KB accumulator, 20 KB of chunk buffers, plus the promoted x/weight
  tiles); not confirmed without Nsight Compute counters.

## Reproducing

1. Compiler: reconfigure the build with `-DIREE_TARGET_BACKEND_CUDA=ON` and
   rebuild `iree-compile`.
2. Runtime on the GX10 (runtime only, gcc 13, Makefiles):

   ```
   cmake -S iree_c -B iree_c-build-cuda -G "Unix Makefiles" \
       -DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_COMPILER=OFF \
       -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
       -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_CUDA=ON \
       -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
       -DCUDAToolkit_ROOT=/usr/local/cuda
   cmake --build iree_c-build-cuda --target iree-run-module -j20
   ```

3. Runtime fix: without it every CUDA run fails with
   `function 'prefill_bs1$async_dispatch_0_...' not found in executable`.
   The compiler stores kernel names through `sanitizeSymbolName`
   (`prefill_bs1_async_...`) but the VM looks them up by the original name
   (`prefill_bs1$async_...`). The fix makes
   `iree_hal_cuda_native_executable_lookup_export_by_name`
   (`runtime/src/iree/hal/drivers/cuda/native_executable.c`) apply the same
   mapping (every char outside `[A-Za-z0-9_]` -> `_`) to the requested name
   before comparing. It is applied only on the GX10 copy of the source so far.
4. Run with `--device=cuda`; inputs as in `~/workspace/run_llama.sh`.
   Transient size: the gdb breakpoint from DocFuseMlp.md works on CUDA too
   (aarch64 host, so print `$x6`); the largest alloca after
   `EXEC @prefill_bs1` other than the 265,551,872 B logits buffer is the
   transient arena.

## Open items

- Speed, MMA path: keep the accumulator in registers (each subgroup owning an
  N slice across the F loop) instead of shared memory. That removes the N
  split, and with it the gate/up recompute and the shared-memory limit on
  occupancy. It does not fit the stock TileAndFuse levels (the F loop holds
  more than the accumulating op), so it needs its own loop structure.
- Decode (M=1) still uses the scalar path and is ~0.17 s slower than
  baseline for a 32 KiB saving; consider not fusing it.
- Confirm the MMA-path bottleneck with Nsight Compute once performance
  counters are enabled for non-admin users on the GX10.
- Upstream the runtime name-lookup fix (or make the compiler emit
  `$`-free export names).
- No lit test for `GPURewriteMLPChainAsForall` yet.

## Single matmul sweep (NV_MMA_SYNC_F32_16x8x16_F16)

How fast is one matmul on the GB10 by itself, and at which size? One
`linalg.matmul` per size, square, `[S, S] f16 x [S, S] f16 -> [S, S] f32`
(zero fill + matmul, static shapes), no model around it:

```mlir
func.func @matmul(%a: tensor<SxSxf16>, %b: tensor<SxSxf16>) -> tensor<SxSxf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<SxSxf32>
  %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<SxSxf32>) -> tensor<SxSxf32>
  %c = linalg.matmul ins(%a, %b : tensor<SxSxf16>, tensor<SxSxf16>)
                     outs(%acc : tensor<SxSxf32>) -> tensor<SxSxf32>
  return %c : tensor<SxSxf32>
}
```

`NV_MMA_SYNC_F32_16x8x16_F16` is the only f16-input / f32-accumulate
`NV_MMA_SYNC` intrinsic (the other `NV_MMA_SYNC` ones are F16-accumulate, BF16
and FP8), so the sweep is over the matmul size, not the intrinsic. sm_121 also
lists `NV_WMMA_F32_16x16x16_F16` (WMMA, 16x16x16), which KernelConfig did not
pick for any of these sizes and which was not tested here. Compiled with the
flags in [Compiling](#compiling), no fusion flags; KernelConfig picks the
intrinsic by itself, and every size's PTX is `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.

Tile config KernelConfig picks (128 threads, `promote_operands = [0, 1]`):

| S | `workgroup` (M, N) | `subgroup` (M, N, intrinsics) | `reduction` (K, intrinsics) |
|---|---|---|---|
| 128 – 1024 | 64 x 64 | 2 x 4 | 8 (K = 128) |
| 2048 – 8192 | 128 x 64 | 4 x 4 | 2 (K = 32) |

Measured on the GX10 (same host as above; GB10: 48 SMs at 2.418 GHz, 24 MiB L2,
LPDDR5x 8533 MT/s x 256 bit = 273 GB/s):

- Kernel time: `nsys` `cuda_gpu_kern_sum`, average over 200 calls of
  `iree-benchmark-module --benchmark_min_time=200x`. This is the GPU time of
  the matmul dispatch alone.
- Per call: `iree-benchmark-module` `real_time` mean of 5 repetitions (1 s
  each): submit + kernel + wait, as `iree-run-module` would see it.
- TFLOP/s = 2·S³ / time. GB/s = minimum traffic / time, minimum traffic = A + B
  read once (2 B each) + C written once (4 B) = 8·S² B. Real DRAM/L2 traffic is
  higher: each operand tile is re-read once per workgroup row/column.

| S | Kernel time | Kernel TFLOP/s | Kernel GB/s | Per call | Per-call TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 7.5 µs | 0.56 | 17.4 | 2.68 ms | 0.0016 |
| 256 | 12.7 µs | 2.6 | 41.1 | 2.59 ms | 0.013 |
| 512 | 20.4 µs | 13.2 | 103 | 2.90 ms | 0.093 |
| 1024 | 76.2 µs | 28.2 | 110 | 2.95 ms | 0.73 |
| **2048** | **337 µs** | **50.9** | 99.5 | 2.75 ms | 6.2 |
| 4096 | 4.68 ms | 29.4 | 28.7 | 9.37 ms | 14.7 |
| 8192 | 49.2 ms | 22.4 | 10.9 | 62.3 ms | 17.6 |

![Single matmul sweep](../../tex/png/fig/matmul_mma_sweep.png)

(`~/tex/png/fig/matmul_mma_sweep.png`, from `python matmul_sweep.py
json/matmul_mma_data.json` in `~/tex/png`.)

- Best kernel throughput is S = 2048: 50.9 TFLOP/s, ~1.7x the GB10's plain-FP32
  FMA peak (29.7 TFLOP/s, see [Why the scalar path is slow](#why-the-scalar-path-is-slow)).
  Below that the kernel is too small to fill 48 SMs: S = 128 is 4 workgroups,
  S = 512 is 64, S = 1024 is 256 (64 x 64 tiles).
- At S = 4096 throughput drops to 29.4 TFLOP/s and the kernel time varies
  4.03–6.05 ms over the 200 calls. A + B are 16 MiB at S = 2048 (fit in the
  24 MiB L2) and 64 MiB at S = 4096 (do not), so the tile re-reads go to DRAM.
  This is the likely cause, not confirmed: Nsight Compute counters are
  admin-only on the GX10.
- Bandwidth (minimum traffic) stays far below the 273 GB/s DRAM peak at every
  size, so even the minimum traffic does not limit any of these sizes; the
  matmul is compute bound or tile-reuse bound, not bound by moving A, B, C once.
- Per-call time has a ~2.6 ms floor up to S = 2048 (the kernel is 0.3 ms or
  less there), so small matmuls run from `iree-run-module` /
  `iree-benchmark-module` measure host overhead, not the GPU. Where the floor
  comes from (output allocation, synchronization) was not traced.
- A llama MLP projection at prefill S = 512 (512 x 8192 x 2048) has the same
  FLOPs as the S = 2048 square matmul (2·512·2048·8192 = 2·2048³) but a
  different shape; it was not run here. The baseline's three MLP matmuls take
  92.0 ms over 16 layers = 1.92 ms per projection, i.e. ~9 TFLOP/s
  (see [Why the scalar path is slow](#why-the-scalar-path-is-slow)), well
  below the square-matmul peak.

Files: `~/workspace/tmp/gx10-mm/` (`gen.sh` writes `mm_<S>.mlir`,
`bench_mm.sh` is the GX10 run script; copies and logs in `~/matmul-mma` on the
GX10). Plot data: `~/tex/png/json/matmul_mma_data.json`.

## Schedule variants (TM = 8 scalar, TM = 16 / N tile 1024 MMA)

All on the GX10, same model and inputs as [Results](#results). Variants are
listed by the schedule they implement; `acc` is always f32 in shared memory.

### TM = 8, no N split (scalar)

`--iree-codegen-gpu-mlp-mma=false`. One workgroup per 8 rows and all 2048
columns, so gate/up are computed once per (M, K2):

```
scf.forall M (TM = 8)                                   // workgroups, 256 threads
  acc = 0                              [8, 2048] f32    // shared memory, 64 KiB
  scf.for K2 (FC = 128)
    gate, up = x[M] @ Wg/Wu[K2]^T      [8, 128]         // per-thread FMAs
    h = silu(gate) * up                [8, 128] f16
    scf.for N (16) / scf.for k2 (16)                    // N over threads: each owns acc[0:8, N:N+16]
      acc[:, N] += h[:, k2] @ Wd[N, k2]^T               // k2 serial in the thread
  out[M] = residual(acc)
```

- `--iree-codegen-gpu-mlp-down-k-tile` sets the k2 step (default 16). `0`
  removes the k2 loop: `acc[:, N] += h @ Wd[N, :]^T` over the whole 128 chunk
  in one op ("no k2" below). The kernel grows from 1,284 to 6,007 PTX lines
  (1,078 -> 5,667 virtual registers, no local spills in the PTX).
- No `mma.sync` in the MLP kernel on either variant: TM = 8 is below the
  16-row intrinsic, and padding to 16 would waste half of each MMA.
- TM = 16 on this schedule does not compile: `[16, 2048]` f32 is 128 KiB,
  151,552 B of shared memory in total against sm_121's 101,376 B.

Baseline → k2 (16) → no k2, one session, 3 runs each:

| LLaMA-3.2-1B f16 on GX10 | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 → 27.25 → 27.25 MiB (−16 MiB) | 119.50 → 87.50 → 87.50 MiB (−32 MiB) | 371.01 → 307.01 → 307.01 MiB (−64 MiB) |
| Decode transient buffer | 132.33 → 132.30 → 132.30 MiB (−32 KiB) | 136.37 → 136.33 → 136.33 MiB (−32 KiB) | 144.43 → 144.40 → 144.40 MiB (−32 KiB) |
| Prefill `DEVICE_LOCAL` peak | 2,781.76 → 2,765.76 → 2,765.76 MiB | 2,983.27 → 2,951.27 → 2,951.27 MiB | 3,485.28 → 3,421.28 → 3,421.28 MiB |
| Decode `DEVICE_LOCAL` peak | 2,617.84 → 2,617.80 → 2,617.80 MiB | 2,621.87 → 2,621.84 → 2,621.84 MiB | 2,629.93 → 2,629.90 → 2,629.90 MiB |
| Prefill `HOST_LOCAL` peak | 253.25 MiB (all) | 378.50 MiB (all) | 629.00 MiB (all) |
| Decode `HOST_LOCAL` peak | 74.01–84.02 MiB (all) | 74.01–84.01 MiB (all) | 74.01–84.01 MiB (all) |
| Prefill wall time | 1.34–1.36 → 1.79 → 2.86–2.97 s | 1.71–1.73 → 2.36–2.41 → 4.74–4.77 s | 2.96–3.01 → 4.24–4.30 → 9.25–9.39 s |
| Decode wall time | 1.05–1.06 → 1.17–1.26 → 1.21–1.24 s | 1.08–1.11 → 1.20–1.24 → 1.21–1.24 s | 1.07–1.14 → 1.25–1.28 → 1.23–1.29 s |

- k2 (16) is faster than the old scalar TM = 4 path (2.26 / 3.33 / 6.02 s in
  [Prefill length S](#prefill-length-s)): 8 rows x 16 columns of `down` per
  thread instead of 1 x 16, and no N split.
- No k2 is 1.6x / 2.0x / 2.2x slower than k2 (16) and slower than the old
  scalar path; likely register pressure / occupancy from the unrolled 128-long
  reduction (not confirmed, no Nsight Compute counters on the GX10).
- Decode (TM = 1) is the same on both; memory is identical on both.

### TM = 16, N tile 1024 (MMA, the default)

```
scf.forall (M, N0) (TM = 16, NT = 1024)                 // workgroups, 256 threads = 8 warps
  acc = 0                              [16, 1024] f32   // shared memory, 64 KiB
  scf.for K2 (FC = 128)
    gate, up = x[M] @ Wg/Wu[K2]^T      [16, 128]        // MMA 16x8x16, K step 64;
                                                        //   2x, once per N0 tile
    h = silu(gate) * up                [16, 128] f16
    scf.for N in N0 .. N0+1024                          // per warp: acc[:, 128 columns]
      acc[:, N] += h @ Wd[N, K2]^T                      // MMA 16x8x16
  out[M, N0] = residual(acc)
```

Same as the earlier [MMA path](#mma-path) with N tile 1024 instead of 512
(gate/up recomputed 2x instead of 4x). The MLP kernel has 144 `mma.sync`.
Decode (M = 1) is not a multiple of 16 and keeps the scalar path.

Promoting gate/up's operands (x and the weight tile) to shared memory does not
fit next to the 64 KiB accumulator at K = 32 (104,448 B > 101,376 B). Gate/up
operand variants, prefill wall time, one session, 1 run each:

| Prefill | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Baseline | 1.30 s | 1.73 s | 2.93 s |
| TM = 8, k2 (16) (scalar) | 1.75 s | 2.34 s | 4.27 s |
| TM = 16 / 1024, K step 16, promoted | 1.83 s | 2.39 s | 4.23 s |
| TM = 16 / 1024, K step 32, not promoted | 1.71 s | 2.28 s | 3.96 s |
| **TM = 16 / 1024, K step 64, not promoted (default)** | **1.65 s** | **2.09 s** | **3.65 s** |

Default vs baseline, 1 run each (memory rows as in the table above):

| LLaMA-3.2-1B f16 on GX10 | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Prefill transient buffer | 43.25 MiB → 27.25 MiB (−16 MiB) | 119.50 MiB → 87.50 MiB (−32 MiB) | 371.01 MiB → 307.01 MiB (−64 MiB) |
| Decode transient buffer | 132.33 MiB → 132.30 MiB (−32 KiB) | 136.37 MiB → 136.33 MiB (−32 KiB) | 144.43 MiB → 144.40 MiB (−32 KiB) |
| Prefill `DEVICE_LOCAL` peak | 2,781.76 MiB → 2,765.76 MiB (−16 MiB) | 2,983.27 MiB → 2,951.27 MiB (−32 MiB) | 3,485.28 MiB → 3,421.28 MiB (−64 MiB) |
| Decode `DEVICE_LOCAL` peak | 2,617.84 MiB → 2,617.80 MiB (−32 KiB) | 2,621.87 MiB → 2,621.84 MiB (−32 KiB) | 2,629.93 MiB → 2,629.90 MiB (−32 KiB) |
| Prefill wall time | 1.30 s → 1.65 s | 1.73 s → 2.09 s | 2.93 s → 3.65 s |
| Decode wall time | 1.13 s → 1.21 s | 1.09 s → 1.26 s | 1.05 s → 1.21 s |

- Fastest fused variant so far: +0.35 / +0.36 / +0.72 s over baseline,
  against +0.45 / +0.61 / +1.34 s for TM = 8 in the same session, and faster
  than the N tile 512 MMA path at S = 1024 / 2048 (2.27–2.33 / 3.97–4.01 s,
  earlier session, [Results](#results-1)).
- Not promoting and taking bigger K steps wins: with promotion only K = 16
  fits, and it is the slowest MMA variant.

Files: x86 `~/workspace/tmp/gx10-cuda/` (`full_tm8`, `full_tm8_origin`,
`opt1_k{1,2,4}_p*`, `full_opt1` = default `.vmfb`s); GX10 `~/llama-mem-cuda/`
(`bench_s_origin.log`, `bench_decode_origin.log`, `bench_s_opt1.log`,
`bench_decode_opt1.log`; 1-run scripts `bench_s_1run.sh`,
`bench_decode_s_1run.sh`).

### Why down is slow (likely, not confirmed)

Likely causes, from the down loop in the current IR (TN = 256, k2 step 64;
N and k2 coalesced into one `scf.for 0 .. 8`, after
`UnrollAnnotatedLoopsPass`, `dispatch_28`). No Nsight Compute counters on the
GX10 to confirm:

- Each down step loads its `Wd` slice straight from global memory into
  registers (`map_load` into a private `alloca`), with no shared-memory
  staging.
- Each warp has only 4 MMA tiles (16x8) per step: 32 columns of the 256-wide
  N slice, 16 `mma.sync` per (N, k2) step.
- `acc` goes through shared memory on every (N, k2) step: read the warp's
  tiles, 16 MMAs, write them back. That is 8 round trips per F chunk.
- There are 64 F chunks (8192 / 128), each with barriers: around gate/up/h,
  and one per (N, k2) step of down.

### Shared memory limit (101,376 B per SM)

The MMA kernel's workgroup buffers are all in shared memory (checked in the
IR after `IREEComprehensiveBufferizePass`, `dispatch_28`):

| Buffer | Type | Size |
|---|---|---|
| acc | `memref<1x1x128x2x8x8xf32, workgroup>` | 64 KiB |
| gate | `memref<1x1x2x8x16x8xf32, workgroup>` | 8 KiB |
| up | `memref<1x1x2x8x16x8xf32, workgroup>` | 8 KiB |
| h | `memref<1x16x128xf16, workgroup>` | 4 KiB |
| **Total** | | **84 KiB** |

The 101,376 B (99 KiB) limit caps two things:

1. **Workgroups resident per SM (occupancy).** A resident workgroup holds its
   shared memory for its whole lifetime, so
   `floor(99 KiB / 84 KiB) = 1` workgroup per SM: 8 of 48 warps (17%), with
   registers, warp slots and block slots to spare. Shared memory runs out first.
2. **Which N tiles are allowed at all (per-workgroup maximum).** The pass
   checks `kMMARowTile * (nTile * 4 + kFfnChunk * (4 + 4 + 2)) <= limit`
   (`getMaxWorkgroupMemoryBytes()`) before taking the MMA path:

| nTile | Shared memory | Fits? |
|---|---|---|
| 2048 (no N split) | 16 × (8,192 + 1,280) = 148 KiB | no, MMA path not used |
| 1024 (default) | 84 KiB | yes, 1 per SM |
| 512 | 52 KiB | yes, 1 per SM (99 / 52 = 1.9) |
| 256 | 36 KiB | yes, 2 per SM |

This limit is why N is split at all: nTile = 2048 would compute gate/up once,
but `acc` alone would need 128 KiB. A large shared-memory `acc` costs occupancy
either way, which is why moving `acc` to registers is in
[Open items](#open-items).

## Current default vs baseline

Current default vs baseline (no fusion), GX10 prefill, 1 run each, back to back:

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Baseline wall time | 1.31 s | 1.70 s | 2.92 s |
| Current default wall time | 1.56 s (+0.25) | 1.94 s (+0.24) | 3.31 s (+0.39) |
| `DEVICE_LOCAL` peak | −16 MiB | −32 MiB | −64 MiB |

nsys (16 layers) shows where the gap is. Baseline gate + up + down is three
separate matmul kernels; the fused MLP is one kernel:

| | Baseline gate + up + down | Fused MLP kernel |
|---|---:|---:|
| S=512 | 34.9 + 37.7 + 13.4 = 86.0 ms | 175.5 ms (2.0×) |
| S=2048 | 144.9 + 149.0 + 51.6 = 345.5 ms | 561.0 ms (1.6×) |

Attention and the other kernels are the same in both. So the whole wall-time
gap is the fused MLP kernel.

### Where the fused kernel's time goes (estimate)

Split from the earlier N-loop experiment, assuming down took the same time in
both builds. S = 2048:

| Part | Fused (current default) | Baseline | Fused vs baseline |
|---|---:|---:|---|
| gate + up (+ h) | ~274 ms, computed 2× (2 N0 tiles) | 293.9 ms, computed 1× | per unit of work, fused is ~2× more efficient |
| down | ~289 ms | 51.6 ms | ~5.6× slower |
| Total | ~563 ms | 345.5 ms | 1.6× |

At S = 512 the same split gives gate/up ≈ 98 ms and down ≈ 76 ms, against
86.0 ms for the whole baseline MLP.

What it would take to beat baseline (S = 2048):

1. **Compute gate/up once instead of twice**: saves ~137 ms (~49 ms at
   S = 512). That needs the whole `[16, 2048]` f32 `acc` on chip: 128 KiB,
   over the 99 KiB of shared memory (see
   [Shared memory limit](#shared-memory-limit-101376-b-per-sm)). In registers
   it would be 64 f32 per thread at 512 threads.
2. **Make down as efficient as gate/up already is**: ~289 → ~70 ms. Down
   does the same kind of MMAs as gate/up at about a quarter of the rate
   (gate/up: ~274 ms for 4 matmul-units of work, ~68 ms each; down is 1 unit).

| | gate/up | down | Total | vs baseline 345.5 ms |
|---|---:|---:|---:|---|
| Now | ~274 | ~289 | ~563 | 1.6× |
| (1) only | ~137 | ~289 | ~426 | slower |
| (2) only | ~274 | ~70 | ~344 | about even |
| (1) + (2) | ~137 | ~70 | ~207 | 0.6× |

Either one alone doesn't beat baseline; both together would.

### Weight memory traffic check

How much of the fused kernel's time is spent loading weights? Two check builds
pin one weight to its first F chunk, so every F chunk reads the same slice,
which then stays cached. The time saved is roughly what loading that weight
costs. Both builds give wrong results and are for timing only; the IR confirms
only the targeted slice was pinned.

- **Wd pinned:** every F chunk reads Wd's chunk 0 instead of its own.
- **Wg/Wu pinned:** the same for Wg and Wu.

MLP kernel, nsys, 16 layers, all three builds in one session:

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Current default | 174.3 ms | 282.2 ms | 557.2 ms |
| Wd pinned to chunk 0 | 146.5 ms (−27.8, −16%) | 225.8 ms (−56.4, −20%) | 447.5 ms (−109.7, −20%) |
| Wg/Wu pinned to chunk 0 | 157.1 ms (−17.2, −10%) | 241.4 ms (−40.8, −14%) | 485.6 ms (−71.6, −13%) |
| Lower bound: both weight costs removed | ~129 ms | ~185 ms | ~376 ms |
| Baseline gate + up + down | 86.0 ms | — | 345.5 ms |

The lower bound subtracts both savings from the default, assuming they add up.

- Weight memory traffic accounts for only about 20% of the fused kernel's
  time (Wd: 16–20%; Wd and Wg/Wu together: 45.0 / 97.2 / 181.3 ms, 26–34%).
  Even with weights free, the kernel would still be
  slower than the baseline's three matmuls: ~129 vs 86.0 ms at S = 512, ~376
  vs 345.5 ms at S = 2048. So the gap comes from how the work is structured,
  not from memory bandwidth. Staging Wd with async copies can't close it by
  itself. Neither can a TM = 8 version with swapped operands, which halves
  weight reuse.
- **The remaining cost is structural:**
  - gate/up is computed twice (2 N0 tiles);
  - with TM = 16 there is one M tile, so each weight fragment loaded into
    registers feeds exactly one `mma.sync`. The baseline's 128×64 tiles reuse
    each weight fragment across 4–8 MMAs. Per MMA, the fused kernel issues
    far more load and address instructions;
  - every F chunk adds barriers and `acc` round trips through shared memory
    (see [Why down is slow](#why-down-is-slow-likely-not-confirmed)).

## Hand-written kernel

Prefill is now faster than the baseline at every S, with the 16 MiB saving at S=512 intact (32 and 64 MiB at S=1024 and 2048). The
compiler-generated kernel couldn't get there, so this uses a hand-written CUDA kernel for the fused MLP, swapped into the IREE build
with a compile flag.

GX10 prefill wall time, 1 run each:

| | S=512 | S=1024 | S=2048 |
|---|---:|---:|---:|
| Baseline (no fusion) | 1.37 s | 1.69 s | 2.99 s |
| IREE-generated fused kernel (compiler default) | 1.49 s | 1.88 s | 3.28 s |
| Hand kernel | 1.34 s | 1.66 s | 2.73 s |
| Transient buffer vs baseline | −16 MiB | −32 MiB | −64 MiB |

The gain is much larger on the MLP itself (nsys, 16 layers): 35.8 vs 86.3 ms at S=512 and 120.0 vs 346.6 ms at S=2048, about 2.4–2.9×
faster than the baseline's three matmuls. The wall-time gain is smaller because attention dominates prefill; at S=2048 it's about 1.24
s of the 1.53 s of GPU time.

### What the hand kernel does

`~/workspace/tmp/gx10-hand/mlp_fused_v2p3f.cu` (`NWARPS = 16`, `DG = 4`):

- **Tiling:** one 512-thread workgroup per 16 rows covering all 2048 output
  columns, so there's no N split and gate/up is computed once instead of twice.
- **acc in registers:** each of the 16 warps keeps its `[16, 128]` slice for
  the whole kernel, so there's no `acc` traffic through shared memory.
- **x staged once:** x goes into shared memory once and is fed to the MMAs
  with `ldmatrix`.
- **One load per fragment:** within each 32-wide k block, k is reordered the
  same way for x/h and the weights, so each lane's weight fragment is a single
  16-byte load.
- **Prefetch:** 3 steps ahead in gate/up (4-slot ring); Wd groups are
  double-buffered in down.
- **h:** computed per warp in registers into a double-buffered f16 tile, with
  one barrier per F chunk.

```
scf.forall M (TM = 16)                                   // grid M / 16, 512 threads = 16 warps
  x[M] -> shared                       [16, 2048] f16    // once, k relabeled
  acc = 0                              [16, 2048] f32    // REGISTERS: warp w owns cols 128w : +128
                                                         //   16 MMA tiles = 64 f32 per thread
  scf.for K2 (FC = 128)                                  // 64 chunks
    per warp w: gate, up [:, 8w : +8]                    // one k loop, prefetch distance 3
    h = silu(gate) * up                [16, 128] f16     // registers -> shared, double-buffered; 1 barrier
    per warp w: acc[:, 128w : +128] += h @ Wd[128w : +128, K2]^T   // Wd groups double-buffered
  out[M] = residual + f16(acc)
```

The header comment of `mlp_fused_v2p3f.cu` still says 256 threads (8 warps);
`NWARPS` defaults to 16, and `h2p3f.mlir` launches 512 threads with
74,496 B of shared memory.

### Compiler changes (uncommitted, all tested)

- `GPURewriteMLPChainAsForall.cpp`: two new defaults for the IREE-generated
  kernel: the fused gate/up loop is unrolled ×2, and h is computed in
  registers. Together they're about 5–8% faster with bit-identical output.
- `FormDispatchRegions.cpp`: new `--iree-dispatch-creation-fuse-mlp-min-rows`
  flag. Gated MLP chains with a static row count below it are left unfused.
  The default 0 leaves behavior unchanged, including on x86; 2 leaves decode
  (M = 1) unfused.

### Building the fastest configuration

```
cd ~/workspace/tmp/gx10-cuda
./compile.sh <name> --iree-dispatch-creation-fuse-mlp=true \
  --iree-dispatch-creation-fuse-mlp-min-rows=2 \
  "--iree-hal-substitute-executable-source=prefill_bs1\$async_dispatch_28=$HOME/workspace/tmp/gx10-hand/h2p3f.mlir"
```

`~/workspace/tmp/gx10-hand/mk.sh <name> <ptx> <threads> <smem_bytes>`
regenerates the substitution executable (`<name>.mlir`, here `h2p3f.mlir`)
from the kernel's PTX and compiles a vmfb with it. `h2p3f.mlir` was made from
`v2p3f.ptx` with 512 threads and 74,496 B.

### First version (`mlp_fused.cu`, 256 threads)

The hand-written kernel beats the baseline clearly (51.7 vs 86.3 ms at S=512, 147.6 vs 346.6 ms at S=2048), with output close to the
baseline (top-1 agreement 100% and 99.7%).

| | S=512 | S=2048 |
|---|---:|---:|
| Baseline gate + up + down | 86.3 ms | 346.6 ms |
| Hand-written fused MLP kernel | 51.7 ms (0.60×) | 147.6 ms (0.43×) |
| Top-1 agreement vs baseline | 100% | 99.7% |

A CUDA kernel replaces the fused MLP dispatch (`prefill_bs1$async_dispatch_28`)
through `--iree-hal-substitute-executable-source`; the compiler-generated kernel is
not used. It follows the IREE ABI of that dispatch (4 constants, 5 bindings),
so the rest of the model is unchanged.

```
scf.forall M (TM = 16)                                   // grid M / 16, 256 threads = 8 warps
  x[M] -> shared                       [16, 2048] f16    // once, k relabeled (below)
  acc = 0                              [16, 2048] f32    // REGISTERS: warp w owns cols 256w : +256
                                                         //   32 MMA tiles = 128 f32 per thread
  scf.for K2 (FC = 128)                                  // 64 chunks
    per warp w: gate, up [:, 16w : +16]                  // one k loop for both, step 32
      Wg/Wu fragments: ld.global.nc 16 B, prefetched one k step ahead
      4 mma.sync per k16 step share one x A-fragment (ldmatrix)
    h = silu(gate) * up                [16, 128] f16     // shared, double-buffered; 1 barrier
    per warp w: acc[:, 256w : +256] += h @ Wd[256w : +256, K2]^T
      Wd fragments straight from global (ld.global.nc 16 B), 64 mma.sync per k32 step
  out[M] = residual + f16(acc)                           // registers -> global
```

First version against the compiler-generated default:

| | Compiler default | Hand-written |
|---|---|---|
| N split | 2 × 1024 (gate/up computed 2×) | none (gate/up computed 1×) |
| `acc` | shared memory, 64 KiB, round trip every (N, k2) step | registers, never leaves the warp |
| Shared memory | 84 KiB (acc, gate, up, h) | 74,496 B (x tile 65,792 B + 2 h buffers 8,704 B) |
| gate / up | two separate k loops | one k loop, A-fragment shared by 4 MMAs |
| Weight loads | `map_load` into registers | 16 B `ld.global.nc`, gate/up prefetched one step ahead |
| Barriers per F chunk | around gate/up/h and every (N, k2) step | 1 (h double-buffered) |
| Workgroups at S = 512 | 64 | 32 |

- **k relabeling.** In an m16n8k16 B fragment, lane `t = lane % 4` needs k
  `{2t, 2t+1, 2t+8, 2t+9}` (+16 for the second k16 step). The kernel relabels
  k inside every 32-wide block so these are physical k `[8t, 8t+8)`. Each lane
  then loads both k16 steps' B fragments with one 16 B load straight from the
  weights in global memory. x and h in shared memory are stored in the same
  relabeled order, so the dot products are unchanged; only the f32 summation
  order inside a block differs.
- **Registers.** `__launch_bounds__(256, 1)`: one 256-thread block per SM, up
  to 255 registers per thread, of which 128 hold `acc`.
- **Numerics.** silu and the gate/up rounding are done in f16 step by step,
  as in the IREE dispatch. The different summation order is why S = 2048
  top-1 is 99.7% rather than bit-identical.
- **Limits.** Shapes are fixed for llama-3.2-1B (H = 2048, F = 8192); M must
  be a multiple of 32. Prefill only.

Files: x86 `~/workspace/tmp/gx10-hand/` (`mlp_fused.cu`, `mlp_fused.ptx`,
`mlp_exe.mlir` = substitution executable); `~/workspace/tmp/gx10-cuda/`
(`hand1.vmfb`, `hand1.compile.log`; outputs `cmp/out_hand1_{512,2048}.npy`).
