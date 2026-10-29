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
| `--iree-codegen-gpu-mlp-mma` | true | MMA path; false gives the scalar path (vmfb byte-identical to before) |
| `--iree-codegen-gpu-mlp-mma-n-tile` | 512 | N per workgroup; gate/up recomputed N / nTile times |
| `--iree-codegen-gpu-mlp-mma-reduction-k` | 2 | gate/up K tile, in intrinsics (x16) |
| `--iree-codegen-gpu-mlp-mma-promote` | true | gate/up operands (x, weight tile) to shared memory |
| `--iree-codegen-gpu-mlp-row-tile` | 0 | scalar path only: force TM (0 = from the shared-memory budget) |

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
