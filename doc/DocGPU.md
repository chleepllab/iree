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

## Files changed (1 new file, 6 modified)

| File | Change |
|---|---|
| `Codegen/Common/GPURewriteMLPChainAsForall.cpp` (new, 558 lines) | The GPU pass |
| `Codegen/Common/Passes.td`, `Common/CMakeLists.txt` | Pass definition and source entry |
| `Codegen/Common/Passes.h` | Exports `isGatedMLPChainDown()` so config selection can use the matcher |
| `Codegen/LLVMGPU/Passes.cpp` | Adds the pass before workgroup tiling in TileAndFuse, under `#ifdef MY_PASS` / `isMLPFusionEnabled()` like the CPU side |
| `Codegen/LLVMGPU/KernelConfig.cpp` | Sends the MLP chain to TileAndFuse. Without this, decode (M=1) goes to VectorDistribute and the pass never runs |
| `Codegen/LLVMGPU/CMakeLists.txt` | `-DMY_PASS`, as in `LLVMCPU/CMakeLists.txt` |

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
| vmfb size | 677,389 B | 741,933 B |

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

- Speed: put the per-chunk gate/up and `down` matmuls on MMA (tensor cores);
  the shared-memory layout and memory saving would stay the same.
- TM=8 needs more than half of sm_121's workgroup memory (75,776 B of
  101,376 B); try a 3/4 budget once nothing else in the kernel needs shared
  memory.
- Upstream the runtime name-lookup fix (or make the compiler emit
  `$`-free export names).
- No lit test for `GPURewriteMLPChainAsForall` yet.
