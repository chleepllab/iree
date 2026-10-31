# Gated-MLP fusion (`--iree-dispatch-creation-fuse-mlp`)

Fuses the gated MLP of a transformer layer into one dispatch so the two
`[S, ffn]` intermediates (`gate`, `up`) are never materialized in the transient
arena. Only the LLVMCPU backend has the codegen side.

```
gate = x * Wg^T            [M, F] f32      (F = ffn width, 8192 for llama-3.2-1b)
up   = x * Wu^T            [M, F] f32
h    = silu(gate) * up     [M, F] f16
down = h * Wd^T            [M, N] f32      (root)
out  = epilogue(down, ...) [M, N]          (trunc + residual add)
```

## Flags

All custom work is off by default; with no flags the compiler behaves as
upstream IREE. The switches live in `compiler/src/iree/compiler/Utils/CustomFusion.{h,cpp}`.

| Flag | Turns on |
|---|---|
| `--iree-dispatch-creation-fuse-mlp=true` | MLP dispatch formation, `RewriteMLPChainAsForall`, canonicalize/CSE after it, custom slice packing |
| `--iree-dispatch-creation-fuse-conv-chain=true` | conv -> conv / conv -> maxpool chain fusion, conv pad-into-producer fusion, conv rewrite passes, conv tile-and-decompose passes, custom slice packing |
| `--iree-stream-custom-packing=true` | custom static slice packer only |

Pass both fusion flags to enable all optimizations. `fuse-mlp` does not turn
on any of the conv chain work.

Recommended: also pass `--iree-llvmcpu-stack-allocation-limit=131072` (see
[Row tile TM](#row-tile-tm)).

```
iree-compile llama-3.2-1b-instruct-f16.mlir -o llama.vmfb \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-dispatch-creation-fuse-mlp=true \
    --iree-llvmcpu-stack-allocation-limit=131072
```

## Where it happens

| Stage | File | What |
|---|---|---|
| Dispatch creation | `DispatchCreation/FormDispatchRegions.cpp` (`tryFuseMLPChain`) | Pulls gate/up matmuls and the activation into the `down` matmul's dispatch, tags the root `iree_dispatch.special_mlp` |
| Codegen | `Codegen/Common/RewriteMLPChainAsForall.cpp` | Tiles the fused dispatch (below), tags the function `iree_codegen.mlp_chain_rewritten` |
| Codegen | `Codegen/Common/TileDispatchUsingForall.cpp` | Skips functions tagged `iree_codegen.mlp_chain_rewritten` |
| Codegen | `Codegen/LLVMCPU/Passes.cpp` | Adds `RewriteMLPChainAsForall` + canonicalize/CSE before workgroup tiling when the flag is on |
| Stream | `Dialect/Stream/Transforms/LayoutSlices.cpp` | Custom static slice packer (no effect on llama, see results) |

## Loop structure

Same M -> K2 -> N nesting as Hexagon's `HexagonMatmulFusionPass` (K2 = F):
fc1 (gate/up/silu) has no N axis and sits above the N loop, so it is computed
once per (M, F) tile, not once per N tile.

```
scf.forall m (TM rows)                               // step 1: M, workgroups
  acc = 0                       [TM, N] f32
  scf.for f (Fc = 128)                               // step 2: F, gate/up/h fused in
    gate, up = x[m] @ Wg/Wu[f]^T  [TM, Fc] f32
    h = silu(gate) * up           [TM, Fc] f16       // elementwise
    scf.for n (vecN) / k (vecK)                      // step 3: down tiled alone
      acc[0:TM, n] += h[0:TM, k] @ Wd[n, k]^T
  out[m] = epilogue(acc)
```

Shapes (llama-3.2-1b, prefill S=512): x `[M, N]` = `[512, 2048]` f16;
Wg, Wu `[F, N]` = `[8192, 2048]` f16 and Wd `[N, F]` = `[2048, 8192]` f16 as
stored (`[out, in]`; "^T" is the reduction over the second dim, no copy is
made); gate/up/h `[M, F]`; down/out `[M, N]`. Decode is M = 1.

Tile sizes come from `down`'s `lowering_config` (KernelDispatch's choice for
the target), falling back to the constants in the pass when a level is
missing:

| Size | Source | Fallback | x86 (AVX-512) | riscv64 (VLEN 256) |
|---|---|---|---:|---:|
| TM (max) | `distribution[0]` | `kVecM` = 8 | 8 | 8 |
| Fc | `cache_reduction[2]`, rounded up to `lcm(vecK, gate's vecN)` | `kFfnChunk` = 128 | 128 | 128 |
| vecM x vecN x vecK | `vector_common_parallel` / `vector_reduction` / `vector_inner_parallel` | 8 x 16 x 16 | 8 x 16 x 16 | 8 x 32 x 1 |

There is no m' loop in practice: `down`'s M vector tile (vecM = 8) is at
least TM (TM <= `distribution[0]` = 8, and 2 at the default stack limit, 1 for
decode), so that loop has one iteration and is canonicalized away; all TM rows
go through each vector op. It would only iterate if a config had
`distribution[0]` larger than vecM.

Differences from Hexagon:

| | Hexagon | IREE `fuse-mlp` |
|---|---|---|
| M loop | `scf.for` | `scf.forall` mapped to workgroups |
| M + K2 | one tile-and-fuse (`{tileM, tileK, 0}`) | two calls: M, then F with gate/up/h fused |
| N loop | tile-and-fuse of fc2 over N | `down` tiled alone to its vector tile (n, k loops) |
| Producers | one matmul + GELU | two matmuls (gate, up) + SiLU*mul |

Every op is tiled alone to vector size and its `lowering_config` dropped;
otherwise the vector-level tile-and-fuse passes would anchor on `down` and
fuse `h` into each N tile, recomputing the chunk N/vecN times.

## Row tile TM

`pickRowTile` picks the largest power of two <= `down`'s `distribution[0]`
(8; `kVecM` if unset) whose stack buffers fit in 3/4 of the target's
`max_stack_allocation_size`:

- bytes per row = N*4 (f32 acc) + Fc*(4+4+2) (gate, up f32; h f16) = 2048*4 + 1280 = 9,472 B
- 32 KiB default limit -> budget 24,576 B -> **TM = 2** (TM=4 needs 37,888 B)
- 128 KiB limit -> budget 98,304 B -> **TM = 8** (75,776 B)

Each workgroup streams all of Wg, Wu, Wd for its TM rows, so TM sets how many
times the MLP weights are read per layer: S/TM = 256 passes at TM=2, 64 at
TM=8. At TM=2 fused prefill is ~1.3 s slower than baseline; TM=8 removes it.

## Results

Host: AMD Ryzen 9 9950X (32 threads), `--device=local-task`.
Model: llama-3.2-1b-instruct f16, `prefill_bs1`, S=512, KV cache 128x524288 f16.
Transient buffer size is captured at runtime with a gdb breakpoint on
`iree_hal_device_queue_alloca` (sizes are dynamic, not readable from the IR);
`HOST_LOCAL`/`DEVICE_LOCAL` from `--print_statistics` do not include it.

### Baseline / custom packing / custom fusion (default 32 KiB stack, TM=2)

| LLaMA-3.2-1B f16, prefill S=512 | Baseline (no flags) | Custom packing (`--iree-stream-custom-packing=true`) | Custom fusion (`--iree-dispatch-creation-fuse-mlp=true`) |
|---|---:|---:|---:|
| Transient buffer (activations) | 45,353,344 B | 45,353,344 B | 28,576,128 B |
| Max RSS (3 runs) | 2,872,576–2,872,832 KB | 2,872,064–2,873,344 KB | 2,855,680–2,856,192 KB |
| `HOST_LOCAL` peak (`--print_statistics`) | 265,551,872 B | 265,551,872 B | 265,551,872 B |
| `DEVICE_LOCAL` peak (`--print_statistics`) | 134,222,088 B | 134,222,088 B | 134,222,088 B |
| Minor page faults | ~766,580 | ~766,360 | ~758,430 |
| Dispatch ops (prefill / whole module) | 363 / 822 | 363 / 822 | 331 / 758 |
| Number of executables (prefill / whole module) | 45 / 122 | 45 / 122 | 43 / 118 |
| Wall time (3 runs) | 2.21–2.37 s | 2.17–2.23 s | 3.47–3.64 s |
| Output (1×512×128256 logits) | — | bit-identical to baseline | bit-identical to baseline |

- Transient buffer: −16,777,216 B (exactly 16 MiB, −37.0%) = the two
  `[512, 8192]` f16 MLP intermediates. Saving scales linearly with S.
- `HOST_LOCAL` is the logits output, `DEVICE_LOCAL` the KV cache; neither
  includes the transient arena, so both are unchanged.
- Custom packing has no effect on llama: its transient slices are
  dynamically sized and the custom packer only handles static slices.
- 2 fewer dispatches per layer (16 layers unrolled): 363 -> 331 in prefill.

### Effect of TM (stack limit)

| LLaMA-3.2-1B f16, prefill S=512 | Baseline (32 KiB stack) | Baseline (128 KiB stack) | Custom fusion, TM=2 (32 KiB stack) | Custom fusion, TM=8 (128 KiB stack) |
|---|---:|---:|---:|---:|
| Transient buffer (activations) | 45,353,344 B | 45,353,344 B | 28,576,128 B | 28,576,128 B |
| Max RSS (3 runs) | 2,872,320–2,872,832 KB | 2,872,064–2,872,832 KB | 2,855,936–2,856,192 KB | 2,856,448–2,856,960 KB |
| `HOST_LOCAL` peak | 265,551,872 B | 265,551,872 B | 265,551,872 B | 265,551,872 B |
| `DEVICE_LOCAL` peak | 134,222,088 B | 134,222,088 B | 134,222,088 B | 134,222,088 B |
| Minor page faults | ~767,030 | ~766,570 | ~758,270 | ~758,310 |
| Wall time (3 runs) | 2.33–2.48 s | 2.17–2.49 s | 3.61–3.69 s | **2.14–2.31 s** |
| Output | — | bit-identical | bit-identical | bit-identical |

- The stack limit alone does not change the baseline.
- TM=8 brings fused prefill back to baseline speed (slightly faster) with the
  same 16 MiB transient saving. Max RSS is ~0.5–0.75 MiB above TM=2 from the
  larger per-thread stack buffers.

### GX10 (aarch64)

Host: ASUS GX10, Cortex-X925 + Cortex-A725 (20 cores), `--device=local-task`.
Cross-compiled on x86 with `--iree-llvmcpu-target-triple=aarch64-unknown-linux-gnu
--iree-llvmcpu-target-cpu=cortex-x925`; runtime-only build (gcc 13) on the
GX10. Same model, function, S and KV cache as above.

| LLaMA-3.2-1B f16, prefill S=512 | Baseline (32 KiB stack) | Custom fusion, TM=2 (32 KiB stack) | Baseline (128 KiB stack) | Custom fusion, TM=8 (128 KiB stack)* |
|---|---:|---:|---:|---:|
| Transient buffer (activations) | 45,353,344 B | 28,576,128 B | 45,353,344 B | 28,576,128 B |
| Max RSS (3 runs) | 2,867,260–2,867,904 KB | 2,850,944–2,851,268 KB | 2,867,256–2,867,424 KB | 2,851,736–2,852,068 KB |
| `HOST_LOCAL` peak | 265,551,872 B | 265,551,872 B | 265,551,872 B | 265,551,872 B |
| `DEVICE_LOCAL` peak | 134,222,088 B | 134,222,088 B | 134,222,088 B | 134,222,088 B |
| Minor page faults | ~759,660 | ~752,890 | ~759,760 | ~753,320 |
| Wall time (3 runs) | 13.03–13.10 s | 17.46–17.64 s | 12.82–12.99 s | 17.08–17.17 s |
| Output | — | bit-identical | bit-identical | bit-identical |

\* Compiled with `--iree-llvmcpu-fail-on-large-vector=false`; see Open items.

- Transient buffer: the same −16,777,216 B (−37.0%) as on x86, byte for byte.
  Max RSS is ~16.1 MiB lower at TM=2 and ~15.6 MiB lower at TM=8.
- Fused prefill is ~34% slower at TM=2 and ~32% slower at TM=8. Unlike x86,
  TM=8 does not recover baseline speed.
- The baseline itself is ~5.6x slower than on the 9950X (13 s vs 2.3 s), so
  aarch64 f16 matmul codegen dominates; not investigated.

### Banana Pi BPI-F3 (riscv64)

Host: SpacemiT K1, 8x SpacemiT X60 (RVV 1.0, VLEN 256), `--device=local-task`.
Cross-compiled on x86 with `--iree-llvmcpu-target-triple=riscv64
--iree-llvmcpu-target-abi=lp64d
--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+zvl256b,+v,+zfh,+zvfh` and
`--iree-llvmcpu-stack-allocation-limit=131072`; runtime from
`~/iree_c-build-riscv`. Same model, function, S and KV cache as above. One run
each (a prefill takes 21–27 minutes).

| LLaMA-3.2-1B f16, prefill S=512 | Baseline (128 KiB stack) | Custom fusion, TM=8, fixed 8x16x16 tile | Custom fusion, TM=8, config tile 8x32x1 |
|---|---:|---:|---:|
| Transient buffer (activations) | 45,353,344 B | 28,576,128 B (−16 MiB) | 28,576,128 B (−16 MiB) |
| Max RSS | 2,858,624 KB | 2,842,496 KB (−15.7 MiB) | 2,842,240 KB (−16.0 MiB) |
| `HOST_LOCAL` peak | 265,551,872 B | 265,551,872 B | 265,551,872 B |
| `DEVICE_LOCAL` peak | 134,222,088 B | 134,222,088 B | 134,222,088 B |
| Minor page faults | 814,438 | 806,552 | 813,195 |
| Wall time | 1,643 s (27.4 min) | 1,277 s (21.3 min, −22%) | 1,902 s (31.7 min, +16%) |
| Output | — | bit-identical | bit-identical |

The fixed-tile column is the pass before tile sizes were read from
`lowering_config`; the last column is the current pass.

- Transient buffer: the same −16,777,216 B (−37.0%) as on x86 and aarch64.
  Max RSS is ~15.7 MiB lower.
- With the fixed 8x16x16 tile, fused prefill is faster than the baseline
  (−366 s); with the config tile (8x32x1, i.e. `down` reduced one k at a time)
  it is slower (+259 s). One run each on a shared board (load average 3–5
  during the last run), so the gap is not confirmed; see Open items.
- TM=8 compiles without `--iree-llvmcpu-fail-on-large-vector=false` on this
  target.
- gdb: `allocation_size` is in `$a6` (RISC-V LP64D; the semaphore lists and
  params are passed by reference, as on aarch64 where it is `$x6`).

## Reproducing

`~/workspace/run_llama.sh [SEQ_LEN]` compiles and runs prefill + decode; add
the flags above to its `iree-compile` line. Transient size:

```
# g.gdb
set pagination off
set breakpoint pending on
break *iree_hal_device_queue_alloca
commands
silent
printf "ALLOCA %lu\n", $rcx
continue
end
run
```

```
gdb -q -batch -x g.gdb --args iree-run-module --module=llama.vmfb ... --function=prefill_bs1 ...
```

`$rcx` is `allocation_size` at function entry (x86-64 SysV; the semaphore
lists and params are passed in memory). The largest alloca after
`EXEC @prefill_bs1` other than the 265,551,872 B logits buffer is the
transient arena.

## Open items

- TM=8 needs `--iree-llvmcpu-stack-allocation-limit=131072` today. The pass
  could choose TM against its own budget instead of the target default.
- Decode (M=1) also goes through the rewrite; not measured.
- riscv64: rerun the fixed-tile build under the same board load to tell
  whether the config tile (vecK = 1) is really slower. If it is, keep TM and
  Fc from the config but use a larger vecK for `down` on RISC-V.
- The x86 tile is unchanged by the config-driven sizes (output bit-identical
  to the fixed-tile build); the aarch64 tile was not re-checked.
