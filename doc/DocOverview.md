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
| `--iree-dispatch-creation-fuse-attention` | attention on IREE's own kernels: K / V / mask K2 dim blocking (MMA config on GPU), causal tile skipping, mask broadcast fold | — | [DocGPUFuseAttn.md](DocGPUFuseAttn.md) |

All three default to off. `fuse-mlp` and `fuse-qkv` also turn on the custom
slice packer (`--iree-stream-custom-packing`).

## CPU (LLVMCPU, x86)

- MLP fusion: [DocFuseMlp.md](DocFuseMlp.md).
- QKV fusion: [DocFuseQKV.md](DocFuseQKV.md).

## GPU (CUDA, GX10 / GB10 sm_121)

- MLP fusion: [DocGPUFuseMlp.md](DocGPUFuseMlp.md) (direct-path kernel, prefill
  faster than baseline with the 16 / 32 / 64 MiB transient saving).
- QKV fusion: [DocGPUFuseQKV.md](DocGPUFuseQKV.md).
- Attention: [DocGPUFuseAttn.md](DocGPUFuseAttn.md).

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
