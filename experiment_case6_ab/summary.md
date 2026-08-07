# Case6 lazy-GM branch A/B

Date: 2026-08-05

Target shape:

```text
(Z=128, H=8, N_CTX=8192, HEAD_DIM=64, causal=False), BM=128, BN=256
```

All measurements used:

```bash
python3 profiling_runs/pipeline_fixp_diagnosis_20260804T062807Z/bench_one_shape.py \
  --z 128 --h 8 --n-ctx 8192 --head-dim 64 --block-m 128 --block-n 256 \
  --warmup 3 --iters 12 --seed 18
```

Results:

| Variant | Runtime path | Output | candidate median us | Result |
|---|---|---:|---:|---|
| current | lazy-GM acc, `use_max=False`, `ACC_IN_UB=False` | match | 438196.285 | Keep |
| `FA_DISABLE_LAZY_GM_ACC_SPECIAL=1` | stable, resident acc, `use_max=True`, `ACC_IN_UB=True` | match | 571618.350 | Reject |
| `FA_USE_MAX=1` | stable, GM acc, `use_max=True`, `ACC_IN_UB=False` | compile fail | n/a | Reject |

The `FA_USE_MAX=1` branch failed during BiShengIR lowering with UB overflow:

```text
requires 2117632 bits while 1572864 bits available
```

Conclusion: the current case6 lazy-GM path is still the best of these source-level policy choices. Stable wide-BN variants either run much slower or do not compile.
