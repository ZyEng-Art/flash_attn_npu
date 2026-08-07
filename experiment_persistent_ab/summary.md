# Case5 persistent program A/B

Date: 2026-08-05

Target shape:

```text
(Z=128, H=8, N_CTX=4096, HEAD_DIM=128, causal=False), BM=128, BN=256
```

This is the accepted lazy-GM + QK-fp16 special path.

All measurements used:

```bash
python3 profiling_runs/pipeline_fixp_diagnosis_20260804T062807Z/bench_one_shape.py \
  --z 128 --h 8 --n-ctx 4096 --head-dim 128 --block-m 128 --block-n 256 \
  --warmup 3 --iters 12 --seed 18
```

Results:

| PERSISTENT_PROGRAMS | launched programs | Output | candidate median us | Result |
|---:|---:|---:|---:|---|
| default 20 | 20 | match | 133478.140 | Keep |
| 16 | 16 | match | 165678.830 | Reject |
| 24 | 24 | match | 212517.035 | Reject |

Conclusion: the current default `DEFAULT_PERSISTENT_PROGRAMS = 20` remains best for the current case5 path. Do not add shape-specific P16/P24 routing for this case.
