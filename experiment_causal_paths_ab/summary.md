# D128 causal pipeline path A/B

Date: 2026-08-05

Target shape:

```text
(Z=128, H=8, N_CTX=1024, HEAD_DIM=128, causal=True), BM=128, BN=64
```

All measurements used:

```bash
python3 profiling_runs/pipeline_fixp_diagnosis_20260804T062807Z/bench_one_shape.py \
  --z 128 --h 8 --n-ctx 1024 --head-dim 128 --causal --block-m 128 --block-n 64 \
  --warmup 3 --iters 20 --seed 18
```

Results:

| Variant | Output | candidate median us | Result |
|---|---:|---:|---|
| current default explicit QK scratch | match | 9306.120 | Keep |
| `FA_EXPLICIT_QK_SCRATCH=0` | match | 11670.015 | Reject |
| `FA_EXPLICIT_QK_SCRATCH=0 FA_PIPELINE_PREFETCH_KV=1` | match | 13872.735 | Reject |

Conclusion: for the current source, the default explicit-QK-scratch route remains the fastest tested D128 causal path. The older no-scratch and prefetch-kv pipeline variants should not be merged.
