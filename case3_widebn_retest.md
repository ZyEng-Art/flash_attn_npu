# Case3 Wide-BN Retest

## Summary

This document records the isolated retest for the case3 wide-BN tiling experiment.
The tested change was rejected because it is a reproducible performance regression.

- Date: 2026-08-05
- Experiment directory: `isolated_case3_retest_20260805_0747_5257/`
- Optimization point: 18, kernel family / shape-specific tiling specialization
- Baseline source: `flash_attention_forward.py` copied to `baseline_current.py`
- Candidate source: `candidate_case3_widebn.py`
- Candidate-only change:
  - `(128, 8, 2048, 128, True): (128, 64)` -> `(64, 256)`

## Isolation

The retest was isolated from shared report directories and Triton cache state:

- New experiment directory: `isolated_case3_retest_20260805_0747_5257/`
- Copied local evaluator: `isolated_case3_retest_20260805_0747_5257/evaluate_attention.py`
- Separate `TRITON_CACHE_DIR` values for correctness, baseline performance, and candidate performance
- NPU idle check before performance runs:
  - AICore: `0%`
  - Process table: no active NPU process

## Correctness

The candidate passed correctness before performance comparison:

- Report: `isolated_case3_retest_20260805_0747_5257/reports/candidate_correctness/evaluation_report.json`
- Score: `40.000 / 40.000`
- Passed cases: `18 / 18`

## Performance

The performance command shape was:

```bash
python3 evaluate_attention.py \
  --candidate <candidate_file> \
  --mode performance \
  --iters 80 \
  --warmup 10 \
  --report-dir <isolated_report_dir>
```

Top-level result:

| version | score / 60 | mean speedup |
|---|---:|---:|
| `baseline_current.py` | 24.837686 | 0.413961 |
| `candidate_case3_widebn.py` | 24.165060 | 0.402751 |
| delta | -0.672626 | -0.011210 |

Per-case candidate-side median comparison:

| case | shape | baseline candidate median us | wide-BN candidate median us | delta |
|---:|---|---:|---:|---:|
| 1 | `(128,8,1024,128, causal=True)` | 9321.020 | 9299.755 | -0.23% |
| 2 | `(128,8,1024,256, causal=True)` | 15739.175 | 15769.470 | +0.19% |
| 3 | `(128,8,2048,128, causal=True)` | 29893.270 | 38251.985 | +27.96% |
| 4 | `(128,8,2048,256, causal=False)` | 65665.365 | 65712.240 | +0.07% |
| 5 | `(128,8,4096,128, causal=False)` | 133541.615 | 133584.445 | +0.03% |
| 6 | `(128,8,8192,64, causal=False)` | 439451.875 | 438510.600 | -0.21% |

The decisive case3 comparison had low noise:

- Baseline route case3 candidate CV: `0.0024`
- Wide-BN route case3 candidate CV: `0.0048`

## Decision

Do not merge the case3 wide-BN preset. Keep the current `(128, 64)` lazy explicit-QK-scratch route for:

```text
(B=128, H=8, N=2048, D=128, causal=True)
```

The `(64, 256)` preset makes the target shape itself slower by `27.96%` in the isolated run, so the aggregate score drop is not explained by report-directory collision, cache contamination, or unrelated shape noise.

## Raw Reports

- Full isolated summary: `isolated_case3_retest_20260805_0747_5257/summary.md`
- Baseline performance JSON: `isolated_case3_retest_20260805_0747_5257/reports/baseline_performance_i80/evaluation_report.json`
- Baseline performance text: `isolated_case3_retest_20260805_0747_5257/reports/baseline_performance_i80/evaluation_report.txt`
- Candidate correctness JSON: `isolated_case3_retest_20260805_0747_5257/reports/candidate_correctness/evaluation_report.json`
- Candidate correctness text: `isolated_case3_retest_20260805_0747_5257/reports/candidate_correctness/evaluation_report.txt`
- Candidate performance JSON: `isolated_case3_retest_20260805_0747_5257/reports/candidate_performance_i80/evaluation_report.json`
- Candidate performance text: `isolated_case3_retest_20260805_0747_5257/reports/candidate_performance_i80/evaluation_report.txt`
