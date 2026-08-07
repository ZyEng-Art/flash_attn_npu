# Isolated Case3 Wide-BN Retest

## Scope

- Date: 2026-08-05
- Experiment directory: `/workspace/new_attn/isolated_case3_retest_20260805_0747_5257`
- Optimization point: 18, kernel family / shape-specific tiling specialization
- Baseline: `baseline_current.py`, copied from `/workspace/new_attn/flash_attention_forward.py`
- Candidate: `candidate_case3_widebn.py`
- Candidate-only code change:
  - `(128, 8, 2048, 128, True): (128, 64)` -> `(64, 256)`

## Isolation Measures

- Used a new experiment directory instead of the shared top-level `evaluation_reports_*` paths.
- Used a copied `evaluate_attention.py` from the same directory.
- Used independent `TRITON_CACHE_DIR` directories for correctness, baseline performance, and candidate performance.
- Checked `npu-smi info` before performance runs:
  - AICore: `0%`
  - NPU process table: no running processes

## Correctness

Candidate correctness passed:

- Report: `reports/candidate_correctness/evaluation_report.json`
- Result: `40.000 / 40.000`
- Passed: `18 / 18`

## Performance

Command shape:

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
| baseline_current.py | 24.837686 | 0.413961 |
| candidate_case3_widebn.py | 24.165060 | 0.402751 |
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

Candidate-side CV for the decisive case3 was low:

- baseline route case3 candidate CV: `0.0024`
- wide-BN route case3 candidate CV: `0.0048`

## Conclusion

The case3 wide-BN preset is a real regression in this isolated retest. The score drop is not explained by report directory collision or unrelated shape noise: the candidate changes only case3 routing, and case3 itself slows down by `27.96%` with low CV.

Do not merge this preset change. Keep the current `(128, 64)` lazy explicit-QK-scratch route for `(128,8,2048,128, causal=True)`.
