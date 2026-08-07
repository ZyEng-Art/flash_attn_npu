# Lazy-GM pair-accumulator experiment

Date: 2026-08-05

Candidate:

```text
/workspace/new_attn/experiment_lazy_gm_pair/candidate_case6_pair_acc.py
```

Target shape:

```text
(Z=128, H=8, N_CTX=8192, HEAD_DIM=64, causal=False), BM=128, BN=256
```

Change:

- Added `_attn_fwd_inner_loop_lazy_gm_pair_acc`.
- Routed only the D64 non-causal lazy-GM path.
- The candidate processes two `BLOCK_N=256` K/V blocks per loop and performs one GM accumulator load/store for the pair:
  `acc = acc + pv0 + pv1`.

Validation:

```bash
python3 -m py_compile experiment_lazy_gm_pair/candidate_case6_pair_acc.py
git diff --check -- experiment_lazy_gm_pair/candidate_case6_pair_acc.py
```

Both passed.

Performance command:

```bash
python3 profiling_runs/pipeline_fixp_diagnosis_20260804T062807Z/bench_one_shape.py \
  --candidate experiment_lazy_gm_pair/candidate_case6_pair_acc.py \
  --z 128 --h 8 --n-ctx 8192 --head-dim 64 --block-m 128 --block-n 256 \
  --warmup 3 --iters 12 --seed 18
```

Result:

| Variant | Output | candidate median us | Result |
|---|---:|---:|---|
| current source from `experiment_case6_ab` | match | 438196.285 | Baseline |
| pair-acc candidate | match | 445733.430 | Reject |

Conclusion: reducing GM accumulator load/store frequency by pairing two K/V blocks did not improve case6. The larger live range and altered schedule likely offset the saved GM traffic. Do not merge this candidate.
