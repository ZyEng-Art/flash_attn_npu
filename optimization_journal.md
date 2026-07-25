# Flash Attention Triton Optimization Journal

## 2026-07-25 - Stable baseline after correctness fix

- Commit: pending at time of writing.
- Optimization point: baseline stabilization, not a latency optimization point.
- Content:
  - Removed launch-time dynamic AICore detection from `_get_persistent_programs()`.
  - Kept `DEFAULT_PERSISTENT_PROGRAMS = 20` as the default path unless `PERSISTENT_PROGRAMS` is explicitly set.
  - Preserved the existing tiling, `tl.constexpr`, and modulo-removal performance changes.
- Effect:
  - Public correctness: `18/18`, `40.000 / 40.000`.
  - Full local score: `59.9605 / 100`.
  - Performance: `6/6`, `19.9605 / 60`, mean speedup `0.3326757973861712`.
  - Custom submit-order regression: `21/21`, including the previously crashing first case
    `Z=1 H=2 N_CTX=1024 HEAD_DIM=64 causal=False dtype=float16`.
- Issues:
  - Dynamic core querying via `torch_npu.npu.current_device()` can trigger NPU runtime cold-start before the first evaluated kernel and matches the reported `LazySetDevice` / `507033` failure.
  - The latency optimizer checklist recommends dynamic core count, but this evaluator is sensitive to pre-launch device initialization. Correctness gate takes priority.
- Reports:
  - `evaluation_reports/codex_fix_correctness/evaluation_report.json`
  - `evaluation_reports/codex_fix_final/evaluation_report.json`
