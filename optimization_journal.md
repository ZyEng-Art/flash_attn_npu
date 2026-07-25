# Flash Attention Triton Optimization Journal

## 2026-07-25 - Stable baseline after correctness fix

- Commit: `abc6cf7`
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

## 2026-07-25 - Optimization point 3: persistent program count sweep

- Commit: `f690f21`
- Optimization point: 3, vector/core partitioning.
- Content:
  - Swept `PERSISTENT_PROGRAMS` values `16`, `20`, `24`, and `32` on the six local performance shapes.
  - Kept `DEFAULT_PERSISTENT_PROGRAMS = 20` unchanged because it was consistently the best median latency in this environment.
  - Did not reintroduce dynamic AICore probing because the earlier `torch_npu.npu.current_device()` path matched the submit-side `LazySetDevice` / `507033` first-test crash.
- Effect:
  - `(128, 8, 1024, 128, causal=True)`: P16 `14693.565 us`, P20 `11975.670 us`, P24 `18568.215 us`, P32 `14257.405 us`.
  - `(128, 8, 1024, 256, causal=True)`: P16 `22028.780 us`, P20 `18569.455 us`, P24 `28442.325 us`, P32 `21939.690 us`.
  - `(128, 8, 2048, 128, causal=True)`: P16 `48169.860 us`, P20 `38974.120 us`, P24 `66674.405 us`, P32 `45966.135 us`.
  - `(128, 8, 2048, 256, causal=False)`: P16 `90849.515 us`, P20 `73823.815 us`, P24 `113447.831 us`, P32 `93169.055 us`.
  - `(128, 8, 4096, 128, causal=False)`: P16 `220966.380 us`, P20 `171107.485 us`, P24 `274473.700 us`, P32 `218354.495 us`.
  - `(128, 8, 8192, 64, causal=False)`: P16 `764178.215 us`, P20 `631072.375 us`, P24 `999213.620 us`, P32 `747206.750 us`.
  - Result: no code change. P20 remains the best measured launch partition across the local performance set.
- Issues:
  - The latency optimizer checklist recommends dynamic core count, but device-property access can initialize the NPU before the evaluator's first kernel and has already caused a correctness-gate failure in submit ordering.
  - P24 and P32 are strongly negative on all tested performance shapes; P16 is consistently slower than P20.
- Verification:
  - No code changed in this optimization point; this entry records the profiling result and rejects the negative alternatives.

## 2026-07-25 - Optimization point 2: tiling sweep around tuned presets

- Commit: pending at time of writing.
- Optimization point: 2, tiling optimization.
- Content:
  - Swept targeted `(BLOCK_M, BLOCK_N)` candidates around the existing presets for all six performance shapes.
  - Kept `DEFAULT_TILING_PRESETS` unchanged because the current presets were best or statistically tied within noise.
  - Checked candidate output against `torch_npu.npu_fusion_attention` before timing each viable tiling.
- Effect:
  - `(128, 8, 1024, 128, causal=True)`: current `(128,64)` `11645.710 us`; `(64,128)` `12842.185 us`; `(64,256)` `11806.495 us`; `(128,128)` and `(128,256)` failed MLIR compilation.
  - `(128, 8, 1024, 256, causal=True)`: current `(64,128)` `18112.785 us`; `(128,64)` `18132.870 us`; `(64,64)` `20702.840 us`; `(32,128)` `29881.350 us`; `(64,256)` failed MLIR compilation.
  - `(128, 8, 2048, 128, causal=True)`: current `(64,256)` `37731.721 us`; `(128,64)` `40085.370 us`; `(64,128)` `43334.230 us`; `(128,128)` and `(128,256)` failed MLIR compilation.
  - `(128, 8, 2048, 256, causal=False)`: current `(128,128)` `73214.190 us`; `(64,128)` `93422.750 us`; `(128,64)` `112174.865 us`; `(64,256)` and `(128,256)` failed MLIR compilation.
  - `(128, 8, 4096, 128, causal=False)`: current `(128,256)` `172015.770 us`; `(128,128)` `205168.070 us`; `(64,256)` `216740.774 us`; `(64,128)` `302766.805 us`; `(256,64)` failed MLIR compilation.
  - `(128, 8, 8192, 64, causal=False)`: current `(128,256)` `617161.525 us`; `(256,64)` `685005.235 us`; `(128,128)` `757361.920 us`; `(64,256)` `879657.120 us`; `(256,128)` failed MLIR compilation.
  - Result: no code change. The current tiling table remains the best measured table in the targeted sweep.
- Issues:
  - Large square or very wide tiles often fail during MLIR lowering because the live `qk` and accumulator footprint overflows local storage.
  - Some alternatives are close on one shape, but lose clearly on adjacent shapes or require a path that fails compilation elsewhere.
- Verification:
  - Viable candidates were output-checked against the NPU fusion attention baseline with `atol=1e-2`, `rtol=1e-2`.
  - No code changed in this optimization point.
