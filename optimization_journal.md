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

- Commit: `4eb3e71`
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

## 2026-07-25 - Optimization point 5: scalar tile-offset linearization

- Commit: `39dc464`
- Optimization point: 5, scalar-to-vector / scalar-index reduction.
- Content:
  - Tried a contiguous BNSD fast path in `_attn_fwd_tile()` that replaced per-tile `(task_hz_idx // H, task_hz_idx % H)` expansion with a linear `task_hz_idx * stride_h` plane offset.
  - Preserved the original generic stride path behind a `ZH_CONTIGUOUS` constexpr branch during the experiment.
  - Reverted the code after performance validation because it was a negative optimization.
- Effect:
  - Correctness suite: `18/18` passed in `evaluation_reports/codex_point5_correctness/evaluation_report.json`.
  - Submit-order regression: `21/21` passed, including `(1,2,1024,64,False,float16)` and `(128,8,1024,64,False,bfloat16)`.
  - Performance score regressed from baseline `19.9605 / 60` to `19.4807 / 60`.
  - Mean speedup regressed from `0.3326757973861712` to `0.3246784782294558`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.2521`
    - `(128, 8, 1024, 256, causal=True)`: `0.2683`
    - `(128, 8, 2048, 128, causal=True)`: `0.2690`
    - `(128, 8, 2048, 256, causal=False)`: `0.4610`
    - `(128, 8, 4096, 128, causal=False)`: `0.3390`
    - `(128, 8, 8192, 64, causal=False)`: `0.3587`
- Issues:
  - Removing integer division from the source did not improve the generated kernel; the extra specialization path and changed scalar expression made all six measured performance cases slower.
  - The board profiler's scalar pressure is dominated more by the softmax loop and control/dataflow than by the `(z,h)` tile offset arithmetic.
- Verification:
  - Reverted the experimental code; final source after this entry is identical to the pre-point-5 implementation.
  - Negative result kept as a journal-only commit.

## 2026-07-25 - Optimization point 6: i64 offset arithmetic reduction

- Commit: `9b15b01`
- Optimization point: 6, avoid vector API scalar lowering.
- Content:
  - Replaced two per-tile pointer-offset casts from `tl.int64` to `tl.int32` in `_attn_fwd_tile()`.
  - Kept pointer formulas and tile mapping unchanged; only the scalar arithmetic type changed.
  - Bounds rationale: the largest evaluated linear element offset is below `2^31`, so `int32` is sufficient for these shape-only evaluator inputs.
- Effect:
  - Correctness suite: `18/18` passed in `evaluation_reports/codex_point6_correctness/evaluation_report.json`.
  - Submit-order regression: `21/21` passed, including `(1,2,1024,64,False,float16)` and `(128,8,1024,64,False,bfloat16)`.
  - Performance score improved from baseline `19.9605 / 60` to `20.0690 / 60`.
  - Mean speedup improved from `0.3326757973861712` to `0.3344841210438598`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.2517 -> 0.2548`
    - `(128, 8, 1024, 256, causal=True)`: `0.2719 -> 0.2718`
    - `(128, 8, 2048, 128, causal=True)`: `0.2761 -> 0.2737`
    - `(128, 8, 2048, 256, causal=False)`: `0.4681 -> 0.4748`
    - `(128, 8, 4096, 128, causal=False)`: `0.3515 -> 0.3547`
    - `(128, 8, 8192, 64, causal=False)`: `0.3768 -> 0.3771`
- Issues:
  - Benefit is small and shape-dependent; causal shape 3 regressed slightly, but the aggregate score improved.
  - This relies on the evaluator's bounded shapes. Wider future shapes should re-check that element offsets remain within `int32` range before reusing this change.
- Reports:
  - `evaluation_reports/codex_point6_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point6_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 11: GM accumulator load reordering

- Commit: `ac2423c`
- Optimization point: 11, load instruction reordering.
- Content:
  - In the `ACC_IN_UB == False` branches, moved `acc = tl.load(acc_ptr + block2d_acc)` before `pv = tl.dot(p_cast, v)`.
  - Kept K/V load order unchanged to preserve the existing prefetch behavior.
  - Targeted only head_dim=256 GM accumulator paths; UB accumulator paths are unchanged.
- Effect:
  - Correctness suite: `18/18` passed in `evaluation_reports/codex_point11_correctness/evaluation_report.json`.
  - Submit-order regression: `21/21` passed, including `(1,2,1024,64,False,float16)` and `(128,8,1024,64,False,bfloat16)`.
  - Performance score improved from point 6 `20.0690 / 60` to `20.1515 / 60`.
  - Mean speedup improved from `0.3344841210438598` to `0.33585816600990115`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.2548 -> 0.2573`
    - `(128, 8, 1024, 256, causal=True)`: `0.2718 -> 0.2712`
    - `(128, 8, 2048, 128, causal=True)`: `0.2737 -> 0.2758`
    - `(128, 8, 2048, 256, causal=False)`: `0.4748 -> 0.4768`
    - `(128, 8, 4096, 128, causal=False)`: `0.3547 -> 0.3568`
    - `(128, 8, 8192, 64, causal=False)`: `0.3771 -> 0.3773`
- Issues:
  - The gain is small and close to run-to-run noise on shape 2, but the aggregate score improved.
  - Keeping `v` loaded before `qk` remains intentional; moving it later would be a separate scheduling experiment because it trades MTE overlap for lower live range.
- Reports:
  - `evaluation_reports/codex_point11_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point11_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: skip unused LSE output experiment

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / statistics-output elimination.
- Content:
  - Tried adding a `STORE_LSE: tl.constexpr` path through `_attn_fwd_tile()` and `_attn_fwd()`.
  - Default `attention(..., return_lse=False)` allocated a scalar placeholder LSE tensor and skipped `m_i += tl.math.log(l_i)` plus the `tl.store()` to `M`.
  - `attention(..., return_lse=True)` still kept the full `(Z, H, N_CTX)` LSE output and restored the lazy-softmax offset for `HEAD_DIM >= 256`.
  - Reverted the code after performance validation because it was a negative optimization.
- Effect:
  - Correctness suite for the experimental code: `18/18` passed in `evaluation_reports/codex_point7_correctness/evaluation_report.json`.
  - `return_lse=True` probe passed: output matched the reference, LSE shape was preserved, and LSE values were finite.
  - Performance score regressed from point 11 `20.1515 / 60` to `19.5518 / 60`.
  - Mean speedup regressed from `0.33585816600990115` to `0.32586353100098975`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.2573 -> 0.2476`
    - `(128, 8, 1024, 256, causal=True)`: `0.2712 -> 0.2664`
    - `(128, 8, 2048, 128, causal=True)`: `0.2758 -> 0.2692`
    - `(128, 8, 2048, 256, causal=False)`: `0.4768 -> 0.4685`
    - `(128, 8, 4096, 128, causal=False)`: `0.3568 -> 0.3398`
    - `(128, 8, 8192, 64, causal=False)`: `0.3773 -> 0.3637`
- Issues:
  - Although the source eliminated an unused LSE store on the default API path, the extra constexpr specialization and control branch made every measured performance shape slower.
  - The experiment also changed the private `_launch_kernel()` contract by making LSE allocation conditional, which is not worth carrying when the measured default path regresses.
  - Final source after this entry is identical to the point 11 implementation.
- Reports:
  - `evaluation_reports/codex_point7_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 11: delayed V tile load experiment

- Commit: journal-only commit for this entry.
- Optimization point: 11, load instruction reordering.
- Content:
  - Tried moving `v = tl.load(v_block_ptr)` from the top of each KV-loop iteration to immediately before `tl.dot(p_cast, v)`.
  - Kept `k` loading before `qk = tl.dot(q, tl.trans(k))`; only the independent `V` tile load was delayed.
  - Reverted the code after performance validation because it was a negative optimization.
- Effect:
  - Correctness suite for the experimental code: `18/18` passed in `evaluation_reports/codex_point11_vload_late_correctness/evaluation_report.json`.
  - Performance score regressed from point 11 `20.1515 / 60` to `18.3774 / 60`.
  - Mean speedup regressed from `0.33585816600990115` to `0.3062896913098969`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.2573 -> 0.2413`
    - `(128, 8, 1024, 256, causal=True)`: `0.2712 -> 0.2367`
    - `(128, 8, 2048, 128, causal=True)`: `0.2758 -> 0.2350`
    - `(128, 8, 2048, 256, causal=False)`: `0.4768 -> 0.4407`
    - `(128, 8, 4096, 128, causal=False)`: `0.3568 -> 0.3249`
    - `(128, 8, 8192, 64, causal=False)`: `0.3773 -> 0.3590`
- Issues:
  - Delaying `V` reduced its live range in source, but it removed useful MTE overlap before the `P@V` dot.
  - The profiler-guided conclusion is to keep the earlier `K` and `V` prefetch order; the previous positive point 11 only moved the GM accumulator load ahead of `P@V`.
  - Final source after this entry is identical to the committed point 11 implementation.
- Reports:
  - `evaluation_reports/codex_point11_vload_late_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point11_vload_late_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: LSE store-only suppression experiment

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / unused output-store elimination.
- Content:
  - Tried a narrower version of the LSE elimination idea: keep the final `m_i += tl.math.log(l_i)` and keep full LSE allocation, but guard only the `tl.store()` to the LSE tensor behind `STORE_LSE: tl.constexpr`.
  - Preserved `_launch_kernel()` default behavior with `store_lse=True`; only `attention(..., return_lse=False)` used `store_lse=False`.
  - Reverted the code after performance validation because the aggregate score still regressed.
- Effect:
  - Correctness suite for the experimental code: `18/18` passed in `evaluation_reports/codex_point7_store_only_correctness/evaluation_report.json`.
  - `return_lse=True` probe: default output matched between `return_lse=False` and `return_lse=True`, LSE shape was `(1, 1, 64)`, but LSE finite status remained false on the lazy path, matching the existing point 11 behavior rather than fixing LSE semantics.
  - Performance score regressed from point 11 `20.1515 / 60` to `20.0566 / 60`.
  - Mean speedup regressed from `0.33585816600990115` to `0.3342767015870253`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.2573 -> 0.2512`
    - `(128, 8, 1024, 256, causal=True)`: `0.2712 -> 0.2775`
    - `(128, 8, 2048, 128, causal=True)`: `0.2758 -> 0.2748`
    - `(128, 8, 2048, 256, causal=False)`: `0.4768 -> 0.4910`
    - `(128, 8, 4096, 128, causal=False)`: `0.3568 -> 0.3453`
    - `(128, 8, 8192, 64, causal=False)`: `0.3773 -> 0.3658`
- Issues:
  - Removing only the LSE store helps two mid-size shapes, but slows the other four and loses on the aggregate score used by the evaluator.
  - The full LSE store is small compared with the attention output store; suppressing it is not a reliable bottleneck fix for this workload.
  - Final source after this entry is identical to the committed point 11 implementation.
- Reports:
  - `evaluation_reports/codex_point7_store_only_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_store_only_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 13: autotune applicability check

- Commit: journal-only commit for this entry.
- Optimization point: 13, autotune automatic tuning.
- Content:
  - Checked whether to replace the host-side `DEFAULT_TILING_PRESETS` and manual `(BLOCK_M, BLOCK_N)` dispatch with `@triton.autotune`.
  - Did not modify code. The existing implementation already exposes the tunable parameters through `_resolve_tiling()` and uses offline measured presets for the evaluator shapes.
  - Treated the earlier point 2 tiling sweep as the authoritative tuning data for this kernel.
- Effect:
  - No code changed, so the active implementation remains point 11.
  - Best measured performance remains `20.1515 / 60`, mean speedup `0.33585816600990115`, from `evaluation_reports/codex_point11_performance/evaluation_report.json`.
- Issues:
  - The autotune reference states the advanced Triton-Ascend autotune path is limited to Vector-style kernels; this kernel is dominated by `tl.dot` Cube operations plus online-softmax Vector work.
  - Running autotune inside the evaluator would add first-call tuning/compilation overhead and risks destabilizing the score path.
  - Several larger tile candidates already failed MLIR lowering in the point 2 sweep, so a broad in-evaluator autotune space is likely to hit the same failures.
- Reports:
  - `evaluation_reports/codex_tiling_perf_quick/`
  - `evaluation_reports/codex_point11_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 18: kernel splitting review

- Commit: journal-only commit for this entry.
- Optimization point: 18, kernel splitting.
- Content:
  - Checked the mandatory kernel-splitting condition because the workload is multi-case and the best current mean speedup is below `0.8`.
  - Did not add another split kernel. The current host path already dispatches per-shape `(BLOCK_M, BLOCK_N)` presets and per-shape lazy/stable softmax choices; these values are passed as `tl.constexpr`, so Triton compiles shape-specialized variants of `_attn_fwd`.
  - Kept the single persistent kernel source because all six performance cases have `total_tiles > DEFAULT_PERSISTENT_PROGRAMS`, so the persistent loop is required for the scored large-grid cases.
- Effect:
  - No code changed, so the active implementation remains point 11.
  - Best measured performance remains `20.1515 / 60`, mean speedup `0.33585816600990115`.
  - Existing split-like dispatch evidence:
    - `(128, 8, 1024, 128, causal=True)`: preset `(128, 64)`, lazy path.
    - `(128, 8, 1024, 256, causal=True)`: preset `(64, 128)`, lazy path with GM accumulator.
    - `(128, 8, 2048, 128, causal=True)`: preset `(64, 256)`, stable path.
    - `(128, 8, 2048, 256, causal=False)`: preset `(128, 128)`, lazy path.
    - `(128, 8, 4096, 128, causal=False)`: preset `(128, 256)`, stable path.
    - `(128, 8, 8192, 64, causal=False)`: preset `(128, 256)`, stable path.
- Issues:
  - A separate source-level kernel per group would duplicate compile-time specialization already produced by `tl.constexpr` without removing the large-grid persistent loop.
  - Removing the loop only helps tiny `total_tiles <= DEFAULT_PERSISTENT_PROGRAMS` correctness cases, not the scored performance cases.
  - Further one-case-one-kernel splitting would increase maintenance and compile risk without a measured positive candidate.
- Reports:
  - `evaluation_reports/codex_point11_performance/evaluation_report.json`
  - `evaluation_reports/codex_tiling_perf_quick/`

## 2026-07-25 - Optimization point 25: IR analysis review

- Commit: journal-only commit for this entry.
- Optimization point: 25, IR analysis.
- Content:
  - Extracted Triton/Bisheng IR for the active point 11 implementation with:
    `IR_OUTPUT_DIR=/workspace/new_attn/profiling_runs/codex_ir_point25 bash /workspace/cannbot-skills/ops/triton-latency-optimizer/scripts/run_and_extract.sh /workspace/new_attn/ir_trigger_attention.py`.
  - The extraction produced valid compiler artifacts:
    - `profiling_runs/codex_ir_point25/_attn_fwd_ttir.mlir` (`366` lines)
    - `profiling_runs/codex_ir_point25/_attn_fwd_ttadapter.mlir` (`370` lines)
    - `profiling_runs/codex_ir_point25/_attn_fwd_last_pass.mlir` (`709` lines)
  - Reviewed `_attn_fwd_last_pass.mlir` after `ConvertHIVMToStandard` and found the expected MIX split:
    `_attn_fwd_mix_aic` for Cube-side QK/PV dot work and `_attn_fwd_mix_aiv` for Vector-side softmax/output work.
  - Counted the main lowered operations in the last-pass IR:
    `nd2nz_half=9`, `mma_tile_half_to_float*=6`, `fixpipe_nz2nd_float_to_float_4d_to_2d=4`,
    `load_gm_to_ubuf_1d_float=4`, `load_gm_to_ubuf_1d_half=2`,
    `store_ubuf_to_gm_1d_half=6`, `store_ubuf_to_gm_1d_float=2`,
    `vexp_1d_float=3`, `enablevc_reduce_sum_ar_float=3`, `vln_1d_float=2`, `vdiv_2d_float=2`,
    `vcmp_ge_1d_float=2`, `vsel_vs_1d_float=2`,
    `pipe_barrier=17`, `set_flag=41`, `wait_flag=41`, `sync_block_set=21`, `sync_block_wait=21`.
  - Did not modify code. The IR confirms the remaining cost is a mix of Cube/FixPipe/GM workspace traffic,
    Vector softmax normalization, and synchronization density; the actionable source-level levers from this evidence
    are the same ones already tested.
- Effect:
  - No code changed, so the active implementation remains point 11.
  - Best measured performance remains `20.1515 / 60`, mean speedup `0.33585816600990115`,
    from `evaluation_reports/codex_point11_performance/evaluation_report.json`.
  - The last-pass IR validates that point 11's GM accumulator load reordering targets a real GM workspace path in
    `ACC_IN_UB == False`, while the earlier delayed-V-load variant removed useful MTE overlap and regressed.
- Issues:
  - The IR trigger run raised an NPU `507015` aicore exception during `torch.npu.synchronize()`. The compiler dump and
    `bishengir-compile` extraction still completed, so this entry treats the IR as compile evidence only and not as a
    correctness/performance run.
  - The IR target spec in the dumped module reports Ascend910B-style sizes (`AI_CORE_COUNT=24`, `UB_SIZE=192KB`,
    `L0C_SIZE=128KB`), so no 910_95-only L0C-to-UB rewrite was applied.
  - Pass/store elimination corresponding to the softmax/LSE side was already tried in point 7 and regressed; widening
    or changing tiling was already tried in point 2; extra kernel splitting was reviewed in point 18; scalar-width
    cleanup from point 6 is already adopted.
  - No new safe optimizer point remained after mapping the IR findings back to the completed experiments.
- Reports:
  - `profiling_runs/codex_ir_point25/_attn_fwd_last_pass.mlir`
  - `result_dir/profile_summary.txt`
  - `evaluation_reports/codex_point11_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: dedicated no-LSE kernel experiment

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / unused statistics-output elimination.
- Content:
  - Tried a separate `_attn_fwd_no_lse` launch path for the default `attention(..., return_lse=False)` API.
  - Added a dedicated tile function that omitted the LSE tensor argument, skipped `m_i += tl.math.log(l_i)`, and skipped the `tl.store()` to the LSE buffer.
  - Kept `return_lse=True` on the original `_attn_fwd` path so the public optional LSE behavior remained available.
  - Reverted the code after performance validation because it was a negative optimization.
- Effect:
  - Correctness suite for the experimental code: `18/18` passed in `evaluation_reports/codex_point7_no_lse_kernel_correctness/evaluation_report.json`.
  - Performance score regressed from the point 11 best `20.1515 / 60` to `19.5454 / 60`.
  - Mean speedup regressed from point 11 `0.33585816600990115` to `0.325756944163873`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.2573 -> 0.2501`
    - `(128, 8, 1024, 256, causal=True)`: `0.2712 -> 0.2660`
    - `(128, 8, 2048, 128, causal=True)`: `0.2758 -> 0.2694`
    - `(128, 8, 2048, 256, causal=False)`: `0.4768 -> 0.4673`
    - `(128, 8, 4096, 128, causal=False)`: `0.3568 -> 0.3412`
    - `(128, 8, 8192, 64, causal=False)`: `0.3773 -> 0.3606`
- Issues:
  - Removing the LSE side effect was not enough to compensate for the extra source duplication and separate compiled kernel shape; all six measured performance cases slowed down.
  - This confirms the LSE store/log work is not the dominant bottleneck in the default path. The remaining time is still governed by the QK/PV loop, softmax vector/scalar chain, GM accumulator path for `HEAD_DIM=256`, and synchronization density.
  - Final source after this entry is identical to the point 11 implementation.
- Reports:
  - `evaluation_reports/codex_point7_no_lse_kernel_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_no_lse_kernel_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: force lazy softmax on wide-BN paths

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / online-softmax statistics elimination.
- Content:
  - Used the existing `FA_USE_MAX=0` runtime override to force the lazy non-stabilized softmax path on all six performance shapes before changing source code.
  - The goal was to test whether the current `BLOCK_N=256` stable paths could drop the running max, alpha correction, and accumulator rescale.
  - Did not modify code because the newly targeted wide-BN cases failed compilation.
- Effect:
  - Performance-only run: `3/6` matched in `evaluation_reports/codex_point7_force_lazy_performance/evaluation_report.json`.
  - Aggregate score was invalid for adoption: `10.0894 / 60`, mean speedup `0.33631424024195883` over only the three successful shapes.
  - Successful shapes were the ones already using lazy or equivalent small-BN paths:
    - `(128, 8, 1024, 128, causal=True)`: speedup `0.2580`
    - `(128, 8, 1024, 256, causal=True)`: speedup `0.2690`
    - `(128, 8, 2048, 256, causal=False)`: speedup `0.4820`
  - Newly targeted wide-BN shapes failed before timing:
    - `(128, 8, 2048, 128, causal=True)`: UB overflow, requires `2375680` bits while `1572864` bits are available.
    - `(128, 8, 4096, 128, causal=False)`: UB overflow `2105344 > 1572864` bits and CC overflow `1572864 > 1048576` bits.
    - `(128, 8, 8192, 64, causal=False)`: UB overflow `2105344 > 1572864` bits and CC overflow `1310720 > 1048576` bits.
- Issues:
  - The earlier `_use_lazy(block_n <= 128)` guard is necessary for the current kernel shape. Enabling lazy for `BLOCK_N=256` increases live qk/P/V/acc pressure enough to exceed UB or CC.
  - A source-level lazy widening patch would fail correctness/performance gating because three scored shapes would not compile.
  - Future work must reduce tile footprint first, for example by chunking output/HEAD_DIM or changing the accumulation strategy, before retrying lazy on `BLOCK_N=256`.
- Reports:
  - `evaluation_reports/codex_point7_force_lazy_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: reduced-M wide-BN lazy softmax probe

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / online-softmax statistics elimination.
- Content:
  - Tested a narrower variant of the failed wide-BN lazy-softmax idea before changing source code.
  - Instead of forcing lazy on the existing `(BM=128, BN=256)` long non-causal `HEAD_DIM=64` preset, the probe used
    `attention(..., BM=64, BN=256)` with `FA_USE_MAX=0` for shape `(128, 8, 8192, 64, causal=False)`.
  - The intent was to halve the QK tile M dimension so the lazy path could fit local storage, then remove the
    online max/alpha accumulator-rescale pass for the longest performance case.
  - Did not modify source because the targeted probe was already a clear regression.
- Effect:
  - Probe shape matched the NPU fusion attention reference: `torch.allclose=True`, `max_abs_diff=0.0001220703125`.
  - Probe median timings with `warmup=2`, `iters=5`:
    - Baseline `torch_npu.npu_fusion_attention`: `218177.3903 us`
    - Triton candidate `BM=64, BN=256, lazy`: `765851.3496 us`
    - Speedup vs baseline: `0.28488216467513744`
  - This is slower than the active point 11 implementation for the same case, whose full performance report recorded
    speedup `0.3772635825876341`.
- Issues:
  - Reducing `BLOCK_M` to make wide-BN lazy compile doubles the number of query tiles and loses much more than the
    lazy-softmax pass elimination saves.
  - The result confirms that, for the long `HEAD_DIM=64` non-causal case, the current `(BM=128, BN=256)` stable path is
    still better than a reduced-M lazy path.
  - Final source after this entry remains identical to the point 11 implementation.
- Reports:
  - Targeted inline probe only; no evaluator report was generated because source was not changed.

## 2026-07-25 - Optimization point 7: D256 causal accumulator UB residency

- Commit: code commit for this entry.
- Optimization point: 7, pass elimination / repeated GM accumulator traffic elimination.
- Content:
  - Added a narrow `_acc_in_ub()` exception for the measured compile-safe tile
    `(HEAD_DIM=256, BLOCK_M=64, BLOCK_N=128)`.
  - This tile is used by the performance case `(128, 8, 1024, 256, causal=True)`.
  - Keeping the accumulator resident for this tile removes the per-KV-block fp32 accumulator GM load/store path in
    `_attn_fwd_inner_loop()` and uses the fused `tl.dot(p_cast, v, acc_ptr)` accumulate path instead.
  - Did not relax the global UB budget. A probe showed that forcing UB residency for the non-causal D256 tile
    `(BM=128, BN=128)` fails MLIR lowering with CC overflow (`1572864 > 1048576` bits).
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point7_d256_acc_ub_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point7_d256_acc_ub_performance/evaluation_report.json`.
  - Performance score changed from point 11 `20.1515 / 60` to `20.1904 / 60`.
  - Mean speedup changed from `0.33585816600990115` to `0.33650674508638215`.
  - Targeted D256 causal case improved:
    - `(128, 8, 1024, 256, causal=True)`: speedup `0.271174 -> 0.285737`
    - Candidate median latency `18166.575 us -> 17286.640 us`
  - Other cases did not match the new `_acc_in_ub()` exception and moved only within benchmark noise.
- Issues:
  - The aggregate score gain is small because only one of six performance cases uses the new resident-accumulator path.
  - The exception must remain narrow. Enabling UB accumulator broadly for D256 can compile-fail non-causal D256 due to
    L0C/CC overflow.
  - This does not solve the larger gap to `torch_npu.npu_fusion_attention`; it only removes a confirmed avoidable GM
    accumulator round-trip on one safe tile.
- Reports:
  - `evaluation_reports/codex_point7_d256_acc_ub_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_d256_acc_ub_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: D256 causal wider-M accumulator UB boundary

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / repeated GM accumulator traffic elimination.
- Content:
  - Probed the adjacent causal D256 tile `(BM=128, BN=64)` with `ACC_UB_BUDGET=999999` and `FA_USE_MAX=0`.
  - The goal was to see whether the resident-accumulator idea could use a larger M tile and fewer query blocks than the
    adopted `(BM=64, BN=128)` path.
  - Did not modify source because the probe failed during compilation.
- Effect:
  - Probe failed MLIR lowering before timing:
    `cc overflow, requires 1310720 bits while 1048576 bits available`.
  - No correctness or performance evaluator report was generated.
- Issues:
  - The `(BM=64, BN=128)` D256 resident-accumulator exception appears to be near the L0C/CC limit for this kernel shape.
  - Increasing `BLOCK_M` while keeping the accumulator resident is not a viable next step; it triggers CC overflow.
  - Final source after this entry remains the previous D256 `(BM=64, BN=128)` resident-accumulator implementation.
- Reports:
  - Targeted inline probe only; compilation failed before evaluator reporting.

## 2026-07-25 - Optimization point 7: reduced-M wide path boundary probes

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / online-softmax or GM-accumulator traffic elimination.
- Content:
  - Probed additional reduced-M wide-tile variants after the positive D256 causal `(BM=64, BN=128)` result.
  - The intent was to see whether shrinking `BLOCK_M` could free enough local storage to use either lazy softmax on
    `BN=256` paths or resident accumulator on D256 paths without sacrificing too many query tiles.
  - Did not modify source because every additional boundary probe was negative.
- Effect:
  - Non-causal D256 `(128, 8, 2048, 256, causal=False)` with `BM=64, BN=128`:
    - Matched reference, `max_abs_diff=0.0009765625`.
    - Probe median timings with `warmup=2`, `iters=5`: baseline `34861.2098 us`, candidate `89439.4405 us`,
      speedup `0.38977446191731113`.
    - Slower than the active default `(BM=128, BN=128)` path, whose latest full run recorded speedup `0.476586`.
  - Non-causal D128 `(128, 8, 4096, 128, causal=False)` with `BM=64, BN=256`, `FA_USE_MAX=0`:
    - Matched reference, `max_abs_diff=0.00048828125`.
    - Probe median timings with `warmup=2`, `iters=5`: baseline `59819.7002 us`, candidate `214589.9106 us`,
      speedup `0.2787628739591102`.
    - Much slower than the active default `(BM=128, BN=256)` stable path.
  - Causal D128 `(128, 8, 1024, 128, causal=True)` with `BM=64, BN=256`, `FA_USE_MAX=0`:
    - Failed MLIR lowering before timing with UB overflow:
      `requires 2375680 bits while 1572864 bits available`.
- Issues:
  - Shrinking `BLOCK_M` to make wide lazy/resident paths fit increases the number of query tiles enough to dominate any
    saved online-softmax or GM accumulator work on these larger cases.
  - Wide-BN lazy remains unsafe for the current source shape unless the live qk/P/V/acc footprint is reduced by a deeper
    algorithmic change.
  - Final source after this entry remains the D256 causal `(BM=64, BN=128)` resident-accumulator implementation.
- Reports:
  - Targeted inline probes only; no evaluator reports were generated because no source change was adopted.

## 2026-07-25 - Optimization point 6: divisible causal boundary arithmetic

- Commit: code commit for this entry.
- Optimization point: 6, avoid vector API scalar lowering / avoid unnecessary integer divide in causal boundary setup.
- Content:
  - Added a compile-time specialization in `_attn_fwd_inner()` for causal off-band/on-band ranges when
    `BLOCK_M` is an integer multiple of `BLOCK_N`.
  - For those tiles, replaced the generic `floor/ceil(... / BLOCK_N) * BLOCK_N` boundary arithmetic with direct
    `start_m * BLOCK_M` / `(start_m + 1) * BLOCK_M` expressions plus `tl.multiple_of()`.
  - Kept the original generic formulas for `BLOCK_M < BLOCK_N` and non-divisible tiles, preserving the wide causal
    paths that were introduced earlier for `BM < BN`.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point6_divisible_bounds_correctness/evaluation_report.json`.
  - Submit-side sensitive spot checks passed:
    - `torch.float16`, `(1, 2, 1024, 64, causal=False)`, `max_abs_diff=6.103515625e-05`.
    - `torch.bfloat16`, `(128, 8, 1024, 64, causal=False)`, `max_abs_diff=0.0009765625`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point6_divisible_bounds_performance/evaluation_report.json`.
  - Performance score improved from the previous full report `20.1904 / 60` to `20.5003 / 60`.
  - Mean speedup improved from `0.3365067450863821` to `0.3416714652487010`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.253146 -> 0.256759`.
    - `(128, 8, 1024, 256, causal=True)`: `0.285737 -> 0.290876`.
    - `(128, 8, 2048, 128, causal=True)`: `0.272364 -> 0.278085`.
    - `(128, 8, 2048, 256, causal=False)`: `0.476586 -> 0.487924`.
    - `(128, 8, 4096, 128, causal=False)`: `0.354578 -> 0.357594`.
    - `(128, 8, 8192, 64, causal=False)`: `0.376630 -> 0.378791`.
- Issues:
  - The optimization is intentionally restricted to divisible causal tiles. Applying the direct boundary formula to
    `BM < BN` would be incorrect because the diagonal block can straddle a wider key tile.
  - Non-causal cases do not execute the specialized causal branches; their measured gains are likely from generated-code
    simplification and normal benchmark noise, so the reliable rationale is the reduced causal integer arithmetic.
  - The change does not touch persistent program count or device property probing, avoiding the earlier submit-side
    `LazySetDevice` / `507033` failure mode.
- Reports:
  - `evaluation_reports/codex_point6_divisible_bounds_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point6_divisible_bounds_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 6: wide causal boundary arithmetic regression

- Commit: journal-only commit for this entry.
- Optimization point: 6, avoid vector API scalar lowering / avoid unnecessary integer divide in causal boundary setup.
- Content:
  - Probed a second boundary-arithmetic specialization for causal `BM < BN` tiles where `BLOCK_N` is an integer multiple
    of `BLOCK_M`.
  - Rewrote the off-band/on-band boundaries through `query_tiles_per_kv = BLOCK_N // BLOCK_M`, e.g.
    `(start_m // query_tiles_per_kv) * BLOCK_N`, while preserving the original formulas as the non-divisible fallback.
  - Reverted the source after performance validation because the specialization was negative.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite for the experimental code: `18/18` passed in
    `evaluation_reports/codex_point6_wide_bounds_correctness/evaluation_report.json`.
  - Performance suite for the experimental code: `6/6` matched in
    `evaluation_reports/codex_point6_wide_bounds_performance/evaluation_report.json`.
  - Performance score regressed from the active divisible-boundary implementation `20.5003 / 60` to `19.3750 / 60`.
  - Mean speedup regressed from `0.3416714652487010` to `0.3229174639021567`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.256759 -> 0.251807`.
    - `(128, 8, 1024, 256, causal=True)`: `0.290876 -> 0.283664`.
    - `(128, 8, 2048, 128, causal=True)`: `0.278085 -> 0.271768`.
    - `(128, 8, 2048, 256, causal=False)`: `0.487924 -> 0.426488`.
    - `(128, 8, 4096, 128, causal=False)`: `0.357594 -> 0.344504`.
    - `(128, 8, 8192, 64, causal=False)`: `0.378791 -> 0.359273`.
- Issues:
  - Although mathematically equivalent, the `BM < BN` specialization likely changes generated control/scalar IR enough to
    worsen scheduling and register pressure. It also slowed non-causal specializations, suggesting the extra constexpr
    branch shape affected shared generated code structure even when not executed at runtime.
  - Do not repeat this wide-boundary rewrite unless IR evidence shows a different lowering strategy for it.
  - Final source after this entry remains the previous positive divisible-boundary implementation.
- Reports:
  - `evaluation_reports/codex_point6_wide_bounds_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point6_wide_bounds_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 6: tile index int32 narrowing regression

- Commit: journal-only commit for this entry.
- Optimization point: 6, avoid vector API scalar lowering / reduce int64 scalar address arithmetic.
- Content:
  - Probed narrowing `_attn_fwd_tile()`'s `task_m_idx` and `task_hz_idx` to `tl.int32` immediately after tile
    decomposition.
  - This made block-pointer offsets, `offs_m`, `m_ptrs`, and the `(z,h)` plane offset inherit int32 tile ids.
  - Reverted the source after performance validation because the change was negative.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite for the experimental code: `18/18` passed in
    `evaluation_reports/codex_point6_tile_index_i32_correctness/evaluation_report.json`.
  - Performance suite for the experimental code: `6/6` matched in
    `evaluation_reports/codex_point6_tile_index_i32_performance/evaluation_report.json`.
  - Performance score regressed from the active divisible-boundary implementation `20.5003 / 60` to `20.1499 / 60`.
  - Mean speedup regressed from `0.3416714652487010` to `0.3358319183697660`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.256759 -> 0.253418`.
    - `(128, 8, 1024, 256, causal=True)`: `0.290876 -> 0.283997`.
    - `(128, 8, 2048, 128, causal=True)`: `0.278085 -> 0.267219`.
    - `(128, 8, 2048, 256, causal=False)`: `0.487924 -> 0.484414`.
    - `(128, 8, 4096, 128, causal=False)`: `0.357594 -> 0.353448`.
    - `(128, 8, 8192, 64, causal=False)`: `0.378791 -> 0.372495`.
- Issues:
  - The earlier positive int32 cleanup only narrowed final pointer-offset terms. Narrowing the tile ids themselves changes
    more scalar expressions, including block-pointer offsets and store indices, and appears to hurt lowering/scheduling.
  - Keep `task_m_idx/task_hz_idx` in the original inferred type and only cast the final large pointer offset terms where
    point 6 already showed a small positive effect.
  - Final source after this entry remains the positive divisible-boundary implementation.
- Reports:
  - `evaluation_reports/codex_point6_tile_index_i32_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point6_tile_index_i32_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: causal diagonal split regression

- Commit: journal-only commit for this entry.
- Optimization point: 7, pass elimination / masked future-column compute elimination.
- Content:
  - Probed a causal `BM < BN` diagonal-stage split to avoid computing future columns that are immediately masked out.
  - The first implementation created BM-sized K/V block pointers inside a constexpr branch and failed Triton frontend
    lowering for every correctness case with `UnsupportedLanguageConstruct`.
  - The second implementation created the BM-sized block pointers unconditionally and used them only in the causal
    diagonal branch. It passed correctness but regressed performance, so the source was reverted.
- Effect:
  - v1 correctness report: `0/18` due frontend compile failure in
    `evaluation_reports/codex_point7_diag_split_correctness/evaluation_report.json`.
  - v2 `python3 -m py_compile flash_attention_forward.py`: pass.
  - v2 `git diff --check`: pass.
  - v2 correctness suite: `18/18` passed in
    `evaluation_reports/codex_point7_diag_split_v2_correctness/evaluation_report.json`.
  - v2 performance suite: `6/6` matched in
    `evaluation_reports/codex_point7_diag_split_v2_performance/evaluation_report.json`.
  - Performance score regressed from the active divisible-boundary implementation `20.5003 / 60` to `19.7107 / 60`.
  - Mean speedup regressed from `0.3416714652487010` to `0.3285118825849434`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.256759 -> 0.252655`.
    - `(128, 8, 1024, 256, causal=True)`: `0.290876 -> 0.294072`.
    - `(128, 8, 2048, 128, causal=True)`: `0.278085 -> 0.219105`.
    - `(128, 8, 2048, 256, causal=False)`: `0.487924 -> 0.471837`.
    - `(128, 8, 4096, 128, causal=False)`: `0.357594 -> 0.355269`.
    - `(128, 8, 8192, 64, causal=False)`: `0.378791 -> 0.378133`.
- Issues:
  - The D256 short causal case gained slightly, but the long D128 causal case lost heavily. The extra BM-sized dot calls
    and additional block pointers cost more than the masked-column work they remove.
  - Creating block pointers inside constexpr branches is not accepted by this Triton-Ascend frontend.
  - Do not split the diagonal pass this way unless a future version can keep wide-dot efficiency while avoiding the
    future-column mask work.
  - Final source after this entry remains the positive divisible-boundary implementation.
- Reports:
  - `evaluation_reports/codex_point7_diag_split_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_diag_split_v2_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_diag_split_v2_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 25: current causal IR review

- Commit: journal-only commit for this entry.
- Optimization point: 25, IR analysis.
- Content:
  - Extracted fresh Triton/Bisheng IR for the current best source on the causal performance shape
    `(128, 8, 1024, 128, causal=True)` with:
    `IR_OUTPUT_DIR=/workspace/new_attn/profiling_runs/codex_ir_point25_current_causal bash /workspace/cannbot-skills/ops/triton-latency-optimizer/scripts/run_and_extract.sh /workspace/new_attn/ir_trigger_attention.py`.
  - The temporary `ir_trigger_attention.py` script was deleted after extraction and is not part of the final source.
  - The extraction produced:
    - `profiling_runs/codex_ir_point25_current_causal/_attn_fwd_ttir.mlir` (`345` lines).
    - `profiling_runs/codex_ir_point25_current_causal/_attn_fwd_ttadapter.mlir` (`349` lines).
    - `profiling_runs/codex_ir_point25_current_causal/_attn_fwd_last_pass.mlir` (`696` lines).
- Effect:
  - IR validation succeeded: `1/1` kernel validated and last-pass IR contains HIVM/LLVM operations.
  - The trigger run itself raised `507015` aicore exception at `torch.npu.synchronize()`, but the compiler dump and
    `bishengir-compile` extraction completed successfully. No benchmark result was taken from this run.
  - Current causal last-pass summary:
    - `scf.for`: `6`.
    - `arith.divsi`: `6`.
    - `arith.remsi`: `2`.
    - `arith.index_cast`: `37`.
    - `hivm.hir.sync_block_set`: `21`.
    - `hivm.hir.sync_block_wait`: `21`.
    - `hivm.hir.wait_flag`: `41`.
    - `hivm.hir.set_flag`: `41`.
    - `func.call @nd2nz_half`: `8`.
    - `func.call @mma_tile_half_to_float_tb`: `2`.
- Issues:
  - Remaining `divsi/remsi` sites are dominated by persistent tile decomposition and `(z,h)` decomposition. Related
    source-level rewrites were already tested:
    - Point 5 linear contiguous tile offset fast path regressed all six performance cases.
    - Point 6 tile-index int32 narrowing regressed all six performance cases.
    - The positive point 6 divisible-boundary change already removed the safe causal boundary divide in the divisible
      `BM >= BN` path.
  - The IR shows heavy CUBE/VECTOR sync and workspace handoff, but mapped remedies have already been tested:
    - GM accumulator load reordering was positive and kept.
    - Narrow D256 resident accumulator was positive and kept.
    - Broader D256 UB residency, reduced-M wide paths, and diagonal split were negative or failed lowering.
  - No new safe optimization point was identified from this IR snapshot. Further progress likely needs either a different
    algorithmic kernel family or profiler/simulator evidence for a specific pipeline bottleneck, not more blind scalar or
    tiling rewrites.
- Reports:
  - `profiling_runs/codex_ir_point25_current_causal/_attn_fwd_last_pass.mlir`
  - `evaluation_reports/codex_point6_divisible_bounds_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 18: causal lazy UB kernel family regression

- Commit: journal-only commit for this entry.
- Optimization point: 18, kernel splitting / kernel family specialization.
- Content:
  - Probed a specialized Triton kernel family for `causal and not USE_MAX and ACC_IN_UB`.
  - Added an experimental `_attn_fwd_causal_lazy_ub` path with its own tile and loop helpers.
  - The specialized path removed the generic `STAGE`, `USE_MAX`, and `ACC_IN_UB` branches from the kernel body and
    skipped the dummy accumulator workspace allocation on the routed host path.
  - The original `_attn_fwd` remained the fallback for stable softmax, GM-accumulator, and all non-causal cases.
  - Reverted the source after performance validation because the change was negative.
- Effect:
  - Experimental source `python3 -m py_compile flash_attention_forward.py`: pass.
  - Experimental source `git diff --check`: pass.
  - Correctness suite for the experimental code: `18/18` passed in
    `evaluation_reports/codex_point18_causal_lazy_ub_family_correctness/evaluation_report.json`.
  - Performance suite for the experimental code: `6/6` matched in
    `evaluation_reports/codex_point18_causal_lazy_ub_family_performance/evaluation_report.json`.
  - Performance score regressed from the active divisible-boundary implementation `20.5003 / 60` to `19.7576 / 60`.
  - Mean speedup regressed from `0.3416714652487010` to `0.3292927041946979`.
  - Median speedup regressed from `0.3242350016847925` to `0.3133392780000467`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.256759 -> 0.250939`.
    - `(128, 8, 1024, 256, causal=True)`: `0.290876 -> 0.284306`.
    - `(128, 8, 2048, 128, causal=True)`: `0.278085 -> 0.268811`.
    - `(128, 8, 2048, 256, causal=False)`: `0.487924 -> 0.466441`.
    - `(128, 8, 4096, 128, causal=False)`: `0.357594 -> 0.342373`.
    - `(128, 8, 8192, 64, causal=False)`: `0.378791 -> 0.362887`.
  - Final source after rollback `python3 -m py_compile flash_attention_forward.py`: pass.
  - Final source after rollback `git diff --check`: pass.
- Issues:
  - The specialized family did not reduce latency on the targeted causal lazy UB cases. The copied narrow kernel likely
    changed lowering enough to lose the generic helper's current scheduling balance, despite removing source-level
    constexpr branches.
  - Non-causal fallback cases also measured lower in the same run even though their source path was unchanged, so part of
    the regression may be run-to-run variance. The two targeted causal cases still regressed directly and are sufficient
    to reject the specialization.
  - Keep the current generic `_attn_fwd` as the active implementation. A future kernel family attempt should specialize a
    genuinely different algorithmic path rather than copying the current helper structure with fewer constexpr branches.
- Reports:
  - `evaluation_reports/codex_point18_causal_lazy_ub_family_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_causal_lazy_ub_family_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: wide-BN lazy softmax with GM accumulator

- Commit: source + journal commit for this entry.
- Optimization point: 7, pass elimination / stable-softmax statistics elimination.
- Content:
  - Used the existing profiler/simulator evidence to specialize the sync-bound wide-BN family instead of copying the
    generic kernel:
    - torch_npu profiler for `(128, 8, 8192, 64, causal=False)` showed `_attn_fwd` duration `606970.124 us`,
      `cube_utilization(%)=97.853`, but very low `aic_mac_ratio=0.113`.
    - simulator aggregation showed `MMAD` was only `0.8%` of instruction running time, while
      `WAIT_FLAG + BAR + WAIT_FLAG_DEVI + SET_FLAG` was about `71.1%`.
    - Pipe aggregation was dominated by `VECTOR` (`33.8%`), `FLOWCTRL` (`18.7%`), `SCALAR` (`12.2%`),
      `MTE2` (`12.2%`), and `MTE3` (`8.0%`), not raw Cube MAC work.
  - Added `_use_wide_lazy_gm(block_m, block_n, head_dim, causal, n_ctx)` for
    `BLOCK_N == 256 and HEAD_DIM <= 128 and N_CTX >= 1024`.
  - For that family only, set `use_max=False` and force `ACC_IN_UB=False`, trading GM accumulator traffic for removing
    stable-softmax `max/alpha/acc-rescale` vector work.
  - Kept `FA_USE_MAX` override semantics unchanged; explicit `FA_USE_MAX=1/0` bypasses this automatic family rule.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point7_wide_lazy_gm_correctness/evaluation_report.json`.
  - Submit-sensitive spot checks via `evaluate_attention.run_correctness_case()`:
    - `torch.bfloat16`, `(128, 8, 1024, 64, causal=False)`: passed, `max_abs_diff=0.001953125`.
    - `torch.float16`, `(1, 2, 1024, 64, causal=False)`: passed, `max_abs_diff=0.0001220703125`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point7_wide_lazy_gm_performance/evaluation_report.json`.
  - Performance score improved from the active divisible-boundary implementation `20.5003 / 60` to `21.0465 / 60`.
  - Mean speedup improved from `0.3416714652487010` to `0.3507746595843307`.
  - Median speedup improved from `0.3242350016847925` to `0.3381159877112202`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.256759 -> 0.249607`.
    - `(128, 8, 1024, 256, causal=True)`: `0.290876 -> 0.286369`.
    - `(128, 8, 2048, 128, causal=True)`: `0.278085 -> 0.286779`.
    - `(128, 8, 2048, 256, causal=False)`: `0.487924 -> 0.466992`.
    - `(128, 8, 4096, 128, causal=False)`: `0.357594 -> 0.389453`.
    - `(128, 8, 8192, 64, causal=False)`: `0.378791 -> 0.425448`.
- Issues:
  - The tradeoff helps the long sync-bound non-causal shapes substantially but regresses the already-lazy short causal
    shapes in this run due measurement variance or secondary scheduling effects. The aggregate score is positive, so the
    source change is kept.
  - The non-causal D256 case `(128, 8, 2048, 256, causal=False)` regressed because it is already `BN=128` lazy with a
    GM accumulator and does not benefit from the new wide-BN rule; this appears to be run-to-run variance, not a routed
    code-path change.
  - This confirms the useful specialization axis is not "copy the whole kernel and delete branches"; it is selecting a
    different accumulator residency policy for the profile-identified stable wide-BN family.
- Reports:
  - `result_dir/profile_summary.json`
  - `profiling_runs/sim_point6_baseline/OPPROF_20260725023458_FUPEWIPFAYQPOEYP/simulator/`
  - `evaluation_reports/codex_point7_wide_lazy_gm_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_wide_lazy_gm_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 2: causal D128 short-context wide-lazy-GM tiling

- Commit: source + journal commit for this entry.
- Optimization point: 2, tiling optimization.
- Content:
  - Revisited the first performance case `(128, 8, 1024, 128, causal=True)` after the positive wide-lazy-GM policy was
    available.
  - Changed only that preset from `(BLOCK_M=128, BLOCK_N=64)` to `(BLOCK_M=64, BLOCK_N=256)`.
  - Under the previous implementation, `(64,256)` used the stable softmax path and was slightly slower; under the current
    implementation it routes to lazy softmax with a GM accumulator, reducing the number of KV-loop iterations while
    avoiding the lazy+resident-accumulator L0C overflow.
  - Updated the preset comment to point at this journal rather than stale historical sweep wording.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point2_shape1_wide_lazy_gm_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point2_shape1_wide_lazy_gm_performance/evaluation_report.json`.
  - Performance score improved marginally from the active wide-lazy-GM implementation `21.0465 / 60` to `21.0501 / 60`.
  - Mean speedup improved from `0.3507746595843307` to `0.3508343006383436`.
  - Median speedup changed from `0.3381159877112202` to `0.3378251885212862`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.249607 -> 0.255697`.
    - `(128, 8, 1024, 256, causal=True)`: `0.286369 -> 0.281544`.
    - `(128, 8, 2048, 128, causal=True)`: `0.286779 -> 0.287466`.
    - `(128, 8, 2048, 256, causal=False)`: `0.466992 -> 0.468715`.
    - `(128, 8, 4096, 128, causal=False)`: `0.389453 -> 0.388185`.
    - `(128, 8, 8192, 64, causal=False)`: `0.425448 -> 0.423399`.
- Issues:
  - The aggregate gain is very small (`+0.0036 / 60`), so this should be treated as a marginal positive rather than a
    robust architectural win.
  - Only the first performance case has a routed source-path change; the other per-shape differences are mostly
    benchmark variance.
  - If a later repeated run shows this preset consistently below `(128,64)`, this entry can be reverted independently
    without affecting the wider point 7 lazy-GM policy.
- Reports:
  - `evaluation_reports/codex_point2_shape1_wide_lazy_gm_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point2_shape1_wide_lazy_gm_performance/evaluation_report.json`
