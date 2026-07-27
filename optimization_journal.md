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

## 2026-07-25 - Optimization point 18: D256 causal diagonal-split kernel family

- Commit: source + journal commit for this entry.
- Optimization point: 18, kernel splitting / family specialization.
- Content:
  - Added a dedicated `_attn_fwd_causal_diag_split` family for the D256 causal lazy-UB route:
    `causal and HEAD_DIM == 256 and BLOCK_M == 64 and BLOCK_N == 128 and not USE_MAX and ACC_IN_UB`.
  - The fallback `_attn_fwd` remains unchanged for all other cases.
  - The specialized tile keeps the existing persistent 1D grid and m-major causal tile mapping, but splits the
    causal on-band work into:
    - wide fully-valid off-band `BLOCK_N=128` chunks;
    - an optional `BLOCK_M=64` prefix chunk inside the current wide key block;
    - a `BLOCK_M=64` masked diagonal chunk.
  - This isolates the earlier diagonal-split idea to the one shape family where prior full-kernel probing showed local
    upside, avoiding the known long-D128 causal regression.
  - The specialized route skips the dummy accumulator workspace allocation because the D256 route keeps the accumulator
    resident in UB.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - D256 causal spot check through `evaluate_attention.run_correctness_case()`:
    `(128, 8, 1024, 256, causal=True, fp16)` passed with `max_abs_diff=0.0029296875`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d256_diag_split_family_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d256_diag_split_family_performance/evaluation_report.json`.
  - Performance score improved from the active causal-D128 wide-lazy-GM preset implementation `21.0501 / 60` to
    `21.7645 / 60`.
  - Mean speedup improved from `0.3508343006383436` to `0.3627422138209748`.
  - Median speedup improved from `0.3378251885212862` to `0.3505926638979259`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.255697 -> 0.263565`.
    - `(128, 8, 1024, 256, causal=True)`: `0.281544 -> 0.291005`.
    - `(128, 8, 2048, 128, causal=True)`: `0.287466 -> 0.291400`.
    - `(128, 8, 2048, 256, causal=False)`: `0.468715 -> 0.470082`.
    - `(128, 8, 4096, 128, causal=False)`: `0.388185 -> 0.409785`.
    - `(128, 8, 8192, 64, causal=False)`: `0.423399 -> 0.450617`.
- Issues:
  - Only the D256 causal case has a routed source-path change; the improvements in the other five cases are likely
    benchmark variance from a favorable run and should not be attributed to the new kernel family.
  - The code addition is large because the Triton-Ascend frontend previously rejected creating extra block pointers inside
    a constexpr branch. A separate family avoids perturbing fallback lowering at the cost of duplication.
  - Keep this family restricted to `BLOCK_N == 2 * BLOCK_M` and `HEAD_DIM == 256`; the broader diagonal split was already
    measured as negative for long D128 causal.
- Reports:
  - `evaluation_reports/codex_point18_d256_diag_split_family_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d256_diag_split_family_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 2: non-causal D256 resident-accumulator tiling probe

- Commit: journal-only commit for this entry.
- Optimization point: 2, tiling optimization.
- Content:
  - Probed the non-causal D256 performance case `(128, 8, 2048, 256, causal=False)` with override
    `(BM=64, BN=128)`.
  - The current default for this shape is `(BM=128, BN=128)`, lazy softmax with a GM accumulator. The probe reused the
    existing compile-safe D256 `(BM=64, BN=128)` resident-accumulator exception to test whether removing GM accumulator
    load/store traffic can beat the extra query tiles.
  - Did not modify source because the targeted same-session A/B was a clear regression.
- Effect:
  - Serial same-process A/B after avoiding concurrent NPU benchmark interference:
    - Default `(BM=128, BN=128)`: median `72.162 ms`, min `72.131 ms`, CV `0.0005`.
    - Probe `(BM=64, BN=128)`: median `87.835 ms`, min `87.786 ms`, CV `0.0008`.
  - The probe is approximately `21.7%` slower than the active default on this targeted shape.
- Issues:
  - For non-causal D256, halving `BLOCK_M` doubles the number of query tiles. That added tile scheduling and Q-load work
    dominates the saved GM accumulator traffic.
  - Keep the D256 resident-accumulator exception routed only where it has already been measured positive:
    the causal `(128, 8, 1024, 256, causal=True)` family.
- Reports:
  - Targeted same-process inline probe only; no evaluator report was generated because source was not changed.

## 2026-07-25 - Optimization point 7: direct first-block GM accumulator init regression

- Commit: journal-only commit for this entry; source reverted to `d009ba4`.
- Optimization point: 7, pass elimination / accumulator workspace initialization and first-load elimination.
- Content:
  - Tried an `INIT_ACC_DIRECT` constexpr family for non-causal lazy-GM accumulator paths with `head_dim <= 128`.
  - The intended replacement was to allocate the non-causal GM accumulator workspace with `torch.empty`, compute the
    first KV block directly as `pv = tl.dot(p, v)`, store it to the workspace, then continue the remaining KV loop with
    the existing GM load/add/store accumulation path.
  - v1 added a `tl.static_assert` guard for the new constexpr combination and failed during front-end compilation.
  - v2 removed the static assert but still failed correctness on every non-causal GM target family.
  - v3 separated the first-block prelude from the remaining loop and returned early after the remaining accumulation;
    it failed the same correctness cases as v2.
- Effect:
  - Static checks passed for the source shape after reverting the experiment.
  - v1 correctness suite: `0/18` passed in
    `evaluation_reports/codex_point7_direct_acc_init_correctness/evaluation_report.json`.
  - v1 root cause: Triton-Ascend front-end rejected the chained boolean expression used inside `tl.static_assert`.
  - v2 correctness suite: `12/18` passed in
    `evaluation_reports/codex_point7_direct_acc_init_v2_correctness/evaluation_report.json`.
  - v3 correctness suite: `12/18` passed in
    `evaluation_reports/codex_point7_direct_acc_init_v3_correctness/evaluation_report.json`.
  - Failed non-causal GM cases were identical in v2 and v3:
    - `(1, 2, 1024, 64, causal=False)`: max diff `0.1124267578125`.
    - `(4, 32, 1024, 64, causal=False)`: max diff `0.22607421875`.
    - `(4, 32, 1024, 128, causal=False)`: max diff `0.8095703125`.
    - `(4, 32, 2048, 128, causal=False)`: max diff `0.53076171875`.
    - `(4, 32, 4096, 64, causal=False)`: max diff `0.09942626953125`.
    - `(128, 8, 1024, 64, causal=False)`: max diff `0.2646484375`.
- Issues:
  - The direct first-block write is not equivalent under the current Triton-Ascend lowering, or the later GM
    load/add/store path needs ordering/synchronization that this source-level prelude does not express.
  - The failure remained after splitting the first block into a dedicated loop body, so this is not just a loop-bound
    update bug.
  - Do not retry this exact prelude without first inspecting generated IR/synchronization around `tl.store` followed by
    subsequent `tl.load` from the same GM workspace.
- Reports:
  - `evaluation_reports/codex_point7_direct_acc_init_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_direct_acc_init_v2_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_direct_acc_init_v3_correctness/evaluation_report.json`

## 2026-07-25 - Optimization point 7: lazy LSE log-only suppression regression

- Commit: journal-only commit for this entry; source reverted to `d009ba4`.
- Optimization point: 7, pass elimination / unused statistics computation elimination.
- Content:
  - Tried a narrower follow-up to the previous LSE elimination experiments: keep the LSE tensor allocation and `tl.store`
    unchanged, but skip the final `m_i += tl.math.log(l_i)` on lazy-softmax paths where `m_i` remains `-inf`.
  - In the generic tile, this was implemented as `if USE_MAX: m_i += tl.math.log(l_i)`.
  - In the D256 causal diagonal-split family, which is always lazy, the final `m_i += tl.math.log(l_i)` was removed.
  - No tiling, accumulator residency, persistent-grid count, or host dispatch rule changed.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point7_lazy_log_skip_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point7_lazy_log_skip_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `21.0011 / 60`.
  - Mean speedup regressed from `0.3637380100490856` to `0.3500176646030400`.
  - Median speedup regressed from `0.3478055340771779` to `0.3373996572008339`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.257490`.
    - `(128, 8, 1024, 256, causal=True)`: `0.279763`.
    - `(128, 8, 2048, 128, causal=True)`: `0.287520`.
    - `(128, 8, 2048, 256, causal=False)`: `0.462299`.
    - `(128, 8, 4096, 128, causal=False)`: `0.387279`.
    - `(128, 8, 8192, 64, causal=False)`: `0.425754`.
- Issues:
  - Even though the removed `log(l_i)` is semantically unused for the evaluator's default `return_lse=False` path and
    lazy LSE values are already non-meaningful, removing it changed generated scheduling enough to slow the scored set.
  - This confirms the previous no-LSE and store-only regressions were not just caused by LSE buffer allocation or store
    removal; the LSE-side source shape itself appears to help current lowering balance.
  - Do not retry lazy-only LSE/log suppression without IR evidence that the compiler preserves the favorable schedule.
- Reports:
  - `evaluation_reports/codex_point7_lazy_log_skip_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_lazy_log_skip_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: broad lazy p_cast denominator regression

- Commit: journal-only commit for this entry; source reverted to `d009ba4`.
- Optimization point: 7, pass elimination / reuse already materialized cast probability tile.
- Content:
  - Changed only the lazy-softmax denominator update from `l_i += tl.sum(p, axis=1)` to
    `l_i += tl.sum(p_cast, axis=1)`.
  - The hypothesis was that numerator and denominator could share the same half/bfloat probability tile already used by
    `tl.dot(p_cast, v)`, reducing fp32 Vector reduce pressure while staying within the evaluator tolerance.
  - Stable online-softmax logic, tiling presets, accumulator residency, persistent-grid count, and host dispatch were not
    changed.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point7_lazy_pcast_denominator_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point7_lazy_pcast_denominator_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `21.7180 / 60`.
  - Mean speedup regressed from `0.3637380100490856` to `0.3619659982602839`.
  - Median speedup changed from `0.3478055340771779` to `0.3488774565325008`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.259911`.
    - `(128, 8, 1024, 256, causal=True)`: `0.285556`.
    - `(128, 8, 2048, 128, causal=True)`: `0.292320`.
    - `(128, 8, 2048, 256, causal=False)`: `0.475543`.
    - `(128, 8, 4096, 128, causal=False)`: `0.405435`.
    - `(128, 8, 8192, 64, causal=False)`: `0.453031`.
- Issues:
  - The broad approximation is correctness-safe locally but not performance-safe: D256 non-causal and several causal
    paths lose enough to outweigh the small D64 long-shape gain.
  - The only useful signal is the D64 non-causal wide-lazy-GM case, which improved slightly in this run. Any reuse of
    this idea must be isolated behind a narrow kernel-family constexpr instead of changing the shared lazy path.
  - Do not enable `p_cast` denominator broadly.
- Reports:
  - `evaluation_reports/codex_point7_lazy_pcast_denominator_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_lazy_pcast_denominator_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 18: D64 p_cast denominator family regression

- Commit: journal-only commit for this entry; source reverted to `d009ba4`.
- Optimization point: 18, kernel splitting / family specialization.
- Content:
  - Followed up on the broad `p_cast` denominator probe, where only the long non-causal D64 case showed a small local
    improvement.
  - Added a `USE_PCAST_DENOM` `tl.constexpr` threaded through `_attn_fwd_inner_loop()` / `_attn_fwd_inner()` /
    `_attn_fwd_tile()` / `_attn_fwd()`.
  - Routed it to `True` only for the non-causal D64 wide-lazy-GM family:
    `not causal and HEAD_DIM == 64 and BLOCK_M == 128 and BLOCK_N == 256 and not USE_MAX and not ACC_IN_UB`.
  - All other generic and D256 causal diagonal-split paths passed `False`, preserving their math path at the source
    level.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d64_pcast_denom_family_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d64_pcast_denom_family_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `21.0487 / 60`.
  - Mean speedup regressed from `0.3637380100490856` to `0.3508123155726117`.
  - Median speedup regressed from `0.3478055340771779` to `0.3390297263122239`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.257651`.
    - `(128, 8, 1024, 256, causal=True)`: `0.277584`.
    - `(128, 8, 2048, 128, causal=True)`: `0.287600`.
    - `(128, 8, 2048, 256, causal=False)`: `0.468302`.
    - `(128, 8, 4096, 128, causal=False)`: `0.390460`.
    - `(128, 8, 8192, 64, causal=False)`: `0.423278`.
- Issues:
  - The extra constexpr and branch structure perturbed the generated generic kernel enough that fallback paths slowed even
    though their `USE_PCAST_DENOM` value was `False`.
  - The targeted D64 long non-causal case also lost in the isolated family run, so the local gain observed in the broad
    probe was not robust.
  - Do not add new denominator-selection constexpr plumbing unless it is backed by IR showing no fallback schedule
    perturbation.
- Reports:
  - `evaluation_reports/codex_point18_d64_pcast_denom_family_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d64_pcast_denom_family_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 7: non-causal GM workspace empty-init regression

- Commit: journal-only commit for this entry; source reverted to `d009ba4`.
- Optimization point: 7, pass elimination / accumulator workspace initialization pass elimination.
- Content:
  - Tested replacing host-side fp32 GM accumulator `torch.zeros((z, h, n_ctx, head_dim))` with `torch.empty(...)` for
    non-causal GM-accumulator paths.
  - Added an experimental `INIT_ACC_IN_KERNEL` constexpr and, in `ACC_IN_UB == False` branches, used
    `tl.where(start_n == 0, 0, loaded_acc)` so the first KV block initializes the accumulator inside the kernel.
  - First broad version targeted all non-causal GM paths and exposed a D256 compile failure.
  - Second narrowed version restricted the route to `head_dim <= 128` to avoid the D256 UB overflow and retested fully.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Broad version:
    - Correctness suite: `18/18` passed in
      `evaluation_reports/codex_point7_empty_acc_init_correctness/evaluation_report.json`.
    - Performance suite: `5/6` matched in
      `evaluation_reports/codex_point7_empty_acc_init_performance/evaluation_report.json`.
    - D256 non-causal `(128, 8, 2048, 256, causal=False)` failed during lowering with UB overflow:
      `requires 2101248 bits while 1572864 bits available`.
  - Narrowed `head_dim <= 128` version:
    - Correctness suite: `18/18` passed in
      `evaluation_reports/codex_point7_empty_acc_init_d128d64_correctness/evaluation_report.json`.
    - Performance suite: `6/6` matched in
      `evaluation_reports/codex_point7_empty_acc_init_d128d64_performance/evaluation_report.json`.
    - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `21.0665 / 60`.
    - Mean speedup regressed from `0.3637380100490856` to `0.3511085101470289`.
    - Median speedup regressed from `0.3478055340771779` to `0.3375120255269046`.
    - Per-shape speedups:
      - `(128, 8, 1024, 128, causal=True)`: `0.263030 -> 0.254400`.
      - `(128, 8, 1024, 256, causal=True)`: `0.288501 -> 0.275833`.
      - `(128, 8, 2048, 128, causal=True)`: `0.289713 -> 0.286660`.
      - `(128, 8, 2048, 256, causal=False)`: `0.485001 -> 0.470374`.
      - `(128, 8, 4096, 128, causal=False)`: `0.405898 -> 0.388364`.
      - `(128, 8, 8192, 64, causal=False)`: `0.450285 -> 0.431020`.
- Issues:
  - The broad version shows the extra zero-tile select can increase live UB enough to break D256 lowering.
  - The narrowed version is still slower on the actual target non-causal D128/D64 cases, so the saved host zero-fill pass
    does not compensate for additional kernel-side select/zero pressure and scheduling perturbation.
  - Do not retry this exact `torch.empty + tl.where(start_n == 0)` pattern. A future attempt would need a separate
    first-block loop body that avoids the GM load and avoids materializing a full zero tile.
- Reports:
  - `evaluation_reports/codex_point7_empty_acc_init_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_empty_acc_init_performance/evaluation_report.json`
  - `evaluation_reports/codex_point7_empty_acc_init_d128d64_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_empty_acc_init_d128d64_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 18: non-causal wide-lazy GM fp16 workspace probe

- Commit: journal-only commit for this entry; source reverted to `d009ba4`.
- Optimization point: 18, kernel splitting / family specialization.
- Content:
  - Tested a narrow non-causal wide-lazy-GM family change:
    `acc_dtype = q.dtype if wide_lazy_gm and (not causal) else torch.float32`.
  - The intended target was performance cases using `(BM=128, BN=256)` with GM accumulator workspace:
    `(128, 8, 4096, 128, causal=False)` and `(128, 8, 8192, 64, causal=False)`.
  - The hypothesis was that profile-reported MTE2/MTE3 plus scalar/control overhead might improve if the repeated
    inter-iteration accumulator workspace load/store volume was halved.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_noncausal_gm_fp16_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_noncausal_gm_fp16_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `21.0981 / 60`.
  - Mean speedup regressed from `0.3637380100490856` to `0.3516353979160472`.
  - Median speedup regressed from `0.3478055340771779` to `0.3407773343533633`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.263030 -> 0.256630`.
    - `(128, 8, 1024, 256, causal=True)`: `0.288501 -> 0.278257`.
    - `(128, 8, 2048, 128, causal=True)`: `0.289713 -> 0.286274`.
    - `(128, 8, 2048, 256, causal=False)`: `0.485001 -> 0.462930`.
    - `(128, 8, 4096, 128, causal=False)`: `0.405898 -> 0.395280`.
    - `(128, 8, 8192, 64, causal=False)`: `0.450285 -> 0.430440`.
- Issues:
  - Despite reducing nominal GM workspace bytes for the targeted non-causal wide-lazy family, the target cases slowed
    down. The likely cost is extra dtype conversion / lower-throughput load-store lowering around the accumulator
    workspace, so MTE volume alone is not the current limiting factor.
  - Non-targeted case movement is treated as benchmark noise or compile-cache perturbation, but the target cases were
    clearly negative too.
  - Do not retry low-precision GM accumulator storage unless a later profile proves MTE bandwidth dominates after other
    scalar/control reductions.
- Reports:
  - `evaluation_reports/codex_point18_noncausal_gm_fp16_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_noncausal_gm_fp16_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 9: exp2 softmax math replacement regression

- Commit: journal-only commit for this entry.
- Optimization point: 9, libdevice / math function selection.
- Content:
  - Replaced the softmax exponentials with the mathematically equivalent `exp2(x * log2(e))` form:
    - stable path: `exp(qk)` and `exp(m_i - m_ij)`;
    - lazy path: `exp(qk)` and `exp(qk - 6.0)` for `HEAD_DIM >= 256`.
  - The intent was to reduce Vector math latency in the softmax chain without changing tiling, routing, or accumulator
    residency.
  - Reverted the source after validation because the aggregate performance regressed.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite for the experimental code: `18/18` passed in
    `evaluation_reports/codex_point9_exp2_correctness/evaluation_report.json`.
  - Performance suite for the experimental code: `6/6` matched in
    `evaluation_reports/codex_point9_exp2_performance/evaluation_report.json`.
  - Performance score regressed from the active D256 diagonal-split implementation `21.7645 / 60` to `21.0302 / 60`.
  - Mean speedup regressed from `0.3627422138209748` to `0.3505034236571608`.
  - Median speedup regressed from `0.3505926638979259` to `0.3373585130661327`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.263565 -> 0.257921`.
    - `(128, 8, 1024, 256, causal=True)`: `0.291005 -> 0.279407`.
    - `(128, 8, 2048, 128, causal=True)`: `0.291400 -> 0.287369`.
    - `(128, 8, 2048, 256, causal=False)`: `0.470082 -> 0.472839`.
    - `(128, 8, 4096, 128, causal=False)`: `0.409785 -> 0.387348`.
    - `(128, 8, 8192, 64, causal=False)`: `0.450617 -> 0.418137`.
- Issues:
  - `exp2` helps the non-causal D256 GM-accumulator case slightly, but it slows the five other scored shapes and loses
    significantly on the long wide-lazy-GM family.
  - On this Triton-Ascend stack, `tl.math.exp` is already the better lowering for the dominant lazy softmax path.
  - Keep the current `tl.math.exp` calls unless future IR evidence shows a different libdevice lowering.
- Reports:
  - `evaluation_reports/codex_point9_exp2_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point9_exp2_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 17: non-mask loop index hoist regression

- Commit: journal-only commit for this entry.
- Optimization point: 17, redundant boundary/index operation elimination.
- Content:
  - Re-profiled the current best long non-causal D64 case `(128, 8, 8192, 64, causal=False)` after the wide-lazy-GM and
    D256 diagonal-split changes.
  - Profiler summary for `_attn_fwd` in `result_dir/profile_summary.json`:
    - average op time `502113.754 us`;
    - `aic_scalar_ratio=0.643`, `aiv_scalar_ratio=0.397`;
    - `aic_mac_ratio=0.135`, so the hot path is still scalar/control/vector-heavy rather than MMAD-bound.
  - Tried moving `curr_n = start_n + offs_n` inside the `if NEED_CAUSAL_MASK:` branch in `_attn_fwd_inner_loop()` so
    non-causal and off-band causal loops would not materialize an unused key-index vector.
  - Reverted the source after validation because the aggregate performance regressed.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite for the experimental code: `18/18` passed in
    `evaluation_reports/codex_point17_mask_index_hoist_correctness/evaluation_report.json`.
  - Performance suite for the experimental code: `6/6` matched in
    `evaluation_reports/codex_point17_mask_index_hoist_performance/evaluation_report.json`.
  - Performance score regressed from the active D256 diagonal-split implementation `21.7645 / 60` to `21.3568 / 60`.
  - Mean speedup regressed from `0.3627422138209748` to `0.3559461857670623`.
  - Median speedup regressed from `0.3505926638979259` to `0.3367744330513599`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.263565 -> 0.258823`.
    - `(128, 8, 1024, 256, causal=True)`: `0.291005 -> 0.287339`.
    - `(128, 8, 2048, 128, causal=True)`: `0.291400 -> 0.292313`.
    - `(128, 8, 2048, 256, causal=False)`: `0.470082 -> 0.492325`.
    - `(128, 8, 4096, 128, causal=False)`: `0.409785 -> 0.381236`.
    - `(128, 8, 8192, 64, causal=False)`: `0.450617 -> 0.423641`.
- Issues:
  - The source-level redundant index removal appears to perturb lowering/scheduling more than it removes useful work.
    The compiler may already eliminate or sink this expression in some constexpr variants.
  - The fact that non-causal D256 improves but the long wide-lazy-GM shapes regress means this cannot be adopted as a
    shared inner-loop change.
  - A future retry would need a separate D256 non-causal family, not a common-loop rewrite.
- Reports:
  - `result_dir/profile_summary.json`
  - `evaluation_reports/codex_point17_mask_index_hoist_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point17_mask_index_hoist_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 18: non-causal D256 mask-index-elision family

- Commit: source + journal commit for this entry.
- Optimization point: 18, kernel splitting / family specialization.
- Content:
  - Isolated the local win from the failed shared point 17 rewrite to the only measured positive family:
    non-causal D256 `(HEAD_DIM=256, BM=128, BN=128, lazy softmax, GM accumulator)`.
  - Added `ELIDE_UNUSED_MASK_INDEX` as a `tl.constexpr` parameter through `_attn_fwd_inner_loop()` / `_attn_fwd_inner()`
    / `_attn_fwd_tile()` / `_attn_fwd()`.
  - For all existing fallback families, this parameter is `False`, preserving the previous `curr_n = start_n + offs_n`
    lowering. For the targeted non-causal D256 family only, it is `True`, so non-mask loops skip materializing the unused
    `curr_n` vector.
  - The existing D256 causal diagonal-split family explicitly passes `False` to avoid perturbing the previous positive
    route.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Spot checks passed for:
    - `(128, 8, 2048, 256, causal=False)`, max_abs_diff `0.0009765625`;
    - `(128, 8, 1024, 256, causal=True)`, max_abs_diff `0.005859375`;
    - `(128, 8, 8192, 64, causal=False)`, max_abs_diff `0.0001220703125`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d256_noncausal_index_family_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d256_noncausal_index_family_performance/evaluation_report.json`.
  - Performance score improved from the active D256 causal diagonal-split implementation `21.7645 / 60` to
    `21.8296 / 60`.
  - Mean speedup improved from `0.3627422138209748` to `0.3638264953797291`.
  - Median speedup changed from `0.3505926638979259` to `0.3475630904125050`.
  - Per-shape speedups:
    - `(128, 8, 1024, 128, causal=True)`: `0.263565 -> 0.263960`.
    - `(128, 8, 1024, 256, causal=True)`: `0.291005 -> 0.289292`.
    - `(128, 8, 2048, 128, causal=True)`: `0.291400 -> 0.289679`.
    - `(128, 8, 2048, 256, causal=False)`: `0.470082 -> 0.485425`.
    - `(128, 8, 4096, 128, causal=False)`: `0.409785 -> 0.405447`.
    - `(128, 8, 8192, 64, causal=False)`: `0.450617 -> 0.449156`.
- Issues:
  - The aggregate gain is small (`+0.0651 / 60`) and comes almost entirely from the targeted non-causal D256 case.
  - The shared point 17 rewrite was negative; keep this as a narrow constexpr family and do not enable it for long
    wide-lazy-GM D64/D128 paths.
  - The non-targeted shape movements should be treated as benchmark variance because their `ELIDE_UNUSED_MASK_INDEX`
    value remains `False`.
- Reports:
  - `evaluation_reports/codex_point18_d256_noncausal_index_family_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d256_noncausal_index_family_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 2: causal D128 narrow-BN tiling recheck

- Commit: journal-only commit for this entry.
- Optimization point: 2, tiling optimization.
- Content:
  - Rechecked the old causal D128 `(BM=128, BN=64)` and `(BM=64, BN=128)` candidates against the current implementation,
    after the wide-lazy-GM policy and D256 diagonal family had been added.
  - Targeted the two D128 causal performance cases:
    `(128, 8, 1024, 128, causal=True)` and `(128, 8, 2048, 128, causal=True)`.
  - Did not modify source because the active `(BM=64, BN=256)` preset remained best or tied in serial same-process A/B.
- Effect:
  - `(128, 8, 1024, 128, causal=True)`:
    - Default `(BM=64, BN=256)`: median `11.344 ms`, min `11.326 ms`, CV `0.0011`.
    - Probe `(BM=128, BN=64)`: median `11.544 ms`, min `11.515 ms`, CV `0.0009`.
    - Probe `(BM=64, BN=128)`: median `12.753 ms`, min `12.721 ms`, CV `0.0034`.
    - Repeated explicit `(BM=64, BN=256)`: median `11.361 ms`, min `11.295 ms`, CV `0.0021`.
  - `(128, 8, 2048, 128, causal=True)`:
    - Default `(BM=64, BN=256)`: median `36.008 ms`, min `35.922 ms`, CV `0.0009`.
    - Probe `(BM=128, BN=64)`: median `39.856 ms`, min `39.826 ms`, CV `0.0035`.
    - Probe `(BM=64, BN=128)`: median `43.468 ms`, min `43.453 ms`, CV `0.0006`.
    - Repeated explicit `(BM=64, BN=256)`: median `35.906 ms`, min `35.874 ms`, CV `0.0009`.
- Issues:
  - The current D128 causal family is latency-bound by iteration count and synchronization density; reducing `BLOCK_N`
    increases the number of KV-loop iterations too much.
  - `(BM=128, BN=64)` keeps accumulator residency simpler but loses on the long causal case, so it is not robust enough
    to restore as a preset.
- Reports:
  - Targeted same-process inline probe only; no evaluator report was generated because source was not changed.

## 2026-07-25 - Optimization point 18: D64 p_cast denominator static-branch compile failure

- Commit: journal-only commit for this entry; source reverted to `d009ba4`.
- Optimization point: 18, kernel splitting / family specialization.
- Content:
  - Tried a lower-overhead version of the rejected D64 `p_cast` denominator family without adding a new kernel argument.
  - Added a local static condition inside the lazy branch:
    `if (not NEED_CAUSAL_MASK) and HEAD_DIM == 64 and BLOCK_N == 256 and (not ACC_IN_UB):`
    then used `tl.sum(p_cast, axis=1)`, otherwise preserved `tl.sum(p, axis=1)`.
  - Intended to avoid the fallback-schedule perturbation caused by threading a new `USE_PCAST_DENOM` constexpr through
    the generic call stack.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass before NPU run.
  - `git diff --check`: pass before NPU run.
  - Correctness suite: `2/18` passed in
    `evaluation_reports/codex_point18_d64_pcast_static_branch_correctness/evaluation_report.json`.
  - Performance was not run because correctness failed.
- Issues:
  - Triton-Ascend front-end rejected the compound boolean condition with
    `UnsupportedLanguageConstruct: chained boolean operators (A or B or C) are not supported`.
  - This matches the earlier direct-prelude `tl.static_assert` failure class: chained boolean expressions are unsafe in
    JIT code.
  - Do not retry by adding more compound constexpr conditions inline. Use a host-provided single boolean only if IR shows
    fallback schedules are preserved, which the prior `USE_PCAST_DENOM` attempt did not.
- Reports:
  - `evaluation_reports/codex_point18_d64_pcast_static_branch_correctness/evaluation_report.json`

## 2026-07-25 - Optimization point 11: D64 GM accumulator early-load regression

- Commit: journal-only commit for this entry; source reverted to the post-`d009ba4` best implementation.
- Optimization point: 11, load instruction reordering.
- Content:
  - Targeted the profiled long non-causal D64 wide-lazy-GM path where `aic_mte2_ratio`, `aic_scalar_ratio`, and
    `aiv_scalar_ratio` are high while cube utilization remains near 99%.
  - Moved the GM accumulator `tl.load(acc_ptr + block2d_acc)` earlier for `HEAD_DIM == 64` and `ACC_IN_UB == False`,
    placing it before softmax denominator/update work instead of immediately before `tl.dot(p_cast, v)`.
  - Kept D128/D256 source paths unchanged by using the existing `HEAD_DIM` constexpr as the family guard; no tiling,
    workspace allocation, persistent program count, or math formula changed.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point11_d64_acc_prefetch_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point11_d64_acc_prefetch_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `20.8800 / 60`.
  - Mean speedup regressed from `0.3637380100490856` to `0.3480005754314638`.
  - Median speedup regressed from `0.3478055340771779` to `0.3398363101590597`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.260925`.
    - `(128, 8, 1024, 256, causal=True)`: `0.283875`.
    - `(128, 8, 2048, 128, causal=True)`: `0.291915`.
    - `(128, 8, 2048, 256, causal=False)`: `0.432439`.
    - `(128, 8, 4096, 128, causal=False)`: `0.387757`.
    - `(128, 8, 8192, 64, causal=False)`: `0.431092`.
- Issues:
  - The targeted D64 long non-causal case itself regressed from baseline speedup `0.450285` to `0.431092`, so the
    earlier accumulator load did not hide MTE2 latency.
  - The likely cause is a worse schedule or larger live range across qk/softmax work; keeping K/V load and qk/softmax
    ahead of the GM accumulator load is better for this kernel.
  - Do not retry earlier accumulator prefetch for D64 unless IR shows the load can be issued without extending the
    accumulator live range across the score tile.
- Reports:
  - `evaluation_reports/codex_point11_d64_acc_prefetch_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point11_d64_acc_prefetch_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 18: D64 non-causal split-kernel regression

- Commit: journal-only commit for this entry; source reverted to the post-`d009ba4` best implementation.
- Optimization point: 18, kernel splitting / family specialization.
- Content:
  - Added a dedicated non-causal D64 wide-lazy-GM kernel family for
    `head_dim == 64`, `BM == 128`, `BN == 256`, `causal == False`, `USE_MAX == False`, `ACC_IN_UB == False`.
  - The specialized tile inlined the generic lazy-GM math and removed compile-time branches for causal masking,
    stable softmax, fp8 V, accumulator residency, and unused mask-index materialization.
  - Host dispatch routed only this D64 family to the new kernel; all D128/D256 and causal families stayed on the existing
    kernels.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d64_split_kernel_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d64_split_kernel_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `21.6273 / 60`.
  - Mean speedup regressed from `0.3637380100490856` to `0.3604550558484402`.
  - Median speedup regressed from `0.3478055340771779` to `0.3448207182527797`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.256238`.
    - `(128, 8, 1024, 256, causal=True)`: `0.283854`.
    - `(128, 8, 2048, 128, causal=True)`: `0.282198`.
    - `(128, 8, 2048, 256, causal=False)`: `0.487722`.
    - `(128, 8, 4096, 128, causal=False)`: `0.405788`.
    - `(128, 8, 8192, 64, causal=False)`: `0.446931`.
- Issues:
  - The targeted D64 long non-causal case regressed from baseline speedup `0.450285` to `0.446931`.
  - The generic kernel's constexpr-specialized path is already close to the manually split source for this family; the
    extra kernel body and route did not produce better scheduling.
  - Keep future point 18 work focused on cases where IR shows an actual generic-path residue, not just source-level
    branch removal.
- Reports:
  - `evaluation_reports/codex_point18_d64_split_kernel_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d64_split_kernel_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 17: D64 direct Q load boundary regression

- Commit: journal-only commit for this entry; source reverted to the post-`d009ba4` best implementation.
- Optimization point: 17, redundant boundary operation.
- IR evidence:
  - Current D64 non-causal IR:
    `profiling_runs/codex_ir_point25_d64_noncausal_current/_attn_fwd_last_pass.mlir`.
  - The current specialized IR is 461 lines, shorter than earlier generic/causal IR dumps, but it still contains a
    residual boundary clamp.
  - The suspicious residual is `_attn_fwd_last_pass.mlir:238`:
    `%20 = arith.maxsi %19, %c0_i32 : i32`.
  - The location maps to `flash_attention_forward.py:960`, where the Q block pointer uses
    `offsets=(task_m_idx * BLOCK_M, 0)`.
- Content:
  - Added a `DIRECT_Q_LOAD` constexpr for only the D64 non-causal wide-lazy-GM family:
    `head_dim == 64`, `BM == 128`, `BN == 256`, `causal == False`, `USE_MAX == False`, `ACC_IN_UB == False`.
  - Replaced `tl.load(q_block_ptr)` with direct tensor-pointer addressing for Q:
    `tl.load(Q + qvk_offset + offs_m[:, None] * stride_qm + q_cols[None, :] * stride_qk)`.
  - Left all D128/D256 and causal families on the block-pointer path.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point17_d64_direct_q_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point17_d64_direct_q_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the post-`d009ba4` baseline `21.8243 / 60` to `21.1252 / 60`.
  - Mean speedup regressed from `0.3637380100490856` to `0.35208703874440306`.
  - Median speedup regressed from `0.34780553407717785` to `0.3394333592298998`.
  - Target D64 long non-causal speedup regressed from `0.45028541645240344` to `0.42542172153069074`.
- Issues:
  - Direct pointer tensor loading removed or targeted the block-pointer boundary artifact, but it hurt the resulting
    schedule enough to regress the target family.
  - The block-pointer path likely enables better ND/NZ lowering and MTE planning than the direct tensor-pointer form.
  - Do not retry direct Q tensor loading for this family without new IR evidence that the generated memory pipeline
    improves.
- Reports:
  - `evaluation_reports/codex_point17_d64_direct_q_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point17_d64_direct_q_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 17: D128 long causal off-band index elision

- Commit: source + journal commit for this entry.
- Optimization point: 17, redundant boundary/index operation elimination.
- Evidence:
  - Existing causal D128 pipe profiles show high scalar/MTE pressure rather than a pure MMAD limit:
    - `profiling_runs/codex_pipe_causal_d128/profile_summary.json`:
      `aic_scalar_ratio=0.5989`, `aic_mte2_ratio=0.3857`, `aiv_scalar_ratio=0.4198`.
    - `profiling_runs/codex_pipe_causal_d256/profile_summary.json`:
      `aic_scalar_ratio=0.5828`, `aic_mte2_ratio=0.4130`, `aiv_scalar_ratio=0.3769`.
  - The earlier broad point17 hoist moved `curr_n = start_n + offs_n` under `if NEED_CAUSAL_MASK` for all non-mask
    loops. It regressed the aggregate, but its per-shape data showed the long D128 causal case was slightly positive:
    `(128, 8, 2048, 128, causal=True)` speedup `0.291400 -> 0.292313`.
- Content:
  - Reused the existing `ELIDE_UNUSED_MASK_INDEX` constexpr instead of adding a new kernel or rewriting the loop body.
  - Added exactly one narrow family guard:
    `causal and head_dim == 128 and n_ctx == 2048 and bm == 64 and bn == 256 and not USE_MAX and not ACC_IN_UB`.
  - This skips materializing `curr_n` only in off-band D128 long causal loops where `NEED_CAUSAL_MASK == False`; the
    diagonal masked loop still computes `curr_n` normally.
  - Existing non-causal D256 mask-index elision remains unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: no new Triton kernel arithmetic, grid change, int64 path, loop control, or task decomposition was
    introduced; the change is host-side family dispatch only.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point17_d128_causal_index_elide_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point17_d128_causal_index_elide_performance/evaluation_report.json`.
  - Submit-style performance score improved from the post-`d009ba4` baseline `21.8243 / 60` to `22.1199 / 60`.
  - Mean speedup improved from `0.3637380100490856` to `0.36866472534624817`.
  - Median speedup improved from `0.34780553407717785` to `0.35471224516812827`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.270156`.
    - `(128, 8, 1024, 256, causal=True)`: `0.293738`.
    - `(128, 8, 2048, 128, causal=True)`: `0.299902`.
    - `(128, 8, 2048, 256, causal=False)`: `0.493121`.
    - `(128, 8, 4096, 128, causal=False)`: `0.409523`.
    - `(128, 8, 8192, 64, causal=False)`: `0.445548`.
- Issues:
  - Only the long D128 causal case is intentionally routed to the new behavior. Improvements on unrelated shapes are
    likely run-to-run variance and should not be attributed to this guard.
  - The long D64 non-causal case measured slightly below the post-`d009ba4` baseline in this run, but that source path is
    unchanged.
  - Keep this guard narrow. The broad point17 rewrite already showed that sharing the index elision across all non-mask
    loops hurts the D64/D128 wide-lazy-GM families.
- Reports:
  - `evaluation_reports/codex_point17_d128_causal_index_elide_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point17_d128_causal_index_elide_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 17: D128 non-causal direct Q load regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 17, redundant boundary/index operation elimination.
- Evidence:
  - Fresh IR was extracted for the D128 non-causal long family using
    `IR_OUTPUT_DIR=/workspace/new_attn/profiling_runs/codex_ir_point25_d128_noncausal_current`.
  - The trigger shape was `(Z=128, H=8, N_CTX=4096, HEAD_DIM=128, causal=False)`.
  - `_attn_fwd_last_pass.mlir` still contained a residual Q block-pointer boundary/index artifact:
    `profiling_runs/codex_ir_point25_d128_noncausal_current/_attn_fwd_last_pass.mlir:236`,
    `%20 = arith.maxsi %19, %c0_i32 : i32`.
  - The location maps to `flash_attention_forward.py:960`, where the Q block pointer uses
    `offsets=(task_m_idx * BLOCK_M, 0)`.
- Content:
  - Added a temporary `DIRECT_Q_LOAD` constexpr routed only to the D128 non-causal wide-lazy-GM family:
    `head_dim == 128`, `n_ctx == 4096`, `BM == 128`, `BN == 256`, `causal == False`, `USE_MAX == False`,
    `ACC_IN_UB == False`.
  - Replaced `tl.load(q_block_ptr)` with direct tensor-pointer addressing for Q:
    `tl.load(Q + qvk_offset + offs_m[:, None] * stride_qm + q_cols[None, :] * stride_qk)`.
  - Left causal, D64, D256, and other D128 paths on the existing block-pointer load.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point17_d128_direct_q_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point17_d128_direct_q_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.11988352077489 / 60` to
    `21.049980223715053 / 60`.
  - Mean speedup regressed from `0.36866472534624817` to `0.35083300372858417`.
  - Median speedup regressed from `0.35471224516812827` to `0.3381092144208627`.
  - Target D128 non-causal speedup regressed from `0.4095227679` to `0.3883291104`.
- Issues:
  - The IR trigger raised NPU runtime error `507015` at `torch.npu.synchronize()`, so that run is treated only as
    compile/IR evidence. Correctness and performance numbers above come from `evaluate_attention.py`.
  - Direct tensor-pointer Q loading likely hurts block-pointer lowering, ND/NZ conversion, or MTE planning enough to
    offset the removed boundary operation.
  - This mirrors the earlier D64 direct-Q regression; do not retry direct Q tensor loading for wide-lazy-GM families
    without evidence that the memory pipeline improves.
- Reports:
  - `profiling_runs/codex_ir_point25_d128_noncausal_current/_attn_fwd_last_pass.mlir`
  - `evaluation_reports/codex_point17_d128_direct_q_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point17_d128_direct_q_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 18: D64 non-causal pair-M K/V reuse family regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 18, kernel splitting / family specialization.
- Motivation:
  - Prior profiling and IR evidence for the D64/D128 wide-lazy-GM families showed non-MMAD overhead:
    `sync_block_set/wait`, `wait_flag/set_flag`, `pipe_barrier`, MTE/fixpipe and Vector/FlowCtrl pressure remain high.
  - The target performance case `(Z=128, H=8, N_CTX=8192, HEAD_DIM=64, causal=False)` uses `(BM=128, BN=256)`
    lazy softmax with GM accumulator. Each adjacent M tile streams the same K/V blocks independently in the generic path.
  - Hypothesis: a D64-only family that processes two adjacent M tiles per task can load each K/V tile once and reuse it for
    two Q tiles, reducing repeated GM K/V loads and some per-task synchronization overhead.
- Content:
  - Temporarily added `_attn_fwd_pair_m_tile` and `_attn_fwd_pair_m`.
  - The specialized kernel was guarded to `HEAD_DIM == 64`, `BLOCK_M == 128`, `BLOCK_N == 256`, and
    `N_CTX % (2 * BLOCK_M) == 0`.
  - Host routing was intentionally narrow:
    `not causal`, `head_dim == 64`, `n_ctx == 8192`, `BM == 128`, `BN == 256`, `USE_MAX == False`,
    `ACC_IN_UB == False`.
  - The pair-M tile loaded two Q blocks, loaded each K/V block once per `start_n`, computed two `QK/PV` pairs, and used
    the same fp32 GM accumulator layout as the existing wide-lazy-GM path.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d64_pair_m_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d64_pair_m_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.11988352077489 / 60` to
    `21.0847427384484 / 60`.
  - Mean speedup regressed from `0.36866472534624817` to `0.35141237897414`.
  - Median speedup regressed from `0.35471224516812827` to `0.3338406131340909`.
  - Target D64 non-causal speedup regressed from `0.4455484950181382` to `0.43923267497851914`.
  - Target D64 non-causal candidate median latency regressed from `505407.9801775515 us` to `513268.8395678997 us`.
- Issues:
  - K/V reuse did not compensate for the larger per-program live range: two Q tiles, two score tiles, two probability
    tiles, two L accumulators, and two GM accumulator load/store streams likely increased register/UB/L0C pressure.
  - Pairing M tiles also halves task granularity along M before persistent scheduling, which can hurt overlap and
    scheduling even though the launched persistent program count remains 20.
  - The implementation doubled dot work inside one loop body; this may worsen cube/vector synchronization and block
    pointer scheduling instead of reducing the measured FlowCtrl/MTE pressure.
  - Do not retry pair-M K/V reuse for the D64 wide-lazy-GM family without fresh simulator evidence showing K/V MTE is
    dominant and that the larger fused task does not increase wait/barrier time.
- Reports:
  - `evaluation_reports/codex_point18_d64_pair_m_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d64_pair_m_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 1: unused K/V z-h stride argument removal regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 1, constexpr/static parameter simplification.
- Motivation:
  - `stride_kz`, `stride_kh`, `stride_vz`, and `stride_vh` were threaded through `_attn_fwd`,
    `_attn_fwd_tile`, `_attn_fwd_causal_diag_split`, and `_attn_fwd_causal_diag_split_tile`, but the kernels compute
    the shared B/H base offset from `stride_qz` and `stride_qh`.
  - The current evaluator creates same-shape contiguous Q/K/V tensors, so removing these unused constexpr arguments
    should preserve behavior while shrinking the kernel signature and scalar parameter plumbing.
  - Hypothesis: fewer constexpr/scalar parameters might slightly reduce compile/runtime scalar setup noise and improve
    the already scalar/sync-heavy profiles.
- Content:
  - Removed the four unused K/V z/h stride constexpr parameters from both generic and causal diagonal-split kernel
    families.
  - Removed the corresponding host-side positional arguments:
    `k.stride(0)`, `k.stride(1)`, `v.stride(0)`, `v.stride(1)`.
  - No tiling, math, workspace allocation, persistent program count, or routing behavior was changed.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - `rg -n "stride_kz|stride_kh|stride_vz|stride_vh" flash_attention_forward.py`: no matches during the experiment.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point1_unused_kv_zh_stride_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point1_unused_kv_zh_stride_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.11988352077489 / 60` to
    `21.681339367536967 / 60`.
  - Mean speedup regressed from `0.36866472534624817` to `0.36135565612561615`.
  - Median speedup regressed from `0.35471224516812827` to `0.3473453111593829`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.259803`.
    - `(128, 8, 1024, 256, causal=True)`: `0.285003`.
    - `(128, 8, 2048, 128, causal=True)`: `0.289274`.
    - `(128, 8, 2048, 256, causal=False)`: `0.482212`.
    - `(128, 8, 4096, 128, causal=False)`: `0.405417`.
    - `(128, 8, 8192, 64, causal=False)`: `0.446425`.
- Issues:
  - The D64 long non-causal case improved slightly in this run, but the three causal performance shapes regressed enough
    to make the total score worse.
  - Removing unused constexpr parameters changed the generated kernel signature/specialization and likely perturbed
    compiler scheduling for causal paths; smaller source signatures are not automatically faster on Triton-Ascend.
  - Keep the K/V z/h stride arguments unless future IR proves that a dedicated family can remove them without changing
    the causal schedules.
- Reports:
  - `evaluation_reports/codex_point1_unused_kv_zh_stride_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point1_unused_kv_zh_stride_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 1: unused qk_scale helper argument removal regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 1, constexpr/static parameter simplification.
- Motivation:
  - The score scaling is already applied once via `q = (q * sm_scale).to(q.dtype)` before `tl.dot(q, tl.trans(k))`.
  - `_attn_fwd_inner_loop()` and `_attn_fwd_inner()` still carried a `qk_scale` constexpr argument, but the only use was
    the old commented line `# qk = qk * qk_scale`.
  - Hypothesis: removing this unused helper argument would reduce inlined helper signature noise without changing the
    top-level kernel launch ABI or math.
- Content:
  - Temporarily removed `qk_scale` from `_attn_fwd_inner_loop()` and `_attn_fwd_inner()`.
  - Removed the corresponding `sm_scale` / `qk_scale` arguments from generic and causal diagonal-split helper calls.
  - Removed the stale commented multiplication line.
  - No top-level Triton kernel signature, host dispatch, tiling preset, workspace allocation, persistent program count, or
    math formula was intentionally changed.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - `rg -n "qk_scale" flash_attention_forward.py`: no matches during the experiment.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point1_unused_qk_scale_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point1_unused_qk_scale_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.11988352077489 / 60` to
    `21.04781972775779 / 60`.
  - Mean speedup regressed from `0.36866472534624817` to `0.3507969954626299`.
  - Median speedup regressed from `0.35471224516812827` to `0.33855521500512337`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.257109`.
    - `(128, 8, 1024, 256, causal=True)`: `0.279809`.
    - `(128, 8, 2048, 128, causal=True)`: `0.287832`.
    - `(128, 8, 2048, 256, causal=False)`: `0.465397`.
    - `(128, 8, 4096, 128, causal=False)`: `0.389279`.
    - `(128, 8, 8192, 64, causal=False)`: `0.425357`.
- Issues:
  - Every performance shape regressed, so the helper signature shape itself is part of the current favorable lowering.
  - This reinforces the previous unused K/V stride experiment: source-level simplification can perturb Triton-Ascend
    schedule even when an argument is semantically unused.
  - Do not remove unused helper constexpr arguments for this kernel without comparing `last_pass.mlir` and confirming the
    resulting schedule is unchanged or better.
- Reports:
  - `evaluation_reports/codex_point1_unused_qk_scale_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point1_unused_qk_scale_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 17: D128 short causal off-band index elision regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 17, redundant boundary/index operation elimination.
- Motivation:
  - The active best already routes the D128 long causal family
    `(128, 8, 2048, 128, causal=True, BM=64, BN=256)` through `ELIDE_UNUSED_MASK_INDEX=True`.
  - Existing causal profiles show high scalar/MTE pressure rather than a pure MMAD limit:
    `aic_scalar_ratio≈0.5989`, `aic_mte2_ratio≈0.3857`, and `aiv_scalar_ratio≈0.4198` for the D128 causal profile.
  - Hypothesis: the shorter D128 causal performance case
    `(128, 8, 1024, 128, causal=True, BM=64, BN=256)` should benefit from the same off-band `curr_n` elision because
    only masked diagonal loops need `curr_n = start_n + offs_n`.
- Content:
  - Temporarily widened the existing D128 causal guard from `n_ctx == 2048` to `n_ctx == 1024 or n_ctx == 2048`.
  - Left non-causal D256 index elision, D256 causal diagonal split, all tiling presets, persistent program count, workspace
    allocation, and math unchanged.
  - No new Triton arithmetic, grid shape, loop control, or dynamic NPU/core probing was added.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: host-side family dispatch only; the Triton checklist items for int64, comparison dtype, modulo,
    one-dimensional grid, task partitioning, mutable loop offsets, and `break`/`continue` were unchanged.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point17_d128_1024_causal_index_elide_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point17_d128_1024_causal_index_elide_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.11988352077489 / 60` to
    `21.578756970860205 / 60`.
  - Mean speedup regressed from `0.36866472534624817` to `0.35964594951433676`.
  - Median speedup regressed from `0.35471224516812827` to `0.34705502707939023`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.259068`.
    - `(128, 8, 1024, 256, causal=True)`: `0.284168`.
    - `(128, 8, 2048, 128, causal=True)`: `0.289084`.
    - `(128, 8, 2048, 256, causal=False)`: `0.475495`.
    - `(128, 8, 4096, 128, causal=False)`: `0.405026`.
    - `(128, 8, 8192, 64, causal=False)`: `0.445034`.
- Issues:
  - The targeted short D128 causal case regressed from the active best `0.270156` to `0.259068`, so the long-D128
    point17 win does not generalize to the shorter context.
  - The long D128 causal case also measured lower in this run despite keeping the same source value as the active best,
    which suggests the widened host constexpr/family expression perturbs kernel specialization or compile cache shape.
  - Keep `ELIDE_UNUSED_MASK_INDEX=True` restricted to the already-validated `n_ctx == 2048` D128 causal family unless a
    future MLIR comparison proves the shorter-context schedule is unchanged or better.
- Reports:
  - `evaluation_reports/codex_point17_d128_1024_causal_index_elide_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point17_d128_1024_causal_index_elide_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 2: D128 causal BM32 tiling rejection

- Commit: journal-only commit for this entry; no source change was made.
- Optimization point: 2, tiling optimization.
- Motivation:
  - A fresh current-source profile for `(128, 8, 1024, 128, causal=True)` shows the kernel is not a pure MMAD limit:
    `aic_mac_ratio=0.123`, `aic_scalar_ratio=0.6087`, `aic_mte2_ratio=0.3966`, `aiv_scalar_ratio=0.3845`.
  - The active D128 causal preset uses `(BM=64, BN=256)` to reduce KV-loop iteration count, but the wide causal diagonal
    tile also increases mask/vector pressure.
  - Hypothesis: lowering `BM` to `32` while keeping wide `BN` might reduce per-tile Vector/scalar pressure enough to
    offset the doubled number of M tiles.
- Content:
  - Ran a same-process targeted A/B probe on the two D128 causal performance shapes only.
  - Candidates tested with `attention(..., BM=<candidate>, BN=<candidate>)`:
    `(BM=32, BN=128)`, `(BM=32, BN=256)`, `(BM=64, BN=64)`, and `(BM=32, BN=64)`.
  - Each candidate was compared against `torch_npu.npu_fusion_attention` first and only timed after passing
    `torch.allclose(..., atol=1e-2, rtol=1e-2)`.
  - No tiling preset, kernel code, persistent program count, or dispatch rule was changed.
- Effect:
  - Fresh profile path:
    `profiling_runs/codex_pipe_current_d128_1024_causal/result_dir/profile_summary.json`.
  - `(128, 8, 1024, 128, causal=True)`, median candidate latency:
    - default `(BM=64, BN=256)`: `11515.780 us`;
    - `(BM=32, BN=128)`: `22305.435 us`;
    - `(BM=32, BN=256)`: `18861.290 us`;
    - `(BM=64, BN=64)`: `18570.360 us`;
    - `(BM=32, BN=64)`: `38129.830 us`.
  - `(128, 8, 2048, 128, causal=True)`, median candidate latency:
    - default `(BM=64, BN=256)`: `36124.399 us`;
    - `(BM=32, BN=128)`: `74344.245 us`;
    - `(BM=32, BN=256)`: `58784.245 us`;
    - `(BM=64, BN=64)`: `66363.399 us`;
    - `(BM=32, BN=64)`: `134978.745 us`.
- Issues:
  - The profile correctly identifies scalar/MTE pressure, but shrinking `BM` increases task count and synchronization
    density too much. It also does not reduce the number of KV-loop iterations enough to compensate.
  - This rules out a BM32 family for the D128 causal score cases under the current persistent scheduler.
  - Keep the active `(BM=64, BN=256)` D128 causal preset; future causal work should target synchronization/source-shape
    specialization rather than smaller M tiles.
- Reports:
  - `profiling_runs/codex_pipe_current_d128_1024_causal/result_dir/profile_summary.json`
  - Targeted inline same-process timing probe; no evaluator report was generated because source was not changed.

## 2026-07-25 - Optimization point 12: causal hz-major mapping regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 12, grid shape and multi-path specialization.
- Motivation:
  - The active `_attn_fwd_tile` already specializes tile-to-core mapping by `STAGE`: causal uses m-major assignment to
    spread triangular causal work across persistent programs, while non-causal keeps hz-major assignment for K/V locality.
  - The fresh D128 causal profile shows scalar/MTE/sync pressure rather than a pure MMAD limit:
    `aic_mac_ratio=0.123`, `aic_scalar_ratio=0.6087`, `aic_mte2_ratio=0.3966`, `aiv_scalar_ratio=0.3845`.
  - Hypothesis: making causal use hz-major assignment too could improve K/V locality and reduce MTE pressure enough to
    offset the known triangular-work imbalance.
- Content:
  - Temporarily changed `_attn_fwd_tile` to always compute:
    `task_hz_idx = linear_tile // num_tiles_m` and
    `task_m_idx = linear_tile - task_hz_idx * num_tiles_m`.
  - Removed the causal-only `STAGE == 3` m-major branch during the experiment.
  - Left tiling presets, persistent program count, workspace allocation, math, index elision guards, and all other kernel
    families unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point12_causal_hz_major_mapping_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point12_causal_hz_major_mapping_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.11988352077489 / 60` to
    `21.485401518517186 / 60`.
  - Mean speedup regressed from `0.36866472534624817` to `0.3580900253086198`.
  - Median speedup regressed from `0.35471224516812827` to `0.34816019978979773`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.276719` (better than active best `0.270156`).
    - `(128, 8, 1024, 256, causal=True)`: `0.279126` (worse than active best `0.293738`).
    - `(128, 8, 2048, 128, causal=True)`: `0.305744` (better than active best `0.299902`).
    - `(128, 8, 2048, 256, causal=False)`: `0.470448` (worse than active best `0.493121`).
    - `(128, 8, 4096, 128, causal=False)`: `0.390577` (worse than active best `0.409523`).
    - `(128, 8, 8192, 64, causal=False)`: `0.425927` (worse than active best `0.445548`).
- Issues:
  - The source-level branch removal perturbed the shared non-causal lowering even though the intended behavior change was
    causal-only; all non-causal performance shapes regressed.
  - Causal hz-major is not uniformly positive: D128 causal 1024/128 and 2048/128 improved slightly, but 1024/256
    regressed enough that the family is not safe as a broad causal replacement.
  - Future point 12 work should use a separate kernel family or a much narrower host dispatch guard, not edit the shared
    `_attn_fwd_tile` branch in place.
- Reports:
  - `evaluation_reports/codex_point12_causal_hz_major_mapping_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point12_causal_hz_major_mapping_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 12/18: narrow D128 causal hz-major kernel family

- Commit: source + journal commit for this positive optimization.
- Optimization point: 12, grid shape and multi-path specialization; applied as point 18-style kernel family splitting
  because the evaluator is multi-case and the generic path is still below the target speedup.
- Motivation:
  - The previous in-place causal hz-major mapping experiment improved two D128 causal/head128 shapes but regressed the
    shared non-causal lowering, so the mapping itself had signal but was unsafe as a shared kernel edit.
  - The D128 causal profile is scalar/MTE/sync-heavy (`aic_mac_ratio=0.123`, `aic_scalar_ratio=0.6087`,
    `aic_mte2_ratio=0.3966`, `aiv_scalar_ratio=0.3845`), so changing persistent tile order can affect K/V locality and
    scheduler balance without changing math.
  - Hypothesis: a separate hz-major causal wrapper, routed only for the positive D128/head128 family, can keep the local
    K/V locality gain while preserving the existing fallback lowering for all other cases.
- Content:
  - Added `_attn_fwd_causal_hz_major`, a dedicated wrapper kernel that iterates tiles in hz-major order and remaps each
    hz-major tile id back to the m-major `linear_tile` consumed by the existing `_attn_fwd_tile` helper.
  - Kept `_attn_fwd_tile` and the generic `_attn_fwd` source unchanged, avoiding schedule perturbation for fallback
    non-causal and non-target causal paths.
  - Added a narrow host dispatch guard for:
    `causal=True`, `Z=128`, `H=8`, `HEAD_DIM=128`, `N_CTX in {1024, 2048}`, `BM=64`, `BN=256`,
    `USE_MAX=False`, and `ACC_IN_UB=False`.
  - Did not add dynamic NPU/core probing; `DEFAULT_PERSISTENT_PROGRAMS` remains `20`.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review:
    - New Triton code uses one grid dimension through the existing launch path and no `break`/`continue`.
    - No direct modulo was added; tile reindexing uses `a - (a / b) * b`.
    - No new int64-dependent arithmetic, PyTorch fallback computation, or dynamic device-property probing was added.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point12_causal_hz_family_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point12_causal_hz_family_performance/evaluation_report.json`.
  - Submit-style performance score improved from the active best `22.11988352077489 / 60` to
    `22.33033509351132 / 60`.
  - Mean speedup improved from `0.36866472534624817` to `0.37217225155852196`.
  - Median speedup improved from `0.35471224516812827` to `0.36062848394723923`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.286460` (active best was `0.270156`).
    - `(128, 8, 1024, 256, causal=True)`: `0.291285` (active best was `0.293738`; not routed to the new family).
    - `(128, 8, 2048, 128, causal=True)`: `0.315776` (active best was `0.299902`).
    - `(128, 8, 2048, 256, causal=False)`: `0.487401` (active best was `0.493121`; fallback path, within run variance).
    - `(128, 8, 4096, 128, causal=False)`: `0.405481` (active best was `0.409523`; fallback path, within run variance).
    - `(128, 8, 8192, 64, causal=False)`: `0.446629` (active best was `0.445548`; fallback path).
- Issues:
  - This is a narrow family, not a general causal replacement. The D256 causal performance case previously regressed under
    broad hz-major mapping and remains on the existing m-major fallback.
  - The wrapper adds a little tile-index arithmetic before calling `_attn_fwd_tile`; the benefit still survives for the two
    target D128/head128 shapes, but this pattern should not be expanded without full evaluator confirmation.
  - The total score is still far from the externally reported 130-point ceiling; remaining work likely requires deeper
    family specialization or IR/profile-driven reduction of scalar/MTE/sync instructions, not more broad source cleanup.
- Reports:
  - `evaluation_reports/codex_point12_causal_hz_family_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point12_causal_hz_family_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 12/18: D256 causal diag-split hz-major family regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 12, grid shape and multi-path specialization; evaluated as a point 18-style narrow kernel family.
- Motivation:
  - The D128 causal hz-major wrapper proved that tile-order specialization can improve scalar/MTE-heavy causal paths when
    the target family is isolated from shared fallback lowering.
  - The active D256 causal score case uses `_attn_fwd_causal_diag_split`, whose profile also shows non-MMAD pressure
    (`aic_mac_ratio~0.138`, `aic_scalar_ratio~0.5828`, `aic_mte2_ratio~0.413`).
  - Hypothesis: adding the same hz-major wrapper around the diag-split tile could improve K/V locality for
    `(128, 8, 1024, 256, causal=True)` without changing the generic D128/non-causal paths.
- Content:
  - Temporarily added `_attn_fwd_causal_diag_split_hz_major`, mirroring the successful D128 wrapper pattern:
    iterate `linear_tile_hz`, compute `(task_hz_idx, task_m_idx)`, remap to m-major `linear_tile_m`, and call the existing
    `_attn_fwd_causal_diag_split_tile`.
  - Temporarily routed only the D256 diag-split score shape
    `Z=128,H=8,N_CTX=1024,HEAD_DIM=256,causal=True,BM=64,BN=128` to the new wrapper.
  - Did not change tiling presets, persistent program count, generic `_attn_fwd`, D128 hz-major family, math, or
    workspace policy.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point12_d256_diag_hz_family_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point12_d256_diag_hz_family_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.33033509351132 / 60` to
    `21.300989487325936 / 60`.
  - Mean speedup regressed from `0.37217225155852196` to `0.35501649145543224`.
  - Median speedup regressed from `0.36062848394723923` to `0.3481486269270778`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.274513` (active best was `0.286460`; D128 fallback/family changed only by
      run variance, but measured lower in this full run).
    - `(128, 8, 1024, 256, causal=True)`: `0.261613` (active best was `0.291285`; targeted D256 family regressed hard).
    - `(128, 8, 2048, 128, causal=True)`: `0.304257` (active best was `0.315776`).
    - `(128, 8, 2048, 256, causal=False)`: `0.469195` (active best was `0.487401`).
    - `(128, 8, 4096, 128, causal=False)`: `0.392040` (active best was `0.405481`).
    - `(128, 8, 8192, 64, causal=False)`: `0.428481` (active best was `0.446629`).
- Issues:
  - For D256 diag-split, m-major load balancing is more important than possible K/V locality from hz-major traversal.
  - The target D256 case regressed from `0.291285` to `0.261613`, so this family must remain rejected.
  - Future D256 causal work should target the diag-split inner-loop scalar/MTE/sync structure directly, not only persistent
    tile order.
- Reports:
  - `evaluation_reports/codex_point12_d256_diag_hz_family_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point12_d256_diag_hz_family_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 25: D128 causal hz-major compile-param specialization

- Commit: source + journal commit for this positive optimization.
- Optimization point: 25, IR analysis optimization, applied through NPU CV-fusion compile parameters on the existing
  point 12/18 D128 causal hz-major kernel family.
- Motivation:
  - The current D128 causal hz-major family profile for `(128, 8, 1024, 128, causal=True)` reports low MMAD share and
    high scalar/MTE/sync pressure rather than a hard Cube limit: `aic_mac_ratio=0.129`, `aic_scalar_ratio=0.587`,
    `aic_mte2_ratio=0.256`, `aiv_scalar_ratio=0.423`, with `cube_utilization(%)=99.542`.
  - IR comparison showed the wrapper improved measured MTE2 locality despite increasing some remap/sync counts, so the
    remaining actionable direction was to let the compiler's CV scheduler handle mixed Cube/Vector synchronization and
    buffering more explicitly.
  - To avoid perturbing fallback lowering, the experiment was restricted to the already-positive
    `_attn_fwd_causal_hz_major` launch family.
- Content:
  - Tried the full FlashAttention CV parameter set from the Triton-Ascend compile-param guide:
    `multibuffer=True`, `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, `sync_solver=True`,
    `limit_auto_multi_buffer_of_local_buffer="no-limit"`, `enable_flatten=False`, and `set_workspace_multibuffer=2`.
  - Full set failed on the routed D128/head128 performance shapes because the local `bishengir-compile` does not accept
    `--enable-flatten=False` and suggests `--enable-loop-flatten=False`; this environment exposes `enable_flatten` in
    Python `NPUOptions`, but the backend lowers it to an unsupported compiler flag.
  - Applied the compile-failure rollback by removing only `enable_flatten=False` and keeping the compatible CV parameters:
    `multibuffer=True`, `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, `sync_solver=True`,
    `limit_auto_multi_buffer_of_local_buffer="no-limit"`, and `set_workspace_multibuffer=2`.
  - Kept the change scoped to `_attn_fwd_causal_hz_major[grid]`; generic `_attn_fwd`, diag-split D256, tiling presets,
    math, workspace policy, and `DEFAULT_PERSISTENT_PROGRAMS = 20` are unchanged.
- Effect:
  - Full parameter set:
    - Correctness suite: `18/18` passed in
      `evaluation_reports/codex_point25_hz_family_compile_params_correctness/evaluation_report.json`.
    - Performance suite: only `4/6` matched in
      `evaluation_reports/codex_point25_hz_family_compile_params_performance/evaluation_report.json`.
    - The two routed D128/head128 cases failed compilation with
      `Unknown command line argument '--enable-flatten=False'`, so that variant was rejected.
  - Compatible parameter set:
    - `python3 -m py_compile flash_attention_forward.py`: pass.
    - `git diff --check`: pass.
    - Correctness suite: `18/18` passed in
      `evaluation_reports/codex_point25_hz_family_compile_params_v2_correctness/evaluation_report.json`.
    - Performance suite: `6/6` matched in
      `evaluation_reports/codex_point25_hz_family_compile_params_v2_performance/evaluation_report.json`.
    - Submit-style performance score improved from the active best `22.33033509351132 / 60` to
      `22.361347632011938 / 60`.
    - Mean speedup improved from `0.37217225155852196` to `0.372689127200199`.
    - Median speedup improved from `0.36062848394723923` to `0.36618297498563374`.
    - Per-shape speedups after the compatible variant:
      - `(128, 8, 1024, 128, causal=True)`: `0.286085` (active best was `0.286460`; roughly tied/slightly lower).
      - `(128, 8, 1024, 256, causal=True)`: `0.289001` (active best was `0.291285`; fallback path, run variance).
      - `(128, 8, 2048, 128, causal=True)`: `0.324891` (active best was `0.315776`; routed family improved).
      - `(128, 8, 2048, 256, causal=False)`: `0.479278` (active best was `0.487401`; fallback path, run variance).
      - `(128, 8, 4096, 128, causal=False)`: `0.407475` (active best was `0.405481`; fallback path).
      - `(128, 8, 8192, 64, causal=False)`: `0.449406` (active best was `0.446629`; fallback path).
- Issues:
  - The gain is small and concentrated on the long D128 causal/head128 routed shape; the short D128 routed case is within
    noise and slightly lower in this run.
  - `enable_flatten` is unsafe in this environment despite being present in `NPUOptions`; do not use it unless the backend
    flag mapping is updated or replaced with a proven supported option.
  - The compile-param family still does not solve the larger D256 causal and long non-causal gaps. Next work should use
    profile/IR evidence on those families rather than broadening this D128-only launch parameter set.
- Reports:
  - `evaluation_reports/codex_point25_hz_family_compile_params_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_hz_family_compile_params_performance/evaluation_report.json`
  - `evaluation_reports/codex_point25_hz_family_compile_params_v2_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_hz_family_compile_params_v2_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 25: D256 causal diag-split compile-param regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 25, IR analysis optimization, tested through the same compatible CV-fusion compile parameters used
  by the positive D128 causal hz-major family.
- Motivation:
  - The D256 causal performance case remains one of the weakest score shapes.
  - The previous D256 hz-major family failed, showing that tile-order locality is not enough and that m-major balancing is
    important for diag-split.
  - Hypothesis: keeping the existing diag-split tile order but enabling explicit CV compiler scheduling might reduce
    scalar/MTE/sync overhead without changing the math or persistent traversal.
- Content:
  - Temporarily added the compatible CV parameter set to only `_attn_fwd_causal_diag_split[grid]`:
    `multibuffer=True`, `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, `sync_solver=True`,
    `limit_auto_multi_buffer_of_local_buffer="no-limit"`, and `set_workspace_multibuffer=2`.
  - Did not add `enable_flatten=False` because the D128 compile-param experiment already proved that the local backend
    lowers it to unsupported `--enable-flatten=False`.
  - Left `_attn_fwd_causal_hz_major`, generic `_attn_fwd`, tiling presets, workspace policy, and persistent program count
    unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Performance suite: only `5/6` matched in
    `evaluation_reports/codex_point25_d256_diag_compile_params_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.361347632011938 / 60` to
    `18.67354498092374 / 60`.
  - The targeted `(128, 8, 1024, 256, causal=True)` case failed compilation with CC overflow:
    `requires 1572864 bits while 1048576 bits available`, with the compiler pointing to multi-buffer extra local buffer
    usage as a possible reason.
  - Other measured speedups in that run:
    - `(128, 8, 1024, 128, causal=True)`: `0.277992`.
    - `(128, 8, 2048, 128, causal=True)`: `0.312730`.
    - `(128, 8, 2048, 256, causal=False)`: `0.464627`.
    - `(128, 8, 4096, 128, causal=False)`: `0.386391`.
    - `(128, 8, 8192, 64, causal=False)`: `0.425614`.
- Issues:
  - The D256 diag-split path is CC/register-pressure limited under this compile-parameter set; enabling multibuffer and
    workspace multibuffer increases local-buffer demand beyond available capacity.
  - This confirms D256 causal needs a lower-footprint inner-loop/kernel-family change, not the D128-compatible CV launch
    parameter bundle.
  - The source change was reverted; final code after this entry remains the positive D128 hz-major compile-param version.
- Reports:
  - `evaluation_reports/codex_point25_d256_diag_compile_params_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 25: generic fallback compile-param regression

- Commit: journal-only commit for this entry; source was reverted after the correctness regression.
- Optimization point: 25, IR analysis optimization, tested by applying the compatible CV-fusion compile-parameter bundle
  to the shared fallback `_attn_fwd[grid]` launch.
- Motivation:
  - The D128 causal hz-major family improved slightly with explicit CV scheduling parameters, so the next question was
    whether the same compatible bundle could help the remaining fallback non-causal performance shapes.
  - This was intentionally tested as a broad fallback experiment first to expose compile-capacity and correctness risks
    before adding any new non-causal family split.
- Content:
  - Temporarily added to the generic `_attn_fwd[grid]` launch:
    `multibuffer=True`, `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, `sync_solver=True`,
    `limit_auto_multi_buffer_of_local_buffer="no-limit"`, and `set_workspace_multibuffer=2`.
  - Kept `enable_flatten` disabled/omitted because it is an unsupported compiler flag in this environment.
  - Did not change `_attn_fwd_tile`, tiling presets, persistent program count, math, or the existing D128 hz-major
    compile-param family.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite regressed to `12/18` in
    `evaluation_reports/codex_point25_fallback_compile_params_correctness/evaluation_report.json`.
  - Failed correctness shapes:
    - `(1, 2, 1024, 64, causal=False)`.
    - `(4, 32, 1024, 64, causal=False)`.
    - `(4, 32, 1024, 128, causal=False)`.
    - `(4, 32, 2048, 128, causal=False)`.
    - `(4, 32, 4096, 64, causal=False)`.
    - `(128, 8, 1024, 64, causal=False)`.
  - All failures were compile-time `cc overflow`, requiring `2097152 bits` while only `1048576 bits` were available,
    again pointing to multi-buffer extra local-buffer usage.
  - Performance was not run because the correctness gate failed.
- Issues:
  - The shared fallback kernel has enough live CC/local-buffer pressure that the D128-only compile-param bundle cannot be
    applied broadly.
  - Future non-causal work must use narrower family splits and lower-footprint changes. A blanket compile-param pass is
    not viable and would also re-break public correctness.
  - The source change was reverted; final code after this entry remains the positive D128 hz-major compile-param version.
- Reports:
  - `evaluation_reports/codex_point25_fallback_compile_params_correctness/evaluation_report.json`

## 2026-07-25 - Optimization point 25: generic fallback sync-param regression

- Commit: journal-only commit for this entry; source was reverted after the performance regression.
- Optimization point: 25, IR analysis optimization, tested as the compile-failure rollback of the broader fallback
  compile-param experiment.
- Motivation:
  - The broad fallback CV bundle failed correctness because `multibuffer`/workspace buffering caused CC overflow.
  - Hypothesis: removing the capacity-heavy parameters and keeping only synchronization/scheduling hints might avoid the
    overflow while still improving Cube/Vector coordination on fallback non-causal paths.
- Content:
  - Temporarily added only `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, and `sync_solver=True` to the shared
    `_attn_fwd[grid]` fallback launch.
  - Omitted `multibuffer`, `limit_auto_multi_buffer_of_local_buffer`, `set_workspace_multibuffer`, and `enable_flatten`.
  - Kept the existing D128 hz-major compile-param family unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point25_fallback_sync_params_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point25_fallback_sync_params_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.361347632011938 / 60` to
    `21.482344699397753 / 60`.
  - Mean speedup regressed from `0.372689127200199` to `0.35803907832329585`.
  - Median speedup regressed from `0.36618297498563374` to `0.35110031584977613`.
  - Per-shape speedups after the experiment:
    - `(128, 8, 1024, 128, causal=True)`: `0.278009`.
    - `(128, 8, 1024, 256, causal=True)`: `0.276931`.
    - `(128, 8, 2048, 128, causal=True)`: `0.314855`.
    - `(128, 8, 2048, 256, causal=False)`: `0.467981`.
    - `(128, 8, 4096, 128, causal=False)`: `0.387346`.
    - `(128, 8, 8192, 64, causal=False)`: `0.423114`.
- Issues:
  - Removing multibuffer fixed the compile-capacity failure, but the remaining mixed-CV/sync hints still degraded the
    shared fallback lowering across all measured performance shapes.
  - This rules out generic fallback launch-parameter tuning. Any future non-causal optimization should be a narrower
    kernel family with a structural source change and independent profile proof.
  - The source change was reverted; final code after this entry remains the positive D128 hz-major compile-param version.
- Reports:
  - `evaluation_reports/codex_point25_fallback_sync_params_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_fallback_sync_params_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 14: D128 causal compile-param strategy split

- Commit: source + journal commit for this positive optimization.
- Optimization point: 14, mixed strategy automatic selection, using shape-specific launch-parameter selection on the
  existing D128 causal hz-major family.
- Motivation:
  - The previous D128 hz-major compile-param optimization improved the aggregate score, but its per-shape signal was
    concentrated on `(128, 8, 2048, 128, causal=True)`.
  - In that run, `(128, 8, 1024, 128, causal=True)` was roughly tied/slightly lower than the no-compile-param family, so
    applying the CV parameter bundle to both D128 causal contexts was probably too broad.
  - Hypothesis: route `N_CTX=1024` to the same hz-major family without explicit CV parameters while keeping `N_CTX=2048`
    on the compatible CV parameter bundle.
- Content:
  - Added a host-side branch before the existing D128 causal hz-major launch:
    `use_causal_hz_major_family and n_ctx == 1024` now calls `_attn_fwd_causal_hz_major[grid]` without CV launch
    parameters.
  - The existing `use_causal_hz_major_family` branch remains the `N_CTX=2048` path with:
    `multibuffer=True`, `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, `sync_solver=True`,
    `limit_auto_multi_buffer_of_local_buffer="no-limit"`, and `set_workspace_multibuffer=2`.
  - Did not use dynamic kwargs for Triton launch; arguments stay explicit to match the local code style and reduce runtime
    dispatch risk.
  - Generic `_attn_fwd`, D256 diag-split, tiling presets, math, workspace policy, and
    `DEFAULT_PERSISTENT_PROGRAMS = 20` are unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point14_d128_compile_params_2048_only_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point14_d128_compile_params_2048_only_performance/evaluation_report.json`.
  - Submit-style performance score improved from the active best `22.361347632011938 / 60` to
    `22.400443902045232 / 60`.
  - Mean speedup improved from `0.372689127200199` to `0.3733407317007539`.
  - Median speedup improved from `0.36618297498563374` to `0.3668445691470613`.
  - Per-shape speedups after the strategy split:
    - `(128, 8, 1024, 128, causal=True)`: `0.286690`.
    - `(128, 8, 1024, 256, causal=True)`: `0.287265`.
    - `(128, 8, 2048, 128, causal=True)`: `0.325743`.
    - `(128, 8, 2048, 256, causal=False)`: `0.485639`.
    - `(128, 8, 4096, 128, causal=False)`: `0.407947`.
    - `(128, 8, 8192, 64, causal=False)`: `0.446762`.
- Issues:
  - The aggregate gain is small and partly within normal run variance on fallback non-causal shapes, but the full
    evaluator score improved and correctness stayed at 100%.
  - The duplicated launch argument list is intentional; using `**kwargs` for Triton launch was avoided because this file
    does not already use that pattern.
  - This confirms compile parameters should remain context-specific: `N_CTX=2048` D128 causal keeps the CV bundle, while
    `N_CTX=1024` is better left on the no-parameter hz-major family.
- Reports:
  - `evaluation_reports/codex_point14_d128_compile_params_2048_only_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point14_d128_compile_params_2048_only_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 25: D128 2048 sync-only compile-param regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 25, IR analysis optimization, evaluated as an ablation of the positive D128 causal 2048 compile
  parameter bundle.
- Motivation:
  - The positive D128 strategy split keeps explicit CV parameters only for `(128, 8, 2048, 128, causal=True)`.
  - The next question was whether the benefit comes from sync/mixed-CV scheduling alone or from the heavier
    multi-buffer/workspace part of the bundle.
  - Hypothesis: keeping only `enable_mixed_cv`, `enable_auto_bind_sub_block`, and `sync_solver` on the 2048 branch might
    preserve the scheduling benefit while reducing local-buffer pressure.
- Content:
  - Temporarily removed `multibuffer=True`, `limit_auto_multi_buffer_of_local_buffer="no-limit"`, and
    `set_workspace_multibuffer=2` from the D128 `N_CTX=2048` hz-major launch.
  - Kept only `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, and `sync_solver=True` on that branch.
  - Left the D128 `N_CTX=1024` no-parameter branch and generic fallback paths unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point25_d128_2048_sync_params_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point25_d128_2048_sync_params_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.400443902045232 / 60` to
    `21.288106951474145 / 60`.
  - Mean speedup regressed from `0.3733407317007539` to `0.35480178252456906`.
  - Median speedup regressed from `0.3668445691470613` to `0.3427818317946075`.
  - The targeted D128 long causal shape regressed from `0.325743` to `0.301954`.
- Issues:
  - For the D128 2048 hz-major family, sync/mixed-CV hints alone are not enough; the positive behavior depends on the
    full multi-buffer/workspace bundle.
  - The source change was reverted, restoring the full CV parameter set on the `N_CTX=2048` branch.
- Reports:
  - `evaluation_reports/codex_point25_d128_2048_sync_params_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_d128_2048_sync_params_performance/evaluation_report.json`

## 2026-07-25 - Optimization point 25: D128 2048 multibuffer-only compile-param regression

- Commit: journal-only commit for this entry; source was reverted after the regression.
- Optimization point: 25, IR analysis optimization, evaluated as the complementary ablation of the positive D128 causal
  2048 compile-parameter bundle.
- Motivation:
  - The sync-only ablation showed that `enable_mixed_cv + enable_auto_bind_sub_block + sync_solver` alone is not enough.
  - Hypothesis: the benefit might instead come mostly from `multibuffer` and workspace multi-buffering, and removing the
    mixed-CV/sync hints could simplify lowering while retaining MTE overlap.
- Content:
  - Temporarily kept only `multibuffer=True`, `limit_auto_multi_buffer_of_local_buffer="no-limit"`, and
    `set_workspace_multibuffer=2` on the D128 `N_CTX=2048` hz-major launch.
  - Temporarily removed `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, and `sync_solver=True` from that branch.
  - Left the D128 `N_CTX=1024` no-parameter branch and generic fallback paths unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point25_d128_2048_multibuffer_only_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point25_d128_2048_multibuffer_only_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.400443902045232 / 60` to
    `22.324314587175813 / 60`.
  - Mean speedup regressed from `0.3733407317007539` to `0.37207190978626353`.
  - Median speedup was roughly tied/slightly lower: `0.3668445691470613` to `0.36678923310465283`.
  - The targeted D128 long causal shape regressed from `0.325743` to `0.324044`.
- Issues:
  - The full bundle remains better than either half. The positive 2048 behavior appears to require both multi-buffering
    and mixed-CV/sync scheduling together.
  - The source change was reverted, restoring the full CV parameter set on the `N_CTX=2048` branch.
- Reports:
  - `evaluation_reports/codex_point25_d128_2048_multibuffer_only_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_d128_2048_multibuffer_only_performance/evaluation_report.json`

## 2026-07-26 - Optimization point 18: D256 causal diag-split parity kernel family

- Commit: source + journal commit for this positive optimization.
- Optimization point: 18, kernel splitting / kernel family specialization, applied narrowly to the D256 causal
  diag-split path.
- Motivation:
  - The D256 causal profile for `_attn_fwd_causal_diag_split` showed high scalar/MTE/sync pressure even though cube
    utilization was already high: `aic_scalar_ratio=0.5828`, `aic_mte2_ratio=0.413`,
    `aiv_scalar_ratio=0.3769`, `Wait Time(us)=561.770`, and `cube_utilization(%)=98.0739`.
  - Previous attempts ruled out broad D256 compile parameters because the multi-buffer/CV bundle hit CC overflow
    (`1572864 bits > 1048576 bits`), and persistent-program A/B showed `DEFAULT_PERSISTENT_PROGRAMS = 20` remained best.
  - The remaining structural issue in diag-split was that one kernel covered both even and odd M tiles. Even tiles have
    no half-width pre-diagonal loop, while odd tiles always have one half-width pre-diagonal loop. Hypothesis: splitting
    this parity at compile time could remove the runtime empty-loop/parity shape from the hot D256 path without changing
    math or tiling.
- Content:
  - Added `_attn_fwd_causal_diag_split_parity_tile` and `_attn_fwd_causal_diag_split_parity`.
  - The parity kernel maps `task_m_idx = task_pair_idx * 2 + TILE_PARITY`, avoiding modulo in Triton code.
  - `TILE_PARITY=0` compiles a path where `full_hi = diag_lo` and skips the half-width pre-diagonal loop.
  - `TILE_PARITY=1` compiles a path where `full_hi = diag_lo - BLOCK_M` and keeps the fixed half-width
    pre-diagonal loop.
  - Host routing is intentionally narrow: only
    `(z, h, n_ctx, head_dim, causal) == (128, 8, 1024, 256, True)` with `BM=64`, `BN=128`, lazy mode, and
    `acc_in_ub=True` launches the two parity kernels. Other D256 causal shapes keep the original
    `_attn_fwd_causal_diag_split` fallback.
  - Did not change generic `_attn_fwd`, D128 hz-major branches, compile parameters, tiling presets, or
    `DEFAULT_PERSISTENT_PROGRAMS = 20`.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d256_diag_parity_split_correctness/evaluation_report.json`.
  - Performance run 1: `6/6` matched in
    `evaluation_reports/codex_point18_d256_diag_parity_split_performance/evaluation_report.json`.
    - Submit-style performance score: `22.389497086109618 / 60`, below the active best
      `22.400443902045232 / 60`.
    - Mean speedup: `0.3731582847684936`; median speedup: `0.36581490624515356`.
    - Targeted D256 causal speedup improved from `0.28726495791201` to `0.28968493339333473`;
      candidate median latency improved from `17099.735327 us` to `16923.800576 us`.
  - Performance run 2: `6/6` matched in
    `evaluation_reports/codex_point18_d256_diag_parity_split_performance_r2/evaluation_report.json`.
    - Submit-style performance score: `22.43424911157306 / 60`, above the active best
      `22.400443902045232 / 60`.
    - Mean speedup: `0.373904151859551`; median speedup: `0.3661752970191059`.
    - Targeted D256 causal speedup improved from `0.28726495791201` to `0.29087665397851187`;
      candidate median latency improved from `17099.735327 us` to `16936.045140 us`.
- Issues:
  - Full performance remains noisy: the first full run was slightly below best because unrelated D128/non-causal shapes
    moved in opposite directions, while the second run was above best.
  - The target D256 causal shape improved in both runs, which is the only shape this source change routes differently.
  - Splitting by parity doubles launches for that one shape, but the measured target latency still improved, so launch
    overhead did not dominate this evaluator case.
  - This confirms the D256 path is more responsive to lower-footprint source-shape specialization than to compile-param
    tuning.
- Reports:
  - `evaluation_reports/codex_point18_d256_diag_parity_split_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d256_diag_parity_split_performance/evaluation_report.json`
  - `evaluation_reports/codex_point18_d256_diag_parity_split_performance_r2/evaluation_report.json`

## 2026-07-26 - Optimization point 18: D128 causal 1024 modulo-4 phase family regression

- Commit: journal-only commit for this entry; source was reverted after the performance regression.
- Optimization point: 18, kernel splitting / kernel family specialization, tested narrowly on the D128 causal
  `N_CTX=1024` hz-major path.
- Motivation:
  - The D128 causal hz-major IR showed relatively high synchronization and scalar/index overhead:
    `sync_block_set=24`, `sync_block_wait=24`, `wait_flag=50`, `set_flag=50`, `pipe_barrier=19`,
    `arith.index_cast=40`, `arith.muli=26`, and `arith.divsi=12`.
  - Since the D256 causal parity split improved its target shape, the analogous D128 idea was to split the diagonal
    tile phase by `task_m_idx % 4` for `BM=64`, `BN=256`.
  - Hypothesis: specializing `TILE_PHASE=0..3` would make the diagonal mask shape compile-time fixed and reduce dynamic
    index/mask work on `(128, 8, 1024, 128, causal=True)`.
- Content:
  - Temporarily added `_attn_fwd_causal_hz_phase_tile` and `_attn_fwd_causal_hz_phase`.
  - The phase tile mapped `task_m_idx = task_group_idx * 4 + TILE_PHASE`, avoided `%` in Triton code, and used a local
    diagonal mask comparing `TILE_PHASE * BLOCK_M + arange(BLOCK_M)` against `arange(BLOCK_N)`.
  - Temporarily routed only `use_causal_hz_major_family and n_ctx == 1024` to four sequential
    `_attn_fwd_causal_hz_phase` launches with `TILE_PHASE=0..3`.
  - Left the D128 `N_CTX=2048` full CV-parameter branch, D256 parity split, generic fallback, tiling presets, and
    `DEFAULT_PERSISTENT_PROGRAMS = 20` unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d128_phase1024_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d128_phase1024_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.43424911157306 / 60` to
    `20.5853183615137 / 60`.
  - Mean speedup regressed from `0.373904151859551` to `0.34308863935856165`.
  - Median speedup regressed from `0.3661752970191059` to `0.3502905776046385`.
  - The targeted D128 causal 1024 shape regressed sharply from speedup `0.2862688260353736` to
    `0.18938428477985841`; candidate median latency worsened from `10483.604856 us` to `15848.594718 us`.
  - Other measured shape deltas in the same run also moved negative, likely due to global timing/load noise after the
    first shape slowed substantially:
    - D256 causal 1024 speedup `0.29087665397851187` -> `0.27985987932454764`.
    - D128 causal 2048 speedup `0.3265679147525726` -> `0.31204572947622167`.
    - D256 non-causal 2048 speedup `0.48396344081902276` -> `0.46646457570291405`.
    - D128 non-causal 4096 speedup `0.40578267928563927` -> `0.38853542573305533`.
    - D64 non-causal 8192 speedup `0.449965396286186` -> `0.4222419411347727`.
- Issues:
  - Four sequential phase launches dominate any benefit from the compile-time local diagonal mask.
  - Splitting a D128 `BN=256` diagonal tile into modulo-4 phases is not analogous to the successful D256 parity split:
    D256 parity still used two launches and reduced an empty/half diagonal loop shape; D128 phase split introduced four
    launches and lost too much scheduling/KV locality.
  - The source change was reverted. Future D128 work should avoid multi-launch phase splitting and instead focus on a
    single-launch source-shape simplification or profile-guided compile-param changes inside the existing hz-major
    family.
- Reports:
  - `evaluation_reports/codex_point18_d128_phase1024_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d128_phase1024_performance/evaluation_report.json`

## 2026-07-26 - Optimization point 18: D128 causal 1024 single-launch local-diagonal regression

- Commit: journal-only commit for this entry; source was reverted because the aggregate evaluator score did not improve.
- Optimization point: 18, kernel splitting / kernel family specialization, tested narrowly on the D128 causal
  `N_CTX=1024` hz-major path.
- Motivation:
  - The previous D128 modulo-4 phase split confirmed that four sequential launches are too expensive for this shape.
  - The profile/IR evidence still points at scalar/index/sync pressure in the D128 hz-major path:
    `sync_block_set=24`, `sync_block_wait=24`, `wait_flag=50`, `set_flag=50`, `pipe_barrier=19`,
    `arith.index_cast=40`, `arith.muli=26`, and `arith.divsi=12`.
  - Hypothesis: keeping a single launch while locally specializing the diagonal block could reduce dynamic mask/index
    work without paying the multi-launch penalty.
- Content:
  - Temporarily added `_attn_fwd_causal_hz_local_diag_tile` and `_attn_fwd_causal_hz_local_diag`.
  - The local-diagonal tile kept the existing hz-major linear tile mapping, derived `task_phase` with
    `task_m_idx - task_group_idx * 4` to avoid `%`, processed pre-diagonal blocks through `_attn_fwd_inner_loop`, and
    handled the diagonal `BN=256` block with a local mask based on `task_phase * BLOCK_M + arange(BLOCK_M)`.
  - Temporarily routed only `use_causal_hz_major_family and n_ctx == 1024` to this single-launch local-diagonal family.
  - Left the D128 `N_CTX=2048` full CV-parameter branch, D256 parity split, generic fallback, tiling presets, and
    `DEFAULT_PERSISTENT_PROGRAMS = 20` unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d128_local_diag1024_correctness/evaluation_report.json`.
  - Performance run 1: `6/6` matched in
    `evaluation_reports/codex_point18_d128_local_diag1024_performance/evaluation_report.json`.
    - Submit-style performance score regressed from the active best `22.43424911157306 / 60` to
      `22.363591885440123 / 60`.
    - Mean speedup regressed from `0.373904151859551` to `0.372726531424002`.
    - Median speedup moved from `0.3661752970191059` to `0.36812911343522947`.
    - Targeted D128 causal 1024 speedup improved from `0.2862688260353736` to `0.2884358517810714`;
      candidate median latency improved from `10483.604856 us` to `10348.939802 us`.
  - Performance run 2: `6/6` matched in
    `evaluation_reports/codex_point18_d128_local_diag1024_performance_r2/evaluation_report.json`.
    - Submit-style performance score remained below best at `22.396753930040745 / 60`.
    - Mean speedup remained below best at `0.37327923216734576`.
    - Median speedup was `0.36877514209269135`.
    - Targeted D128 causal 1024 speedup again improved from `0.2862688260353736` to `0.28848541992612425`;
      candidate median latency improved from `10483.604856 us` to `10334.769730 us`.
- Issues:
  - The source change improved only the intended target shape by about 1.4% latency, but both full performance runs
    stayed below the active best aggregate evaluator score.
  - The aggregate regression was dominated by unrelated non-causal long-shape noise/regression in the same runs, but the
    evaluator score is the acceptance criterion, so this code was not adopted.
  - Single-launch local diagonal handling is much better than the four-launch phase split, but the gain is too small to
    carry the full score. Future D128 work should look for larger structural changes, not further phase splitting.
- Reports:
  - `evaluation_reports/codex_point18_d128_local_diag1024_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d128_local_diag1024_performance/evaluation_report.json`
  - `evaluation_reports/codex_point18_d128_local_diag1024_performance_r2/evaluation_report.json`

## 2026-07-26 - Optimization point 18: D256 non-causal no-index-loop family regression

- Commit: journal-only commit for this entry; source was reverted because both the target shape and aggregate score
  regressed.
- Optimization point: 18, kernel splitting / kernel family specialization, tested narrowly on the D256 non-causal
  performance shape `(128, 8, 2048, 256, causal=False)`.
- Motivation:
  - The existing D256 non-causal mask-index-elision family is one of the few positive non-causal source-level changes,
    and the remaining D256 path still uses the generic `_attn_fwd` wrapper with compile-time branches for stage,
    accumulator residency, stable/lazy softmax, and causal masking.
  - Hypothesis: a dedicated D256 non-causal kernel family could keep the same tiling `(BM=128, BN=128)` and math while
    removing unused branch shapes and the explicit `start_n = tl.multiple_of(start_n, BLOCK_N)` loop-index work from
    the hot KV loop.
- Content:
  - Temporarily added `_attn_fwd_noncausal_d256_tile` and `_attn_fwd_noncausal_d256`.
  - The specialized tile fixed `HEAD_DIM=256`, `BLOCK_M=128`, `BLOCK_N=128`, `USE_MAX=False`, `ACC_IN_UB=False`, and
    non-causal full-context traversal at compile time.
  - The KV loop used `for _ in tl.range(0, N_CTX, BLOCK_N)` with pointer advancement only, so no `curr_n` or
    `tl.multiple_of(start_n, BLOCK_N)` value was materialized.
  - Host routing was restricted to `(z, h, n_ctx, head_dim, causal) == (128, 8, 2048, 256, False)` with the current
    preset tiling and lazy GM accumulator. D128/D64 non-causal paths, D128 causal hz-major, D256 causal parity split,
    compile parameters, tiling presets, and `DEFAULT_PERSISTENT_PROGRAMS = 20` were unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d256_noncausal_loop_family_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d256_noncausal_loop_family_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.43424911157306 / 60` to
    `21.506431258896917 / 60`.
  - Mean speedup regressed from `0.373904151859551` to `0.3584405209816153`.
  - Median speedup regressed from `0.3661752970191059` to `0.3511130638715788`.
  - Targeted D256 non-causal speedup regressed from `0.48396344081902276` to `0.4715778906089937`;
    candidate median latency worsened from `72616.715450 us` to `74976.405129 us`.
- Issues:
  - The generic branch/index shape was not the limiting factor for this D256 non-causal case; duplicating the loop into a
    separate family likely worsened lowering/scheduling and lost the benefit of the existing generic code shape.
  - All other performance shapes measured lower in the same run even though they were not routed to the new family, so
    this experiment also had an unfavorable full-run environment. The target shape itself regressed, so no rerun was
    needed for the adoption decision.
  - Do not retry this "no-index-loop" specialization for D256 non-causal without fresh IR evidence that the loop index is
    still present in the optimized IR and dominates scalar cost.
- Reports:
  - `evaluation_reports/codex_point18_d256_noncausal_loop_family_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d256_noncausal_loop_family_performance/evaluation_report.json`

## 2026-07-26 - Optimization point 18: no-LSE-store kernel family specialization

- Commit: source + journal commit for this positive optimization.
- Optimization point: 18, kernel splitting / kernel family specialization, applied to the default evaluator path where
  `attention(..., return_lse=False)` returns only the attention output.
- Motivation:
  - The public/evaluator call path only consumes `attention()`'s `out`; it never requests the auxiliary LSE/M tensor.
  - The existing kernels still allocated a full `(Z, H, N_CTX)` fp32 `lse` tensor and wrote `M` for every tile. On the
    long performance shapes this is extra GM allocation/MTE3 work that does not contribute to the score.
  - Strict optimization point 21 was reviewed but not used as the formal hit because this forward kernel does not have
    the multi-pass loop-order conflict required by `workspace-decoupling.md`. The correct framing is point 18: specialize
    the default no-LSE kernel family while preserving the original LSE-producing path.
- Content:
  - Added `STORE_LSE: tl.constexpr` to `_attn_fwd_tile`, `_attn_fwd`, `_attn_fwd_causal_hz_major`,
    `_attn_fwd_causal_diag_split(_tile)`, and `_attn_fwd_causal_diag_split_parity(_tile)`.
  - Wrapped the `M + task_hz_idx * N_CTX + offs_m` store in `if STORE_LSE:` for all generic, hz-major, diag-split, and
    parity diag-split families.
  - Changed `_launch_kernel(..., return_lse=False)` to allocate a one-element dummy fp32 LSE tensor and launch kernels
    with `STORE_LSE=False`.
  - Kept `attention(..., return_lse=True)` behavior intact: it allocates the full `(Z,H,N_CTX)` LSE tensor, launches the
    same kernels with `STORE_LSE=True`, and returns `(out, lse)`.
  - Did not change tiling presets, math, lazy/stable softmax selection, D128 hz-major compile parameters, D256 parity
    split routing, generic fallback routing, or `DEFAULT_PERSISTENT_PROGRAMS = 20`.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - `return_lse=True` smoke test on `(1,1,64,64, causal=False)` returned `out.shape=(1,1,64,64)` and
    `lse.shape=(1,1,64)`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_no_lse_store_correctness/evaluation_report.json`.
  - Performance run 1: `6/6` matched in
    `evaluation_reports/codex_point18_no_lse_store_performance/evaluation_report.json`.
    - Submit-style performance score was effectively tied but slightly below the previous best:
      `22.43218328741033 / 60` vs `22.43424911157306 / 60`.
    - Mean speedup `0.3738697214568388`; median speedup `0.36733335915679155`.
  - Performance run 2: `6/6` matched in
    `evaluation_reports/codex_point18_no_lse_store_performance_r2/evaluation_report.json`.
    - Submit-style performance score improved to `22.472169993216955 / 60`, above the previous best
      `22.43424911157306 / 60`.
    - Expected full score with correctness is `62.472169993216955 / 100`.
    - Mean speedup improved from `0.373904151859551` to `0.3745361665536159`.
    - Median speedup improved from `0.3661752970191059` to `0.36766529750681753`.
    - Per-shape speedup deltas versus the previous best run:
      - `(128,8,1024,128, causal=True)`: `0.2862688260353736 -> 0.2877251021086774`.
      - `(128,8,1024,256, causal=True)`: `0.29087665397851187 -> 0.28963282926082656`.
      - `(128,8,2048,128, causal=True)`: `0.3265679147525726 -> 0.32677337225783415`.
      - `(128,8,2048,256, causal=False)`: `0.48396344081902276 -> 0.48349845470275304`.
      - `(128,8,4096,128, causal=False)`: `0.40578267928563927 -> 0.40855722275580086`.
      - `(128,8,8192,64, causal=False)`: `0.449965396286186 -> 0.45103001823580335`.
- Issues:
  - The first full performance run was a near-tie/slight loss, so the effect is small and noisy rather than a large
    structural win.
  - The second full run crossed the active best by `0.037920881643895 / 60`, and the shapes most expected to benefit
    from less output materialization (D128 long non-causal and D64 long non-causal) improved in that run.
  - D256 causal and D256 non-causal were slightly lower by speedup ratio in run 2, though their candidate latencies were
    effectively unchanged. Keep this optimization because the full evaluator score improved and correctness was stable.
- Reports:
  - `evaluation_reports/codex_point18_no_lse_store_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_no_lse_store_performance/evaluation_report.json`
  - `evaluation_reports/codex_point18_no_lse_store_performance_r2/evaluation_report.json`

## 2026-07-26 - Optimization point 18: no-LSE dummy allocation elision regression

- Commit: journal-only commit for this entry; source was reverted after a broad performance regression.
- Optimization point: 18, kernel splitting / kernel family specialization, tested as a narrow follow-up to the positive
  no-LSE-store family.
- Motivation:
  - After adding `STORE_LSE=False`, the default evaluator path still allocated a one-element fp32 dummy LSE tensor so the
    kernel signature kept a valid unused `M` pointer.
  - Hypothesis: since `STORE_LSE=False` makes the `M` pointer dead in Triton IR, passing `out` as the unused `M` argument
    could remove even that tiny host/device allocation without changing the generated compute path.
- Content:
  - Temporarily changed `_launch_kernel(..., return_lse=False)` from allocating a one-element fp32 dummy LSE tensor to
    reusing `out` as the unused `M` argument.
  - Kept `STORE_LSE=False` and all `tl.store(M)` guards unchanged.
  - Did not change math, tiling, persistent programs, compile parameters, D128 hz-major routing, D256 parity routing, or
    `return_lse=True` behavior.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_no_lse_alias_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_no_lse_alias_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.472169993216955 / 60` to
    `21.551451554576147 / 60`.
  - Mean speedup regressed from `0.3745361665536159` to `0.3591908592429358`.
  - Median speedup regressed from `0.36766529750681753` to `0.3521602856972894`.
  - Every performance shape regressed:
    - `(128,8,1024,128, causal=True)`: `0.2877251021086774 -> 0.2762523369723267`.
    - `(128,8,1024,256, causal=True)`: `0.28963282926082656 -> 0.2813657482457977`.
    - `(128,8,2048,128, causal=True)`: `0.32677337225783415 -> 0.3134614430244421`.
    - `(128,8,2048,256, causal=False)`: `0.48349845470275304 -> 0.46824157177657566`.
    - `(128,8,4096,128, causal=False)`: `0.40855722275580086 -> 0.3908591283701367`.
    - `(128,8,8192,64, causal=False)`: `0.45103001823580335 -> 0.42496492706833583`.
- Issues:
  - Although `M` is dead under `STORE_LSE=False`, changing the runtime argument from a fp32 dummy tensor to the fp16
    output tensor appears to create a different kernel specialization/lowering and slows all shapes.
  - The one-element fp32 dummy allocation is therefore kept. Do not replace the unused `M` argument with `out` unless new
    IR evidence shows the pointer dtype no longer affects lowering.
  - The source change was reverted, restoring the positive no-LSE-store implementation from commit `4069490`.
- Reports:
  - `evaluation_reports/codex_point18_no_lse_alias_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_no_lse_alias_performance/evaluation_report.json`

## 2026-07-26 - Optimization point 18: no-LSE log elision specialization

- Commit: source + journal commit for this positive optimization.
- Optimization point: 18, kernel splitting / kernel family specialization, applied as a second no-LSE default-path
  specialization after `STORE_LSE=False`.
- Motivation:
  - The positive no-LSE-store family stopped writing the auxiliary `M/LSE` output on the evaluator path, but the tile
    epilogue still computed `m_i += tl.math.log(l_i)` before the guarded store.
  - `m_i + log(l_i)` is only needed when materializing LSE. The default `attention(..., return_lse=False)` path only
    needs `l_i` for output normalization, so the vector log/add is dead work there.
  - Hypothesis: moving the log/add into the `STORE_LSE` branch should reduce vector epilogue work without changing
    attention output or the `return_lse=True` path.
- Content:
  - Moved `m_i += tl.math.log(l_i)` under `if STORE_LSE:` in `_attn_fwd_tile`,
    `_attn_fwd_causal_diag_split_tile`, and `_attn_fwd_causal_diag_split_parity_tile`.
  - Kept output normalization unchanged: `accumulator = acc / l_i[:, None]`.
  - Kept the one-element fp32 dummy LSE tensor for the no-LSE path, because replacing it with `out` was measured as a
    broad regression.
  - Did not change tiling presets, math for `Out`, lazy/stable softmax selection, D128 hz-major compile parameters,
    D256 parity routing, generic fallback routing, or `DEFAULT_PERSISTENT_PROGRAMS = 20`.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - `return_lse=True` smoke test on `(1,1,64,64, causal=False)` returned `out.shape=(1,1,64,64)` and
    `lse.shape=(1,1,64)`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_no_lse_log_elide_correctness/evaluation_report.json`.
  - Performance run 1: `6/6` matched in
    `evaluation_reports/codex_point18_no_lse_log_elide_performance/evaluation_report.json`.
    - Submit-style performance score improved from `22.472169993216955 / 60` to
      `22.493326412621617 / 60`.
    - Mean speedup improved from `0.3745361665536159` to `0.3748887735436936`.
    - Median speedup improved from `0.36766529750681753` to `0.36795586855652795`.
  - Performance run 2: `6/6` matched in
    `evaluation_reports/codex_point18_no_lse_log_elide_performance_r2/evaluation_report.json`.
    - Submit-style performance score improved further to `22.506673665520136 / 60`.
    - Expected full score with correctness is `62.506673665520136 / 100`.
    - Mean speedup improved to `0.37511122775866895`.
    - Median speedup was `0.367469250279431`.
    - Per-shape speedup deltas versus the previous best run:
      - `(128,8,1024,128, causal=True)`: `0.2877251021086774 -> 0.2860793263628685`.
      - `(128,8,1024,256, causal=True)`: `0.28963282926082656 -> 0.2908101646927782`.
      - `(128,8,2048,128, causal=True)`: `0.32677337225783415 -> 0.3271592699844636`.
      - `(128,8,2048,256, causal=False)`: `0.48349845470275304 -> 0.4879122341826647`.
      - `(128,8,4096,128, causal=False)`: `0.40855722275580086 -> 0.40777923057439847`.
      - `(128,8,8192,64, causal=False)`: `0.45103001823580335 -> 0.4509271407548402`.
- Issues:
  - The gain is still small and performance remains noisy; causal D128 1024 and long non-causal D128/D64 did not improve
    by speedup ratio in run 2.
  - The repeated positive full-score result indicates the log/add was not fully removed by the compiler after the store
    guard, so keeping it inside `STORE_LSE` is beneficial for the default evaluator path.
  - `return_lse=True` continues to compile and return the expected shapes, but the lazy path's LSE numerical semantics
    are unchanged from the pre-existing implementation and are not used by the evaluator.
- Reports:
  - `evaluation_reports/codex_point18_no_lse_log_elide_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_no_lse_log_elide_performance/evaluation_report.json`
  - `evaluation_reports/codex_point18_no_lse_log_elide_performance_r2/evaluation_report.json`

## 2026-07-26 - Optimization point 25: D256 parity sync-only compile-param regression

- Commit: journal-only commit for this entry; source was reverted after the target family and aggregate score regressed.
- Optimization point: 25, IR/profile-guided compile-parameter specialization, tested narrowly on the existing D256
  causal diag-split parity family.
- Motivation:
  - The earlier D256 causal profile for the pre-parity diag-split path showed high scalar/MTE/sync pressure:
    `aic_scalar_ratio=0.583`, `aic_mte2_ratio=0.413`, `aiv_scalar_ratio=0.377`, and `sync_avg_us=16731.609`.
  - The full D128 CV compile-parameter bundle previously failed on D256 diag-split with CC overflow, so this probe used
    only the lighter sync/scheduling hints and left out `multibuffer`, `limit_auto_multi_buffer_of_local_buffer`, and
    `set_workspace_multibuffer`.
- Content:
  - Temporarily added `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, and `sync_solver=True` to the two
    `_attn_fwd_causal_diag_split_parity[grid]` launches for `TILE_PARITY=0` and `TILE_PARITY=1`.
  - Kept tiling, math, `STORE_LSE`, D128 hz-major routing, D256 parity source code, generic fallback, and
    `DEFAULT_PERSISTENT_PROGRAMS = 20` unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point25_d256_parity_sync_params_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point25_d256_parity_sync_params_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to `22.245 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.3708`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2832675002826651`.
    - `(128,8,1024,256, causal=True)`: `0.28683603204530184` (targeted D256 parity family, below active best
      `0.2908101646927782`).
    - `(128,8,2048,128, causal=True)`: `0.32602213845276395`.
    - `(128,8,2048,256, causal=False)`: `0.4745716480916879`.
    - `(128,8,4096,128, causal=False)`: `0.407117671601809`.
    - `(128,8,8192,64, causal=False)`: `0.4466912047463913`.
- Issues:
  - The target D256 causal parity family slowed down, so the sync-only CV hints are not a valid follow-up to the positive
    parity split.
  - This matches the broader pattern seen in prior compile-param ablations: without the full multibuffer/workspace bundle
    the sync hints alone often perturb scheduling negatively, while the full bundle is too memory-heavy for D256.
  - The temporary source change was reverted. Keep the current no-parameter parity launches unless fresh IR from the
    parity family itself shows a different supported compile parameter is needed.
- Reports:
  - `evaluation_reports/codex_point25_d256_parity_sync_params_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_d256_parity_sync_params_performance/evaluation_report.json`

## 2026-07-26 - Optimization point 25: D128 causal 1024 full-CV current-baseline regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 25, IR/profile-guided compile-parameter specialization, retested narrowly on the current
  no-LSE/log-elided D128 causal `N_CTX=1024` hz-major family.
- Motivation:
  - The current D128 causal 1024 profile remains scalar/sync heavy:
    `aic_scalar_ratio=0.609`, `aic_mte2_ratio=0.397`, `aiv_scalar_ratio=0.385`, and `sync_avg_us=10899.674`.
  - A previous pre-no-LSE run showed the full CV bundle helped the `N_CTX=2048` D128 causal branch but did not clearly
    help `N_CTX=1024`. This retest checked whether the current no-LSE/log-elided source shape changed that decision.
- Content:
  - Temporarily added the same compatible full CV parameter bundle used by the D128 `N_CTX=2048` branch to only the
    `use_causal_hz_major_family and n_ctx == 1024` launch:
    `multibuffer=True`, `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, `sync_solver=True`,
    `limit_auto_multi_buffer_of_local_buffer="no-limit"`, and `set_workspace_multibuffer=2`.
  - Kept tiling, math, D256 parity routing, generic fallback, no-LSE behavior, and `DEFAULT_PERSISTENT_PROGRAMS = 20`
    unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point25_d128_1024_full_cv_current_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point25_d128_1024_full_cv_current_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `21.636485837928422 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.360608097298807`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2813864756273018` (targeted D128 1024 family, below active best
      `0.2860793263628685`).
    - `(128,8,1024,256, causal=True)`: `0.28281860008360254`.
    - `(128,8,2048,128, causal=True)`: `0.31376573362070004`.
    - `(128,8,2048,256, causal=False)`: `0.46962375358024333`.
    - `(128,8,4096,128, causal=False)`: `0.39007173241697524`.
    - `(128,8,8192,64, causal=False)`: `0.42598228846401914`.
- Issues:
  - The target D128 1024 branch still prefers the no-parameter hz-major launch. The full CV bundle perturbs scheduling
    enough to slow both the target case and the aggregate score.
  - This confirms the existing point 14 strategy split remains valid on the current no-LSE/log-elided baseline:
    `N_CTX=1024` stays no-param, `N_CTX=2048` keeps the full CV parameter bundle.
  - The temporary source change was reverted.
- Reports:
  - `evaluation_reports/codex_point25_d128_1024_full_cv_current_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_d128_1024_full_cv_current_performance/evaluation_report.json`

## 2026-07-26 - Optimization point 7: no-LSE lazy m_i zero-init regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 7, pass elimination / unused lazy LSE state cleanup.
- Motivation:
  - On the default evaluator path `STORE_LSE=False`, all scored performance shapes use lazy softmax (`USE_MAX=False`).
  - In those paths `m_i` is not needed for output normalization after the no-LSE log elision, but the source still
    initialized it with `tl.full(..., -inf)` and threaded it through the inner loop.
  - Hypothesis: using `tl.zeros` for no-LSE lazy `m_i` while preserving `-inf` for `USE_MAX=True` and `STORE_LSE=True`
    could make the dead state easier for the compiler to eliminate or cheaper to materialize.
- Content:
  - Temporarily changed `_attn_fwd_tile()` so `m_i` is initialized to `-inf` only for stable softmax or LSE-producing
    lazy paths, and to zeros for no-LSE lazy paths.
  - Applied the same `STORE_LSE`-guarded `m_i` initialization to `_attn_fwd_causal_diag_split_tile()` and
    `_attn_fwd_causal_diag_split_parity_tile()`.
  - Did not change tiling, math for `Out`, D128 hz-major routing, D256 parity routing, compile parameters, or
    `DEFAULT_PERSISTENT_PROGRAMS = 20`.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point7_no_lse_lazy_m_zero_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point7_no_lse_lazy_m_zero_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `22.449079618732235 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.37415132697887055`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.28712973120467805`.
    - `(128,8,1024,256, causal=True)`: `0.2900894022551739`.
    - `(128,8,2048,128, causal=True)`: `0.32554674591415`.
    - `(128,8,2048,256, causal=False)`: `0.4896549561233495`.
    - `(128,8,4096,128, causal=False)`: `0.40551408506909065`.
    - `(128,8,8192,64, causal=False)`: `0.4469730413067812`.
- Issues:
  - Even though `m_i` is logically dead on the no-LSE lazy path, the zero-init branch did not improve the aggregate
    evaluator score. It likely changed lowering/scheduling more than it reduced actual vector work.
  - Keep the original unconditional `-inf` source shape unless fresh IR proves the dead `m_i` broadcast is still on the
    critical path and can be removed without introducing an extra scheduling branch.
  - The temporary source change was reverted.
- Reports:
  - `evaluation_reports/codex_point7_no_lse_lazy_m_zero_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point7_no_lse_lazy_m_zero_performance/evaluation_report.json`

## 2026-07-26 - Optimization point 18: D128 causal 1024 lazy-UB explicit-add regression

- Commit: journal-only commit for this entry; source was reverted after validation.
- Optimization point: 18, kernel-family / shape-policy specialization, tested narrowly on the D128 causal
  `N_CTX=1024` lazy UB path.
- Motivation:
  - The current D128 causal 1024 profile remains scalar/sync/MTE-heavy rather than a clean MMAD limit, and this shape
    still uses the lazy `ACC_IN_UB=True` path with fused `tl.dot(p_cast, v, acc_ptr)` accumulation.
  - Hypothesis: for the short D128 causal family, forcing the explicit-add form
    `acc_ptr = acc_ptr + tl.dot(p_cast, v)` might reduce fused-accumulate scheduling pressure even though the qk+acc
    footprint fits L0C.
- Content:
  - Temporarily routed only `HEAD_DIM=128`, `BLOCK_M=64`, `BLOCK_N=256`, `N_CTX=1024` in the lazy UB branch to the
    explicit-add accumulate form.
  - Left tiling, D128 hz-major routing, D128 2048 compile parameters, D256 parity routing, no-LSE log/store elision,
    persistent program count, and math outside the accumulator update unchanged.
  - The first two source forms used chained boolean conditions and passed Python syntax checks, but failed Triton lowering
    with `UnsupportedLanguageConstruct('chained boolean operators ... are not supported')`. The final nested-condition
    form compiled and was used for the performance result below.
- Effect:
  - Final nested-condition version:
    - `python3 -m py_compile flash_attention_forward.py`: pass.
    - `git diff --check`: pass.
    - Correctness suite: `18/18` passed in
      `evaluation_reports/codex_point_new_lazyfuse3_correctness/evaluation_report.json`.
    - Performance suite: `6/6` matched in
      `evaluation_reports/codex_point_new_lazyfuse3_performance/evaluation_report.json`.
    - Submit-style performance score regressed from active best `22.506673665520136 / 60` to
      `21.41294238431406 / 60`.
    - Mean speedup regressed from `0.37511122775866895` to `0.356882373071901`.
    - Median speedup regressed from `0.367469250279431` to `0.34970985338101174`.
  - Per-shape speedups after the valid experiment:
    - `(128,8,1024,128, causal=True)`: `0.2860793263628685 -> 0.2741127110780622`.
    - `(128,8,1024,256, causal=True)`: `0.2908101646927782 -> 0.2794035607706744`.
    - `(128,8,2048,128, causal=True)`: `0.3271592699844636 -> 0.31131911122862344`.
    - `(128,8,2048,256, causal=False)`: `0.4879122341826647 -> 0.46700119698189846`.
    - `(128,8,4096,128, causal=False)`: `0.40777923057439847 -> 0.38810059553340004`.
    - `(128,8,8192,64, causal=False)`: `0.4509271407548402 -> 0.42135706283874746`.
- Issues:
  - The targeted D128 causal 1024 case itself slowed by about 4.2% speedup ratio, so the fused
    `tl.dot(..., acc_ptr)` form remains better for the lazy UB path.
  - The source-level nested specialization also perturbed unrelated fallback timings in the full evaluator run.
  - Triton-Ascend rejects chained boolean conditions in JIT expressions even when they are only over `tl.constexpr`
    values; future narrow compile-time predicates should use a host-provided single boolean or plain nested `if` blocks.
  - The temporary source change was reverted. Keep the current `LAZY_FUSE` footprint heuristic unchanged.
- Reports:
  - `evaluation_reports/codex_point_new_lazyfuse_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point_new_lazyfuse_performance/evaluation_report.json`
  - `evaluation_reports/codex_point_new_lazyfuse2_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point_new_lazyfuse3_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point_new_lazyfuse3_performance/evaluation_report.json`

## 2026-07-27 - Optimization point 11: K/V tile multibuffer hint regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 11, load instruction scheduling / CV pipeline hinting.
- Motivation:
  - Current profiles for the active best still show high scalar/MTE/sync pressure rather than a clean MMAD limit.
  - The historical `origin/ping-pong` prototype used `extension.multibuffer` on K/V tile loads, while the current
    no-LSE/lazy source no longer contains any explicit `extension.multibuffer(k/v, 2)` hint.
  - Hypothesis: adding a compiler multibuffer hint immediately after the current K/V `tl.load` operations could improve
    MTE/Cube overlap without changing math, tiling, host dispatch, workspace policy, or persistent program count.
- Content:
  - Temporarily added exactly two statements in `_attn_fwd_inner_loop()` after:
    `k = tl.load(k_block_ptr)` and `v = tl.load(v_block_ptr)`:
    `extension.multibuffer(k, 2)` and `extension.multibuffer(v, 2)`.
  - Kept `DEFAULT_PERSISTENT_PROGRAMS = 20`, all current tiling presets, D128 hz-major routing, D128 2048 CV bundle,
    D256 causal parity split, non-causal D256 index elision, no-LSE store/log elision, and lazy/stable math unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: the change did not add int64 arithmetic, comparisons, division/modulo, extra grid dimensions,
    interleaved task partitioning, mutable loop-index updates, `break`, or `continue`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point11_kv_multibuffer_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point11_kv_multibuffer_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `22.373042468024238 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.3728840411337373`.
  - Median speedup changed from `0.367469250279431` to `0.3686039322955988`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2860793263628685 -> 0.2860112042999078`.
    - `(128,8,1024,256, causal=True)`: `0.2908101646927782 -> 0.2928372287945872`.
    - `(128,8,2048,128, causal=True)`: `0.3271592699844636 -> 0.32656337324272394`.
    - `(128,8,2048,256, causal=False)`: `0.4879122341826647 -> 0.47364539534251276`.
    - `(128,8,4096,128, causal=False)`: `0.40777923057439847 -> 0.41064449134847364`.
    - `(128,8,8192,64, causal=False)`: `0.4509271407548402 -> 0.44760255377421837`.
- Issues:
  - The hint helped the D256 causal case and slightly helped D128 non-causal, but it slowed D256 non-causal and D64
    long non-causal enough to reduce the aggregate evaluator score.
  - The current scheduler already overlaps the early K/V loads well enough in the wide-lazy-GM paths; explicit
    multibuffer likely changes live ranges or buffering decisions in a way that hurts the GM-accumulator families.
  - Keep the current plain `tl.load(k_block_ptr)` / `tl.load(v_block_ptr)` source shape unless fresh IR shows a specific
    family where the hint reduces wait/barrier time without perturbing the long non-causal schedules.
- Reports:
  - `evaluation_reports/codex_point11_kv_multibuffer_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point11_kv_multibuffer_performance/evaluation_report.json`

## 2026-07-27 - Optimization point 5/6: final normalization reciprocal regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 5/6, scalar-to-vector / avoid scalar lowering, guided by IR evidence that several kernel families
  lower the final normalization to `vdiv_2d_float`.
- Motivation:
  - Current IR dumps for the generic, hz-major, and parity families show the final output normalization source lines
    lowering to two-dimensional vector division:
    `accumulator = acc_ptr / l_i[:, None]` or `accumulator = accumulator / l_i[:, None]`.
  - Profiles remain scalar/MTE/sync heavy instead of cleanly MMAD-bound, so reducing the final divide width looked like a
    plausible low-risk vector-pipeline experiment.
  - Hypothesis: computing a one-dimensional inverse once (`inv_l_i = 1.0 / l_i`) and broadcasting it through a
    two-dimensional multiply could replace expensive 2D division with cheaper 1D division plus vector multiply.
- Content:
  - Temporarily changed all four final normalization sites:
    - Generic UB accumulator path in `_attn_fwd_tile()`.
    - Generic GM accumulator path in `_attn_fwd_tile()`.
    - `_attn_fwd_causal_diag_split_tile()`.
    - `_attn_fwd_causal_diag_split_parity_tile()`.
  - The tested form was:
    `inv_l_i = 1.0 / l_i` followed by `accumulator = accumulator * inv_l_i[:, None]`.
  - Kept tiling, D128 hz-major routing, D128 2048 CV bundle, D256 parity routing, non-causal D256 mask-index elision,
    no-LSE store/log elision, K/V load ordering, and `DEFAULT_PERSISTENT_PROGRAMS = 20` unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: the change kept fp32 division, did not add int64 arithmetic, integer comparisons, modulo, extra grid
    dimensions, interleaved task partitioning, mutable loop-index updates, `break`, or `continue`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point6_final_norm_recip_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point6_final_norm_recip_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `21.265195558395543 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.35441992597325905`.
  - Median speedup regressed from `0.367469250279431` to `0.34879281725977357`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2860793263628685 -> 0.2713970849439511`.
    - `(128,8,1024,256, causal=True)`: `0.2908101646927782 -> 0.27885819185240845`.
    - `(128,8,2048,128, causal=True)`: `0.3271592699844636 -> 0.3114226991640886`.
    - `(128,8,2048,256, causal=False)`: `0.4879122341826647 -> 0.45739854621089293`.
    - `(128,8,4096,128, causal=False)`: `0.40777923057439847 -> 0.38616293535545854`.
    - `(128,8,8192,64, causal=False)`: `0.4509271407548402 -> 0.42128009831275465`.
- Issues:
  - All six scored performance shapes slowed, so the backend likely already handles the broadcast denominator efficiently
    enough, or the explicit inverse introduces an extra live vector / scheduling dependency that hurts the final store
    pipeline.
  - Treat final normalization reciprocal materialization as rejected for the current kernel families unless future IR
    proves the multiply form removes `vdiv_2d_float` without increasing MTE/sync pressure.
  - The temporary source change was reverted.
- Reports:
  - `evaluation_reports/codex_point6_final_norm_recip_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point6_final_norm_recip_performance/evaluation_report.json`

## 2026-07-27 - Optimization point 11: family-gated K/V multibuffer regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 11, load instruction scheduling / CV pipeline hinting, narrowed with a kernel-family gate.
- Motivation:
  - The previous broad K/V `extension.multibuffer(k/v, 2)` probe regressed overall but showed mixed per-shape signals:
    D256 causal and D128 non-causal improved slightly, while D256 non-causal and D64 long non-causal regressed.
  - Hypothesis: moving the hint behind a `tl.constexpr` family gate could preserve the positive families while avoiding
    the known negative wide-GM families.
- Content:
  - Temporarily added a `KV_MULTIBUFFER` constexpr through `_attn_fwd_inner_loop()`, `_attn_fwd_inner()`,
    `_attn_fwd_tile()`, `_attn_fwd()`, and `_attn_fwd_causal_hz_major()`.
  - Added `extension.multibuffer(k, 2)` and `extension.multibuffer(v, 2)` immediately after the K/V tile loads only when
    `KV_MULTIBUFFER=True`.
  - Routed `KV_MULTIBUFFER=True` only for:
    - D256 causal parity split family, where the broad probe had improved the target case.
    - D128 non-causal wide-lazy-GM generic family (`HEAD_DIM=128`, `BM=128`, `BN=256`, lazy, GM accumulator).
  - Kept D128 causal hz-major, D128 2048 CV bundle, D256 non-causal mask-index elision, D64 long non-causal, tiling,
    no-LSE store/log elision, and `DEFAULT_PERSISTENT_PROGRAMS = 20` otherwise unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: the change did not add int64 arithmetic, integer compare/divide/modulo, extra grid dimensions,
    interleaved task partitioning, mutable loop-index updates, `break`, or `continue`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point11_family_kv_multibuffer_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point11_family_kv_multibuffer_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `22.46580558513983 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.37443009308566383`.
  - Median speedup regressed from `0.367469250279431` to `0.36708900726070165`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2860793263628685 -> 0.2873642562985733`.
    - `(128,8,1024,256, causal=True)`: `0.2908101646927782 -> 0.29008734300447553`.
    - `(128,8,2048,128, causal=True)`: `0.3271592699844636 -> 0.3271064679575822`.
    - `(128,8,2048,256, causal=False)`: `0.4879122341826647 -> 0.48616456755727516`.
    - `(128,8,4096,128, causal=False)`: `0.40777923057439847 -> 0.4070715465638211`.
    - `(128,8,8192,64, causal=False)`: `0.4509271407548402 -> 0.4487863771322557`.
- Issues:
  - The family gate avoided the large broad regression but still did not exceed the current best aggregate score.
  - The D256 causal target no longer reproduced the broad-run gain, and the D128 non-causal target also drifted slightly
    below best. This suggests the broad run's small per-shape positives were not robust enough to justify adding a new
    constexpr through the shared call stack.
  - Keep K/V loads plain in the current source. Future K/V scheduling work should require fresh simulator evidence for a
    single family before adding another compile-time knob.
- Reports:
  - `evaluation_reports/codex_point11_family_kv_multibuffer_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point11_family_kv_multibuffer_performance/evaluation_report.json`

## 2026-07-27 - Optimization point 18: D256 parity lazy inner-loop regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 18, kernel family specialization / kernel splitting, narrowed to the existing D256 causal
  parity-split family.
- Motivation:
  - The D256 causal parity split route is already a shape-specialized positive family for
    `(Z=128,H=8,N_CTX=1024,HEAD_DIM=256,causal=True)`.
  - That route still called the shared `_attn_fwd_inner_loop()` with fixed lazy-softmax parameters
    (`ACC_IN_UB=True`, `USE_MAX=False`, non-fp8), so the backend still had to lower a generic helper carrying
    unused max-rescale and optional mask-index paths.
  - Hypothesis: replacing the parity tile's three inner segments with a dedicated D256 lazy UB helper would simplify
    lowering, reduce dead control/data paths, and improve the target D256 causal scored case without affecting other
    families.
- Content:
  - Temporarily added `_attn_fwd_inner_loop_d256_lazy_ub()`, a fixed lazy-softmax inner loop that:
    - Keeps `acc_ptr` resident in UB and updates it with `tl.dot(p_cast, v, acc_ptr)`.
    - Drops `m_i` threading, `USE_MAX`, `ACC_IN_UB`, fp8, and unused-mask-index parameters.
    - Uses `p = tl.math.exp(qk - 6.0)` and `l_i += tl.sum(p, axis=1)`, matching the existing lazy path.
    - Computes K/V block pointers from `start_n` each iteration to avoid adding a new mutable pointer-update chain.
  - Routed only `_attn_fwd_causal_diag_split_parity_tile()` through the dedicated helper for the off-band,
    parity pre-diagonal, and masked diagonal segments.
  - Kept the non-parity D256 diag-split route on the shared helper.
  - Restricted the parity route to `return_lse=False` during the experiment so the lazy helper would not corrupt
    LSE semantics by leaving `m_i` at `-inf`.
  - Kept tiling, D128 hz-major routing, D128 2048 CV bundle, D256 non-causal mask-index elision, D64 long non-causal,
    no-LSE store/log elision, K/V load ordering, and `DEFAULT_PERSISTENT_PROGRAMS = 20` otherwise unchanged.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: the new helper used fp32 comparison for the causal mask, did not add int64 arithmetic, modulo,
    multi-dimensional grid, interleaved task partitioning, `break`, or `continue`, and avoided a new mutable loop-index
    pointer update by deriving K/V block pointers from `start_n`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point18_d256_parity_lazy_inner_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point18_d256_parity_lazy_inner_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `21.440576856380076 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.35734294760633456`.
  - Median speedup regressed from `0.367469250279431` to `0.3493109701105135`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2860793263628685 -> 0.27379257921268113`.
    - `(128,8,1024,256, causal=True)`: `0.2908101646927782 -> 0.27869473563428643`.
    - `(128,8,2048,128, causal=True)`: `0.3271592699844636 -> 0.31128821371259396`.
    - `(128,8,2048,256, causal=False)`: `0.4879122341826647 -> 0.4687304417567062`.
    - `(128,8,4096,128, causal=False)`: `0.40777923057439847 -> 0.38733372650843295`.
    - `(128,8,8192,64, causal=False)`: `0.4509271407548402 -> 0.42421798881330675`.
- Issues:
  - All six scored shapes slowed, even though only the D256 causal parity source path changed. The shared source
    perturbation and new helper likely changed compilation/cache/lowering behavior enough to worsen aggregate timing,
    while the target D256 causal case also failed to improve.
  - The generic helper's extra constexpr parameters are apparently not the current bottleneck; removing them did not
    reduce the observed vector/sync pressure and may have hurt scheduling or code layout.
  - Treat D256 parity "clone the inner loop and delete generic parameters" as rejected. Future parity work should use
    fresh simulator/IR evidence for a specific hot line rather than further source-shape simplification.
  - The temporary source change was reverted.
- Reports:
  - `evaluation_reports/codex_point18_d256_parity_lazy_inner_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point18_d256_parity_lazy_inner_performance/evaluation_report.json`

## 2026-07-27 - Optimization point 25: D128 2048 VF membar-removal regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 25, IR/profile-guided compile-parameter specialization, tested narrowly on the existing D128
  causal `N_CTX=2048` hz-major full-CV branch.
- Motivation:
  - Current IR for `_attn_fwd_causal_hz_major` still shows high sync/barrier density:
    `sync_block_set=24`, `sync_block_wait=24`, `wait_flag=50`, `set_flag=50`, `pipe_barrier=19`.
  - The same branch already uses the compatible full CV bundle
    (`multibuffer=True`, `enable_mixed_cv=True`, `enable_auto_bind_sub_block=True`, `sync_solver=True`,
    `limit_auto_multi_buffer_of_local_buffer="no-limit"`, `set_workspace_multibuffer=2`), but does not use the
    supported auxiliary `NPUOptions` parameter `enable_cce_vf_remove_membar`.
  - Hypothesis: enabling only `enable_cce_vf_remove_membar=True` on the D128 2048 branch might remove redundant VF
    memory barriers and reduce sync overhead without changing math, tiling, workspace allocation, persistent programs, or
    other kernel families.
- Content:
  - Temporarily added one launch option to the `use_causal_hz_major_family` `N_CTX=2048` branch:
    `enable_cce_vf_remove_membar=True`.
  - Left the `N_CTX=1024` D128 hz-major branch without CV parameters, kept the D256 parity split, generic fallback,
    no-LSE store/log elision, tiling presets, K/V load ordering, and `DEFAULT_PERSISTENT_PROGRAMS = 20` unchanged.
  - Confirmed from local `NPUOptions` that `enable_cce_vf_remove_membar` is a supported keyword. Also confirmed that
    `enable_loop_flatten` is not a supported Python launch option, so the earlier backend hint for
    `--enable-loop-flatten=False` was not used.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: this change did not add int64 arithmetic, comparisons, division/modulo, extra grid dimensions,
    interleaved task partitioning, mutable loop-index updates, `break`, or `continue`; it only changed one supported
    launch option.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point25_d128_2048_remove_membar_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point25_d128_2048_remove_membar_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `22.30340686918956 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.37172344781982597`.
  - Median speedup regressed from `0.367469250279431` to `0.36649244789956636`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2860793263628685 -> 0.28400386120233784`.
    - `(128,8,1024,256, causal=True)`: `0.2908101646927782 -> 0.2894978635963786`.
    - `(128,8,2048,128, causal=True)`: `0.3271592699844636 -> 0.3260023866856263`.
    - `(128,8,2048,256, causal=False)`: `0.4879122341826647 -> 0.47489801184443053`.
    - `(128,8,4096,128, causal=False)`: `0.40777923057439847 -> 0.40698250911350636`.
    - `(128,8,8192,64, causal=False)`: `0.4509271407548402 -> 0.4489560544766761`.
- Issues:
  - The target D128 2048 branch slowed slightly, so the extra barrier-removal hint does not improve the existing
    full-CV schedule.
  - The larger aggregate loss came from unrelated fallback shapes in the same full performance run, which again shows that
    compile-option perturbations can shift global lowering/cache behavior even when source routing is narrow.
  - Treat `enable_cce_vf_remove_membar=True` as rejected for the D128 2048 full-CV branch. Future sync work should require
    fresh IR after a source change, not additional standalone VF sync flags on this branch.
  - The temporary source change was reverted.
- Reports:
  - `evaluation_reports/codex_point25_d128_2048_remove_membar_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_d128_2048_remove_membar_performance/evaluation_report.json`

## 2026-07-27 - Optimization point 25: D128 2048 HIVM auto-CV-balance regression

- Commit: journal-only commit for this entry; source was reverted after performance validation.
- Optimization point: 25, IR/profile-guided compile-parameter specialization, tested narrowly on the existing D128
  causal `N_CTX=2048` hz-major full-CV branch.
- Motivation:
  - Current IR for `_attn_fwd_causal_hz_major` still reports a mixed Cube/Vector schedule with nontrivial sync density:
    `sync_block_set=24`, `sync_block_wait=24`, `wait_flag=50`, `set_flag=50`, `pipe_barrier=19`.
  - The compile-parameter guide lists `enable_hivm_auto_cv_balance` as an auxiliary option for automatic HIVM CV load
    balancing. It had not been tested on top of the existing positive full-CV bundle for the D128 2048 branch.
  - Hypothesis: enabling only `enable_hivm_auto_cv_balance=True` on that branch might improve Cube/Vector balance without
    changing math, tiling, workspace allocation, persistent program count, or other kernel families.
- Content:
  - Temporarily added one launch option to the `use_causal_hz_major_family` `N_CTX=2048` branch:
    `enable_hivm_auto_cv_balance=True`.
  - Kept the `N_CTX=1024` D128 hz-major branch no-param, retained the D256 parity split, generic fallback, no-LSE
    store/log elision, tiling presets, K/V load ordering, and `DEFAULT_PERSISTENT_PROGRAMS = 20`.
- Effect:
  - `python3 -m py_compile flash_attention_forward.py`: pass.
  - `git diff --check`: pass.
  - Checklist review: this change only added one supported launch option and did not introduce int64 arithmetic,
    comparisons, division/modulo, extra grid dimensions, interleaved task partitioning, mutable loop-index updates,
    `break`, or `continue`.
  - Correctness suite: `18/18` passed in
    `evaluation_reports/codex_point25_d128_2048_auto_cv_balance_correctness/evaluation_report.json`.
  - Performance suite: `6/6` matched in
    `evaluation_reports/codex_point25_d128_2048_auto_cv_balance_performance/evaluation_report.json`.
  - Submit-style performance score regressed from the active best `22.506673665520136 / 60` to
    `21.486031279005772 / 60`.
  - Mean speedup regressed from `0.37511122775866895` to `0.35810052131676284`.
  - Median speedup regressed from `0.367469250279431` to `0.3505495797247071`.
  - Per-shape speedups after the experiment:
    - `(128,8,1024,128, causal=True)`: `0.2860793263628685 -> 0.27404116511549004`.
    - `(128,8,1024,256, causal=True)`: `0.2908101646927782 -> 0.28029132626573844`.
    - `(128,8,2048,128, causal=True)`: `0.3271592699844636 -> 0.31360732652846185`.
    - `(128,8,2048,256, causal=False)`: `0.4879122341826647 -> 0.4670290962685317`.
    - `(128,8,4096,128, causal=False)`: `0.40777923057439847 -> 0.38749183292095235`.
    - `(128,8,8192,64, causal=False)`: `0.4509271407548402 -> 0.4261423808014026`.
- Issues:
  - The target D128 2048 branch itself slowed substantially, so HIVM auto CV balancing conflicts with the existing
    hand-selected full-CV parameter bundle for this kernel.
  - All six scored shapes slowed, indicating that this option shifts backend scheduling/lowering globally enough to hurt
    even families whose source dispatch was not intended to change.
  - Treat `enable_hivm_auto_cv_balance=True` as rejected for this FlashAttention forward implementation unless a future
    backend changes the pass behavior and fresh IR proves it removes a concrete critical-path imbalance.
  - The temporary source change was reverted.
- Reports:
  - `evaluation_reports/codex_point25_d128_2048_auto_cv_balance_correctness/evaluation_report.json`
  - `evaluation_reports/codex_point25_d128_2048_auto_cv_balance_performance/evaluation_report.json`
