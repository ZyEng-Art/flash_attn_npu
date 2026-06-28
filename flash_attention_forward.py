"""
Simple single-path Flash Attention forward for Ascend NPU.

This module uses one persistent Triton kernel with offline-tuned tiling presets
and detailed torch_npu profiling summaries.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.backends.ascend.runtime  # noqa: F401
import triton.language as tl
import triton.language.extra.cann.extension as extension


DEVICE = "npu"
RESULT_DIR_NAME = "result_dir"
DEFAULT_PERSISTENT_PROGRAMS = 20
PROFILE_OP_NAME = "_attn_fwd"

SUMMARY_KERNEL_CORE_COLUMNS = [
    "Duration(us)",
    "Wait Time(us)",
    "Block Num",
    "Mix Block Num",
    "aicore_time(us)",
    "aic_total_cycles",
    "aiv_time(us)",
    "aiv_total_cycles",
]

SUMMARY_KERNEL_PREFERRED_COLUMNS = SUMMARY_KERNEL_CORE_COLUMNS + [
    "aic_mac_ratio",
    "aic_scalar_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_fixpipe_ratio",
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
    "cube_utilization(%)",
    "aic_mac_fp16_ratio",
    "aic_mac_int8_ratio",
    "aic_cube_fops",
    "aiv_vec_fp32_ratio",
    "aiv_vec_fp16_ratio",
    "aiv_vec_int32_ratio",
    "aiv_vec_misc_ratio",
    "aiv_vector_fops",
    "aic_l1_read_bw(GB/s)",
    "aic_l1_write_bw(GB/s)",
    "aic_main_mem_read_bw(GB/s)",
    "aic_main_mem_write_bw(GB/s)",
    "aic_l2_read_bw(GB/s)",
    "aic_l2_write_bw(GB/s)",
    "aiv_ub_read_bw(GB/s)",
    "aiv_ub_write_bw(GB/s)",
    "aiv_main_mem_read_bw(GB/s)",
    "aiv_main_mem_write_bw(GB/s)",
    "aiv_l2_read_bw(GB/s)",
    "aiv_l2_write_bw(GB/s)",
    "aic_l0a_read_bw(GB/s)",
    "aic_l0a_write_bw(GB/s)",
    "aic_l0b_read_bw(GB/s)",
    "aic_l0b_write_bw(GB/s)",
    "aic_l0c_read_bw_cube(GB/s)",
    "aic_l0c_write_bw_cube(GB/s)",
    "aiv_l0c_read_bw(GB/s)",
    "aiv_l0c_write_bw(GB/s)",
    "aiv_vec_bankgroup_cflt_ratio",
    "aiv_vec_bank_cflt_ratio",
    "aiv_vec_resc_cflt_ratio",
    "aic_write_cache_hit",
    "aic_write_cache_miss_allocate",
    "aic_r0_read_cache_hit",
    "aic_r0_read_cache_miss_allocate",
    "aic_r1_read_cache_hit",
    "aic_r1_read_cache_miss_allocate",
    "aiv_write_cache_hit",
    "aiv_write_cache_miss_allocate",
    "aiv_r0_read_cache_hit",
    "aiv_r0_read_cache_miss_allocate",
    "aiv_r1_read_cache_hit",
    "aiv_r1_read_cache_miss_allocate",
]

SUMMARY_KERNEL_EXCLUDED_COLUMNS = {
    "Step Id",
    "Device_id",
    "Model ID",
    "Task ID",
    "Stream ID",
    "Name",
    "Type",
    "OP State",
    "Accelerator Core",
    "Start Time(us)",
    "HF32 Eligible",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "Context ID",
}

SUMMARY_L2_PREFERRED_COLUMNS = [
    "Hit Rate",
    "Victim Rate",
]

SUMMARY_L2_EXCLUDED_COLUMNS = {
    "Device_id",
    "Stream Id",
    "Task Id",
    "Op Name",
}

SUMMARY_STEP_COLUMNS = [
    "Computing",
    "Communication(Not Overlapped)",
    "Overlapped",
    "Communication",
    "Free",
    "Stage",
    "Bubble",
    "Communication(Not Overlapped and Exclude Receive)",
    "Preparing",
]

SUMMARY_API_COLUMNS = [
    "Time(us)",
    "Count",
    "Avg(us)",
    "Min(us)",
    "Max(us)",
]

DEFAULT_AIC_METRIC_CANDIDATES = [
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "ResourceConflictRatio",
    "L2Cache",
]

# BEGIN AUTO-TUNED DEFAULT TILING PRESETS
# Default tiling presets captured from offline tuning.
# This block is rewritten by `tune_flash_attention_forward.py`.
DEFAULT_TILING_PRESETS = {
    (128, 8, 1024, 128, True): (128, 64),
    (128, 8, 1024, 256, True): (64, 64),
    (128, 8, 2048, 128, True): (128, 64),
    (128, 8, 2048, 256, False): (64, 128),
    (128, 8, 4096, 128, False): (128, 256),
    (128, 8, 8192, 64, False): (128, 256),
}
# END AUTO-TUNED DEFAULT TILING PRESETS


def _default_tiling(z, h, n_ctx, head_dim, causal):
    preset = DEFAULT_TILING_PRESETS.get((z, h, n_ctx, head_dim, causal))
    if preset is not None:
        return preset

    if causal:
        if head_dim >= 128:
            return (64, 32) if n_ctx >= 2048 else (32, 32)
        if n_ctx >= 8192:
            return 128, 32
        return 64, 32

    # Non-causal is Vector/softmax-bound on Ascend; a large BLOCK_N consistently
    # wins (measured 5-16x vs the old small-block heuristic). Size BN against UB:
    # head_dim<=128 -> BN=256, head_dim>=256 -> BN=128 (qk fp32 + acc must fit 192KB).
    if head_dim >= 256:
        block_m, block_n = 64, 128
    else:
        block_m, block_n = 128, 256
    return min(block_m, n_ctx), min(block_n, n_ctx)


def get_tiling(z, h, n_ctx, head_dim, causal):
    return _default_tiling(z, h, n_ctx, head_dim, causal)


def get_tiling_source(z, h, n_ctx, head_dim, causal):
    if (z, h, n_ctx, head_dim, causal) in DEFAULT_TILING_PRESETS:
        return "preset"
    return "fallback"


def _to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value, precision=3):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.{precision}f}"


def _read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _ordered_columns(rows):
    ordered = []
    seen = set()
    for row in rows:
        for col in row.keys():
            if col not in seen:
                ordered.append(col)
                seen.add(col)
    return ordered


def _prioritize_columns(columns, preferred):
    prioritized = [col for col in preferred if col in columns]
    prioritized.extend(col for col in columns if col not in prioritized)
    return prioritized


def _find_profiler_output(root):
    root = Path(root).resolve()
    if (root / "op_statistic.csv").exists():
        return root

    candidates = sorted(root.rglob("ASCEND_PROFILER_OUTPUT/op_statistic.csv"))
    if candidates:
        return candidates[-1].parent

    candidates = sorted(root.rglob("op_statistic.csv"))
    if candidates:
        return candidates[-1].parent

    raise FileNotFoundError(f"cannot find op_statistic.csv under {root}")


def _pick_main_op(rows, preferred_op=PROFILE_OP_NAME):
    if not rows:
        return None
    matches = [row for row in rows if row.get("OP Type") == preferred_op]
    if matches:
        return matches[0]
    return max(rows, key=lambda row: _to_float(row.get("Total Time(us)")) or 0.0)


def _summarize_kernel_details(rows, op_name):
    if op_name:
        rows = [row for row in rows if row.get("Name") == op_name or row.get("Type") == op_name]
    if not rows:
        return {}, []

    summary = {"kernel_count": len(rows)}
    numeric_cols = []
    for col in _ordered_columns(rows):
        if col in SUMMARY_KERNEL_EXCLUDED_COLUMNS:
            continue
        values = [_to_float(row.get(col)) for row in rows]
        if not any(value is not None for value in values):
            continue
        summary[col] = _mean(values)
        numeric_cols.append(col)
    numeric_cols = _prioritize_columns(numeric_cols, SUMMARY_KERNEL_PREFERRED_COLUMNS)
    ordered_summary = {"kernel_count": summary["kernel_count"]}
    for col in numeric_cols:
        ordered_summary[col] = summary[col]
    return ordered_summary, numeric_cols


def _summarize_l2_cache(rows, op_name):
    if op_name:
        rows = [row for row in rows if row.get("Op Name") == op_name]
    if not rows:
        return {}, []

    summary = {"l2_count": len(rows)}
    l2_cols = []
    for col in SUMMARY_L2_PREFERRED_COLUMNS:
        if any(col in row for row in rows):
            summary[col] = _mean([_to_float(row.get(col)) for row in rows])
            l2_cols.append(col)
    for col in _ordered_columns(rows):
        if col in SUMMARY_L2_EXCLUDED_COLUMNS or col in l2_cols:
            continue
        values = [_to_float(row.get(col)) for row in rows]
        if not any(value is not None for value in values):
            continue
        summary[col] = _mean(values)
        l2_cols.append(col)
    return summary, l2_cols


def _summarize_step(rows):
    if not rows:
        return {}
    return {col: _mean([_to_float(row.get(col)) for row in rows]) for col in SUMMARY_STEP_COLUMNS}


def _summarize_api(rows):
    if not rows:
        return {}

    summary = {}
    for api_name in ("launch", "aclrtSynchronizeDeviceWithTimeout"):
        match = next((row for row in rows if row.get("API Name") == api_name), None)
        if match is not None:
            summary[api_name] = {col: _to_float(match.get(col)) for col in SUMMARY_API_COLUMNS}
    return summary


def summarize_profile_output(root, preferred_op=PROFILE_OP_NAME):
    out_dir = _find_profiler_output(root)
    op_rows = _read_csv_rows(out_dir / "op_statistic.csv")
    kernel_rows = _read_csv_rows(out_dir / "kernel_details.csv")
    step_rows = _read_csv_rows(out_dir / "step_trace_time.csv")
    api_rows = _read_csv_rows(out_dir / "api_statistic.csv")
    l2_rows = _read_csv_rows(out_dir / "l2_cache.csv")

    main_op = _pick_main_op(op_rows, preferred_op)
    op_name = main_op.get("OP Type") if main_op else preferred_op
    op_summary = {}
    if main_op:
        for key in [
            "OP Type",
            "Core Type",
            "Count",
            "Total Time(us)",
            "Min Time(us)",
            "Avg Time(us)",
            "Max Time(us)",
            "Ratio(%)",
        ]:
            value = main_op.get(key)
            op_summary[key] = _to_float(value) if "Type" not in key else value

    kernel_summary, kernel_columns = _summarize_kernel_details(kernel_rows, op_name)
    l2_summary, l2_columns = _summarize_l2_cache(l2_rows, op_name)
    return {
        "profiler_output": str(out_dir),
        "op_name": op_name,
        "op": op_summary,
        "kernel": kernel_summary,
        "kernel_columns": kernel_columns,
        "step": _summarize_step(step_rows),
        "api": _summarize_api(api_rows),
        "l2": l2_summary,
        "l2_columns": l2_columns,
    }


def write_profile_summary_files(root, summary_by_metric):
    root = Path(root)
    json_path = root / "profile_summary.json"
    text_path = root / "profile_summary.txt"

    import json

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_by_metric, f, indent=2, ensure_ascii=True)

    lines = []
    for metric_name, summary in summary_by_metric.items():
        lines.append(f"[{metric_name}]")
        lines.append(f"profiler_output={summary.get('profiler_output', 'n/a')}")

        op = summary.get("op", {})
        lines.append(
            "op: type={op_type}, count={count}, avg_us={avg}, min_us={minv}, max_us={maxv}, ratio_pct={ratio}".format(
                op_type=op.get("OP Type", "n/a"),
                count=_fmt(op.get("Count")),
                avg=_fmt(op.get("Avg Time(us)")),
                minv=_fmt(op.get("Min Time(us)")),
                maxv=_fmt(op.get("Max Time(us)")),
                ratio=_fmt(op.get("Ratio(%)")),
            )
        )

        kernel = summary.get("kernel", {})
        kernel_columns = summary.get("kernel_columns", [])
        lines.append(
            (
                "kernel: duration_us={duration}, wait_us={wait}, block_num={block_num}, "
                "mix_block_num={mix_block_num}, cube_pct={cube}, aic_mac_ratio={mac}, "
                "aic_scalar_ratio={scalar}, aiv_vec_ratio={aiv_vec}"
            ).format(
                duration=_fmt(kernel.get("Duration(us)")),
                wait=_fmt(kernel.get("Wait Time(us)")),
                block_num=_fmt(kernel.get("Block Num")),
                mix_block_num=_fmt(kernel.get("Mix Block Num")),
                cube=_fmt(kernel.get("cube_utilization(%)")),
                mac=_fmt(kernel.get("aic_mac_ratio")),
                scalar=_fmt(kernel.get("aic_scalar_ratio")),
                aiv_vec=_fmt(kernel.get("aiv_vec_ratio")),
            )
        )
        extra_kernel_columns = [
            col for col in kernel_columns
            if col not in SUMMARY_KERNEL_CORE_COLUMNS and col not in {"Block Num", "Mix Block Num"}
        ]
        if extra_kernel_columns:
            lines.append(
                "kernel_metrics: "
                + ", ".join(f"{col}={_fmt(kernel.get(col))}" for col in extra_kernel_columns)
            )

        step = summary.get("step", {})
        lines.append(
            "step: computing_us={computing}, free_us={free}, bubble_us={bubble}, preparing_us={preparing}".format(
                computing=_fmt(step.get("Computing")),
                free=_fmt(step.get("Free")),
                bubble=_fmt(step.get("Bubble")),
                preparing=_fmt(step.get("Preparing")),
            )
        )

        api = summary.get("api", {})
        launch = api.get("launch", {})
        sync = api.get("aclrtSynchronizeDeviceWithTimeout", {})
        lines.append(
            "api: launch_avg_us={launch_avg}, sync_avg_us={sync_avg}".format(
                launch_avg=_fmt(launch.get("Avg(us)")),
                sync_avg=_fmt(sync.get("Avg(us)")),
            )
        )
        l2 = summary.get("l2", {})
        l2_columns = summary.get("l2_columns", [])
        if l2_columns:
            lines.append(
                "l2_cache: "
                + ", ".join(f"{col}={_fmt(l2.get(col), 6)}" for col in l2_columns)
            )
        lines.append("")

    with text_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return json_path, text_path


def print_profile_summary(summary_by_metric):
    print("\nProfile summary")
    print("---------------")
    for metric_name, summary in summary_by_metric.items():
        print(f"[{metric_name}]")
        print(f"output: {summary.get('profiler_output', 'n/a')}")

        op = summary.get("op", {})
        print(
            "  op avg/min/max us: "
            f"{_fmt(op.get('Avg Time(us)'))} / {_fmt(op.get('Min Time(us)'))} / {_fmt(op.get('Max Time(us)'))}"
        )

        kernel = summary.get("kernel", {})
        kernel_columns = summary.get("kernel_columns", [])
        print(
            "  kernel duration/wait/block/mix: "
            f"{_fmt(kernel.get('Duration(us)'))} / {_fmt(kernel.get('Wait Time(us)'))} / "
            f"{_fmt(kernel.get('Block Num'))} / {_fmt(kernel.get('Mix Block Num'))}"
        )
        extra_kernel_columns = [
            col for col in kernel_columns
            if col not in SUMMARY_KERNEL_CORE_COLUMNS and col not in {"Block Num", "Mix Block Num"}
        ]
        if extra_kernel_columns:
            print(
                "  kernel metrics: "
                + ", ".join(f"{col}={_fmt(kernel.get(col))}" for col in extra_kernel_columns)
            )

        l2 = summary.get("l2", {})
        l2_columns = summary.get("l2_columns", [])
        if l2_columns:
            print(
                "  l2 cache: "
                + ", ".join(f"{col}={_fmt(l2.get(col), 6)}" for col in l2_columns)
            )


def _resolve_requested_metric_names():
    raw = os.environ.get("PROFILE_METRICS")
    if raw:
        return [name.strip() for name in raw.split(",") if name.strip()]
    return list(DEFAULT_AIC_METRIC_CANDIDATES)


def _resolve_available_metrics():
    metrics_enum = getattr(torch_npu.profiler, "AiCMetrics", None)
    if metrics_enum is None:
        return {}

    available = {}
    for attr in dir(metrics_enum):
        if attr.startswith("_"):
            continue
        value = getattr(metrics_enum, attr)
        if callable(value):
            continue
        available[attr] = value
    return available


def _build_metric_plan():
    available = _resolve_available_metrics()
    requested = _resolve_requested_metric_names()
    selected = [(name, available[name]) for name in requested if name in available]
    if not selected and "PipeUtilization" in available:
        selected.append(("PipeUtilization", available["PipeUtilization"]))
    if not selected and available:
        first_name = sorted(available)[0]
        selected.append((first_name, available[first_name]))
    return selected


def _build_profiler_activities():
    activities = [torch_npu.profiler.ProfilerActivity.NPU]
    cpu_activity = getattr(torch_npu.profiler.ProfilerActivity, "CPU", None)
    if cpu_activity is not None:
        activities.append(cpu_activity)
    return activities


def _make_experimental_config(aic_metric):
    return torch_npu.profiler._ExperimentalConfig(
        aic_metrics=aic_metric,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=True,
        data_simplification=False,
    )


def _get_persistent_programs():
    raw = os.environ.get("PERSISTENT_PROGRAMS")
    if raw is None:
        return DEFAULT_PERSISTENT_PROGRAMS
    value = int(raw)
    if value < 1:
        raise ValueError("PERSISTENT_PROGRAMS must be >= 1")
    return value


def _resolve_tiling(z, h, n_ctx, head_dim, causal, bm=None, bn=None):
    if (bm is None) != (bn is None):
        raise ValueError("BM and BN must be provided together")
    if bm is not None:
        return bm, bn, "override"
    return (*get_tiling(z, h, n_ctx, head_dim, causal), get_tiling_source(z, h, n_ctx, head_dim, causal))


def _describe_runtime(z, h, n_ctx, head_dim, causal, bm=None, bn=None):
    block_m, block_n, tiling_source = _resolve_tiling(z, h, n_ctx, head_dim, causal, bm, bn)
    total_tiles = triton.cdiv(n_ctx, block_m) * z * h
    launched_programs = min(total_tiles, _get_persistent_programs())
    return {
        "selected_config": {"BLOCK_M": block_m, "BLOCK_N": block_n},
        "tiling_source": tiling_source,
        "total_tiles": total_tiles,
        "launched_programs": launched_programs,
        "stage": 3 if causal else 1,
    }


@triton.jit
def _attn_fwd_inner_loop(
    acc_ptr,
    l_i,
    m_i,
    q,
    k_block_ptr,
    v_block_ptr,
    lo,
    hi,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    NEED_CAUSAL_MASK: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    k_block_ptr = tl.advance(k_block_ptr, (lo, 0))
    v_block_ptr = tl.advance(v_block_ptr, (lo, 0))

    if HEAD_DIM >= 256:
        row = tl.arange(0, BLOCK_M)[:, None]
        col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
        block2d_acc = row * HEAD_DIM + col_head_dim

    if NEED_CAUSAL_MASK:
        # Ascend compare paths are much more likely to stay vectorized with fp32 than int64/int32.
        offs_m_for_cmp = offs_m.to(tl.float32)

    
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n

        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        if NEED_CAUSAL_MASK:
            curr_n_for_cmp = curr_n.to(tl.float32)

        qk = tl.dot(q, tl.trans(k))
        # qk = qk * qk_scale

        if NEED_CAUSAL_MASK:
            causal_mask = offs_m_for_cmp[:, None] >= curr_n_for_cmp[None, :]
            qk = tl.where(causal_mask, qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        p_cast = p.to(tl.float8e5) if fp8_v else p.to(k.dtype)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        if HEAD_DIM < 256:
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)
        else:
            pv = tl.dot(p_cast, v)
            acc = tl.load(acc_ptr + block2d_acc)
            for slice_idx in range(4):
                offset = slice_idx * (BLOCK_M // 4)
                acc_i = extension.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                alpha_i = extension.extract_slice(alpha, [offset], [BLOCK_M // 4], [1])
                pv_i = extension.extract_slice(pv, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
                acc_i = acc_i * alpha_i[:, None] + pv_i
                acc = extension.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
            tl.store(acc_ptr + block2d_acc, acc)

        m_i = m_ij
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))

    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd_inner(
    acc_ptr,
    l_i,
    m_i,
    q,
    k_block_ptr,
    v_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        stage_lo = 0
        stage_hi = start_m * BLOCK_M
        full_hi = stage_hi - (stage_hi % BLOCK_N)
        acc_ptr, l_i, m_i = _attn_fwd_inner_loop(
            acc_ptr,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            stage_lo,
            full_hi,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
            False,
            N_CTX,
            fp8_v,
        )
        return acc_ptr, l_i, m_i

    if STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        stage_lo = tl.multiple_of(start_m * BLOCK_M, BLOCK_M)
        stage_hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        full_hi = stage_hi - ((stage_hi - stage_lo) % BLOCK_N)
        acc_ptr, l_i, m_i = _attn_fwd_inner_loop(
            acc_ptr,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            stage_lo,
            full_hi,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
            True,
            N_CTX,
            fp8_v,
        )
        return acc_ptr, l_i, m_i

    stage_lo = 0
    full_hi = N_CTX - (N_CTX % BLOCK_N)
    acc_ptr, l_i, m_i = _attn_fwd_inner_loop(
        acc_ptr,
        l_i,
        m_i,
        q,
        k_block_ptr,
        v_block_ptr,
        stage_lo,
        full_hi,
        qk_scale,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        offs_m,
        offs_n,
        False,
        N_CTX,
        fp8_v,
    )
    return acc_ptr, l_i, m_i


@triton.jit
def _attn_fwd_tile(
    Q,
    K,
    V,
    M,
    Out,
    acc,
    sm_scale,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    linear_tile,
):
    # Tile-to-core decomposition depends on STAGE (compile-time constant), trading off
    # load balance vs K/V cache reuse on the persistent (stride-P) grid:
    #   causal (STAGE==3): work grows with m_idx, so hz-major mapping leaves each core
    #     seeing only m_idx == pid (mod gcd(P, num_tiles_m)) -> stragglers. m-major
    #     (num_hz >> P) makes every core sample all m_idx evenly -> balanced.
    #   non-causal (STAGE==1): all tiles cost the same, so keep hz-major to hold
    #     consecutive same-(z,h) tiles -> K/V stay hot in L2 (m-major loses ~6%).
    num_tiles_m = tl.cdiv(N_CTX, BLOCK_M)
    if STAGE == 3:
        num_hz = Z * H
        task_m_idx = linear_tile // num_hz
        task_hz_idx = linear_tile - task_m_idx * num_hz
    else:
        task_hz_idx = linear_tile // num_tiles_m
        task_m_idx = linear_tile - task_hz_idx * num_tiles_m
    off_z = task_hz_idx // H
    off_h = task_hz_idx % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32) + 1.0
    if HEAD_DIM < 256:
        acc_ptr = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    else:
        acc_offset = (((off_z.to(tl.int64) * H + off_h.to(tl.int64)) * N_CTX + task_m_idx * BLOCK_M) * HEAD_DIM)
        acc_ptr = acc + acc_offset

    tl.static_assert(N_CTX % BLOCK_M == 0)
    q = tl.load(q_block_ptr)
    q = (q * sm_scale).to(q.dtype)
    if STAGE & 1:
        acc_ptr, l_i, m_i = _attn_fwd_inner(
            acc_ptr,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            task_m_idx,
            sm_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )
    if STAGE & 2:
        acc_ptr, l_i, m_i = _attn_fwd_inner(
            acc_ptr,
            l_i,
            m_i,
            q,
            k_block_ptr,
            v_block_ptr,
            task_m_idx,
            sm_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,
        )

    m_i += tl.math.log(l_i)
    if HEAD_DIM < 256:
        accumulator = acc_ptr / l_i[:, None]
    else:
        row = tl.arange(0, BLOCK_M)[:, None]
        col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
        accumulator = tl.load(acc_ptr + row * HEAD_DIM + col_head_dim)
        accumulator = accumulator / l_i[:, None]

    m_ptrs = M + task_hz_idx * N_CTX + offs_m
    tl.store(m_ptrs, m_i.to(tl.float32))
    tl.store(o_block_ptr, accumulator.to(Out.type.element_ty))


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    M,
    Out,
    acc,
    sm_scale,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    num_tiles_m = tl.cdiv(N_CTX, BLOCK_M)
    total_tiles = num_tiles_m * Z * H
    pid = tl.program_id(0)
    program_count = tl.num_programs(0)

    for linear_tile in tl.range(pid, total_tiles, program_count):
        _attn_fwd_tile(
            Q,
            K,
            V,
            M,
            Out,
            acc,
            sm_scale,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_oz,
            stride_oh,
            stride_om,
            stride_on,
            Z,
            H,
            N_CTX,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            STAGE,
            linear_tile,
        )


def _validate_inputs(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must all be 4D tensors shaped [Z, H, N_CTX, HEAD_DIM]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if q.shape[-1] not in {16, 32, 64, 128, 256}:
        raise ValueError("HEAD_DIM must be one of {16, 32, 64, 128, 256}")


def _build_grid(z, h, n_ctx, block_m):
    total_tiles = triton.cdiv(n_ctx, block_m) * z * h
    return (min(total_tiles, _get_persistent_programs()), 1, 1)


def _launch_kernel(q, k, v, causal, sm_scale, bm=None, bn=None):
    _validate_inputs(q, k, v)

    z, h, n_ctx, head_dim = q.shape
    bm, bn, _ = _resolve_tiling(z, h, n_ctx, head_dim, causal, bm, bn)
    out = torch.empty_like(q)
    lse = torch.empty((z, h, n_ctx), device=q.device, dtype=torch.float32)
    acc = (
        torch.empty((1,), dtype=torch.float32, device=q.device)
        if head_dim < 256
        else torch.zeros((z, h, n_ctx, head_dim), dtype=torch.float32, device=q.device)
    )
    stage = 3 if causal else 1
    grid = _build_grid(z, h, n_ctx, bm)

    _attn_fwd[grid](
        q,
        k,
        v,
        lse,
        out,
        acc,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        z,
        h,
        N_CTX=n_ctx,
        HEAD_DIM=head_dim,
        BLOCK_M=bm,
        BLOCK_N=bn,
        STAGE=stage,
        debug=False,
    )
    return out, lse


def attention(q, k, v, causal, sm_scale, BM=None, BN=None, return_lse=False):
    out, lse = _launch_kernel(q, k, v, causal, sm_scale, bm=BM, bn=BN)
    if return_lse:
        return out, lse
    return out


def profile_once(z, h, n_ctx, head_dim, causal, q, k, v, sm_scale):
    out = attention(q, k, v, causal, sm_scale)
    torch.npu.synchronize()
    runtime = _describe_runtime(z, h, n_ctx, head_dim, causal)
    print("Triton implementation: single persistent fixed-tiling kernel")
    print(f"Selected tiling config: {runtime['selected_config']} (source={runtime['tiling_source']})")
    print(
        "Launch summary: "
        f"stage={runtime['stage']}, total_tiles={runtime['total_tiles']}, "
        f"launched_programs={runtime['launched_programs']}"
    )
    return out


def profiling(z, h, n_ctx, head_dim, causal, *args, **kwargs):
    sm_scale_kw = kwargs.pop("sm_scale", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"profiling() got unexpected keyword argument(s): {unexpected}")

    if sm_scale_kw is None:
        if len(args) == 4:
            q, k, v, sm_scale = args
        elif len(args) == 5:
            _dtype, q, k, v, sm_scale = args
        else:
            raise TypeError(
                "profiling expects either (q, k, v, sm_scale) or "
                "(dtype, q, k, v, sm_scale) after causal"
            )
    else:
        if len(args) == 3:
            q, k, v = args
            sm_scale = sm_scale_kw
        elif len(args) == 4:
            _dtype, q, k, v = args
            sm_scale = sm_scale_kw
        else:
            raise TypeError(
                "profiling expects either (q, k, v, sm_scale=<value>) or "
                "(dtype, q, k, v, sm_scale=<value>) after causal"
            )

    attention(q, k, v, causal, sm_scale)
    torch.npu.synchronize()
    runtime = _describe_runtime(z, h, n_ctx, head_dim, causal)
    print("Triton implementation: single persistent fixed-tiling kernel")
    print(f"Selected tiling config: {runtime['selected_config']} (source={runtime['tiling_source']})")
    print(
        "Launch summary: "
        f"stage={runtime['stage']}, total_tiles={runtime['total_tiles']}, "
        f"launched_programs={runtime['launched_programs']}"
    )

    result_dir = Path.cwd() / RESULT_DIR_NAME
    if result_dir.exists():
        shutil.rmtree(result_dir)

    active = 30
    total_steps = 1 + 1 + 1 + active
    metric_plan = _build_metric_plan()
    if not metric_plan:
        raise RuntimeError("No available AiCMetrics found in torch_npu.profiler.AiCMetrics.")

    summary_by_metric = {}
    for metric_idx, (metric_name, metric_value) in enumerate(metric_plan):
        profile_root = result_dir if metric_idx == 0 else result_dir / metric_name
        print(f"Profiling metric: {metric_name}")

        with torch_npu.profiler.profile(
            activities=_build_profiler_activities(),
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=active, repeat=1, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_root / "triton")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=False,
            experimental_config=_make_experimental_config(metric_value),
        ) as prof:
            for _ in range(total_steps):
                attention(q, k, v, causal, sm_scale)
                torch.npu.synchronize()
                prof.step()

        summary_by_metric[metric_name] = summarize_profile_output(profile_root, preferred_op=PROFILE_OP_NAME)

    summary_json, summary_txt = write_profile_summary_files(result_dir, summary_by_metric)
    print("Profiling complete. Results saved to:", result_dir)
    print("Machine-readable summary:", summary_json)
    print("Text summary:", summary_txt)
    print_profile_summary(summary_by_metric)
    return summary_by_metric


def _dtype_from_name(name):
    normalized = name.strip().lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _make_inputs(z, h, n_ctx, head_dim, dtype):
    q = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    return q, k, v


def _apply_runtime_overrides(args):
    if getattr(args, "profile_metrics", None):
        os.environ["PROFILE_METRICS"] = args.profile_metrics
    if getattr(args, "persistent_programs", None) is not None:
        os.environ["PERSISTENT_PROGRAMS"] = str(args.persistent_programs)


def _run_tune(args):
    dtype = _dtype_from_name(args.dtype)
    _apply_runtime_overrides(args)
    q, k, v = _make_inputs(args.z, args.h, args.n_ctx, args.head_dim, dtype)
    profile_once(args.z, args.h, args.n_ctx, args.head_dim, args.causal, q, k, v, args.sm_scale)


def _run_bench(args):
    dtype = _dtype_from_name(args.dtype)
    _apply_runtime_overrides(args)
    q, k, v = _make_inputs(args.z, args.h, args.n_ctx, args.head_dim, dtype)

    for _ in range(args.warmup):
        attention(q, k, v, args.causal, args.sm_scale)
    torch.npu.synchronize()

    for _ in range(args.iters):
        attention(q, k, v, args.causal, args.sm_scale)
    torch.npu.synchronize()

    runtime = _describe_runtime(args.z, args.h, args.n_ctx, args.head_dim, args.causal)
    print(f"Selected tiling config: {runtime['selected_config']} (source={runtime['tiling_source']})")


def _run_torch_profile(args):
    dtype = _dtype_from_name(args.dtype)
    _apply_runtime_overrides(args)
    q, k, v = _make_inputs(args.z, args.h, args.n_ctx, args.head_dim, dtype)
    profiling(args.z, args.h, args.n_ctx, args.head_dim, args.causal, q, k, v, args.sm_scale)


def _build_cli():
    parser = argparse.ArgumentParser(description="Simple Flash Attention forward with offline-tuned tiling and profiling.")
    subparsers = parser.add_subparsers(dest="command")

    def add_common_args(subparser):
        subparser.add_argument("--z", type=int, required=True)
        subparser.add_argument("--h", type=int, required=True)
        subparser.add_argument("--n-ctx", type=int, required=True, dest="n_ctx")
        subparser.add_argument("--head-dim", type=int, required=True, dest="head_dim")
        subparser.add_argument("--causal", action="store_true")
        subparser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16"])
        subparser.add_argument("--sm-scale", type=float, default=0.5, dest="sm_scale")
        subparser.add_argument("--persistent-programs", type=int)

    tune = subparsers.add_parser("tune")
    add_common_args(tune)
    tune.set_defaults(handler=_run_tune)

    bench = subparsers.add_parser("bench")
    add_common_args(bench)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--iters", type=int, default=20)
    bench.set_defaults(handler=_run_bench)

    torch_profile = subparsers.add_parser("torch-profile")
    add_common_args(torch_profile)
    torch_profile.add_argument("--profile-metrics")
    torch_profile.set_defaults(handler=_run_torch_profile)
    return parser


def _run_default_profile():
    z, h, n_ctx, head_dim = 128, 8, 8192, 64
    causal, dtype = False, torch.float16
    q, k, v = _make_inputs(z, h, n_ctx, head_dim, dtype)
    profiling(z, h, n_ctx, head_dim, causal, q, k, v, sm_scale=0.5)


def main():
    parser = _build_cli()
    args = parser.parse_args()
    if getattr(args, "handler", None) is None:
        _run_default_profile()
        return
    args.handler(args)


__all__ = [
    "DEVICE",
    "RESULT_DIR_NAME",
    "PROFILE_OP_NAME",
    "get_tiling",
    "attention",
    "profile_once",
    "profiling",
    "summarize_profile_output",
    "write_profile_summary_files",
    "_attn_fwd",
]


if __name__ == "__main__":
    main()
