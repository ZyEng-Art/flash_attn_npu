"""
Pure-Triton Flash Attention forward for Ascend NPU.

This module keeps the forward kernel, launch config selection, and lightweight
profiling helpers used by the workspace scripts.
"""

import argparse
import csv
import json
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
RESULT_DIR_NAME = "result_dir_blockptr_persistent"
TUNED_CONFIG_JSON = Path(__file__).resolve().with_name("flash_attention_forward_tuned.json")

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

DEFAULT_FORWARD_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 32,
    "persistent_blocks": 20,
    "num_stages": 2,
    "enable_hivm_auto_cv_balance": True,
    "tile_mix_vector_loop": 2,
    "tile_mix_cube_loop": 2,
    "enable_ubuf_saving": True,
}

BUILTIN_TUNED_CONFIGS = {
    (128, 8, 1024, 128, True): {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "persistent_blocks": 20,
        "num_stages": 1,
        "enable_hivm_auto_cv_balance": False,
        "tile_mix_vector_loop": 4,
        "tile_mix_cube_loop": 2,
        "enable_ubuf_saving": True,
    },
    (128, 8, 1024, 256, True): {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "persistent_blocks": 20,
        "num_stages": 1,
        "enable_hivm_auto_cv_balance": True,
        "tile_mix_vector_loop": 2,
        "tile_mix_cube_loop": 4,
        "enable_ubuf_saving": True,
    },
    (128, 8, 2048, 128, True): {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "persistent_blocks": 20,
        "num_stages": 2,
        "enable_hivm_auto_cv_balance": True,
        "tile_mix_vector_loop": 2,
        "tile_mix_cube_loop": 2,
        "enable_ubuf_saving": True,
    },
    (128, 8, 2048, 256, False): {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "persistent_blocks": 40,
        "num_stages": 2,
        "enable_hivm_auto_cv_balance": False,
        "tile_mix_vector_loop": 4,
        "tile_mix_cube_loop": 4,
        "enable_ubuf_saving": True,
    },
    (128, 8, 4096, 128, False): {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "persistent_blocks": 20,
        "num_stages": 2,
        "enable_hivm_auto_cv_balance": True,
        "tile_mix_vector_loop": 2,
        "tile_mix_cube_loop": 2,
        "enable_ubuf_saving": True,
    },
    (128, 8, 8192, 64, False): {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "persistent_blocks": 40,
        "num_stages": 1,
        "enable_hivm_auto_cv_balance": False,
        "tile_mix_vector_loop": 2,
        "tile_mix_cube_loop": 4,
        "enable_ubuf_saving": True,
    },
}

TILING_HINTS = {
    (1, 1, 64, 64, False): (64, 16),
    (1, 1, 64, 64, True): (64, 16),
    (1, 1, 128, 128, False): (32, 32),
    (1, 1, 128, 128, True): (32, 32),
    (1, 2, 1024, 64, False): (64, 32),
    (1, 2, 1024, 64, True): (64, 32),
    (4, 32, 64, 64, False): (64, 32),
    (4, 32, 64, 64, True): (64, 32),
    (4, 32, 128, 128, False): (32, 64),
    (4, 32, 256, 128, True): (64, 64),
    (4, 32, 512, 64, True): (64, 64),
    (4, 32, 1024, 64, False): (64, 64),
    (4, 32, 1024, 64, True): (64, 64),
    (4, 32, 1024, 128, False): (32, 32),
    (4, 32, 2048, 64, True): (64, 64),
    (4, 32, 2048, 128, False): (32, 32),
    (4, 32, 4096, 64, False): (64, 64),
    (128, 8, 1024, 64, False): (64, 64),
    (128, 8, 1024, 128, True): (32, 32),
    (128, 8, 1024, 256, True): (32, 32),
    (128, 8, 2048, 128, True): (64, 32),
    (128, 8, 2048, 256, False): (32, 32),
    (128, 8, 4096, 128, False): (64, 64),
    (128, 8, 8192, 64, False): (128, 32),
}


def _parse_shape_key(raw_key):
    z, h, n_ctx, head_dim, causal = raw_key.split(":")
    return int(z), int(h), int(n_ctx), int(head_dim), bool(int(causal))


def _load_tuned_configs():
    configs = dict(BUILTIN_TUNED_CONFIGS)
    if not TUNED_CONFIG_JSON.exists():
        return configs

    try:
        payload = json.loads(TUNED_CONFIG_JSON.read_text(encoding="utf-8"))
    except Exception:
        return configs

    for raw_key, config in payload.get("tuned_configs", {}).items():
        configs[_parse_shape_key(raw_key)] = dict(config)
    return configs


TUNED_CONFIGS = _load_tuned_configs()


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


def _pick_main_op(rows, preferred_op="_attn_fwd"):
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


def summarize_profile_output(root, preferred_op="_attn_fwd"):
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


def get_tiling(z, h, n_ctx, head_dim, causal):
    override_bm = os.environ.get("OVERRIDE_BM")
    override_bn = os.environ.get("OVERRIDE_BN")
    if override_bm and override_bn:
        return int(override_bm), int(override_bn)

    exact = TILING_HINTS.get((z, h, n_ctx, head_dim, causal))
    if exact is not None:
        return exact

    if causal:
        if head_dim >= 128:
            return (64, 32) if n_ctx >= 2048 else (32, 32)
        if n_ctx >= 8192:
            return 128, 32
        return 64, 32

    if head_dim >= 256:
        return 32, 32
    if head_dim >= 128:
        return (64, 32) if n_ctx >= 2048 else (32, 32)
    if n_ctx >= 8192:
        return 128, 32
    if n_ctx >= 2048:
        return 64, 64
    if n_ctx >= 1024:
        return (64, 64) if head_dim <= 64 else (64, 32)
    return (64, 32) if head_dim <= 64 else (32, 32)


def _resolve_launch_programs(z, h, n_ctx, block_m, persistent_blocks=None):
    total_tiles = triton.cdiv(n_ctx, block_m) * z * h
    max_programs = int(os.environ.get("MAX_LAUNCHED_PROGRAMS", "65535"))
    if max_programs < 1:
        raise ValueError("MAX_LAUNCHED_PROGRAMS must be >= 1")

    requested = os.environ.get("PERSISTENT_BLOCKS")
    if requested is not None:
        persistent_blocks = int(requested)
    elif persistent_blocks is None:
        persistent_blocks = int(os.environ.get("DEFAULT_LAUNCHED_PROGRAMS", "20"))

    if persistent_blocks < 1:
        raise ValueError("persistent block count must be >= 1")
    return min(total_tiles, persistent_blocks, max_programs)


def _resolve_forward_config(z, h, n_ctx, head_dim, causal):
    config = dict(DEFAULT_FORWARD_CONFIG)
    config["BLOCK_M"], config["BLOCK_N"] = get_tiling(z, h, n_ctx, head_dim, causal)

    if os.environ.get("OVERRIDE_BM") and os.environ.get("OVERRIDE_BN"):
        return config

    exact = TUNED_CONFIGS.get((z, h, n_ctx, head_dim, causal))
    if exact is not None:
        config.update(exact)
    return config


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
    row_mask_cmp,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    k_block_ptr = tl.advance(k_block_ptr, (lo, 0))
    v_block_ptr = tl.advance(v_block_ptr, (lo, 0))

    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
    block2d_acc = row * HEAD_DIM + col_head_dim
    offs_m_cmp = offs_m.to(tl.float32)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n
        curr_n_cmp = curr_n.to(tl.float32)

        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        qk = tl.where(curr_n_cmp[None, :] < N_CTX, qk, -1.0e6)
        qk = tl.where(row_mask_cmp[:, None], qk, -1.0e6)

        if STAGE == 2:
            causal_mask = offs_m_cmp[:, None] >= curr_n_cmp[None, :]
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
def _attn_fwd_manual(
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
        row_mask = offs_m < N_CTX
        row_mask_cmp = offs_m.to(tl.float32) < N_CTX

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32) + 1.0

        if HEAD_DIM < 256:
            acc_ptr = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        else:
            acc_offset = (
                ((off_z.to(tl.int64) * H + off_h.to(tl.int64)) * N_CTX + task_m_idx * BLOCK_M) * HEAD_DIM
            )
            acc_ptr = acc + acc_offset

        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

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
                row_mask_cmp,
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
                row_mask_cmp,
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
        tl.store(m_ptrs, m_i.to(tl.float32), mask=row_mask)
        tl.store(o_block_ptr, accumulator.to(Out.type.element_ty), boundary_check=(0, 1))


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
    _attn_fwd_manual(
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


def _launch_kernel(kernel, q, k, v, causal, sm_scale, config):
    _validate_inputs(q, k, v)

    z, h, n_ctx, head_dim = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((z, h, n_ctx), device=q.device, dtype=torch.float32)
    acc = (
        torch.empty((1,), dtype=torch.float32, device=q.device)
        if head_dim < 256
        else torch.zeros((z, h, n_ctx, head_dim), dtype=torch.float32, device=q.device)
    )

    grid = (_resolve_launch_programs(z, h, n_ctx, config["BLOCK_M"], config.get("persistent_blocks")), 1, 1)
    stage = 3 if causal else 1

    kernel[grid](
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
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        STAGE=stage,
        num_stages=config["num_stages"],
        enable_hivm_auto_cv_balance=config["enable_hivm_auto_cv_balance"],
        tile_mix_vector_loop=config["tile_mix_vector_loop"],
        tile_mix_cube_loop=config["tile_mix_cube_loop"],
        enable_ubuf_saving=config["enable_ubuf_saving"],
        debug=False,
    )
    return out, lse


def _launch_attention(q, k, v, causal, sm_scale, bm, bn):
    config = dict(DEFAULT_FORWARD_CONFIG)
    config["BLOCK_M"] = bm
    config["BLOCK_N"] = bn
    return _launch_kernel(_attn_fwd, q, k, v, causal, sm_scale, config)


def _launch_attention_forward(q, k, v, causal, sm_scale):
    config = _resolve_forward_config(q.shape[0], q.shape[1], q.shape[2], q.shape[3], causal)
    return _launch_kernel(_attn_fwd, q, k, v, causal, sm_scale, config)


def profile_once(z, h, n_ctx, head_dim, causal, dtype, q, k, v, sm_scale):
    config = _resolve_forward_config(z, h, n_ctx, head_dim, causal)
    launched_programs = _resolve_launch_programs(z, h, n_ctx, config["BLOCK_M"], config.get("persistent_blocks"))
    print("Triton implementation: block_ptr + persistent + offline tuned fixed config")
    print(
        "Triton launch config: "
        f"BM={config['BLOCK_M']}, BN={config['BLOCK_N']}, launched_programs={launched_programs}"
    )
    print(f"Selected forward config: {config}")
    torch.npu.synchronize()
    return attention(q, k, v, causal, sm_scale).to(dtype)


def profiling(z, h, n_ctx, head_dim, causal, dtype, q, k, v, sm_scale):
    config = _resolve_forward_config(z, h, n_ctx, head_dim, causal)
    launched_programs = _resolve_launch_programs(z, h, n_ctx, config["BLOCK_M"], config.get("persistent_blocks"))
    print("Triton implementation: block_ptr + persistent + offline tuned fixed config")
    print(
        "Triton launch config: "
        f"BM={config['BLOCK_M']}, BN={config['BLOCK_N']}, launched_programs={launched_programs}"
    )
    print(f"Selected forward config: {config}")

    result_dir = Path.cwd() / RESULT_DIR_NAME
    if result_dir.exists():
        shutil.rmtree(result_dir)

    active = 30
    total_steps = 1 + 1 + 1 + active
    metric_plan = _build_metric_plan()
    if not metric_plan:
        raise RuntimeError("No available AiCMetrics found in torch_npu.profiler.AiCMetrics.")

    summary_by_metric = {}
    for metric_name, metric_value in metric_plan:
        metric_result_dir = result_dir / metric_name
        print(f"Profiling metric: {metric_name}")

        with torch_npu.profiler.profile(
            activities=_build_profiler_activities(),
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=active, repeat=1, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(metric_result_dir / "triton")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=False,
            experimental_config=_make_experimental_config(metric_value),
        ) as prof:
            for _ in range(total_steps):
                attention(q, k, v, causal, sm_scale).to(dtype)
                torch.npu.synchronize()
                prof.step()

        summary_by_metric[metric_name] = summarize_profile_output(metric_result_dir, preferred_op="_attn_fwd")

    summary_json, summary_txt = write_profile_summary_files(result_dir, summary_by_metric)
    print("Profiling complete. Results saved to:", result_dir)
    print("Machine-readable summary:", summary_json)
    print("Text summary:", summary_txt)
    print_profile_summary(summary_by_metric)
    return summary_by_metric


def attention_persistent(q, k, v, causal, sm_scale, bm, bn, return_lse=False):
    out, lse = _launch_attention(q, k, v, causal, sm_scale, bm, bn)
    if return_lse:
        return out, lse
    return out


def attention(q, k, v, causal, sm_scale, BM=None, BN=None, return_lse=False):
    if BM is None or BN is None:
        out, lse = _launch_attention_forward(q, k, v, causal, sm_scale)
    else:
        out, lse = _launch_attention(q, k, v, causal, sm_scale, BM, BN)
    if return_lse:
        return out, lse
    return out


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
    if getattr(args, "override_bm", None) is not None:
        os.environ["OVERRIDE_BM"] = str(args.override_bm)
    if getattr(args, "override_bn", None) is not None:
        os.environ["OVERRIDE_BN"] = str(args.override_bn)
    if getattr(args, "persistent_blocks", None) is not None:
        os.environ["PERSISTENT_BLOCKS"] = str(args.persistent_blocks)


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


def _run_torch_profile(args):
    dtype = _dtype_from_name(args.dtype)
    _apply_runtime_overrides(args)
    q, k, v = _make_inputs(args.z, args.h, args.n_ctx, args.head_dim, dtype)
    profiling(args.z, args.h, args.n_ctx, args.head_dim, args.causal, dtype, q, k, v, args.sm_scale)


def _build_cli():
    parser = argparse.ArgumentParser(description="Flash attention forward kernel and profiling entrypoints.")
    subparsers = parser.add_subparsers(dest="command")

    def add_shape_args(subparser):
        subparser.add_argument("--z", type=int, required=True)
        subparser.add_argument("--h", type=int, required=True)
        subparser.add_argument("--n-ctx", type=int, required=True, dest="n_ctx")
        subparser.add_argument("--head-dim", type=int, required=True, dest="head_dim")
        subparser.add_argument("--causal", action="store_true")
        subparser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16"])
        subparser.add_argument("--sm-scale", type=float, default=0.5, dest="sm_scale")
        subparser.add_argument("--profile-metrics")
        subparser.add_argument("--override-bm", type=int)
        subparser.add_argument("--override-bn", type=int)
        subparser.add_argument("--persistent-blocks", type=int)

    bench = subparsers.add_parser("bench")
    add_shape_args(bench)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--iters", type=int, default=20)
    bench.set_defaults(handler=_run_bench)

    torch_profile = subparsers.add_parser("torch-profile")
    add_shape_args(torch_profile)
    torch_profile.set_defaults(handler=_run_torch_profile)
    return parser


def _run_default_profile():
    z, h, n_ctx, head_dim = 128, 8, 8192, 64
    causal, dtype = False, torch.float16
    q, k, v = _make_inputs(z, h, n_ctx, head_dim, dtype)
    profiling(z, h, n_ctx, head_dim, causal, dtype, q, k, v, sm_scale=0.5)


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
    "DEFAULT_FORWARD_CONFIG",
    "TUNED_CONFIGS",
    "attention",
    "attention_persistent",
    "get_tiling",
    "profile_once",
    "profiling",
    "summarize_profile_output",
    "write_profile_summary_files",
    "_attn_fwd",
    "_attn_fwd_manual",
]


if __name__ == "__main__":
    main()
