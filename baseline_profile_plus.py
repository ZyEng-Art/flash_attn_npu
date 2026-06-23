import csv
import json
import os
import shutil
from pathlib import Path

import torch
import torch_npu


DEVICE = "npu"

SUMMARY_KERNEL_CORE_COLUMNS = [
    "Duration(us)",
    "Wait Time(us)",
    "aicore_time(us)",
    "aic_total_cycles",
    "aiv_time(us)",
    "aiv_total_cycles",
    "Block Num",
    "Mix Block Num",
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


def _pick_main_op(rows, preferred_op=None):
    if not rows:
        return None
    if preferred_op:
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
    summary = {}
    for col in SUMMARY_STEP_COLUMNS:
        summary[col] = _mean([_to_float(row.get(col)) for row in rows])
    return summary


def _summarize_api(rows):
    if not rows:
        return {}

    launch_rows = [row for row in rows if row.get("API Name") == "launch"]
    sync_rows = [row for row in rows if row.get("API Name") == "aclrtSynchronizeDeviceWithTimeout"]
    summary = {}

    if launch_rows:
        launch = launch_rows[0]
        summary["launch"] = {col: _to_float(launch.get(col)) for col in SUMMARY_API_COLUMNS}

    if sync_rows:
        sync = sync_rows[0]
        summary["aclrtSynchronizeDeviceWithTimeout"] = {
            col: _to_float(sync.get(col)) for col in SUMMARY_API_COLUMNS
        }

    return summary


def summarize_profile_output(root, preferred_op=None):
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
                "kernel: duration_us={duration}, wait_us={wait}, cube_pct={cube}, "
                "aic_mac_ratio={mac}, aic_scalar_ratio={scalar}, "
                "aic_mte1_ratio={mte1}, aic_mte2_ratio={mte2}, "
                "aic_fixpipe_ratio={fixpipe}, aiv_vec_ratio={aiv_vec}"
            ).format(
                duration=_fmt(kernel.get("Duration(us)")),
                wait=_fmt(kernel.get("Wait Time(us)")),
                cube=_fmt(kernel.get("cube_utilization(%)")),
                mac=_fmt(kernel.get("aic_mac_ratio")),
                scalar=_fmt(kernel.get("aic_scalar_ratio")),
                mte1=_fmt(kernel.get("aic_mte1_ratio")),
                mte2=_fmt(kernel.get("aic_mte2_ratio")),
                fixpipe=_fmt(kernel.get("aic_fixpipe_ratio")),
                aiv_vec=_fmt(kernel.get("aiv_vec_ratio")),
            )
        )
        extra_kernel_columns = [col for col in kernel_columns if col not in SUMMARY_KERNEL_CORE_COLUMNS]
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
    print("\nBaseline profile summary")
    print("------------------------")
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
            "  kernel duration/wait us: "
            f"{_fmt(kernel.get('Duration(us)'))} / {_fmt(kernel.get('Wait Time(us)'))}"
        )
        extra_kernel_columns = [col for col in kernel_columns if col not in SUMMARY_KERNEL_CORE_COLUMNS]
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

        step = summary.get("step", {})
        print(
            "  step computing/free/bubble/preparing us: "
            f"{_fmt(step.get('Computing'))} / {_fmt(step.get('Free'))} / "
            f"{_fmt(step.get('Bubble'))} / {_fmt(step.get('Preparing'))}"
        )

        api = summary.get("api", {})
        launch = api.get("launch", {})
        sync = api.get("aclrtSynchronizeDeviceWithTimeout", {})
        print(
            "  api launch/sync avg us: "
            f"{_fmt(launch.get('Avg(us)'))} / {_fmt(sync.get('Avg(us)'))}"
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
    selected = []

    for name in requested:
        if name in available:
            selected.append((name, available[name]))

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


def build_attention_mask(n_ctx, causal, device):
    if not causal:
        return None, 0
    mask = torch.triu(torch.ones(n_ctx, n_ctx, device=device), diagonal=1).bool()
    return mask, 2


def profile_baseline(Z, H, N_CTX, HEAD_DIM, causal, dtype, sm_scale=0.5, result_root=None):
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

    atten_mask, sparse_mode = build_attention_mask(N_CTX, causal, q.device)

    if result_root is None:
        result_root = Path(os.getcwd()) / "baseline_result_dir_plus"
    else:
        result_root = Path(result_root)
    if result_root.exists():
        shutil.rmtree(result_root)

    active = 30
    total_steps = 1 + 1 + 1 + active
    activities = _build_profiler_activities()
    metric_plan = _build_metric_plan()
    if not metric_plan:
        raise RuntimeError("No available AiCMetrics found in torch_npu.profiler.AiCMetrics.")

    summary_by_metric = {}
    for metric_name, metric_value in metric_plan:
        metric_result_dir = result_root / metric_name
        experimental_config = _make_experimental_config(metric_value)
        print(f"Profiling baseline metric: {metric_name}")

        with torch_npu.profiler.profile(
            activities=activities,
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=active, repeat=1, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(metric_result_dir / "baseline")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(total_steps):
                torch_npu.npu_fusion_attention(
                    q,
                    k,
                    v,
                    H,
                    padding_mask=None,
                    atten_mask=atten_mask,
                    scale=sm_scale,
                    keep_prob=1.0,
                    input_layout="BNSD",
                    pre_tockens=65535,
                    next_tockens=65535,
                    sparse_mode=sparse_mode,
                )[0]
                torch.npu.synchronize()
                prof.step()

        summary_by_metric[metric_name] = summarize_profile_output(metric_result_dir)

    summary_json, summary_txt = write_profile_summary_files(result_root, summary_by_metric)

    print("Baseline profiling complete.")
    print("Results saved to:", result_root)
    print("Machine-readable summary:", summary_json)
    print("Text summary:", summary_txt)
    print_profile_summary(summary_by_metric)
    print("Inspect each metric subdirectory under baseline_result_dir_plus/*/baseline for full profiler output.")


if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM = 128, 8, 8192, 64
    causal = False
    dtype = torch.float16
    profile_baseline(Z, H, N_CTX, HEAD_DIM, causal, dtype, sm_scale=0.5)
