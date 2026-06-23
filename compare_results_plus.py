import argparse
import csv
import json
from pathlib import Path


KEY_KERNEL_CORE_COLUMNS = [
    "Duration(us)",
    "Wait Time(us)",
    "Block Num",
    "Mix Block Num",
    "aicore_time(us)",
    "aic_total_cycles",
    "aiv_time(us)",
    "aiv_total_cycles",
]

KEY_KERNEL_PREFERRED_COLUMNS = KEY_KERNEL_CORE_COLUMNS + [
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

KEY_KERNEL_EXCLUDED_COLUMNS = {
    "kernel_count",
}

KEY_L2_PREFERRED_COLUMNS = [
    "Hit Rate",
    "Victim Rate",
]

KEY_STEP_COLUMNS = [
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

KEY_API_NAMES = [
    "launch",
    "aclrtSynchronizeDeviceWithTimeout",
]

KEY_API_COLUMNS = [
    "Time(us)",
    "Count",
    "Avg(us)",
    "Min(us)",
    "Max(us)",
]


def find_profiler_output(root):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"path does not exist: {root}")

    if (root / "op_statistic.csv").exists():
        return root

    candidates = sorted(root.rglob("ASCEND_PROFILER_OUTPUT/op_statistic.csv"))
    if candidates:
        return candidates[-1].parent

    candidates = sorted(root.rglob("op_statistic.csv"))
    if candidates:
        return candidates[-1].parent

    raise FileNotFoundError(f"cannot find op_statistic.csv under {root}")


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def ordered_columns(rows):
    ordered = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def prioritize_columns(columns, preferred):
    prioritized = [col for col in preferred if col in columns]
    prioritized.extend(col for col in columns if col not in prioritized)
    return prioritized


def pick_main_op(rows, preferred_op=None):
    if not rows:
        return None
    if preferred_op:
        matches = [row for row in rows if row.get("OP Type") == preferred_op]
        if matches:
            return matches[0]
    return max(rows, key=lambda row: to_float(row.get("Total Time(us)")) or 0.0)


def summarize_kernel_details(rows, op_name=None):
    if op_name:
        rows = [row for row in rows if row.get("Name") == op_name or row.get("Type") == op_name]
    if not rows:
        return {}, []

    summary = {"kernel_count": len(rows)}
    numeric_cols = []
    excluded = {
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
    for col in ordered_columns(rows):
        if col in excluded:
            continue
        values = [to_float(row.get(col)) for row in rows]
        if not any(value is not None for value in values):
            continue
        summary[col] = mean(values)
        numeric_cols.append(col)
    numeric_cols = prioritize_columns(numeric_cols, KEY_KERNEL_PREFERRED_COLUMNS)
    ordered_summary = {"kernel_count": summary["kernel_count"]}
    for col in numeric_cols:
        ordered_summary[col] = summary[col]
    return ordered_summary, numeric_cols


def summarize_l2(rows, op_name=None):
    if op_name:
        rows = [row for row in rows if row.get("Op Name") == op_name]
    if not rows:
        return {}, []

    summary = {"l2_count": len(rows)}
    l2_cols = []
    for col in KEY_L2_PREFERRED_COLUMNS:
        if any(col in row for row in rows):
            summary[col] = mean([to_float(row.get(col)) for row in rows])
            l2_cols.append(col)
    for col in ordered_columns(rows):
        if col in {"Device_id", "Stream Id", "Task Id", "Op Name"} or col in l2_cols:
            continue
        values = [to_float(row.get(col)) for row in rows]
        if not any(value is not None for value in values):
            continue
        summary[col] = mean(values)
        l2_cols.append(col)
    return summary, l2_cols


def summarize_step(rows):
    if not rows:
        return {}
    summary = {}
    for col in KEY_STEP_COLUMNS:
        summary[col] = mean([to_float(row.get(col)) for row in rows])
    return summary


def summarize_api(rows):
    if not rows:
        return {}

    summary = {}
    for api_name in KEY_API_NAMES:
        matches = [row for row in rows if row.get("API Name") == api_name]
        if matches:
            row = matches[0]
            summary[api_name] = {col: to_float(row.get(col)) for col in KEY_API_COLUMNS}
    return summary


def load_raw_profile(root, preferred_op=None):
    out_dir = find_profiler_output(root)
    op_rows = read_csv_rows(out_dir / "op_statistic.csv")
    kernel_rows = read_csv_rows(out_dir / "kernel_details.csv")
    step_rows = read_csv_rows(out_dir / "step_trace_time.csv")
    api_rows = read_csv_rows(out_dir / "api_statistic.csv")
    l2_rows = read_csv_rows(out_dir / "l2_cache.csv")

    main_op = pick_main_op(op_rows, preferred_op)
    op_name = main_op.get("OP Type") if main_op else preferred_op

    op_summary = {}
    if main_op:
        op_summary = {
            "OP Type": main_op.get("OP Type"),
            "Core Type": main_op.get("Core Type"),
            "Count": to_float(main_op.get("Count")),
            "Total Time(us)": to_float(main_op.get("Total Time(us)")),
            "Min Time(us)": to_float(main_op.get("Min Time(us)")),
            "Avg Time(us)": to_float(main_op.get("Avg Time(us)")),
            "Max Time(us)": to_float(main_op.get("Max Time(us)")),
            "Ratio(%)": to_float(main_op.get("Ratio(%)")),
        }

    kernel_summary, kernel_columns = summarize_kernel_details(kernel_rows, op_name)
    l2_summary, l2_columns = summarize_l2(l2_rows, op_name)
    return {
        "__kind__": "single",
        "__source__": str(Path(root).expanduser()),
        "default": {
            "input_root": str(Path(root).expanduser()),
            "profiler_output": str(out_dir),
            "op_name": op_name,
            "op": op_summary,
            "kernel": kernel_summary,
            "kernel_columns": kernel_columns,
            "step": summarize_step(step_rows),
            "api": summarize_api(api_rows),
            "l2": l2_summary,
            "l2_columns": l2_columns,
        },
    }


def normalize_summary_entry(entry):
    kernel = entry.get("kernel", {})
    kernel_columns = entry.get("kernel_columns")
    if kernel_columns is None:
        kernel_columns = [key for key in kernel.keys() if key not in KEY_KERNEL_EXCLUDED_COLUMNS]
        kernel_columns = prioritize_columns(kernel_columns, KEY_KERNEL_PREFERRED_COLUMNS)

    l2 = entry.get("l2", {})
    l2_columns = entry.get("l2_columns")
    if l2_columns is None:
        l2_columns = [key for key in l2.keys() if key != "l2_count"]
        l2_columns = prioritize_columns(l2_columns, KEY_L2_PREFERRED_COLUMNS)

    return {
        "profiler_output": entry.get("profiler_output"),
        "op_name": entry.get("op_name"),
        "op": entry.get("op", {}),
        "kernel": kernel,
        "kernel_columns": kernel_columns,
        "step": entry.get("step", {}),
        "api": entry.get("api", {}),
        "l2": l2,
        "l2_columns": l2_columns,
    }


def load_summary_profile(path):
    path = Path(path).expanduser().resolve()
    data = read_json(path)
    normalized = {metric_name: normalize_summary_entry(entry) for metric_name, entry in data.items()}
    return {
        "__kind__": "summary",
        "__source__": str(path),
        **normalized,
    }


def load_profile(path, preferred_op=None):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")

    if path.is_file() and path.name == "profile_summary.json":
        return load_summary_profile(path)

    summary_json = path / "profile_summary.json"
    if summary_json.exists():
        return load_summary_profile(summary_json)

    return load_raw_profile(path, preferred_op)


def fmt(value, precision=3):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.{precision}f}"


def diff_values(a, b):
    if a is None or b is None:
        return None, None
    delta = b - a
    pct = None if a == 0 else delta / a * 100.0
    return delta, pct


def print_section(title):
    print()
    print(title)
    print("-" * len(title))


def print_comparison(label, a, b, lower_is_better=False):
    delta, pct = diff_values(a, b)
    if delta is None:
        print(f"{label:34} {fmt(a):>14} {fmt(b):>14} {'n/a':>14} {'n/a':>12}")
        return

    if lower_is_better:
        verdict = "better" if delta < 0 else "worse" if delta > 0 else "same"
    else:
        verdict = "higher" if delta > 0 else "lower" if delta < 0 else "same"

    print(f"{label:34} {fmt(a):>14} {fmt(b):>14} {fmt(delta):>14} {fmt(pct, 2) + '%':>12}  {verdict}")


def resolve_dynamic_columns(cols_a, cols_b, preferred):
    combined = []
    seen = set()
    for col in preferred:
        if col in cols_a or col in cols_b:
            combined.append(col)
            seen.add(col)
    for source in (cols_a, cols_b):
        for col in source:
            if col not in seen:
                combined.append(col)
                seen.add(col)
    return combined


def comparison_prefers_lower(key):
    if key in {"Duration(us)", "Wait Time(us)", "aicore_time(us)", "aiv_time(us)", "aic_total_cycles", "aiv_total_cycles"}:
        return True
    lower_markers = ("Time", "Wait", "Victim", "miss", "Miss", "cflt", "Conflict", "bank", "resc")
    return any(marker in key for marker in lower_markers)


def print_api_comparison(name_a, name_b, api_a, api_b):
    print_section("API Statistic")
    print(f"{'metric':34} {name_a:>14} {name_b:>14} {'delta':>14} {'delta %':>12}")
    for api_name in KEY_API_NAMES:
        for col in KEY_API_COLUMNS:
            label = f"{api_name}:{col}"
            lower = col != "Count"
            print_comparison(label, api_a.get(api_name, {}).get(col), api_b.get(api_name, {}).get(col), lower_is_better=lower)


def print_profile_pair(metric_name, name_a, entry_a, name_b, entry_b):
    print_section(f"Metric: {metric_name}")
    print(f"{name_a} output: {entry_a.get('profiler_output') or 'n/a'}")
    print(f"{name_b} output: {entry_b.get('profiler_output') or 'n/a'}")
    print(f"{name_a} op: {entry_a.get('op_name') or entry_a.get('op', {}).get('OP Type') or 'n/a'}")
    print(f"{name_b} op: {entry_b.get('op_name') or entry_b.get('op', {}).get('OP Type') or 'n/a'}")

    print_section("Op Statistic")
    print(f"{'metric':34} {name_a:>14} {name_b:>14} {'delta':>14} {'delta %':>12}")
    for key in ["Count", "Total Time(us)", "Min Time(us)", "Avg Time(us)", "Max Time(us)", "Ratio(%)"]:
        print_comparison(key, entry_a["op"].get(key), entry_b["op"].get(key), lower_is_better="Time" in key)

    print_section("Kernel Details")
    print(f"{'metric':34} {name_a:>14} {name_b:>14} {'delta':>14} {'delta %':>12}")
    print_comparison("kernel_count", entry_a["kernel"].get("kernel_count"), entry_b["kernel"].get("kernel_count"))
    kernel_keys = resolve_dynamic_columns(
        entry_a.get("kernel_columns", []),
        entry_b.get("kernel_columns", []),
        KEY_KERNEL_PREFERRED_COLUMNS,
    )
    for key in kernel_keys:
        lower = comparison_prefers_lower(key)
        print_comparison(key, entry_a["kernel"].get(key), entry_b["kernel"].get(key), lower_is_better=lower)

    l2_keys = resolve_dynamic_columns(
        entry_a.get("l2_columns", []),
        entry_b.get("l2_columns", []),
        KEY_L2_PREFERRED_COLUMNS,
    )
    if l2_keys:
        print_section("L2 Cache")
        print(f"{'metric':34} {name_a:>14} {name_b:>14} {'delta':>14} {'delta %':>12}")
        print_comparison("l2_count", entry_a["l2"].get("l2_count"), entry_b["l2"].get("l2_count"))
        for key in l2_keys:
            lower = comparison_prefers_lower(key)
            print_comparison(key, entry_a["l2"].get(key), entry_b["l2"].get(key), lower_is_better=lower)

    print_section("Step Trace")
    print(f"{'metric':34} {name_a:>14} {name_b:>14} {'delta':>14} {'delta %':>12}")
    for key in KEY_STEP_COLUMNS:
        lower = key not in {"Overlapped"}
        print_comparison(key, entry_a["step"].get(key), entry_b["step"].get(key), lower_is_better=lower)

    print_api_comparison(name_a, name_b, entry_a.get("api", {}), entry_b.get("api", {}))

    avg_a = entry_a["op"].get("Avg Time(us)")
    avg_b = entry_b["op"].get("Avg Time(us)")
    if avg_a and avg_b:
        speedup = avg_a / avg_b
        print_section("Summary")
        print(f"{name_b} speedup vs {name_a}: {speedup:.4f}x")
        print(f"{name_b} Avg Time change: {(avg_b - avg_a) / avg_a * 100.0:.2f}%")


def resolve_metric_keys(profile_a, profile_b):
    keys_a = {key for key in profile_a.keys() if not key.startswith("__")}
    keys_b = {key for key in profile_b.keys() if not key.startswith("__")}
    if keys_a == {"default"} and keys_b == {"default"}:
        return ["default"]

    shared = sorted(keys_a & keys_b)
    if shared:
        return shared

    raise ValueError(
        "No shared metrics found between the two inputs. "
        f"left={sorted(keys_a)}, right={sorted(keys_b)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare key metrics from two Ascend/Torch-NPU profiler result directories or profile_summary.json files."
    )
    parser.add_argument("dir_a", help="first profiler result directory or profile_summary.json, usually baseline")
    parser.add_argument("dir_b", help="second profiler result directory or profile_summary.json, usually candidate")
    parser.add_argument("--name-a", default="A", help="display name for first directory")
    parser.add_argument("--name-b", default="B", help="display name for second directory")
    parser.add_argument("--op", default=None, help="preferred op name when reading raw profiler output, e.g. _attn_fwd")
    args = parser.parse_args()

    prof_a = load_profile(args.dir_a, args.op)
    prof_b = load_profile(args.dir_b, args.op)
    metric_keys = resolve_metric_keys(prof_a, prof_b)

    print("Profile inputs")
    print("--------------")
    print(f"{args.name_a}: {prof_a['__source__']}")
    print(f"{args.name_b}: {prof_b['__source__']}")
    print(f"Shared metrics: {', '.join(metric_keys)}")

    for metric_name in metric_keys:
        print_profile_pair(metric_name, args.name_a, prof_a[metric_name], args.name_b, prof_b[metric_name])


if __name__ == "__main__":
    main()
