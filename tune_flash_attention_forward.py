#!/usr/bin/env python3
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

DEFAULT_SM_SCALE = 0.5
DEFAULT_SEED = 0
DEFAULT_TUNE_ITERS = 3
DEFAULT_FINAL_ITERS = 5
DEFAULT_OUTPUT_JSON = "flash_attention_forward_tuned.json"
DEFAULT_CANDIDATE_MODULE = "flash_attention_forward.py"
DEFAULT_EVALUATE_MODULE = "evaluate_attention.py"

BLOCK_M_CANDIDATES = (16, 32, 64, 128, 256)
BLOCK_N_CANDIDATES = (16, 32, 64, 128, 256)

PRESET_BEGIN = "# BEGIN AUTO-TUNED DEFAULT TILING PRESETS"
PRESET_END = "# END AUTO-TUNED DEFAULT TILING PRESETS"

torch = None


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_shape_arg(raw: str):
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) != 5:
        raise ValueError("shape must be formatted as 'Z,H,N_CTX,HEAD_DIM,causal'")
    z, h, n_ctx, head_dim = (int(value) for value in parts[:4])
    causal_raw = parts[4].lower()
    if causal_raw in {"1", "true", "t", "yes", "y"}:
        causal = True
    elif causal_raw in {"0", "false", "f", "no", "n"}:
        causal = False
    else:
        raise ValueError(f"invalid causal flag: {parts[4]}")
    return (z, h, n_ctx, head_dim, causal)


def _shape_record(shape):
    z, h, n_ctx, head_dim, causal = shape
    return {
        "Z": z,
        "H": h,
        "N_CTX": n_ctx,
        "HEAD_DIM": head_dim,
        "causal": causal,
    }


def _shape_key_string(shape):
    z, h, n_ctx, head_dim, causal = shape
    return f"{z}:{h}:{n_ctx}:{head_dim}:{int(causal)}"


def _shape_key_from_record(record: dict):
    return f"{record['Z']}:{record['H']}:{record['N_CTX']}:{record['HEAD_DIM']}:{int(record['causal'])}"


def _make_inputs(device: str, dtype, z: int, h: int, n_ctx: int, head_dim: int, seed: int):
    torch.manual_seed(seed)
    q = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    v = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    return q, k, v


def _generate_tiling_candidates(shape):
    _z, _h, n_ctx, head_dim, causal = shape
    candidates = []
    seen = set()

    for bm in BLOCK_M_CANDIDATES:
        if bm > max(256, n_ctx):
            continue
        if head_dim >= 256 and bm > 64:
            continue
        for bn in BLOCK_N_CANDIDATES:
            if bn > max(256, n_ctx):
                continue
            if bm < bn:
                continue
            if head_dim >= 256 and bn > 64:
                continue
            if causal and bn > bm:
                continue

            candidate = (bm, bn)
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    fallback_priority = [
        (32, 32),
        (64, 32),
        (64, 64),
        (128, 32),
        (128, 64),
        (128, 128),
    ]
    for candidate in fallback_priority:
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    return candidates


def _resolve_programs(module, z: int, h: int, n_ctx: int, block_m: int):
    total_tiles = ((n_ctx + block_m - 1) // block_m) * z * h
    persistent = module._get_persistent_programs()
    return (min(total_tiles, persistent), 1, 1)


def _launch_fixed(module, q, k, v, causal: bool, sm_scale: float, bm: int, bn: int):
    if hasattr(module, "_launch_kernel"):
        out, _lse = module._launch_kernel(q, k, v, causal, sm_scale, bm=bm, bn=bn)
        return out

    if hasattr(module, "attention"):
        return module.attention(q, k, v, causal, sm_scale, BM=bm, BN=bn)

    raise AttributeError("candidate module does not expose `_launch_kernel` or `attention`")


def _measure_candidate(module, q, k, v, causal: bool, sm_scale: float, bm: int, bn: int, iters: int):
    _launch_fixed(module, q, k, v, causal, sm_scale, bm, bn)
    torch.npu.synchronize()

    samples_us = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = _launch_fixed(module, q, k, v, causal, sm_scale, bm, bn)
        torch.npu.synchronize()
        samples_us.append((time.perf_counter() - start) * 1e6)
    return out, {
        "avg_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "samples_us": samples_us,
        "iters": iters,
    }


def _measure_baseline(eval_module, q, k, v, h: int, causal: bool, sm_scale: float, iters: int):
    eval_module._baseline_attention(q, k, v, h, causal, sm_scale)
    torch.npu.synchronize()

    samples_us = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = eval_module._baseline_attention(q, k, v, h, causal, sm_scale)
        torch.npu.synchronize()
        samples_us.append((time.perf_counter() - start) * 1e6)
    return out, {
        "avg_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "samples_us": samples_us,
        "iters": iters,
    }


def _error_summary(exc: Exception):
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return message.splitlines()[0][:240]


def _merge_with_existing_output(output_path: Path, payload: dict):
    if not output_path.exists():
        return payload

    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return payload

    merged_configs = dict(existing.get("tuned_configs", {}))
    merged_configs.update(payload.get("tuned_configs", {}))

    merged_results = {}
    ordered_keys = []
    for item in existing.get("results", []):
        if not isinstance(item, dict):
            continue
        key = _shape_key_from_record(item)
        if key not in ordered_keys:
            ordered_keys.append(key)
        merged_results[key] = item
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        key = _shape_key_from_record(item)
        if key not in ordered_keys:
            ordered_keys.append(key)
        merged_results[key] = item

    merged_payload = dict(payload)
    merged_payload["tuned_configs"] = merged_configs
    merged_payload["results"] = [merged_results[key] for key in ordered_keys]
    return merged_payload


def tune_case(module, eval_module, shape, dtype, sm_scale: float, seed: int, tune_iters: int, final_iters: int):
    z, h, n_ctx, head_dim, causal = shape
    q, k, v = _make_inputs(module.DEVICE, dtype, z, h, n_ctx, head_dim, seed)

    tiling_candidates = _generate_tiling_candidates(shape)
    candidate_rows = []
    for bm, bn in tiling_candidates:
        try:
            _, stats = _measure_candidate(module, q, k, v, causal, sm_scale, bm, bn, tune_iters)
            candidate_rows.append({
                "config": {"BLOCK_M": bm, "BLOCK_N": bn},
                "avg_us": stats["avg_us"],
                "min_us": stats["min_us"],
                "max_us": stats["max_us"],
            })
        except Exception as exc:  # noqa: BLE001
            candidate_rows.append({
                "config": {"BLOCK_M": bm, "BLOCK_N": bn},
                "avg_us": None,
                "error": _error_summary(exc),
            })

    viable = [row for row in candidate_rows if row.get("avg_us") is not None]
    viable.sort(key=lambda row: row["avg_us"])
    if not viable:
        raise RuntimeError(f"no viable config found for shape {shape}")

    best = viable[0]["config"]
    candidate_out, candidate_stats = _measure_candidate(
        module, q, k, v, causal, sm_scale, best["BLOCK_M"], best["BLOCK_N"], final_iters
    )
    ref_out, baseline_stats = _measure_baseline(eval_module, q, k, v, h, causal, sm_scale, final_iters)

    try:
        ref_out = ref_out.to(dtype)
        max_abs_diff = torch.max(torch.abs(candidate_out.to(dtype) - ref_out)).item()
        correctness_error = None
    except Exception as exc:  # noqa: BLE001
        max_abs_diff = None
        correctness_error = str(exc)

    result = {
        **_shape_record(shape),
        "best_config": best,
        "candidate_avg_us": candidate_stats["avg_us"],
        "candidate_min_us": candidate_stats["min_us"],
        "candidate_max_us": candidate_stats["max_us"],
        "baseline_avg_us": baseline_stats["avg_us"],
        "baseline_min_us": baseline_stats["min_us"],
        "baseline_max_us": baseline_stats["max_us"],
        "speedup_vs_baseline": baseline_stats["avg_us"] / candidate_stats["avg_us"],
        "max_abs_diff": max_abs_diff,
        "correctness_error": correctness_error,
        "candidates": candidate_rows,
    }

    del q, k, v, candidate_out, ref_out
    return result


def _format_presets_block(tuned_configs: dict):
    lines = [
        PRESET_BEGIN,
        "# Default tiling presets captured from offline tuning.",
        "# This block is rewritten by `tune_flash_attention_forward.py`.",
        "DEFAULT_TILING_PRESETS = {",
    ]
    sorted_items = sorted(
        tuned_configs.items(),
        key=lambda item: tuple(int(part) for part in item[0].split(":")),
    )
    for key, config in sorted_items:
        z, h, n_ctx, head_dim, causal = key.split(":")
        causal_bool = "True" if int(causal) else "False"
        lines.append(
            f"    ({int(z)}, {int(h)}, {int(n_ctx)}, {int(head_dim)}, {causal_bool}): "
            f"({int(config['BLOCK_M'])}, {int(config['BLOCK_N'])}),"
        )
    lines.append("}")
    lines.append(PRESET_END)
    return "\n".join(lines)


def apply_tuned_presets(candidate_module_path: Path, tuned_json_path: Path):
    payload = json.loads(tuned_json_path.read_text(encoding="utf-8"))
    tuned_configs = payload.get("tuned_configs", {})
    if not tuned_configs:
        raise ValueError(f"no tuned_configs found in {tuned_json_path}")

    source = candidate_module_path.read_text(encoding="utf-8")
    begin = source.find(PRESET_BEGIN)
    end = source.find(PRESET_END)
    if begin < 0 or end < 0 or end < begin:
        raise RuntimeError(
            f"cannot locate preset markers in {candidate_module_path}; expected {PRESET_BEGIN!r} and {PRESET_END!r}"
        )

    end += len(PRESET_END)
    replacement = _format_presets_block(tuned_configs)
    updated = source[:begin] + replacement + source[end:]
    candidate_module_path.write_text(updated, encoding="utf-8")


def _ensure_torch_runtime():
    global torch
    if torch is not None:
        return torch

    import torch as _torch
    import torch_npu  # noqa: F401

    torch = _torch
    return torch


def main():
    parser = argparse.ArgumentParser(
        description="Offline tune flash_attention_forward.py and optionally apply tuned tiling presets."
    )
    parser.add_argument(
        "--candidate-module",
        default=str(Path(__file__).resolve().with_name(DEFAULT_CANDIDATE_MODULE)),
        help="Path to flash_attention_forward.py",
    )
    parser.add_argument(
        "--evaluate-module",
        default=str(Path(__file__).resolve().with_name(DEFAULT_EVALUATE_MODULE)),
        help="Path to evaluate_attention.py",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().with_name(DEFAULT_OUTPUT_JSON)),
        help="Path to write tuned configs JSON",
    )
    parser.add_argument("--sm-scale", type=float, default=DEFAULT_SM_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16"])
    parser.add_argument("--tune-iters", type=int, default=DEFAULT_TUNE_ITERS)
    parser.add_argument("--final-iters", type=int, default=DEFAULT_FINAL_ITERS)
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Overwrite output JSON instead of merging new shapes into an existing file.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        help="Optional shape override formatted as 'Z,H,N_CTX,HEAD_DIM,causal'. May be passed multiple times.",
    )
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="Only write tuned JSON, do not rewrite flash_attention_forward.py presets.",
    )
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Skip tuning and only rewrite flash_attention_forward.py from the existing tuned JSON.",
    )
    args = parser.parse_args()

    candidate_module_path = Path(args.candidate_module).resolve()
    output_json_path = Path(args.output_json).resolve()

    if args.apply_only:
        apply_tuned_presets(candidate_module_path, output_json_path)
        print(f"Applied tuned presets from {output_json_path} to {candidate_module_path}")
        return

    torch_runtime = _ensure_torch_runtime()
    dtype = torch_runtime.float16
    candidate_module = _load_module(candidate_module_path, "flash_attention_candidate")
    evaluate_module = _load_module(Path(args.evaluate_module).resolve(), "flash_attention_evaluate")
    performance_cases = (
        [_parse_shape_arg(raw_shape) for raw_shape in args.shape]
        if args.shape
        else list(evaluate_module.PERFORMANCE_CASES)
    )

    results = []
    for idx, shape in enumerate(performance_cases, start=1):
        print(f"[{idx}/{len(performance_cases)}] tuning shape={shape}")
        result = tune_case(
            candidate_module,
            evaluate_module,
            shape,
            dtype,
            args.sm_scale,
            args.seed,
            args.tune_iters,
            args.final_iters,
        )
        print(
            "  best: "
            f"config={result['best_config']}, "
            f"candidate_avg_us={result['candidate_avg_us']:.3f}, "
            f"baseline_avg_us={result['baseline_avg_us']:.3f}, "
            f"speedup={result['speedup_vs_baseline']:.3f}"
        )
        results.append(result)

    tuned_table = {}
    for item in results:
        tuned_table[_shape_key_string((
            item["Z"],
            item["H"],
            item["N_CTX"],
            item["HEAD_DIM"],
            item["causal"],
        ))] = item["best_config"]

    payload = {
        "meta": {
            "candidate_module": str(candidate_module_path),
            "evaluate_module": str(Path(args.evaluate_module).resolve()),
            "dtype": args.dtype,
            "sm_scale": args.sm_scale,
            "seed": args.seed,
            "tune_iters": args.tune_iters,
            "final_iters": args.final_iters,
        },
        "tuned_configs": tuned_table,
        "results": results,
    }

    if not args.replace_output:
        payload = _merge_with_existing_output(output_json_path, payload)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Tuned config JSON written to: {output_json_path}")

    if not args.skip_apply:
        apply_tuned_presets(candidate_module_path, output_json_path)
        print(f"Applied tuned presets to: {candidate_module_path}")


if __name__ == "__main__":
    main()
