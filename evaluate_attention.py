#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch_npu


CORRECTNESS_CASES = [
    (1, 1, 64, 64, False),
    (1, 1, 64, 64, True),
    (1, 1, 128, 128, False),
    (1, 1, 128, 128, True),
    (1, 2, 1024, 64, False),
    (1, 2, 1024, 64, True),
    (4, 32, 64, 64, False),
    (4, 32, 64, 64, True),
    (4, 32, 128, 128, False),
    (4, 32, 256, 128, True),
    (4, 32, 512, 64, True),
    (4, 32, 1024, 64, False),
    (4, 32, 1024, 64, True),
    (4, 32, 1024, 128, False),
    (4, 32, 2048, 64, True),
    (4, 32, 2048, 128, False),
    (4, 32, 4096, 64, False),
    (128, 8, 1024, 64, False),
]

PERFORMANCE_CASES = [
    (128, 8, 1024, 128, True),
    (128, 8, 1024, 256, True),
    (128, 8, 2048, 128, True),
    (128, 8, 2048, 256, False),
    (128, 8, 4096, 128, False),
    (128, 8, 8192, 64, False),
]
CORRECTNESS_TOTAL_POINTS = 40.0
PERFORMANCE_TOTAL_POINTS = 60.0
DEFAULT_SM_SCALE = 0.5
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def _load_candidate_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("flash_attention_candidate", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dtype_from_name(name: str):
    normalized = name.strip().lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _dtype_to_name(dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype)


def _shape_key(shape):
    z, h, n_ctx, head_dim, causal = shape
    return {
        "Z": z,
        "H": h,
        "N_CTX": n_ctx,
        "HEAD_DIM": head_dim,
        "causal": causal,
    }


def _make_inputs(device: str, dtype, z: int, h: int, n_ctx: int, head_dim: int, seed: int):
    torch.manual_seed(seed)
    q = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    v = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    return q, k, v


def _maybe_empty_cache():
    empty_cache = getattr(torch.npu, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _baseline_attention(q, k, v, h: int, causal: bool, sm_scale: float):
    atten_mask = None
    sparse_mode = 0
    pre_tockens = 65535
    next_tockens = 65535
    if causal:
        # `sparse_mode=2` only accepts a compressed 2048x2048 causal mask on this torch_npu build.
        # Use defaultMask mode so generic causal sequence lengths, such as 1024, remain supported.
        atten_mask = torch.triu(
            torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        pre_tockens = q.shape[-2]
        next_tockens = 0
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        h,
        padding_mask=None,
        atten_mask=atten_mask,
        scale=sm_scale,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        sparse_mode=sparse_mode,
    )[0]


def _candidate_attention(module, q, k, v, causal: bool, sm_scale: float):
    attention_cls = getattr(module, "attention")
    return attention_cls.forward(q, k, v, causal, sm_scale).to(q.dtype)


def _reference_attention(module, q, k, v, h: int, causal: bool, sm_scale: float):
    try:
        return _baseline_attention(q, k, v, h, causal, sm_scale)
    except RuntimeError as exc:
        fallback = getattr(module, "_torch_attention_reference", None)
        if fallback is None:
            raise RuntimeError(f"torch_npu baseline unavailable and no fallback reference exists: {exc}") from exc
        return fallback(q, k, v, causal, sm_scale)


def _measure_us(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    times_us = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        torch.npu.synchronize()
        times_us.append((time.perf_counter() - start) * 1e6)

    return out, {
        "avg_us": statistics.mean(times_us),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "iters": iters,
        "warmup": warmup,
        "samples_us": times_us,
    }


def _safe_speedup(baseline_us, candidate_us):
    if baseline_us is None or candidate_us is None:
        return None
    if candidate_us <= 0.0:
        return None
    return baseline_us / candidate_us


def _correctness_case_points():
    return CORRECTNESS_TOTAL_POINTS / len(CORRECTNESS_CASES)


def _performance_case_points():
    return PERFORMANCE_TOTAL_POINTS / len(PERFORMANCE_CASES)


@torch.no_grad()
def run_correctness_case(module, device, dtype, shape, sm_scale, seed, atol, rtol):
    z, h, n_ctx, head_dim, causal = shape
    q, k, v = _make_inputs(device, dtype, z, h, n_ctx, head_dim, seed)

    error = None
    passed = False
    max_abs_diff = None
    points = 0.0
    try:
        ref_out = _reference_attention(module, q, k, v, h, causal, sm_scale).to(dtype)
        tri_out = _candidate_attention(module, q, k, v, causal, sm_scale)
        max_abs_diff = torch.max(torch.abs(tri_out - ref_out)).item()
        passed = torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
        if passed:
            points = _correctness_case_points()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    result = {
        **_shape_key(shape),
        "dtype": _dtype_to_name(dtype),
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
        "max_abs_diff": max_abs_diff,
        "score": points,
        "error": error,
    }

    del q, k, v
    _maybe_empty_cache()
    return result


@torch.no_grad()
def run_performance_case(module, device, dtype, shape, sm_scale, seed, atol, rtol, warmup, iters):
    z, h, n_ctx, head_dim, causal = shape
    q, k, v = _make_inputs(device, dtype, z, h, n_ctx, head_dim, seed)

    error = None
    output_match = False
    max_abs_diff = None
    baseline_stats = None
    candidate_stats = None
    speedup = None
    points = 0.0

    try:
        ref_out = _baseline_attention(q, k, v, h, causal, sm_scale).to(dtype)
        tri_out = _candidate_attention(module, q, k, v, causal, sm_scale)
        max_abs_diff = torch.max(torch.abs(tri_out - ref_out)).item()
        output_match = torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)

        if output_match:
            _, baseline_stats = _measure_us(
                lambda: _baseline_attention(q, k, v, h, causal, sm_scale),
                warmup,
                iters,
            )
            _, candidate_stats = _measure_us(
                lambda: _candidate_attention(module, q, k, v, causal, sm_scale),
                warmup,
                iters,
            )
            speedup = _safe_speedup(baseline_stats["avg_us"], candidate_stats["avg_us"])
            if speedup is not None:
                points = _performance_case_points() * min(max(speedup, 0.0), 1.0)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    result = {
        **_shape_key(shape),
        "dtype": _dtype_to_name(dtype),
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "output_match": output_match,
        "max_abs_diff": max_abs_diff,
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "speedup": speedup,
        "score": points,
        "error": error,
    }

    del q, k, v
    _maybe_empty_cache()
    return result


def _summarize_correctness(results):
    passed_cases = sum(1 for item in results if item["passed"])
    score = sum(item["score"] for item in results)
    return {
        "total_cases": len(results),
        "passed_cases": passed_cases,
        "points_per_case": _correctness_case_points(),
        "score": score,
        "max_score": CORRECTNESS_TOTAL_POINTS,
        "cases": results,
    }


def _summarize_performance(results):
    matched_cases = sum(1 for item in results if item["output_match"])
    score = sum(item["score"] for item in results)
    valid_speedups = [item["speedup"] for item in results if item["speedup"] is not None]
    return {
        "total_cases": len(results),
        "matched_cases": matched_cases,
        "points_per_case": _performance_case_points(),
        "score": score,
        "max_score": PERFORMANCE_TOTAL_POINTS,
        "mean_speedup": statistics.mean(valid_speedups) if valid_speedups else None,
        "cases": results,
    }


def _format_float(value, precision=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return str(value)
        return f"{value:.{precision}f}"
    return str(value)


def _write_reports(report_root: Path, summary: dict):
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "evaluation_report.json"
    txt_path = report_root / "evaluation_report.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    lines = []
    lines.append("Evaluation Summary")
    lines.append("==================")
    lines.append(f"candidate_module={summary['config']['candidate_module']}")
    lines.append(f"dtype={summary['config']['dtype']}, sm_scale={summary['config']['sm_scale']}")
    lines.append(
        "correctness_score={correctness} / {correctness_max}, performance_score={performance} / {performance_max}, total_score={total} / 100".format(
            correctness=_format_float(summary["correctness"]["score"]),
            correctness_max=_format_float(summary["correctness"]["max_score"]),
            performance=_format_float(summary["performance"]["score"]),
            performance_max=_format_float(summary["performance"]["max_score"]),
            total=_format_float(summary["total_score"]),
        )
    )
    lines.append("")

    lines.append("Correctness Cases")
    lines.append("-----------------")
    for idx, case in enumerate(summary["correctness"]["cases"], start=1):
        lines.append(
            "[{idx}] shape=({Z},{H},{N_CTX},{HEAD_DIM}, causal={causal}) passed={passed} score={score} max_abs_diff={diff} error={error}".format(
                idx=idx,
                Z=case["Z"],
                H=case["H"],
                N_CTX=case["N_CTX"],
                HEAD_DIM=case["HEAD_DIM"],
                causal=case["causal"],
                passed=case["passed"],
                score=_format_float(case["score"]),
                diff=_format_float(case["max_abs_diff"], precision=6),
                error=case["error"] or "",
            )
        )
    lines.append("")

    lines.append("Performance Cases")
    lines.append("-----------------")
    for idx, case in enumerate(summary["performance"]["cases"], start=1):
        baseline_avg = case["baseline"]["avg_us"] if case["baseline"] else None
        candidate_avg = case["candidate"]["avg_us"] if case["candidate"] else None
        lines.append(
            "[{idx}] shape=({Z},{H},{N_CTX},{HEAD_DIM}, causal={causal}) match={output_match} score={score} baseline_avg_us={baseline} candidate_avg_us={candidate} speedup={speedup} max_abs_diff={diff} error={error}".format(
                idx=idx,
                Z=case["Z"],
                H=case["H"],
                N_CTX=case["N_CTX"],
                HEAD_DIM=case["HEAD_DIM"],
                causal=case["causal"],
                output_match=case["output_match"],
                baseline=_format_float(baseline_avg),
                candidate=_format_float(candidate_avg),
                speedup=_format_float(case["speedup"], precision=4),
                score=_format_float(case["score"]),
                diff=_format_float(case["max_abs_diff"], precision=6),
                error=case["error"] or "",
            )
        )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return json_path, txt_path


def _print_summary(summary: dict, json_path: Path, txt_path: Path):
    print("Evaluation complete")
    print(f"JSON report: {json_path}")
    print(f"Text report: {txt_path}")
    print(
        "Correctness: {score:.3f} / {max_score:.3f} ({passed}/{total} passed)".format(
            score=summary["correctness"]["score"],
            max_score=summary["correctness"]["max_score"],
            passed=summary["correctness"]["passed_cases"],
            total=summary["correctness"]["total_cases"],
        )
    )
    print(
        "Performance: {score:.3f} / {max_score:.3f} ({matched}/{total} matched, mean_speedup={speedup})".format(
            score=summary["performance"]["score"],
            max_score=summary["performance"]["max_score"],
            matched=summary["performance"]["matched_cases"],
            total=summary["performance"]["total_cases"],
            speedup=_format_float(summary["performance"]["mean_speedup"], precision=4),
        )
    )
    print(f"Total score: {_format_float(summary['total_score'])} / 100")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate /workspace/new_attn/flash_attention_forward.py correctness and performance using shape-only inputs."
    )
    parser.add_argument(
        "--candidate",
        default=str(Path(__file__).resolve().with_name("flash_attention_forward.py")),
        help="Path to the candidate flash attention module.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "correctness", "performance"],
        default="all",
        help="Select which suites to run.",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--sm-scale", dest="sm_scale", type=float, default=DEFAULT_SM_SCALE)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report-dir",
        default=str(Path(__file__).resolve().with_name("evaluation_reports")),
        help="Directory where reports are written.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    candidate_path = Path(args.candidate).resolve()
    report_dir = Path(args.report_dir).resolve()
    module = _load_candidate_module(candidate_path)
    dtype = _dtype_from_name(args.dtype)
    device = getattr(module, "DEVICE", "npu")

    correctness_results = []
    performance_results = []

    if args.mode in {"all", "correctness"}:
        for idx, shape in enumerate(CORRECTNESS_CASES):
            print(f"Running correctness case {idx + 1}/{len(CORRECTNESS_CASES)}: {shape}")
            correctness_results.append(
                run_correctness_case(
                    module,
                    device,
                    dtype,
                    shape,
                    args.sm_scale,
                    args.seed + idx,
                    args.atol,
                    args.rtol,
                )
            )

    if args.mode in {"all", "performance"}:
        offset = len(CORRECTNESS_CASES)
        for idx, shape in enumerate(PERFORMANCE_CASES):
            print(f"Running performance case {idx + 1}/{len(PERFORMANCE_CASES)}: {shape}")
            performance_results.append(
                run_performance_case(
                    module,
                    device,
                    dtype,
                    shape,
                    args.sm_scale,
                    args.seed + offset + idx,
                    args.atol,
                    args.rtol,
                    args.warmup,
                    args.iters,
                )
            )

    correctness_summary = _summarize_correctness(correctness_results)
    performance_summary = _summarize_performance(performance_results)
    total_score = correctness_summary["score"] + performance_summary["score"]

    summary = {
        "config": {
            "candidate_module": str(candidate_path),
            "report_dir": str(report_dir),
            "dtype": _dtype_to_name(dtype),
            "sm_scale": args.sm_scale,
            "atol": args.atol,
            "rtol": args.rtol,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "scoring": {
                "correctness_total": CORRECTNESS_TOTAL_POINTS,
                "performance_total": PERFORMANCE_TOTAL_POINTS,
                "correctness_points_per_case": _correctness_case_points(),
                "performance_points_per_case": _performance_case_points(),
                "performance_formula": "per_case_score = 10 * min(baseline_avg_us / candidate_avg_us, 1.0) when output matches",
            },
        },
        "correctness": correctness_summary,
        "performance": performance_summary,
        "total_score": total_score,
    }

    json_path, txt_path = _write_reports(report_dir, summary)
    _print_summary(summary, json_path, txt_path)


if __name__ == "__main__":
    main()
