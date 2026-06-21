#!/usr/bin/env python3
import argparse
import importlib.util
import itertools
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu


DEFAULT_SM_SCALE = 0.5
DEFAULT_SEED = 0
DEFAULT_STAGE_A_ITERS = 1
DEFAULT_STAGE_B_ITERS = 1
DEFAULT_STAGE_C_ITERS = 3
DEFAULT_FINAL_ITERS = 5
DEFAULT_TOP_TILINGS = 2
DEFAULT_TOP_CONFIGS = 3
DEFAULT_PERSISTENT_CANDIDATES = [8, 12, 16, 20, 24, 32, 40, 48, 64]

TILING_CANDIDATES = [
    (32, 32),
    (64, 32),
    (64, 64),
    (128, 32),
]

COMPILER_SEARCH_SPACE = {
    "num_stages": [1, 2],
    "enable_hivm_auto_cv_balance": [True, False],
    "tile_mix_vector_loop": [2, 4],
    "tile_mix_cube_loop": [2, 4],
    "enable_ubuf_saving": [True],
}

DEFAULT_COMPILER_CONFIG = {
    "num_stages": 2,
    "enable_hivm_auto_cv_balance": True,
    "tile_mix_vector_loop": 2,
    "tile_mix_cube_loop": 2,
    "enable_ubuf_saving": True,
}


def _load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape_key(shape):
    z, h, n_ctx, head_dim, causal = shape
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


def _resolve_programs(z: int, h: int, n_ctx: int, block_m: int, persistent_blocks: int, max_programs: int = 65535):
    total_tiles = ((n_ctx + block_m - 1) // block_m) * z * h
    launched_programs = min(total_tiles, persistent_blocks, max_programs)
    return max(1, launched_programs)


def _make_inputs(device: str, dtype, z: int, h: int, n_ctx: int, head_dim: int, seed: int):
    torch.manual_seed(seed)
    q = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    v = torch.empty((z, h, n_ctx, head_dim), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    return q, k, v


def _launch_fixed(module, q, k, v, causal: bool, sm_scale: float, config: dict):
    z, h, n_ctx, head_dim = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((z, h, n_ctx), device=q.device, dtype=torch.float32)
    if head_dim < 256:
        acc = torch.empty((1,), dtype=torch.float32, device=q.device)
    else:
        acc = torch.zeros((z, h, n_ctx, head_dim), dtype=torch.float32, device=q.device)

    stage = 3 if causal else 1
    launched_programs = _resolve_programs(
        z,
        h,
        n_ctx,
        config["BLOCK_M"],
        config["persistent_blocks"],
    )
    grid = (launched_programs, 1, 1)

    module._attn_fwd_manual[grid](
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
    return out


def _measure_config(module, q, k, v, causal: bool, sm_scale: float, config: dict, timed_iters: int):
    _launch_fixed(module, q, k, v, causal, sm_scale, config)
    torch.npu.synchronize()

    samples_us = []
    out = None
    for _ in range(timed_iters):
        start = time.perf_counter()
        out = _launch_fixed(module, q, k, v, causal, sm_scale, config)
        torch.npu.synchronize()
        samples_us.append((time.perf_counter() - start) * 1e6)

    return out, {
        "avg_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "samples_us": samples_us,
        "iters": timed_iters,
    }


def _measure_baseline(eval_module, q, k, v, h: int, causal: bool, sm_scale: float, timed_iters: int):
    _baseline_attention(eval_module, q, k, v, h, causal, sm_scale)
    torch.npu.synchronize()

    samples_us = []
    out = None
    for _ in range(timed_iters):
        start = time.perf_counter()
        out = _baseline_attention(eval_module, q, k, v, h, causal, sm_scale)
        torch.npu.synchronize()
        samples_us.append((time.perf_counter() - start) * 1e6)

    return out, {
        "avg_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "samples_us": samples_us,
        "iters": timed_iters,
    }


def _compiler_candidates():
    keys = list(COMPILER_SEARCH_SPACE)
    values = [COMPILER_SEARCH_SPACE[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _persistent_candidates(shape, block_m: int):
    z, h, n_ctx, _, _ = shape
    total_tiles = ((n_ctx + block_m - 1) // block_m) * z * h
    candidates = []
    for value in DEFAULT_PERSISTENT_CANDIDATES:
        launched = min(total_tiles, value)
        if launched < 1:
            continue
        if launched not in candidates:
            candidates.append(launched)
    default_programs = min(total_tiles, 20)
    if default_programs not in candidates:
        candidates.append(default_programs)
    return sorted(candidates)


def _baseline_attention(eval_module, q, k, v, h: int, causal: bool, sm_scale: float):
    return eval_module._baseline_attention(q, k, v, h, causal, sm_scale)


@torch.no_grad()
def tune_case(module, eval_module, shape, dtype, sm_scale: float, seed: int, top_tilings: int, top_configs: int,
              stage_a_iters: int, stage_b_iters: int, stage_c_iters: int, final_iters: int):
    z, h, n_ctx, head_dim, causal = shape
    q, k, v = _make_inputs(module.DEVICE, dtype, z, h, n_ctx, head_dim, seed)

    stage_a_rows = []
    for bm, bn in TILING_CANDIDATES:
        config = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "persistent_blocks": min(((n_ctx + bm - 1) // bm) * z * h, 20),
            **DEFAULT_COMPILER_CONFIG,
        }
        try:
            _, stats = _measure_config(module, q, k, v, causal, sm_scale, config, stage_a_iters)
            stage_a_rows.append({"config": config, "avg_us": stats["avg_us"]})
        except Exception as exc:  # noqa: BLE001
            stage_a_rows.append({"config": config, "avg_us": None, "error": str(exc).splitlines()[0][:240]})

    viable_tilings = [row for row in stage_a_rows if row.get("avg_us") is not None]
    viable_tilings.sort(key=lambda row: row["avg_us"])
    selected_tilings = []
    for row in viable_tilings[:top_tilings]:
        selected_tilings.append((row["config"]["BLOCK_M"], row["config"]["BLOCK_N"]))

    stage_b_rows = []
    for bm, bn in selected_tilings:
        for compiler_cfg in _compiler_candidates():
            config = {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "persistent_blocks": min(((n_ctx + bm - 1) // bm) * z * h, 20),
                **compiler_cfg,
            }
            try:
                _, stats = _measure_config(module, q, k, v, causal, sm_scale, config, stage_b_iters)
                stage_b_rows.append({"config": config, "avg_us": stats["avg_us"]})
            except Exception as exc:  # noqa: BLE001
                stage_b_rows.append({"config": config, "avg_us": None, "error": str(exc).splitlines()[0][:240]})

    viable_stage_b = [row for row in stage_b_rows if row.get("avg_us") is not None]
    viable_stage_b.sort(key=lambda row: row["avg_us"])
    finalist_rows = viable_stage_b[:top_configs]

    stage_c_rows = []
    for row in finalist_rows:
        base_config = row["config"]
        for persistent_blocks in _persistent_candidates(shape, base_config["BLOCK_M"]):
            config = dict(base_config)
            config["persistent_blocks"] = persistent_blocks
            try:
                _, stats = _measure_config(module, q, k, v, causal, sm_scale, config, stage_c_iters)
                stage_c_rows.append({"config": config, "avg_us": stats["avg_us"]})
            except Exception as exc:  # noqa: BLE001
                stage_c_rows.append({"config": config, "avg_us": None, "error": str(exc).splitlines()[0][:240]})

    viable_stage_c = [row for row in stage_c_rows if row.get("avg_us") is not None]
    viable_stage_c.sort(key=lambda row: row["avg_us"])
    if not viable_stage_c:
        raise RuntimeError(f"no viable config found for shape {shape}")

    best_config = dict(viable_stage_c[0]["config"])
    candidate_out, candidate_stats = _measure_config(module, q, k, v, causal, sm_scale, best_config, final_iters)
    ref_out, baseline_stats = _measure_baseline(eval_module, q, k, v, h, causal, sm_scale, final_iters)

    try:
        ref_out = ref_out.to(dtype)
        max_abs_diff = torch.max(torch.abs(candidate_out.to(dtype) - ref_out)).item()
    except Exception as exc:  # noqa: BLE001
        max_abs_diff = None
        ref_out = None
        correctness_error = str(exc)
    else:
        correctness_error = None

    result = {
        **_shape_record(shape),
        "best_config": best_config,
        "candidate_avg_us": candidate_stats["avg_us"],
        "candidate_min_us": candidate_stats["min_us"],
        "candidate_max_us": candidate_stats["max_us"],
        "baseline_avg_us": baseline_stats["avg_us"],
        "baseline_min_us": baseline_stats["min_us"],
        "baseline_max_us": baseline_stats["max_us"],
        "speedup_vs_baseline": baseline_stats["avg_us"] / candidate_stats["avg_us"],
        "max_abs_diff": max_abs_diff,
        "correctness_error": correctness_error,
        "stage_a": stage_a_rows,
        "stage_b_top": viable_stage_b[: min(5, len(viable_stage_b))],
        "stage_c_top": viable_stage_c[: min(5, len(viable_stage_c))],
    }

    del q, k, v, candidate_out
    if ref_out is not None:
        del ref_out
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline tune flash_attention_forward.py for evaluate_attention.py performance cases.")
    parser.add_argument(
        "--candidate-module",
        default=str(Path(__file__).resolve().with_name("flash_attention_forward.py")),
        help="Path to flash_attention_forward.py",
    )
    parser.add_argument(
        "--evaluate-module",
        default=str(Path(__file__).resolve().with_name("evaluate_attention.py")),
        help="Path to evaluate_attention.py",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().with_name("flash_attention_forward_tuned.json")),
        help="Path to write tuned configs JSON",
    )
    parser.add_argument("--sm-scale", type=float, default=DEFAULT_SM_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16"])
    parser.add_argument("--top-tilings", type=int, default=DEFAULT_TOP_TILINGS)
    parser.add_argument("--top-configs", type=int, default=DEFAULT_TOP_CONFIGS)
    parser.add_argument("--stage-a-iters", type=int, default=DEFAULT_STAGE_A_ITERS)
    parser.add_argument("--stage-b-iters", type=int, default=DEFAULT_STAGE_B_ITERS)
    parser.add_argument("--stage-c-iters", type=int, default=DEFAULT_STAGE_C_ITERS)
    parser.add_argument("--final-iters", type=int, default=DEFAULT_FINAL_ITERS)
    args = parser.parse_args()

    dtype = torch.float16
    candidate_module = _load_module(Path(args.candidate_module), "flash_attention_candidate")
    evaluate_module = _load_module(Path(args.evaluate_module), "flash_attention_evaluate")
    performance_cases = list(evaluate_module.PERFORMANCE_CASES)

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
            args.top_tilings,
            args.top_configs,
            args.stage_a_iters,
            args.stage_b_iters,
            args.stage_c_iters,
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
        key = f"{item['Z']}:{item['H']}:{item['N_CTX']}:{item['HEAD_DIM']}:{int(item['causal'])}"
        tuned_table[key] = item["best_config"]

    payload = {
        "meta": {
            "candidate_module": str(Path(args.candidate_module).resolve()),
            "evaluate_module": str(Path(args.evaluate_module).resolve()),
            "dtype": args.dtype,
            "sm_scale": args.sm_scale,
            "seed": args.seed,
        },
        "tuned_configs": tuned_table,
        "results": results,
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Tuned config JSON written to: {output_path}")


if __name__ == "__main__":
    main()
