#!/usr/bin/env python3
import argparse
import inspect
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import flash_attention_forward as flash_attn


def _dtype_from_name(name: str):
    normalized = name.strip().lower()
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Standalone runner for msprof collection using flash_attention_forward.attention."
    )
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--n-ctx", type=int, required=True, dest="n_ctx")
    parser.add_argument("--head-dim", type=int, required=True, dest="head_dim")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--sm-scale", type=float, default=0.5, dest="sm_scale")
    parser.add_argument("--block-m", type=int, dest="block_m")
    parser.add_argument("--block-n", type=int, dest="block_n")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--persistent-programs", type=int)
    parser.add_argument("--print-debug-context", action="store_true")
    return parser


def _safe_call(func, default=None):
    if func is None:
        return default
    try:
        return func()
    except Exception:
        return default


def _resolve_triton_paths():
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    dump_dir = os.environ.get("TRITON_DUMP_DIR")

    try:
        import triton.runtime.cache as triton_cache
    except Exception:
        return cache_dir, dump_dir

    if cache_dir is None:
        cache_dir = _safe_call(getattr(triton_cache, "default_cache_dir", None))
    if dump_dir is None:
        dump_dir = _safe_call(getattr(triton_cache, "default_dump_dir", None))
    return cache_dir, dump_dir


def _print_debug_context(args, runtime):
    kernel_fn = getattr(flash_attn._attn_fwd, "fn", None)
    kernel_source = inspect.getsourcefile(kernel_fn) if kernel_fn is not None else None
    kernel_code = getattr(kernel_fn, "__code__", None)
    cache_dir, dump_dir = _resolve_triton_paths()

    print("Debug context")
    print("-------------")
    print(f"module_file={Path(flash_attn.__file__).resolve()}")
    print(f"kernel_name={flash_attn.PROFILE_OP_NAME}")
    print(f"kernel_source_file={kernel_source or 'n/a'}")
    print(f"kernel_source_line={getattr(kernel_code, 'co_firstlineno', 'n/a')}")
    print(f"shape=(z={args.z}, h={args.h}, n_ctx={args.n_ctx}, head_dim={args.head_dim})")
    print(f"causal={args.causal}")
    print(f"dtype={args.dtype}")
    print(f"sm_scale={args.sm_scale}")
    print(f"selected_tiling={runtime['selected_config']}")
    print(f"tiling_source={runtime['tiling_source']}")
    print(f"stage={runtime['stage']}")
    print(f"total_tiles={runtime['total_tiles']}")
    print(f"launched_programs={runtime['launched_programs']}")
    print(f"triton_cache_dir={cache_dir or 'n/a'}")
    print(f"triton_dump_dir={dump_dir or 'n/a'}")
    for env_name in (
        "TRITON_DISABLE_LINE_INFO",
        "TRITON_ALWAYS_COMPILE",
        "TRITON_DEBUG",
        "TRITON_CACHE_DIR",
        "TRITON_DUMP_DIR",
    ):
        print(f"{env_name}={os.environ.get(env_name, '<unset>')}")
    print("")


def main():
    args = _build_arg_parser().parse_args()
    if (args.block_m is None) != (args.block_n is None):
        raise ValueError("--block-m and --block-n must be provided together")

    if args.persistent_programs is not None:
        import os
        os.environ["PERSISTENT_PROGRAMS"] = str(args.persistent_programs)

    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    q, k, v = flash_attn._make_inputs(args.z, args.h, args.n_ctx, args.head_dim, dtype)

    bm = args.block_m
    bn = args.block_n

    runtime = flash_attn._describe_runtime(
        args.z, args.h, args.n_ctx, args.head_dim, args.causal, bm=bm, bn=bn
    )
    print(f"Selected tiling config: {runtime['selected_config']} (source={runtime['tiling_source']})")
    if args.print_debug_context:
        _print_debug_context(args, runtime)

    for _ in range(args.warmup):
        flash_attn.attention(q, k, v, args.causal, args.sm_scale, BM=bm, BN=bn)
    torch.npu.synchronize()

    for _ in range(args.iters):
        flash_attn.attention(q, k, v, args.causal, args.sm_scale, BM=bm, BN=bn)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
