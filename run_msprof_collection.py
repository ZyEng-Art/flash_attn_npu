#!/usr/bin/env python3
"""
Collect msprof simulator outputs and Triton intermediate artifacts under msprof_runs/.
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MSPROF_CANDIDATES = [
    "/usr/local/Ascend/cann-9.1.0-beta.1/bin/msprof",
    "/usr/local/Ascend/cann/tools/profiler/bin/msprof",
]
DEFAULT_SOC_VERSION = "Ascend910B3"
RUNNER_PATH = Path("/workspace/new_attn/msprof_attention_runner.py")
RUNS_ROOT = Path("/workspace/new_attn/msprof_runs")


def _find_msprof():
    for candidate in MSPROF_CANDIDATES:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    resolved = shutil_which("msprof")
    if resolved:
        return resolved
    raise FileNotFoundError("Cannot find msprof in known locations or PATH")


def shutil_which(name):
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry) / name
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _quote_cmd(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def _build_runner_command(args):
    cmd = [
        sys.executable,
        str(RUNNER_PATH),
        "--z", str(args.z),
        "--h", str(args.h),
        "--n-ctx", str(args.n_ctx),
        "--head-dim", str(args.head_dim),
        "--dtype", args.dtype,
        "--sm-scale", str(args.sm_scale),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--seed", str(args.seed),
        "--print-debug-context",
    ]
    if args.causal:
        cmd.append("--causal")
    if args.block_m is not None:
        cmd.extend(["--block-m", str(args.block_m), "--block-n", str(args.block_n)])
    if args.persistent_programs is not None:
        cmd.extend(["--persistent-programs", str(args.persistent_programs)])
    return cmd


def _build_msprof_command(msprof_path, args, output_dir, runner_cmd):
    return [
        msprof_path,
        "op",
        "simulator",
        "--kernel-name", args.kernel_name,
        "--soc-version", args.soc_version,
        "--output", str(output_dir),
        "--application", _quote_cmd(runner_cmd),
    ]


def _write_run_metadata(run_dir, env_vars, runner_cmd, msprof_cmd):
    (run_dir / "runner_command.txt").write_text(_quote_cmd(runner_cmd) + "\n", encoding="utf-8")
    (run_dir / "msprof_command.txt").write_text(_quote_cmd(msprof_cmd) + "\n", encoding="utf-8")

    lines = []
    for key in sorted(env_vars):
        lines.append(f"{key}={env_vars[key]}")
    (run_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tail_log(path, lines=80):
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _write_artifact_manifest(run_dir):
    manifest_path = run_dir / "artifact_manifest.txt"
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path == manifest_path:
            continue
        entries.append(str(path.relative_to(run_dir)))
    manifest_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run msprof op simulator and archive all outputs under msprof_runs/."
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
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--persistent-programs", type=int)
    parser.add_argument("--kernel-name", default="_attn_fwd")
    parser.add_argument("--soc-version", default=DEFAULT_SOC_VERSION)
    parser.add_argument("--run-name")
    parser.add_argument("--keep-existing-dir", action="store_true")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    if (args.block_m is None) != (args.block_n is None):
        raise ValueError("--block-m and --block-n must be provided together")

    msprof_path = _find_msprof()
    run_name = args.run_name or _timestamp()
    run_dir = RUNS_ROOT / run_name
    if run_dir.exists() and not args.keep_existing_dir:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    triton_cache_dir = run_dir / "triton_cache"
    triton_dump_dir = run_dir / "triton_dump"
    simulator_out_dir = run_dir / "simulator_out"
    stdout_path = run_dir / "msprof_stdout.log"
    stderr_path = run_dir / "msprof_stderr.log"

    env = os.environ.copy()
    env_updates = {
        "TRITON_DISABLE_LINE_INFO": "false",
        "TRITON_ALWAYS_COMPILE": "1",
        "TRITON_DEBUG": "1",
        "TRITON_CACHE_DIR": str(triton_cache_dir),
        "TRITON_DUMP_DIR": str(triton_dump_dir),
    }
    env.update(env_updates)

    runner_cmd = _build_runner_command(args)
    msprof_cmd = _build_msprof_command(msprof_path, args, simulator_out_dir, runner_cmd)
    _write_run_metadata(run_dir, env_updates, runner_cmd, msprof_cmd)

    print(f"run_dir={run_dir}")
    print(f"msprof={msprof_path}")
    print(f"runner_command={_quote_cmd(runner_cmd)}")
    print(f"msprof_command={_quote_cmd(msprof_cmd)}")

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(msprof_cmd, cwd="/workspace", env=env, stdout=stdout_f, stderr=stderr_f, text=True)

    print(f"returncode={proc.returncode}")
    print(f"stdout_log={stdout_path}")
    print(f"stderr_log={stderr_path}")
    print(f"simulator_out={simulator_out_dir}")
    print(f"triton_cache={triton_cache_dir}")
    print(f"triton_dump={triton_dump_dir}")

    if proc.returncode != 0:
        stderr_tail = _tail_log(stderr_path)
        stdout_tail = _tail_log(stdout_path)
        if stdout_tail:
            print("\nLast stdout lines:")
            print(stdout_tail)
        if stderr_tail:
            print("\nLast stderr lines:")
            print(stderr_tail)
        raise SystemExit(proc.returncode)

    manifest_path = _write_artifact_manifest(run_dir)
    print(f"artifact_manifest={manifest_path}")


if __name__ == "__main__":
    main()
