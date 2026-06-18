"""
Pure-Triton Flash Attention forward for Ascend NPU.

This variant rebuilds the forward kernel around the Triton-Ascend fused
attention structure:

- block pointer based Q/K/V/O access
- persistent-style launch to cap large grid dispatch overhead
- profiling helpers and sweep utilities for tiling and launch count tuning

The forward path is intentionally standalone and does not implement backward.
"""

import csv
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


DEVICE = "npu"
RESULT_DIR_NAME = "result_dir_blockptr_persistent"

SUMMARY_KERNEL_COLUMNS = [
    "Duration(us)",
    "Wait Time(us)",
    "Block Num",
    "Mix Block Num",
    "aicore_time(us)",
    "aic_mac_ratio",
    "aic_scalar_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_fixpipe_ratio",
    "aiv_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
    "cube_utilization(%)",
]

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
        return {}

    summary = {"kernel_count": len(rows)}
    for col in SUMMARY_KERNEL_COLUMNS:
        summary[col] = _mean([_to_float(row.get(col)) for row in rows])
    return summary


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


def summarize_profile_output(root, preferred_op="_attn_fwd"):
    out_dir = _find_profiler_output(root)
    op_rows = _read_csv_rows(out_dir / "op_statistic.csv")
    kernel_rows = _read_csv_rows(out_dir / "kernel_details.csv")
    step_rows = _read_csv_rows(out_dir / "step_trace_time.csv")
    api_rows = _read_csv_rows(out_dir / "api_statistic.csv")

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

    return {
        "profiler_output": str(out_dir),
        "op_name": op_name,
        "op": op_summary,
        "kernel": _summarize_kernel_details(kernel_rows, op_name),
        "step": _summarize_step(step_rows),
        "api": _summarize_api(api_rows),
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
        print(
            "  kernel duration/wait/block/mix: "
            f"{_fmt(kernel.get('Duration(us)'))} / {_fmt(kernel.get('Wait Time(us)'))} / "
            f"{_fmt(kernel.get('Block Num'))} / {_fmt(kernel.get('Mix Block Num'))}"
        )
        print(
            "  cube/mac/scalar/aiv_vec: "
            f"{_fmt(kernel.get('cube_utilization(%)'))} / {_fmt(kernel.get('aic_mac_ratio'))} / "
            f"{_fmt(kernel.get('aic_scalar_ratio'))} / {_fmt(kernel.get('aiv_vec_ratio'))}"
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


def _resolve_launch_schedule(z, h, n_ctx, bm):
    num_tiles_m = triton.cdiv(n_ctx, bm)
    total_tiles = num_tiles_m * z * h

    max_programs = int(os.environ.get("MAX_LAUNCHED_PROGRAMS", "65535"))
    if max_programs < 1:
        raise ValueError("MAX_LAUNCHED_PROGRAMS must be >= 1")

    requested = os.environ.get("PERSISTENT_BLOCKS")
    if requested is None:
        default_programs = int(os.environ.get("DEFAULT_LAUNCHED_PROGRAMS", "20"))
        if default_programs < 1:
            raise ValueError("DEFAULT_LAUNCHED_PROGRAMS must be >= 1")
        launched_programs = min(total_tiles, default_programs, max_programs)
    else:
        launched_programs = int(requested)
        if launched_programs < 1:
            raise ValueError("PERSISTENT_BLOCKS must be >= 1")
        launched_programs = min(total_tiles, launched_programs, max_programs)

    return {
        "num_tiles_m": num_tiles_m,
        "total_tiles": total_tiles,
        "launched_programs": launched_programs,
    }


@triton.jit
def _attn_fwd_inner(
    acc_ptr,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
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

    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    row = tl.arange(0, BLOCK_M)[:, None]
    col_head_dim = tl.arange(0, HEAD_DIM)[None, :]
    block2d_acc = row * HEAD_DIM + col_head_dim
    offs_m_cmp = offs_m.to(tl.float32)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_n = start_n + offs_n
        curr_n_cmp = curr_n.to(tl.float32)

        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

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
        pv = tl.dot(p_cast, v)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        if HEAD_DIM < 256:
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)
        else:
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
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc_ptr, l_i, m_i


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
        task_hz_idx = linear_tile // num_tiles_m
        task_m_idx = linear_tile - task_hz_idx * num_tiles_m
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
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
                ((off_z.to(tl.int64) * H + off_h.to(tl.int64)) * N_CTX + task_m_idx * BLOCK_M)
                * HEAD_DIM
            )
            acc_ptr = acc + acc_offset

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        if STAGE & 1:
            acc_ptr, l_i, m_i = _attn_fwd_inner(
                acc_ptr,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
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
                K_block_ptr,
                V_block_ptr,
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
            block2d_acc = row * HEAD_DIM + col_head_dim
            accumulator = tl.load(acc_ptr + block2d_acc)
            accumulator = accumulator / l_i[:, None]

        m_ptrs = M + task_hz_idx * N_CTX + offs_m
        tl.store(m_ptrs, m_i.to(tl.float32), mask=row_mask)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty), boundary_check=(0, 1))


def _launch_attention(q, k, v, causal, sm_scale, bm, bn):
    head_dim_q, head_dim_k = q.shape[-1], k.shape[-1]
    head_dim_v = v.shape[-1]
    assert head_dim_q == head_dim_k and head_dim_k == head_dim_v
    assert head_dim_k in {16, 32, 64, 128, 256}

    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    stage = 3 if causal else 1
    launch = _resolve_launch_schedule(q.shape[0], q.shape[1], q.shape[2], bm)
    grid = (launch["launched_programs"], 1, 1)

    if head_dim_k < 256:
        acc = torch.empty((1,), dtype=torch.float32, device=q.device)
    else:
        acc = torch.zeros((q.shape[0], q.shape[1], q.shape[2], head_dim_k), dtype=torch.float32, device=q.device)

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
        q.shape[0],
        q.shape[1],
        N_CTX=q.shape[2],
        HEAD_DIM=head_dim_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        STAGE=stage,
        debug=True,
    )
    return out, lse, launch


def attention_persistent(q, k, v, causal, sm_scale, bm, bn, return_lse=False):
    out, lse, _ = _launch_attention(q, k, v, causal, sm_scale, bm, bn)
    if return_lse:
        return out, lse
    return out


class attention:
    @staticmethod
    def forward(q, k, v, causal, sm_scale, return_lse=False):
        z, h, n_ctx, head_dim = q.shape
        bm, bn = get_tiling(z, h, n_ctx, head_dim, causal)
        return attention_persistent(q, k, v, causal, sm_scale, bm, bn, return_lse=return_lse)


def get_tiling(Z, H, N_CTX, HEAD_DIM, causal):
    override_bm = os.environ.get("OVERRIDE_BM")
    override_bn = os.environ.get("OVERRIDE_BN")
    if override_bm and override_bn:
        return int(override_bm), int(override_bn)

    defaults = {
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
    exact = defaults.get((Z, H, N_CTX, HEAD_DIM, causal))
    if exact is not None:
        return exact

    if causal:
        if HEAD_DIM >= 128:
            return (64, 32) if N_CTX >= 2048 else (32, 32)
        if N_CTX >= 8192:
            return 128, 32
        if N_CTX >= 2048:
            return 64, 32
        return 64, 32

    if HEAD_DIM >= 256:
        return 32, 32
    if HEAD_DIM >= 128:
        return (64, 32) if N_CTX >= 2048 else (32, 32)
    if N_CTX >= 8192:
        return 128, 32
    if N_CTX >= 2048:
        return 64, 64
    if N_CTX >= 1024:
        return (64, 64) if HEAD_DIM <= 64 else (64, 32)
    return (64, 32) if HEAD_DIM <= 64 else (32, 32)


def profile_once(Z, H, N_CTX, HEAD_DIM, causal, dtype, q, k, v, sm_scale):
    bm, bn = get_tiling(Z, H, N_CTX, HEAD_DIM, causal)
    _, _, launch = _launch_attention(q, k, v, causal, sm_scale, bm, bn)
    print("Triton implementation: block_ptr + persistent")
    print(f"Triton launch config: BM={bm}, BN={bn}, launched_programs={launch['launched_programs']}")
    torch.npu.synchronize()
    return attention_persistent(q, k, v, causal, sm_scale, bm, bn).to(dtype)


def profiling(Z, H, N_CTX, HEAD_DIM, causal, dtype, q, k, v, sm_scale):
    bm, bn = get_tiling(Z, H, N_CTX, HEAD_DIM, causal)
    launch = _resolve_launch_schedule(Z, H, N_CTX, bm)
    print("Triton implementation: block_ptr + persistent")
    print(f"Triton launch config: BM={bm}, BN={bn}, launched_programs={launch['launched_programs']}")

    result_dir = Path(os.getcwd()) / RESULT_DIR_NAME
    if result_dir.exists():
        shutil.rmtree(result_dir)

    active = 30
    total_steps = 1 + 1 + 1 + active
    activities = _build_profiler_activities()
    metric_plan = _build_metric_plan()
    if not metric_plan:
        raise RuntimeError("No available AiCMetrics found in torch_npu.profiler.AiCMetrics.")

    summary_by_metric = {}
    for metric_name, metric_value in metric_plan:
        metric_result_dir = result_dir / metric_name
        experimental_config = _make_experimental_config(metric_value)
        print(f"Profiling metric: {metric_name}")

        with torch_npu.profiler.profile(
            activities=activities,
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=active, repeat=1, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(metric_result_dir / "triton")),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(total_steps):
                tri_out = attention_persistent(q, k, v, causal, sm_scale, bm, bn).to(dtype)
                torch.npu.synchronize()
                prof.step()

        summary_by_metric[metric_name] = summarize_profile_output(metric_result_dir, preferred_op="_attn_fwd")

    summary_json, summary_txt = write_profile_summary_files(result_dir, summary_by_metric)
    print("Profiling complete. Results saved to:", result_dir)
    print("Machine-readable summary:", summary_json)
    print("Text summary:", summary_txt)
    print_profile_summary(summary_by_metric)
    print(f"Inspect each metric subdirectory under {RESULT_DIR_NAME}/*/triton for full profiler output.")


def _load_pipe_summary():
    summary_path = Path(os.getcwd()) / RESULT_DIR_NAME / "profile_summary.json"
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    pipe = summary["PipeUtilization"]
    op = pipe.get("op", {})
    kernel = pipe.get("kernel", {})
    step = pipe.get("step", {})
    api = pipe.get("api", {})
    launch = api.get("launch", {})
    return {
        "avg_us": op.get("Avg Time(us)"),
        "min_us": op.get("Min Time(us)"),
        "max_us": op.get("Max Time(us)"),
        "block_num": kernel.get("Block Num"),
        "mix_block_num": kernel.get("Mix Block Num"),
        "wait_us": kernel.get("Wait Time(us)"),
        "cube_pct": kernel.get("cube_utilization(%)"),
        "aic_mac_ratio": kernel.get("aic_mac_ratio"),
        "aic_scalar_ratio": kernel.get("aic_scalar_ratio"),
        "aiv_vec_ratio": kernel.get("aiv_vec_ratio"),
        "preparing_us": step.get("Preparing"),
        "launch_avg_us": launch.get("Avg(us)"),
    }


def _write_sweep_reports(sweep_dir, title, shape_desc, rows, best_key):
    csv_path = sweep_dir / "sweep_summary.csv"
    md_path = sweep_dir / "sweep_summary.md"

    fieldnames = list(rows[0].keys()) if rows else []
    if fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    successful_rows = [row for row in rows if row["status"] == "ok" and row[best_key] is not None]
    best = min(successful_rows, key=lambda row: row[best_key]) if successful_rows else None

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(shape_desc)
    lines.append("")

    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            values = []
            for key in headers:
                value = row[key]
                if isinstance(value, (int, str)) or value is None:
                    formatted = value if isinstance(value, str) else _fmt(value)
                else:
                    formatted = _fmt(value)
                if formatted is None:
                    formatted = ""
                values.append(str(formatted).replace("|", "/"))
            lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    if best is not None:
        best_desc = ", ".join(f"{key}={best[key]}" for key in rows[0].keys() if key in ("bm", "bn", "persistent_blocks"))
        lines.append(f"Best by {best_key}: {best_desc}, {best_key}={_fmt(best[best_key])}")
    else:
        lines.append(f"Best by {best_key}: no successful candidate")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return csv_path, md_path


def sweep_persistent_blocks(Z, H, N_CTX, HEAD_DIM, causal, dtype, sm_scale, block_candidates):
    original_metrics = os.environ.get("PROFILE_METRICS")
    original_requested = os.environ.get("PERSISTENT_BLOCKS")
    os.environ["PROFILE_METRICS"] = "PipeUtilization"

    bm, bn = get_tiling(Z, H, N_CTX, HEAD_DIM, causal)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

    sweep_dir = Path(os.getcwd()) / "persistent_block_sweep"
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for candidate in block_candidates:
        os.environ["PERSISTENT_BLOCKS"] = str(candidate)
        resolved = _resolve_launch_schedule(Z, H, N_CTX, bm)
        row = {
            "persistent_blocks": candidate,
            "launched_programs": resolved["launched_programs"],
            "status": "ok",
            "error": "",
            "avg_us": None,
            "min_us": None,
            "max_us": None,
            "block_num": None,
            "mix_block_num": None,
            "wait_us": None,
            "cube_pct": None,
            "aic_mac_ratio": None,
            "aic_scalar_ratio": None,
            "aiv_vec_ratio": None,
            "preparing_us": None,
            "launch_avg_us": None,
        }
        print(f"\n=== Sweep persistent_blocks={candidate} (resolved={resolved['launched_programs']}) ===")
        try:
            profiling(Z, H, N_CTX, HEAD_DIM, causal, dtype, q, k, v, sm_scale)
            row.update(_load_pipe_summary())
            candidate_dir = sweep_dir / f"persistent_blocks_{candidate}"
            shutil.copytree(Path(os.getcwd()) / RESULT_DIR_NAME, candidate_dir)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc).splitlines()[0][:240]
            print(f"Sweep persistent_blocks={candidate} failed: {row['error']}")
        rows.append(row)

    csv_path, md_path = _write_sweep_reports(
        sweep_dir,
        "Persistent Block Sweep",
        f"Target shape: `{Z} x {H} x {N_CTX} x {HEAD_DIM}`, causal=`{causal}`, dtype=`{dtype}`, BM=`{bm}`, BN=`{bn}`",
        rows,
        "avg_us",
    )

    if original_metrics is None:
        os.environ.pop("PROFILE_METRICS", None)
    else:
        os.environ["PROFILE_METRICS"] = original_metrics

    if original_requested is None:
        os.environ.pop("PERSISTENT_BLOCKS", None)
    else:
        os.environ["PERSISTENT_BLOCKS"] = original_requested

    print(f"Persistent block sweep summary CSV: {csv_path}")
    print(f"Persistent block sweep summary Markdown: {md_path}")
    return rows


def sweep_block_tiling(Z, H, N_CTX, HEAD_DIM, causal, dtype, sm_scale, tiling_candidates):
    original_metrics = os.environ.get("PROFILE_METRICS")
    original_bm = os.environ.get("OVERRIDE_BM")
    original_bn = os.environ.get("OVERRIDE_BN")
    os.environ["PROFILE_METRICS"] = "PipeUtilization"

    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

    sweep_dir = Path(os.getcwd()) / "persistent_tiling_sweep"
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for bm, bn in tiling_candidates:
        os.environ["OVERRIDE_BM"] = str(bm)
        os.environ["OVERRIDE_BN"] = str(bn)
        print(f"\n=== Sweep BM={bm}, BN={bn} ===")
        row = {
            "bm": bm,
            "bn": bn,
            "status": "ok",
            "error": "",
            "avg_us": None,
            "min_us": None,
            "max_us": None,
            "block_num": None,
            "mix_block_num": None,
            "wait_us": None,
            "cube_pct": None,
            "aic_mac_ratio": None,
            "aic_scalar_ratio": None,
            "aiv_vec_ratio": None,
            "preparing_us": None,
            "launch_avg_us": None,
        }
        try:
            profiling(Z, H, N_CTX, HEAD_DIM, causal, dtype, q, k, v, sm_scale)
            row.update(_load_pipe_summary())
            tiling_dir = sweep_dir / f"bm_{bm}_bn_{bn}"
            shutil.copytree(Path(os.getcwd()) / RESULT_DIR_NAME, tiling_dir)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc).splitlines()[0][:240]
            print(f"Sweep BM={bm}, BN={bn} failed: {row['error']}")
        rows.append(row)

    csv_path, md_path = _write_sweep_reports(
        sweep_dir,
        "Persistent Tiling Sweep",
        f"Target shape: `{Z} x {H} x {N_CTX} x {HEAD_DIM}`, causal=`{causal}`, dtype=`{dtype}`",
        rows,
        "avg_us",
    )

    if original_metrics is None:
        os.environ.pop("PROFILE_METRICS", None)
    else:
        os.environ["PROFILE_METRICS"] = original_metrics

    if original_bm is None:
        os.environ.pop("OVERRIDE_BM", None)
    else:
        os.environ["OVERRIDE_BM"] = original_bm

    if original_bn is None:
        os.environ.pop("OVERRIDE_BN", None)
    else:
        os.environ["OVERRIDE_BN"] = original_bn

    print(f"Tiling sweep summary CSV: {csv_path}")
    print(f"Tiling sweep summary Markdown: {md_path}")
    return rows


def _torch_attention_reference(q, k, v, causal, sm_scale):
    qf = q.float()
    kf = k.float()
    vf = v.float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale
    if causal:
        seq_q = q.shape[-2]
        seq_k = k.shape[-2]
        causal_mask = torch.triu(
            torch.ones((seq_q, seq_k), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).to(q.dtype)


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN",
    [
        (1, 1, 64, 64, False, torch.float16, 64, 16),
        (1, 1, 64, 64, True, torch.float16, 64, 16),
        (1, 1, 96, 64, False, torch.float16, 64, 32),
        (1, 1, 128, 128, False, torch.float16, 32, 32),
        (1, 2, 1024, 64, False, torch.float16, 64, 32),
    ],
)
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    if causal and BM < BN:
        pytest.skip("Causal path expects BLOCK_M >= BLOCK_N.")

    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

    sm_scale = 0.5

    atten_mask = None
    sparse_mode = 0
    pre_tockens = 65535
    next_tockens = 65535
    if causal:
        # `sparse_mode=2` requires a compressed 2048x2048 causal mask on this torch_npu build.
        # Keep the reference path on defaultMask mode so smaller causal shapes can still run.
        atten_mask = torch.triu(
            torch.ones((N_CTX, N_CTX), device=DEVICE, dtype=torch.bool),
            diagonal=1,
        )
        pre_tockens = N_CTX
        next_tockens = 0

    try:
        ref_out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            H,
            padding_mask=None,
            atten_mask=atten_mask,
            scale=sm_scale,
            keep_prob=1.0,
            input_layout="BNSD",
            pre_tockens=pre_tockens,
            next_tockens=next_tockens,
            sparse_mode=sparse_mode,
        )[0]
    except RuntimeError as exc:
        print(f"npu_fusion_attention reference unavailable, falling back to torch reference: {exc}")
        ref_out = _torch_attention_reference(q, k, v, causal, sm_scale)

    tri_out = attention_persistent(q, k, v, causal, sm_scale, BM, BN).to(dtype)
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM = 128, 8, 1024, 128
    causal, dtype = True, torch.float16
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

    profiling(Z, H, N_CTX, HEAD_DIM, causal, dtype, q, k, v, sm_scale=0.5)
