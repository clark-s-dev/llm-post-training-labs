"""
硬件与精度选择工具。

为什么单独抽一个文件出来？
    每个 lab 都要回答同样三个问题：跑在哪块设备上、用什么精度、用哪种 attention 实现。
    这三件事在不同显卡上答案不同，写死会导致换机器就跑不了。这里统一处理。

针对 NVIDIA L4 的说明（本仓库的目标硬件）：
    · 架构 Ada Lovelace，compute capability = 8.9
    · 显存 24 GB —— 本仓库所有 lab 都按这个上限设计
    · 原生支持 bfloat16（不需要像 T4/V100 那样退回 fp16 + GradScaler）
    · 显存带宽约 300 GB/s，比 A100（~2 TB/s）低不少。
      自回归生成是「带宽瓶颈」型负载，所以 L4 上 rollout 会明显偏慢，
      这也是本仓库默认用 0.5B 模型、并把生成长度设得比较短的原因。
"""

from __future__ import annotations

import os
import platform
import subprocess

import torch


# --------------------------------------------------------------------------
# 设备与精度
# --------------------------------------------------------------------------
def pick_device() -> str:
    """按优先级挑一个可用设备：CUDA > Apple MPS > CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str | None = None) -> torch.dtype:
    """
    挑训练用的浮点精度。

    bfloat16 vs float16 的区别：
        两者都是 16 位，但 bf16 的指数位和 fp32 一样多（8 位），
        所以动态范围和 fp32 相同，几乎不会溢出 —— 训练时不需要 loss scaling。
        代价是尾数位少（7 位），精度低。对深度学习来说这个取舍非常划算。
        Ada（L4）、Ampere（A100）、Hopper（H100）都原生支持 bf16。
    """
    device = device or pick_device()
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "cuda":
        return torch.float16          # 老卡（如 T4）只能用 fp16
    return torch.float32              # CPU / MPS 上直接用 fp32，省心


def pick_attn_impl() -> str:
    """
    挑 attention 实现。

    "flash_attention_2"：最快、最省显存，但需要单独编译安装 flash-attn，
                         版本不匹配时非常容易装挂。
    "sdpa"             ：PyTorch 内置的 scaled_dot_product_attention，
                         底层同样会调用 FlashAttention / Memory-Efficient 内核，
                         零安装成本。**本仓库默认用它。**

    想强制指定可以设环境变量：ATTN_IMPL=flash_attention_2 python train.py
    """
    forced = os.environ.get("ATTN_IMPL")
    if forced:
        return forced
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


# --------------------------------------------------------------------------
# 显存
# --------------------------------------------------------------------------
def gpu_mem_gb() -> tuple[float, float]:
    """返回 (已分配 GB, 峰值已分配 GB)。只有 CUDA 有意义。"""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024 ** 3,
        torch.cuda.max_memory_allocated() / 1024 ** 3,
    )


def reset_peak_mem() -> None:
    """把峰值显存统计清零。每个训练步开头调一次，就能测出单步峰值。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def total_vram_gb() -> float:
    """这块卡一共有多少显存（GB）。L4 应该返回约 22.x（24GB 标称，扣掉保留部分）。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def assert_enough_vram(need_gb: float, what: str = "这个 lab") -> None:
    """
    显存不够时早点报错，而不是训练跑了 5 分钟才 OOM。
    只在 CUDA 上检查；CPU/MPS 直接放行（会很慢，但能跑通逻辑）。
    """
    if not torch.cuda.is_available():
        return
    have = total_vram_gb()
    if have < need_gb:
        raise RuntimeError(
            f"{what} 大约需要 {need_gb:.0f} GB 显存，当前这块卡只有 {have:.1f} GB。\n"
            f"可以试试：减小 --micro-bs、减小 --max-new-tokens、或换更小的模型。"
        )


# --------------------------------------------------------------------------
# 环境摘要（lab00 会调用，出问题时贴这段给别人看最省事）
# --------------------------------------------------------------------------
def describe_env() -> str:
    lines = [
        f"Python            : {platform.python_version()}  ({platform.machine()})",
        f"PyTorch           : {torch.__version__}",
        f"CUDA available    : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"
        lines += [
            f"CUDA runtime      : {torch.version.cuda}",
            f"GPU               : {props.name}",
            f"Compute capability: sm_{props.major}{props.minor}  (cc {cc})",
            f"VRAM              : {props.total_memory / 1024 ** 3:.1f} GB",
            f"bfloat16 支持     : {torch.cuda.is_bf16_supported()}",
            f"GPU 数量          : {torch.cuda.device_count()}",
        ]
        # L4 = sm_89。给个明确提示，避免用户在别的卡上照抄本仓库的超参。
        if (props.major, props.minor) == (8, 9):
            lines.append("→ 检测到 Ada 架构（L4 / L40S / RTX 40 系），本仓库默认参数就是按这个调的 ✓")
        else:
            lines.append(f"→ 注意：本仓库默认参数是按 L4（sm_89, 24GB）调的，当前是 sm_{props.major}{props.minor}，可能需要调整 batch size")
    else:
        lines.append("→ 没有检测到 CUDA。所有 lab 仍可在 CPU 上跑通逻辑（加 --smoke），但会非常慢。")

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            lines.append(f"nvidia-smi        : {smi.stdout.strip()}")
    except Exception:
        pass

    lines.append(f"选定设备/精度     : {pick_device()} / {pick_dtype()}")
    lines.append(f"选定 attention    : {pick_attn_impl()}")
    return "\n".join(lines)
