"""
训练循环里反复用到的零件：优化器、学习率调度、日志、checkpoint。

这些东西每个 lab 都要用，抽出来避免复制粘贴。
每个函数里的默认值都是本仓库推荐的起点，注释说明了「为什么是这个值」。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, is_dataclass

import torch


# --------------------------------------------------------------------------
# 优化器
# --------------------------------------------------------------------------
def build_optimizer(model, lr: float, weight_decay: float = 0.0, betas=(0.9, 0.95)):
    """
    构造 AdamW，并把参数分成两组：

      · 二维及以上的权重矩阵      → 施加 weight decay
      · LayerNorm / bias / 一维   → **不施加** weight decay

    为什么要分组？对 LayerNorm 的 scale 和 bias 做衰减会把它们往 0 拉，
    损害模型表达能力。这是 GPT/LLaMA 系列训练脚本里的标准做法。

    betas=(0.9, 0.95) 而不是默认的 (0.9, 0.999)：
        大模型训练的惯例。RL 阶段梯度噪声大，β2 小一点响应更快；
        如果你发现 RL 训练震荡，可以把 β2 调到 0.99 让它更平滑。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name.lower() or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas,
        eps=1e-8,       # RLHF 场景有人推荐 1e-5（梯度很小时 1e-8 会让步长过于激进）
    )


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.03,
                    schedule: str = "cosine", min_lr_ratio: float = 0.1):
    """
    学习率调度：先线性 warmup，再按 cosine / linear / constant 衰减。

    为什么必须 warmup？
        优化器刚开始时 Adam 的一阶/二阶矩估计还不准，
        直接用峰值学习率会让第一步的更新方向很随机，
        足以破坏预训练权重（表现为 loss 先暴涨）。

    RL 阶段（lab04/lab07）建议用 schedule="constant"：
        RL 的总步数事先难以确定，cosine 衰减到一半停掉的话，
        实际用到的学习率轨迹跟你预期的完全不同。
    """
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup                        # 线性 warmup
        if schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        if schedule == "linear":
            return max(min_lr_ratio, 1.0 - progress)
        # cosine：平滑地从 1 降到 min_lr_ratio
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# 日志
# --------------------------------------------------------------------------
class Logger:
    """
    极简日志器：同时写终端、JSONL 文件，可选写 wandb。

    为什么要写 JSONL？训练崩了之后你需要一份完整记录来复盘。
    终端输出会被 tmux 滚掉，wandb 可能没配。文件永远在。
    """

    def __init__(self, out_dir: str, use_wandb: bool = False,
                 project: str = "llm-post-training-labs", run_name: str | None = None,
                 config: dict | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "metrics.jsonl")
        self.f = open(self.path, "a", encoding="utf-8")
        self.t0 = time.time()
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name, config=config or {})
                self.wandb = wandb
                print(f"[log] wandb 已启用: {wandb.run.url}")
            except Exception as e:
                print(f"[log] wandb 启用失败（继续用本地日志）: {e}")

        if config:
            with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(config, fh, ensure_ascii=False, indent=2, default=str)

    def log(self, step: int, metrics: dict, prefix: str = "") -> None:
        rec = {"step": step, "elapsed_s": round(time.time() - self.t0, 1)}
        rec.update({f"{prefix}{k}": _to_py(v) for k, v in metrics.items()})
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        if self.wandb:
            self.wandb.log(rec, step=step)

    def print_row(self, step: int, metrics: dict, every: int = 1) -> None:
        """终端里打一行紧凑的指标。"""
        if step % every:
            return
        parts = []
        for k, v in metrics.items():
            v = _to_py(v)
            parts.append(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}")
        print(f"step {step:5d} | " + " | ".join(parts), flush=True)

    def close(self) -> None:
        self.f.close()
        if self.wandb:
            self.wandb.finish()


def _to_py(v):
    """把 tensor / numpy 标量转成普通 Python 数，方便 json 序列化。"""
    if torch.is_tensor(v):
        return v.detach().float().mean().item() if v.numel() > 1 else v.item()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except Exception:
            return v
    return v


# --------------------------------------------------------------------------
# 保存 / 加载
# --------------------------------------------------------------------------
def save_checkpoint(model, tokenizer, out_dir: str, step: int | None = None,
                    extra: dict | None = None) -> str:
    """
    存成 HuggingFace 标准格式，之后可以直接
        AutoModelForCausalLM.from_pretrained(路径)
    加载，也能直接喂给 vLLM。
    """
    path = out_dir if step is None else os.path.join(out_dir, f"step_{step}")
    os.makedirs(path, exist_ok=True)
    # PEFT/LoRA 模型有自己的 save_pretrained，会只存 adapter，这里直接调用即可
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra:
        with open(os.path.join(path, "train_state.json"), "w", encoding="utf-8") as f:
            json.dump(extra, f, ensure_ascii=False, indent=2, default=str)
    print(f"[ckpt] 已保存到 {path}")
    return path


def cfg_to_dict(cfg) -> dict:
    return asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))


# --------------------------------------------------------------------------
# 梯度
# --------------------------------------------------------------------------
def clip_and_step(model, optimizer, scheduler, max_norm: float = 1.0) -> float:
    """
    梯度裁剪 + 优化器步进 + 清零。返回裁剪前的梯度范数。

    ★ 一定要把 grad_norm 打进日志。它是训练健康度最灵敏的指标之一：
      · 平稳的小值            → 正常
      · 偶发尖峰              → 数据里有脏样本
      · 持续飙升然后 loss=NaN → 学习率太大 / 数值溢出
    """
    gnorm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm
    )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(gnorm)
