#!/usr/bin/env python3
"""
Lab 08 —— 用 TRL 的 GRPOTrainer 复现 lab04

目的
----
lab04 从零手写了 GRPO 的每一行（rollout、组内优势、PPO 裁剪、归一化）。
本 lab 用同一份奖励函数（common/rewards.py）和同一份数据（GSM8K），
换成 HuggingFace TRL 的 GRPOTrainer，让你对比：
    · 手写版暴露的 12 个坑（padding 方向、EOS mask、归一化分母……）
      在框架里变成了哪几个配置项
    · 生产代码大概长什么样

TRL 不是从零实现，它是这个仓库里**唯一**依赖 RL 框架的 lab —— 目的正是
让你把 lab04 的理解拿来对照，而不是重新学一遍。

用法
----
    python labs/lab08_trl_grpo/train_grpo_trl.py --smoke              # 冒烟
    python labs/lab08_trl_grpo/train_grpo_trl.py                      # 默认配置
    python labs/lab08_trl_grpo/train_grpo_trl.py --model outputs/lab01_sft --total-steps 300

依赖
----
    pip install "trl>=0.12" —— 见 requirements-frameworks.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common import gpu
from common.data import GSM8K_SYSTEM, load_gsm8k, set_seed
from common.rewards import RewardConfig, compute_reward


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "outputs/lab08_trl_grpo"

    group_size: int = 8            # TRL 里叫 num_generations，含义和 lab04 的 G 完全一样
    per_device_prompts: int = 8    # 每个 device 每步几道题
    max_prompt_len: int = 320
    max_new_tokens: int = 384
    temperature: float = 1.0

    beta: float = 0.0              # KL 系数，语义同 lab04
    lr: float = 1e-6
    max_grad_norm: float = 1.0
    total_steps: int = 300

    seed: int = 0
    log_every: int = 1
    save_every: int = 50
    smoke: bool = False


# ==========================================================================
# 数据集：一行 = 一道题。TRL 支持「对话式 prompt」——
# 给一个 messages 列表，它会自动用 tokenizer 的 chat template 渲染，
# 效果等价于 lab04 里手写的 build_prompt()。
# ==========================================================================
def build_dataset(n: int | None, seed: int):
    from datasets import Dataset

    items = load_gsm8k("train", n=n, seed=seed)
    return Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": GSM8K_SYSTEM},
                {"role": "user", "content": it["question"]},
            ],
            "answer": it["answer"],
        }
        for it in items
    ])


# ==========================================================================
# 奖励函数：直接复用 common/rewards.py，和 lab04 完全一样的判分逻辑
# （格式门控 + 正确性 + 超长/复读软惩罚）。
#
# TRL 的约定：reward_funcs 接收 completions（补全内容列表）和数据集里
# 除 prompt 外的其余列（这里是 answer），必须原样透传同名的关键字参数。
# ==========================================================================
def gsm8k_reward(completions, answer, **kwargs) -> list[float]:
    rcfg = RewardConfig()
    rewards = []
    for completion, gt in zip(completions, answer):
        # 对话式 prompt 下 completions 是 [{"role": "assistant", "content": "..."}]
        text = completion[-1]["content"] if isinstance(completion, list) else completion
        r, _ = compute_reward(text, gt, rcfg)
        rewards.append(r)
    return rewards


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise SystemExit(
            "没装 trl。先跑：pip install -r labs/lab08_trl_grpo/requirements-trl.txt"
        ) from e

    print(f"[env] device={gpu.pick_device()} dtype={gpu.pick_dtype()}")
    gpu.assert_enough_vram(14, "Lab 08 (TRL GRPO, 0.5B)")

    dataset = build_dataset(n=None, seed=cfg.seed)
    print(f"[data] 训练题库 {len(dataset)} 道")

    # ---- GRPOConfig 里能对上 lab04 Config 的字段，注释标出对应关系 ----
    args = GRPOConfig(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_prompts * cfg.group_size,
        num_generations=cfg.group_size,            # ↔ lab04 的 group_size
        max_prompt_length=cfg.max_prompt_len,
        max_completion_length=cfg.max_new_tokens,   # ↔ lab04 的 max_new_tokens
        temperature=cfg.temperature,
        beta=cfg.beta,                              # ↔ lab04 的 beta（KL 系数）
        learning_rate=cfg.lr,                       # ↔ lab04 的 lr（同样要比 SFT 小 10~20 倍）
        max_grad_norm=cfg.max_grad_norm,
        max_steps=cfg.total_steps,
        logging_steps=cfg.log_every,
        save_steps=cfg.save_every,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=cfg.seed,
    )

    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=gsm8k_reward,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(os.path.join(cfg.out_dir, "final"))
    print(f"\n下一步：python labs/lab06_eval_passk/evaluate.py --model "
          f"{os.path.join(cfg.out_dir, 'final')} --n 200")


def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in vars(cfg).items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(val, bool):
            ap.add_argument(flag, action="store_true", default=val)
        else:
            ap.add_argument(flag, type=type(val), default=val)
    ns = ap.parse_args()
    for k in vars(cfg):
        setattr(cfg, k, getattr(ns, k))
    if cfg.smoke:
        cfg.total_steps, cfg.group_size, cfg.per_device_prompts = 3, 4, 2
        cfg.max_new_tokens, cfg.save_every, cfg.log_every = 96, 999, 1
        cfg.out_dir += "_smoke"
    return cfg


if __name__ == "__main__":
    train(parse_args())
