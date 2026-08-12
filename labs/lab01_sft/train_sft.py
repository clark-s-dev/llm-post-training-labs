#!/usr/bin/env python3
"""
Lab 01 —— 从零实现 SFT（监督微调）

目标
----
把 Qwen2.5-0.5B-Instruct 微调成「先在 <think></think> 里推理、再用 \\boxed{} 给答案」的格式。
这一步不是为了提升数学能力（SFT 学不到新能力），而是为了**把输出格式钉死**，
好让 lab04 的 GRPO 一开始就有合规的输出可以打分。

不用任何 Trainer 封装，整个训练循环都写出来，这样你能看清每一步在做什么。

对应教程章节：第 03 章「SFT 监督微调全解」

本 lab 演示的三个关键点
--------------------
1. loss mask —— 只对 assistant 的回答内容算损失（prompt 和角色头都要屏蔽）
2. 梯度累积下的**正确归一化** —— 这是一个真实存在过的框架级 bug（见 --naive-accum）
3. LoRA 与全参微调的显存/效果对比

L4 (24GB) 参考耗时
-----------------
    全参微调  2000 样本 × 2 epoch ≈ 8 分钟，峰值显存约 11 GB
    LoRA      同上                ≈ 6 分钟，峰值显存约 5 GB

用法
----
    python labs/lab01_sft/train_sft.py --smoke              # 3 步冒烟测试（1 分钟）
    python labs/lab01_sft/train_sft.py                      # 完整训练
    python labs/lab01_sft/train_sft.py --lora               # 用 LoRA
    python labs/lab01_sft/train_sft.py --naive-accum        # 复现梯度累积 bug（教学用）
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import gpu
from common.data import (IGNORE_INDEX, build_sft_example, collate_sft, describe_example,
                         gsm8k_to_sft_messages, load_gsm8k, sanity_check_example, set_seed)
from common.train_utils import (Logger, build_optimizer, build_scheduler, cfg_to_dict,
                                clip_and_step, save_checkpoint)


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "outputs/lab01_sft"

    # ---- 数据 ----
    n_train: int = 2000            # GSM8K 训练样本数（全量 7473）
    n_eval: int = 200
    max_len: int = 640             # 覆盖 GSM8K 绝大多数样本；太小会截断掉 EOS！

    # ---- 优化 ----
    lr: float = 2e-5               # 全参 SFT 的典型值。LoRA 会自动放大到 10 倍
    epochs: int = 2                # SFT 极易过拟合，2~3 轮足够
    micro_bs: int = 4              # 单次前向的样本数，OOM 时先调小它
    accum: int = 4                 # 梯度累积 → 等效 batch = 4×4 = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ---- LoRA ----
    lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64

    # ---- 其他 ----
    seed: int = 0
    eval_every: int = 50
    log_every: int = 5
    naive_accum: bool = False      # True = 故意用错误的归一化方式（教学演示）
    smoke: bool = False
    wandb: bool = False


# ==========================================================================
# 损失
# ==========================================================================
def sft_loss_sum(model, batch: dict) -> tuple[torch.Tensor, int]:
    """
    返回 (这个 micro-batch 的**交叉熵总和**, 参与计算的 token 数)。

    ★ 注意这里返回的是 **sum** 而不是 mean。原因见下面 train() 里的说明 ——
      只有拿到 sum，才能在梯度累积时用「全局 token 数」做正确的归一化。
    """
    # 左移对齐：位置 i 的 logits 预测位置 i+1 的 token
    logits = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"],
                   use_cache=False).logits          # [B, L, V]
    shift_logits = logits[:, :-1, :]                # [B, L-1, V]
    shift_labels = batch["labels"][:, 1:]           # [B, L-1]

    # reduction="sum" + ignore_index=-100：
    #   被屏蔽的位置（prompt / padding）不产生任何损失，也不计入分母
    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),   # ★ .float()：bf16 精度不够
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    n_tokens = int((shift_labels != IGNORE_INDEX).sum())
    return loss_sum, n_tokens


# ==========================================================================
# 训练
# ==========================================================================
def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device, dtype = gpu.pick_device(), gpu.pick_dtype()
    print(f"[env] device={device} dtype={dtype} attn={gpu.pick_attn_impl()}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=dtype, attn_implementation=gpu.pick_attn_impl(),
    ).to(device)

    # gradient checkpointing：用时间换显存（重算激活值而不是存下来）
    # 大约慢 30%，但激活显存降到 1/sqrt(L)。0.5B 模型上不开也行，开着更安全。
    model.gradient_checkpointing_enable()
    model.config.use_cache = False        # ★ 必须关，否则和 checkpointing 冲突

    if cfg.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            # ★ 一定要包含 MLP 的三个投影。只加 q_proj/v_proj 是 2021 年的老做法，
            #   MLP 占了模型 2/3 的参数，不动它等于放弃大部分容量。
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
        model.print_trainable_parameters()
        cfg.lr = cfg.lr * 10           # ★ LoRA 的学习率要比全参大约 10 倍

    # ---------------- 数据 ----------------
    n_tr = 32 if cfg.smoke else cfg.n_train
    n_ev = 16 if cfg.smoke else cfg.n_eval
    train_items = load_gsm8k("train", n=n_tr, seed=cfg.seed)
    eval_items = load_gsm8k("test", n=n_ev, seed=cfg.seed)

    def to_example(item):
        return build_sft_example(tok, gsm8k_to_sft_messages(item), cfg.max_len)

    train_ds = [to_example(x) for x in train_items]
    eval_ds = [to_example(x) for x in eval_items]

    # ★★★ 开跑前必做：肉眼确认第一条样本的 mask 是对的 ★★★
    print("\n" + "=" * 72)
    print("第一条训练样本（务必确认绿色部分正好是你想让模型学的内容）：")
    sanity_check_example(tok, train_ds[0], verbose=False)
    print(describe_example(tok, train_ds[0]))
    trunc = sum(1 for e in train_ds if e.labels[-1] == IGNORE_INDEX)
    if trunc:
        print(f"⚠️  有 {trunc}/{len(train_ds)} 条样本的最后一个 token 被屏蔽了 "
              f"—— 多半是被 max_len={cfg.max_len} 截断，模型会学不会停止。建议调大 max_len。")
    print("=" * 72 + "\n")

    collate = lambda b: collate_sft(b, tok.pad_token_id)
    loader = DataLoader(train_ds, batch_size=cfg.micro_bs, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.micro_bs, shuffle=False, collate_fn=collate)

    # ---------------- 优化器 ----------------
    steps_per_epoch = max(1, len(loader) // cfg.accum)
    total_steps = 3 if cfg.smoke else steps_per_epoch * cfg.epochs
    opt = build_optimizer(model, cfg.lr, cfg.weight_decay)
    sched = build_scheduler(opt, total_steps, cfg.warmup_ratio, "cosine")

    logger = Logger(cfg.out_dir, cfg.wandb, run_name="lab01-sft", config=cfg_to_dict(cfg))
    print(f"[train] {len(train_ds)} 条样本 | 等效 batch={cfg.micro_bs * cfg.accum} "
          f"| 共 {total_steps} 步 | lr={cfg.lr:g}")

    step, done = 0, False
    model.train()
    for epoch in range(cfg.epochs):
        it = iter(loader)
        while not done:
            # ---- 攒够 accum 个 micro-batch ----
            micro_batches = []
            for _ in range(cfg.accum):
                try:
                    micro_batches.append(next(it))
                except StopIteration:
                    break
            if not micro_batches:
                break

            gpu.reset_peak_mem()

            # ═══════════════════════════════════════════════════════════════
            # ★★★ 本 lab 的教学重点：梯度累积怎样归一化才是对的 ★★★
            #
            # 正确做法：分母是**这一整个全局 batch** 的有效 token 总数
            #     loss = Σ_all_micro_batches(交叉熵) / Σ_all_micro_batches(token 数)
            #
            # 错误做法（--naive-accum）：每个 micro-batch 先各自求平均，再对
            # micro-batch 求平均
            #     loss = (1/K) Σ_k [ Σ_t(交叉熵_k) / n_k ]
            #
            # 只有当所有 micro-batch 的 token 数完全相等时两者才等价。实际上
            # 它们不等（样本长度不同），于是 accum=1 和 accum=8 会训出不一样
            # 的模型 —— 这正是 2024 年底 HF Trainer / TRL / Axolotl 集体中招
            # 的那个 bug。同样的错误在 RL 的 loss 聚合里以完全相同的形式重现
            # （见教程第 07 章 DAPO 的 token-level loss）。
            # ═══════════════════════════════════════════════════════════════
            batch_tokens = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum())
                               for b in micro_batches)
            batch_tokens = max(1, batch_tokens)

            total_loss = 0.0
            for b in micro_batches:
                b = {k: v.to(device) for k, v in b.items()}
                loss_sum, n_tok = sft_loss_sum(model, b)

                if cfg.naive_accum:
                    loss = (loss_sum / max(1, n_tok)) / len(micro_batches)   # ❌ 错误
                else:
                    loss = loss_sum / batch_tokens                            # ✅ 正确

                loss.backward()
                total_loss += float(loss_sum.detach()) / batch_tokens

            gnorm = clip_and_step(model, opt, sched, cfg.max_grad_norm)
            step += 1

            _, peak = gpu.gpu_mem_gb()
            metrics = {
                "loss": total_loss,
                "ppl": min(1e4, torch.exp(torch.tensor(total_loss)).item()),
                "lr": sched.get_last_lr()[0],
                "gnorm": gnorm,
                "tokens": batch_tokens,
                "peak_gb": peak,
            }
            logger.log(step, metrics)
            logger.print_row(step, metrics, every=cfg.log_every)

            if step % cfg.eval_every == 0 or step == total_steps:
                ev = evaluate(model, eval_loader, device)
                logger.log(step, ev, prefix="eval/")
                print(f"      └─ eval loss {ev['loss']:.4f}  ppl {ev['ppl']:.2f}")
                model.train()

            if step >= total_steps:
                done = True
        if done:
            break

    # ---------------- 收尾 ----------------
    ev = evaluate(model, eval_loader, device)
    print(f"\n[final] eval loss={ev['loss']:.4f} ppl={ev['ppl']:.2f}")
    model.config.use_cache = True
    save_checkpoint(model, tok, cfg.out_dir, extra={"final_eval": ev, "steps": step})
    logger.close()

    print("\n" + "=" * 72)
    print("训练完成。下一步：")
    print("  1) 看看模型现在会不会用正确格式回答：")
    print(f"       python scripts/generate.py --model {cfg.out_dir}")
    print("  2) 量化评估它的准确率：")
    print(f"       python labs/lab06_eval_passk/evaluate.py --model {cfg.out_dir} --n 100")
    print("  3) 用它当 GRPO 的起点（推荐）：")
    print(f"       python labs/lab04_grpo/train_grpo.py --model {cfg.out_dir}")
    print("=" * 72)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """验证集损失。★ 这是唯一可靠的早停信号 —— 它开始上升就该停了。"""
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        loss_sum, n = sft_loss_sum(model, b)
        tot_loss += float(loss_sum)
        tot_tok += n
    loss = tot_loss / max(1, tot_tok)
    return {"loss": loss, "ppl": min(1e4, float(torch.exp(torch.tensor(loss))))}


def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in vars(cfg).items():
        if isinstance(val, bool):
            ap.add_argument(f"--{name.replace('_', '-')}", action="store_true", default=val)
        else:
            ap.add_argument(f"--{name.replace('_', '-')}", type=type(val), default=val)
    ns = ap.parse_args()
    default_out = cfg.out_dir
    for k in vars(cfg):
        setattr(cfg, k, getattr(ns, k))
    if cfg.smoke:                       # 冒烟模式：把一切都调到最小
        cfg.micro_bs, cfg.accum, cfg.eval_every, cfg.log_every = 2, 2, 3, 1
        # 只有用户没指定 --out-dir 时才自动加后缀，避免覆盖正式训练的结果
        if cfg.out_dir == default_out:
            cfg.out_dir += "_smoke"
    return cfg


if __name__ == "__main__":
    train(parse_args())
