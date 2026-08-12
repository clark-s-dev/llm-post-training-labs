#!/usr/bin/env python3
"""
Lab 02 —— 从零训练一个奖励模型（Reward Model）

目标
----
把「人类的成对偏好」变成一个可微的打分函数 r_φ(x, y)。
这是经典 RLHF（第 05 章）的第二步，也是理解 DPO 推导的前提。

核心数学：Bradley-Terry 模型
--------------------------
假设每个回答有一个潜在「实力」分数，那么 A 打败 B 的概率是

    P(y_w ≻ y_l | x) = σ( r(x, y_w) − r(x, y_l) )

对观察到的偏好做最大似然估计，取负对数就得到损失：

    L = − E[ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]

直觉：让「好回答的分」减「坏回答的分」尽可能大。
     分差为 0 时 loss = log2 ≈ 0.693（等于瞎猜）；分差 → +∞ 时 loss → 0。

★ 关键性质：这个损失**只约束分差，不约束绝对值**。
  所有分数同时 +100，损失完全不变。这带来两个后果：
    1. 训练时分数可能整体漂移到很大的数 → 需要加 L2 正则（本文件有）
    2. "reward = 3.5" 这个数本身没有跨模型可比性
  这个「平移不变性」正是 DPO 推导中配分函数 Z(x) 能被消掉的原因（见第 06 章）。

对应教程章节：第 05 章 5.3 / 5.4

L4 (24GB) 参考耗时
-----------------
    2000 对偏好数据 × 1 epoch ≈ 6 分钟，峰值显存约 9 GB

用法
----
    python labs/lab02_reward_model/train_rm.py --smoke
    python labs/lab02_reward_model/train_rm.py --n-train 4000
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import gpu
from common.data import build_pair_tensors, load_preference_data, pad_stack, set_seed
from common.train_utils import Logger, build_optimizer, build_scheduler, cfg_to_dict, clip_and_step


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "outputs/lab02_rm"
    n_train: int = 2000
    n_eval: int = 300
    max_len: int = 640
    max_prompt: int = 320

    lr: float = 5e-6               # ★ 比 SFT 小。RM 极易过拟合
    epochs: int = 1                # ★ 只训 1 轮！第 2 轮验证准确率就开始掉了
    micro_bs: int = 2              # 每条样本要过两遍（chosen + rejected），所以比 SFT 小
    accum: int = 8
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    reg_coef: float = 1e-3         # 分数绝对值的 L2 正则，防止整体漂移

    seed: int = 0
    log_every: int = 5
    eval_every: int = 50
    smoke: bool = False
    wandb: bool = False


# ==========================================================================
# 模型：LM 主干 + 一个标量头
# ==========================================================================
class RewardModel(nn.Module):
    """
    结构 = 预训练 LM 去掉 lm_head（那个 [hidden, 151936] 的大矩阵），
           换成一个 [hidden, 1] 的线性层，输出一个标量分数。

    为什么从 Instruct 模型初始化而不是 base？
        RM 要判断「什么是好回答」，Instruct 模型已经有这个先验了。
        实践中从 SFT 后的模型初始化效果最好。
    """

    def __init__(self, model_name: str, dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel
        # AutoModel（而不是 AutoModelForCausalLM）加载的就是不带 lm_head 的主干
        self.backbone = AutoModel.from_pretrained(
            model_name, dtype=dtype, attn_implementation=gpu.pick_attn_impl()
        )
        h = self.backbone.config.hidden_size
        self.v_head = nn.Linear(h, 1, bias=False)
        # ★ 小方差初始化。用默认初始化的话，初始 reward 的方差会很大，
        #   训练前几十步极不稳定（loss 剧烈震荡）。
        nn.init.normal_(self.v_head.weight, std=1.0 / (h + 1) ** 0.5)

    def forward(self, input_ids, attention_mask):
        hs = self.backbone(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state   # [B, L, H]

        # ★ 取「最后一个非 padding token」的隐状态作为整条序列的表示。
        #   常见 bug：直接写 hs[:, -1]。右 padding 时那取到的是 pad token，
        #   等于对着空气打分，训练完全学不动（而且不会报错）。
        last_idx = attention_mask.sum(dim=1) - 1                              # [B]
        pooled = hs[torch.arange(hs.size(0), device=hs.device), last_idx]     # [B, H]
        return self.v_head(pooled.float()).squeeze(-1)                        # [B]

    def gradient_checkpointing_enable(self):
        self.backbone.gradient_checkpointing_enable()


# ==========================================================================
# 损失
# ==========================================================================
def bt_loss(r_chosen, r_rejected, margin=None, reg_coef: float = 1e-3):
    """
    Bradley-Terry 损失 + 可选的 margin + L2 正则。

    margin（Llama 2 的做法）：标注时如果记录了偏好强度
    （"显著更好" / "略好"），可以要求强偏好的分差更大：
        L = − log σ( r_w − r_l − m )
    本 lab 的数据集没有强度标注，所以默认 margin=None。
    """
    diff = r_chosen - r_rejected
    if margin is not None:
        diff = diff - margin
    loss = -F.logsigmoid(diff).mean()

    # ★ 正则项：BT 损失只约束分差，分数绝对值可以自由漂移到 ±1000。
    #   那会让后续 PPO 的数值稳定性变差，所以拉一把。
    loss = loss + reg_coef * (torch.cat([r_chosen, r_rejected]) ** 2).mean()
    return loss


# ==========================================================================
# 数据
# ==========================================================================
def make_batches(tok, items, cfg, shuffle=True):
    """把偏好数据打包成 batch。chosen 和 rejected 拼在一起只跑一次前向。"""
    import random
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for s in range(0, len(idx), cfg.micro_bs):
        chunk = [items[i] for i in idx[s: s + cfg.micro_bs]]
        pairs = [build_pair_tensors(tok, it, cfg.max_prompt, cfg.max_len) for it in chunk]
        if not pairs:
            continue
        pad_id = tok.pad_token_id or tok.eos_token_id
        # 顺序很重要：前半是 chosen，后半是 rejected，后面用 chunk(2) 拆开
        yield {
            "input_ids": pad_stack([p["chosen"]["input_ids"] for p in pairs] +
                                   [p["rejected"]["input_ids"] for p in pairs], pad_id),
            "attention_mask": pad_stack([p["chosen"]["attention_mask"] for p in pairs] +
                                        [p["rejected"]["attention_mask"] for p in pairs], 0),
        }


# ==========================================================================
# 训练
# ==========================================================================
def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device, dtype = gpu.pick_device(), gpu.pick_dtype()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = RewardModel(cfg.model, dtype).to(device)
    model.gradient_checkpointing_enable()

    n_tr = 32 if cfg.smoke else cfg.n_train
    n_ev = 16 if cfg.smoke else cfg.n_eval
    data = load_preference_data(n=n_tr + n_ev, seed=cfg.seed)
    train_items, eval_items = data[:n_tr], data[n_tr:n_tr + n_ev]
    print(f"[data] 训练 {len(train_items)} 对，验证 {len(eval_items)} 对")
    print(f"[data] 样例 prompt: {train_items[0]['prompt'][:100]!r}")

    total_steps = 3 if cfg.smoke else max(1, len(train_items) // (cfg.micro_bs * cfg.accum)) * cfg.epochs
    opt = build_optimizer(model, cfg.lr, weight_decay=0.0)
    sched = build_scheduler(opt, total_steps, cfg.warmup_ratio, "cosine")
    logger = Logger(cfg.out_dir, cfg.wandb, run_name="lab02-rm", config=cfg_to_dict(cfg))
    print(f"[train] 共 {total_steps} 步，等效 batch = {cfg.micro_bs * cfg.accum} 对")

    step, done = 0, False
    model.train()
    for _ in range(cfg.epochs):
        gen = make_batches(tok, train_items, cfg)
        while not done:
            micros = []
            for _ in range(cfg.accum):
                try:
                    micros.append(next(gen))
                except StopIteration:
                    break
            if not micros:
                break

            gpu.reset_peak_mem()
            agg = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "r_chosen": 0.0, "r_rejected": 0.0}
            for b in micros:
                b = {k: v.to(device) for k, v in b.items()}
                rewards = model(b["input_ids"], b["attention_mask"])     # [2B]
                r_c, r_r = rewards.chunk(2, dim=0)                        # 各 [B]

                loss = bt_loss(r_c, r_r, reg_coef=cfg.reg_coef) / len(micros)
                loss.backward()

                with torch.no_grad():
                    agg["loss"] += float(loss) * len(micros) / len(micros)
                    agg["acc"] += float((r_c > r_r).float().mean()) / len(micros)
                    agg["margin"] += float((r_c - r_r).mean()) / len(micros)
                    agg["r_chosen"] += float(r_c.mean()) / len(micros)
                    agg["r_rejected"] += float(r_r.mean()) / len(micros)

            gnorm = clip_and_step(model, opt, sched, cfg.max_grad_norm)
            step += 1
            _, peak = gpu.gpu_mem_gb()
            agg.update({"lr": sched.get_last_lr()[0], "gnorm": gnorm, "peak_gb": peak})
            logger.log(step, agg)
            logger.print_row(step, agg, every=cfg.log_every)

            if step % cfg.eval_every == 0 or step >= total_steps:
                ev = evaluate(model, tok, eval_items, cfg, device)
                logger.log(step, ev, prefix="eval/")
                print(f"      └─ eval acc {ev['acc']:.3f}  margin {ev['margin']:.3f}")
                model.train()
            if step >= total_steps:
                done = True
        if done:
            break

    ev = evaluate(model, tok, eval_items, cfg, device)
    print(f"\n[final] 验证集 pairwise 准确率 = {ev['acc']:.3f}")
    print(interpret_accuracy(ev["acc"]))

    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg_to_dict(cfg)},
               os.path.join(cfg.out_dir, "reward_model.pt"))
    tok.save_pretrained(cfg.out_dir)
    print(f"[ckpt] 已保存到 {cfg.out_dir}/reward_model.pt")
    logger.close()


@torch.no_grad()
def evaluate(model, tok, items, cfg, device) -> dict:
    model.eval()
    accs, margins = [], []
    for b in make_batches(tok, items, cfg, shuffle=False):
        b = {k: v.to(device) for k, v in b.items()}
        r_c, r_r = model(b["input_ids"], b["attention_mask"]).chunk(2, dim=0)
        accs.append(float((r_c > r_r).float().mean()))
        margins.append(float((r_c - r_r).mean()))
    return {"acc": sum(accs) / max(1, len(accs)), "margin": sum(margins) / max(1, len(margins))}


def interpret_accuracy(acc: float) -> str:
    """帮你判断这个数字是好是坏 —— 新手常常不知道该期待多少。"""
    if acc < 0.55:
        return ("  → 接近瞎猜。检查：数据加载对不对？学习率是不是太小？"
                "pooling 是不是取到了 pad token？")
    if acc < 0.65:
        return "  → 偏低但在学。可以多给点数据，或者换更大的 backbone。"
    if acc <= 0.80:
        return ("  → ✅ 正常区间。人类标注员之间的一致率本身也只有 70~80%，"
                "所以这已经接近数据的信息上限了。")
    return ("  → ⚠️ 高得可疑。要么过拟合了（RM 训超过 1 个 epoch 的典型症状），"
            "要么 RM 找到了捷径（比如只数长度）。建议查一下 reward 与回答长度的相关性。")


def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in vars(cfg).items():
        if isinstance(val, bool):
            ap.add_argument(f"--{name.replace('_','-')}", action="store_true", default=val)
        else:
            ap.add_argument(f"--{name.replace('_','-')}", type=type(val), default=val)
    ns = ap.parse_args()
    default_out = cfg.out_dir
    for k in vars(cfg):
        setattr(cfg, k, getattr(ns, k))
    if cfg.smoke:
        cfg.micro_bs, cfg.accum, cfg.log_every, cfg.eval_every = 1, 2, 1, 3
        if cfg.out_dir == default_out:
            cfg.out_dir += "_smoke"
    return cfg


if __name__ == "__main__":
    train(parse_args())
