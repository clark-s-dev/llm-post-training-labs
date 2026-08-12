#!/usr/bin/env python3
"""
Lab 03 —— 从零实现 DPO（直接偏好优化）

目标
----
不训奖励模型、不做采样、不用 PPO，直接从成对偏好数据训出对齐后的策略。

核心公式（推导过程见教程第 06 章，六步走完）
------------------------------------------
    L_DPO = − E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x))
                      − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]

其中 r̂_θ(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) 叫做「隐式奖励」——
这就是论文标题的意思：**你的语言模型偷偷就是一个奖励模型**。

梯度长这样：
    ∇L = −β·E[ σ(r̂_l − r̂_w) · ( ∇log π(y_w) − ∇log π(y_l) ) ]
             └─── 自适应权重 ───┘  └─ 提高好的 ─┘ └─ 压低坏的 ─┘
    模型已经判断正确的样本，权重趋近 0（不再贡献梯度）——内置的困难样本挖掘。

本 lab 演示的教学重点：似然位移（likelihood displacement）
------------------------------------------------------
训练时盯着 logps/chosen 这个指标，你会看到一个反直觉现象：
**chosen 的对数概率也在下降**，只是比 rejected 降得慢。

原因：损失只约束「差值」，不约束绝对值。而「压低一个序列」在优化上
比「抬高一个序列」容易得多，于是优化器走了阻力最小的路。被压走的概率
质量流向了既不是 chosen 也不是 rejected 的第三类输出（可能是乱码）。
这就是 DPO 训久了模型会「变傻」的机制。

用 --rpo-alpha 1.0 打开 NLL 正则项即可缓解（这是目前最常用的修法）。

L4 (24GB) 参考耗时
-----------------
    2000 对 × 1 epoch ≈ 9 分钟，峰值显存约 12 GB（ref 的 logp 预先算好可再省 2GB）

用法
----
    python labs/lab03_dpo/train_dpo.py --smoke
    python labs/lab03_dpo/train_dpo.py                        # 原版 DPO，能看到似然位移
    python labs/lab03_dpo/train_dpo.py --rpo-alpha 1.0        # 加 NLL 正则，位移被抑制
    python labs/lab03_dpo/train_dpo.py --loss-type ipo        # 换成 IPO
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

from common import gpu
from common.data import build_pair_tensors, load_preference_data, pad_stack, set_seed
from common.logprobs import per_token_logps, sequence_logps
from common.train_utils import (Logger, build_optimizer, build_scheduler, cfg_to_dict,
                                clip_and_step, save_checkpoint)


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "outputs/lab03_dpo"
    n_train: int = 2000
    n_eval: int = 200
    max_len: int = 640
    max_prompt: int = 320

    # ---- DPO 超参 ----
    beta: float = 0.1              # ★ 越小越激进地偏离 ref。0.1 是标准起点
    loss_type: str = "sigmoid"     # sigmoid(原版DPO) | ipo | cdpo
    label_smoothing: float = 0.0   # cDPO：假设有这么大比例的标注是反的
    rpo_alpha: float = 0.0         # >0 时加 NLL 正则，防似然位移。推荐 1.0

    # ---- 优化 ----
    lr: float = 5e-7               # ★★ 比 SFT 小 10~40 倍！用 SFT 的 lr 会在 50 步内训废
    epochs: int = 1                # DPO 过拟合极快
    micro_bs: int = 2
    accum: int = 8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    seed: int = 0
    log_every: int = 5
    eval_every: int = 50
    smoke: bool = False
    wandb: bool = False


# ==========================================================================
# 损失
# ==========================================================================
def dpo_loss(policy_c, policy_r, ref_c, ref_r, cfg: Config):
    """
    参数都是 [B] 的**序列级** log-prob（token log-prob 之和，不是平均！）。

    ★ 为什么必须求和不能平均？
      推导里的 log π(y|x) = Σ_t log π(y_t|·) 就是求和，这是序列概率的定义。
      除以长度会脱离推导（隐式奖励 β·log(π_θ/π_ref) 不再成立），
      变成另一个算法（那是 SimPO 干的事）。
    """
    # 隐式奖励（除掉 β 之前的部分）
    chosen_logratio = policy_c - ref_c        # log(π_θ(y_w)/π_ref(y_w))
    rejected_logratio = policy_r - ref_r      # log(π_θ(y_l)/π_ref(y_l))
    logits = chosen_logratio - rejected_logratio          # 括号里那一坨 / β

    if cfg.loss_type == "sigmoid":            # 原版 DPO
        loss = -F.logsigmoid(cfg.beta * logits)
    elif cfg.loss_type == "cdpo":             # 带标签平滑，对噪声标注更鲁棒
        eps = cfg.label_smoothing or 0.1
        loss = (-F.logsigmoid(cfg.beta * logits) * (1 - eps)
                - F.logsigmoid(-cfg.beta * logits) * eps)
    elif cfg.loss_type == "ipo":
        # IPO：把 log-sigmoid 换成平方损失。
        # 动机：log σ 在 margin 已经很大时仍有梯度，会一直把 margin 往大推，
        #       最终过拟合成确定性策略。平方损失到达目标 margin 后梯度归零。
        loss = (logits - 1 / (2 * cfg.beta)) ** 2
    else:
        raise ValueError(f"未知 loss_type: {cfg.loss_type}")

    metrics = {
        "rewards/chosen": (cfg.beta * chosen_logratio).mean(),
        "rewards/rejected": (cfg.beta * rejected_logratio).mean(),
        "rewards/margin": (cfg.beta * (chosen_logratio - rejected_logratio)).mean(),
        "rewards/accuracy": (chosen_logratio > rejected_logratio).float().mean(),
        # ★★ 下面两个才是要盯死的指标 —— 它们暴露似然位移 ★★
        "logps/chosen": policy_c.mean(),
        "logps/rejected": policy_r.mean(),
    }
    return loss.mean(), metrics


# ==========================================================================
# 数据
# ==========================================================================
def make_batches(tok, items, cfg, shuffle=True):
    """
    产出的 batch 里，前半是 chosen、后半是 rejected（拼一起只跑一次前向）。
    completion_mask 标记「回答部分」，只有这部分的 log-prob 会被累加。
    """
    import random
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    pad_id = tok.pad_token_id or tok.eos_token_id
    for s in range(0, len(idx), cfg.micro_bs):
        chunk = [items[i] for i in idx[s: s + cfg.micro_bs]]
        pairs = [build_pair_tensors(tok, it, cfg.max_prompt, cfg.max_len) for it in chunk]
        if not pairs:
            continue
        yield {
            "input_ids": pad_stack([p["chosen"]["input_ids"] for p in pairs] +
                                   [p["rejected"]["input_ids"] for p in pairs], pad_id),
            "attention_mask": pad_stack([p["chosen"]["attention_mask"] for p in pairs] +
                                        [p["rejected"]["attention_mask"] for p in pairs], 0),
            "completion_mask": pad_stack([p["chosen"]["completion_mask"] for p in pairs] +
                                         [p["rejected"]["completion_mask"] for p in pairs], 0),
        }


def batch_seq_logps(model, b, device):
    """算 [2B] 的序列级 log-prob，同时返回逐 token 的（RPO 正则要用）。"""
    ids, attn, cmask = b["input_ids"].to(device), b["attention_mask"].to(device), b["completion_mask"].to(device)
    # completion 部分从整条序列的末尾往前数：这里直接用整条长度，靠 mask 过滤
    C = ids.size(1) - 1
    per_tok = per_token_logps(model, ids, attn, completion_len=C, temperature=1.0)  # [2B, C]
    mask = cmask[:, 1:]                       # 与 per_tok 对齐（per_tok 对应 token 1..L-1）
    return sequence_logps(per_tok, mask, average=False), per_tok, mask


# ==========================================================================
# 训练
# ==========================================================================
def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device, dtype = gpu.pick_device(), gpu.pick_dtype()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(train_mode: bool):
        m = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype=dtype, attn_implementation=gpu.pick_attn_impl()).to(device)
        m.config.use_cache = False
        if train_mode:
            m.gradient_checkpointing_enable()
            m.train()
        else:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        return m

    policy = load(True)
    # ★ 参考模型必须和 policy 的初始权重完全一致（通常就是 SFT 后那份）。
    #   ref 用别的模型的话，训练开始时 log(π_θ/π_ref) 就不是 0，损失从奇怪的点出发。
    ref = load(False)
    print("[model] policy + ref 已加载（ref 冻结）")

    n_tr = 16 if cfg.smoke else cfg.n_train
    n_ev = 8 if cfg.smoke else cfg.n_eval
    data = load_preference_data(n=n_tr + n_ev, seed=cfg.seed)
    train_items, eval_items = data[:n_tr], data[n_tr:n_tr + n_ev]

    total_steps = 3 if cfg.smoke else max(1, len(train_items) // (cfg.micro_bs * cfg.accum)) * cfg.epochs
    opt = build_optimizer(policy, cfg.lr, weight_decay=0.0)
    sched = build_scheduler(opt, total_steps, cfg.warmup_ratio, "cosine")
    logger = Logger(cfg.out_dir, cfg.wandb, run_name="lab03-dpo", config=cfg_to_dict(cfg))
    print(f"[train] {len(train_items)} 对 | {total_steps} 步 | β={cfg.beta} lr={cfg.lr:g} "
          f"loss={cfg.loss_type} rpo_alpha={cfg.rpo_alpha}")

    logps_chosen_start = None
    step, done = 0, False
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
            agg: dict[str, float] = {}
            for b in micros:
                pol_logps, pol_per_tok, mask = batch_seq_logps(policy, b, device)
                with torch.no_grad():
                    ref_logps, _, _ = batch_seq_logps(ref, b, device)

                pol_c, pol_r = pol_logps.chunk(2, dim=0)
                ref_c, ref_r = ref_logps.chunk(2, dim=0)
                loss, m = dpo_loss(pol_c, pol_r, ref_c, ref_r, cfg)

                # ---- RPO / DPO+NLL 正则 ----
                # 直接把「chosen 的概率不许降」写进损失，是治似机位移最有效的手段。
                if cfg.rpo_alpha > 0:
                    per_tok_c = pol_per_tok.chunk(2, dim=0)[0]
                    mask_c = mask.chunk(2, dim=0)[0]
                    nll = -(per_tok_c * mask_c).sum() / mask_c.sum().clamp(min=1)
                    loss = loss + cfg.rpo_alpha * nll
                    m["nll_chosen"] = nll

                (loss / len(micros)).backward()
                for k, v in m.items():
                    agg[k] = agg.get(k, 0.0) + float(v) / len(micros)
                agg["loss"] = agg.get("loss", 0.0) + float(loss) / len(micros)

            gnorm = clip_and_step(policy, opt, sched, cfg.max_grad_norm)
            step += 1
            if logps_chosen_start is None:
                logps_chosen_start = agg["logps/chosen"]

            _, peak = gpu.gpu_mem_gb()
            agg.update({"lr": sched.get_last_lr()[0], "gnorm": gnorm, "peak_gb": peak,
                        # 这个派生指标直接量化似然位移的严重程度
                        "logps/chosen_drift": agg["logps/chosen"] - logps_chosen_start})
            logger.log(step, agg)
            logger.print_row(step, {k: agg[k] for k in
                                    ["loss", "rewards/accuracy", "rewards/margin",
                                     "logps/chosen", "logps/rejected", "gnorm"]},
                             every=cfg.log_every)
            if step >= total_steps:
                done = True
        if done:
            break

    # ---------------- 收尾：把似然位移讲清楚 ----------------
    drift = agg["logps/chosen"] - logps_chosen_start
    print("\n" + "=" * 76)
    print("【似然位移诊断】")
    print(f"  logps/chosen  起始 {logps_chosen_start:8.2f}  →  结束 {agg['logps/chosen']:8.2f}"
          f"   （变化 {drift:+.2f}）")
    print(f"  rewards/margin  {agg['rewards/margin']:.3f}   accuracy {agg['rewards/accuracy']:.3f}")
    if drift < -5 and cfg.rpo_alpha == 0:
        print("  ⚠️  chosen 的对数概率明显下降了 —— 这就是似然位移。")
        print("     margin 和 accuracy 看起来很好，但模型可能正在变差。")
        print("     试试： --rpo-alpha 1.0   再跑一遍，对比这个数字。")
    elif cfg.rpo_alpha > 0:
        print("  ✅ 已开启 RPO 正则（--rpo-alpha），chosen 概率被 NLL 项拉住了。")
    print("=" * 76)

    policy.config.use_cache = True
    save_checkpoint(policy, tok, cfg.out_dir, extra={"final_metrics": agg})
    logger.close()


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
        cfg.micro_bs, cfg.accum, cfg.log_every = 1, 2, 1
        if cfg.out_dir == default_out:
            cfg.out_dir += "_smoke"
    return cfg


if __name__ == "__main__":
    train(parse_args())
