#!/usr/bin/env python3
"""
Lab 04 —— 从零实现 GRPO（本仓库的核心实验）

目标
----
用可验证奖励（答案对不对）做强化学习，在 GSM8K 上把 0.5B 模型的准确率显著提上去。
不用任何 RL 框架，整个算法从头写，包含 2025 年那几篇关键论文的改进。

算法回顾（教程第 07 章）
---------------------
GRPO = PPO 的裁剪目标 + 用「同一道题采 G 条答案的组内均值」替代 critic 网络。

    优势：   Â_i = (r_i − mean(r)) / std(r)          ← 整条回答共用一个值
    目标：   min( ρ_{i,t}·Â_i , clip(ρ_{i,t}, 1−ε_low, 1+ε_high)·Â_i )
    其中     ρ_{i,t} = π_θ(o_{i,t}|·) / π_old(o_{i,t}|·)

为什么可以不要 critic？
    LLM 的 MDP 里 γ=1 且奖励只在最后一个 token 给，
    所以每个位置的回报都等于同一个终局奖励 —— baseline 只需要一个数，
    而不需要一个能预测每个位置的网络。组内均值就是这个数的无偏蒙特卡洛估计。

本 lab 支持的算法变体（--loss-type）
---------------------------------
    grpo     原版：Σ_i (1/|o_i|) Σ_t ...          有长度偏差
    dr_grpo  Dr.GRPO：去掉 1/|o_i| 和 ÷std        修长度偏差 + 难度偏差
    dapo     DAPO：分母用全 batch 的 token 总数    修长度偏差（推荐）
    gspo     GSPO：把 clip 提到序列级              长序列/MoE 更稳
    cispo    CISPO：裁剪 IS 权重而非梯度           所有 token 都贡献梯度

L4 (24GB) 参考配置与耗时
----------------------
    0.5B 模型，G=8，8 prompt/步，max_new=384：约 25~40 秒/步，峰值显存约 18 GB
    跑 300 步约 3 小时，GSM8K 准确率通常能从 ~25% 提到 ~55%

用法
----
    python labs/lab04_grpo/train_grpo.py --smoke                 # 3 步冒烟（几分钟）
    python labs/lab04_grpo/train_grpo.py                         # 默认 DAPO 风格
    python labs/lab04_grpo/train_grpo.py --model outputs/lab01_sft   # 从 SFT 起步（推荐）
    python labs/lab04_grpo/train_grpo.py --loss-type grpo --no-dynamic-sampling
                                                                 # 复现原版 GRPO 的长度爆炸
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from common import gpu
from common.data import GSM8K_SYSTEM, build_prompt, load_gsm8k, set_seed
from common.logprobs import kl_penalty, per_token_logps, per_token_logps_chunked
from common.rewards import RewardConfig, compute_reward
from common.train_utils import (Logger, build_optimizer, build_scheduler, cfg_to_dict,
                                clip_and_step, save_checkpoint)


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "outputs/lab04_grpo"

    # ---- 采样 ----
    group_size: int = 8            # G：每题采几条。<8 时组内均值噪声太大
    prompts_per_step: int = 8      # 每步几道题 → 每步 8×8 = 64 条回答
    max_prompt_len: int = 320
    max_new_tokens: int = 384      # ★ 显存与速度的头号开关
    temperature: float = 1.0       # ★ 必须 >0；训练时算 logp 也要除以它
    top_p: float = 1.0             # ★ 不做截断采样，否则采样分布 ≠ 计算 logp 的分布
    top_k: int = 0

    # ---- 算法 ----
    loss_type: str = "dapo"        # grpo | dr_grpo | dapo | gspo | cispo
    eps_low: float = 0.2
    eps_high: float = 0.28         # ★ DAPO clip-higher：给低概率 token 上升空间，防熵坍塌
    beta: float = 0.0              # KL 系数。RLVR 里常设 0（可验证奖励不容易被 hack）
    kl_estimator: str = "k3"
    mu: int = 1                    # 每批数据做几次梯度更新。1 = 严格 on-policy
    dynamic_sampling: bool = True  # ★ DAPO：丢掉全对/全错的组（它们梯度为 0）
    norm_adv_by_std: bool = True   # False = Dr.GRPO（去掉难度偏差）

    # ---- 优化 ----
    lr: float = 1e-6               # ★★ 比 SFT 小 10~20 倍。用 SFT 的 lr 会在 20 步内训崩
    micro_bs: int = 4              # 前向/反向的微批。OOM 时先调小它
    max_grad_norm: float = 1.0
    total_steps: int = 300

    # ---- 其他 ----
    seed: int = 0
    log_every: int = 1
    eval_every: int = 25
    save_every: int = 50
    sample_every: int = 10         # 每隔多少步打印一条真实输出
    n_eval: int = 100
    smoke: bool = False
    wandb: bool = False
    use_vllm: bool = False         # 有装 vLLM 的话开它，rollout 快 5~15 倍


# ==========================================================================
# Rollout：对每道题采 G 条回答
# ==========================================================================
class Rollout:
    """
    把「批量生成 + 构造 mask」这件事封装起来。

    这里的每个细节都有讲究，尤其是 padding 方向和 completion_mask 的边界。
    """

    def __init__(self, model, tok, cfg: Config, device):
        self.model, self.tok, self.cfg, self.device = model, tok, cfg, device

    @torch.no_grad()
    def __call__(self, questions: list[str]) -> dict:
        cfg, tok = self.cfg, self.tok

        # 每题重复 G 次 —— 同一题的 G 条在 batch 里是连续的，
        # 后面 view(-1, G) 就能按组切分
        prompts = []
        for q in questions:
            prompts += [build_prompt(tok, q, GSM8K_SYSTEM)] * cfg.group_size

        # ★ 左 padding！批量生成时 pad 必须在左边，
        #   否则 pad token 夹在 prompt 和生成内容中间，破坏因果结构。
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_prompt_len).to(self.device)

        self.model.eval()
        gen = self.model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,                    # ★ 绝不能 greedy：整组会完全相同，优势全 0
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k if cfg.top_k > 0 else None,
            pad_token_id=tok.pad_token_id,
        )
        self.model.train()

        prompt_len = enc.input_ids.shape[1]
        completions = gen[:, prompt_len:]                      # [B, C]
        texts = tok.batch_decode(completions, skip_special_tokens=True)

        # ---- completion_mask：EOS 之后的位置全部置 0 ----
        # ★ EOS 本身要保留（mask=1），模型需要学会「在这里停」。
        eos_id = tok.eos_token_id
        is_eos = completions == eos_id
        C = completions.size(1)
        eos_idx = torch.full((completions.size(0),), C - 1, dtype=torch.long, device=self.device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.float().argmax(dim=1)[has_eos]
        pos = torch.arange(C, device=self.device).expand_as(completions)
        comp_mask = (pos <= eos_idx.unsqueeze(1)).long()

        return {
            "input_ids": gen,                                                    # [B, P+C]
            # 注意 attention_mask 要把 prompt 的左 padding 和 completion 的尾部一起处理
            "attention_mask": torch.cat([enc.attention_mask, comp_mask], dim=1),
            "completion_mask": comp_mask,
            "completion_len": C,
            "texts": texts,
            "n_tokens": comp_mask.sum(dim=1),                                    # [B]
            "truncated": ~has_eos,                                               # [B]
        }


# ==========================================================================
# 优势：GRPO 的灵魂
# ==========================================================================
def group_advantages(rewards: torch.Tensor, group_size: int, norm_by_std: bool = True):
    """
    组内标准化。

    rewards : [B]，其中 B = n_prompts × G，同一题的 G 条连续排列
    返回     : [B] 的优势，以及每组的统计量

    norm_by_std=True  → 原版 GRPO：Â = (r − mean) / std
    norm_by_std=False → Dr.GRPO   ：Â = r − mean

    ★ 为什么 Dr.GRPO 要去掉 ÷std？
      二值奖励下 std = sqrt(p(1−p))，p 是组内正确率。
      p=0.5（中等难度）时 std=0.5，放大 2 倍；
      p=0.9（很简单）  时 std=0.3，放大 3.3 倍。
      也就是说「太简单」和「太难」的题反而拿到更大的梯度权重 —— 这是难度偏差。
    """
    r = rewards.view(-1, group_size)                       # [n_prompts, G]
    mean = r.mean(dim=1, keepdim=True)
    adv = r - mean
    if norm_by_std:
        # ★ 用 1e-4 而不是 1e-8：全对/全错的组 std=0，
        #   除以 1e-8 会得到天文数字甚至 inf，一步就把模型训炸。
        adv = adv / (r.std(dim=1, keepdim=True) + 1e-4)
    return adv.view(-1), {"group_mean": mean.squeeze(1), "group_std": r.std(dim=1)}


# ==========================================================================
# 损失
# ==========================================================================
def grpo_loss(policy, batch, old_logps, ref_logps, advantages, normalizer, cfg: Config):
    """
    返回 (loss, 指标字典)。

    normalizer 是分母，不同 loss_type 取值不同 —— 这一个数就是
    GRPO / Dr.GRPO / DAPO 三篇论文的全部区别（详见教程第 07 章 7.10）。
    """
    logps = per_token_logps(policy, batch["input_ids"], batch["attention_mask"],
                            batch["completion_len"], cfg.temperature)     # [b, C]
    mask = batch["completion_mask"]
    adv = advantages.unsqueeze(1)                                          # [b, 1] 广播到每个 token

    log_ratio = logps - old_logps

    if cfg.loss_type == "gspo":
        # GSPO：序列级重要性比率 s_i = (π_θ(y)/π_old(y))^(1/|y|)，几何平均。
        # 动机：GRPO 的奖励是序列级的，但比率是 token 级的 —— 粒度错配，
        #       几千个高方差的 token 比率累加到同一个序列信号上，方差爆炸。
        # ★ 注意 GSPO 的 ε 要比 GRPO 小两个数量级（论文用 3e-4/4e-4），
        #   因为几何平均后波动幅度小得多。这里按比例缩放。
        seq_lr = (log_ratio * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1)
        # straight-through：数值上等于 s_i，梯度却从当前 token 的 logp 流出
        log_w = logps - logps.detach() + seq_lr.detach()
        ratio = torch.exp(log_w.clamp(max=10.0))
        eps_lo, eps_hi = cfg.eps_low * 1e-3, cfg.eps_high * 1e-3
    else:
        ratio = torch.exp(log_ratio.clamp(-20, 20))
        eps_lo, eps_hi = cfg.eps_low, cfg.eps_high

    if cfg.loss_type == "cispo":
        # CISPO：裁剪的是 IS 权重（带 stop-gradient），不是梯度本身。
        # 标准 clip 会让被裁 token 的梯度**完全消失**，而那些 token 常常正是
        # "However" / "Wait" / "Recheck" 这类低概率但对推理至关重要的词。
        # CISPO 保证每个 token 都贡献梯度，只是权重被限制。
        w = torch.clamp(ratio, max=1 + eps_hi).detach()
        per_token_loss = -w * adv * logps
        clipfrac = ((ratio > 1 + eps_hi) * mask).sum() / mask.sum().clamp(min=1)
    else:
        # 标准 PPO/GRPO 裁剪目标
        loss1 = ratio * adv
        loss2 = torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi) * adv
        per_token_loss = -torch.min(loss1, loss2)          # 负号：最大化目标 → 最小化损失
        clipfrac = (((ratio < 1 - eps_lo) | (ratio > 1 + eps_hi)) * mask).sum() / mask.sum().clamp(min=1)

    # ---- KL 惩罚（β=0 时跳过，RLVR 常见做法）----
    kl_val = torch.tensor(0.0, device=logps.device)
    if cfg.beta > 0 and ref_logps is not None:
        kl = kl_penalty(logps, ref_logps, cfg.kl_estimator)
        per_token_loss = per_token_loss + cfg.beta * kl
        kl_val = (kl * mask).sum() / mask.sum().clamp(min=1)

    # ═══════════════════════════════════════════════════════════════════
    # ★★★ 归一化：一个分母引发的三篇论文 ★★★
    #   grpo    : 先按序列内 token 数平均，再按序列平均 → 短序列的每个 token 权重更大
    #             = 长度偏差 = "要错就错得长一点"，导致长度爆炸
    #   dr_grpo : 除以常数 (batch × max_len)，完全无偏
    #   dapo    : 除以整个全局 batch 的有效 token 总数，所有 token 一视同仁
    # ═══════════════════════════════════════════════════════════════════
    if cfg.loss_type == "grpo":
        loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1)).sum() / normalizer
    elif cfg.loss_type == "dr_grpo":
        loss = (per_token_loss * mask).sum() / (normalizer * cfg.max_new_tokens)
    else:  # dapo / gspo / cispo：全局 token 归一化
        loss = (per_token_loss * mask).sum() / normalizer

    # 指标一律 detach —— 不 detach 的话 float() 会告警，而且会让这些张量
    # 意外地把计算图挂住不释放（长跑时表现为显存缓慢泄漏）
    with torch.no_grad():
        r_det, m_bool = ratio.detach(), mask.bool()
        metrics = {
            "clipfrac": float(clipfrac.detach()),
            "kl": float(kl_val.detach()),
            # ★ ratio_mean 应该≈1.0（μ=1 时严格等于 1）。明显偏离说明
            #   采样分布和训练分布不一致 —— 查温度、精度、padding（教程第 09 章）
            "ratio_mean": float((r_det * mask).sum() / mask.sum().clamp(min=1)),
            "ratio_max": float(r_det[m_bool].max()) if m_bool.any() else 0.0,
        }
    return loss, metrics


# ==========================================================================
# 训练主循环
# ==========================================================================
def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device, dtype = gpu.pick_device(), gpu.pick_dtype()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[env] device={device} dtype={dtype} attn={gpu.pick_attn_impl()}")
    gpu.assert_enough_vram(14, "Lab 04 (GRPO, 0.5B)")

    tok = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")   # ★ 生成必须左 padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=dtype, attn_implementation=gpu.pick_attn_impl()).to(device)
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = True     # 生成时需要 KV cache；per_token_logps 里会临时关掉

    ref = None
    if cfg.beta > 0:
        ref = AutoModelForCausalLM.from_pretrained(
            cfg.model, dtype=dtype, attn_implementation=gpu.pick_attn_impl()).to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        print(f"[model] 已加载参考模型（β={cfg.beta}）")
    else:
        print("[model] β=0，不加载参考模型（省一份显存）")

    data = load_gsm8k("train", n=None, seed=cfg.seed)
    eval_data = load_gsm8k("test", n=cfg.n_eval, seed=cfg.seed)
    print(f"[data] 训练题库 {len(data)} 道，评测 {len(eval_data)} 道")

    rollout = Rollout(policy, tok, cfg, device)
    rcfg = RewardConfig()
    opt = build_optimizer(policy, cfg.lr, weight_decay=0.0)
    # ★ RL 用 constant 调度：总步数事先难以确定，cosine 跑一半停掉的话
    #   实际学习率轨迹和你预期的完全不同
    sched = build_scheduler(opt, cfg.total_steps, warmup_ratio=0.0, schedule="constant")
    logger = Logger(cfg.out_dir, cfg.wandb, run_name=f"lab04-{cfg.loss_type}",
                    config=cfg_to_dict(cfg))

    print(f"\n[train] loss_type={cfg.loss_type}  G={cfg.group_size}  "
          f"prompts/step={cfg.prompts_per_step}  clip=({cfg.eps_low},{cfg.eps_high})  "
          f"β={cfg.beta}  lr={cfg.lr:g}  dynamic_sampling={cfg.dynamic_sampling}")
    print("=" * 88)

    import random
    rng = random.Random(cfg.seed)

    for step in range(1, cfg.total_steps + 1):
        t0 = time.time()
        gpu.reset_peak_mem()

        # ---------------- 1. 采样（含 DAPO 动态采样）----------------
        batch, rewards, parts, n_resample = None, None, None, 0
        for attempt in range(4):
            qs_items = rng.sample(data, cfg.prompts_per_step)
            cand = rollout([it["question"] for it in qs_items])

            # 打分：每条回答算一次奖励
            gts = [it["answer"] for it in qs_items for _ in range(cfg.group_size)]
            r_list, p_list = [], []
            for txt, gt, n_tok, trunc in zip(cand["texts"], gts,
                                             cand["n_tokens"].tolist(),
                                             cand["truncated"].tolist()):
                r, p = compute_reward(txt, gt, rcfg, n_tokens=n_tok,
                                      max_tokens=cfg.max_new_tokens, truncated=trunc)
                r_list.append(r); p_list.append(p)
            r_t = torch.tensor(r_list, dtype=torch.float32, device=device)

            if not cfg.dynamic_sampling:
                batch, rewards, parts = cand, r_t, p_list
                break

            # ★ DAPO 动态采样：只保留「既不全对也不全错」的组。
            #   全对/全错的组 Â 全是 0，梯度为 0 —— 白白花了 G 次生成的算力。
            acc = torch.tensor([p["correct"] for p in p_list], device=device).view(-1, cfg.group_size).mean(1)
            keep = ((acc > 0) & (acc < 1)).nonzero(as_tuple=True)[0]
            if len(keep) == 0:
                n_resample += 1
                continue
            sel = torch.cat([torch.arange(k * cfg.group_size, (k + 1) * cfg.group_size,
                                          device=device) for k in keep])
            batch = {k: (v[sel] if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == len(r_list)
                         else ([v[i] for i in sel.tolist()] if isinstance(v, list) else v))
                     for k, v in cand.items()}
            rewards = r_t[sel]
            parts = [p_list[i] for i in sel.tolist()]
            break
        if batch is None:                       # 连续几次都凑不到有效组 → 用最后一次的原始 batch
            batch, rewards, parts = cand, r_t, p_list
        t_rollout = time.time() - t0

        # ---------------- 2. 优势 ----------------
        advantages, _ = group_advantages(rewards, cfg.group_size, cfg.norm_adv_by_std)

        # ---------------- 3. 记录 π_old（和 π_ref）----------------
        # ★ 这一步很关键：old_logps 必须用**训练引擎**重新前向算一遍，
        #   而不是复用 generate 时的分数。因为 generate 内部可能有不同的数值路径。
        #   （生产环境用 vLLM 采样时，两边的 logp 会有 1e-3 量级的系统差异，
        #     需要 TIS 修正 —— 见教程第 09 章。）
        old_logps = per_token_logps_chunked(
            policy, batch["input_ids"], batch["attention_mask"],
            batch["completion_len"], cfg.temperature, micro_bs=cfg.micro_bs)
        ref_logps = None
        if ref is not None:
            ref_logps = per_token_logps_chunked(
                ref, batch["input_ids"], batch["attention_mask"],
                batch["completion_len"], cfg.temperature, micro_bs=cfg.micro_bs)

        # 归一化分母（见 grpo_loss 里的说明）
        n_seq = batch["input_ids"].size(0)
        total_tokens = batch["completion_mask"].sum().clamp(min=1)
        normalizer = float(total_tokens) if cfg.loss_type in ("dapo", "gspo", "cispo") else n_seq

        # ---------------- 4. 更新 ----------------
        loss_metrics = {}
        for _ in range(cfg.mu):
            opt.zero_grad(set_to_none=True)
            for s in range(0, n_seq, cfg.micro_bs):
                sl = slice(s, s + cfg.micro_bs)
                micro = {"input_ids": batch["input_ids"][sl],
                         "attention_mask": batch["attention_mask"][sl],
                         "completion_mask": batch["completion_mask"][sl],
                         "completion_len": batch["completion_len"]}
                loss, m = grpo_loss(policy, micro, old_logps[sl],
                                    ref_logps[sl] if ref_logps is not None else None,
                                    advantages[sl], normalizer, cfg)
                loss.backward()          # 已按全局分母归一化，直接累加即可
                for k, v in m.items():
                    loss_metrics[k] = loss_metrics.get(k, 0.0) + v * (micro["input_ids"].size(0) / n_seq)
            gnorm = clip_and_step(policy, opt, sched, cfg.max_grad_norm)

        # ---------------- 5. 监控 ----------------
        _, peak = gpu.gpu_mem_gb()
        lens = batch["completion_mask"].sum(1).float()
        metrics = {
            "reward/mean": float(rewards.mean()),
            "reward/std": float(rewards.std()),                     # → 0 说明没信号了
            "acc": sum(p["correct"] for p in parts) / len(parts),   # ★ 真正关心的指标
            "fmt": sum(p["format"] for p in parts) / len(parts),
            "len/mean": float(lens.mean()),
            "len/max": float(lens.max()),
            "trunc": float(batch["truncated"].float().mean()),
            "adv/absmean": float(advantages.abs().mean()),          # → 0 说明优势没信号
            "gnorm": gnorm,
            "peak_gb": peak,
            "t_rollout": t_rollout,
            "t_total": time.time() - t0,
            "n_resample": n_resample,
            **loss_metrics,
        }
        # ★ 训练停滞的头号原因：优势全是 0（组内所有回答得分相同 → 没有可学的信号）。
        #   连续出现就主动报警，而不是让你盯着一条平线看半小时。
        if metrics["adv/absmean"] < 1e-6:
            zero_adv_streak = getattr(train, "_zero_adv", 0) + 1
            train._zero_adv = zero_adv_streak
            if zero_adv_streak in (3, 10, 30):
                print(f"  ⚠️  连续 {zero_adv_streak} 步优势全为 0 —— 组内所有回答得分相同，"
                      f"梯度为 0。\n     可能原因：题目太简单/太难（看 acc={metrics['acc']:.2f}）、"
                      f"模型还不会输出合规格式（fmt={metrics['fmt']:.2f}，"
                      f"建议先跑 lab01 做冷启动 SFT）、或 group_size 太小。")
        else:
            train._zero_adv = 0

        logger.log(step, metrics)
        if step % cfg.log_every == 0:
            print(f"step {step:4d} | acc {metrics['acc']:.3f} | fmt {metrics['fmt']:.3f} "
                  f"| R {metrics['reward/mean']:+.3f}±{metrics['reward/std']:.2f} "
                  f"| len {metrics['len/mean']:4.0f} | trunc {metrics['trunc']:.2f} "
                  f"| clip {metrics.get('clipfrac', 0):.3f} | gn {gnorm:5.2f} "
                  f"| {metrics['t_total']:.0f}s | {peak:.1f}GB", flush=True)

        # 定期打印一条真实输出 —— 数字看不出 reward hacking，眼睛能
        if step % cfg.sample_every == 0:
            best = int(rewards.argmax())
            print("─" * 88)
            print(f"  [最高奖励样本 R={rewards[best]:.2f}] {batch['texts'][best][:500]}")
            print("─" * 88)

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            ev = quick_eval(policy, tok, eval_data, cfg, device)
            logger.log(step, ev, prefix="eval/")
            print(f"      └─ 【独立评测】GSM8K 准确率 {ev['acc']:.3f} "
                  f"（格式合规 {ev['fmt']:.3f}，平均 {ev['len']:.0f} tokens）")

        if step % cfg.save_every == 0:
            save_checkpoint(policy, tok, cfg.out_dir, step=step)

    save_checkpoint(policy, tok, os.path.join(cfg.out_dir, "final"))
    logger.close()
    print("\n下一步：python labs/lab06_eval_passk/evaluate.py --model "
          f"{os.path.join(cfg.out_dir, 'final')} --n 200")


@torch.no_grad()
def quick_eval(model, tok, items, cfg: Config, device) -> dict:
    """
    独立评测集上的准确率。★ 这是唯一可信的指标 ——
    训练 reward 几乎总是单调上升，它跟评测分数背离就说明在 hack 奖励。
    """
    model.eval()
    rcfg = RewardConfig()
    n_ok = n_fmt = n_len = 0
    bs = max(1, cfg.micro_bs * 2)
    for s in range(0, len(items), bs):
        chunk = items[s: s + bs]
        prompts = [build_prompt(tok, it["question"], GSM8K_SYSTEM) for it in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_prompt_len).to(device)
        out = model.generate(**enc, max_new_tokens=cfg.max_new_tokens,
                             do_sample=True, temperature=0.7, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for txt, it in zip(texts, chunk):
            _, p = compute_reward(txt, it["answer"], rcfg)
            n_ok += p["correct"]; n_fmt += p["format"]; n_len += len(txt.split())
    model.train()
    n = max(1, len(items))
    return {"acc": n_ok / n, "fmt": n_fmt / n, "len": n_len / n}


def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in vars(cfg).items():
        flag = f"--{name.replace('_','-')}"
        if isinstance(val, bool):
            ap.add_argument(flag, action="store_true", default=val)
            if val:   # 默认为 True 的开关，额外提供 --no-xxx 关掉
                ap.add_argument(f"--no-{name.replace('_','-')}", dest=name, action="store_false")
        else:
            ap.add_argument(flag, type=type(val), default=val)
    ns = ap.parse_args()
    default_out = cfg.out_dir
    for k in vars(cfg):
        setattr(cfg, k, getattr(ns, k))
    if cfg.smoke:
        cfg.total_steps, cfg.group_size, cfg.prompts_per_step = 3, 4, 2
        cfg.max_new_tokens, cfg.micro_bs = 96, 2
        cfg.eval_every, cfg.save_every, cfg.sample_every, cfg.n_eval = 3, 999, 3, 8
        if cfg.out_dir == default_out:
            cfg.out_dir += "_smoke"
    return cfg


if __name__ == "__main__":
    train(parse_args())
