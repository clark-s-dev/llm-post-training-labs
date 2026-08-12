#!/usr/bin/env python3
"""
Lab 07 —— Agentic RL：多轮工具调用

任务
----
给模型一个计算器工具。它必须先输出 <tool>表达式</tool> 调用工具、
读到 <result>…</result> 之后再继续推理，最后给出 \\boxed{} 答案。
用 GRPO 训练它学会「什么时候该调工具、该算什么」。

★★ 本 lab 最重要的一件事：loss mask ★★
    工具返回的 token **不是模型的动作**，是环境的输出。
    如果不屏蔽它们：
      1. 策略梯度会去提高「生成工具返回内容」的概率 → 模型开始复读计算结果
      2. 更糟：这些 token 的重要性比率 ρ 完全没有意义（分母是随机的），
         会出现 10^±5 这种值 → 梯度爆炸
    如果工具返回占了轨迹的 60%，那你 60% 的梯度都是垃圾。

    用 --no-mask 可以亲眼看到这个失败模式（ratio_max 会飙升、gnorm 爆炸）。

对应教程章节：第 10 章「Agentic RL：多轮与工具」

L4 (24GB) 参考耗时
-----------------
    G=8，4 题/步，max 3 轮：约 40~60 秒/步（多轮生成比单轮慢）

用法
----
    python labs/lab07_agentic_rl/train_agent_grpo.py --smoke
    python labs/lab07_agentic_rl/train_agent_grpo.py --steps 100
    python labs/lab07_agentic_rl/train_agent_grpo.py --steps 30 --no-mask   # 看错误示范
"""

from __future__ import annotations

import argparse
import ast
import operator
import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from common import gpu
from common.data import load_gsm8k, set_seed
from common.logprobs import per_token_logps, per_token_logps_chunked
from common.rewards import answers_equal, extract_last_boxed
from common.train_utils import (Logger, build_optimizer, build_scheduler, cfg_to_dict,
                                clip_and_step, save_checkpoint)

# ==========================================================================
# 环境：一个安全的计算器工具
# ==========================================================================
SYSTEM = """你是一个会用计算器的数学助手。规则：
1. 需要算术时，输出 <tool>表达式</tool>，系统会返回 <result>结果</result>。
2. 表达式只能包含数字和 + - * / ( ) 。
3. 拿到结果后继续推理，可以多次调用工具。
4. 最后用 \\boxed{} 给出答案。

示例：
<think>需要算 15 乘以 4</think>
<tool>15*4</tool>
<result>60</result>
<think>所以答案是 60</think>
\\boxed{60}"""

TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)

# 只允许这几种运算 —— 绝不能用 eval()！
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_calc(expr: str) -> str:
    """
    安全地求值一个算术表达式。

    ★ 绝对不要用 eval()。RL 训练会跑几百万次模型生成的字符串，
      模型不需要有恶意，它只要碰巧生成了 __import__('os').system('rm -rf /')
      你的机器就完了。这里用 AST 白名单，只放行数字和四则运算。
    """
    MAX_ABS = 1e15          # 中间结果的上限
    MAX_EXP = 32            # 指数上限
    expr = expr.strip()[:200]
    try:
        node = ast.parse(expr, mode="eval").body

        def guard(v):
            """每一步都检查，而不是等算完再检查。"""
            if isinstance(v, complex) or v != v or abs(v) > MAX_ABS:   # v!=v 判 NaN
                raise ValueError("数值过大")
            return v

        def ev(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return guard(n.value)
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                a, b = ev(n.left), ev(n.right)
                # ★★ 关键安全检查：幂运算必须在**计算之前**限制指数。
                #    Python 的大整数是任意精度的，9**9**9 会先算出一个
                #    有 3.7 亿位的数 —— 进程直接卡死，整个训练 run 挂掉。
                #    这不是理论风险：RL 会跑几百万次模型生成的表达式，
                #    这种输入迟早会出现。
                if isinstance(n.op, ast.Pow) and (abs(b) > MAX_EXP or abs(a) > 1e6):
                    raise ValueError("指数过大")
                return guard(_OPS[type(n.op)](a, b))
            if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
                return guard(_OPS[type(n.op)](ev(n.operand)))
            raise ValueError("不支持的表达式")

        val = ev(node)
        return str(int(val)) if float(val).is_integer() else f"{val:.6g}"
    except ZeroDivisionError:
        return "ERROR: 除零"
    except RecursionError:
        return "ERROR: 表达式嵌套过深"
    except Exception as e:
        # ★ 报错也要如实喂回给模型 —— 我们希望它学会从错误中恢复，
        #   而不是假装什么都没发生。
        return f"ERROR: {type(e).__name__}"


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "outputs/lab07_agent"
    group_size: int = 8
    prompts_per_step: int = 4
    max_turns: int = 3                 # 最多调几次工具
    max_tokens_per_turn: int = 200
    max_total_tokens: int = 700
    max_prompt_len: int = 400
    temperature: float = 1.0

    loss_type: str = "dapo"
    eps_low: float = 0.2
    eps_high: float = 0.28
    lr: float = 1e-6
    micro_bs: int = 2
    steps: int = 100
    mask_tool_output: bool = True      # ★ 本 lab 的教学开关

    seed: int = 0
    log_every: int = 1
    sample_every: int = 10
    smoke: bool = False
    wandb: bool = False


# ==========================================================================
# 多轮 rollout
# ==========================================================================
@torch.no_grad()
def rollout_trajectory(model, tok, question: str, cfg: Config, device) -> dict:
    """
    采一条多轮轨迹。返回 token 序列 + **loss mask**。

    loss_mask[i] = 1  → 第 i 个 token 由模型生成，参与训练
    loss_mask[i] = 0  → prompt 或工具返回，屏蔽
    """
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)[-cfg.max_prompt_len:]

    token_ids = list(ids)
    loss_mask = [0] * len(ids)          # prompt 全屏蔽
    n_turns, n_tool_ok, n_tool_err = 0, 0, 0
    finished = False

    for turn in range(cfg.max_turns + 1):
        budget = min(cfg.max_tokens_per_turn, cfg.max_total_tokens - len(token_ids))
        if budget <= 8:
            break

        inp = torch.tensor([token_ids], device=device)
        out = model.generate(
            input_ids=inp,
            attention_mask=torch.ones_like(inp),
            max_new_tokens=budget,
            do_sample=True, temperature=cfg.temperature, top_p=1.0,
            pad_token_id=tok.pad_token_id,
            # 一生成到 </tool> 就停下来，把控制权交回给环境
            stop_strings=["</tool>"], tokenizer=tok,
        )
        gen_ids = out[0, len(token_ids):].tolist()
        if not gen_ids:
            break
        gen_txt = tok.decode(gen_ids, skip_special_tokens=True)

        token_ids += gen_ids
        loss_mask += [1] * len(gen_ids)                 # ★ 模型生成 → 参与训练

        # 有没有工具调用？
        m = TOOL_RE.search(gen_txt)
        if m is None:
            finished = True                              # 没调工具 = 直接给答案，结束
            break

        n_turns += 1
        result = safe_calc(m.group(1))
        n_tool_err += result.startswith("ERROR")
        n_tool_ok += not result.startswith("ERROR")

        obs = f"\n<result>{result}</result>\n"
        obs_ids = tok.encode(obs, add_special_tokens=False)
        token_ids += obs_ids
        # ★★★ 整个 Agentic RL 最关键的一行 ★★★
        loss_mask += [0 if cfg.mask_tool_output else 1] * len(obs_ids)

    text = tok.decode(token_ids[len(ids):], skip_special_tokens=True)
    return {"token_ids": token_ids, "loss_mask": loss_mask, "prompt_len": len(ids),
            "text": text, "n_turns": n_turns, "finished": finished,
            "n_tool_ok": n_tool_ok, "n_tool_err": n_tool_err}


def agent_reward(traj: dict, gold: str, cfg: Config) -> tuple[float, dict]:
    """
    奖励设计（教程第 10 章的建议）：
      · 主奖励：最终答案对不对
      · 门控：工具调用格式必须合法（有 <tool> 就必须能被解析）
      · 轻微效率惩罚：防止模型学会「反复调工具拖时间」
      · ❌ 不给「调用了工具」本身发奖励 —— 什么时候该调，正是我们想让 RL 学的
    """
    parts = {}
    pred = extract_last_boxed(traj["text"])
    parts["correct"] = 1.0 if answers_equal(pred, gold) else 0.0
    parts["used_tool"] = 1.0 if traj["n_turns"] > 0 else 0.0
    parts["tool_err"] = float(traj["n_tool_err"])

    r = parts["correct"]
    if traj["n_tool_err"] > 0:                      # 工具调用格式错误 → 门控扣分
        r -= 0.3 * min(1.0, traj["n_tool_err"])
        parts["tool_err_penalty"] = -0.3 * min(1.0, traj["n_tool_err"])
    if not traj["finished"]:                        # 撞轮数/token 上限
        r -= 0.3
        parts["unfinished"] = -0.3
    if traj["n_turns"] > 2:                         # 效率惩罚（系数很小）
        r -= 0.05 * (traj["n_turns"] - 2)
    parts["total"] = r
    return r, parts


def collate(trajs: list[dict], pad_id: int, device):
    """
    把变长轨迹右侧 pad 成矩形。completion 部分统一从「最长 prompt」之后开始。

    ★ C = L − P 必须 > 0。如果某条轨迹一个 token 都没生成
      （prompt 已经吃满 max_total_tokens），C 可能变成 0，
      而 Python 的 x[-0:] 等价于 x[:]，会引出一个完全看不懂的报错。
      调用方必须先过滤掉这类轨迹（见 train() 里的 keep 逻辑）。
    """
    L = max(len(t["token_ids"]) for t in trajs)
    P = max(t["prompt_len"] for t in trajs)
    assert L > P, (f"没有任何新生成的 token（L={L} <= P={P}）。"
                   f"多半是 max_total_tokens 太小，塞不下 system prompt + 生成内容。")
    ids, attn, lmask = [], [], []
    for t in trajs:
        pad = L - len(t["token_ids"])
        ids.append(t["token_ids"] + [pad_id] * pad)
        attn.append([1] * len(t["token_ids"]) + [0] * pad)
        lmask.append(t["loss_mask"] + [0] * pad)
    ids = torch.tensor(ids, device=device)
    attn = torch.tensor(attn, device=device)
    lmask = torch.tensor(lmask, device=device)
    C = L - P                                        # completion 长度（统一按最长 prompt 切）
    return {"input_ids": ids, "attention_mask": attn,
            "loss_mask": lmask[:, -C:], "completion_len": C}


# ==========================================================================
# 训练
# ==========================================================================
def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device, dtype = gpu.pick_device(), gpu.pick_dtype()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=dtype, attn_implementation=gpu.pick_attn_impl()).to(device)
    policy.gradient_checkpointing_enable()

    data = load_gsm8k("train", n=None, seed=cfg.seed)
    opt = build_optimizer(policy, cfg.lr, 0.0)
    sched = build_scheduler(opt, cfg.steps, 0.0, "constant")
    logger = Logger(cfg.out_dir, cfg.wandb, run_name="lab07-agent", config=cfg_to_dict(cfg))

    if not cfg.mask_tool_output:
        print("\033[91m" + "!" * 78)
        print("!! --no-mask：工具返回的 token 也会参与训练。这是**错误示范**。")
        print("!! 预期现象：ratio_max 飙升、gnorm 爆炸、模型开始复读 <result> 的内容。")
        print("!" * 78 + "\033[0m")

    print(f"[train] G={cfg.group_size} 题/步={cfg.prompts_per_step} "
          f"最多 {cfg.max_turns} 轮  mask_tool_output={cfg.mask_tool_output}")
    print("=" * 92)

    import random
    rng = random.Random(cfg.seed)

    for step in range(1, cfg.steps + 1):
        t0 = time.time()
        gpu.reset_peak_mem()
        policy.eval()

        # ---- 1. 采样：每题 G 条多轮轨迹 ----
        items = rng.sample(data, cfg.prompts_per_step)
        trajs, rewards, parts = [], [], []
        n_empty = 0
        for it in items:
            for _ in range(cfg.group_size):
                tr = rollout_trajectory(policy, tok, it["question"], cfg, device)
                # ★ 过滤掉「一个 token 都没生成」的退化轨迹。
                #   它们没有任何可学的内容，而且会让后面的 C = L − P 变成 0。
                if len(tr["token_ids"]) <= tr["prompt_len"]:
                    n_empty += 1
                    continue
                r, p = agent_reward(tr, it["answer"], cfg)
                trajs.append(tr); rewards.append(r); parts.append(p)
        policy.train()

        if len(trajs) < cfg.group_size:
            print(f"  ⚠️  本步只采到 {len(trajs)} 条有效轨迹（{n_empty} 条为空），跳过。"
                  f"\n     请调大 --max-total-tokens（当前 {cfg.max_total_tokens}，"
                  f"system prompt 本身就要 ~250 token）。")
            continue
        # 组内优势要求每组条数一致，这里按 group_size 截齐
        n_keep = (len(trajs) // cfg.group_size) * cfg.group_size
        trajs, rewards, parts = trajs[:n_keep], rewards[:n_keep], parts[:n_keep]

        batch = collate(trajs, tok.pad_token_id, device)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # ---- 2. 组内优势 ----
        from labs.lab04_grpo.train_grpo import group_advantages
        adv, _ = group_advantages(r_t, cfg.group_size, norm_by_std=True)

        # ---- 3. old logp ----
        old = per_token_logps_chunked(policy, batch["input_ids"], batch["attention_mask"],
                                      batch["completion_len"], cfg.temperature, cfg.micro_bs)

        # ---- 4. 更新（损失和 lab04 完全一样，只是 mask 换了）----
        n_seq = batch["input_ids"].size(0)
        normalizer = float(batch["loss_mask"].sum().clamp(min=1))
        opt.zero_grad(set_to_none=True)
        ratio_max, clipfrac = 0.0, 0.0
        for s in range(0, n_seq, cfg.micro_bs):
            sl = slice(s, s + cfg.micro_bs)
            logps = per_token_logps(policy, batch["input_ids"][sl], batch["attention_mask"][sl],
                                    batch["completion_len"], cfg.temperature)
            mask = batch["loss_mask"][sl]              # ★ 唯一的改动：用 loss_mask 而不是 completion_mask
            ratio = torch.exp((logps - old[sl]).clamp(-20, 20))
            a = adv[sl].unsqueeze(1)
            loss1 = ratio * a
            loss2 = torch.clamp(ratio, 1 - cfg.eps_low, 1 + cfg.eps_high) * a
            per_tok = -torch.min(loss1, loss2)
            (per_tok * mask).sum().div(normalizer).backward()
            with torch.no_grad():
                mb = mask.bool()
                if mb.any():
                    ratio_max = max(ratio_max, float(ratio.detach()[mb].max()))
                    clipfrac += float((((ratio < 1 - cfg.eps_low) | (ratio > 1 + cfg.eps_high))
                                       * mask).sum()) / normalizer
        gnorm = clip_and_step(policy, opt, sched, 1.0)

        # ---- 5. 监控 ----
        _, peak = gpu.gpu_mem_gb()
        learn_frac = float(batch["loss_mask"].sum()) / float(batch["attention_mask"].sum())
        m = {
            "reward/mean": float(r_t.mean()), "reward/std": float(r_t.std()),
            "acc": sum(p["correct"] for p in parts) / len(parts),
            "tool_use_rate": sum(p["used_tool"] for p in parts) / len(parts),
            "tool_err": sum(p["tool_err"] for p in parts) / len(parts),
            "turns": sum(t["n_turns"] for t in trajs) / len(trajs),
            "finished": sum(t["finished"] for t in trajs) / len(trajs),
            "learnable_frac": learn_frac,       # ★ 健康值 20%~50%
            "ratio_max": ratio_max,             # ★ 不 mask 时这个数会爆
            "clipfrac": clipfrac,
            "gnorm": gnorm, "peak_gb": peak, "t": time.time() - t0,
        }
        logger.log(step, m)
        if step % cfg.log_every == 0:
            print(f"step {step:4d} | acc {m['acc']:.3f} | 用工具 {m['tool_use_rate']:.2f} "
                  f"| 轮数 {m['turns']:.1f} | 工具报错 {m['tool_err']:.2f} "
                  f"| 可学token {learn_frac:.1%} | ratio_max {ratio_max:7.2f} "
                  f"| gn {gnorm:6.2f} | {m['t']:.0f}s", flush=True)

        if not cfg.mask_tool_output and ratio_max > 50:
            print("      \033[91m← ratio_max 已经爆了。这就是不 mask 工具输出的后果。\033[0m")

        if step % cfg.sample_every == 0:
            best = int(r_t.argmax())
            print("─" * 92)
            print(f"  [最高奖励轨迹 R={rewards[best]:.2f} 轮数={trajs[best]['n_turns']}]")
            print("  " + trajs[best]["text"][:600].replace("\n", "\n  "))
            print("─" * 92)

    save_checkpoint(policy, tok, cfg.out_dir)
    logger.close()


def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in vars(cfg).items():
        flag = f"--{name.replace('_','-')}"
        if isinstance(val, bool):
            ap.add_argument(flag, action="store_true", default=val)
            if val:
                ap.add_argument(f"--no-{name.replace('_','-')}", dest=name, action="store_false")
        else:
            ap.add_argument(flag, type=type(val), default=val)
    # 额外提供一个更好记的别名
    ap.add_argument("--no-mask", dest="mask_tool_output", action="store_false")
    ns = ap.parse_args()
    default_out = cfg.out_dir
    for k in vars(cfg):
        setattr(cfg, k, getattr(ns, k))
    if cfg.smoke:
        cfg.steps, cfg.group_size, cfg.prompts_per_step = 2, 4, 1
        cfg.max_turns, cfg.max_tokens_per_turn, cfg.max_total_tokens = 2, 64, 640
        cfg.micro_bs, cfg.sample_every = 2, 2
        if cfg.out_dir == default_out:
            cfg.out_dir += "_smoke"
    return cfg


if __name__ == "__main__":
    train(parse_args())
