#!/usr/bin/env python3
"""
Lab 06 —— 正确地评估一个推理模型

评估比训练更容易做错。这个 lab 实现四个指标，并解释它们各自在测什么：

    greedy / pass@1   temperature=0，采一次
                      ❌ 推理模型不要用：长推理链上一个 token 走错就全错，
                         单次结果方差极大；很多推理模型在 temp=0 下还会复读。

    avg@k             采 k 次求平均正确率
                      ✅ 报告推理模型的标准做法。AIME 只有 30 题，
                         一题 = 3.3 分，必须 k≥16 并报标准差。

    pass@k            采 k 次至少对一次 —— 衡量「能力边界」
                      ★ 必须用无偏估计式，直接数是有偏且高方差的：
                            pass@k = E[ 1 − C(n−c, k) / C(n, k) ]
                         其中每题采 n 个样本（n ≥ 4k），其中 c 个正确。

    maj@k             采 k 次多数投票 —— 衡量「自洽性」，通常明显高于 avg@k

另外还做一件事：**污染检查**。GSM8K/MATH 很可能已经在预训练语料里，
用 13-gram 匹配估计测试题与训练题的重叠度。这个数高的话，你的分数不可信。

对应教程章节：第 11 章 11.3

用法
----
    python labs/lab06_eval_passk/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --n 100
    python labs/lab06_eval_passk/evaluate.py --model outputs/lab04_grpo/final --n 200 --k 8
    python labs/lab06_eval_passk/evaluate.py --compare Qwen/Qwen2.5-0.5B-Instruct outputs/lab01_sft outputs/lab04_grpo/final
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from common import gpu
from common.data import GSM8K_SYSTEM, build_prompt, load_gsm8k, set_seed
from common.rewards import RewardConfig, compute_reward, extract_last_boxed


# ==========================================================================
# 指标
# ==========================================================================
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    pass@k 的**无偏**估计（Codex 论文的做法）。

    n 个样本里有 c 个正确，问「随机抽 k 个至少有一个正确」的概率：
        1 − C(n−c, k) / C(n, k)

    直接实现组合数会溢出，用连乘形式：
        C(n−c,k)/C(n,k) = Π_{i=n−c+1}^{n} (1 − k/i)
    """
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def majority_vote(answers: list[str | None]) -> str | None:
    """多数投票（self-consistency）。None（没抽出答案）不参与投票。"""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


# ==========================================================================
# 采样
# ==========================================================================
@torch.no_grad()
def sample_answers(model, tok, items, n_samples: int, cfg) -> list[dict]:
    """
    对每道题采 n_samples 条回答。返回每题的统计。

    ★ 注意所有模型必须用**完全相同的解码参数**比较，
      否则「你的模型 temp=0.6 / baseline temp=0」这种比较毫无意义。
    """
    results = []
    bs = cfg.batch_size
    for qi, item in enumerate(items):
        prompt = build_prompt(tok, item["question"], GSM8K_SYSTEM)
        texts: list[str] = []
        remaining = n_samples
        while remaining > 0:
            cur = min(bs, remaining)
            enc = tok([prompt] * cur, return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.max_prompt_len).to(model.device)
            out = model.generate(
                **enc, max_new_tokens=cfg.max_new_tokens,
                do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                pad_token_id=tok.pad_token_id,
            )
            texts += tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            remaining -= cur

        rcfg = RewardConfig()
        correct = [compute_reward(t, item["answer"], rcfg)[1]["correct"] == 1.0 for t in texts]
        preds = [extract_last_boxed(t) for t in texts]
        results.append({
            "question": item["question"],
            "gold": item["answer"],
            "n": len(texts),
            "c": int(sum(correct)),
            "preds": preds,
            "maj": majority_vote(preds),
            "fmt": sum(1 for t in texts if compute_reward(t, item["answer"], rcfg)[1]["format"] == 1.0),
            "mean_len": float(np.mean([len(t.split()) for t in texts])),
            "sample": texts[0],
        })
        if (qi + 1) % 10 == 0:
            acc = np.mean([r["c"] / r["n"] for r in results])
            print(f"  ...{qi + 1}/{len(items)} 题  当前 avg@{n_samples}={acc:.3f}", flush=True)
    return results


def summarize(results: list[dict], ks: list[int]) -> dict:
    """把每题的统计汇总成报告。同时给出 avg@k 的标准误 —— 这个必须报。"""
    n = results[0]["n"]
    per_q_acc = np.array([r["c"] / r["n"] for r in results])

    out = {
        "n_questions": len(results),
        "samples_per_question": n,
        f"avg@{n}": float(per_q_acc.mean()),
        # 标准误 = std / sqrt(题数)。AIME 30 题时这个数会很大，说明±5分的差异没意义
        f"avg@{n}_stderr": float(per_q_acc.std(ddof=1) / np.sqrt(len(results))),
        "maj@%d" % n: float(np.mean([
            r["maj"] is not None and _eq(r["maj"], r["gold"]) for r in results])),
        "format_rate": float(np.mean([r["fmt"] / r["n"] for r in results])),
        "mean_response_words": float(np.mean([r["mean_len"] for r in results])),
    }
    for k in ks:
        if k <= n:
            out[f"pass@{k}"] = float(np.mean([pass_at_k(r["n"], r["c"], k) for r in results]))
    return out


def _eq(a, b) -> bool:
    from common.rewards import answers_equal
    return answers_equal(a, b)


# ==========================================================================
# 污染检查
# ==========================================================================
def contamination_check(test_items, train_items, n_gram: int = 13) -> dict:
    """
    用 n-gram 重叠估计测试集与训练集的污染程度。

    13-gram 是社区常用阈值（GPT-3 论文起）。命中率高说明测试题
    （或它的近似）出现在训练数据里，分数不可信。

    ⚠️ 这只能查「你自己的训练集」。模型预训练语料里有没有 GSM8K
       你查不到 —— 这正是为什么要用 AIME 2025、LiveCodeBench 最新分片
       这类**训练截止之后**的题目来评测。
    """
    def grams(text: str):
        w = text.split()
        return {" ".join(w[i:i + n_gram]) for i in range(max(0, len(w) - n_gram + 1))}

    train_grams = set()
    for it in train_items:
        train_grams |= grams(it["question"])

    hits = [1 for it in test_items if grams(it["question"]) & train_grams]
    return {"n_gram": n_gram, "overlap_rate": len(hits) / max(1, len(test_items)),
            "n_overlap": len(hits), "n_test": len(test_items)}


# ==========================================================================
# 主流程
# ==========================================================================
def evaluate_one(model_path: str, items, cfg) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device, dtype = gpu.pick_device(), gpu.pick_dtype()
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation=gpu.pick_attn_impl()).to(device).eval()

    print(f"\n[eval] {model_path}   题数={len(items)}  每题采样={cfg.k}  "
          f"temp={cfg.temperature} top_p={cfg.top_p}")
    results = sample_answers(model, tok, items, cfg.k, cfg)
    summary = summarize(results, ks=[1, 2, 4, 8, 16, 32])
    summary["model"] = model_path

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"summary": summary, "per_question": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--compare", nargs="+", default=None, help="一次评多个 checkpoint 并列对比")
    ap.add_argument("--n", type=int, default=100, help="评多少道题")
    ap.add_argument("--k", type=int, default=8, help="每题采几次（pass@k 建议 n≥4k）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-prompt-len", type=int, default=320)
    ap.add_argument("--out", default="outputs/lab06_eval")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    set_seed(cfg.seed)
    items = load_gsm8k("test", n=cfg.n, seed=cfg.seed)

    # 污染检查
    print("\033[1m污染检查\033[0m")
    contam = contamination_check(items, load_gsm8k("train", n=2000, seed=0))
    print(f"  测试题与训练题的 13-gram 重叠率: {contam['overlap_rate']:.1%} "
          f"({contam['n_overlap']}/{contam['n_test']})")
    print("  ⚠️  这只查了本仓库的训练集。模型预训练语料里是否含 GSM8K 无法查证 ——")
    print("     严肃评测请用训练截止之后的题目（AIME 2025 / LiveCodeBench 最新分片）。")

    models = cfg.compare or [cfg.model]
    all_reports = []
    for mp in models:
        rep = evaluate_one(mp, items, cfg)
        all_reports.append(rep)

    # ---------------- 报告 ----------------
    print("\n" + "═" * 96)
    print("\033[1m评测结果\033[0m")
    print("═" * 96)
    keys = [k for k in all_reports[0]["summary"] if k != "model"]
    print(f"{'指标':<24s}" + "".join(f"{os.path.basename(r['summary']['model'])[:20]:>22s}"
                                     for r in all_reports))
    print("─" * 96)
    for key in keys:
        row = f"{key:<24s}"
        for r in all_reports:
            v = r["summary"][key]
            row += f"{v:>22.4f}" if isinstance(v, float) else f"{v:>22}"
        print(row)
    print("═" * 96)

    a = all_reports[0]["summary"]
    print(f"""
\033[1m怎么读这些数字\033[0m
  avg@{cfg.k} = {a[f'avg@{cfg.k}']:.3f} ± {a[f'avg@{cfg.k}_stderr']:.3f}
      ← 报告分数时必须带上这个 ± 。两个模型差距小于 2 倍标准误就不能说谁更好。
  pass@1 vs pass@{cfg.k}
      ← 差距大 = 模型「偶尔能做对但不稳定」。RL 的主要作用就是把 pass@k 转成 pass@1。
  maj@{cfg.k} vs avg@{cfg.k}
      ← maj 明显更高 = 模型的错误是随机分布的（投票能纠正）；
        两者接近 = 模型的错误是系统性的（投票救不了）。
  format_rate
      ← 低于 0.9 说明格式没学稳，RL 的格式奖励还没饱和。
""")

    os.makedirs(cfg.out, exist_ok=True)
    path = os.path.join(cfg.out, "eval_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"summary": r["summary"],
                    "per_question": [{k: v for k, v in q.items() if k != "sample"}
                                     for q in r["per_question"]]} for r in all_reports],
                  f, ensure_ascii=False, indent=2)
    print(f"完整结果已保存到 {path}")


if __name__ == "__main__":
    main()
