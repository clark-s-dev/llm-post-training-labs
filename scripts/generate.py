#!/usr/bin/env python3
"""
通用采样脚本 —— 拿任意 checkpoint 生成几条回答，用眼睛看看模型现在什么样。

这是最被低估的调试手段。任何自动指标都不如「亲眼看 5 条输出」信息量大：
是不是停不下来？格式对不对？开始复读了吗？推理是真的还是装的？

用法:
    python scripts/generate.py                                   # 用基础模型
    python scripts/generate.py --model outputs/lab01_sft         # 用 SFT 后的
    python scripts/generate.py --model outputs/lab04_grpo/final --n 5 --temp 0.7
    python scripts/generate.py --prompt "9.11 和 9.9 哪个大？"
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from common import gpu
from common.data import GSM8K_SYSTEM, build_prompt, load_gsm8k
from common.rewards import compute_reward, extract_last_boxed, has_valid_format

DEMO_QUESTIONS = [
    "Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?",
    "9.11 和 9.9 哪个更大？",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default=None, help="自定义问题；不给就用 GSM8K 测试集")
    ap.add_argument("--n", type=int, default=3, help="生成几条")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--gsm8k", action="store_true", help="从 GSM8K 测试集取题（能判分）")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = gpu.pick_device(), gpu.pick_dtype()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=gpu.pick_attn_impl()
    ).to(device).eval()
    print(f"[model] {args.model}  device={device} dtype={dtype}\n")

    # 准备问题（带标准答案的话可以顺便判分）
    if args.prompt:
        items = [{"question": args.prompt, "answer": None}]
    elif args.gsm8k:
        items = load_gsm8k("test", n=args.n, seed=0)
    else:
        items = [{"question": q, "answer": None} for q in DEMO_QUESTIONS[: args.n]]

    for i, item in enumerate(items, 1):
        prompt = build_prompt(tok, item["question"], GSM8K_SYSTEM)
        enc = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temp > 0,
                temperature=args.temp if args.temp > 0 else None,
                top_p=1.0, top_k=0,          # 不做截断采样，看到模型真实的分布
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        completion = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        n_new = out.shape[1] - enc.input_ids.shape[1]

        print("═" * 76)
        print(f"[{i}] 问题: {item['question'][:160]}")
        print("─" * 76)
        print(completion)
        print("─" * 76)

        # 自动体检：这几项就是训练时最该关注的信号
        flags = []
        flags.append(("格式合规", has_valid_format(completion)))
        flags.append(("正常结束", n_new < args.max_new_tokens))
        pred = extract_last_boxed(completion)
        if item.get("answer") is not None:
            r, parts = compute_reward(completion, item["answer"])
            flags.append((f"答案正确(gt={item['answer']}, pred={pred})", parts["correct"] == 1.0))
            flags.append((f"总奖励={r:.2f}", r > 0))
        print("  " + "   ".join(f"{'✓' if ok else '✗'} {name}" for name, ok in flags)
              + f"   |  {n_new} tokens")
        print()


if __name__ == "__main__":
    main()
