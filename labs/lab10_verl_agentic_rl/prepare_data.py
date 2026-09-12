#!/usr/bin/env python3
"""
Lab 10 —— 数据准备：把 GSM8K 转成 veRL 多轮 Agentic RL 要的 parquet 格式

和 lab09 的区别只有 system prompt：lab09 是单轮，这里要告诉模型
「可以调用 calculator 工具」。至于工具调用具体怎么触发、怎么解析——
那是 veRL 的 ToolAgentLoop + tool_parser（`format=hermes`）在 rollout 时
自动处理的，不需要在 prompt 里手写 <tool>...</tool> 这种自定义标签
（这是和 lab07 手写版最大的不同，见 README）。

用法
----
    python labs/lab10_verl_agentic_rl/prepare_data.py
    # 产出 labs/lab10_verl_agentic_rl/data/{train,test}.parquet
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.data import load_gsm8k

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# ★ 依旧要求 <think>…</think> + \boxed{}——common/rewards.py 的格式门控靠这个判分，
#   和 lab04/07/08/09 保持完全一致的奖励口径。工具调用本身走原生 function-calling，
#   不需要在 prompt 里教格式。
AGENT_SYSTEM = (
    "你是一个数学助手，可以调用 calculator 工具做四则运算。"
    "请先在 <think> 和 </think> 之间逐步推理，需要计算时调用 calculator 工具，"
    "拿到结果后继续推理，最后用 \\boxed{} 给出最终的数值答案。"
)


def to_verl_row(item: dict, idx: int, split: str) -> dict:
    return {
        "data_source": "gsm8k_lab10",
        "prompt": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": item["question"]},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": item["answer"]},
        "extra_info": {"split": split, "index": idx},
    }


def main(n_train: int | None = None, n_test: int = 200, seed: int = 0) -> None:
    import pandas as pd

    os.makedirs(OUT_DIR, exist_ok=True)
    for split, n in (("train", n_train), ("test", n_test)):
        items = load_gsm8k(split, n=n, seed=seed)
        rows = [to_verl_row(it, i, split) for i, it in enumerate(items)]
        path = os.path.join(OUT_DIR, f"{split}.parquet")
        pd.DataFrame(rows).to_parquet(path)
        print(f"[data] {split}: {len(rows)} 条 → {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n_train, a.n_test, a.seed)
