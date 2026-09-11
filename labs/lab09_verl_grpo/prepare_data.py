#!/usr/bin/env python3
"""
Lab 09 —— 数据准备：把 GSM8K 转成 veRL 要的 parquet 格式

veRL 不直接吃 HuggingFace Dataset，它要求每行是这样的 schema：

    data_source   str    奖励函数用它决定用哪套判分逻辑（我们只有一种，固定写死）
    prompt        list   [{"role": "system"/"user", "content": str}, ...]，
                          veRL 自己套 chat template，语义同 lab04 的 build_prompt()
    ability       str    随便填，veRL 内部有些统计逻辑会按它分组
    reward_model  dict   {"style": "rule", "ground_truth": str} —— 送进 reward_fn.py
    extra_info    dict   透传给奖励函数的额外信息，这里没用到，留空占位

用法
----
    python labs/lab09_verl_grpo/prepare_data.py
    # 产出 labs/lab09_verl_grpo/data/{train,test}.parquet
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.data import GSM8K_SYSTEM, load_gsm8k

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def to_verl_row(item: dict, idx: int, split: str) -> dict:
    return {
        "data_source": "gsm8k_lab09",
        "prompt": [
            {"role": "system", "content": GSM8K_SYSTEM},
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
