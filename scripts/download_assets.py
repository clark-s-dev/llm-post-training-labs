#!/usr/bin/env python3
"""
预下载所有 lab 需要的模型与数据集。

什么时候用：
  · 服务器网络慢 —— 先下载好，训练时不会卡在中途
  · 离线环境    —— 在有网的机器上跑一次，把 ~/.cache/huggingface 整个拷过去
  · 想确认磁盘够不够 —— 会打印每项的大小

用法:
    python scripts/download_assets.py
    python scripts/download_assets.py --model Qwen/Qwen2.5-1.5B-Instruct
    HF_HOME=/data/hf python scripts/download_assets.py   # 换缓存盘（磁盘小的时候有用）
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def human(n: float) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

def dir_size(p: str) -> int:
    t = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                t += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return t

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--skip-prefs", action="store_true", help="跳过偏好数据（lab02/03 才用）")
    args = ap.parse_args()

    from common.data import hf_cache_dir
    cache = hf_cache_dir()
    print(f"缓存目录: {cache}")
    before = dir_size(cache) if os.path.exists(cache) else 0
    ok = True

    # 1) 模型 + tokenizer
    print(f"\n[1/3] 模型 {args.model} ...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        AutoTokenizer.from_pretrained(args.model)
        AutoModelForCausalLM.from_pretrained(args.model)
        print(f"      ✓ 完成（{time.time()-t0:.0f}s）")
    except Exception as e:
        ok = False
        print(f"      ✗ 失败: {e}")

    # 2) GSM8K（lab01 / lab04 / lab06 / lab07 都要）
    print("\n[2/3] 数据集 GSM8K ...")
    try:
        from common.data import load_gsm8k
        tr, te = load_gsm8k("train"), load_gsm8k("test")
        print(f"      ✓ 训练 {len(tr)} 题，测试 {len(te)} 题")
    except Exception as e:
        ok = False
        print(f"      ✗ 失败: {e}")

    # 3) 偏好数据（lab02 / lab03）
    if not args.skip_prefs:
        print("\n[3/3] 偏好数据（lab02 奖励模型 / lab03 DPO）...")
        try:
            from common.data import load_preference_data
            d = load_preference_data(n=100)
            print(f"      ✓ 可用，示例条数 {len(d)}")
        except Exception as e:
            ok = False
            print(f"      ✗ 失败: {e}\n        （只影响 lab02/lab03，其他 lab 不受影响）")
    else:
        print("\n[3/3] 已跳过偏好数据")

    after = dir_size(cache) if os.path.exists(cache) else 0
    print(f"\n缓存占用: {human(after)}（本次新增 {human(max(0, after - before))}）")
    print("✓ 全部就绪，可以开始跑 lab 了" if ok else "⚠️ 有项目失败，见上面的错误信息")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
