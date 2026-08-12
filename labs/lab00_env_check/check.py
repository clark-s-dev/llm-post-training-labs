#!/usr/bin/env python3
"""
Lab 00 —— 环境自检

在跑任何训练之前先跑这个。它会依次检查：

  1. GPU 是否可见、是不是 L4、显存够不够、支不支持 bfloat16
  2. PyTorch 能不能真的在 GPU 上算东西（并测一下算力）
  3. HuggingFace 能不能下载模型和数据集
  4. ★ 本仓库的 per-token log-prob 实现是否与 HF 官方实现完全一致
  5. chat template + loss mask 是否正确
  6. 一次最小的前向 + 反向，看看真实显存占用

第 4 项是最重要的。它验证了「差一位对齐」这个最隐蔽的 bug 不存在 ——
后面 SFT / DPO / GRPO 的所有数学都建立在它之上。

用法:
    python labs/lab00_env_check/check.py                 # 完整检查（会下载 0.5B 模型）
    python labs/lab00_env_check/check.py --no-download   # 跳过下载，只查硬件
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# 让脚本可以直接跑，不需要先 pip install -e .
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from common import gpu
from common.data import (IGNORE_INDEX, build_prompt, build_sft_example,
                         gsm8k_to_sft_messages, sanity_check_example)
from common.logprobs import kl_penalty, selftest_against_hf
from common.rewards import compute_reward, extract_last_boxed, has_valid_format

DEFAULT_MODEL = os.environ.get("LAB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PASS, FAIL, WARN = "\033[92m✓\033[0m", "\033[91m✗\033[0m", "\033[93m!\033[0m"
results: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "", fatal: bool = True) -> bool:
    mark = PASS if ok else (FAIL if fatal else WARN)
    print(f" {mark} {name}" + (f"  —  {detail}" if detail else ""))
    results.append((ok or not fatal, name))
    return ok


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-download", action="store_true",
                    help="跳过所有需要联网的检查")
    args = ap.parse_args()

    print("\033[1mLab 00 · 环境自检\033[0m")

    # ---------------------------------------------------------------- 1. 硬件
    section("1. 硬件与驱动")
    print(gpu.describe_env())
    print()

    has_cuda = torch.cuda.is_available()
    check("CUDA 可用", has_cuda,
          "没有 GPU 的话只能用 --smoke 跑通逻辑，训练会慢几百倍", fatal=False)

    if has_cuda:
        vram = gpu.total_vram_gb()
        check("显存 ≥ 20 GB", vram >= 20, f"实测 {vram:.1f} GB（L4 应为 ~22.3）", fatal=False)
        check("支持 bfloat16", torch.cuda.is_bf16_supported(),
              "不支持的话会退回 fp16，需要额外的 loss scaling", fatal=False)
        cc = torch.cuda.get_device_capability()
        check("Ada 架构 (sm_89)", cc == (8, 9),
              f"实测 sm_{cc[0]}{cc[1]}；不是 L4 的话默认超参可能需要调", fatal=False)

    # ---------------------------------------------------------------- 2. 算力
    section("2. PyTorch GPU 计算")
    if has_cuda:
        try:
            dtype = gpu.pick_dtype()
            a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
            b = torch.randn(4096, 4096, device="cuda", dtype=dtype)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                a @ b
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 20
            tflops = 2 * 4096 ** 3 / dt / 1e12
            check("矩阵乘法可执行", True, f"{tflops:.1f} TFLOPS ({dtype})")
            # L4 的 bf16 稠密算力标称约 121 TFLOPS，实测能到 60~100 就正常
            check("算力在合理区间", tflops > 20,
                  "明显偏低的话检查是不是被别的进程占着，或功耗被限制", fatal=False)
        except Exception as e:
            check("矩阵乘法可执行", False, str(e))
    else:
        print("  (跳过 —— 没有 CUDA)")

    # ---------------------------------------------------------------- 3. 联网
    if args.no_download:
        section("3-6. 已用 --no-download 跳过")
        return summarize()

    section("3. HuggingFace 下载")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(args.model)
        check("tokenizer 下载/加载", True, f"{args.model}  vocab={tok.vocab_size}  用时 {time.time()-t0:.1f}s")
    except Exception as e:
        check("tokenizer 下载/加载", False, f"{e}\n     离线环境请先跑 scripts/download_assets.py")
        return summarize()

    try:
        t0 = time.time()
        device, dtype = gpu.pick_device(), gpu.pick_dtype()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=gpu.pick_attn_impl(),
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        check("模型下载/加载", True,
              f"{n_params:.0f}M 参数, dtype={dtype}, attn={gpu.pick_attn_impl()}, 用时 {time.time()-t0:.1f}s")
    except Exception as e:
        check("模型下载/加载", False, str(e))
        return summarize()

    try:
        from common.data import load_gsm8k
        items = load_gsm8k("test", n=4)
        check("数据集下载 (GSM8K)", len(items) == 4, f"示例答案: {items[0]['answer']}")
    except Exception as e:
        check("数据集下载 (GSM8K)", False, str(e), fatal=False)

    # ------------------------------------------------- 4. ★ logprob 正确性 ★
    section("4. ★ 核心正确性：per-token log-prob 是否与 HF 官方实现一致")
    print("   原理：HF 的 model(labels=ids).loss 是所有位置交叉熵的平均，")
    print("        我们自己算的 per-token log-prob 取负号求平均应该完全相等。")
    print("        对不上 = 存在『差一位』对齐 bug，后面所有训练都会静默失败。\n")
    r = selftest_against_hf(model, tok)
    check("log-prob 对齐", r["ok"],
          f"HF={r['hf_loss']:.6f}  本仓库={r['my_loss']:.6f}  差={r['abs_diff']:.2e}")

    # KL 估计器的数学性质
    lp = torch.randn(4, 32, device=device)
    rlp = lp + torch.randn(4, 32, device=device) * 0.1
    k3 = kl_penalty(lp, rlp, "k3")
    check("KL 估计器 k3 恒非负", bool((k3 >= -1e-6).all()), f"min={k3.min().item():.2e}")

    # ---------------------------------------------- 5. 模板与 loss mask
    section("5. chat template 与 loss mask")
    item = {"question": "Tom has 3 apples and buys 5 more. How many does he have?",
            "answer": "8", "solution": "He starts with 3.\n3 + 5 = <<3+5=8>>8"}
    msgs = gsm8k_to_sft_messages(item)
    ex = build_sft_example(tok, msgs, max_len=512)
    try:
        sanity_check_example(tok, ex, verbose=False)
        check("loss mask 三项断言", True,
              f"可学 {ex.n_train_tokens}/{len(ex.input_ids)} tokens "
              f"({ex.n_train_tokens/len(ex.input_ids):.0%})")
    except AssertionError as e:
        check("loss mask 三项断言", False, str(e))

    check("EOS 在可学的一侧", ex.labels[-1] != IGNORE_INDEX,
          "否则模型学不会停止，推理时会一直生成到上限")

    train_text = tok.apply_chat_template(msgs, tokenize=False)
    infer_text = build_prompt(tok, item["question"])
    check("训练文本以推理 prompt 为前缀", train_text.startswith(infer_text),
          "不一致的话，训练分布和推理分布对不上（SFT 头号事故）")

    print("\n   可视化（绿=参与训练，灰=被屏蔽）：")
    from common.data import describe_example
    print("   " + describe_example(tok, ex).replace("\n", "\n   ")[:1400])

    # 奖励函数
    good = "<think>\n" + "step by step reasoning here. " * 3 + "\n</think>\n答案是 \\boxed{8}"
    check("奖励函数：格式+答案都对 → 1.0", compute_reward(good, "8")[0] == 1.0)
    check("奖励函数：无 think 标签 → 0.0（门控）", compute_reward("\\boxed{8}", "8")[0] == 0.0)
    check("只认最后一个 boxed", extract_last_boxed(r"\boxed{1} x \boxed{2}") == "2",
          "防止模型穷举答案骗分")

    # ---------------------------------------------- 6. 前向 + 反向显存
    section("6. 一次真实的前向 + 反向（看显存占用）")
    try:
        gpu.reset_peak_mem()
        model.train()
        model.gradient_checkpointing_enable()
        ids = torch.randint(0, 1000, (2, 512), device=device)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        alloc, peak = gpu.gpu_mem_gb()
        model.zero_grad(set_to_none=True)
        check("前向 + 反向可执行", True,
              f"loss={out.loss.item():.3f}  峰值显存={peak:.2f} GB (batch=2, len=512, 仅权重+梯度)")
        if has_cuda:
            check("显存余量充足", peak < gpu.total_vram_gb() * 0.5,
                  "剩余空间要留给优化器状态和 rollout", fatal=False)
    except Exception as e:
        check("前向 + 反向可执行", False, str(e))

    return summarize()


def summarize() -> int:
    section("结论")
    bad = [n for ok, n in results if not ok]
    if bad:
        print(f" {FAIL} {len(bad)} 项检查未通过：")
        for n in bad:
            print(f"     · {n}")
        print("\n 请先解决上面的问题再跑训练。常见处理办法见 docs/L4_NOTES.md")
        return 1
    print(f" {PASS} 全部 {len(results)} 项检查通过 —— 可以开始跑 lab01 了：")
    print("     make lab01     # 或 bash labs/lab01_sft/run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
