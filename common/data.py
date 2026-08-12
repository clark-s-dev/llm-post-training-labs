"""
数据处理：chat template、loss mask、GSM8K 加载、偏好数据加载。

这个文件里的 bug 是最难查的一类 —— 因为它们**不会报错**。
模板差一个换行、loss mask 多屏蔽一个 token，训练曲线看起来完全正常，
但模型就是训不好。所以本文件提供了 `describe_example()` 这样的可视化自检工具，
**开跑之前请务必用它肉眼看一遍第一条样本。**
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

IGNORE_INDEX = -100        # PyTorch CrossEntropyLoss 的默认 ignore_index


# --------------------------------------------------------------------------
# 1. SFT 样本构造（带 loss mask）
# --------------------------------------------------------------------------
@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]              # prompt 部分为 IGNORE_INDEX
    attention_mask: list[int]

    @property
    def n_train_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def build_sft_example(tokenizer, messages: list[dict], max_len: int = 1024) -> SFTExample:
    """
    把一段多轮对话变成一条训练样本，只对 assistant 说的话计算 loss。

    做法：逐轮增量渲染 —— 用 apply_chat_template 分别渲染「前 i 条」和「前 i+1 条」，
    两者的字符串差值就是第 i 条消息实际新增的文本。这样不用手写模板，
    也不会因为模板里有 system prompt、BOS 之类的东西而错位。

    ★ 为什么要屏蔽 prompt？
      我们想让模型学「给定问题如何回答」，而不是「如何生成问题」。
      不屏蔽会把模型容量浪费在拟合用户输入的分布上。

    ★ 为什么 EOS 一定要保留在可学的一侧？
      模型靠 EOS 学会「什么时候停下来」。把 EOS 屏蔽掉（或者被 max_len 截断掉）
      是 SFT 最常见的事故 —— 现象是推理时模型停不下来，一直生成到上限。
    """
    input_ids: list[int] = []
    labels: list[int] = []

    def _render(msgs: list[dict], gen_prompt: bool = False) -> str:
        # 注意：transformers 5.x 对空消息列表会直接抛 ValueError，所以空列表要特判。
        if not msgs:
            return ""
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=gen_prompt)

    for i, msg in enumerate(messages):
        prefix = _render(messages[:i])
        full = _render(messages[: i + 1])

        if msg["role"] != "assistant":
            ids = tokenizer.encode(full[len(prefix):], add_special_tokens=False)
            input_ids += ids
            labels += [IGNORE_INDEX] * len(ids)          # ❌ 用户/系统消息：全屏蔽
            continue

        # assistant 这一轮要再切一刀，把「角色头」和「真正的回答内容」分开：
        #     <|im_start|>assistant\n   ← 角色头。推理时由 add_generation_prompt 喂进去，
        #                                  模型不需要自己生成它，所以要屏蔽。
        #     答案正文……<|im_end|>       ← 这才是要学的（EOS/<|im_end|> 必须包含在内，
        #                                  否则模型学不会什么时候停下来）。
        header = _render(messages[:i], gen_prompt=True)
        if header.startswith(prefix) and full.startswith(header):
            head_ids = tokenizer.encode(header[len(prefix):], add_special_tokens=False)
            body_ids = tokenizer.encode(full[len(header):], add_special_tokens=False)
        else:
            # 某些自定义模板不支持 add_generation_prompt，退回「整轮都学」的保守做法
            head_ids, body_ids = [], tokenizer.encode(full[len(prefix):], add_special_tokens=False)

        input_ids += head_ids + body_ids
        labels += [IGNORE_INDEX] * len(head_ids) + body_ids

    # 从右边截断（保留 prompt 开头）。注意截断可能会砍掉 EOS —— 下面会检查。
    input_ids, labels = input_ids[:max_len], labels[:max_len]
    return SFTExample(input_ids, labels, [1] * len(input_ids))


def sanity_check_example(tokenizer, ex: SFTExample, verbose: bool = True) -> None:
    """
    三条断言能挡掉绝大多数 SFT 数据 bug。开跑前跑一次，30 秒的事。
    """
    assert len(ex.input_ids) == len(ex.labels) == len(ex.attention_mask), "三个字段长度不一致"
    n = ex.n_train_tokens
    assert n > 0, "整条样本没有任何可学的 token —— 多半是 chat template 匹配失败了"
    assert n < len(ex.input_ids), "所有 token 都在学 —— 忘了屏蔽 prompt？"
    if verbose:
        print(describe_example(tokenizer, ex))


def describe_example(tokenizer, ex: SFTExample) -> str:
    """
    把一条样本可视化：绿色 = 参与训练，灰色 = 被屏蔽。
    在终端里一眼就能看出 mask 对不对。
    """
    GREEN, GREY, RESET = "\033[92m", "\033[90m", "\033[0m"
    out, cur, buf = [], ex.labels[0] != IGNORE_INDEX, []
    for tid, lab in zip(ex.input_ids, ex.labels):
        trainable = lab != IGNORE_INDEX
        if trainable != cur:
            out.append((GREEN if cur else GREY) + tokenizer.decode(buf) + RESET)
            buf, cur = [], trainable
        buf.append(tid)
    out.append((GREEN if cur else GREY) + tokenizer.decode(buf) + RESET)

    n, total = ex.n_train_tokens, len(ex.input_ids)
    header = (f"{GREEN}绿色=参与训练{RESET}  {GREY}灰色=被屏蔽{RESET}   "
              f"可学 {n}/{total} = {n / total:.1%}\n" + "─" * 70 + "\n")
    return header + "".join(out) + "\n" + "─" * 70


def collate_sft(batch: list[SFTExample], pad_token_id: int) -> dict:
    """把一批变长样本 pad 成矩形张量。右侧 padding（训练用；生成时要左 padding）。"""
    import torch
    L = max(len(b.input_ids) for b in batch)
    ids, labs, masks = [], [], []
    for b in batch:
        pad = L - len(b.input_ids)
        ids.append(b.input_ids + [pad_token_id] * pad)
        labs.append(b.labels + [IGNORE_INDEX] * pad)        # padding 不算 loss
        masks.append(b.attention_mask + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


# --------------------------------------------------------------------------
# 2. GSM8K（本仓库的主力数据集）
# --------------------------------------------------------------------------
GSM8K_SYSTEM = (
    "你是一个数学助手。请先在 <think> 和 </think> 之间逐步推理，"
    "然后用 \\boxed{} 给出最终的数值答案。"
)


def load_gsm8k(split: str = "train", n: int | None = None, seed: int = 0) -> list[dict]:
    """
    加载 GSM8K（小学数学应用题，7473 训练 / 1319 测试）。

    返回 [{"question": str, "answer": str, "solution": str}, ...]
      answer   = 标准答案（纯数字字符串），用来判分
      solution = 原始的人写解题过程，lab01 SFT 会用到

    第一次运行会从 HuggingFace 下载（约 5 MB）。离线服务器可以先跑：
        python scripts/download_assets.py
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = [
        {
            "question": r["question"],
            "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
            "solution": r["answer"].split("####")[0].strip(),
        }
        for r in ds
    ]
    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def gsm8k_to_sft_messages(item: dict) -> list[dict]:
    """
    把 GSM8K 的一条数据变成本仓库统一格式的对话。

    ★ 这一步是 lab01 → lab04 能串起来的关键：
      SFT 教会模型输出 <think>…</think> + \\boxed{}，
      GRPO 的格式奖励才有东西可奖励。如果直接对 base 模型跑 GRPO，
      前几十步会全部因为格式不合规拿 0 分，训练启动非常慢。
    """
    # 原始 solution 里有 <<48/2=24>> 这种计算器标注，去掉更干净
    import re
    reasoning = re.sub(r"<<[^>]*>>", "", item["solution"]).strip()
    return [
        {"role": "system", "content": GSM8K_SYSTEM},
        {"role": "user", "content": item["question"]},
        {"role": "assistant",
         "content": f"<think>\n{reasoning}\n</think>\n\n答案是 \\boxed{{{item['answer']}}}"},
    ]


def build_prompt(tokenizer, question: str, system: str = GSM8K_SYSTEM) -> str:
    """构造推理/rollout 用的 prompt（以 assistant 开头结束，让模型接着写）。"""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,      # ★ 关键：末尾补上 "<|im_start|>assistant\n"
    )


# --------------------------------------------------------------------------
# 3. 偏好数据（lab02 奖励模型 / lab03 DPO 用）
# --------------------------------------------------------------------------
def load_preference_data(n: int | None = None, seed: int = 0) -> list[dict]:
    """
    加载成对偏好数据，返回 [{"prompt", "chosen", "rejected"}, ...]。

    默认用 UltraFeedback（GPT-4 标注，质量较高）。
    数据集在 HF 上偶尔会改名/下线，所以这里准备了多个候选，逐个尝试。
    """
    from datasets import load_dataset

    candidates = [
        ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs"),
        ("trl-lib/ultrafeedback_binarized", "train"),
        ("Anthropic/hh-rlhf", "train"),
    ]
    last_err: Exception | None = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            items = _normalize_preference(ds, name)
            if items:
                print(f"[data] 使用偏好数据集: {name} ({len(items)} 条)")
                break
        except Exception as e:                       # 网络/改名/字段变化都在这里兜住
            last_err = e
            continue
    else:
        raise RuntimeError(f"所有候选偏好数据集都加载失败，最后一个错误: {last_err}")

    if n is not None and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def _normalize_preference(ds, name: str) -> list[dict]:
    """不同数据集字段不一样，统一成 {prompt, chosen, rejected} 三个字符串。"""
    items: list[dict] = []
    for r in ds:
        if "chosen" not in r or "rejected" not in r:
            continue
        chosen, rejected = r["chosen"], r["rejected"]

        # UltraFeedback 格式：chosen/rejected 是消息列表 [{role, content}, ...]
        if isinstance(chosen, list):
            prompt = r.get("prompt") or (chosen[0]["content"] if chosen else "")
            chosen_txt = chosen[-1]["content"] if chosen else ""
            rejected_txt = rejected[-1]["content"] if rejected else ""
        # HH-RLHF 格式：整段对话拼成一个字符串，用 "\n\nAssistant:" 分隔
        elif "hh-rlhf" in name:
            sep = "\n\nAssistant:"
            if sep not in chosen:
                continue
            prompt = chosen.rsplit(sep, 1)[0] + sep
            chosen_txt = chosen.rsplit(sep, 1)[1].strip()
            rejected_txt = rejected.rsplit(sep, 1)[1].strip() if sep in rejected else ""
        else:
            prompt = r.get("prompt", "")
            chosen_txt, rejected_txt = str(chosen), str(rejected)

        if prompt and chosen_txt and rejected_txt and chosen_txt != rejected_txt:
            items.append({"prompt": prompt, "chosen": chosen_txt, "rejected": rejected_txt})
    return items


def build_pair_tensors(tokenizer, item: dict, max_prompt: int = 384, max_len: int = 768):
    """
    把一条偏好数据变成 DPO / RM 需要的张量。

    返回 (ids, attn, completion_mask)，其中 completion_mask 标记「回答部分」——
    只有这部分的 log-prob 会被累加成 log π(y|x)。
    """
    import torch

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    )
    p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[-max_prompt:]

    out = {}
    for key in ("chosen", "rejected"):
        c_ids = tokenizer.encode(item[key], add_special_tokens=False)
        c_ids = c_ids[: max_len - len(p_ids) - 1] + [tokenizer.eos_token_id]   # ★ 补 EOS
        ids = p_ids + c_ids
        out[key] = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "completion_mask": torch.tensor([0] * len(p_ids) + [1] * len(c_ids)),
        }
    return out


def pad_stack(tensors: list, pad_value: int = 0):
    """把一组 1-D 张量右侧 pad 后堆成 2-D。"""
    import torch
    L = max(t.size(0) for t in tensors)
    return torch.stack([
        torch.cat([t, torch.full((L - t.size(0),), pad_value, dtype=t.dtype)]) for t in tensors
    ])


# --------------------------------------------------------------------------
# 4. 杂项
# --------------------------------------------------------------------------
def set_seed(seed: int = 0) -> None:
    """固定随机种子。注意：GPU 上仍不是 bit 级可复现（cuDNN 算法选择有随机性）。"""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hf_cache_dir() -> str:
    """HuggingFace 模型/数据缓存位置。磁盘紧张时可以用 HF_HOME 环境变量改到大盘上。"""
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
