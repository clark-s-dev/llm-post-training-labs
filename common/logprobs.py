"""
全仓库最核心的一个文件：把「一段文本在某个模型下的对数概率」算出来。

SFT 的损失、DPO 的隐式奖励、GRPO 的重要性比率 —— 全都建立在这里的函数之上。
如果这里错了（最常见的是差一位的对齐错误），训练不会报错，
但 loss 降不下去、ratio 恒等于奇怪的值，非常难查。所以这里写了详细注释和自检函数。

关键概念回顾
-----------
自回归模型在位置 i 输出的 logits，预测的是位置 i+1 的 token。
所以要拿到 token y_t 的概率，得看位置 t-1 的 logits：

    input_ids :  [ x0  x1  x2 | y0  y1  y2 ]        (| 左边是 prompt，右边是 completion)
    logits    :  [ ?   ?   L2 | L3  L4  L5 ]        L2 预测 y0，L3 预测 y1，...
    对齐做法  :  logits[:, :-1] 与 input_ids[:, 1:] 对齐
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. 内存友好的 log_softmax + gather
# --------------------------------------------------------------------------
def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    等价于 torch.gather(logits.log_softmax(-1), 2, index.unsqueeze(-1)).squeeze(-1)，
    但**不会一次性物化整个 [B, T, V] 的 log_softmax 结果**。

    为什么要这么费劲？算一笔账（Qwen2.5 的词表 V = 151936）：
        B=4, T=768, V=151936, fp32 → 4 × 768 × 151936 × 4 B ≈ 1.87 GB
    只是一个中间张量就快 2 GB。逐行处理后峰值降到 1/B，L4 的 24 GB 才够用。

    参数
    ----
    logits : [B, T, V]  未归一化的分数
    index  : [B, T]     每个位置实际出现的 token id
    返回
    ----
    [B, T]  每个位置上「实际出现的那个 token」的 log 概率
    """
    per_token = []
    for row_logits, row_index in zip(logits, index):        # 一次只处理一条序列
        # .float() 很重要：bf16 的 log_softmax 在长尾上精度不足，
        # 会让 RL 的重要性比率 ratio = exp(logp - old_logp) 产生肉眼可见的偏差。
        row_logps = row_logits.float().log_softmax(dim=-1)   # [T, V]
        picked = torch.gather(row_logps, dim=1, index=row_index.unsqueeze(-1)).squeeze(-1)
        per_token.append(picked)                             # [T]
    return torch.stack(per_token)                            # [B, T]


# --------------------------------------------------------------------------
# 2. 兼容不同 transformers 版本的 "只算最后 N 个位置的 logits" 参数
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _logits_to_keep_kwarg(forward_qualname: str, param_names: tuple[str, ...]) -> str | None:
    """
    transformers 里这个参数改过名：
        4.45 ~ 4.47 叫 num_logits_to_keep
        4.48+       叫 logits_to_keep
    传错名字会被 **kwargs 静默吞掉（不报错，但白白算了全部位置的 logits）。
    这里做一次运行时探测，结果缓存起来。
    """
    if "logits_to_keep" in param_names:
        return "logits_to_keep"
    if "num_logits_to_keep" in param_names:
        return "num_logits_to_keep"
    return None


def _keep_kwargs(model, n_keep: int) -> dict:
    """构造 {参数名: n_keep}；模型不支持则返回空 dict（退化成算全部 logits）。"""
    try:
        sig = inspect.signature(model.forward)
        name = _logits_to_keep_kwarg(model.forward.__qualname__, tuple(sig.parameters))
    except (TypeError, ValueError):
        name = None
    return {name: n_keep} if name else {}


# --------------------------------------------------------------------------
# 3. 主函数：算 completion 部分每个 token 的 log-prob
# --------------------------------------------------------------------------
def per_token_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    参数
    ----
    model          : 一个 AutoModelForCausalLM
    input_ids      : [B, L]，prompt 与 completion 已经拼在一起（completion 在最右边）
    attention_mask : [B, L]
    completion_len : completion 有多少个 token（记作 C）
    temperature    : ★ 必须和采样时用的温度一致！
                     如果 rollout 时用 temperature=1.0 采样，这里也要传 1.0。
                     不一致的话，π_old 和 π_θ 就不是同一个分布，
                     重要性比率 ρ 从第一步起就是错的（RL 训练会莫名其妙地不稳）。

    返回
    ----
    [B, C]  completion 每个 token 的 log 概率

    注意：本函数**不做 mask**。padding / EOS 之后的位置也会算出一个值，
          调用方需要自己乘上 completion_mask。这样设计是为了职责单一。
    """
    # ★ 必须显式拦住 completion_len <= 0。
    #   Python 的负零切片有个坑：x[-0:] 等价于 x[0:]，也就是**整个张量**，
    #   于是 targets 会变成整条序列而 logits 是空的 —— 报错信息完全看不出根因
    #   （"Size does not match at dimension 0"）。多轮 rollout 里只要某条轨迹
    #   一个 token 都没生成，就会撞上这个坑。
    if completion_len <= 0:
        raise ValueError(
            f"completion_len 必须 > 0，收到 {completion_len}。"
            "通常意味着这一批里有『没有生成任何 token』的样本（prompt 已经吃满了预算），"
            "请在调用前过滤掉它们。"
        )

    # logits_to_keep = C + 1：
    #   我们需要位置 L-C-1 .. L-2 的 logits（它们分别预测 completion 的第 0..C-1 个 token）。
    #   取最后 C+1 个位置（L-C-1 .. L-1）再丢掉最后一个，正好就是想要的那 C 个。
    #   这样 lm_head（一个 [hidden, 151936] 的大矩阵乘）只在 C+1 个位置上算，
    #   而不是全部 L 个位置 —— prompt 很长时能省下大量显存和时间。
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,                       # 训练时必须关，否则和 gradient checkpointing 冲突
        **_keep_kwargs(model, completion_len + 1),
    )
    logits = out.logits                        # [B, C+1, V]（或 [B, L, V]，取决于是否支持该参数）

    logits = logits[:, :-1, :]                 # ← 左移对齐：丢掉最后一个位置
    logits = logits[:, -completion_len:, :]    # ← 只保留 completion 对应的 C 个位置
    logits = logits / temperature              # ← 与采样温度保持一致

    targets = input_ids[:, -completion_len:]   # [B, C] 实际生成的 token
    return selective_log_softmax(logits, targets)


def per_token_logps_chunked(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_len: int,
    temperature: float = 1.0,
    micro_bs: int = 2,
    no_grad: bool = True,
) -> torch.Tensor:
    """
    per_token_logps 的分批版本，用于「只前向不反向」的场景
    （比如记录 π_old 的 log-prob、或算参考模型 π_ref 的 log-prob）。

    micro_bs 是显存的第一道闸门：OOM 的时候先把它调小。
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    outs = []
    with ctx:
        for s in range(0, input_ids.size(0), micro_bs):
            sl = slice(s, s + micro_bs)
            outs.append(
                per_token_logps(model, input_ids[sl], attention_mask[sl],
                                completion_len, temperature)
            )
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------
# 4. 序列级 log-prob（DPO 用）
# --------------------------------------------------------------------------
def sequence_logps(
    per_token: torch.Tensor,
    mask: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    """
    把 [B, C] 的逐 token log-prob 汇总成 [B] 的整句 log-prob。

    average=False（默认）→ 求和 = log π(y|x)，这是概率的定义。**DPO 必须用这个。**
    average=True          → 除以长度 = 平均每 token 的 log-prob。SimPO 用这个。

    新手最常见的错误就是在 DPO 里用了 average=True：
    那会脱离 DPO 的推导（隐式奖励 β·log(π_θ/π_ref) 不再成立），变成另一个算法。
    """
    summed = (per_token * mask).sum(dim=-1)
    if average:
        return summed / mask.sum(dim=-1).clamp(min=1)
    return summed


# --------------------------------------------------------------------------
# 5. KL 散度的三种估计器（John Schulman, http://joschu.net/blog/kl-approx.html）
# --------------------------------------------------------------------------
def kl_penalty(logp: torch.Tensor, ref_logp: torch.Tensor, kind: str = "k3") -> torch.Tensor:
    """
    估计 KL(π_θ ‖ π_ref)，样本来自 π_θ。

    令 ℓ = log π_ref − log π_θ，r = exp(ℓ)：
        k1 = −ℓ            无偏，但方差大，单样本可能是负数
        k2 = ½ℓ²           有偏，但方差小，**梯度是对的**
        k3 = r − ℓ − 1     无偏 **且** 恒 ≥ 0，方差比 k1 小 —— GRPO 论文用的就是它

    k3 为什么恒非负？因为 e^ℓ ≥ ℓ+1 对任意实数成立
    （e^ℓ 是凸函数，ℓ+1 是它在 ℓ=0 处的切线）。
    """
    diff = ref_logp - logp                       # ℓ
    if kind == "k1":
        return -diff
    if kind == "k2":
        return 0.5 * diff.square()
    if kind == "k3":
        # clamp 是数值保护：ratio 里有 exp，diff 太大会溢出成 inf 然后传染成 NaN
        diff = torch.clamp(diff, min=-20.0, max=20.0)
        return torch.exp(diff) - diff - 1.0
    raise ValueError(f"未知的 KL 估计器: {kind}（可选 k1 / k2 / k3）")


# --------------------------------------------------------------------------
# 6. 自检：把本文件的实现和 HuggingFace 官方实现对一下
# --------------------------------------------------------------------------
def selftest_against_hf(model, tokenizer, text: str = "1 + 1 = 2. The answer is two.") -> dict:
    """
    验证思路：
        HF 的 model(input_ids, labels=input_ids).loss 是「所有位置交叉熵的平均」。
        我们自己算的 per-token log-prob 取负号求平均，应该得到同一个数。
        对不上就说明对齐（差一位）或温度处理有问题。

    lab00 会调用这个函数。**任何时候你怀疑自己的 logprob 实现有问题，先跑它。**
    """
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    attn = torch.ones_like(ids)

    with torch.no_grad():
        hf_loss = model(input_ids=ids, attention_mask=attn, labels=ids).loss.item()
        # labels=ids 时 HF 内部也会做左移，算的是第 1..L-1 个 token 的交叉熵。
        # 所以这里 completion_len 取 L-1，覆盖同样的位置集合。
        mine = per_token_logps(model, ids, attn, completion_len=ids.size(1) - 1)
        my_loss = (-mine).mean().item()

    return {
        "hf_loss": hf_loss,
        "my_loss": my_loss,
        "abs_diff": abs(hf_loss - my_loss),
        # bf16 下 1e-2 是合理容差；fp32 下应该小于 1e-4
        "ok": abs(hf_loss - my_loss) < 1e-2,
    }
