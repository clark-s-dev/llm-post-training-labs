#!/usr/bin/env python3
"""
核心算法单元测试 —— 纯 CPU，几秒钟跑完，不需要下载任何模型。

它验证的是「数学有没有写对」，而不是「模型训得好不好」：
    · per-token log-prob 的对齐（差一位 bug）
    · 组内优势的计算（GRPO / Dr.GRPO）
    · GRPO 损失的**梯度方向**是否正确（优势为正 → 概率上升）
    · PPO 裁剪是否真的在该切断梯度的时候切断了
    · 五种 loss_type 是否都能跑通并产生非零梯度
    · KL 估计器的数学性质
    · 奖励函数的防 hack 行为

改了 common/ 或 lab04 的算法之后，先跑这个再去烧 GPU：
    python scripts/test_core.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from common.logprobs import kl_penalty, per_token_logps, selective_log_softmax, sequence_logps
from common.rewards import (RewardConfig, answers_equal, compute_reward, extract_last_boxed,
                            has_valid_format, naive_reward_contains_answer)

PASS, FAIL = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f" {PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def section(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 70)


# ==========================================================================
# 一个最小的假 LM：只实现 per_token_logps 需要的接口
# ==========================================================================
class FakeLM(nn.Module):
    """
    行为像 AutoModelForCausalLM，但只有一个 embedding + 线性层。
    用它测试算法逻辑，避免下载真实模型（几秒 vs 几分钟）。
    """

    def __init__(self, vocab: int = 64, hidden: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                logits_to_keep=None, **kw):
        h = self.emb(input_ids)
        logits = self.head(h)
        if logits_to_keep:                      # 模拟 transformers 的行为：只返回最后 N 个位置
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


# ==========================================================================
# 1. log-prob 对齐
# ==========================================================================
def test_logprobs() -> None:
    section("1. per-token log-prob（对齐正确性）")
    torch.manual_seed(0)
    m = FakeLM()
    ids = torch.randint(0, 64, (2, 10))
    attn = torch.ones_like(ids)
    C = 4

    got = per_token_logps(m, ids, attn, completion_len=C)
    check("输出形状", tuple(got.shape) == (2, C), f"{tuple(got.shape)}")

    # 手工算一遍参考答案：位置 i 的 logits 预测位置 i+1 的 token
    full = m(ids).logits                                  # [2, 10, V]
    ref = []
    for b in range(2):
        row = []
        for t in range(10 - C, 10):                       # completion 是最后 C 个 token
            lp = full[b, t - 1].log_softmax(-1)           # ← 用 t-1 位置的 logits
            row.append(lp[ids[b, t]])
        ref.append(torch.stack(row))
    ref = torch.stack(ref)
    check("与手工计算一致（差一位对齐）", torch.allclose(got, ref, atol=1e-5),
          f"max diff = {(got - ref).abs().max():.2e}")

    # 故意错开一位，确认测试本身有区分力（否则这个测试等于没测）
    wrong = torch.stack([torch.stack([full[b, t].log_softmax(-1)[ids[b, t]]
                                      for t in range(10 - C, 10)]) for b in range(2)])
    check("测试有区分力（错位版本确实不同）", not torch.allclose(got, wrong, atol=1e-5))

    # 温度
    check("温度会改变结果",
          not torch.allclose(per_token_logps(m, ids, attn, C, temperature=2.0), got))

    # selective_log_softmax 与朴素实现等价
    logits = torch.randn(3, 5, 20)
    idx = torch.randint(0, 20, (3, 5))
    naive = torch.gather(logits.float().log_softmax(-1), 2, idx.unsqueeze(-1)).squeeze(-1)
    check("selective_log_softmax 与朴素实现等价",
          torch.allclose(selective_log_softmax(logits, idx), naive, atol=1e-6))

    # 序列级汇总
    per_tok = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    check("sequence_logps 求和（DPO 用）",
          torch.allclose(sequence_logps(per_tok, mask), torch.tensor([-3.0, -3.0])))
    check("sequence_logps 平均（SimPO 用）",
          torch.allclose(sequence_logps(per_tok, mask, average=True), torch.tensor([-1.5, -1.0])))


# ==========================================================================
# 2. 组内优势
# ==========================================================================
def test_advantages() -> None:
    section("2. 组内优势（GRPO 的灵魂）")
    from labs.lab04_grpo.train_grpo import group_advantages

    # 一组 4 条：2 对 2 错
    r = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv, _ = group_advantages(r, 4, norm_by_std=True)
    check("均值为 0", abs(float(adv.mean())) < 1e-5, f"mean={float(adv.mean()):.2e}")
    check("答对的优势为正、答错的为负",
          bool((adv[[0, 2]] > 0).all() and (adv[[1, 3]] < 0).all()), f"{adv.tolist()}")

    # Dr.GRPO：不除 std
    adv_dr, _ = group_advantages(r, 4, norm_by_std=False)
    check("Dr.GRPO 优势 = r − mean", torch.allclose(adv_dr, torch.tensor([.5, -.5, .5, -.5])),
          f"{adv_dr.tolist()}")

    # 全对 / 全错 → 优势必须是 0（这类组对梯度零贡献，DAPO 会把它们丢掉）
    for name, rr in [("全对", torch.ones(4)), ("全错", torch.zeros(4))]:
        a, _ = group_advantages(rr, 4, True)
        check(f"{name}的组优势为 0（且不是 NaN/inf）",
              bool(torch.isfinite(a).all()) and float(a.abs().max()) < 1e-3,
              f"max|adv|={float(a.abs().max()):.2e}")

    # 难度偏差：÷std 会放大「几乎全对」的组
    a_mid, _ = group_advantages(torch.tensor([1., 1., 0., 0.]), 4, True)   # p=0.5
    a_easy, _ = group_advantages(torch.tensor([1., 1., 1., 0.]), 4, True)  # p=0.75
    check("÷std 引入难度偏差（Dr.GRPO 批评的点）",
          float(a_easy.abs().max()) > float(a_mid.abs().max()),
          f"p=0.5 → {float(a_mid.abs().max()):.2f}, p=0.75 → {float(a_easy.abs().max()):.2f}")

    # 多组独立
    r2 = torch.tensor([1., 0., 0., 0.,  1., 1., 1., 0.])
    a2, _ = group_advantages(r2, 4, False)
    check("多组各自独立标准化",
          abs(float(a2[:4].mean())) < 1e-6 and abs(float(a2[4:].mean())) < 1e-6)


# ==========================================================================
# 3. GRPO 损失与梯度
# ==========================================================================
def _make_batch(m: FakeLM, B=4, L=10, C=4):
    ids = torch.randint(0, m.vocab, (B, L))
    return {"input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "completion_mask": torch.ones(B, C, dtype=torch.long),
            "completion_len": C}


def test_grpo_loss() -> None:
    section("3. GRPO 损失与梯度方向")
    from labs.lab04_grpo.train_grpo import Config, grpo_loss

    torch.manual_seed(0)
    m = FakeLM()
    batch = _make_batch(m)
    C = batch["completion_len"]
    old = per_token_logps(m, batch["input_ids"], batch["attention_mask"], C).detach()

    cfg = Config(loss_type="dapo", beta=0.0, temperature=1.0)

    # ---- 梯度方向：优势为正，应当推高这些 token 的 log-prob ----
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss, met = grpo_loss(m, batch, old, None, adv, normalizer=float(batch["completion_mask"].sum()), cfg=cfg)
    m.zero_grad(); loss.backward()
    gnorm = sum(p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
    check("产生非零梯度", float(gnorm) > 1e-8, f"‖g‖={float(gnorm):.4f}")
    check("ratio 初始 ≈ 1.0（π_θ == π_old）", abs(met["ratio_mean"] - 1.0) < 1e-4,
          f"ratio_mean={met['ratio_mean']:.6f}")

    # 沿负梯度走一小步，正优势样本的 log-prob 应当上升
    before = per_token_logps(m, batch["input_ids"], batch["attention_mask"], C).detach()
    with torch.no_grad():
        for p in m.parameters():
            if p.grad is not None:
                p -= 0.5 * p.grad
    after = per_token_logps(m, batch["input_ids"], batch["attention_mask"], C).detach()
    up = (after[:2] - before[:2]).mean()      # 优势为正的两条
    down = (after[2:] - before[2:]).mean()    # 优势为负的两条
    check("优势>0 的序列 log-prob 上升", float(up) > 0, f"Δ={float(up):+.4f}")
    check("优势<0 的序列 log-prob 下降", float(down) < 0, f"Δ={float(down):+.4f}")

    # ---- 裁剪：ratio 超出上界且优势为正时，梯度应被切断 ----
    torch.manual_seed(1)
    m2 = FakeLM()
    b2 = _make_batch(m2)
    cur = per_token_logps(m2, b2["input_ids"], b2["attention_mask"], b2["completion_len"])
    # 人为把 old 调低很多 → ratio = exp(cur - old) 远大于 1+ε
    fake_old = (cur - 3.0).detach()
    adv_pos = torch.ones(4)
    loss_c, met_c = grpo_loss(m2, b2, fake_old, None, adv_pos,
                              normalizer=float(b2["completion_mask"].sum()), cfg=cfg)
    m2.zero_grad(); loss_c.backward()
    g_clipped = float(sum(p.grad.norm() ** 2 for p in m2.parameters() if p.grad is not None) ** 0.5)
    check("裁剪比例 = 100%（ratio 全部越界）", met_c["clipfrac"] > 0.99,
          f"clipfrac={met_c['clipfrac']:.3f}")
    check("被裁后梯度≈0（正优势不再继续推高）", g_clipped < 1e-6, f"‖g‖={g_clipped:.2e}")

    # ---- 五种 loss_type 都能跑通并产生梯度 ----
    for lt in ["grpo", "dr_grpo", "dapo", "gspo", "cispo"]:
        torch.manual_seed(2)
        mm = FakeLM(); bb = _make_batch(mm)
        oo = per_token_logps(mm, bb["input_ids"], bb["attention_mask"], bb["completion_len"]).detach()
        c = Config(loss_type=lt, beta=0.0, max_new_tokens=bb["completion_len"])
        nz = float(bb["completion_mask"].sum()) if lt in ("dapo", "gspo", "cispo") else bb["input_ids"].size(0)
        l, _ = grpo_loss(mm, bb, oo, None, torch.tensor([1., -1., 1., -1.]), nz, c)
        mm.zero_grad(); l.backward()
        g = float(sum(p.grad.norm() ** 2 for p in mm.parameters() if p.grad is not None) ** 0.5)
        check(f"loss_type={lt:<8s} 可跑且梯度非零", torch.isfinite(l) and g > 1e-9,
              f"loss={float(l):+.5f}  ‖g‖={g:.4f}")

    # ---- KL 惩罚生效 ----
    torch.manual_seed(3)
    m3 = FakeLM(); b3 = _make_batch(m3)
    o3 = per_token_logps(m3, b3["input_ids"], b3["attention_mask"], b3["completion_len"]).detach()
    ref = (o3 - 0.5).detach()
    l_nokl, _ = grpo_loss(m3, b3, o3, ref, torch.ones(4), 16.0, Config(loss_type="dapo", beta=0.0))
    l_kl, met_kl = grpo_loss(m3, b3, o3, ref, torch.ones(4), 16.0, Config(loss_type="dapo", beta=1.0))
    check("β>0 时 KL 项进入损失", float(l_kl) > float(l_nokl) and met_kl["kl"] > 0,
          f"无KL={float(l_nokl):+.4f}  有KL={float(l_kl):+.4f}  kl={met_kl['kl']:.4f}")


# ==========================================================================
# 4. KL 估计器
# ==========================================================================
def test_kl() -> None:
    section("4. KL 估计器")
    torch.manual_seed(0)
    logp = torch.randn(64, 32)
    ref = logp + torch.randn(64, 32) * 0.2
    k1, k2, k3 = (kl_penalty(logp, ref, k) for k in ("k1", "k2", "k3"))
    check("k3 恒非负（k1 不是）", bool((k3 >= -1e-6).all()) and bool((k1 < 0).any()),
          f"k3.min={float(k3.min()):.2e}  k1.min={float(k1.min()):.3f}")
    check("k3 方差 < k1 方差", float(k3.var()) < float(k1.var()),
          f"var(k1)={float(k1.var()):.4f}  var(k3)={float(k3.var()):.4f}")
    check("π_θ == π_ref 时三者都为 0",
          all(float(kl_penalty(logp, logp, k).abs().max()) < 1e-6 for k in ("k1", "k2", "k3")))


# ==========================================================================
# 5. 奖励函数
# ==========================================================================
def test_rewards() -> None:
    section("5. 奖励函数（防 hack）")
    check("只认最后一个 boxed", extract_last_boxed(r"\boxed{1} x \boxed{2}") == "2")
    check("支持嵌套花括号", extract_last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}")
    check("未闭合的 boxed 返回 None", extract_last_boxed(r"\boxed{12") is None)
    check("数值等价：1,000 == 1000", answers_equal("1,000", "1000"))
    check("数值等价：\\frac{1}{2} == 0.5", answers_equal(r"\frac{1}{2}", "0.5"))
    check("不等的答案判错", not answers_equal("42", "43"))

    good = "<think>\n" + "reasoning " * 10 + "\n</think>\n答案是 \\boxed{8}"
    check("空 think 标签被拒（hack）", not has_valid_format("<think></think>\\boxed{1}"))
    cfg = RewardConfig()
    check("格式对+答案对 → 1.0", compute_reward(good, "8", cfg)[0] == 1.0)
    check("格式错+答案对 → 0.0（门控生效）", compute_reward("\\boxed{8}", "8", cfg)[0] == 0.0)

    hack = "答案可能是 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
    check("穷举 hack 能骗过 naive reward（lab05 会演示）",
          naive_reward_contains_answer(hack, "8")[0] == 1.0)
    check("同一段文本被正确的奖励函数拒绝", compute_reward(hack, "8", cfg)[0] == 0.0)

    r_long, p_long = compute_reward(good, "8", cfg, n_tokens=1020, max_tokens=1024)
    check("超长软惩罚生效", "overlong" in p_long and r_long < 1.0,
          f"total={r_long:.3f}  overlong={p_long.get('overlong'):.3f}")


# ==========================================================================
# 6. 工具沙箱（lab07）
# ==========================================================================
def test_tool_sandbox() -> None:
    section("6. 计算器工具的安全性（lab07）")
    import time
    from labs.lab07_agentic_rl.train_agent_grpo import safe_calc

    check("正常算式", safe_calc("15*4") == "60")
    check("括号与除法", safe_calc("(3+5)/2") == "4")
    check("小指数可用", safe_calc("2**10") == "1024")
    check("除零被捕获", safe_calc("1/0").startswith("ERROR"))

    # ★ 代码注入：RL 会跑几百万次模型生成的字符串，用 eval() 就是灾难
    for evil in ['__import__("os").system("echo pwned")',
                 'open("/etc/passwd").read()',
                 '[].__class__.__mro__[1].__subclasses__()']:
        check(f"拒绝代码注入: {evil[:34]}", safe_calc(evil).startswith("ERROR"))

    # ★ DoS：Python 大整数是任意精度的，9**9**9 会算出 3.7 亿位的数，进程卡死
    for bomb in ["9**9**9", "10**10**10", "((((1+1))))**9999"]:
        t0 = time.time()
        r = safe_calc(bomb)
        dt = time.time() - t0
        check(f"抵御指数炸弹 {bomb:<18s}", r.startswith("ERROR") and dt < 0.5,
              f"{r}  用时 {dt*1000:.1f}ms")


def main() -> int:
    print("\033[1m核心算法单元测试\033[0m（纯 CPU，无需下载模型）")
    test_logprobs()
    test_advantages()
    test_grpo_loss()
    test_kl()
    test_rewards()
    test_tool_sandbox()

    print("\n" + "═" * 70)
    if _failures:
        print(f"{FAIL} {len(_failures)} 项失败：")
        for f in _failures:
            print(f"   · {f}")
        return 1
    print(f"{PASS} 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
