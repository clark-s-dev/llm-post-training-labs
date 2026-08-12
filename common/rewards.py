"""
奖励函数（RLVR 的核心）。

在 RLHF 里，奖励来自一个学出来的神经网络（奖励模型）；
在 RLVR（可验证奖励强化学习）里，奖励就是几个 if 语句 —— 但这几个 if 极其难写对。

本文件的设计原则（每一条都对应一个真实踩过的坑）：
    1. 只认最后一个 \\boxed{}      —— 否则模型会学会「把所有候选答案都列一遍」来碰运气
    2. 格式奖励做成「门控」而非加分 —— 否则模型会学会「格式完美 + 答案乱写」
    3. 数值等价而非字符串相等      —— "1,000" / "1000.0" / "$1000" 得判成同一个答案
    4. 任何解析都不能抛异常        —— 一条脏样本不该让整个训练崩掉
    5. 返回分项而非只返回总分      —— 训练时要能看出「总分涨了但正确率没涨」
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 1. 从模型输出里把答案抠出来
# --------------------------------------------------------------------------
def extract_last_boxed(text: str) -> str | None:
    r"""
    抓取**最后一个** \boxed{...} 里的内容，支持嵌套花括号（如 \boxed{\frac{1}{2}}）。

    为什么必须是「最后一个」而不是「第一个」或「全文搜索」？
        这是一个真实的 reward hacking 入口。如果用全文匹配，模型会学会输出：
            "答案可能是 1，也可能是 2，也可能是 42……"
        总有一个命中标准答案，白拿满分。只认最后一个 boxed 就堵死了这条路
        （模型必须做出唯一承诺）。
    """
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx == -1:
        return None

    i = idx + len(key)
    depth = 1                 # 已经进入一层花括号
    buf: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:            # 找到配对的右括号
                return "".join(buf).strip()
        buf.append(ch)
        i += 1
    return None                       # 括号没闭合（通常是被 max_tokens 截断了）


def extract_gsm8k_answer(solution: str) -> str:
    """GSM8K 数据集的标准答案写在 '#### 42' 后面。"""
    return solution.split("####")[-1].strip().replace(",", "")


# 清洗数值字符串：去掉千分位逗号、货币符号、百分号、空白、LaTeX 的 \! \, 之类的间隔符
_CLEAN = re.compile(r"[,\s$%]|\\[!,;:]|\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\left|\\right")


def _to_number(s: str) -> float | None:
    """尽力把字符串转成数字；转不了返回 None（不抛异常）。"""
    if s is None:
        return None
    s = _CLEAN.sub("", str(s)).strip()
    if not s:
        return None

    # 处理 LaTeX 分数：\frac{3}{4} 或 \dfrac{3}{4}
    m = re.fullmatch(r"\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        try:
            den = float(m.group(2))
            return float(m.group(1)) / den if den != 0 else None
        except ValueError:
            return None

    # 处理 "3/4" 这种朴素分数
    m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", s)
    if m:
        try:
            den = float(m.group(2))
            return float(m.group(1)) / den if den != 0 else None
        except ValueError:
            return None

    try:
        return float(s)
    except ValueError:
        return None


def answers_equal(pred: str | None, gold: str | None, tol: float = 1e-6) -> bool:
    """
    判断两个答案是否等价。先试数值比较，失败再退回字符串比较。

    可选增强：装了 math-verify 之后会优先用它（能处理符号表达式、区间、集合等）。
        pip install math-verify
    """
    if pred is None or gold is None:
        return False

    a, b = _to_number(pred), _to_number(gold)
    if a is not None and b is not None:
        return abs(a - b) < tol

    # 数值路走不通 —— 试试 math-verify（可选依赖，没装就跳过）
    try:
        from math_verify import parse, verify           # type: ignore
        return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))
    except Exception:
        pass

    return str(pred).strip() == str(gold).strip()


# --------------------------------------------------------------------------
# 2. 格式检查
# --------------------------------------------------------------------------
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# 本仓库统一使用的输出格式。lab01(SFT) 教它，lab04(GRPO) 奖励它。
RESPONSE_FORMAT = "<think>\n{reasoning}\n</think>\n\n答案是 \\boxed{{{answer}}}"


def has_valid_format(text: str, min_think_chars: int = 20) -> bool:
    """
    格式是否合规：既要有非空的 <think>…</think>，也要有 \\boxed{}。

    min_think_chars 是防 hack 的：不加这个限制，模型会输出空的
    <think></think> 来白拿格式分（真实发生过）。
    """
    m = THINK_RE.search(text)
    if m is None or len(m.group(1).strip()) < min_think_chars:
        return False
    return extract_last_boxed(text) is not None


def count_repetition(text: str, n: int = 12) -> float:
    """
    n-gram 重复率，用来检测「复读机」失败模式。
    返回 0~1，越接近 1 说明重复越严重。正常文本一般 < 0.2。
    """
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


# --------------------------------------------------------------------------
# 3. 组合奖励
# --------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """
    奖励的配置。默认值就是本仓库推荐的「门控式」设计：

        格式不合规            → 直接 0 分，没有任何商量余地（gate）
        答对                  → +1.0                        （主奖励）
        超长被截断            → 软惩罚，随超出长度线性增大   （塑形，DAPO 的做法）
        复读                  → 软惩罚                       （塑形）

    为什么是「门控」而不是「加权和」？
        加权和允许模型做交易 —— 牺牲难拿的正确性分，去刷容易拿的格式分。
        门控把「必须满足的条件」变成硬前提，消除了这种交易空间。
    """
    correct_reward: float = 1.0
    format_is_gate: bool = True        # True=不合格式直接 0；False=当作 0.2 的加分项
    format_bonus: float = 0.2          # 仅当 format_is_gate=False 时生效
    min_think_chars: int = 20

    # 超长软惩罚（DAPO: Soft Overlong Punishment）
    overlong_penalty: float = 0.5      # 最大惩罚幅度
    overlong_cache: int = 128          # 缓冲区大小（token）；在最后这么多 token 内线性加罚

    # 复读惩罚
    repetition_penalty: float = 0.5
    repetition_threshold: float = 0.35

    tags: list[str] = field(default_factory=list)


def compute_reward(
    completion: str,
    ground_truth: str,
    cfg: RewardConfig | None = None,
    n_tokens: int | None = None,
    max_tokens: int | None = None,
    truncated: bool = False,
) -> tuple[float, dict[str, float]]:
    """
    返回 (总奖励, 分项字典)。

    ★ 一定要把分项也记到日志里。训练时你会经常看到这种情况：
          total  ↗↗↗  看起来很棒
          correct →→→  其实一动没动
      只看总分完全发现不了模型在刷格式分/长度分。
    """
    cfg = cfg or RewardConfig()
    parts: dict[str, float] = {}

    ok_format = has_valid_format(completion, cfg.min_think_chars)
    parts["format"] = 1.0 if ok_format else 0.0

    pred = extract_last_boxed(completion)
    correct = answers_equal(pred, ground_truth)
    parts["correct"] = 1.0 if correct else 0.0

    # ---- 门控：格式不合规直接判 0，后面的奖励一概不给 ----
    if cfg.format_is_gate and not ok_format:
        parts["total"] = 0.0
        return 0.0, parts

    total = cfg.correct_reward * parts["correct"]
    if not cfg.format_is_gate:
        total += cfg.format_bonus * parts["format"]

    # ---- 塑形项 1：超长软惩罚 ----
    # 被 max_tokens 截断的回答，本来可能是对的，只是没写完。
    # 直接给 0 等于用噪声惩罚模型（"你这个推理是错的"），会明显拖慢收敛。
    # 这里改成随长度平滑过渡的惩罚。
    if n_tokens is not None and max_tokens is not None and cfg.overlong_penalty > 0:
        soft_start = max_tokens - cfg.overlong_cache
        if n_tokens > soft_start:
            frac = min(1.0, (n_tokens - soft_start) / max(1, cfg.overlong_cache))
            parts["overlong"] = -cfg.overlong_penalty * frac
            total += parts["overlong"]

    # ---- 塑形项 2：复读惩罚 ----
    if cfg.repetition_penalty > 0:
        rep = count_repetition(completion)
        if rep > cfg.repetition_threshold:
            parts["repetition"] = -cfg.repetition_penalty
            total += parts["repetition"]

    parts["total"] = total
    return total, parts


# --------------------------------------------------------------------------
# 4. 故意写坏的奖励函数 —— lab05 用它演示 reward hacking
# --------------------------------------------------------------------------
def naive_reward_contains_answer(completion: str, ground_truth: str) -> tuple[float, dict]:
    """
    ❌ 一个看起来很合理、实际漏洞巨大的奖励函数：
       「只要回答里出现了正确答案这个数字，就给 1 分。」

    模型学到的 hack：把 0 到 100 全列一遍，或者反复罗列候选答案。
    lab05 会真的把它跑起来，让你看到 reward 曲线飙升而准确率不动。
    """
    hit = str(ground_truth).strip() in completion
    return (1.0 if hit else 0.0), {"correct": float(hit), "total": float(hit)}


def naive_reward_length(completion: str, ground_truth: str) -> tuple[float, dict]:
    """
    ❌ 另一个经典坏奖励：「回答越长越好」（模拟 RM 学到的长度偏见）。
       模型学到的 hack：疯狂输出废话或复读。
    """
    r = min(1.0, len(completion) / 2000)
    return r, {"total": r, "correct": 0.0}
