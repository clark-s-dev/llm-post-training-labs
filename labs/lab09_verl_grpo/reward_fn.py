"""
Lab 09 —— veRL 的自定义奖励函数

veRL 通过 `custom_reward_function.path` / `.name` 加载这个文件里的一个函数，
签名是固定的（veRL 框架的约定，不能改名/改参数顺序）：

    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

这里直接复用 common/rewards.py 的判分逻辑 —— 和 lab04/lab08 完全同一套规则
（格式门控 + 正确性 + 超长/复读软惩罚），保证三个 lab 的奖励含义一致，
差异只来自训练框架本身。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.rewards import RewardConfig, compute_reward

_RCFG = RewardConfig()


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    reward, _parts = compute_reward(solution_str, ground_truth, _RCFG)
    return reward
