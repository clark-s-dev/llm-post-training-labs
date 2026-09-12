"""
Lab 10 —— veRL 的自定义奖励函数

和 lab09 完全一样：直接复用 common/rewards.py，只判最终答案（<think> + \\boxed{}）。
工具调用本身不参与打分 —— 和 lab07 的 agent_reward() 设计原则一致：
「什么时候该调工具」正是我们想让 RL 自己学的，不能直接奖励「调用了工具」这件事。
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
