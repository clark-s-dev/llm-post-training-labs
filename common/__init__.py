"""所有 lab 共用的工具模块。

  gpu.py         设备/精度/attention 实现的选择，显存检查
  logprobs.py    ★ 核心：per-token log-prob、KL 估计器
  rewards.py     RLVR 的奖励函数（含防 hack 设计）
  data.py        chat template、loss mask、GSM8K / 偏好数据加载
  train_utils.py 优化器、调度器、日志、checkpoint
"""
