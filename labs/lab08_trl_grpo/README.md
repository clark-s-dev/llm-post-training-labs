# Lab 08 · 用 TRL 复现 GRPO

把 [lab04](../lab04_grpo/README.md) 换成 HuggingFace [TRL](https://github.com/huggingface/trl) 的
`GRPOTrainer`。奖励函数和数据集**原封不动复用** `common/rewards.py` / `common/data.py` ——
唯一变的是训练循环本身。

```bash
pip install -r labs/lab08_trl_grpo/requirements-trl.txt

make lab08 ARGS="--smoke"                              # 3 步冒烟
make lab08 ARGS="--model outputs/lab01_sft"             # 推荐：从 SFT 起步，同 lab04
```

## 和 lab04 的对应关系

lab04 手写暴露的每个细节，在 TRL 里都变成了一个配置项：

| lab04（手写） | lab08（TRL） | 说明 |
|:---|:---|:---|
| `group_size` | `num_generations` | 同一题采几条回答，含义完全一样 |
| 左 padding + `completion_mask` | 内部处理 | TRL 自己管生成时的 padding 和 EOS mask，你不用再手写这部分 |
| `beta`（KL 系数） | `beta` | 语义相同，0 表示不加参考模型 |
| `eps_low` / `eps_high`（clip-higher） | `epsilon` / `epsilon_high` | 版本较新的 TRL 才有非对称裁剪，见下方「版本注意」 |
| `--loss-type dapo/dr_grpo/...` | `loss_type` | TRL 的 `GRPOConfig.loss_type` 支持类似的几种归一化方式，具体可选项随版本变化 |
| `RewardConfig` 门控奖励 | `reward_funcs=gsm8k_reward` | 直接传函数，签名 `(completions, **dataset_columns) -> list[float]` |
| 手写训练循环 | `trainer.train()` | 采样、算 old_logps、裁剪、反传、日志全部封装掉了 |

## 该看什么

跟 lab04 一样看 `acc`（TRL 会把奖励均值/标准差打到日志里），
以及 `make lab06` 的独立评测 —— 训练 reward 涨、独立评测不涨就是在 hack 奖励，
这一点**换了框架也不会变**，因为奖励函数是同一份代码。

## 版本注意

TRL 的 `GRPOConfig` 字段随版本迭代较快（`loss_type` 的可选值、`epsilon_high` 是否存在
都可能因版本而异）。如果某个参数报 `unexpected keyword argument`，
先跑 `python -c "from trl import GRPOConfig; help(GRPOConfig)"` 看当前版本支持什么，
再对照上表调整 `train_grpo_trl.py` 里 `GRPOConfig(...)` 的构造参数。

## 什么时候该用框架而不是手写

- **学原理**：手写（lab04/lab07）—— 每一行都能改，能真正验证「我理解对了」
- **训更大模型 / 要多卡 / 要接入现成的 vLLM 加速 rollout**：TRL —— 成熟、维护活跃、生态大
- **要 Megatron 级别的大规模训练（千卡、MoE）**：见 [lab09（veRL）](../lab09_verl_grpo/README.md)
