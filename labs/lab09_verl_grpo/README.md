# Lab 09 · 用 veRL 复现 GRPO

把 [lab04](../lab04_grpo/README.md) 换成字节跳动开源的 [veRL](https://github.com/volcengine/verl)。
奖励逻辑和数据依旧复用 `common/rewards.py`，只是要先转成 veRL 要求的 parquet 格式。

```bash
pip install -r labs/lab09_verl_grpo/requirements-verl.txt
python labs/lab09_verl_grpo/prepare_data.py          # 产出 data/{train,test}.parquet

bash labs/lab09_verl_grpo/run_lab09_l4.sh --trainer.total_training_steps=3   # 冒烟
bash labs/lab09_verl_grpo/run_lab09_l4.sh                                    # 完整跑
```

## 和 lab04 / lab08 的关系

三个 lab 的**判分逻辑是同一份代码**（`common/rewards.compute_reward`），
只有训练引擎不同，方便你横向对比：

| | lab04 | lab08 | lab09 |
|:---|:---|:---|:---|
| 训练引擎 | 手写 | TRL `GRPOTrainer` | veRL `main_ppo` |
| rollout 引擎 | HF `generate`（可选 vLLM） | HF `generate` | vLLM（默认） |
| 定位 | 学原理 | 单卡/小规模生产 | 大规模分布式生产（FSDP/Megatron + 千卡） |
| 单卡 24GB 上的意义 | 完整体验 | 完整体验 | 只能跑通流程，**发挥不出它的优势** |

## 为什么 lab09 明显更"重"

veRL 是为千卡级训练设计的，即便只用单卡也绕不开它的几个核心概念：

- **数据必须是 parquet**，每行要有 `prompt` / `reward_model.ground_truth` 等固定字段
  （见 `prepare_data.py`），不能像 lab04/08 那样直接喂 Python list。
- **奖励函数走独立进程加载**：`custom_reward_function.path/.name` 指向
  `reward_fn.py::compute_score(data_source, solution_str, ground_truth, extra_info)`，
  这是 veRL 的固定接口，不能像 TRL 那样直接传函数对象。
- **配置是 Hydra 覆盖参数**，不是一个 Python dataclass —— `run_lab09_l4.sh` 里那一长串
  `actor_rollout_ref.xxx=yyy` 才是 veRL 用户真正打交道的东西。
- **rollout 默认用 vLLM**，`gpu_memory_utilization=0.5` 是专门为单卡 24GB 留出训练用的显存
  （vLLM 和训练引擎要共享同一块卡，这在 lab04/08 里不存在，是 veRL 单卡场景特有的坑）。
- **没有 critic**（`critic.enable=false`）：GRPO 本来就不需要 critic 网络，
  这一点和 lab04「为什么可以不要 critic」的结论完全一致，只是 veRL 把 PPO/GRPO
  统一在一套 trainer 里，需要显式关掉 critic。

## 关键 override 对照 lab04 的字段

| `run_lab09_l4.sh` 里的 override | ↔ lab04 `Config` | 说明 |
|:---|:---|:---|
| `algorithm.adv_estimator=grpo` | （固定用组内优势） | 选择优势估计方式，GRPO 而非 GAE |
| `actor_rollout_ref.rollout.n=8` | `group_size` | 同一题采几条 |
| `data.max_prompt_length` / `max_response_length` | `max_prompt_len` / `max_new_tokens` | 长度上限 |
| `actor_rollout_ref.actor.optim.lr` | `lr` | 同样是 `1e-6` 量级 |
| `actor_rollout_ref.actor.use_kl_loss=false` | `beta=0` | RLVR 常见做法：不加 KL 惩罚 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `micro_bs` | OOM 时先调这个 |
| `trainer.total_training_steps` | `total_steps` | 训练步数 |

## 该看什么

veRL 默认把训练指标打到 console（`trainer.logger=console`），关注 `critic/score/mean`
（等价于 lab04 的 `reward/mean`）和验证集上的 `val-core/.../acc`（等价于 lab04 的
【独立评测】那一行）。想接 wandb：把 `trainer.logger=console` 换成 `trainer.logger=[console,wandb]`。

## 什么时候真的该用 veRL

单卡 24GB 场景下用 veRL 纯粹是"学一遍它的接口长什么样"——性能上不会比 lab04/lab08 更快，
配置复杂度反而最高。它真正的价值在你需要下面这些东西的时候：

- 模型大到单卡放不下，要 FSDP/Megatron 做模型并行
- rollout 吞吐是瓶颈，要多副本 vLLM 引擎并行采样
- 要在几十~几百张卡上做长时间的生产级 RL 训练

如果只是想跑一次 GRPO 看看效果，**优先 lab04（学原理）或 lab08（TRL，够用且轻量）**。

## 版本注意

veRL 的 Hydra 配置项随版本变化较快（尤其 `actor_rollout_ref.rollout.*` 下的字段）。
如果某个 override 报 `Could not find` 之类的 Hydra 报错：

```bash
python3 -c "import verl, os; print(os.path.dirname(verl.__file__))"
# 去它的 trainer/config/*.yaml 里对照当前版本的字段名
```
