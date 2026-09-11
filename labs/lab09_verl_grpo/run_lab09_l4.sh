#!/usr/bin/env bash
# Lab 09 —— 用 veRL 复现 lab04 的 GRPO 实验，单张 L4 (24GB)
#
# veRL 面向千卡级生产训练，配置项比 TRL（lab08）多得多。这份脚本只是
# 把它硬压到单卡 0.5B 的规模来跑通，目的是让你看懂 GRPO 在「重型框架」里
# 长什么样，不是把 veRL 的能力发挥出来（它的强项是多卡 FSDP/Megatron + vLLM
# 大规模 rollout，单卡 24GB 跑不出这个优势）。
#
# 前置：
#   pip install -r labs/lab09_verl_grpo/requirements-verl.txt
#   python labs/lab09_verl_grpo/prepare_data.py
#
# 用法：
#   bash labs/lab09_verl_grpo/run_lab09_l4.sh
#   bash labs/lab09_verl_grpo/run_lab09_l4.sh --trainer.total_training_steps=3   # 冒烟

set -euo pipefail
cd "$(dirname "$0")/../.."   # 回到仓库根目录，parquet 路径按仓库根写

DATA_DIR=labs/lab09_verl_grpo/data
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=320 \
    data.max_response_length=384 \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.enable=false \
    custom_reward_function.path=labs/lab09_verl_grpo/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=300 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.default_local_dir=outputs/lab09_verl_grpo \
    trainer.logger=console \
    "$@"

echo
echo "下一步：python labs/lab06_eval_passk/evaluate.py --model outputs/lab09_verl_grpo/<最新checkpoint> --n 200"
