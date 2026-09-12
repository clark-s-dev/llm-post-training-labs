#!/usr/bin/env bash
# Lab 10 —— 用 veRL 复现 lab07 的 Agentic RL（多轮工具调用），单张 L4 (24GB)
#
# 和 lab09（单轮）的区别集中在 actor_rollout_ref.rollout.multi_turn.* 这几行：
# 开多轮、指定工具、指定工具调用的文本格式解析器。loss mask（哪些 token 参与
# 梯度）不需要你自己写 —— ToolAgentLoop 会自动把工具返回的 token 从
# response_mask 里排除，这正是 lab07 里"整个 Agentic RL 最关键的一行"在
# 框架里的等价物。
#
# ★ 多轮工具调用官方推荐 rollout.name=sglang（vLLM 的 async 模式也能支持，
#   但 sglang 是当前测试最充分的路径，见 README「版本注意」）。
#
# 前置：
#   pip install -r labs/lab10_verl_agentic_rl/requirements-verl.txt
#   python labs/lab10_verl_agentic_rl/prepare_data.py
#
# 用法：
#   bash labs/lab10_verl_agentic_rl/run_lab10_l4.sh
#   bash labs/lab10_verl_agentic_rl/run_lab10_l4.sh --trainer.total_training_steps=3   # 冒烟

set -euo pipefail
cd "$(dirname "$0")/../.."   # 回到仓库根目录，parquet/工具路径按仓库根写

DATA_DIR=labs/lab10_verl_agentic_rl/data
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=400 \
    data.max_response_length=700 \
    actor_rollout_ref.model.path="${MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.function_tool_path=labs/lab10_verl_agentic_rl/tools.py \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=128 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    agent.default_agent_loop=tool_agent \
    reward.custom_reward_function.path=labs/lab10_verl_agentic_rl/reward_fn.py \
    reward.custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=100 \
    trainer.save_freq=25 \
    trainer.test_freq=10 \
    trainer.default_local_dir=outputs/lab10_verl_agentic_rl \
    trainer.logger=console \
    "$@"

echo
echo "下一步：python labs/lab06_eval_passk/evaluate.py --model outputs/lab10_verl_agentic_rl/<最新checkpoint> --n 200"
