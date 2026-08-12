# ==============================================================================
# 快捷命令。所有目标都可以加参数覆盖默认值，例如：
#     make lab04 ARGS="--loss-type dr_grpo --total-steps 500"
#     make lab01 ARGS="--lora"
# ==============================================================================
PY   ?= python
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help setup test smoke download clean \
        lab00 lab01 lab02 lab03 lab04 lab05 lab06 lab07 gen eval

help:                     ## 显示所有可用命令
	@echo ""
	@echo "  \033[1mLLM 后训练与 RL 实验室\033[0m  （目标硬件：NVIDIA L4 24GB）"
	@echo ""
	@grep -E '^[a-zA-Z_0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  例：make lab04 ARGS=\"--loss-type dr_grpo --total-steps 500\""
	@echo ""

# ---------------------------------------------------------------- 环境
setup:                    ## 安装依赖并自检（第一次上服务器跑这个）
	bash setup.sh

test:                     ## 核心算法单元测试（纯 CPU，几秒钟，改代码后必跑）
	$(PY) scripts/test_core.py

download:                 ## 预下载所有模型与数据集（离线环境先跑这个）
	$(PY) scripts/download_assets.py

smoke:                    ## 所有 lab 各跑 2~3 步，验证端到端能跑通
	bash scripts/smoke_test.sh

clean:                    ## 删除训练产物（outputs/ 与 __pycache__）
	rm -rf outputs/ && find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "已清理"

# ---------------------------------------------------------------- 实验
lab00:                    ## 环境自检：GPU / logprob 正确性 / loss mask
	$(PY) labs/lab00_env_check/check.py $(ARGS)

lab01:                    ## SFT 监督微调（~8 min，教输出格式）
	$(PY) labs/lab01_sft/train_sft.py $(ARGS)

lab02:                    ## 奖励模型（Bradley-Terry，~6 min）
	$(PY) labs/lab02_reward_model/train_rm.py $(ARGS)

lab03:                    ## DPO 直接偏好优化（~9 min，观察似然位移）
	$(PY) labs/lab03_dpo/train_dpo.py $(ARGS)

lab04:                    ## ★ GRPO 强化学习（核心实验，~3 h 跑 300 步）
	$(PY) labs/lab04_grpo/train_grpo.py $(ARGS)

lab05:                    ## Reward Hacking 演示（Part A 秒级，加 --run-rl 跑 RL）
	$(PY) labs/lab05_reward_hacking/demo_hacking.py $(ARGS)

lab06:                    ## 评估：avg@k / pass@k / maj@k / 污染检查
	$(PY) labs/lab06_eval_passk/evaluate.py $(ARGS)

lab07:                    ## Agentic RL：多轮工具调用（loss mask 是重点）
	$(PY) labs/lab07_agentic_rl/train_agent_grpo.py $(ARGS)

# ---------------------------------------------------------------- 工具
gen:                      ## 采样看输出：make gen ARGS="--model outputs/lab01_sft"
	$(PY) scripts/generate.py $(ARGS)

# ---------------------------------------------------------------- 组合流程
pipeline:                 ## 完整主线：SFT → GRPO → 三方对比评测
	$(PY) labs/lab01_sft/train_sft.py
	$(PY) labs/lab04_grpo/train_grpo.py --model outputs/lab01_sft --total-steps 300
	$(PY) labs/lab06_eval_passk/evaluate.py --n 200 --k 8 --compare \
	    Qwen/Qwen2.5-0.5B-Instruct outputs/lab01_sft outputs/lab04_grpo/final
