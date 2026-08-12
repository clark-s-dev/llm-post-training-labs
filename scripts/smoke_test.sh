#!/usr/bin/env bash
# ==============================================================================
# 全 lab 冒烟测试：每个 lab 只跑 2~3 步，验证端到端能跑通。
# 目的是「代码没坏」，不是「模型训得好」—— 所以指标数字难看是正常的。
#
#   bash scripts/smoke_test.sh              # 全部
#   bash scripts/smoke_test.sh 04 07        # 只测指定的 lab
#
# L4 上全跑一遍约 8~15 分钟（主要花在下载和生成上）。
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
BOLD=$'\033[1m'; GREEN=$'\033[92m'; RED=$'\033[91m'; NC=$'\033[0m'

WANT="${*:-00 01 02 03 04 05 06 07}"
declare -a FAILED=()

run() {   # run <编号> <说明> <命令...>
  local id="$1" desc="$2"; shift 2
  [[ " $WANT " == *" $id "* ]] || return 0
  echo; echo "${BOLD}━━━ lab$id · $desc ━━━${NC}"
  if "$@" > "/tmp/smoke_lab$id.log" 2>&1; then
    echo "${GREEN}✓ lab$id 通过${NC}"
  else
    echo "${RED}✗ lab$id 失败${NC}  —— 最后 15 行："
    tail -15 "/tmp/smoke_lab$id.log" | sed 's/^/    /'
    FAILED+=("$id")
  fi
}

echo "${BOLD}冒烟测试开始${NC}（日志在 /tmp/smoke_lab*.log）"

run 00 "环境自检"        $PY labs/lab00_env_check/check.py
run 01 "SFT"             $PY labs/lab01_sft/train_sft.py --smoke
run 02 "奖励模型"        $PY labs/lab02_reward_model/train_rm.py --smoke
run 03 "DPO"             $PY labs/lab03_dpo/train_dpo.py --smoke
run 04 "GRPO"            $PY labs/lab04_grpo/train_grpo.py --smoke
run 05 "Reward Hacking"  $PY labs/lab05_reward_hacking/demo_hacking.py
run 06 "评估"            $PY labs/lab06_eval_passk/evaluate.py --n 4 --k 4 --max-new-tokens 128
run 07 "Agentic RL"      $PY labs/lab07_agentic_rl/train_agent_grpo.py --smoke

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "${GREEN}${BOLD}全部通过 ✓${NC}  可以开始正式训练了：make lab01"
  exit 0
fi
echo "${RED}${BOLD}失败的 lab: ${FAILED[*]}${NC}"
echo "查看详细日志：cat /tmp/smoke_lab<编号>.log"
exit 1
