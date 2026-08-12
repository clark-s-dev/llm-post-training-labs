#!/usr/bin/env bash
# ==============================================================================
# 一键环境安装 —— 在 L4 GPU 服务器上跑这个脚本就行
#
#   bash setup.sh              # 默认：建 venv + 装依赖 + 自检
#   bash setup.sh --no-venv    # 装到当前 Python 环境（已有 conda env 时用这个）
#   bash setup.sh --with-vllm  # 额外装 vLLM（rollout 快 5~15 倍，但版本较挑）
#
# 脚本做的事：
#   1. 检查 GPU 与驱动
#   2. 建 venv（可跳过）
#   3. 装 PyTorch —— **如果已有带 CUDA 的 torch 就不动它**，避免破坏现成环境
#   4. 装其余依赖
#   5. 跑纯 CPU 的算法自测（几秒）
#   6. 提示下一步
# ==============================================================================
set -euo pipefail

USE_VENV=1
WITH_VLLM=0
for arg in "$@"; do
  case "$arg" in
    --no-venv)   USE_VENV=0 ;;
    --with-vllm) WITH_VLLM=1 ;;
    -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg"; exit 1 ;;
  esac
done

BOLD=$'\033[1m'; GREEN=$'\033[92m'; YELLOW=$'\033[93m'; RED=$'\033[91m'; NC=$'\033[0m'
step() { echo; echo "${BOLD}▸ $*${NC}"; }

# ------------------------------------------------------------------ 1. GPU
step "1/6  检查 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    /'
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  if echo "$GPU_NAME" | grep -qi "L4"; then
    echo "    ${GREEN}✓ 检测到 L4 —— 本仓库的默认参数就是按它调的${NC}"
  else
    echo "    ${YELLOW}! 不是 L4（$GPU_NAME）。参数按 24GB 显存调的，显存更小的话要减 --micro-bs${NC}"
  fi
else
  echo "    ${YELLOW}! 没找到 nvidia-smi。CPU 上只能用 --smoke 跑通逻辑，训练会非常慢。${NC}"
fi

# ------------------------------------------------------------------ 2. venv
step "2/6  Python 环境"
PY=python3
if [ "$USE_VENV" = "1" ]; then
  if [ ! -d .venv ]; then
    echo "    创建 .venv ..."
    $PY -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
  echo "    ${GREEN}✓ 已激活 .venv${NC}（之后每次开新终端记得 source .venv/bin/activate）"
else
  echo "    使用当前环境: $(which $PY)"
fi
$PY -c "import sys; print(f'    Python {sys.version.split()[0]}')"
$PY -m pip install -q --upgrade pip

# ------------------------------------------------------------------ 3. torch
step "3/6  PyTorch"
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  V=$($PY -c "import torch; print(torch.__version__)")
  echo "    ${GREEN}✓ 已有可用的 CUDA PyTorch ($V) —— 跳过安装，避免破坏现成环境${NC}"
else
  echo "    安装 PyTorch (CUDA 12.4 wheel)..."
  # cu124 的 wheel 兼容 CUDA 12.x 驱动，覆盖绝大多数 L4 云主机
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124 \
    || { echo "    ${YELLOW}! cu124 源失败，退回默认源${NC}"; $PY -m pip install -q torch; }
fi

# ------------------------------------------------------------------ 4. 其余依赖
step "4/6  其余依赖"
$PY -m pip install -q -r requirements.txt
echo "    ${GREEN}✓ 完成${NC}"

if [ "$WITH_VLLM" = "1" ]; then
  step "4b/6  vLLM（可选）"
  echo "    ${YELLOW}注意：vLLM 会按自己的需求锁定 torch 版本，可能覆盖上面装的${NC}"
  $PY -m pip install -q vllm || echo "    ${RED}✗ vLLM 安装失败 —— 不影响其他 lab，rollout 会退回 HF generate${NC}"
fi

# ------------------------------------------------------------------ 5. 自测
step "5/6  算法自测（纯 CPU，不下载模型）"
$PY scripts/test_core.py

# ------------------------------------------------------------------ 6. 下一步
step "6/6  下一步"
cat <<EOF
    ${GREEN}环境就绪。${NC}建议按顺序：

      1) 硬件与正确性全面自检（会下载 0.5B 模型，约 1GB）
           ${BOLD}make lab00${NC}      或  python labs/lab00_env_check/check.py

      2) 预下载所有模型/数据（可选，适合网络慢或要离线跑的场景）
           ${BOLD}python scripts/download_assets.py${NC}

      3) 全部 lab 冒烟测试（每个 2~3 步，约 10 分钟，验证端到端能跑）
           ${BOLD}make smoke${NC}

      4) 开始第一个真正的训练
           ${BOLD}make lab01${NC}      # SFT，约 8 分钟

    完整说明见 README.md，L4 专属调优见 docs/L4_NOTES.md
EOF
