# NVIDIA L4 专属说明

## 硬件参数

| 项目 | L4 | 参考：A100 40GB |
|:---|:---|:---|
| 架构 | Ada Lovelace, **sm_89** | Ampere, sm_80 |
| 显存 | 24 GB（可用 ~22.3） | 40 GB |
| 显存带宽 | **~300 GB/s** | ~1555 GB/s |
| bf16 稠密算力 | ~121 TFLOPS | ~312 TFLOPS |
| 功耗 | 72 W | 400 W |
| NVLink | 无 | 有 |

**最需要注意的是显存带宽**：只有 A100 的 1/5。自回归生成是典型的带宽瓶颈负载
（每生成一个 token 都要把整个模型权重从显存读一遍），所以：

- 训练（前向+反向，算力瓶颈）在 L4 上大约是 A100 的 1/3 速度
- **生成（rollout）大约是 A100 的 1/5 速度**
- 而 RL 有 60~80% 的时间花在生成上 → L4 上跑 RL 会明显慢

这就是本仓库默认用 0.5B 模型、`max_new_tokens=384` 的原因。

## 显存预算（0.5B 模型，bf16）

| 项目 | 大小 | 能省吗 |
|:---|:---|:---|
| 模型权重 | 1.0 GB | LoRA |
| 梯度 | 1.0 GB | LoRA |
| AdamW 状态（fp32 m/v + master） | 6.0 GB | LoRA / 8-bit Adam |
| 激活值（开了 gradient checkpointing） | 0.5–3 GB | 减 micro_bs / max_len |
| **logits 中间张量** | **可达 8 GB** | `logits_to_keep`（本仓库已用）/ 减 micro_bs |
| 生成时的 KV cache | 1–4 GB | 减 group_size / max_new_tokens |
| 参考模型（仅 β>0 时） | 1.0 GB | 设 β=0，或用 LoRA 时禁用 adapter 当 ref |

`logits` 是最容易被忽略的一项。算一笔账：
`micro_bs=4 × seq_len=768 × vocab=151936 × 4 字节(fp32) ≈ 1.87 GB`
—— 只是一个中间张量。本仓库的 `common/logprobs.py` 用两个手段压下来：
`logits_to_keep`（只对 completion 位置算 lm_head）+ 逐行 log_softmax。

## 各 lab 的实测占用与耗时（0.5B，默认参数）

| Lab | 峰值显存 | 耗时 | 说明 |
|:---|:---|:---|:---|
| lab00 | 3 GB | 1 min | 含模型下载 |
| lab01 SFT | 11 GB | 8 min | 2000 样本 × 2 epoch |
| lab01 SFT (LoRA) | 5 GB | 6 min | `--lora` |
| lab02 RM | 9 GB | 6 min | 2000 对 |
| lab03 DPO | 12 GB | 9 min | policy + ref 两份模型 |
| lab04 GRPO | 18 GB | 25–40 s/步 | G=8, 8 题/步, max_new=384 |
| lab05 (Part B) | 12 GB | 20 min | 40 步 |
| lab06 评估 | 6 GB | 15 min | 100 题 × 8 采样 |
| lab07 Agent | 14 GB | 40–60 s/步 | 多轮生成，比单轮慢 |

## 装环境

```bash
bash setup.sh
```

脚本会自动跳过已有的 CUDA PyTorch，不会破坏你现成的环境。

### 关于 FlashAttention

L4 是 sm_89，**支持 FlashAttention-2**。但 flash-attn 需要编译安装，容易和 torch 版本冲突。
本仓库默认用 PyTorch 内置的 `sdpa`（底层同样会调 FlashAttention 内核），零安装成本，
性能差距在 0.5B 规模上可以忽略。

想强制用 flash-attn：

```bash
pip install flash-attn --no-build-isolation      # 需要几十分钟编译
ATTN_IMPL=flash_attention_2 make lab04
```

### 关于 vLLM

装了 vLLM 之后 rollout 能快 5~15 倍，L4 上对 lab04 是很大的提升。但 vLLM 会锁定
自己需要的 torch 版本，可能覆盖你装好的。建议：

```bash
bash setup.sh --with-vllm     # 或者单独装到另一个 venv
make test                     # 装完立刻验证核心逻辑还在
```

## 常见报错

### `CUDA out of memory`

按影响从大到小调：

```bash
make lab04 ARGS="--micro-bs 2"                    # ① 最有效
make lab04 ARGS="--micro-bs 2 --max-new-tokens 256"  # ②
make lab04 ARGS="--group-size 4"                  # ③ 最后手段，会让 baseline 更噪
```

顺便确认卡上没有别的进程：`nvidia-smi`。

### `RuntimeError: expected scalar type BFloat16 but found Float`

某个地方 dtype 混了。本仓库在 log_softmax 前统一转 fp32（精度需要），
如果你改过 `common/logprobs.py`，先跑 `make test`。

### 生成特别慢

L4 带宽低，这在预期内。加速手段：

1. 装 vLLM（最有效）
2. 减 `--max-new-tokens`
3. 增大 `--group-size`（批量生成比逐条生成的带宽利用率高），但会增加显存

### `nvidia-smi` 显示利用率只有 20~40%

rollout 阶段这是**正常**的 —— 自回归生成每步只算一个 token，算术强度极低，
GPU 大部分时间在等显存。这正是教程第 09 章讲的问题，也是工业界要做
异步 rollout / partial rollout 的原因。

### 多卡

本仓库为**单卡**设计，没有做分布式。有多张 L4 的话，最实用的做法是并行跑不同配置的实验：

```bash
CUDA_VISIBLE_DEVICES=0 make lab04 ARGS="--loss-type dapo    --out-dir outputs/dapo" &
CUDA_VISIBLE_DEVICES=1 make lab04 ARGS="--loss-type dr_grpo --out-dir outputs/drgrpo" &
wait
```

真要做多卡分布式训练，请用 [verl](https://github.com/volcengine/verl)。

## 长时间训练

```bash
tmux new -s grpo
source .venv/bin/activate
make lab04 ARGS="--total-steps 500" 2>&1 | tee outputs/lab04.log
# Ctrl+B 然后 D 断开
```

lab04 每 50 步存一次 checkpoint（`--save-every` 可调）。RL 崩掉是常事，
崩了之后从最近的 checkpoint 恢复：`make lab04 ARGS="--model outputs/lab04_grpo/step_150"`。
