# Lab 01 · SFT 监督微调

把 base/instruct 模型微调成「先在 `<think></think>` 里推理、再用 `\boxed{}` 给答案」的格式。

```bash
make lab01                             # 全参微调，约 8 分钟
make lab01 ARGS="--lora"               # LoRA，约 6 分钟，显存减半
make lab01 ARGS="--smoke"              # 3 步冒烟
```

## 目标不是提升能力，是钉住格式

SFT 学不到新能力（它的最优解就是复现示范分布）。这一步的价值是让
lab04 的 GRPO 一开始就有**合规的输出可以打分** —— 否则前几十步会因为
格式不合规全部拿 0 分，训练启动极慢。

## 三个教学重点

### 1. loss mask

只对 assistant 的**回答正文**算损失。prompt 和角色头 `<|im_start|>assistant\n` 都要屏蔽
（推理时角色头是喂进去的，模型不需要生成它）。

脚本开头会把第一条样本可视化打印出来，**绿色 = 参与训练**：

```
灰色: <|im_start|>system\n你是一个数学助手...<|im_end|>\n<|im_start|>user\n...<|im_start|>assistant\n
绿色: <think>\n...\n</think>\n\n答案是 \boxed{4}<|im_end|>
```

★ 注意 `<|im_end|>` 必须在绿色一侧 —— 模型靠它学会「什么时候停」。
被 `--max-len` 截断掉 EOS 是 SFT 最常见的事故（现象：推理时停不下来）。

### 2. 梯度累积的正确归一化

```bash
make lab01                          # 正确：分母 = 全局 batch 的总 token 数
make lab01 ARGS="--naive-accum"     # 错误：每个 micro-batch 各自平均再平均
```

只有当所有 micro-batch 的 token 数完全相同时两者才等价。实际上不等，
于是 `accum=1` 和 `accum=8` 会训出不一样的模型 —— 这正是 2024 年底
HF Trainer / TRL / Axolotl 集体中招的那个 bug。

**同样的错误在 RL 的 loss 聚合里以完全相同的形式重现**（见 lab04 的 `--loss-type grpo`）。

### 3. LoRA

```bash
make lab01 ARGS="--lora --lora-r 32"
```

注意两点（代码里有注释说明）：
- `target_modules` 必须包含 MLP 的 `gate/up/down_proj`。只加 `q_proj/v_proj` 是 2021 年的老做法，MLP 占了 2/3 参数。
- LoRA 的学习率要比全参**大 10 倍**（脚本会自动乘）。用全参的 lr 训 LoRA 基本等于没训。

## 该看什么

| 指标 | 健康值 | 异常含义 |
|:---|:---|:---|
| `loss` | 2.0 → 0.7~1.2 | 降到 0.2 以下几乎肯定过拟合 |
| `eval loss` | 跟随下降 | **开始上升就该停** |
| `gnorm` | 平稳的小值 | 频繁尖峰 = 数据里有脏样本 |
| 可学 token 占比 | 30%~50% | 太低说明模板匹配有问题 |

跑完看看效果：`make gen ARGS="--model outputs/lab01_sft --gsm8k"`
