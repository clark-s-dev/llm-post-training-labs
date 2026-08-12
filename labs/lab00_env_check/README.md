# Lab 00 · 环境自检

跑任何训练之前先跑这个。**它不训练模型，只验证环境和代码是对的。**

```bash
make lab00
# 或 python labs/lab00_env_check/check.py --no-download   # 只查硬件，不下模型
```

## 它检查什么

| # | 检查项 | 为什么重要 |
|:--|:---|:---|
| 1 | GPU 型号、compute capability、显存、bf16 支持 | 不是 L4 的话默认参数可能要调 |
| 2 | 矩阵乘法能跑、算力在合理区间 | 排除驱动问题和被别的进程占卡 |
| 3 | HuggingFace 能下模型和数据集 | 排除网络/代理问题 |
| 4 | **per-token log-prob 与 HF 官方 loss 逐位一致** | ★ 最重要 |
| 5 | chat template 与 loss mask 正确 | SFT 头号事故来源 |
| 6 | 真实前向+反向的显存占用 | 提前知道够不够 |

## 第 4 项为什么最重要

自回归模型在位置 `i` 输出的 logits 预测的是位置 `i+1` 的 token。
如果对齐错了一位，**训练不会报错**，但：

- SFT 的 loss 降不下去
- DPO 的隐式奖励是错的
- GRPO 的重要性比率 `ρ` 恒等于奇怪的值

这个检查用 HF 官方的 `model(labels=ids).loss` 当参照物，两者必须完全相等。

## 预期输出

全绿：

```
 ✓ log-prob 对齐   HF=1.234567  本仓库=1.234567  差=0.00e+00
 ✓ loss mask 三项断言   可学 36/99 tokens (36%)
 ✓ EOS 在可学的一侧
 ✓ 训练文本以推理 prompt 为前缀
 ✓ 全部 N 项检查通过
```

任何一项 ✗ 都不要往下跑，先解决。处理办法见 [`docs/L4_NOTES.md`](../../docs/L4_NOTES.md)。
