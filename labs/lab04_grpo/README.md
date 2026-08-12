# Lab 04 · GRPO 强化学习 ★ 核心实验

用可验证奖励（答案对不对）做 RL，在 GSM8K 上真正提升 0.5B 模型的准确率。

```bash
make lab04 ARGS="--model outputs/lab01_sft"      # 推荐：从 SFT 起步
make lab04 ARGS="--smoke"                        # 3 步冒烟
```

## 算法

**GRPO = PPO 的裁剪目标 + 用「同一道题采 G 条答案的组内均值」替代 critic 网络。**

```
优势   Â_i = (r_i − mean(r)) / std(r)              ← 整条回答共用一个值
目标   min( ρ_{i,t}·Â_i , clip(ρ_{i,t}, 1−ε_low, 1+ε_high)·Â_i )
其中   ρ_{i,t} = π_θ(o_{i,t}|·) / π_old(o_{i,t}|·)
```

**为什么可以不要 critic？** LLM 的 MDP 里 γ=1 且奖励只在最后一个 token 给，
所以每个位置的回报都等于同一个终局奖励 —— baseline 只需要一个数，
而不需要一个能预测每个位置的网络。组内均值就是这个数的无偏蒙特卡洛估计。
顺带省掉一个和 policy 同样大的模型。

核心代码就是 `grpo_loss()` 里的这几行：

```python
ratio = torch.exp(logps - old_logps)
loss1 = ratio * adv
loss2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
per_token_loss = -torch.min(loss1, loss2)          # 负号：最大化 → 最小化
```

## 五种变体：一个分母引发的三篇论文

```bash
make lab04 ARGS="--loss-type grpo"      # 原版：Σ_i (1/|o_i|) Σ_t  → 有长度偏差
make lab04 ARGS="--loss-type dr_grpo"   # Dr.GRPO：去掉 1/|o_i| 和 ÷std
make lab04 ARGS="--loss-type dapo"      # DAPO（默认）：分母 = 全 batch token 总数
make lab04 ARGS="--loss-type gspo"      # GSPO：clip 提到序列级
make lab04 ARGS="--loss-type cispo"     # CISPO：裁 IS 权重而非梯度
```

**长度偏差**（`--loss-type grpo` 能复现）：因为除以 `|o_i|`，一条 100 token 的回答里
每个 token 的权重是一条 10000 token 回答的 100 倍。对**错误**回答来说，
写得越长惩罚被摊得越薄 → 模型学会「要错就错得长一点」→ 长度爆炸。

```bash
# 复现长度爆炸：看 len/mean 一路涨而 acc 不动
make lab04 ARGS="--loss-type grpo --no-dynamic-sampling --total-steps 200"
```

**难度偏差**（`--no-norm-adv-by-std` 可关掉）：二值奖励下 `std = √(p(1−p))`，
p=0.5 时放大 2 倍，p=0.9 时放大 3.3 倍 —— 太简单和太难的题反而拿到更大权重。

## DAPO 的两个技巧（默认开启）

**Clip-Higher**（`--eps-high 0.28`）：对称裁剪对低概率 token 极不公平 ——
概率 0.01 的 token 上界 1.2 意味着最多涨到 0.012，而概率 0.9 的可以涨到 1.0。
低概率 token 正是探索的来源，它们没有上升空间就会**熵坍塌**。
只调高上界、下界保持 0.2（下界也放大的话，低概率 token 会被压到 0，更糟）。

**动态采样**（`--no-dynamic-sampling` 可关）：丢掉「全对」和「全错」的组 ——
它们的优势全是 0，梯度为 0，白白花了 G 次生成的算力。

## 该看什么

| 指标 | 健康值 | 异常含义 |
|:---|:---|:---|
| `acc` | 稳步上升 | ★ 真正关心的指标 |
| `reward/std` | 0.3~0.5 | → 0 表示所有样本得分相同，**没信号了** |
| `adv/absmean` | > 0 | → 0 表示梯度为 0（脚本会自动报警） |
| `len` | 缓慢上升后稳定 | 暴涨到 max = 长度偏差没修 |
| `trunc` | < 10% | > 20% 说明 max_new_tokens 不够 |
| `clip` | 1%~15% | > 40% 步子太大；= 0 且 μ=1 是正常的 |
| `ratio_mean` | ≈ 1.0 | 偏离说明采样与训练分布不一致（查温度/精度） |
| 【独立评测】 | 跟随 acc | **与训练 reward 背离 = 在 hack 奖励，立刻停** |

每 10 步会打印一条**奖励最高**的真实输出。数字看不出 reward hacking，眼睛能。

## 12 个代码里标了 ★ 的坑

1. `padding_side="left"` —— 生成必须左 padding
2. `logits / temperature` —— 训练时也要除，和采样一致
3. `log_softmax` 前转 fp32 —— bf16 精度不够
4. `do_sample=True` —— greedy 会让整组相同，优势全 0
5. `top_p=1.0, top_k=0` —— 截断采样会让实际分布 ≠ 算 logp 的分布
6. `lr=1e-6` —— 比 SFT 小 10~20 倍
7. completion_mask 保留 EOS —— 模型要学会停
8. 全 batch token 数做分母 —— 消除长度偏差
9. `std + 1e-4` 而非 `1e-8` —— 全对/全错组的 std=0
10. 动态采样 —— 过滤零梯度组
11. `mu=1` 时 ρ≡1，clip 不触发（想复用数据就调大，但要盯 clipfrac）
12. `logits_to_keep` —— 只对需要的位置算 lm_head，省 GB 级显存

## 预期结果

0.5B 从 SFT 起步，300 步后 GSM8K avg@8 大约从 0.34 → 0.55。
用 lab06 做严谨对比：

```bash
make lab06 ARGS="--n 200 --k 8 --compare Qwen/Qwen2.5-0.5B-Instruct outputs/lab01_sft outputs/lab04_grpo/final"
```
