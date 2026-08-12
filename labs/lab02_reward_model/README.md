# Lab 02 · 奖励模型（Bradley-Terry）

把「人类的成对偏好」变成一个可微的打分函数 `r_φ(x, y)`。

```bash
make lab02
make lab02 ARGS="--n-train 4000"
```

## 核心数学

假设每个回答有一个潜在「实力」分数，A 打败 B 的概率是 `σ(r_A − r_B)`（Bradley-Terry 模型，
和国际象棋 Elo 分是同一套）。对观察到的偏好做最大似然，取负对数：

```
L = − E[ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]
```

**这个损失只约束分差，不约束绝对值** —— 所有分数同时 +100，损失完全不变。
两个后果：

1. 分数可能整体漂移到 ±1000 → 代码里加了 L2 正则拉住
2. `reward = 3.5` 这个数没有跨模型可比性

这个「平移不变性」正是 DPO 推导中配分函数 `Z(x)` 能被消掉的原因（见 lab03）。

## 两个容易错的实现细节

```python
# ❌ 常见 bug：右 padding 时 hs[:, -1] 取到的是 pad token
pooled = hs[:, -1]

# ✅ 取最后一个非 padding token
last_idx = attention_mask.sum(dim=1) - 1
pooled = hs[torch.arange(B), last_idx]
```

```python
# ✅ 小方差初始化。默认初始化会让初始 reward 方差很大，前几十步剧烈震荡
nn.init.normal_(self.v_head.weight, std=1.0 / (hidden + 1) ** 0.5)
```

## 该看什么

**只训 1 个 epoch**。RM 极易过拟合 —— 第 2 轮训练准确率冲到 95%，验证准确率反而下降。

| 验证集准确率 | 含义 |
|:---|:---|
| < 0.55 | 接近瞎猜。查数据加载、学习率、pooling 位置 |
| 0.65 ~ 0.80 | ✅ 正常。人类标注员之间的一致率也只有 70~80% |
| > 0.85 | ⚠️ 可疑：过拟合，或 RM 找到了捷径（比如只数长度） |

脚本跑完会自动给出这个解读。
