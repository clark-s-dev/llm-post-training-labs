# LLM 后训练与强化学习 · 动手实验室

> 8 个从零实现的实验，覆盖 SFT → 奖励模型 → DPO → GRPO → 奖励工程 → 评估 → Agentic RL，
> 外加 2 个用主流 RL 框架（TRL / veRL）复现 GRPO 的对照实验。
> 主线不用任何 RL 框架封装，所有算法从头写，**每一行关键代码都有中文注释解释「为什么这么写」**。
> 目标硬件：**单张 NVIDIA L4（24GB）**，10 个实验全部能在上面跑完。

代码注释里提到的「教程第 N 章」指配套的理论教程（公式推导 / 论文精讲 / workflow），
本仓库不含它，但每个 lab 都可以独立读懂 —— 关键推导都写在对应的 README 和代码注释里。

---

## 30 秒上手

在你的 L4 服务器上：

```bash
ssh <你的服务器>

git clone https://github.com/clark-s-dev/llm-post-training-labs.git
cd llm-post-training-labs

bash setup.sh          # 建 venv、装依赖、跑算法自测（约 3~5 分钟）
source .venv/bin/activate

make lab00             # 环境自检：GPU / logprob 正确性 / loss mask
make lab01             # 第一个真正的训练：SFT（约 8 分钟）
```

跑完 `make lab00` 看到全绿就说明环境没问题，可以放心往下做。

---

## 实验清单

| Lab | 主题 | 你会亲手实现 | L4 耗时 | 峰值显存 |
|:---|:---|:---|:---|:---|
| **00** | 环境自检 | 验证 log-prob 实现与 HF 官方**逐位一致**、loss mask 可视化 | 1 min | 3 GB |
| **01** | SFT 监督微调 | 训练循环、loss mask、**梯度累积的正确归一化**、LoRA | 8 min | 11 GB |
| **02** | 奖励模型 | Bradley-Terry 损失、标量头、pooling 取位、防分数漂移 | 6 min | 9 GB |
| **03** | DPO | 六步推导对应的损失、隐式奖励、**似然位移现象**、IPO/cDPO/RPO | 9 min | 12 GB |
| **04** | ★ **GRPO** | 组内优势、PPO 裁剪、DAPO/Dr.GRPO/GSPO/CISPO 五种变体 | 3 h / 300 步 | 18 GB |
| **05** | Reward Hacking | 奖励函数攻防演练 + **真的用坏奖励训一次，看曲线骗过你** | 1 min / 20 min | 12 GB |
| **06** | 评估 | avg@k、**pass@k 无偏估计**、maj@k、n-gram 污染检查 | 15 min | 6 GB |
| **07** | Agentic RL | 多轮工具调用、**工具输出的 loss mask**、安全沙箱 | 1.5 h / 100 步 | 14 GB |

### 框架对照（可选，需额外安装依赖）

同一份奖励函数、同一份 GSM8K 数据，换成主流 RL 框架复现 lab04，方便对比「手写 vs 框架」：

| Lab | 主题 | 框架 | 定位 |
|:---|:---|:---|:---|
| **08** | GRPO（TRL 版） | [TRL](https://github.com/huggingface/trl) `GRPOTrainer` | 单卡/小规模生产，成熟轻量 |
| **09** | GRPO（veRL 版） | [veRL](https://github.com/volcengine/verl) `main_ppo` | 大规模分布式生产（FSDP/Megatron + 千卡），单卡上只是走通流程 |

每个 lab 目录下都有自己的 `README.md`，说明原理、参数含义、预期输出和「该看什么指标」。

---

## 推荐路线

### 主线（一天跑完，能看到真实提升）

```bash
make lab00                                        # 自检
make lab01                                        # SFT：教会模型输出 <think>…</think> + \boxed{}
make lab04 ARGS="--model outputs/lab01_sft"       # GRPO：从 SFT 起步做强化学习
make lab06 ARGS="--n 200 --k 8 --compare \
    Qwen/Qwen2.5-0.5B-Instruct outputs/lab01_sft outputs/lab04_grpo/final"
```

或者一条命令：`make pipeline`

典型结果（0.5B 模型，GSM8K 测试集，avg@8）：

```
                    base    →   SFT    →   GRPO
准确率              ~0.23      ~0.34      ~0.55
格式合规率          ~0.05      ~0.98      ~1.00
```

> 具体数字会随随机种子、题目子集和步数波动。**报告分数时一定要带上 `avg@k_stderr`** —— lab06 会自动算给你。

### 支线（理解某个具体问题）

| 你想搞懂 | 跑这个 |
|:---|:---|
| 梯度累积的归一化 bug 长什么样 | `make lab01 ARGS="--naive-accum"` 对比默认跑法的 loss 曲线 |
| DPO 为什么训久了会变傻 | `make lab03` 然后 `make lab03 ARGS="--rpo-alpha 1.0"`，对比 `logps/chosen` |
| GRPO 的长度偏差 | `make lab04 ARGS="--loss-type grpo --no-dynamic-sampling"`，看 `len/mean` 一路涨 |
| 奖励怎么被 hack | `make lab05`（秒级），再 `make lab05 ARGS="--run-rl --reward naive"` |
| 不 mask 工具输出会怎样 | `make lab07 ARGS="--steps 30 --no-mask"`，看 `ratio_max` 爆掉 |

---

## 代码结构

```
llm-post-training-labs/
├── setup.sh                    一键装环境
├── Makefile                    所有快捷命令（make help 看列表）
├── requirements.txt
│
├── common/                     所有 lab 共用的核心模块
│   ├── logprobs.py             ★ 全仓库最核心：per-token log-prob、KL 估计器 k1/k2/k3
│   ├── rewards.py              RLVR 奖励函数（含防 hack 的门控设计）
│   ├── data.py                 chat template、loss mask、GSM8K / 偏好数据
│   ├── gpu.py                  设备/精度/attention 选择，显存检查
│   └── train_utils.py          优化器、调度器、日志、checkpoint
│
├── labs/
│   ├── lab00_env_check/        自检
│   ├── lab01_sft/              SFT
│   ├── lab02_reward_model/     奖励模型
│   ├── lab03_dpo/              DPO
│   ├── lab04_grpo/             ★ GRPO（核心）
│   ├── lab05_reward_hacking/   奖励工程
│   ├── lab06_eval_passk/       评估
│   ├── lab07_agentic_rl/       多轮工具调用
│   ├── lab08_trl_grpo/         GRPO（TRL 版，对照 lab04）
│   └── lab09_verl_grpo/        GRPO（veRL 版，对照 lab04）
│
├── scripts/
│   ├── test_core.py            ★ 算法单元测试（纯 CPU，几秒，改代码后必跑）
│   ├── generate.py             采样看输出（最被低估的调试手段）
│   ├── download_assets.py      预下载模型与数据
│   └── smoke_test.sh           全 lab 冒烟测试
│
└── docs/L4_NOTES.md            L4 专属：显存表、常见报错、调优建议
```

### 从哪读起

想看懂代码的话，建议这个顺序：

1. `common/logprobs.py` —— 30 行，但 SFT/DPO/GRPO 全都建立在它之上
2. `labs/lab01_sft/train_sft.py` —— 最简单的完整训练循环
3. `labs/lab04_grpo/train_grpo.py` 里的 `grpo_loss()` —— RL 的全部核心就那 10 行
4. `scripts/test_core.py` —— 看每个算法性质是怎么被验证的

---

## 改代码之后

```bash
make test          # 算法单元测试，纯 CPU 几秒钟，33 项检查
make smoke         # 全 lab 各跑 2~3 步，验证端到端没断
```

`make test` 会验证这些容易静默出错的地方：

- per-token log-prob 与手工计算逐位一致（差一位对齐 bug）
- 优势为正时 log-prob **确实上升**，为负时**确实下降**
- PPO 裁剪在该切断梯度的时候**真的切断了**
- 五种 loss_type 都能产生非零梯度
- k3 估计器恒非负、方差小于 k1
- 奖励函数能挡住穷举答案、空标签、复读等 hack
- 计算器沙箱能挡住代码注入和 `9**9**9` 这类指数炸弹

---

## L4 上的资源规划

| 项目 | 数值 |
|:---|:---|
| 显存 | 22.3 GB 可用（24 GB 标称） |
| 算力 | bf16 稠密约 121 TFLOPS 标称，实测矩阵乘 60–100 |
| 显存带宽 | ~300 GB/s（**比 A100 低 6 倍** —— 生成是带宽瓶颈，所以 rollout 明显偏慢） |
| 磁盘 | 模型 + 数据缓存约 3 GB；每个 checkpoint 约 1 GB |

**OOM 时按这个顺序调**：

1. `--micro-bs` 减半（对显存影响最大）
2. `--max-new-tokens` 减小（lab04/07）
3. `--group-size` 减到 4（会让 baseline 噪声变大，是最后手段）
4. `--max-len` / `--max-prompt-len` 减小

更多见 [`docs/L4_NOTES.md`](docs/L4_NOTES.md)。

---

## 长时间训练怎么挂后台

lab04 要跑几小时，SSH 断了就没了。用 tmux：

```bash
tmux new -s grpo
source .venv/bin/activate
make lab04 ARGS="--model outputs/lab01_sft --total-steps 500" 2>&1 | tee outputs/lab04.log
# Ctrl+B 然后按 D 断开；重新连上用 tmux attach -t grpo
```

想在网页上看曲线：`pip install wandb && wandb login`，然后给任意 lab 加 `--wandb`。
不用 wandb 的话，指标也都写在 `outputs/<lab>/metrics.jsonl`，可以自己画：

```bash
python -c "
import json, sys
rows=[json.loads(l) for l in open('outputs/lab04_grpo/metrics.jsonl')]
for r in rows[::10]: print(r['step'], round(r.get('acc',0),3), round(r.get('reward/mean',0),3))
"
```

---

## 常见问题

<details>
<summary><b>训练一直不动，reward 不涨</b></summary>

按顺序查：

1. `adv/absmean` 是不是 ≈ 0 —— 组内所有回答得分相同，没有信号。看 `acc`：接近 0 说明题太难（先做 lab01 冷启动），接近 1 说明题太简单。
2. `fmt` 是不是很低 —— 模型还不会输出合规格式，格式门控让所有样本都拿 0 分。**先跑 lab01**。
3. `ratio_mean` 是不是 ≈ 1.0 —— 不是的话说明采样分布与训练分布不一致，查温度是否两处一致。
4. 学习率是不是太小 —— RL 用 `1e-6` 量级，但比这更小就学不动了。

lab04 会在连续 3 步优势全 0 时自动打印警告和排查建议。
</details>

<details>
<summary><b>显存不够 / CUDA out of memory</b></summary>

先跑 `make lab00` 看看基线占用。然后按上面「OOM 时按这个顺序调」处理。

另外确认没有别的进程占着卡：`nvidia-smi`。
</details>

<details>
<summary><b>模型下载很慢或失败</b></summary>

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 国内镜像
python scripts/download_assets.py            # 先把东西下全
```

磁盘小的话可以换缓存盘：`export HF_HOME=/data/hf`
</details>

<details>
<summary><b>想换更大的模型</b></summary>

所有 lab 都支持 `--model`。L4 24GB 上的建议：

- `Qwen2.5-0.5B-Instruct`（默认）—— 全部 lab 全参训练都够
- `Qwen2.5-1.5B-Instruct` —— lab01/03 全参可以；lab04 建议配 `--micro-bs 2 --max-new-tokens 256`
- `Qwen2.5-7B-Instruct` —— 只能配 LoRA，且 lab04 需要 `--micro-bs 1`，会很慢

换模型后记得重跑 `make lab00` 确认显存够。
</details>

<details>
<summary><b>transformers 版本报错</b></summary>

本仓库在 transformers 4.45 ~ 5.x 上都测过。`common/logprobs.py` 会在运行时探测
`logits_to_keep` / `num_logits_to_keep` 参数改名的问题，不需要你手动处理。

如果遇到别的版本问题，`make test` 能在几秒内告诉你是不是核心逻辑坏了。
</details>

---

## 说明

- 所有实验用 **GSM8K**（小学数学应用题）和 **UltraFeedback**（偏好数据），都是公开数据集，首次运行自动下载。
- 主线（lab00~07）代码为教学目的编写，优先考虑**可读性**而非极致性能。生产环境请用
  [TRL](https://github.com/huggingface/trl)（lab08 已给出示例）或 [veRL](https://github.com/volcengine/verl)（lab09 已给出示例）。
- lab04/lab07 的 `grpo_loss()` 与 veRL 的 `core_algos.py`、TRL 的 `grpo_trainer.py` 结构一致，读懂这里就能读懂它们。

## License

MIT
