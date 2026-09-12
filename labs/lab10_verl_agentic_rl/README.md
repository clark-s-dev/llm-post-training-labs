# Lab 10 · 用 veRL 复现 Agentic RL（多轮工具调用）

把 [lab07](../lab07_agentic_rl/README.md) 换成 veRL 内置的多轮 Agent Loop
（`ToolAgentLoop` + `@function_tool`）。奖励函数依旧是 `common/rewards.py`
那一套，只判最终答案；工具本身是同一段 AST 白名单计算器沙箱。

```bash
pip install -r labs/lab10_verl_agentic_rl/requirements-verl.txt
python labs/lab10_verl_agentic_rl/prepare_data.py          # 产出 data/{train,test}.parquet

bash labs/lab10_verl_agentic_rl/run_lab10_l4.sh --trainer.total_training_steps=3   # 冒烟
bash labs/lab10_verl_agentic_rl/run_lab10_l4.sh                                    # 完整跑
```

## 和 lab07 最大的不同：工具调用协议

lab07 是手写的，工具调用协议也是手写的——prompt 里教模型输出
`<tool>表达式</tool>`，代码用正则 `TOOL_RE` 去匹配。这是「教学版协议」，
简单直观，但不是模型的原生能力。

veRL 用的是**模型自带的 function-calling 能力**：

- 工具 schema（名字、参数、描述）通过 `tools.py` 里 `calculator()` 的类型标注
  和 docstring 自动推导成 OpenAI 格式，喂进 chat template
- 模型按训练时学到的格式输出工具调用（Qwen2.5 系列是类似 Hermes 的
  `<tool_call>{...}</tool_call>` JSON），veRL 用 `tool_parser`（`format: hermes`）解析
- 不需要你在 prompt 里手写协议，也不需要自己写正则去抓取调用

代价是**可控性变低**——协议由模型的预训练/对齐方式决定，你不能像 lab07 那样
随意改成任意自定义标签。

## 和 lab07 的对应关系

| lab07（手写） | lab10（veRL） | 说明 |
|:---|:---|:---|
| `<tool>...</tool>` 正则解析 | 原生 function-calling + `tool_parser(format="hermes")` | 协议来源不同，见上 |
| `safe_calc()` | `tools.py::calculator()`（`@function_tool`） | **同一份 AST 白名单沙箱逻辑**，包括 `9**9**9` 指数炸弹防护 |
| `loss_mask`（手动标 0/1，工具返回置 0） | `ToolAgentLoop` 自动维护 `response_mask` | ★ 本 lab 最重要的等价点：veRL 内部同样只对模型真正生成的 token 计梯度，工具返回自动被排除，不需要你手写 |
| `max_turns` / `max_tokens_per_turn` | `multi_turn.max_assistant_turns` / `max_tool_response_length` | 轮数与工具返回长度上限 |
| `agent_reward()`（只判最终答案，不奖励"调用工具"本身） | `reward_fn.py::compute_score()` | 同一条设计原则：调不调工具是 RL 要自己学的，不能直接奖励 |
| `--no-mask` 错误示范 | （无对应开关） | veRL 的 mask 是框架强制正确的，没有暴露"故意错误"的接口——这也是用框架的代价：少了一个亲手验证理解的机会，建议先做 lab07 再来对比 |

## 该看什么

除了 lab09 提到的 `critic/score/mean`、`val-core/.../acc`，多轮场景下额外关注：

- 工具调用是否发生（veRL 的 rollout trace / metrics 里能看到 assistant turn 数）
- 如果开了 `trainer.logger=[console,wandb]`，可以对比 rollout 里带工具调用的样本
  和不带的样本谁的奖励更高——和 lab07"该看什么"里的 `用工具` 指标是同一个问题

## 版本注意（比 lab08/09 更明显）

多轮 Agentic RL 是 veRL 里迭代最快的部分，几乎每个小版本都在动：

- `actor_rollout_ref.rollout.multi_turn.function_tool_path`（无状态函数工具，本 lab 用的）
  和 `tool_config_path`（有状态的 `BaseTool` 子类，需要 yaml 配置）是两条并存的代码路径，
  二者可以同时用，但工具名不能重复。如果你的 veRL 版本较旧，可能只支持后者——
  参考同版本的 `verl/trainer/config/rollout/rollout.yaml` 里 `multi_turn:` 那一段自己的字段列表。
- `actor_rollout_ref.rollout.name=sglang` 是多轮工具调用目前官方推荐、测试最充分的引擎；
  部分较新版本的 vLLM async 模式（`rollout.name=vllm rollout.mode=async`）也开始支持，
  但成熟度不如 sglang，报错时优先切回 sglang 排查。
- `agent.default_agent_loop=tool_agent` 依赖 veRL 内置注册的 `ToolAgentLoop`
  （`@register("tool_agent")`）。如果这个名字在你的版本里不存在，
  说明该版本的多轮 Agent Loop 还是旧的 `SGLangRollout` 直接实现，
  需要参考对应版本的 `docs/sglang_multiturn/multiturn.rst`。

报错时先跑：

```bash
python3 -c "import verl; print(verl.__version__ if hasattr(verl, '__version__') else 'unknown')"
python3 -c "from verl.trainer.config.rollout import rollout"   # 存在即说明是较新的分层配置
```

## 什么时候真的该用 veRL 做 agentic RL

单卡场景下和 lab09 的结论一样：意义在"看懂接口"，不在性能。真正的价值在于：

- 工具环境需要**有状态**（比如带浏览器 session 的 web agent）——这时候要用
  `tool_config_path` 的 `BaseTool` 子类（有 `create`/`release` 生命周期钩子），
  比 lab07 手写的无状态 rollout 循环更适合
- 需要多个工具、多个 agent loop 混合调度
- 需要在生产规模（多卡、长时间）上稳定跑多轮 RL，而不是验证一个算法想法
