"""
Lab 10 —— 计算器工具（veRL `@function_tool` 版）

和 lab07 的 safe_calc() 是**同一段沙箱逻辑**，独立复制一份而不是 import lab07 的模块，
是因为这个文件会被 veRL 的 rollout worker 进程按路径动态 exec（`function_tool_path`），
不应该连带拉入 lab07 训练脚本里的 torch/transformers 顶层导入。

veRL 用 `transformers.utils.get_json_schema` 从函数签名 + docstring 自动推导
OpenAI function-calling schema，所以这里的类型标注和 Google 风格 docstring
不是写着好看的 —— 少一个都会在加载时报错。
"""

from __future__ import annotations

import ast
import operator

from verl.tools.function_tool import function_tool

# 只允许这几种运算 —— 绝不能用 eval()！道理和 lab07 一模一样：
# RL 训练会跑几百万次模型生成的表达式，模型不需要有恶意，
# 只要碰巧生成 __import__('os').system(...) 就能让宿主机完蛋。
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

_MAX_ABS = 1e15
_MAX_EXP = 32


def _safe_calc(expr: str) -> str:
    expr = expr.strip()[:200]
    try:
        node = ast.parse(expr, mode="eval").body

        def guard(v):
            if isinstance(v, complex) or v != v or abs(v) > _MAX_ABS:
                raise ValueError("数值过大")
            return v

        def ev(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return guard(n.value)
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                a, b = ev(n.left), ev(n.right)
                # ★ 幂运算必须在计算之前限制指数，否则 9**9**9 这种大整数
                #   会算到进程卡死（Python 大整数是任意精度的）。
                if isinstance(n.op, ast.Pow) and (abs(b) > _MAX_EXP or abs(a) > 1e6):
                    raise ValueError("指数过大")
                return guard(_OPS[type(n.op)](a, b))
            if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
                return guard(_OPS[type(n.op)](ev(n.operand)))
            raise ValueError("不支持的表达式")

        val = ev(node)
        return str(int(val)) if float(val).is_integer() else f"{val:.6g}"
    except ZeroDivisionError:
        return "ERROR: 除零"
    except RecursionError:
        return "ERROR: 表达式嵌套过深"
    except Exception as e:
        # ★ 报错也如实喂回给模型，训练它学会从错误里恢复，而不是假装没发生。
        return f"ERROR: {type(e).__name__}"


@function_tool
def calculator(expression: str) -> str:
    """对一个四则运算表达式求值。

    只支持数字和 + - * / ( ) ** 这几种符号，不支持变量、函数调用或其他 Python 语法。

    Args:
        expression: 要计算的算术表达式，例如 "15 * 4" 或 "(3 + 5) / 2"。

    Returns:
        计算结果的字符串，或者以 "ERROR: " 开头的错误信息（例如除零、表达式非法）。
    """
    return _safe_calc(expression)
