"""工具定义、注册与分发。

设计要点：用 @tool 装饰器把「函数实现」和「给模型看的 schema」绑定在一起，
两者不会再不同步；JSON Schema 由 Pydantic 模型自动生成，无需手写。
"""

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

_REGISTRY: dict[str, tuple[Callable[..., str], type[BaseModel]]] = {}
_SCHEMAS: list[dict[str, Any]] = []


def tool(*, description: str, params: type[BaseModel]) -> Callable:
    """把一个函数注册成模型可调用的工具。"""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        _REGISTRY[fn.__name__] = (fn, params)
        _SCHEMAS.append(
            {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": params.model_json_schema(),
                },
            }
        )
        return fn

    return decorator


def get_tool_schemas() -> list[dict[str, Any]]:
    """返回给模型的工具清单。"""
    return _SCHEMAS


def dispatch(name: str, raw_args: str) -> str:
    """执行工具。

    所有失败都转换成字符串返回给模型，不向上抛异常——
    让模型有机会理解错误并补救，而不是让程序崩溃。
    """
    if name not in _REGISTRY:
        return f"错误：不存在名为 {name} 的工具。可用工具：{list(_REGISTRY)}"

    fn, params_model = _REGISTRY[name]

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return f"错误：参数不是合法的 JSON（{exc}）。请检查格式后重试。"

    try:
        validated = params_model(**args)
    except ValidationError as exc:
        return f"错误：参数校验未通过。{exc.errors()}"

    try:
        return fn(**validated.model_dump())
    except Exception as exc:  # noqa: BLE001
        return f"错误：工具执行失败（{type(exc).__name__}: {exc}）"


# ---------------- 工具实现 ----------------


class CalculatorArgs(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="要计算的数学表达式，使用 Python 语法，例如 '3 ** 15'",
    )


@tool(
    description=(
        "计算一个数学表达式并返回结果。"
        "当用户询问算术、幂运算、百分比等需要精确数值计算的问题时使用。"
    ),
    params=CalculatorArgs,
)
def calculator(expression: str) -> str:
    """计算数学表达式。

    注意：eval 能执行任意代码，有安全风险，这里仅用于教学。
    Day 17 会换成安全的表达式求值。
    """

    return str(eval(expression, {"__builtins__": {}}, {}))
