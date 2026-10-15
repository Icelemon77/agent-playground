"""工具定义、注册与分发。

设计要点：用 @tool 装饰器把「函数实现」和「给模型看的 schema」绑定在一起，
两者不会再不同步；JSON Schema 由 Pydantic 模型自动生成，无需手写。
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, ValidationError

from agent_playground.rag import DEFAULT_TOP_K, get_index, retrieve

logger = logging.getLogger("agent_playground.tools")
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
    except Exception as exc:
        # 关键：非预期异常必须留下痕迹，
        # 否则你自己的 bug 会被伪装成「工具坏了」
        logger.exception("tool_internal_error name=%s args=%s", name, raw_args)
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


# ---------------- 工具 2：当前时间 ----------------


class TimeArgs(BaseModel):
    timezone: str = Field(
        default="Asia/Shanghai",
        description="IANA 时区名，例如 'Asia/Shanghai'、'UTC'、'America/New_York'",
    )


@tool(
    description=(
        "获取当前日期和时间。"
        "当用户询问现在几点、今天几号、星期几时必须使用此工具——"
        "你没有时间概念，不知道现在是哪一天。"
    ),
    params=TimeArgs,
)
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """返回指定时区的当前时间。"""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"错误：未知时区 {timezone}，请使用 IANA 时区名，例如 'Asia/Shanghai'"

    now = datetime.now(tz)
    weekday_cn = "一二三四五六日"[now.weekday()]
    return f"{now:%Y-%m-%d %H:%M:%S} 星期{weekday_cn}（{timezone}）"


# ---------------- 工具 3：读取文件 ----------------

DATA_DIR = Path("data")


class ReadFileArgs(BaseModel):
    filename: str = Field(description="data/ 目录下的文件名，例如 'numbers.txt'")


@tool(
    description="读取 data/ 目录下的文本文件内容。当用户要求查看某个文件时使用。",
    params=ReadFileArgs,
)
def read_file(filename: str) -> str:
    """读取 data/ 下的文本文件，限制在该目录内。"""
    base = DATA_DIR.resolve()
    target = (base / filename).resolve()

    # 安全：禁止路径穿越（../ 逃逸出 data/）
    if not target.is_relative_to(base):
        return f"错误：路径 {filename} 超出允许范围，只能读取 data/ 目录内的文件"

    if not target.exists():
        available = [p.name for p in base.glob("*")] if base.exists() else []
        return f"错误：文件 {filename} 不存在。可用文件：{available}"

    text = target.read_text(encoding="utf-8")
    if len(text) > 100_000:  # 只防资源耗尽，正常文件不会触发
        return text[:100_000] + f"\n...（文件过大已截断，原文共 {len(text)} 字符）"
    return text


# ---------------- 工具 4：列出文件 ----------------


class ListFilesArgs(BaseModel):
    """无需参数。"""


@tool(
    description=(
        "列出 data/ 目录下的所有文件名。"
        "当用户询问有哪些文件、想查看文件清单时使用此工具。"
    ),
    params=ListFilesArgs,
)
def list_files() -> str:
    """列出 data/ 目录下的文件名。"""
    base = DATA_DIR
    if not base.exists():
        return "错误：data/ 目录不存在"

    names = sorted(p.name for p in base.iterdir() if p.is_file())
    if not names:
        return "（data/ 目录为空）"
    return "、".join(names)


# ---------------- 工具 5：检索知识库（RAG）----------------


class SearchKnowledgeArgs(BaseModel):
    """检索知识库的参数。"""

    query: str = Field(
        min_length=1,
        max_length=200,
        description="检索用的关键词，用关键词比用整句提问更容易命中",
    )


TOOL_OUTPUT_LIMIT = 780
"""工具输出的字符上限。

比 context.MAX_OBSERVATION_CHARS（800）小一点，留出余量：
工具结果回到 agent 层还会过一次上下文压缩，超了就会被腰斩。
检索结果是**按相关性排序**的，信息集中在头部——被保留头尾的策略一裁，
受损的正好是次相关的那块。所以这里自己按优先级装箱，保证最相关的完整送达。
"""


@tool(
    description=(
        "在项目自己的知识库（docs/ 下的笔记与 README）里检索相关片段。"
        "当问题涉及本项目的实现细节、实测数据或踩坑记录时，先调用它查证再回答，不要凭记忆作答。"
    ),
    params=SearchKnowledgeArgs,
)
def search_knowledge(query: str) -> str:
    """检索知识库，返回最相关的几个片段（带来源文件与相似度）。"""
    index, sources = get_index()
    hits = retrieve(query, index, top_k=DEFAULT_TOP_K)
    if not hits:
        return (
            "知识库中没有检索到相关内容。换一组关键词再试，或说明这一点不在知识库中。"
        )

    blocks: list[str] = []
    used = 0
    for i, (score, index_no, chunk) in enumerate(hits, start=1):
        block = f"[{i}] 来源 {sources[index_no]}（相似度 {score:.3f}）\n{chunk}"
        if blocks and used + len(block) + 2 > TOOL_OUTPUT_LIMIT:
            break
        blocks.append(block)
        used += len(block) + 2

    text = "\n\n".join(blocks)
    if len(blocks) < len(hits):
        # 告诉模型「还有内容没给」——它才有机会换关键词再查一次，
        # 这就是 Agentic RAG 和「一次检索定生死」的区别。
        text += (
            f"\n\n（因长度限制只返回了 {len(blocks)}/{len(hits)} 条，可换关键词再检索）"
        )
    return text


async def adispatch(name: str, raw_args: str) -> str:
    """异步版的工具分发：支持同步工具和异步工具共存。"""
    if name not in _REGISTRY:
        return f"错误：不存在名为 {name} 的工具。可用工具：{list(_REGISTRY)}"

    fn, params_model = _REGISTRY[name]

    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return f"错误：参数不是合法的 JSON（{exc}）。请检查格式后重试。"

    try:
        validated = params_model(**args).model_dump()
    except ValidationError as exc:
        return f"错误：参数校验未通过。{exc.errors()}"

    try:
        if inspect.iscoroutinefunction(fn):
            return await fn(**validated)  # 异步工具直接 await
        return await asyncio.to_thread(fn, **validated)  # 同步工具丢到线程池
    except Exception as exc:
        logger.exception("tool_internal_error name=%s", name)
        return f"错误：工具执行失败（{type(exc).__name__}: {exc}）"


class WeatherArgs(BaseModel):
    city: str = Field(min_length=1, max_length=50, description="城市名，例如 '北京'")


@tool(
    description="查询指定城市的天气。当用户询问天气时使用。",
    params=WeatherArgs,
)
async def fetch_weather(city: str) -> str:
    """模拟一个慢速 API 调用。"""
    await asyncio.sleep(1.5)  # 模拟网络延迟
    return f"{city} 今天晴，22-28 度"
