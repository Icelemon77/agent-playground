"""LLM 客户端：流式调用大模型接口，返回回答、思考、用量与耗时。"""

import asyncio
import json
import logging
import time
from typing import Any, Protocol

import httpx
from rich.console import Console

from agent_playground.config import Settings
from agent_playground.errors import LLMCallError

console = Console()


class ResponseLike(Protocol):
    """HTTP 响应需要满足的接口（让测试能注入假响应）。"""

    @property
    def status_code(self) -> int: ...

    def json(self) -> dict[str, Any]: ...

    # 返回类型只能是 Any：httpx.Response.raise_for_status() 返回 Response 自身，
    # 而假响应返回 None。声明成 None 会让真实的 httpx.Response 不满足这个协议。
    def raise_for_status(self) -> Any: ...


class ChatClient(Protocol):
    """聊天客户端需要满足的接口（让测试能注入假 client）。

    这里用 Protocol 而不是 httpx.AsyncClient，是为了表达「鸭子类型」：
    只要长得像就行，不需要真的继承——这正是假对象能在测试里工作的原因。
    """

    async def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> ResponseLike: ...

    # achat 在自己创建 client 时会调用它来释放连接，
    # 所以协议里必须声明——哪怕测试用的假对象永远不会被调用到。
    async def aclose(self) -> None: ...


logger = logging.getLogger("agent_playground.llm_client")


def stream_chat(
    messages: list[dict[str, str]],
    settings: Settings,
    *,
    tools: list[dict] | None = None,
    show_reasoning: bool = False,
) -> dict:
    """流式调用 chat/completions，边收边打印，返回本次调用的统计信息。"""
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "none",
    }
    if tools:
        payload["tools"] = tools
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    usage: dict = {}
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    thinking_first_at: float | None = None
    answer_first_at: float | None = None
    finish_reason: str | None = None
    started = time.perf_counter()
    tool_calls_acc: dict[int, dict[str, str]] = {}

    with (
        httpx.Client(timeout=settings.request_timeout) as client,
        client.stream("POST", url, json=payload, headers=headers) as resp,
    ):
        if resp.status_code != 200:
            raise LLMCallError(
                f"大模型接口返回 HTTP {resp.status_code}",
                status_code=resp.status_code,
            )

        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage = chunk["usage"]

            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if reason := choice.get("finish_reason"):
                    finish_reason = reason
                if piece := delta.get("reasoning_content"):
                    if thinking_first_at is None:
                        thinking_first_at = time.perf_counter()
                    reasoning_parts.append(piece)
                    if show_reasoning:
                        console.print(piece, end="", style="grey50")
                if piece := delta.get("content"):
                    if answer_first_at is None:
                        answer_first_at = time.perf_counter()
                    answer_parts.append(piece)
                    console.print(piece, end="")
                # 累积流式工具调用碎片：按 index 分组，arguments 是字符串拼接
                for tc in delta.get("tool_calls") or []:
                    idx = tc["index"]
                    slot = tool_calls_acc.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

    elapsed = time.perf_counter() - started
    return {
        "answer": "".join(answer_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": [tool_calls_acc[i] for i in sorted(tool_calls_acc)],
        "usage": usage,
        "finish_reason": finish_reason,
        "think_ttft": (thinking_first_at - started) if thinking_first_at else 0.0,
        "answer_ttft": (answer_first_at - started) if answer_first_at else 0.0,
        "elapsed": elapsed,
    }


async def achat(
    messages: list[dict],
    settings: Settings,
    *,
    tools: list[dict] | None = None,
    client: ChatClient | None = None,
) -> dict:
    """异步调用（非流式）。

    返回 {answer, tool_calls, usage}。
    tool_calls 已归一化成 [{id, name, arguments}]，可直接喂给 adispatch。
    """
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "reasoning_effort": "none",
    }
    if tools:
        payload["tools"] = tools
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=settings.request_timeout)

    try:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        return {
            "answer": message.get("content") or "",
            "tool_calls": [
                {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
                for call in (message.get("tool_calls") or [])
            ],
            "usage": data.get("usage", {}),
        }
    finally:
        if owns_client:
            await client.aclose()


# 这些 HTTP 状态码代表「服务端临时问题」，值得重试
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


async def achat_with_retry(
    messages: list[dict],
    settings: Settings,
    *,
    tools: list[dict] | None = None,
    client: ChatClient | None = None,
    max_attempts: int = 3,
    base_backoff: float = 0.5,
) -> dict:
    """带重试的 achat。

    只重试「可能恢复」的错误：
      - 网络层异常（超时、断连、DNS）—— 通常抖动
      - HTTP 429 / 5xx —— 限流或服务端故障
    4xx（除了 429）立刻放弃——参数错了重试一万次也一样。

    base_backoff 是首次重试的等待秒数，后续按指数增长（0.5 → 1 → 2）。
    测试时传 0 可以让重试瞬间完成。
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.wait_for(
                achat(messages, settings, tools=tools, client=client),
                timeout=settings.request_timeout,
            )
        except TimeoutError as exc:
            last_error = exc
        except httpx.TransportError as exc:
            last_error = exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRYABLE_STATUS:
                raise  # 4xx：不重试，直接上抛
            last_error = exc

        if attempt < max_attempts:
            wait = base_backoff * 2 ** (attempt - 1)  # 指数退避 0.5s → 1s → 2s
            logger.warning(
                "llm_retry attempt=%d/%d wait=%.1fs error=%s",
                attempt,
                max_attempts,
                wait,
                type(last_error).__name__,
            )
            await asyncio.sleep(wait)

    logger.error("llm_retry_exhausted attempts=%d error=%s", max_attempts, last_error)
    raise last_error if last_error else RuntimeError("重试全部失败")
