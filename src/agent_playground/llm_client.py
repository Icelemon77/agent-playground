"""LLM 客户端：流式调用大模型接口，返回回答、思考、用量与耗时。"""

import json
import time

import httpx
from rich.console import Console

from agent_playground.config import Settings
from agent_playground.errors import LLMCallError

console = Console()


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
    client: httpx.AsyncClient | None = None,
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
