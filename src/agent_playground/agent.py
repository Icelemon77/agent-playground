"""Function Calling 最小闭环。

链路：模型请求调用工具 → 我们执行 → 回传结果 → 模型生成最终答案。
"""

import asyncio
import json
import logging
import sys
import time

import httpx

from agent_playground.config import Settings
from agent_playground.context import compact_observation
from agent_playground.llm_client import ChatClient, achat_with_retry, stream_chat
from agent_playground.logging_config import setup_logging
from agent_playground.prompts import TOOL_PROMPT
from agent_playground.tools import adispatch, dispatch, get_tool_schemas

logger = logging.getLogger("agent_playground.agent")


MAX_STEPS = 6


def run_agent(user_input: str, settings: Settings) -> str:
    """回答一个问题，过程中可以多次调用工具，回答实时流式输出。"""
    messages: list[dict] = [
        {"role": "system", "content": TOOL_PROMPT},
        {"role": "user", "content": user_input},
    ]

    total_prompt = 0
    total_completion = 0
    started_all = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        result = stream_chat(messages, settings, tools=get_tool_schemas())
        usage = result["usage"]
        total_prompt += usage.get("prompt_tokens", 0)
        total_completion += usage.get("completion_tokens", 0)
        calls = result["tool_calls"]

        # 没有工具调用 → 这就是最终答案
        if not calls:
            elapsed_all = time.perf_counter() - started_all
            print(
                f"\n[汇总] {step} 步 · 输入 {total_prompt} tokens · "
                f"输出 {total_completion} tokens · {elapsed_all:.2f}s"
            )
            logger.info(
                "agent_final step=%d prompt_tokens=%d completion_tokens=%d",
                step,
                total_prompt,
                total_completion,
            )
            return result["answer"] or "（模型返回了空内容）"

        # ★ 重构 assistant 消息：流式下没有现成的 message 对象，必须自己拼
        messages.append(
            {
                "role": "assistant",
                "content": result["answer"] or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            }
        )

        print()  # 工具调用前换行，避免和流式文本黏在一起

        for c in calls:
            name = c["name"]
            raw_args = c["arguments"]
            started = time.perf_counter()
            args = json.loads(raw_args)  # ← 现在才是解析的时机
            result_text = dispatch(name, raw_args)
            elapsed = time.perf_counter() - started

            logger.info(
                "tool_call name=%s duration=%.3fs args=%s result=%s",
                name,
                elapsed,
                raw_args,
                result_text[:200],
            )
            print(f"[tool] {name}  {args}  {elapsed:.3f}s  → {result_text[:100]}")

            compacted = compact_observation(result_text)
            messages.append(
                {"role": "tool", "tool_call_id": c["id"], "content": compacted}
            )

    return f"达到最大步数 {MAX_STEPS}，任务未完成"
    # ---- 执行模型要求的每个工具 ----


def main() -> int:
    settings = Settings()
    setup_logging(settings.log_level)
    question = sys.argv[1] if len(sys.argv) > 1 else "3 的 15 次方是多少？"
    print(f"问题：{question}")
    run_agent(question, settings)  # 内容已经在流式过程中打印过了
    print()  # 只补一个结尾换行
    return 0


async def arun_agent(
    user_input: str,
    settings: Settings,
    *,
    client: ChatClient | None = None,
) -> dict:
    """异步版 Agent 主循环：并发执行同一轮的多个工具。

    返回 {answer, steps, prompt_tokens, completion_tokens, elapsed_s}。
    """
    messages: list[dict] = [
        {"role": "system", "content": TOOL_PROMPT},
        {"role": "user", "content": user_input},
    ]
    total_prompt = 0
    total_completion = 0
    started = time.perf_counter()
    try:
        for step in range(1, MAX_STEPS + 1):
            result = await achat_with_retry(
                messages, settings, tools=get_tool_schemas(), client=client
            )

            usage = result["usage"]
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)

            calls = result["tool_calls"]

            if not calls:
                return {
                    "answer": result["answer"],
                    "error": None,
                    "steps": step,
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "elapsed_s": time.perf_counter() - started,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"] or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": c["arguments"],
                            },
                        }
                        for c in calls
                    ],
                }
            )

            # 并发执行本轮的所有工具
            observations = await asyncio.gather(
                *(adispatch(c["name"], c["arguments"]) for c in calls)
            )

            for c, observation in zip(calls, observations, strict=True):
                logger.info(
                    "tool_call name=%s args=%s result=%s",
                    c["name"],
                    c["arguments"],
                    observation[:200],
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "content": compact_observation(observation),
                    }
                )

        return {
            "answer": f"达到最大步数 {MAX_STEPS}，任务未完成",
            "error": "MaxStepsReached",
            "steps": MAX_STEPS,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "elapsed_s": time.perf_counter() - started,
        }
    except httpx.HTTPError as exc:
        logger.warning(
            "agent_http_error input=%s error=%s: %s",
            user_input[:40],
            type(exc).__name__,
            str(exc)[:120],
        )
        return {
            "answer": f"（网络错误，任务未能完成：{type(exc).__name__}）",
            "error": type(exc).__name__,
            "steps": 0,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "elapsed_s": time.perf_counter() - started,
        }


if __name__ == "__main__":
    raise SystemExit(main())
