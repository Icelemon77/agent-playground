"""Function Calling 最小闭环。

链路：模型请求调用工具 → 我们执行 → 回传结果 → 模型生成最终答案。
"""

import json
import sys
import logging
import time
import httpx

from agent_playground.config import Settings
from agent_playground.tools import dispatch, get_tool_schemas
from agent_playground.prompts import TOOL_PROMPT
from agent_playground.logging_config import setup_logging

logger = logging.getLogger("agent_playground.agent")


def post_chat(messages: list[dict], settings: Settings) -> dict:
    """发一次非流式请求，返回完整响应。"""
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "tools": get_tool_schemas(),
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    with httpx.Client(timeout=settings.request_timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


MAX_STEPS = 5


def run_agent(user_input: str, settings: Settings) -> str:
    """回答一个问题，过程中可以多次调用工具。"""
    messages: list[dict] = [
        {"role": "system", "content": TOOL_PROMPT},  # ← 加这一条，必须在第一位
        {"role": "user", "content": user_input},
    ]

    for step in range(MAX_STEPS):
        data = post_chat(messages, settings)
        choice = data["choices"][0]
        msg = choice["message"]
        messages.append(msg)

        # 终止条件：模型不再请求工具，说明它要给出答案了
        if not msg.get("tool_calls"):
            return msg.get("content") or "（模型返回了空内容）"

        print(f"[step {step + 1}] 模型请求调用 {len(msg['tool_calls'])} 个工具")
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            raw_args = call["function"]["arguments"]

            started = time.perf_counter()
            result = dispatch(name, raw_args)
            elapsed = time.perf_counter() - started

            logger.info(
                "tool_call name=%s duration=%.3fs args=%s result=%s",
                name,
                elapsed,
                raw_args,
                result[:200],
            )
            # 屏幕上仍然打印，方便交互时观察
            print(f"[tool] {name}  {raw_args}  {elapsed:.3f}s  → {result[:100]}")

            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    return f"达到最大步数 {MAX_STEPS}，任务未完成"
    # ---- 执行模型要求的每个工具 ----


def main() -> int:
    settings = Settings()
    setup_logging(settings.log_level)
    question = sys.argv[1] if len(sys.argv) > 1 else "3 的 15 次方是多少？"
    print(f"问题：{question}")
    answer = run_agent(question, settings)
    print(f"\n回答：{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
