"""文本式 ReAct Agent。

靠 prompt 约定格式，自己解析 Thought / Action / Observation。
与 agent.py 的区别：不走 tool_calls 协议，工具调用完全靠自己解析文本。
"""

import logging
import re
import sys
import time

import httpx
import json
from agent_playground.trace import Tracer
from agent_playground.config import Settings
from agent_playground.logging_config import setup_logging
from agent_playground.prompts import REACT_PROMPT
from agent_playground.tools import dispatch, get_tool_schemas
from agent_playground.context import compact_observation

logger = logging.getLogger("agent_playground.react")

MAX_STEPS = 6


def build_tool_descriptions() -> str:
    """把工具 schema 渲染成给模型看的自然语言列表。"""
    lines: list[str] = []
    for schema in get_tool_schemas():
        fn = schema["function"]
        params = ", ".join(fn["parameters"].get("properties", {}))
        lines.append(f"- {fn['name']}({params}): {fn['description']}")
    return "\n".join(lines)


def parse_response(text: str) -> tuple[str, str, str] | None:
    """解析模型输出。

    返回 ("final", 答案, "") 或 ("action", 工具名, 参数JSON)；解析失败返回 None。

    注意顺序：必须先判断 Final Answer，否则会把「直接回答」误判成格式错误。
    """
    if match := re.search(r"Final Answer:\s*(.+)", text, re.DOTALL):
        return ("final", match.group(1).strip(), "")

    action = re.search(r"Action:\s*([A-Za-z_]\w*)", text)
    action_input = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if action and action_input:
        return ("action", action.group(1), action_input.group(1))

    return None


def chat(messages: list[dict], settings: Settings) -> tuple[str, dict]:
    """发一次非流式请求。注意：payload 里没有 tools 字段。"""
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "reasoning_effort": "none",
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}

    with httpx.Client(timeout=settings.request_timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], data.get("usage", {})


def run_react(user_input: str, settings: Settings) -> str:
    """ReAct 主循环：Thought → Action → Observation → ... → Final Answer。"""
    messages: list[dict] = [
        {
            "role": "system",
            "content": REACT_PROMPT.format(tool_descriptions=build_tool_descriptions()),
        },
        {"role": "user", "content": user_input},
    ]

    tracer = Tracer("logs", user_input, settings.llm_model, MAX_STEPS)
    total_prompt = 0
    total_completion = 0
    started_all = time.perf_counter()

    try:
        for step in range(1, MAX_STEPS + 1):
            text, usage = chat(messages, settings)
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)

            print(f"\n--- step {step} ---")
            print(text.strip())

            # 防御：模型可能自己编造 Observation，截断后再入历史
            if "Observation:" in text:
                logger.warning("hallucinated_observation step=%d", step)
                print("[warn] 模型自己编造了 Observation，已截断")
                text = text.split("Observation:", 1)[0].rstrip()

            parsed = parse_response(text)

            if parsed is None:
                print("[warn] 格式解析失败，要求模型重试")
                logger.warning("parse_failed step=%d", step)
                tracer.write({"type": "parse_error", "step": step, "raw": text})
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "格式错误：请只输出 Thought/Action/Action Input，"
                            "或 Thought/Final Answer，不要输出其他内容。"
                        ),
                    }
                )
                continue

            kind, value, args = parsed

            if kind == "final":
                elapsed_all = time.perf_counter() - started_all
                tracer.write(
                    {
                        "type": "final",
                        "step": step,
                        "answer": value,
                        "total_prompt_tokens": total_prompt,
                        "total_completion_tokens": total_completion,
                        "elapsed_s": elapsed_all,
                    }
                )
                logger.info(
                    "react_final step=%d prompt_tokens=%d completion_tokens=%d answer=%s",
                    step,
                    total_prompt,
                    total_completion,
                    value[:200],
                )
                print(
                    f"\n[汇总] {step} 步 · 输入 {total_prompt} tokens · "
                    f"输出 {total_completion} tokens · {elapsed_all:.2f}s"
                )
                return value

            started = time.perf_counter()
            observation = dispatch(value, args)
            elapsed = time.perf_counter() - started

            print(f"[observation] {observation}")
            tracer.write(
                {
                    "type": "step",
                    "step": step,
                    "thought": extract_thought(text),
                    "action": value,
                    "action_input": safe_json(args),
                    "observation": observation,
                    "duration_s": elapsed,
                    "usage": usage,
                }
            )
            logger.info(
                "react_action step=%d tool=%s args=%s duration=%.3fs",
                step,
                value,
                args,
                elapsed,
            )

            compacted = compact_observation(observation)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {compacted}"})

        tracer.write({"type": "max_steps", "steps": MAX_STEPS})
        logger.warning("react_max_steps steps=%d", MAX_STEPS)
        return f"达到最大步数 {MAX_STEPS}，任务未完成"

    finally:
        tracer.close()


def extract_thought(text: str) -> str:
    """从模型输出里取出 Thought 的内容。"""
    match = re.search(
        r"Thought:\s*(.+?)(?=\n\s*(?:Action|Final Answer):|$)", text, re.DOTALL
    )
    return match.group(1).strip() if match else ""


def safe_json(raw: str) -> object:
    """能解析就返回对象，否则原样返回——保证 trace 不会因为脏数据写不进去。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


MAX_OBSERVATION_CHARS = 800


LOG_LEVEL_KEYWORDS = ("ERROR", "WARN", "Exception", "失败", "超时")


def main() -> int:
    settings = Settings()
    setup_logging(settings.log_level)

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "读取 data/numbers.txt，然后把里面所有数字相加"
    )
    print(f"问题：{question}")

    answer = run_react(question, settings)
    print(f"\n{'=' * 20} 最终答案 {'=' * 20}")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
