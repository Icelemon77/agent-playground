"""交互式命令行对话：多轮历史、上下文截断与会话统计。"""

from agent_playground.config import Settings
from agent_playground.cost import estimate_cost_cny
from agent_playground.llm_client import stream_chat

SYSTEM_PROMPT = "你是 agent-playground 项目的助手，回答简洁准确，不确定时直说不确定。"
MAX_TURNS = 10


def build_messages(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """拼出发给接口的 messages：system + 最近 MAX_TURNS 轮对话。"""
    recent = history[-(MAX_TURNS * 2) :]
    return [{"role": "system", "content": SYSTEM_PROMPT}, *recent]


def main() -> int:
    settings = Settings()
    history: list[dict[str, str]] = []
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    show_reasoning = False

    print(
        "输入问题开始对话。命令：/exit 退出  /clear 清空  /stats 统计  /think 思考显示\n"
    )

    while True:
        try:
            raw = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw == "/exit":
            break
        if raw == "/clear":
            history.clear()
            print("（历史已清空）\n")
            continue
        if raw == "/stats":
            print(f"累计：输入 {total_prompt} tokens，输出 {total_completion} tokens")
            print(f"预估花费：¥{total_cost:.6f}\n")
            continue
        if raw == "/think":
            show_reasoning = not show_reasoning
            print(f"（思考显示：{'开' if show_reasoning else '关'}）\n")
            continue

        history.append({"role": "user", "content": raw})
        print("AI > ", end="")
        result = stream_chat(
            build_messages(history), settings, show_reasoning=show_reasoning
        )
        history.append({"role": "assistant", "content": result["answer"]})

        usage = result["usage"]
        total_prompt += usage.get("prompt_tokens", 0)
        total_completion += usage.get("completion_tokens", 0)
        total_cost += estimate_cost_cny(settings.llm_model, usage)
        print(
            f"\n[本轮 {usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)} tokens"
            f" / {result['elapsed']:.2f}s / finish={result['finish_reason']}]\n"
        )

    print(f"\n本次会话：输入 {total_prompt} tokens，输出 {total_completion} tokens")
    print(f"预估花费：¥{total_cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
