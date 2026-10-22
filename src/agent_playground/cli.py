"""交互式命令行对话：多轮历史、上下文截断、长期记忆与会话统计。"""

from agent_playground import memory
from agent_playground.config import Settings
from agent_playground.cost import estimate_cost_cny
from agent_playground.llm_client import stream_chat
from agent_playground.prompts import BASE_PROMPT

MAX_TURNS = 10
MEMORY_PREFIXES = ("记住：", "记住:", "记住 ", "请记住：", "请记住")


def extract_memory(text: str) -> str | None:
    """从「记住：xxx」这类显式表达里抽出要记住的内容。"""
    for prefix in MEMORY_PREFIXES:
        if text.startswith(prefix):
            body = text[len(prefix) :].strip()
            return body or None
    return None


def build_messages(
    history: list[dict[str, str]],
    memories: list[dict] | None = None,
) -> list[dict[str, str]]:
    """拼出发给接口的 messages：system（含记忆）+ 最近 MAX_TURNS 轮对话。"""
    prompt = BASE_PROMPT
    if memories:
        lines = "\n".join(f"- {item['text']}" for item in memories)
        prompt += f"\n\n已知关于用户的信息（供参考，不要复述）：\n{lines}"

    recent = history[-(MAX_TURNS * 2) :]
    return [{"role": "system", "content": prompt}, *recent]


def main() -> int:
    settings = Settings()
    history: list[dict[str, str]] = []
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    show_reasoning = False

    print(
        "输入问题开始对话。命令：/exit 退出  /clear 清空  /stats 统计  "
        "/think 思考显示  /mem 记忆  /forget <id> 删除记忆\n"
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
        if raw == "/mem":
            items = memory.all_memories()
            if not items:
                print("（还没有记忆）\n")
                continue
            print(f"共 {len(items)} 条记忆：")
            for item in items:
                print(f"  [{item['id']}] 用过 {item['use_count']} 次 · {item['text']}")
            print()
            continue
        if raw.startswith("/forget "):
            arg = raw[len("/forget ") :].strip()
            if not arg.isdigit():
                print("用法：/forget <id>\n")
                continue
            if memory.delete(int(arg)):
                print(f"（已删除记忆 {arg}）\n")
            else:
                print(f"（没有 id={arg} 的记忆）\n")
            continue

        remembered = extract_memory(raw)
        if remembered is not None:
            memory_id = memory.add(remembered)
            print(f"（已记住 [{memory_id}]：{remembered}）\n")
            continue

        history.append({"role": "user", "content": raw})
        print("AI > ", end="")
        memories = memory.search(raw)
        result = stream_chat(
            build_messages(history, memories),
            settings,
            show_reasoning=show_reasoning,
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