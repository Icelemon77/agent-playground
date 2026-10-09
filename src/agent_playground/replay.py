"""回放一次 Agent 运行的 trace 文件。

用法：uv run python -m agent_playground.replay logs\trace_20261010_153000.jsonl
"""

import json
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def render(record: dict) -> None:
    kind = record.get("type")

    if kind == "start":
        console.rule(f"[bold]运行开始[/bold] {record.get('time', '')}")
        console.print(f"[bold]问题：[/bold]{record.get('question', '')}")
        console.print(
            f"[dim]模型 {record.get('model')} · 最大步数 {record.get('max_steps')}[/dim]"
        )

    elif kind == "step":
        console.rule(f"step {record.get('step')}")
        if thought := record.get("thought"):
            console.print(f"[dim]Thought:[/dim] {thought}")
        console.print(f"[cyan]Action:[/cyan] {record.get('action')}")
        console.print(
            f"[cyan]Action Input:[/cyan] "
            f"{json.dumps(record.get('action_input'), ensure_ascii=False)}"
        )
        console.print(f"[green]Observation:[/green] {record.get('observation')}")
        if usage := record.get("usage"):
            console.print(
                f"[dim]{record.get('duration_s', 0):.3f}s · "
                f"输入 {usage.get('prompt_tokens', 0)} / "
                f"输出 {usage.get('completion_tokens', 0)} tokens[/dim]"
            )

    elif kind == "parse_error":
        console.print(f"[yellow]⚠ step {record.get('step')} 解析失败[/yellow]")
        console.print(f"[dim]{record.get('raw', '')}[/dim]")

    elif kind == "final":
        console.rule("[bold]最终答案[/bold]")
        console.print(record.get("answer", ""))
        console.print(
            f"[dim]{record.get('step')} 步 · "
            f"输入 {record.get('total_prompt_tokens', 0)} tokens · "
            f"输出 {record.get('total_completion_tokens', 0)} tokens · "
            f"{record.get('elapsed_s', 0):.2f}s[/dim]"
        )

    elif kind == "max_steps":
        console.print("[red]达到最大步数，任务未完成[/red]")

    else:
        console.print(f"[dim]（未知记录类型：{kind}）[/dim]")


def load(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python -m agent_playground.replay <trace 文件路径>")
        return 1

    for record in load(sys.argv[1]):
        render(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
