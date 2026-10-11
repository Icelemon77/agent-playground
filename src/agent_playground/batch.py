"""批量任务：串行 vs 并发跑一批问题，对比耗时与成本。

用法：
    uv run python -m agent_playground.batch
"""

import asyncio
import sys
import time

import httpx
from rich.console import Console

from agent_playground.agent import arun_agent
from agent_playground.config import Settings
from agent_playground.cost import estimate_cost_cny

console = Console()

QUESTIONS = [
    "3 的 15 次方是多少？",
    "现在几点了？",
    "data 目录下有哪些文件？",
    "读取 data/numbers.txt 并把里面的数字相加",
    "读取 data/app.log，用一句话总结主要问题",
]


async def run_serial(settings: Settings, client: httpx.AsyncClient) -> list[dict]:
    """串行：一个问题跑完再跑下一个。"""
    results: list[dict] = []
    for question in QUESTIONS:
        console.print(f"  [dim]→ {question}[/dim]")
        results.append(await arun_agent(question, settings, client=client))
    return results


async def run_concurrent(
    settings: Settings,
    client: httpx.AsyncClient,
    limit: int = 3,
) -> list[dict]:
    """并发：最多 limit 个任务同时跑。"""
    semaphore = asyncio.Semaphore(limit)

    async def one(question: str) -> dict:
        async with semaphore:
            return await arun_agent(question, settings, client=client)

    return list(await asyncio.gather(*(one(q) for q in QUESTIONS)))


def summarize(label: str, results: list[dict], elapsed: float, model: str) -> dict:
    total_in = sum(r["prompt_tokens"] for r in results)
    total_out = sum(r["completion_tokens"] for r in results)
    cost = estimate_cost_cny(
        model, {"prompt_tokens": total_in, "completion_tokens": total_out}
    )
    console.print(
        f"  [bold]{label}[/bold]  耗时 [cyan]{elapsed:6.2f}s[/cyan]  "
        f"输入 {total_in:>5}  输出 {total_out:>4}  成本 ¥{cost:.6f}"
    )
    return {"elapsed": elapsed, "in": total_in, "out": total_out, "cost": cost}


async def main() -> int:
    settings = Settings()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    console.print(
        f"模型：{settings.llm_model}    问题数：{len(QUESTIONS)}    并发上限：{limit}\n"
    )

    async with httpx.AsyncClient(timeout=settings.request_timeout) as shared:
        console.print("[bold]=== 串行 ===[/bold]")
        started = time.perf_counter()
        serial_results = await run_serial(settings, shared)
        serial = summarize(
            "串行", serial_results, time.perf_counter() - started, settings.llm_model
        )

        console.print(f"\n[bold]=== 并发（limit={limit}）===[/bold]")
        started = time.perf_counter()
        concurrent_results = await run_concurrent(settings, shared, limit=limit)
        concurrent = summarize(
            "并发",
            concurrent_results,
            time.perf_counter() - started,
            settings.llm_model,
        )

    console.print("\n" + "=" * 58)
    console.print(
        f"串行 {serial['elapsed']:.2f}s  →  并发 {concurrent['elapsed']:.2f}s"
    )
    console.print(f"加速 {serial['elapsed'] / concurrent['elapsed']:.1f}x")
    console.print(
        f"token 总数：串行 {serial['in'] + serial['out']}，"
        f"并发 {concurrent['in'] + concurrent['out']}"
    )
    console.print("\n[dim]注意：并发不减少 token 总量——它只压缩时间。[/dim]")

    for label, results in (("串行", serial_results), ("并发", concurrent_results)):
        failures = [
            r for r in results if r.get("answer", "").startswith("达到最大步数")
        ]
        if failures:
            console.print(f"[yellow]{label} 里有 {len(failures)} 个任务未完成[/yellow]")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
