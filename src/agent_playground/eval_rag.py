"""RAG 评测：一条命令跑出检索与生成的量化报告。

三层指标，按成本从低到高，方便反复跑便宜的那层：

1. **检索召回 recall@k**：期望答案有没有出现在前 k 个块里。纯计算，零 API 成本。
2. **无答案拒答率**：文档里没有答案时，会不会硬凑一篇垃圾。纯计算。
3. **端到端准确率**：把检索结果交给模型，答案里有没有期望的内容。这一步花钱。

召回直接在检索层测 k=1/3/5，**不经过工具**——工具的输出有条数上限，
如果从工具那条路测，测到的是「投递限制」而不是「检索能力」，k=5 永远测不出来。

用法：
    uv run python -m agent_playground.eval_rag --tag baseline
    uv run python -m agent_playground.eval_rag --tag baseline --no-e2e
"""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_playground.config import Settings
from agent_playground.cost import estimate_cost_cny
from agent_playground.eval_set import (
    DEFAULT_PATH,
    EvalItem,
    fingerprint,
    load_eval_set,
    matches_any,
)
from agent_playground.rag import TfidfIndex, generate, get_index, retrieve

K_VALUES = (1, 3, 5)
TOP_K_FOR_GENERATION = 3
REPORT_DIR = Path("data/eval")

# 模型说「资料里没有」时常见的说法。命中任意一个就算正确拒答。
REFUSAL_MARKERS = (
    "资料中没有",
    "没有相关信息",
    "没有找到",
    "未找到",
    "无法回答",
    "知识库中没有",
)


def hit_text(hits: list[tuple[float, int, str]]) -> str:
    """把命中的块拼成一段文本，用于做字符串匹配。"""
    return "\n".join(chunk for _score, _index, chunk in hits)


def score_item(item: EvalItem, text: str) -> float:
    """单条问题的得分。

    fact 类是全中制（要的是那个具体的数字，差一点都不算），
    synth 类按关键词命中比例给部分分（需要凑齐多个值，全中要求过严）。

    expect 的每一项支持「可选写法」（用 | 分隔），避免同一个事实的
    不同表述被字符串匹配误判成答错。
    """
    if not item.expect:
        return 0.0
    matched = sum(1 for token in item.expect if matches_any(token, text))
    if item.category == "fact":
        return 1.0 if matched == len(item.expect) else 0.0
    return matched / len(item.expect)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    items: list[EvalItem],
    index: TfidfIndex,
    k_values: tuple[int, ...] = K_VALUES,
) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    """测检索召回。返回 (汇总, {问题 id: {recall@k: 分数}})。"""
    answerable = [item for item in items if item.category != "none"]
    per_question: dict[int, dict[str, float]] = {}

    by_k: dict[int, list[float]] = defaultdict(list)
    by_k_category: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for item in answerable:
        per_question[item.id] = {}
        for k in k_values:
            hits = retrieve(item.question, index, top_k=k)
            score = score_item(item, hit_text(hits))
            per_question[item.id][f"recall@{k}"] = score
            by_k[k].append(score)
            by_k_category[k][item.category].append(score)

    summary: dict[str, Any] = {
        "questions": len(answerable),
        "overall": {f"recall@{k}": mean(by_k[k]) for k in k_values},
        "by_category": {
            category: {
                f"recall@{k}": mean(by_k_category[k][category]) for k in k_values
            }
            for category in ("fact", "synth")
        },
    }
    return summary, per_question


def evaluate_no_answer(
    items: list[EvalItem], index: TfidfIndex, top_k: int = 3
) -> dict:
    """测无答案类问题：检索会不会硬凑，以及正负样本的分数能不能分开。"""
    negatives = [item for item in items if item.category == "none"]
    max_scores: list[float] = []
    for item in negatives:
        hits = retrieve(item.question, index, top_k=top_k)
        max_scores.append(hits[0][0] if hits else 0.0)

    return {
        "questions": len(negatives),
        "returned_hits": sum(1 for score in max_scores if score > 0),
        "max_scores": [round(score, 4) for score in max_scores],
    }


def evaluate_end_to_end(
    items: list[EvalItem],
    index: TfidfIndex,
    settings: Settings,
    sources: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict]]:
    """把检索结果交给模型，检查答案本身。这一步花钱。"""
    results: list[dict] = []
    prompt_tokens = 0
    completion_tokens = 0

    for item in items:
        hits = retrieve(item.question, index, top_k=TOP_K_FOR_GENERATION)
        output = generate(item.question, hits, settings)
        answer = str(output["answer"])
        usage = output.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", 0))

        if item.category == "none":
            passed = any(marker in answer for marker in REFUSAL_MARKERS)
        else:
            passed = score_item(item, answer) == 1.0

        results.append(
            {
                "id": item.id,
                "category": item.category,
                "passed": passed,
                "answer": answer[:120],
                "sources": [sources[i] for _s, i, _c in hits],
            }
        )

    by_category: dict[str, list[float]] = defaultdict(list)
    for record in results:
        by_category[record["category"]].append(1.0 if record["passed"] else 0.0)

    cost = estimate_cost_cny(
        settings.llm_model,
        {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        is_peak=False,
    )
    summary: dict[str, Any] = {
        "questions": len(items),
        "accuracy": mean([1.0 if r["passed"] else 0.0 for r in results]),
        "by_category": {
            category: mean(scores) for category, scores in by_category.items()
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_cny": round(cost, 5),
    }
    return summary, results


def separation(
    no_answer: dict[str, Any], retrieval_scores: list[float]
) -> dict[str, Any]:
    """判断「用相似度阈值过滤无关问题」可不可行。

    如果负样本的最高分比正样本的最低分还高，两个分布重叠，
    任何阈值都会同时误伤一边——这条路走不通，该换特征而不是调阈值。
    """
    negatives = no_answer.get("max_scores") or []
    if not negatives or not retrieval_scores:
        return {"decidable": False, "reason": "数据不足"}

    positive_min = min(retrieval_scores)
    negative_max = max(negatives)
    return {
        "decidable": positive_min > negative_max,
        "positive_min": round(positive_min, 4),
        "negative_max": round(negative_max, 4),
    }


def load_existing_reports(except_tag: str) -> dict[str, dict[str, Any]]:
    """读回之前跑过的报告，用于并排对比。"""
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(REPORT_DIR.glob("report_*.json")):
        tag = path.stem.removeprefix("report_")
        if tag == except_tag:
            continue
        try:
            reports[tag] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return reports


def stale_warning(
    current: dict[str, Any], others: dict[str, dict[str, Any]]
) -> str | None:
    """语料或评测集变了就提示基线过期。

    报告的可比性依赖两个会变的东西：**检索语料**（改一份文档，连 README 也算，
    切块和排名就会变）和**评测集本身**（改个措辞、更新一个过期的 expect，
    召回和准确率也会变）。两个都记指纹，每次运行比一次。
    """
    stale = [
        tag
        for tag, report in others.items()
        if report.get("corpus", {}).get("fingerprint")
        != current["corpus"]["fingerprint"]
        or report.get("eval_set", {}).get("fingerprint")
        != current["eval_set"]["fingerprint"]
    ]
    if not stale:
        return None
    return (
        f"⚠ 语料或评测集与 {len(stale)} 份历史报告不一致"
        f"（{', '.join(sorted(stale))}）——"
        "那些数字已不代表当前状态，对比仅供参考，基线需要重跑。"
    )


def print_comparison(
    current_tag: str, current: dict[str, Any], others: dict[str, dict[str, Any]]
) -> None:
    """把当前结果和历史报告并排打印——这就是「优化前后对比表」。"""
    if not others:
        return

    def cell(report: dict[str, Any], key: str) -> str:
        retrieval = report.get("retrieval", {}).get("overall", {})
        if key in retrieval:
            return f"{retrieval[key]:.2f}"
        if key == "e2e":
            value = report.get("end_to_end", {}).get("accuracy")
            return f"{value:.2f}" if isinstance(value, float) else "—"
        return "—"

    columns = [*others.keys(), current_tag]
    rows = ["recall@1", "recall@3", "recall@5", "e2e"]
    width = max(len(row) for row in rows) + 2
    print("\n=== 与历史报告对比 ===")
    print("".ljust(width) + "".join(tag.rjust(12) for tag in columns))
    for row in rows:
        line = row.ljust(width)
        for tag in columns:
            report = current if tag == current_tag else others[tag]
            line += cell(report, row).rjust(12)
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--tag", default="baseline", help="本次报告的名字")
    parser.add_argument("--no-e2e", action="store_true", help="跳过端到端，只测检索")
    parser.add_argument("--questions", default=str(DEFAULT_PATH), help="评测集路径")
    args = parser.parse_args()

    settings = Settings()
    items = load_eval_set(args.questions)
    index, sources = get_index()

    print(
        f"评测集 {len(items)} 条 | 知识库 {index.size} 个块 / {len(set(sources))} 个文件"
    )
    print(f"知识库指纹：{index.fingerprint()}")
    print(f"评测集指纹：{fingerprint(items)}")

    retrieval, per_question = evaluate_retrieval(items, index)
    no_answer = evaluate_no_answer(items, index)

    # 正样本的分数：每条问题在 top-3 里的最高分
    positive_scores = [
        retrieve(item.question, index, top_k=1)[0][0]
        for item in items
        if item.category != "none"
    ]
    report: dict[str, Any] = {
        "tag": args.tag,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": {
            "chunks": index.size,
            "files": len(set(sources)),
            "fingerprint": index.fingerprint(),
        },
        "eval_set": {
            "questions": len(items),
            "fingerprint": fingerprint(items),
        },
        "retrieval": retrieval,
        "no_answer": no_answer,
        "separation": separation(no_answer, positive_scores),
        "per_question": per_question,
    }

    print("\n=== 检索召回 ===")
    overall = retrieval["overall"]
    for k in K_VALUES:
        print(f"  recall@{k}: {overall[f'recall@{k}']:.2f}")
    print("\n=== 按类别 ===")
    by_category = retrieval["by_category"]
    for category, scores in by_category.items():
        line = "  " + category.ljust(6)
        line += "".join(f"recall@{k}={scores[f'recall@{k}']:.2f}  " for k in K_VALUES)
        print(line)

    print("\n=== 无答案类 ===")
    print(
        f"  {no_answer['questions']} 条，返回了结果的有 {no_answer['returned_hits']} 条"
    )
    print(f"  打分: {no_answer['max_scores']}")
    print(f"  分离度: {report['separation']}")

    if args.no_e2e:
        print("\n（跳过了端到端，只测检索）")
    else:
        print("\n=== 端到端（会花 API 费用）===")
        e2e, results = evaluate_end_to_end(items, index, settings, sources)
        report["end_to_end"] = e2e
        report["e2e_results"] = results
        print(f"  总体准确率: {e2e['accuracy']:.2f}")
        for category, value in e2e["by_category"].items():
            print(f"  {category:<6}{value:.2f}")
        print(
            f"  tokens {e2e['prompt_tokens']}+{e2e['completion_tokens']}"
            f" | 花费 ¥{e2e['cost_cny']}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"report_{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {out}")

    others = load_existing_reports(args.tag)
    warning = stale_warning(report, others)
    if warning:
        print(f"\n{warning}")
    print_comparison(args.tag, report, others)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
