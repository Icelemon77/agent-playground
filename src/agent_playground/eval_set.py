"""评测集的加载与校验。

评测集是要提交进仓库的项目资产，所以它自己也需要被检查：
JSON 格式错、expect 在知识库里不存在、分类配比失衡——
任何一项都会让后面的评测结果失去意义。

校验用的知识库必须和 rag.py 完全一致，否则会出现
「expect 在文档里存在，但不在检索范围内」这种自欺欺人的通过。
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from agent_playground.rag import load_documents

CATEGORIES = ("fact", "synth", "none")
REQUIRED_FIELDS = ("id", "category", "question", "expect")
DEFAULT_PATH = Path("data/eval/rag_questions.jsonl")
MIN_NONE_RATIO = 0.15


@dataclass(frozen=True)
class EvalItem:
    """一条评测问题。"""

    id: int
    category: str
    question: str
    expect: list[str]


def fingerprint(items: list[EvalItem]) -> str:
    """评测集指纹：题目与期望答案的内容哈希。

    改了评测集，历史报告同样不可比——加一条、改个措辞、更新一个过期的
    `expect`，召回和准确率都会变。报告里必须记下它对应哪一版评测集。
    实测过：把覆盖率那条的 expect 从过期的 40% 更正为当前值，端到端就差一道题。
    """
    payload = "\n".join(
        f"{item.id}|{item.category}|{item.question}|{'~'.join(item.expect)}"
        for item in sorted(items, key=lambda x: x.id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def knowledge_text() -> str:
    """把知识库拼成一个大字符串，用于检查 expect 是否真实存在。"""
    return "".join(text for _source, text in load_documents())


def alternatives(token: str) -> list[str]:
    """拆开 expect 里的「可选写法」。

    同一个事实在文档里常有多种表述（`0.6 秒` 与 `0.57 秒`、`约 52%` 与 `52%`），
    用字符串匹配判定时会误判成答错。所以 expect 的一项可以写成
    `"0.6 秒|0.57 秒"`，任一写法出现就算命中。
    """
    return [part.strip() for part in token.split("|") if part.strip()]


def matches_any(token: str, text: str) -> bool:
    """text 里是否出现了该 expect 项认可的任意一种写法。"""
    return any(alt in text for alt in alternatives(token))


def load_eval_set(path: Path | str = DEFAULT_PATH) -> list[EvalItem]:
    """读取并校验评测集的格式。

    把所有问题收集完再一次性抛出，而不是遇到第一个就停——
    这样改一轮就能看到全部问题，不用「改一处跑一次」。
    """
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"评测集不存在：{file}（当前工作目录是 {Path.cwd()}）")

    lines = [ln for ln in file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"评测集是空的：{file}")

    problems: list[str] = []
    items: list[EvalItem] = []
    for lineno, line in enumerate(lines, 1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"第 {lineno} 行: JSON 解析失败（{exc.msg}）")
            continue
        if not isinstance(raw, dict):
            problems.append(f"第 {lineno} 行: 不是一个 JSON 对象")
            continue

        missing = [name for name in REQUIRED_FIELDS if name not in raw]
        if missing:
            problems.append(f"第 {lineno} 行: 缺少字段 {missing}")
            continue

        category = raw["category"]
        expect = raw["expect"]
        if category not in CATEGORIES:
            problems.append(
                f"第 {lineno} 行: category 只能是 {CATEGORIES}，收到 {category!r}"
            )
            continue
        if not isinstance(expect, list):
            problems.append(f"第 {lineno} 行: expect 必须是数组")
            continue
        if category == "none" and expect:
            problems.append(f"第 {lineno} 行: none 类问题的 expect 必须为空")
            continue
        if category != "none" and not expect:
            problems.append(f"第 {lineno} 行: {category} 类问题的 expect 不能为空")
            continue

        items.append(
            EvalItem(
                id=int(raw["id"]),
                category=str(category),
                question=str(raw["question"]),
                expect=[str(token) for token in expect],
            )
        )

    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise ValueError(f"评测集有 {len(problems)} 个格式问题：\n{detail}")
    return items


def audit(items: list[EvalItem]) -> list[str]:
    """检查内容层面的问题，返回问题清单（空列表 = 全部通过）。"""
    problems: list[str] = []

    ids = [item.id for item in items]
    duplicated = [value for value, count in Counter(ids).items() if count > 1]
    if duplicated:
        problems.append(f"id 有重复：{sorted(duplicated)}")
    if sorted(ids) != list(range(1, len(ids) + 1)):
        problems.append(f"id 不是 1..{len(ids)} 的连续编号")

    text = knowledge_text()
    for item in items:
        for token in item.expect:
            if not matches_any(token, text):
                problems.append(
                    f"问题 {item.id}: expect {token!r} 的所有写法在知识库里都找不到"
                )

    counts = Counter(item.category for item in items)
    none_ratio = counts["none"] / len(items)
    if none_ratio < MIN_NONE_RATIO:
        problems.append(
            f"none 类只占 {none_ratio:.0%}，低于 {MIN_NONE_RATIO:.0%}——测不出拒答能力"
        )

    return problems


def main() -> int:
    path = DEFAULT_PATH
    try:
        items = load_eval_set(path)
    except ValueError as exc:
        print(exc)
        return 1

    counts = Counter(item.category for item in items)
    print(f"评测集：{path}（{len(items)} 条）")
    for category in CATEGORIES:
        share = counts[category] / len(items)
        print(f"  {category:<6}{counts[category]:>4} 条  {share:>5.0%}")

    problems = audit(items)
    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\n全部检查通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
