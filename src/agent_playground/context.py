"""上下文管理：控制发给模型的 Observation 长度。

集中在这里，agent.py 和 react_agent.py 共用——
保证两种范式在同样的上下文条件下对比。
"""

MAX_OBSERVATION_CHARS = 800
MAX_LOG_LINES = 20
LOG_LEVEL_KEYWORDS = ("ERROR", "WARN", "Exception", "失败", "超时")


def compact_head_tail(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    """通用策略：截断中间，保留头尾。"""
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    if half == 0:
        # max_chars < 2 时无法保留头尾，退化成只截头部。
        # 不加这个分支的话 text[-0:] 等于 text[0:]——
        # 会把整个字符串又拼回来，结果比原文还长。
        return text[:max_chars] + "..."

    omitted = len(text) - max_chars
    return f"{text[:half]}\n\n...（中间省略 {omitted} 字符）...\n\n{text[-half:]}"


def compact_log(text: str, max_lines: int = MAX_LOG_LINES) -> str:
    """日志类内容：优先保留有问题的行。"""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    # 用「行号」而不是「行内容」来记录保留哪些行。
    # 按内容匹配会有个陷阱：内容相同的行只要 keep 里有 1 个就全部被保留，
    # 大量重复行时压缩会完全失效。
    important_idx = [
        i for i, ln in enumerate(lines) if any(k in ln for k in LOG_LEVEL_KEYWORDS)
    ]
    important_set = set(important_idx)
    normal_idx = [i for i in range(len(lines)) if i not in important_set]

    # 问题行优先，但至少留 3 个位置给正常行做上下文。
    # 用 max(1, ...) 是因为 max_lines 很小时 max_lines - 3 会变成负数，
    # 而负数切片 important_idx[:-1] 会静默丢掉最后一个元素。
    budget_for_important = max(1, max_lines - 3)
    keep_idx = important_idx[:budget_for_important]

    remaining = max_lines - len(keep_idx)
    if remaining > 0:
        keep_idx += normal_idx[:remaining]

    kept = [lines[i] for i in sorted(keep_idx)]
    return "\n".join(kept) + f"\n\n...（省略 {len(lines) - len(kept)} 行）..."


def compact_observation(text: str) -> str:
    """按内容类型选择压缩策略。"""
    looks_like_log = text.count("\n") > 20 and ("ERROR" in text or "WARN" in text)
    if looks_like_log:
        return compact_log(text)
    return compact_head_tail(text)
