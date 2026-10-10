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
    omitted = len(text) - max_chars
    return f"{text[:half]}\n\n...（中间省略 {omitted} 字符）...\n\n{text[-half:]}"


def compact_log(text: str, max_lines: int = MAX_LOG_LINES) -> str:
    """日志类内容：优先保留有问题的行。"""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    important = [ln for ln in lines if any(k in ln for k in LOG_LEVEL_KEYWORDS)]
    normal = [ln for ln in lines if ln not in important]

    keep = important[: max_lines - 3]
    keep += normal[: max_lines - len(keep)]
    kept = [ln for ln in lines if ln in keep]
    return "\n".join(kept) + f"\n\n...（省略 {len(lines) - len(kept)} 行正常日志）..."


def compact_observation(text: str) -> str:
    """按内容类型选择压缩策略。"""
    looks_like_log = text.count("\n") > 20 and ("ERROR" in text or "WARN" in text)
    if looks_like_log:
        return compact_log(text)
    return compact_head_tail(text)
