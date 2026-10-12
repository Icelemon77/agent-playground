"""上下文压缩的测试。

这个模块最容易出问题的地方是**边界**——参数很小、内容重复、
行数少但每行超长等等。所以这里的测试重点在边界。
"""

from agent_playground.context import (
    compact_head_tail,
    compact_log,
    compact_observation,
)

# ---------------- compact_head_tail ----------------


def test_head_tail_short_text_unchanged() -> None:
    """没超过上限时原样返回。"""
    assert compact_head_tail("短文本", max_chars=100) == "短文本"


def test_head_tail_marks_omitted_count() -> None:
    """截断时标明省略了多少字符——让模型知道「中间有东西被删了」。"""
    result = compact_head_tail("A" * 5000, max_chars=100)

    assert "省略" in result
    assert "4900" in result  # 5000 - 100


def test_head_tail_keeps_both_ends() -> None:
    """头尾都要保留——长文本的关键信息通常在两端。"""
    text = "HEAD" + "x" * 1000 + "TAIL"
    result = compact_head_tail(text, max_chars=100)

    assert result.startswith("HEAD")
    assert result.endswith("TAIL")


def test_head_tail_never_grows_text() -> None:
    """★ 边界：压缩后的长度永远不该超过原文。

    发现过程（写测试时）：max_chars=1 时 half = 1 // 2 = 0，
    而 text[-0:] 等于 text[0:]——等于把整个字符串又拼了回来，
    结果比原文还长。
    """
    original = "A" * 100
    result = compact_head_tail(original, max_chars=1)

    assert len(result) <= len(original)


# ---------------- compact_log ----------------


def test_log_short_text_unchanged() -> None:
    """行数没超过上限时原样返回。"""
    text = "\n".join(["INFO a"] * 5)
    assert compact_log(text, max_lines=20) == text


def test_log_keeps_error_lines_in_middle() -> None:
    """★★ 回归用例：必须保留分散在中间的 ERROR/WARN 行。

    背景：Day 4 实测发现「头尾截断」会丢掉中间的慢查询日志，
    导致模型只能发现 3/5 类问题。这个测试防止它再次发生。
    """
    lines = (
        ["INFO 正常"] * 5
        + ["WARN 中间的重要警告"]
        + ["INFO 正常"] * 30
        + ["ERROR 结尾的错误"]
    )
    result = compact_log("\n".join(lines), max_lines=20)

    assert "中间的重要警告" in result
    assert "ERROR 结尾的错误" in result


def test_log_respects_max_lines_with_duplicate_lines() -> None:
    """★★ 即使有大量重复行，保留的行数也不能超过 max_lines。

    发现过程（写测试时）：kept 用 `ln in keep` **按值**匹配——
    30 行完全相同的 "INFO 重复行" 只要 keep 里有 1 个，就被全部保留了。
    """
    lines = ["INFO 重复行"] * 30 + ["ERROR 唯一"] * 5
    result = compact_log("\n".join(lines), max_lines=20)

    kept = [ln for ln in result.splitlines() if ln and not ln.startswith("...")]
    assert len(kept) <= 20


def test_log_works_with_tiny_max_lines() -> None:
    """★ 边界：max_lines 很小时不该因为负数切片而行为错乱。"""
    lines = ["INFO a"] * 30 + ["ERROR b"] * 3
    result = compact_log("\n".join(lines), max_lines=2)

    kept = [ln for ln in result.splitlines() if ln and not ln.startswith("...")]
    assert len(kept) <= 2


# ---------------- compact_observation：策略选择 ----------------


def test_observation_uses_log_strategy_for_logs() -> None:
    """★ 长日志应该走「按级别筛选」的策略。"""
    lines = ["INFO 正常"] * 25 + ["ERROR 重要错误"]
    result = compact_observation("\n".join(lines))

    assert "重要错误" in result


def test_observation_uses_head_tail_for_generic_text() -> None:
    """★ 没有日志特征时走「保留头尾」的策略。"""
    text = "HEAD" + "x" * 2000 + "TAIL"
    result = compact_observation(text)

    assert result.startswith("HEAD")
    assert result.endswith("TAIL")


def test_observation_short_but_long_text_is_compacted() -> None:
    """★★ 只有几行、但总长度很大的文本，仍然必须被压缩。

    这一条守着 Day 4 修过的**运算符优先级 bug**：
    原来的写法 `a and b or c` 等价于 `(a and b) or c`，
    导致「只要含 WARN 就判定成日志」——
    而日志策略按**行数**判断，面对「行数少但每行超长」的内容会原样返回，
    等于**完全没压缩**。
    """
    long_line = "WARN " + "x" * 1000
    text = "\n".join([long_line] * 3)  # 只有 3 行，但 3000+ 字符

    result = compact_observation(text)

    assert len(result) < len(text)  # 必须真的压缩了


def test_observation_empty_input() -> None:
    """★ 边界：空输入返回空串，不该崩。"""
    assert compact_observation("") == ""
