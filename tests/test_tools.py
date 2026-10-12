"""工具的单元测试。

为什么工具最值得测：Day 3 的实验证明——把 calculator 改成恒定返回 "9999" 后，
模型会一本正经地告诉你「3 的 15 次方等于 9999」。

**模型对工具返回值是 100% 信任的，所以工具的可靠性就是 Agent 的上限。**
"""

import pytest

from agent_playground.tools import (
    calculator,
    dispatch,
    get_current_time,
    list_files,
    read_file,
)

# ---------------- calculator：工具层 ----------------


def test_calculator_basic() -> None:
    """基本运算。"""
    assert calculator("3 ** 15") == "14348907"


def test_calculator_returns_string() -> None:
    """★ 返回值必须是字符串——这是工具的契约。

    如果哪天有人改成返回 int，调用方的字符串拼接会立刻崩。
    """
    assert isinstance(calculator("1 + 1"), str)


def test_calculator_invalid_expression_raises() -> None:
    """★ 非法表达式由工具层抛异常，转成错误字符串是 dispatch 的职责。

    这个分层很重要：工具函数只管干活，错误处理集中在分发层。
    """
    with pytest.raises(SyntaxError):
        calculator("3 +")


# ---------------- get_current_time ----------------


def test_get_current_time_default_timezone() -> None:
    """默认时区返回可读的时间字符串。"""
    result = get_current_time()
    assert "星期" in result
    assert "Asia/Shanghai" in result


def test_get_current_time_invalid_timezone() -> None:
    """★ 边界：未知时区返回错误字符串，而不是抛异常。"""
    result = get_current_time("Mars/Olympus")
    assert result.startswith("错误")


# ---------------- read_file（用 tmp_path 造数据） ----------------


def test_read_file_returns_content(tmp_path, monkeypatch) -> None:
    """正常读取。

    用 tmp_path + chdir 造临时数据——测试不该依赖仓库里的 data/ 目录。
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample.txt").write_text("hello", encoding="utf-8")

    assert read_file("sample.txt") == "hello"


def test_read_file_missing_lists_available(tmp_path, monkeypatch) -> None:
    """★ 文件不存在时，错误信息里要列出可用文件——让模型能自己纠正。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "exists.txt").write_text("x", encoding="utf-8")

    result = read_file("missing.txt")

    assert "不存在" in result
    assert "exists.txt" in result


def test_read_file_path_traversal_blocked(tmp_path, monkeypatch) -> None:
    """★★ 安全边界：路径穿越必须被拦住，且不能泄露内容。

    这是 Day 3 写的防护，现在用测试把它锁住——
    以后谁改坏了这行，这条测试会立刻失败。
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "secret.txt").write_text("绝密内容", encoding="utf-8")

    result = read_file("../secret.txt")

    assert "超出允许范围" in result
    assert "绝密内容" not in result  # ← 关键：内容绝对不能泄露


# ---------------- list_files ----------------


def test_list_files_returns_sorted_names(tmp_path, monkeypatch) -> None:
    """列出文件名，且是排序的。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "b.txt").write_text("", encoding="utf-8")
    (tmp_path / "data" / "a.txt").write_text("", encoding="utf-8")

    result = list_files()

    assert "a.txt" in result
    assert "b.txt" in result
    assert result.index("a.txt") < result.index("b.txt")


def test_list_files_empty_dir(tmp_path, monkeypatch) -> None:
    """★ 边界：空目录给出明确提示，而不是空字符串。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    assert "为空" in list_files()


def test_list_files_missing_dir(tmp_path, monkeypatch) -> None:
    """★ 边界：目录不存在时返回错误字符串。"""
    monkeypatch.chdir(tmp_path)
    assert "不存在" in list_files()


# ---------------- dispatch：分发层 ----------------


def test_dispatch_executes_tool() -> None:
    """正常分发。"""
    assert dispatch("calculator", '{"expression": "1 + 1"}') == "2"


def test_dispatch_unknown_tool_returns_error() -> None:
    """★ 模型可能编出不存在的工具名——必须返回错误而不是抛异常。"""
    result = dispatch("no_such_tool", "{}")

    assert "不存在" in result
    assert "calculator" in result  # 应该告诉模型有哪些可用工具


def test_dispatch_invalid_json_returns_error() -> None:
    """★ 模型可能给出畸形 JSON。"""
    result = dispatch("calculator", "{not valid json}")

    assert "JSON" in result


def test_dispatch_missing_required_param() -> None:
    """★ 参数缺失时由 Pydantic 拦截。"""
    result = dispatch("calculator", "{}")

    assert result.startswith("错误")


def test_dispatch_tool_exception_becomes_error_string() -> None:
    """★★ 工具内部抛异常时，dispatch 必须捕获并转成错误字符串。

    这是 Agent 不崩溃的关键：错误变成给模型的信息，而不是让程序挂掉。
    """
    result = dispatch("calculator", '{"expression": "3 +"}')

    assert result.startswith("错误")
    assert "SyntaxError" in result
