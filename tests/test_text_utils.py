"""text_utils 的单元测试。

选它们做第一批测试的理由：
  1. 纯函数——输入确定、输出确定，没有网络也没有状态
  2. 跑得飞快——毫秒级，可以跑几百个
  3. 边界行为多——正好练「怎么测边界」
"""

from agent_playground.text_utils import (
    build_llm_headers,
    count_tokens_approx,
    merge_config,
    parse_env_pairs,
    truncate,
)

# ---------------- count_tokens_approx ----------------


def test_count_tokens_none_returns_zero() -> None:
    """None 输入返回 0，不该崩。"""
    assert count_tokens_approx(None) == 0


def test_count_tokens_empty_returns_zero() -> None:
    """空字符串返回 0。"""
    assert count_tokens_approx("") == 0


def test_count_tokens_estimates_by_length() -> None:
    """按 chars_per_token 估算：100 字符 ÷ 4 = 25。"""
    assert count_tokens_approx("a" * 100) == 25


def test_count_tokens_nonempty_returns_at_least_one() -> None:
    """★ 边界：非空文本至少返回 1。

    3 个字符除以 4 是 0，但函数保证最小为 1。
    为什么这个边界重要：返回 0 会让调用方误判成「没有内容」。
    """
    assert count_tokens_approx("abc") == 1


# ---------------- truncate ----------------


def test_truncate_short_text_unchanged() -> None:
    """没超过上限的文本原样返回。"""
    assert truncate("hello", max_chars=100) == "hello"


def test_truncate_exactly_at_limit_unchanged() -> None:
    """★ 边界：长度恰好等于上限时不该截断（用的是 <= 不是 <）。"""
    assert truncate("a" * 10, max_chars=10) == "a" * 10


def test_truncate_over_limit_appends_suffix() -> None:
    """超过上限时截断并加省略号。"""
    assert truncate("a" * 200, max_chars=10) == "a" * 10 + "..."


def test_truncate_custom_suffix() -> None:
    """后缀可以自定义。"""
    assert truncate("a" * 200, max_chars=5, suffix="…") == "a" * 5 + "…"


# ---------------- parse_env_pairs ----------------


def test_parse_env_pairs_basic() -> None:
    """最基础的 key=value。"""
    assert parse_env_pairs(["KEY=value"]) == {"KEY": "value"}


def test_parse_env_pairs_skips_comments_and_blank_lines() -> None:
    """注释行和空行被跳过。"""
    lines = ["# 这是注释", "", "KEY=value", "   "]
    assert parse_env_pairs(lines) == {"KEY": "value"}


def test_parse_env_pairs_strips_whitespace() -> None:
    """键和值两边的空格都被去掉。"""
    assert parse_env_pairs(["  KEY  =  value  "]) == {"KEY": "value"}


def test_parse_env_pairs_value_containing_equals() -> None:
    """★ 边界：值里含 = 时不该被切断（partition 只切第一个 =）。"""
    assert parse_env_pairs(["URL=a=b=c"]) == {"URL": "a=b=c"}


def test_parse_env_pairs_line_without_equals() -> None:
    """★ 边界：没有 = 的行，值应该是空串而不是报错。"""
    assert parse_env_pairs(["KEY"]) == {"KEY": ""}


# ---------------- build_llm_headers ----------------


def test_build_headers_without_key() -> None:
    """没传 key 时只有 Content-Type。"""
    assert build_llm_headers() == {"Content-Type": "application/json"}


def test_build_headers_with_key() -> None:
    """传了 key 时加上 Authorization。"""
    headers = build_llm_headers("sk-test")
    assert headers["Authorization"] == "Bearer sk-test"


def test_build_headers_empty_key_ignored() -> None:
    """★ 边界：空字符串 key 不该产生 Authorization（if api_key 是假值）。"""
    assert "Authorization" not in build_llm_headers("")


def test_build_headers_extra_merged() -> None:
    """额外的 header 会被合并进去。"""
    headers = build_llm_headers(extra={"X-Trace": "abc"})
    assert headers["X-Trace"] == "abc"
    assert headers["Content-Type"] == "application/json"


# ---------------- merge_config ----------------


def test_merge_config_without_override() -> None:
    """没有 override 时返回 base 的副本。"""
    assert merge_config({"a": 1}) == {"a": 1}


def test_merge_config_override_wins() -> None:
    """override 里的键覆盖 base。"""
    assert merge_config({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_merge_config_does_not_mutate_base() -> None:
    """★ 重要：不该修改传入的字典。

    有副作用的函数会让调用方困惑——尤其是 base 来自配置对象时，
    被偷偷改掉会导致很难查的 bug。
    """
    base = {"a": 1}
    merge_config(base, {"b": 2})
    assert base == {"a": 1}
