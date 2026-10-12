"""成本估算的测试。

为什么值得测：
  1. 算钱不能错——它是给用户看的数字
  2. 纯函数——最好测的那类
  3. 边界多——缓存命中/未命中、高峰/闲时、未知模型名
"""

from datetime import UTC, datetime

import pytest

from agent_playground.cost import (
    PEAK_MULTIPLIER,
    USD_TO_CNY,
    estimate_cost_cny,
    is_peak_now,
)

# ---------------- is_peak_now ----------------


def test_peak_on_weekday_night() -> None:
    """★ 工作日 UTC 02:00 属于高峰（在 01:00-04:00 窗口内）。"""
    monday_2am = datetime(2026, 10, 12, 2, 0, tzinfo=UTC)  # 周一
    assert is_peak_now(monday_2am) is True


def test_peak_on_weekday_morning_window() -> None:
    """★ 工作日 UTC 07:00 也属于高峰（在 06:00-10:00 窗口内）。"""
    monday_7am = datetime(2026, 10, 12, 7, 0, tzinfo=UTC)
    assert is_peak_now(monday_7am) is True


def test_off_peak_on_weekday_midday() -> None:
    """★ 工作日 UTC 12:00 是闲时（两个窗口都不覆盖）。"""
    monday_noon = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
    assert is_peak_now(monday_noon) is False


def test_weekend_is_always_off_peak() -> None:
    """★ 周末全算闲时——即使在高峰时段的钟点。

    这条最有价值：它验证的是「周末优先于钟点判断」这个顺序。
    如果哪天有人把 weekday 检查挪到后面，它会立刻失败。
    """
    saturday_2am = datetime(2026, 10, 17, 2, 0, tzinfo=UTC)  # 周六，钟点在高峰窗口内
    sunday_7am = datetime(2026, 10, 18, 7, 0, tzinfo=UTC)

    assert is_peak_now(saturday_2am) is False
    assert is_peak_now(sunday_7am) is False


def test_is_peak_now_returns_bool() -> None:
    """不传参数时用当前时间，返回布尔值。"""
    assert isinstance(is_peak_now(), bool)


# ---------------- estimate_cost_cny ----------------


def test_cost_input_only() -> None:
    """100 万输入 token（未命中缓存）：0.15 USD × 7.2 = 1.08 CNY。"""
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}

    result = estimate_cost_cny("deepseek-flash", usage, is_peak=False)

    assert result == pytest.approx(0.15 * USD_TO_CNY)


def test_cost_output_only() -> None:
    """100 万输出 token：0.6 USD × 7.2 = 4.32 CNY。"""
    usage = {"prompt_tokens": 0, "completion_tokens": 1_000_000}

    result = estimate_cost_cny("deepseek-flash", usage, is_peak=False)

    assert result == pytest.approx(0.6 * USD_TO_CNY)


def test_peak_costs_double() -> None:
    """★ 高峰时段的价格正好是闲时的 PEAK_MULTIPLIER 倍。"""
    usage = {"prompt_tokens": 500_000, "completion_tokens": 200_000}

    off_peak = estimate_cost_cny("deepseek-flash", usage, is_peak=False)
    peak = estimate_cost_cny("deepseek-flash", usage, is_peak=True)

    assert peak == pytest.approx(off_peak * PEAK_MULTIPLIER)


def test_cache_hit_is_50x_cheaper_than_miss() -> None:
    """★ 缓存命中的单价便宜 50 倍（0.15 / 0.003）。"""
    million = 1_000_000

    hit = estimate_cost_cny(
        "deepseek-flash", {"prompt_cache_hit_tokens": million}, is_peak=False
    )
    miss = estimate_cost_cny(
        "deepseek-flash", {"prompt_cache_miss_tokens": million}, is_peak=False
    )

    assert miss == pytest.approx(hit * 50)


def test_cache_detail_takes_priority_over_total() -> None:
    """★ 有缓存明细时，不该把总输入当成全部未命中。

    如果实现写成「总是用 prompt_tokens 算未命中」，
    100 万全命中会被算成 100 万未命中——成本高估 50 倍。
    """
    usage = {
        "prompt_tokens": 1_000_000,
        "prompt_cache_hit_tokens": 1_000_000,
        "prompt_cache_miss_tokens": 0,
    }

    result = estimate_cost_cny("deepseek-flash", usage, is_peak=False)

    assert result == pytest.approx(1_000_000 / 1_000_000 * 0.003 * USD_TO_CNY)


def test_unknown_model_falls_back_to_flash() -> None:
    """★ 未知模型名退回 flash 价格，而不是崩。

    这是必要的兜底：模型名可能被改错、或者是新出的遗留名。
    """
    usage = {"prompt_tokens": 1_000_000}

    unknown = estimate_cost_cny("no-such-model", usage, is_peak=False)
    flash = estimate_cost_cny("deepseek-flash", usage, is_peak=False)

    assert unknown == pytest.approx(flash)


def test_pro_is_more_expensive_than_flash() -> None:
    """价格表里的 pro 应该比 flash 贵。"""
    usage = {"completion_tokens": 1_000_000}

    flash = estimate_cost_cny("deepseek-flash", usage, is_peak=False)
    pro = estimate_cost_cny("deepseek-v4-pro", usage, is_peak=False)

    assert pro > flash


def test_empty_usage_costs_zero() -> None:
    """★ 边界：没有用量数据时返回 0，不该崩。"""
    assert estimate_cost_cny("deepseek-flash", {}, is_peak=False) == 0.0
