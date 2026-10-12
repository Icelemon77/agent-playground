"""token 成本估算。

价格来源：DeepSeek 官方定价页，抓取日期 2026-10-08，单位：美元 / 每 100 万 token。
高峰时段：UTC 01:00-04:00 与 06:00-10:00 的工作日（北京时间 09:00-12:00、14:00-18:00），
高峰价 = 闲时价 × 2。
价格会变，改动时请同步更新这里和抓取日期。
"""

from datetime import UTC, datetime

USD_PER_MTOK: dict[str, dict[str, float]] = {
    "deepseek-flash": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6},
    "deepseek-v4-pro": {"cache_hit": 0.022, "cache_miss": 0.66, "output": 1.98},
}
PEAK_MULTIPLIER = 2.0
USD_TO_CNY = 7.2


def is_peak_now(now: datetime | None = None) -> bool:
    """是否处于高峰时段（周末和节假日全为闲时）。

    now 参数只为可测试性而存在——不传就用当前时间。
    如果把 datetime.now() 写死在函数里，测试就没法验证
    「周六凌晨 2 点算不算高峰」这种问题。
    """
    moment = now or datetime.now(UTC)
    if moment.weekday() >= 5:
        return False
    return 1 <= moment.hour < 4 or 6 <= moment.hour < 10


def estimate_cost_cny(model: str, usage: dict, *, is_peak: bool | None = None) -> float:
    """按用量估算花费（人民币元）。"""
    price = USD_PER_MTOK.get(model, USD_PER_MTOK["deepseek-flash"])
    peak = is_peak_now() if is_peak is None else is_peak
    multiplier = PEAK_MULTIPLIER if peak else 1.0

    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    cache_miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0))
    output = usage.get("completion_tokens", 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens", 0
    )

    usd = (
        cache_hit / 1_000_000 * price["cache_hit"]
        + cache_miss / 1_000_000 * price["cache_miss"]
        + output / 1_000_000 * price["output"]
    ) * multiplier
    _ = reasoning  # 思考 token 已含在 output 里，这里只是留个引用点
    return usd * USD_TO_CNY
