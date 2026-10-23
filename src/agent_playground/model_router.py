CHEAP = "deepseek-flash"
STRONG = "deepseek-v4-pro"

COMPLEX_HINTS = ("分析", "对比", "设计", "方案", "为什么", "步骤", "推导", "权衡")


def pick_model(question: str, *, default: str = CHEAP) -> str:
    """按问题复杂度选模型：命中复杂信号或问题较长时用强模型。"""
    if len(question) > 80 or any(h in question for h in COMPLEX_HINTS):
        return STRONG
    return default

