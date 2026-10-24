"""SQL Agent 的抽取与安全校验测试（零 API 成本）。"""

import pytest

from agent_playground.sql_agent import extract_sql, validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM customers",
        "SELECT * FROM orders LIMIT 10",
        "WITH t AS (SELECT 1) SELECT * FROM t",
    ],
)
def test_validate_allows_read_only_queries(sql: str) -> None:
    assert validate_sql(sql) == sql


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM orders", "写操作"),
        ("UPDATE orders SET status = 'x'", "更新"),
        ("SELECT 1; DROP TABLE orders", "多语句"),
        ("SELECT * FROM orders -- 注释", "注释"),
        ("PRAGMA table_info(orders)", "PRAGMA"),
    ],
)
def test_validate_rejects_unsafe_sql(sql: str, reason: str) -> None:
    with pytest.raises(ValueError):
        validate_sql(sql)


def test_extract_strips_code_fence() -> None:
    text = "```sql\nSELECT COUNT(*) FROM customers;\n```"
    assert extract_sql(text) == "SELECT COUNT(*) FROM customers;"


def test_extract_drops_preamble() -> None:
    text = "好的，这是查询：SELECT * FROM orders"
    assert extract_sql(text) == "SELECT * FROM orders"


def test_extract_rejects_prose_refusal() -> None:
    with pytest.raises(ValueError):
        extract_sql("只能生成 SELECT 查询，不能生成删除语句。")


def test_extract_rejects_reject_marker() -> None:
    with pytest.raises(ValueError):
        extract_sql("REJECT: 只支持查询，不支持修改数据")