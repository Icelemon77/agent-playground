"""只读 SQL 数据分析 Agent：自然语言 → SQL → 校验 → 执行。"""

import asyncio
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from agent_playground.config import Settings
from agent_playground.errors import LLMCallError
from agent_playground.llm_client import achat_with_retry

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sales.db"
MAX_ROWS = 200
MAX_SQL_CHARS = 5000

FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "truncate",
)

SCHEMA = """
customers(
    id INTEGER PRIMARY KEY,
    name TEXT,
    city TEXT,
    signup_date TEXT  -- 'YYYY-MM-DD'
)
products(
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price REAL
)
orders(
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,   -- 对应 customers.id
    product_id INTEGER,    -- 对应 products.id
    quantity INTEGER,
    order_date TEXT,       -- 'YYYY-MM-DD'
    status TEXT            -- 'paid' / 'refunded' / 'cancelled'
)
"""

SYSTEM_PROMPT = (
    "你是 SQLite 数据分析师。根据用户问题生成一条 SQL 查询。\n"
    "规则：\n"
    "1. 只输出一条 SELECT（或以 WITH 开头的查询），不要解释。\n"
    "2. 只能使用下面列出的表和字段。\n"
    "3. 禁止 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH。\n"
    "4. 只输出一条语句，不要用分号分隔多条。\n"
    "5. 如果问题问的是单个对象（哪个、最高、最多、最低），用 LIMIT 1；其他可能返回多行的情况，加 LIMIT 200。\n"
    "6. 如果用户要求删除、修改、插入或清空数据，不要生成任何 SQL，只输出一行：REJECT: 只支持查询，不支持修改数据。\n"
    "7.业务口径以问题描述为准；问题没有说明状态时，不要自行添加 status 过滤。 \n"
    "8. 日期字段是 'YYYY-MM-DD' 文本，可以直接用字符串比较。\n\n"
    f"数据库结构：\n{SCHEMA}"
)
def extract_sql(text: str) -> str:
    stripped = text.strip()
    if stripped.upper().startswith("REJECT"):
        raise ValueError(stripped[6:].lstrip(": ").strip() or "只支持查询")
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            stripped = text.lstrip()
            if stripped.lower().startswith("sql"):
                text = stripped[3:]

    match = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if match is None:
        raise ValueError("模型没有返回 SQL")

    candidate = text[match.start() :].strip()

    # 去掉字符串字面量后再检查：SQL 关键字与列名都是 ASCII
    outside = re.sub(r"'[^']*'", "", candidate)
    if re.search(r"[\u4e00-\u9fff]", outside):
        raise ValueError("模型返回的是解释而不是 SQL")

    return candidate


async def generate_sql(question: str, settings: Settings) -> str:
    """调用模型生成 SQL，失败时抛 LLMCallError。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = await achat_with_retry(messages, settings)
    return extract_sql(result["answer"])
def validate_sql(sql: str) -> str:
    """只允许单条只读查询；不合法就抛 ValueError。"""
    text = sql.strip()
    if not text:
        raise ValueError("SQL 为空")
    if len(text) > MAX_SQL_CHARS:
        raise ValueError("SQL 过长")

    lowered = text.lower()
    if not lowered.startswith(("select", "with")):
        raise ValueError("只允许 SELECT / WITH 查询")

    body = text.rstrip(";")
    if ";" in body:
        raise ValueError("不允许多条语句")
    if "--" in body or "/*" in body:
        raise ValueError("不允许 SQL 注释")

    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            raise ValueError(f"禁止关键字：{word}")

    return body.strip()
def execute_sql(sql: str) -> dict[str, Any]:
    """在只读连接上执行查询，返回列名与行。"""
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")

        cursor = conn.execute(sql)
        columns = [col[0] for col in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_ROWS)
        return {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "truncated": len(rows) == MAX_ROWS,
        }
    finally:
        conn.close()

async def ask(question: str, settings: Settings) -> dict[str, Any]:
    """端到端：生成 → 校验 → 执行，所有失败都走返回值。"""
    started = time.perf_counter()
    result: dict[str, Any] = {
        "question": question,
        "sql": None,
        "columns": [],
        "rows": [],
        "truncated": False,
        "error": None,
        "detail": None,
        "elapsed_s": 0.0,
    }

    try:
        raw_sql = await generate_sql(question, settings)
    except (LLMCallError, httpx.HTTPError, TimeoutError) as exc:
        result["error"] = type(exc).__name__
        result["detail"] = str(exc)
        result["elapsed_s"] = time.perf_counter() - started
        return result
    except ValueError as exc:
        result["error"] = "NoSQL"
        result["detail"] = str(exc)
        result["elapsed_s"] = time.perf_counter() - started
        return result

    result["sql"] = raw_sql

    try:
        sql = validate_sql(raw_sql)
    except ValueError as exc:
        result["error"] = "UnsafeSQL"
        result["detail"] = str(exc)
        result["elapsed_s"] = time.perf_counter() - started
        return result

    try:
        data = execute_sql(sql)
    except sqlite3.Error as exc:
        result["error"] = "SQLError"
        result["detail"] = str(exc)
        result["elapsed_s"] = time.perf_counter() - started
        return result

    result.update(data)
    result["sql"] = sql
    result["elapsed_s"] = time.perf_counter() - started
    return result


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "一共有多少客户？"
    result = asyncio.run(ask(question, Settings()))

    print(f"问题：{result['question']}")
    print(f"SQL：{result['sql']}")
    if result["error"]:
        print(f"错误：{result['error']}（{result['detail']}）")
        return 1

    print(f"列：{result['columns']}")
    preview = result["rows"][:10]
    for row in preview:
        print(row)

    if len(result["rows"]) > len(preview):
        print(f"（共 {len(result['rows'])} 行，只显示前 {len(preview)} 行）")

    if result["truncated"]:
        print(f"（结果超过 {MAX_ROWS} 行，只取前 {MAX_ROWS} 行）")
    print(f"耗时：{result['elapsed_s']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
