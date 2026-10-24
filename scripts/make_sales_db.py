"""生成 SQL 数据分析 Agent 用的演示库（固定种子，可复现）。"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sales.db"

SCHEMA = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,
    price    REAL NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id  INTEGER NOT NULL,
    quantity    INTEGER NOT NULL,
    order_date  TEXT NOT NULL,
    status      TEXT NOT NULL
);
"""
CITIES = ["北京", "上海", "广州", "成都", "杭州"]

PRODUCTS = [
    (1, "无线鼠标", "电子产品", 89.0),
    (2, "机械键盘", "电子产品", 329.0),
    (3, "蓝牙耳机", "电子产品", 199.0),
    (4, "台灯", "家居", 79.0),
    (5, "抱枕", "家居", 49.0),
    (6, "收纳盒", "家居", 29.0),
    (7, "Python 入门", "图书", 68.0),
    (8, "算法图解", "图书", 59.0),
    (9, "数据库原理", "图书", 45.0),
    (10, "瑜伽垫", "运动", 99.0),
]

STATUSES = ["paid", "refunded", "cancelled"]
STATUS_WEIGHTS = [70, 15, 15]

def random_date(start: date, end: date) -> str:
    span = (end - start).days
    return (start + timedelta(days=random.randint(0, span))).isoformat()
customers = [
    (i, f"客户{i:02d}", random.choice(CITIES),
     random_date(date(2026, 6, 1), date(2026, 10, 10)))
    for i in range(1, 31)
]
orders = [
    (i,
     random.randint(1, 30),      # customer_id
     random.randint(1, 10),      # product_id
     random.randint(1, 4),       # quantity
     random_date(date(2026, 8, 1), date(2026, 10, 15)),
     random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0])
    for i in range(1, 121)
]
def main() -> None:
    random.seed(42)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()          # 每次重建，保证结果可复现

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO customers (id, name, city, signup_date) VALUES (?, ?, ?, ?)",
            customers,
        )
        conn.executemany(
            "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
            PRODUCTS,
        )
        conn.executemany(
            "INSERT INTO orders "
            "(id, customer_id, product_id, quantity, order_date, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            orders,
        )
        conn.commit()
        conn.executescript(
            "CREATE INDEX idx_orders_customer ON orders(customer_id);"
            "CREATE INDEX idx_orders_date ON orders(order_date);"
        )
        assert len(customers) == 30
        assert len(PRODUCTS) == 10
        assert len(orders) == 120
        assert len({c[2] for c in customers}) >= 4
        assert sum(1 for o in orders if o[4].startswith("2026-09")) >= 30
        assert sum(1 for c in customers if c[3].startswith("2026-09")) >= 5
        assert len({o[5] for o in orders}) == 3
    finally:

        conn.close()


if __name__ == "__main__":
    main()