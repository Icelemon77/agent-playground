# scripts/db_fingerprint.py
import hashlib
import sqlite3

TABLES = ("customers", "products", "orders")


def fingerprint(db_path: str = "data/sales.db") -> str:
    conn = sqlite3.connect(db_path)
    digest = hashlib.sha256()
    try:
        for table in TABLES:
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY id"):
                digest.update(repr(tuple(row)).encode("utf-8"))
    finally:
        conn.close()
    return digest.hexdigest()[:16]


if __name__ == "__main__":
    print(fingerprint())