"""长期记忆：SQLite 存储 + 基于字符 bigram 的召回。"""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from agent_playground.rag import tokenize

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "memory.db"
TOP_K = 3
MAX_CHARS = 200


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            text         TEXT NOT NULL,
            tags         TEXT NOT NULL DEFAULT '',
            created_at   TEXT NOT NULL,
            last_used_at TEXT,
            use_count    INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return conn


def add(text: str, tags: str = "") -> int:
    """写入一条记忆；内容相同则直接返回已有 id。"""
    clean = text.strip()[:MAX_CHARS]
    if not clean:
        raise ValueError("记忆内容不能为空")

    now = datetime.now(UTC).isoformat(timespec="seconds")
    with closing(_connect()) as conn, conn:
        row = conn.execute(
            "SELECT id FROM memories WHERE text = ?", (clean,)
        ).fetchone()
        if row is not None:
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO memories (text, tags, created_at) VALUES (?, ?, ?)",
            (clean, tags.strip(), now),
        )
        return int(cur.lastrowid)


def all_memories() -> list[dict]:
    """全量返回，新的在前。"""
    with closing(_connect()) as conn, conn:
        rows = conn.execute(
            "SELECT id, text, tags, created_at, last_used_at, use_count "
            "FROM memories ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def search(query: str, top_k: int = TOP_K, min_overlap: int = 1) -> list[dict]:
    """按 bigram 重叠度召回，相关度相同再看使用次数和新鲜度。"""
    query = query.strip()
    if not query or top_k <= 0:
        return []

    q = set(tokenize(query))
    scored: list[tuple[int, int, str, dict]] = []
    for memory in all_memories():
        overlap = len(q & set(tokenize(f"{memory['text']} {memory['tags']}")))
        if overlap >= min_overlap:
            scored.append((overlap, memory["use_count"], memory["created_at"], memory))

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    hits = [memory for _, _, _, memory in scored[:top_k]]

    for memory in hits:
        touch(memory["id"])
        memory["use_count"] += 1
    return hits


def touch(memory_id: int) -> None:
    """召回一次就记一次使用，为将来的淘汰策略留数据。"""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
            (now, memory_id),
        )


def delete(memory_id: int) -> bool:
    """删一条记忆，返回是否真的删到了。"""
    with closing(_connect()) as conn, conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0
