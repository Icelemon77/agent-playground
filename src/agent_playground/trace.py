"""Trace 记录：把一次 Agent 运行的每一步写进独立的文件。

与 logs/app.jsonl 的区别：
  app.jsonl      按「事件」记录，用于排查和统计，长期保留
  trace_*.jsonl  按「一次运行」记录，用于回放和评测，可以定期清理
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Asia/Shanghai")


class Tracer:
    """一次运行的 trace 记录器。"""

    def __init__(
        self, log_dir: str | Path, question: str, model: str, max_steps: int
    ) -> None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"trace_{stamp}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")
        self.write(
            {
                "type": "start",
                "time": datetime.now(tz=LOCAL_TZ).isoformat(timespec="seconds"),
                "question": question,
                "model": model,
                "max_steps": max_steps,
            }
        )

    def write(self, record: dict[str, Any]) -> None:
        """写一条记录。每次都 flush——程序崩溃时 trace 才是最有用的。"""
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
