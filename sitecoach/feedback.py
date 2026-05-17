"""Feedback store — TASK 5b.

Persists user-reported wrong→correct pairs to a JSONL file. Production
deployments should swap this for a real database, but JSONL is enough
for low-volume cloud agents and keeps the file readable/auditable.
"""

from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Optional


class FeedbackStore:
    """Thread-safe append-only JSONL persistence."""

    def __init__(self, path: str | Path = "feedback/corrections.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self,
               wrong_word: str,
               correct_word: str,
               audio_id: Optional[str] = None,
               user_id: Optional[str] = None,
               category: str = "USER_REPORT",
               notes: str = "") -> dict:
        if not wrong_word or not correct_word:
            raise ValueError("wrong_word and correct_word are required")

        entry = {
            "ts": int(time.time()),
            "wrongWord": wrong_word.strip(),
            "correctWord": correct_word.strip(),
            "category": category,
            "audioId": audio_id,
            "userId": user_id,
            "notes": notes,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def stats(self) -> dict:
        entries = self.all()
        by_category: dict[str, int] = {}
        for e in entries:
            by_category[e.get("category", "UNKNOWN")] = (
                by_category.get(e.get("category", "UNKNOWN"), 0) + 1
            )
        return {
            "total": len(entries),
            "by_category": by_category,
            "path": str(self.path),
        }
