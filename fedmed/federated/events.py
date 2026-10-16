"""Append-only JSONL event log - the single source of truth for runs.

The federation server writes it, the dashboard API tails it (and can replay a finished
run), and ``fedmed audit`` reads it. Keeping it a plain file means no extra services.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, type_: str, **payload: Any) -> Dict[str, Any]:
        event = {"type": type_, "t": round(time.time(), 3), **payload}
        line = json.dumps(event, separators=(",", ":"), default=_json_default)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return event

    def log(self, message: str, level: str = "info") -> None:
        self.emit("log", level=level, message=message)


def read_events(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written trailing line while the run is live


def _json_default(o: Any):
    try:
        import numpy as np
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
    except ImportError:  # pragma: no cover
        pass
    if o == float("inf"):
        return None
    raise TypeError(f"not JSON serialisable: {type(o)}")
