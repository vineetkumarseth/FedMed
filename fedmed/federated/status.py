"""Tiny HTTP status endpoint each hospital node exposes on its own local port.

``GET /health`` -> ``ok`` and ``GET /status`` -> JSON about the node. The dashboard API
polls these to show which nodes are reachable. (In Flower the node dials *out* to the
server for training traffic; this port is purely for operations/monitoring.)
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict


class NodeStatus:
    def __init__(self, hospital_id: int, name: str, n_train: int, n_val: int):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "hospital": hospital_id, "name": name, "state": "idle", "round": 0,
            "n_train": n_train, "n_val": n_val, "last_loss": None, "last_dice": None,
            "last_payload_mb": None, "started": time.time(),
        }

    def update(self, **kw: Any) -> None:
        with self._lock:
            self._data.update(kw)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            d = dict(self._data)
        d["uptime_s"] = round(time.time() - d.pop("started"), 1)
        return d


def serve_status(status: NodeStatus, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                body, ctype = b"ok", "text/plain"
            elif self.path == "/status":
                body, ctype = json.dumps(status.snapshot()).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # silence default access log
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name=f"status-{port}").start()
    return httpd
