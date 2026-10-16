"""Dashboard backend: tails the run's JSONL event log and streams it over a WebSocket.

    GET  /api/health          liveness
    GET  /api/run             {run_id, events[]}   full history (for page loads / replays)
    GET  /api/nodes           reachability + status of each hospital node (polls their local ports)
    GET  /api/segmentation    exported MRI + mask volumes for the viewer
    GET  /api/baseline        centralized baseline curve
    GET  /api/audit           audit report
    WS   /ws                  history replay, then live events
"""
from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .federated.events import read_events


def latest_run(runs_dir: Path) -> Optional[Path]:
    marker = runs_dir / "LATEST"
    if marker.exists():
        p = runs_dir / marker.read_text().strip()
        if p.exists():
            return p
    runs = sorted((p for p in runs_dir.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="FedMed dashboard API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
    runs_dir = Path(cfg.runs_dir)

    def run_dir() -> Optional[Path]:
        return latest_run(runs_dir)

    def read_json(name: str) -> Any:
        rd = run_dir()
        if rd is None or not (rd / name).exists():
            raise HTTPException(404, f"{name} not available yet")
        return json.loads((rd / name).read_text())

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        rd = run_dir()
        return {"ok": True, "run": rd.name if rd else None}

    @app.get("/api/run")
    def run() -> Dict[str, Any]:
        rd = run_dir()
        if rd is None:
            raise HTTPException(404, "no runs yet - start one with `fedmed federate`")
        return {"run_id": rd.name, "events": list(read_events(rd / "metrics.jsonl"))}

    @app.get("/api/segmentation")
    def segmentation():
        return read_json("segmentation.json")

    @app.get("/api/baseline")
    def baseline():
        return read_json("baseline.json")

    @app.get("/api/audit")
    def audit():
        return read_json("audit.json")

    @app.get("/api/nodes")
    async def nodes() -> List[Dict[str, Any]]:
        def probe(hid: int) -> Dict[str, Any]:
            port = cfg.federation.node_status_base_port + hid - 1
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=0.6) as r:
                    return {"reachable": True, "port": port, **json.loads(r.read())}
            except Exception:
                return {"reachable": False, "hospital": hid, "port": port}
        loop = asyncio.get_running_loop()
        return await asyncio.gather(*(loop.run_in_executor(None, probe, h)
                                      for h in range(1, cfg.federation.num_hospitals + 1)))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        offset, current = 0, None
        try:
            while True:
                rd = run_dir()
                if rd is not None and rd != current:  # a new run started: restart the stream
                    current, offset = rd, 0
                    await sock.send_text(json.dumps({"type": "reset", "run_id": rd.name}))
                if current is not None:
                    path = current / "metrics.jsonl"
                    if path.exists():
                        with path.open("rb") as fh:
                            fh.seek(offset)
                            chunk = fh.read()
                        # only consume complete lines
                        cut = chunk.rfind(b"\n") + 1
                        for line in chunk[:cut].splitlines():
                            if line.strip():
                                await sock.send_text(line.decode("utf-8"))
                        offset += cut
                await asyncio.sleep(0.3)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
