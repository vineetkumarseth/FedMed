"""The aggregation server (Flower).

Privacy boundary: this process has no patient data and no ML framework. It never imports
``fedmed.data``, ``fedmed.models`` or ``torch`` - ``fedmed audit`` enforces that statically.
It holds only the *public* HE context and adds ciphertexts.
"""
from __future__ import annotations

import time
from pathlib import Path

import flwr as fl

from ..config import Config
from ..privacy import he as hemod
from ..privacy import tls
from .events import EventLog
from .strategy import FedMedStrategy


def run_server(cfg: Config, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    events = EventLog(run_dir / "metrics.jsonl")

    public_ctx = None
    if cfg.he.enabled:
        public_ctx = hemod.load_context(Path(cfg.security.key_dir) / "he_public.ctx")
        if public_ctx.has_secret_key():
            raise RuntimeError("Refusing to start: the server was given a context that contains the SECRET key. "
                               "Give it he_public.ctx only.")

    events.log(f"FedMed server up on {cfg.federation.address} "
               f"(TLS={'on' if cfg.security.tls else 'off'}, HE={'on' if cfg.he.enabled else 'off'}, "
               f"DP={'on' if cfg.dp.enabled else 'off'})")

    strategy = FedMedStrategy(cfg, events, public_ctx)
    kwargs = {}
    if cfg.security.tls:
        kwargs["certificates"] = tls.load_server_certs(cfg.security.cert_dir)

    t0 = time.time()
    history = fl.server.start_server(
        server_address=cfg.federation.address,
        config=fl.server.ServerConfig(num_rounds=cfg.federation.rounds, round_timeout=cfg.federation.round_timeout_s),
        strategy=strategy, **kwargs,
    )
    dice = history.metrics_distributed.get("dice", []) if history.metrics_distributed else []
    events.emit("run_end", rounds_completed=len(history.losses_distributed), final_dice=dice[-1][1] if dice else None,
                wall_seconds=round(time.time() - t0, 1),
                epsilon=strategy.accountant.epsilon() if strategy.accountant else None)
    events.log("Federation finished")
