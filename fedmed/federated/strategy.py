"""Aggregation strategy for the federation server.

The server never sees patient data and - in HE mode - never sees plaintext model weights
either: it adds ciphertexts and relays the encrypted sum. All training telemetry comes
from scalar metrics the hospitals report.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from ..config import Config
from ..privacy import dp as dpmod
from ..privacy import he as hemod
from .events import EventLog
from .memory import release_memory


def _payload_label(params: Parameters) -> str:
    """'tenseal.ckks.v1' for encrypted payloads, otherwise 'plaintext' (whatever Flower calls the numpy type)."""
    return params.tensor_type if params.tensor_type == hemod.CIPHERTEXT_TYPE else "plaintext"


class FedMedStrategy(Strategy):
    def __init__(self, cfg: Config, events: EventLog, public_ctx=None):
        super().__init__()
        self.cfg = cfg
        self.events = events
        self.public_ctx = public_ctx
        self.params: Optional[Parameters] = None   # initial weights come from a hospital (see below)
        self.total_examples: int = 0          # N for the current (encrypted) global model
        self.n_params = 0
        self.accountant = dpmod.RDPAccountant(cfg.dp.noise_multiplier, cfg.dp.delta) if cfg.dp.enabled else None
        self.expected = set(range(1, cfg.federation.num_hospitals + 1))
        self._round_started: Dict[int, float] = {}

    def __repr__(self) -> str:
        return "FedMedStrategy"

    # --- model init --------------------------------------------------------------------
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        # Returning None makes Flower ask one connected hospital for its (seeded) initial weights.
        # The server therefore needs no model code - and no ML framework - at all.
        return None

    def evaluate(self, server_round: int, parameters: Parameters):
        return None  # the server has no data; evaluation is federated

    def _announce(self, n_params: int) -> None:
        cfg = self.cfg
        self.n_params = n_params
        n_chunks = -(-n_params // hemod.slots(cfg.he)) if cfg.he.enabled else 0
        self.events.emit(
            "run_start", run_id=self.events.path.parent.name, rounds=cfg.federation.rounds,
            hospitals=cfg.federation.num_hospitals, model=cfg.model.name, n_params=n_params,
            volume_size=cfg.data.volume_size,
            he={"enabled": cfg.he.enabled, "scheme": "CKKS", "poly_modulus_degree": cfg.he.poly_modulus_degree,
                "coeff_mod_bit_sizes": cfg.he.coeff_mod_bit_sizes, "ciphertexts_per_update": n_chunks},
            dp={"enabled": cfg.dp.enabled, "clip_norm": cfg.dp.clip_norm,
                "noise_multiplier": cfg.dp.noise_multiplier, "delta": cfg.dp.delta},
            tls=cfg.security.tls, local_epochs=cfg.train.local_epochs, fedprox_mu=cfg.train.fedprox_mu,
            data_source=cfg.data.source,
        )

    # --- fit ---------------------------------------------------------------------------
    def _config(self, server_round: int) -> Dict[str, Scalar]:
        return {"round": server_round, "total_examples": self.total_examples,
                "local_epochs": self.cfg.train.local_epochs, "fedprox_mu": self.cfg.train.fedprox_mu}

    def _sample(self, client_manager: ClientManager, server_round: int) -> List[ClientProxy]:
        fed = self.cfg.federation
        # Give every hospital a chance to be present (a long grace at start-up, a short one later so a
        # crashed node that restarts can rejoin), but never block the federation on a missing node.
        grace = fed.join_grace_s if server_round == 1 else fed.rejoin_grace_s
        client_manager.wait_for(fed.num_hospitals, timeout=max(1, int(grace)))
        client_manager.wait_for(fed.min_clients, timeout=int(fed.round_timeout_s))
        n = min(fed.num_hospitals, len(client_manager.all()))
        return client_manager.sample(num_clients=n, min_num_clients=fed.min_clients)

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        if self.params is None:  # round 1: `parameters` are the initial weights a hospital handed over
            self.params = parameters
            self._announce(len(parameters_to_ndarrays(parameters)[0]))
        clients = self._sample(client_manager, server_round)
        self._round_started[server_round] = time.time()
        self.events.emit("round_start", round=server_round, nodes_online=len(clients),
                         payload_type=_payload_label(self.params))
        ins = FitIns(self.params, self._config(server_round))
        return [(c, ins) for c in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
        t0 = time.perf_counter()
        responded = {int(r.metrics["hospital"]) for _, r in results}
        dropped = sorted(self.expected - responded)
        if dropped:
            self.events.log(f"Round {server_round}: node(s) {dropped} did not respond - continuing with {sorted(responded)}",
                            level="warn")

        if len(results) < self.cfg.federation.min_clients:
            self.events.emit("round_skipped", round=server_round, responded=sorted(responded),
                             reason=f"need >= {self.cfg.federation.min_clients} nodes")
            return None, {}

        n_by_hospital = {int(r.metrics["hospital"]): int(r.num_examples) for _, r in results}
        total = sum(n_by_hospital.values())

        if self.public_ctx is not None:
            summed = hemod.add_ciphertexts(self.public_ctx, [r.parameters.tensors for _, r in results])
            new_params = Parameters(tensors=summed, tensor_type=hemod.CIPHERTEXT_TYPE)
        else:
            vecs = [parameters_to_ndarrays(r.parameters)[0] for _, r in results]
            weights = np.array([r.num_examples for _, r in results], dtype=np.float64) / total
            avg = np.sum([w * v.astype(np.float64) for w, v in zip(weights, vecs)], axis=0)
            new_params = ndarrays_to_parameters([avg.astype(np.float32)])
        self.params, self.total_examples = new_params, total
        if self.accountant:
            self.accountant.step()
        agg_ms = (time.perf_counter() - t0) * 1000

        clients = []
        for _, r in sorted(results, key=lambda cr: int(cr[1].metrics["hospital"])):
            m = r.metrics
            clients.append({
                "hospital": int(m["hospital"]), "n": int(r.num_examples),
                "train_loss": float(m["train_loss"]), "train_dice": float(m["train_dice"]),
                "train_seconds": float(m["train_seconds"]), "update_norm": float(m["update_norm"]),
                "clipped": bool(m["clipped"]), "encrypt_ms": float(m["encrypt_ms"]),
                "payload_mb": float(m["payload_mb"]),
            })
        eps = self.accountant.epsilon() if self.accountant else None
        noise_to_signal = None
        if self.cfg.dp.enabled and self.n_params:
            # Size of the injected Gaussian noise relative to the (clipped) update it is added to.
            dp = self.cfg.dp
            signal = float(np.mean([min(c["update_norm"], dp.clip_norm) for c in clients]))
            noise_to_signal = dp.noise_multiplier * dp.clip_norm * math.sqrt(self.n_params) / max(signal, 1e-12)
        self.events.emit(
            "fit_round", round=server_round, clients=clients, dropped=dropped, total_examples=total,
            aggregate_ms=round(agg_ms, 1), payload_type=_payload_label(new_params),
            server_saw_plaintext_weights=self.public_ctx is None, epsilon=eps,
            noise_to_signal=noise_to_signal,
            seconds=round(time.time() - self._round_started.get(server_round, time.time()), 2),
        )
        release_memory()
        return new_params, {}

    # --- evaluate ----------------------------------------------------------------------
    def configure_evaluate(self, server_round: int, parameters: Parameters,
                           client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = self._sample(client_manager, server_round)
        ins = EvaluateIns(self.params, self._config(server_round))
        return [(c, ins) for c in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]):
        if not results:
            return None, {}
        n = np.array([r.num_examples for _, r in results], dtype=np.float64)
        w = n / n.sum()
        loss = float(np.sum(w * np.array([r.loss for _, r in results])))
        dice = float(np.sum(w * np.array([r.metrics["dice"] for _, r in results])))
        iou = float(np.sum(w * np.array([r.metrics["iou"] for _, r in results])))
        clients = sorted(({"hospital": int(r.metrics["hospital"]), "loss": float(r.loss),
                           "dice": float(r.metrics["dice"]), "iou": float(r.metrics["iou"]),
                           "n": int(r.num_examples)} for _, r in results), key=lambda c: c["hospital"])
        eps = self.accountant.epsilon() if self.accountant else None
        self.events.emit("eval_round", round=server_round, loss=loss, dice=dice, iou=iou,
                         clients=clients, epsilon=eps)
        release_memory()
        return loss, {"dice": dice}
