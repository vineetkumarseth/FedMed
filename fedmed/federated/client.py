"""A hospital node.

Everything touching patient data (loading, training, evaluation) happens here, on the
hospital's machine. What leaves the node is a *clipped, noised, encrypted* weight vector
plus a handful of scalar metrics.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import (Code, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns,
                         GetParametersRes, Parameters, Status, ndarrays_to_parameters,
                         parameters_to_ndarrays)

from ..config import Config
from ..data import load_hospital_data
from ..data.synthetic import HOSPITAL_PROFILES
from ..models import build_model
from ..privacy import dp as dpmod
from ..privacy import he as hemod
from ..privacy import tls
from ..training.local import evaluate, get_flat, resolve_device, set_flat, set_threads, train_epochs
from .constants import CHAOS_EXIT_CODE
from .status import NodeStatus, serve_status

OK = Status(code=Code.OK, message="OK")


class HospitalClient(fl.client.Client):
    def __init__(self, cfg: Config, hospital_id: int, run_dir: Optional[Path] = None,
                 crash_at_round: Optional[int] = None):
        self.cfg = cfg
        self.hid = hospital_id
        self.run_dir = Path(run_dir) if run_dir else None
        self.crash_at_round = crash_at_round
        set_threads(cfg.train.torch_threads)
        self.device = resolve_device(cfg.train.device)

        self.train_ds, self.val_ds = load_hospital_data(cfg, hospital_id)
        torch.manual_seed(cfg.train.seed)  # identical initial weights on every node
        self.model = build_model(cfg)
        self.n_params = sum(p.numel() for p in self.model.parameters())

        self.he_ctx = None
        if cfg.he.enabled:
            self.he_ctx = hemod.load_context(Path(cfg.security.key_dir) / "he_secret.ctx")
            if not self.he_ctx.has_secret_key():
                raise RuntimeError("Hospital nodes need the *secret* HE context (he_secret.ctx)")

        name = HOSPITAL_PROFILES.get(hospital_id, None)
        self.status = NodeStatus(hospital_id, name.name if name else f"Hospital {hospital_id}",
                                 len(self.train_ds), len(self.val_ds))
        self._httpd = serve_status(self.status, cfg.federation.node_status_base_port + hospital_id - 1)

    # --- helpers -----------------------------------------------------------------------
    def _receive(self, parameters: Parameters, config) -> np.ndarray:
        """Return the global model as a plaintext flat vector (decrypting if needed)."""
        if parameters.tensor_type == hemod.CIPHERTEXT_TYPE:
            if self.he_ctx is None:
                raise RuntimeError("Received ciphertext but HE is disabled on this node")
            total = float(config["total_examples"])
            summed = hemod.decrypt_vector(self.he_ctx, parameters.tensors, self.n_params)
            return (summed / total).astype(np.float32)
        return parameters_to_ndarrays(parameters)[0].astype(np.float32)

    # --- Flower Client API -------------------------------------------------------------
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(status=OK, parameters=ndarrays_to_parameters([get_flat(self.model)]))

    def fit(self, ins: FitIns) -> FitRes:
        rnd = int(ins.config["round"])
        cfg = self.cfg
        self.status.update(state="training", round=rnd)
        t_all = time.perf_counter()

        global_flat = self._receive(ins.parameters, ins.config)
        set_flat(self.model, global_flat)

        if self.crash_at_round == rnd:
            self.status.update(state="crashed")
            time.sleep(0.5)
            os._exit(CHAOS_EXIT_CODE)  # simulate the node dying mid-epoch

        stats = train_epochs(
            self.model, self.train_ds, epochs=int(ins.config["local_epochs"]),
            batch_size=cfg.train.batch_size, lr=cfg.train.lr, device=self.device,
            seed=cfg.train.seed + 97 * rnd + self.hid,
            proximal_mu=float(ins.config.get("fedprox_mu", 0.0)), global_flat=global_flat,
        )

        delta = get_flat(self.model) - global_flat
        pre_norm, clipped = float(np.linalg.norm(delta)), False
        if cfg.dp.enabled:
            delta, pre_norm, clipped = dpmod.clip_update(delta, cfg.dp.clip_norm)
            # Fresh OS entropy on purpose: DP noise must not be reproducible from a known seed.
            delta = dpmod.add_gaussian_noise(delta, cfg.dp.clip_norm, cfg.dp.noise_multiplier,
                                             np.random.default_rng())
        sent = (global_flat + delta).astype(np.float32)
        n = len(self.train_ds)

        t_enc = time.perf_counter()
        if self.he_ctx is not None:
            chunks = hemod.encrypt_vector(self.he_ctx, n * sent.astype(np.float64), hemod.slots(cfg.he))
            params = Parameters(tensors=chunks, tensor_type=hemod.CIPHERTEXT_TYPE)
        else:
            params = ndarrays_to_parameters([sent])
        enc_ms = (time.perf_counter() - t_enc) * 1000
        payload_mb = sum(len(t) for t in params.tensors) / 1e6

        self.status.update(state="idle", last_loss=round(stats["loss"], 4),
                           last_dice=round(stats["dice"], 4), last_payload_mb=round(payload_mb, 2))
        return FitRes(status=OK, parameters=params, num_examples=n, metrics={
            "hospital": self.hid, "train_loss": stats["loss"], "train_dice": stats["dice"],
            "train_seconds": stats["seconds"], "update_norm": pre_norm, "clipped": int(clipped),
            "encrypt_ms": enc_ms, "payload_mb": payload_mb,
            "round_seconds": time.perf_counter() - t_all,
        })

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        rnd = int(ins.config["round"])
        self.status.update(state="evaluating", round=rnd)
        flat = self._receive(ins.parameters, ins.config)
        set_flat(self.model, flat)
        m = evaluate(self.model, self.val_ds, batch_size=self.cfg.train.batch_size, device=self.device)
        if self.run_dir is not None:  # the *hospital* keeps the decrypted global model
            d = self.run_dir / "models"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"round": rnd, "state_dict": self.model.state_dict()}, d / f"hospital_{self.hid}.pt")
        self.status.update(state="idle", last_dice=round(m["dice"], 4))
        return EvaluateRes(status=OK, loss=m["loss"], num_examples=m["n"], metrics={
            "hospital": self.hid, "dice": m["dice"], "iou": m["iou"]})


def run_client(cfg: Config, hospital_id: int, run_dir: Optional[Path] = None,
               crash_at_round: Optional[int] = None) -> None:
    client = HospitalClient(cfg, hospital_id, run_dir, crash_at_round)
    kwargs = dict(server_address=cfg.federation.address, client=client)
    if cfg.security.tls:
        kwargs["root_certificates"] = tls.load_ca(cfg.security.cert_dir)
    else:
        kwargs["insecure"] = True
    fl.client.start_client(**kwargs)
