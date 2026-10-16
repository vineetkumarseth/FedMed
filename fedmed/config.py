"""Typed configuration loaded from YAML.

Every knob of an experiment lives in a single YAML file so a run can be
reproduced from ``configs/<name>.yaml`` plus a seed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"          # "synthetic" | "brats"
    brats_dir: Optional[str] = None    # root folder of BraTS cases (source="brats")
    volume_size: int = 32              # cubic crop / synthetic volume edge length
    n_channels: int = 2                # MRI modalities (e.g. FLAIR + T1ce)
    train_per_hospital: int = 24
    val_per_hospital: int = 8
    test_cases: int = 12               # held-out cases used for visual export
    seed: int = 7


@dataclass
class ModelConfig:
    name: str = "unet3d"               # "unet3d" | "monai_unet"
    base_filters: int = 8
    depth: int = 3


@dataclass
class TrainConfig:
    batch_size: int = 2
    lr: float = 2e-3
    local_epochs: int = 1
    fedprox_mu: float = 0.0            # >0 enables the FedProx proximal term
    device: str = "auto"
    torch_threads: int = 1             # per process; 3 hospitals + server share one machine in the local demo
    seed: int = 11


@dataclass
class FederationConfig:
    num_hospitals: int = 3
    rounds: int = 8
    host: str = "127.0.0.1"
    port: int = 8080
    min_clients: int = 2               # a round proceeds with at least this many nodes
    round_timeout_s: float = 600.0
    join_grace_s: float = 30.0         # round 1 waits this long for ALL hospitals before starting
    rejoin_grace_s: float = 3.0        # later rounds wait this long for a missing/rejoining hospital
    node_status_base_port: int = 8101  # hospital i exposes /status on base_port + i - 1

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class HEConfig:
    enabled: bool = True
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [60, 40, 60])
    global_scale_bits: int = 40


@dataclass
class DPConfig:
    enabled: bool = True
    clip_norm: float = 8.0
    noise_multiplier: float = 0.0005
    delta: float = 1e-5


@dataclass
class SecurityConfig:
    tls: bool = True
    cert_dir: str = "certs"
    key_dir: str = "keys"


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class AuditConfig:
    min_baseline_ratio: float = 0.85   # federated Dice must reach >= ratio * centralized Dice


@dataclass
class Config:
    run_name: str = "fedmed"
    runs_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    he: HEConfig = field(default_factory=HEConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, raw: Dict[str, Any]):
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    if raw is None:
        return cls()
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys {sorted(unknown)} for {cls.__name__}")
    kwargs = {}
    for name, value in raw.items():
        ftype = known[name].type
        target = globals().get(ftype) if isinstance(ftype, str) else ftype
        if dataclasses.is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _build(target, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    raw: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    cfg = _build(Config, raw)
    for dotted, value in (overrides or {}).items():
        target: Any = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            target = getattr(target, p)
        if not hasattr(target, leaf):
            raise ValueError(f"Unknown override '{dotted}'")
        setattr(target, leaf, value)
    return cfg
