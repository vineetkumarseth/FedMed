"""CKKS homomorphic encryption helpers (TenSEAL / Microsoft SEAL).

Protocol
--------
* A **consortium key** is generated once (``fedmed gen-keys``). The *secret* context
  (``keys/he_secret.ctx``) is distributed to hospitals only. The *public* context
  (``keys/he_public.ctx``) - which cannot decrypt - goes to the aggregation server.
* Each hospital encrypts ``n_i * w_i`` (its sample count times its locally trained,
  DP-protected weights). The server only ever performs **ciphertext additions**, so no
  multiplicative depth, relinearisation or Galois keys are required.
* The server returns ``E(sum_i n_i * w_i)`` and the plaintext integer ``N = sum_i n_i``.
  Hospitals decrypt and divide by ``N`` to obtain the FedAvg model.

Limitation: this is single-key HE (all hospitals share one secret key). A production
deployment would use threshold / multi-key HE so no single party can decrypt alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import tenseal as ts

from ..config import HEConfig

CIPHERTEXT_TYPE = "tenseal.ckks.v1"


def make_context(cfg: HEConfig) -> ts.Context:
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, cfg.poly_modulus_degree, -1, cfg.coeff_mod_bit_sizes)
    ctx.global_scale = 2.0 ** cfg.global_scale_bits
    return ctx


def slots(cfg: HEConfig) -> int:
    """CKKS packs poly_modulus_degree / 2 real numbers into one ciphertext."""
    return cfg.poly_modulus_degree // 2


def generate_keys(cfg: HEConfig, key_dir: str | Path) -> tuple[Path, Path]:
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    ctx = make_context(cfg)
    secret_path = key_dir / "he_secret.ctx"
    public_path = key_dir / "he_public.ctx"
    secret_path.write_bytes(ctx.serialize(save_secret_key=True))
    pub = ctx.copy()
    pub.make_context_public()
    public_path.write_bytes(pub.serialize())
    return secret_path, public_path


def load_context(path: str | Path) -> ts.Context:
    return ts.context_from(Path(path).read_bytes())


def encrypt_vector(ctx: ts.Context, vec: np.ndarray, chunk: int) -> List[bytes]:
    """Encrypt a flat float vector as a list of CKKS ciphertexts of at most ``chunk`` slots."""
    vec = np.asarray(vec, dtype=np.float64)
    out = []
    for start in range(0, len(vec), chunk):
        out.append(ts.ckks_vector(ctx, vec[start:start + chunk].tolist()).serialize())
    return out


def decrypt_vector(ctx: ts.Context, chunks: Sequence[bytes], length: int) -> np.ndarray:
    if not ctx.has_secret_key():
        raise PermissionError("This context holds no secret key and cannot decrypt")
    parts = [np.asarray(ts.ckks_vector_from(ctx, c).decrypt(), dtype=np.float64) for c in chunks]
    flat = np.concatenate(parts) if parts else np.zeros(0)
    if len(flat) != length:
        raise ValueError(f"Decrypted length {len(flat)} != expected {length}")
    return flat.astype(np.float32)


def add_ciphertexts(ctx: ts.Context, contributions: Sequence[Sequence[bytes]]) -> List[bytes]:
    """Homomorphically sum several encrypted vectors chunk-by-chunk (server side, public key only)."""
    n_chunks = {len(c) for c in contributions}
    if len(n_chunks) != 1:
        raise ValueError(f"Contributions disagree on number of ciphertext chunks: {sorted(n_chunks)}")
    summed = []
    for i in range(n_chunks.pop()):
        acc = ts.ckks_vector_from(ctx, contributions[0][i])
        for contrib in contributions[1:]:
            acc = acc + ts.ckks_vector_from(ctx, contrib[i])
        summed.append(acc.serialize())
    return summed
