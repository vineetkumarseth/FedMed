import numpy as np
import pytest

from fedmed.config import HEConfig
from fedmed.privacy import he


@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    d = tmp_path_factory.mktemp("keys")
    cfg = HEConfig()
    secret, public = he.generate_keys(cfg, d)
    return cfg, he.load_context(secret), he.load_context(public)


def test_server_context_cannot_decrypt(keys):
    cfg, secret, public = keys
    assert secret.has_secret_key() and not public.has_secret_key()
    chunks = he.encrypt_vector(secret, np.ones(10), he.slots(cfg))
    with pytest.raises(PermissionError):
        he.decrypt_vector(public, chunks, 10)


def test_encrypted_weighted_average_matches_plaintext(keys):
    cfg, secret, public = keys
    rng = np.random.default_rng(0)
    d = 4096 * 2 + 123            # spans three ciphertexts, last one partial
    ws = [rng.normal(0, 0.2, d).astype(np.float32) for _ in range(3)]
    ns = [24, 18, 30]
    sent = [he.encrypt_vector(secret, n * w.astype(np.float64), he.slots(cfg)) for n, w in zip(ns, ws)]
    summed = he.add_ciphertexts(public, sent)          # the server only has the public context
    got = he.decrypt_vector(secret, summed, d) / sum(ns)
    want = sum(n * w for n, w in zip(ns, ws)) / sum(ns)
    assert np.max(np.abs(got - want)) < 1e-4


def test_survivor_only_aggregation(keys):
    """If a node drops out, summing the remaining ciphertexts with the matching N still works."""
    cfg, secret, public = keys
    w = [np.full(50, 1.0), np.full(50, 3.0), np.full(50, 100.0)]
    ns = [10, 30, 999]
    enc = [he.encrypt_vector(secret, n * x, he.slots(cfg)) for n, x in zip(ns, w)]
    got = he.decrypt_vector(secret, he.add_ciphertexts(public, enc[:2]), 50) / 40
    assert np.allclose(got, 2.5, atol=1e-4)
