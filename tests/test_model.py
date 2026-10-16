import numpy as np
import torch

from fedmed.config import load_config
from fedmed.models import build_model
from fedmed.training.local import get_flat, set_flat


def test_forward_shape_and_no_buffers():
    cfg = load_config(None, {"data.n_channels": 2, "model.base_filters": 4, "model.depth": 2})
    model = build_model(cfg)
    out = model(torch.randn(2, 2, 16, 16, 16))
    assert tuple(out.shape) == (2, 1, 16, 16, 16)
    # The model must be exactly its parameters, otherwise federated averaging silently drops state.
    assert len(list(model.buffers())) == 0


def test_flat_roundtrip():
    cfg = load_config(None, {"model.base_filters": 4, "model.depth": 2})
    a, b = build_model(cfg), build_model(cfg)
    set_flat(b, get_flat(a))
    assert np.array_equal(get_flat(a), get_flat(b))
