import numpy as np

from fedmed.config import load_config
from fedmed.data import load_hospital_data, load_test_data
from fedmed.data.partition import iid_partition, lesion_size_partition


def small_cfg():
    return load_config(None, {"data.volume_size": 16, "data.n_channels": 2,
                              "data.train_per_hospital": 4, "data.val_per_hospital": 2})


def test_shapes_and_nonempty_masks():
    cfg = small_cfg()
    train, val = load_hospital_data(cfg, 1)
    x, y = train[0]
    assert tuple(x.shape) == (2, 16, 16, 16) and tuple(y.shape) == (1, 16, 16, 16)
    assert len(train) == 4 and len(val) == 2
    assert all(train[i][1].sum() > 0 for i in range(len(train)))


def test_reproducible_and_hospitals_differ():
    cfg = small_cfg()
    a, _ = load_hospital_data(cfg, 1)
    b, _ = load_hospital_data(cfg, 1)
    c, _ = load_hospital_data(cfg, 2)
    assert np.allclose(a[0][0].numpy(), b[0][0].numpy())
    assert not np.allclose(a[0][0].numpy(), c[0][0].numpy())


def test_test_split_is_disjoint_from_training_seeds():
    cfg = small_cfg()
    train, _ = load_hospital_data(cfg, 1)
    test = load_test_data(cfg)
    assert not np.allclose(train[0][0].numpy(), test[0][0].numpy())


def test_partitions_cover_everything_once():
    for parts in (iid_partition(30, 3, 0), lesion_size_partition(np.random.default_rng(0).random(30), 3, 0.5, 0)):
        merged = np.sort(np.concatenate(parts))
        assert merged.tolist() == list(range(30))
