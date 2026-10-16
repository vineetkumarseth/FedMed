import pytest

from fedmed.config import load_config


def test_defaults_and_overrides():
    cfg = load_config(None, {"federation.rounds": 3, "dp.enabled": False})
    assert cfg.federation.rounds == 3 and cfg.dp.enabled is False
    assert cfg.federation.address == "127.0.0.1:8080"


def test_yaml_files_load():
    for name in ("default", "fast"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.federation.num_hospitals == 3


def test_unknown_keys_are_rejected(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("federation:\n  roundz: 3\n")
    with pytest.raises(ValueError):
        load_config(f)
    with pytest.raises(ValueError):
        load_config(None, {"federation.nope": 1})
