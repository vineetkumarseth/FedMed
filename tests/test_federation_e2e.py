"""End-to-end: real Flower server + 3 hospital subprocesses over gRPC/TLS with CKKS aggregation."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_federation(tmp_path, ports, extra=()):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, "-m", "fedmed.cli", "federate", "--config", str(ROOT / "configs/fast.yaml"),
           "--name", "t", "--skip-baseline", "--no-api",
           "--set", f"federation.port={ports[0]}", "--set", f"federation.node_status_base_port={ports[1]}",
           "--set", "federation.round_timeout_s=90", *extra]
    return subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "runs/t/metrics.jsonl").read_text().splitlines()]


@pytest.mark.slow
def test_encrypted_federation_runs_end_to_end(tmp_path, ports):
    r = run_federation(tmp_path, ports)
    assert r.returncode == 0, r.stdout + r.stderr
    ev = events(tmp_path)
    fits = [e for e in ev if e["type"] == "fit_round"]
    evals = [e for e in ev if e["type"] == "eval_round"]
    assert len(fits) == 3 and len(evals) == 3
    assert all(f["payload_type"] == "tenseal.ckks.v1" and not f["server_saw_plaintext_weights"] for f in fits)
    assert all(len(f["clients"]) == 3 for f in fits)
    assert (tmp_path / "runs/t/models/hospital_1.pt").exists()


@pytest.mark.slow
def test_training_survives_a_node_dropping_out(tmp_path, ports):
    r = run_federation(tmp_path, ports, extra=("--chaos", "3:2", "--rejoin-delay", "-1"))
    assert r.returncode == 0, r.stdout + r.stderr
    fits = [e for e in events(tmp_path) if e["type"] == "fit_round"]
    by_round = {f["round"]: f for f in fits}
    assert by_round[2]["dropped"] == [3] and len(by_round[2]["clients"]) == 2


@pytest.mark.slow
def test_plaintext_mode_still_works_and_is_reported_honestly(tmp_path, ports):
    r = run_federation(tmp_path, ports, extra=("--set", "he.enabled=false"))
    assert r.returncode == 0, r.stdout + r.stderr
    fits = [e for e in events(tmp_path) if e["type"] == "fit_round"]
    assert len(fits) == 3
    assert all(f["payload_type"] == "plaintext" and f["server_saw_plaintext_weights"] for f in fits)
