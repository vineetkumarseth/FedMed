"""Federated audit (mid-project review).

Checks the claims the project makes instead of just asserting them:

1. **Convergence** - federated Dice reaches ``audit.min_baseline_ratio`` of the centralized baseline.
2. **No raw-data path on the server** - the server modules never import ``fedmed.data``.
3. **Server cannot decrypt** - the key material the server loads is the public context only.
4. **Ciphertext on the wire** - every aggregation round shipped ciphertext and the server never held plaintext weights.
5. **Resilience** - rounds with a missing node still completed (reported, not required).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List

from .config import Config
from .federated.events import read_events

SERVER_SIDE_FILES = ("federated/server.py", "federated/strategy.py", "federated/events.py")


def _imports(path: Path) -> List[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            names.append(mod)
            names += [f"{mod}.{a.name}" for a in node.names]
    return names


FORBIDDEN_ON_SERVER = ("data", "models", "training", "torch")


def static_server_check() -> Dict[str, Any]:
    pkg = Path(__file__).parent
    offenders = {}
    for rel in SERVER_SIDE_FILES:
        bad = []
        for name in _imports(pkg / rel):
            head = name.replace("fedmed.", "..").lstrip(".").split(".")[0]
            if head in FORBIDDEN_ON_SERVER:
                bad.append(name)
        if bad:
            offenders[rel] = bad
    return {"name": "Server has no path to patient data or model code", "passed": not offenders,
            "detail": "no server-side module imports fedmed.data, fedmed.models, fedmed.training or torch"
            if not offenders else f"offending imports: {offenders}"}


def run_audit(cfg: Config, run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    events = list(read_events(run_dir / "metrics.jsonl"))
    checks: List[Dict[str, Any]] = [static_server_check()]

    # 1. convergence vs centralized baseline
    base_path = run_dir / "baseline.json"
    evals = [e for e in events if e["type"] == "eval_round"]
    if base_path.exists() and evals:
        base = json.loads(base_path.read_text())["final_dice"]
        fed = evals[-1]["dice"]
        ratio = fed / base if base > 0 else 0.0
        checks.append({"name": "Federated model approaches the centralized baseline",
                       "passed": ratio >= cfg.audit.min_baseline_ratio,
                       "detail": f"federated Dice {fed:.3f} vs centralized {base:.3f} "
                                 f"({ratio * 100:.1f}% of baseline, need {cfg.audit.min_baseline_ratio * 100:.0f}%)"})
    else:
        checks.append({"name": "Federated model approaches the centralized baseline", "passed": False,
                       "detail": "missing baseline.json or evaluation events - run `fedmed baseline` and `fedmed federate`"})

    fits = [e for e in events if e["type"] == "fit_round"]
    if cfg.he.enabled:
        # 3. server key material
        pub = Path(cfg.security.key_dir) / "he_public.ctx"
        if pub.exists():
            from .privacy.he import load_context
            can_decrypt = load_context(pub).has_secret_key()
            checks.append({"name": "Server holds a public-only HE context", "passed": not can_decrypt,
                           "detail": "he_public.ctx contains no secret key" if not can_decrypt else "he_public.ctx CAN decrypt!"})
        # 4. ciphertext on the wire
        ok = bool(fits) and all(f["payload_type"] == "tenseal.ckks.v1" and not f["server_saw_plaintext_weights"] for f in fits)
        checks.append({"name": "Every aggregation round was ciphertext-only", "passed": ok,
                       "detail": f"{len(fits)} rounds, payload_type=tenseal.ckks.v1, server never held plaintext weights"
                       if ok else "found a round with plaintext weights on the server"})
    else:
        checks.append({"name": "Every aggregation round was ciphertext-only", "passed": False,
                       "detail": "HE is disabled in this run (he.enabled=false); the server saw plaintext weights"})

    # 5. resilience (informational)
    dropped_rounds = [f["round"] for f in fits if f["dropped"]]
    completed = len(fits)
    checks.append({"name": "Training survived node dropouts", "passed": completed >= cfg.federation.rounds - len(
        [e for e in events if e["type"] == "round_skipped"]), "informational": True,
        "detail": (f"rounds with a missing node: {dropped_rounds}; all {completed} rounds still aggregated"
                   if dropped_rounds else "no dropouts occurred in this run")})

    result = {"passed": all(c["passed"] for c in checks if not c.get("informational")), "checks": checks}
    (run_dir / "audit.json").write_text(json.dumps(result, indent=2))
    return result
