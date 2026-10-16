"""``fedmed`` command line.

    fedmed gen-certs | gen-keys        one-time key material (TLS + CKKS)
    fedmed baseline                    centralized reference model
    fedmed server | client --id N      run one side of the federation by hand
    fedmed federate                    launch server + 3 hospital nodes + dashboard API, then audit
    fedmed export | audit | api        post-run tooling
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import Config, load_config


def _parse_overrides(pairs: Optional[List[str]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _cfg(args) -> Config:
    return load_config(args.config, _parse_overrides(args.set))


def _resolve_run_dir(cfg: Config, run: Optional[str], create: bool = False) -> Path:
    runs = Path(cfg.runs_dir)
    if run in (None, "latest"):
        marker = runs / "LATEST"
        if marker.exists():
            return runs / marker.read_text().strip()
        if not create:
            raise SystemExit("No runs yet. Start one with `fedmed federate`.")
        run = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs / run
    if create:
        path.mkdir(parents=True, exist_ok=True)
        (runs / "LATEST").write_text(run)
    return path


def ensure_key_material(cfg: Config) -> None:
    if cfg.security.tls and not (Path(cfg.security.cert_dir) / "server.crt").exists():
        from .privacy.tls import generate_dev_certs
        generate_dev_certs(cfg.security.cert_dir)
        print(f"[setup] generated dev TLS certificates in {cfg.security.cert_dir}/")
    if cfg.he.enabled and not (Path(cfg.security.key_dir) / "he_secret.ctx").exists():
        from .privacy.he import generate_keys
        generate_keys(cfg.he, cfg.security.key_dir)
        print(f"[setup] generated CKKS keys in {cfg.security.key_dir}/")


# --- subcommands ---------------------------------------------------------------------------
def cmd_gen_certs(args):
    from .privacy.tls import generate_dev_certs
    cfg = _cfg(args)
    print("wrote", generate_dev_certs(cfg.security.cert_dir))


def cmd_gen_keys(args):
    from .privacy.he import generate_keys
    cfg = _cfg(args)
    s, p = generate_keys(cfg.he, cfg.security.key_dir)
    print(f"secret (hospitals only): {s}\npublic (server):         {p}")


def cmd_baseline(args):
    from .baseline import run_baseline
    cfg = _cfg(args)
    result = run_baseline(cfg, _resolve_run_dir(cfg, args.run, create=True), args.epochs)
    print(f"centralized baseline: final Dice {result['final_dice']:.4f}")


def cmd_server(args):
    from .federated.server import run_server
    cfg = _cfg(args)
    ensure_key_material(cfg)
    run_server(cfg, _resolve_run_dir(cfg, args.run, create=True))


def cmd_client(args):
    from .federated.client import run_client
    cfg = _cfg(args)
    run_client(cfg, args.id, _resolve_run_dir(cfg, args.run), args.crash_at_round)


def cmd_export(args):
    from .export import export_segmentations
    cfg = _cfg(args)
    print("wrote", export_segmentations(cfg, _resolve_run_dir(cfg, args.run)))


def cmd_audit(args):
    from .audit import run_audit
    cfg = _cfg(args)
    result = run_audit(cfg, _resolve_run_dir(cfg, args.run))
    print("\nFederated audit")
    for c in result["checks"]:
        mark = "PASS" if c["passed"] else ("INFO" if c.get("informational") else "FAIL")
        print(f"  [{mark}] {c['name']}\n         {c['detail']}")
    print("\nResult:", "PASSED" if result["passed"] else "FAILED")
    sys.exit(0 if result["passed"] else 1)


def cmd_api(args):
    import uvicorn
    from .api import create_app
    cfg = _cfg(args)
    uvicorn.run(create_app(cfg), host=cfg.dashboard.host, port=cfg.dashboard.port, log_level="warning")


def _wait_port(host: str, port: int, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


def cmd_federate(args):
    cfg = _cfg(args)
    ensure_key_material(cfg)
    run_dir = _resolve_run_dir(cfg, args.name, create=True)
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    base = [sys.executable, "-m", "fedmed.cli"]
    common = ["--config", args.config] if args.config else []
    for k, v in _parse_overrides(args.set).items():
        common += ["--set", f"{k}={json.dumps(v)}"]
    common += ["--run", run_dir.name]
    procs: Dict[str, subprocess.Popen] = {}

    def spawn(name: str, cmd: List[str]) -> subprocess.Popen:
        fh = open(logs / f"{name}.log", "a", buffering=1)
        p = subprocess.Popen(base + cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        procs[name] = p
        return p

    def shutdown(*_):
        for p in procs.values():
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, lambda *a: (shutdown(), sys.exit(130)))
    try:
        if not args.skip_baseline:
            print(f"[federate] run {run_dir.name}: training centralized baseline first (reference only)...")
            # separate process so its memory (PyTorch) is released before the federation starts
            rc = subprocess.call(base + ["baseline", *common])
            if rc != 0:
                raise SystemExit("centralized baseline failed")
        if not args.no_api:
            spawn("api", ["api", *common])
            print(f"[federate] dashboard API on http://{cfg.dashboard.host}:{cfg.dashboard.port}")

        spawn("server", ["server", *common])
        if not _wait_port(cfg.federation.host, cfg.federation.port):
            raise SystemExit("server did not come up - see " + str(logs / "server.log"))

        crash_node, crash_round = None, None
        if args.chaos:
            crash_node, crash_round = (int(x) for x in args.chaos.split(":"))
            print(f"[federate] chaos: hospital {crash_node} will crash in round {crash_round}"
                  + (f" and rejoin after {args.rejoin_delay:.0f}s" if args.rejoin_delay >= 0 else " and stay down"))

        def spawn_client(hid: int, crash: bool):
            cmd = ["client", "--id", str(hid), *common]
            if crash:
                cmd += ["--crash-at-round", str(crash_round)]
            spawn(f"hospital_{hid}", cmd)

        for hid in range(1, cfg.federation.num_hospitals + 1):
            spawn_client(hid, crash=(hid == crash_node))
        restart_at: Dict[int, float] = {}

        server = procs["server"]
        from .federated.constants import CHAOS_EXIT_CODE
        while server.poll() is None:
            time.sleep(0.5)
            for hid in range(1, cfg.federation.num_hospitals + 1):
                p = procs[f"hospital_{hid}"]
                if p.poll() == CHAOS_EXIT_CODE and hid not in restart_at and args.rejoin_delay >= 0:
                    restart_at[hid] = time.time() + args.rejoin_delay
                    print(f"[federate] hospital {hid} crashed (simulated); restarting in {args.rejoin_delay:.0f}s")
                if hid in restart_at and time.time() >= restart_at[hid]:
                    del restart_at[hid]
                    spawn_client(hid, crash=False)
        code = server.returncode
        if code not in (0, None):
            raise SystemExit(f"server exited with code {code} - see {logs / 'server.log'}")

        from .audit import run_audit
        from .export import export_segmentations
        if (run_dir / "models" / "hospital_1.pt").exists():
            export_segmentations(cfg, run_dir)
        audit = run_audit(cfg, run_dir)
        print(f"[federate] done. audit: {'PASSED' if audit['passed'] else 'FAILED'} (details: {run_dir / 'audit.json'})")
        if args.keep_api and not args.no_api:
            print("[federate] API left running - Ctrl+C to stop")
            procs["api"].wait()
    finally:
        shutdown()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="fedmed", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    common.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. --set federation.rounds=4")
    common.add_argument("--run", default=None, help="run id under runs/ (default: latest)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn, help_ in [("gen-certs", cmd_gen_certs, "create dev TLS CA + server cert"),
                            ("gen-keys", cmd_gen_keys, "create the consortium CKKS keys"),
                            ("server", cmd_server, "run the aggregation server"),
                            ("export", cmd_export, "export segmentation overlays for the dashboard"),
                            ("audit", cmd_audit, "verify the privacy and convergence claims of a run"),
                            ("api", cmd_api, "serve the dashboard API")]:
        sub.add_parser(name, parents=[common], help=help_).set_defaults(fn=fn)

    p = sub.add_parser("baseline", parents=[common], help="train the centralized reference model")
    p.add_argument("--epochs", type=int, default=None)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("client", parents=[common], help="run one hospital node")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--crash-at-round", type=int, default=None, help="chaos testing: exit abruptly in this round")
    p.set_defaults(fn=cmd_client)

    p = sub.add_parser("federate", parents=[common], help="launch the whole federation locally")
    p.add_argument("--name", default=None, help="run id (default: timestamp)")
    p.add_argument("--chaos", default=None, metavar="NODE:ROUND", help="crash hospital NODE in round ROUND")
    p.add_argument("--rejoin-delay", type=float, default=8.0, help="seconds before a crashed node restarts (-1 = never)")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--keep-api", action="store_true", help="keep the dashboard API running after training")
    p.set_defaults(fn=cmd_federate)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
