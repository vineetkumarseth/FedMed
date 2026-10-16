# Architecture

```
 Hospital 1            Hospital 2            Hospital 3                 Aggregation server
┌─────────────┐       ┌─────────────┐       ┌─────────────┐            ┌───────────────────────┐
│ MRI + masks │       │ MRI + masks │       │ MRI + masks │            │  no data, no model    │
│ (never      │       │ (never      │       │ (never      │            │  weights in plaintext │
│  leave)     │       │  leave)     │       │  leave)     │            │                       │
│      │      │       │      │      │       │      │      │            │  holds PUBLIC HE key  │
│ local U-Net │       │ local U-Net │       │ local U-Net │            │  adds ciphertexts     │
│ train       │       │ train       │       │ train       │            │                       │
│      │      │       │      │      │       │      │      │            └───────────▲───────────┘
│ clip + noise│       │ clip + noise│       │ clip + noise│                        │
│ (DP)        │       │ (DP)        │       │ (DP)        │                        │
│      │      │       │      │      │       │      │      │      gRPC over TLS     │
│ n·w → CKKS  │──────▶│ n·w → CKKS  │──────▶│ n·w → CKKS  │────────────────────────┘
│ encrypt     │       │ encrypt     │       │ encrypt     │      E(n₁w₁), E(n₂w₂), E(n₃w₃)
└──────▲──────┘       └──────▲──────┘       └──────▲──────┘
       │   E(Σ nᵢwᵢ) and N = Σ nᵢ, broadcast back by the server
       └── decrypt with the SECRET key (hospitals only), divide by N, continue training
```

## One round, step by step

1. **Broadcast.** The server sends the current global model. In round 1 this is the random
   initialisation (plaintext, carries no information). From then on it is a list of CKKS
   ciphertexts, `E(Σ nᵢ·wᵢ)`, plus the integer `N = Σ nᵢ` in the round config.
2. **Decrypt.** Each hospital decrypts with the consortium *secret* key and divides by `N`.
   That is exactly the FedAvg model.
3. **Train locally** for `local_epochs` on private scans (optionally with a FedProx proximal term).
4. **Protect the update.** `Δ = w_local − w_global` is clipped to `‖Δ‖₂ ≤ C`, Gaussian noise
   `N(0, (σC)²)` is added, and the hospital forms `w_sent = w_global + Δ̃`.
5. **Encrypt.** The hospital encrypts `nᵢ · w_sent` into ⌈d / 4096⌉ CKKS ciphertexts.
6. **Aggregate.** The server **adds ciphertexts** (no multiplication, so no relinearisation or
   Galois keys, no noise-budget problems) and stores the result as the next global model.
7. **Evaluate.** The same ciphertext goes back; each hospital decrypts it and reports Dice, IoU
   and loss on its *own* validation scans. These scalar metrics are all the server learns.

Why encrypt `nᵢ·w` instead of `Δ`? A hospital that was offline last round can still join,
because the broadcast always carries the *complete* model, not a delta against a history it missed.

## Components

| Path | Role |
|---|---|
| `fedmed/models/unet3d.py` | 3D U-Net with InstanceNorm (no running statistics to aggregate) |
| `fedmed/training/` | Dice+BCE loss, metrics, local train/eval loops, flat-parameter helpers |
| `fedmed/privacy/he.py` | CKKS context/key management, chunked encrypt/decrypt, ciphertext addition |
| `fedmed/privacy/dp.py` | Update clipping, Gaussian noise, Rényi-DP accountant |
| `fedmed/privacy/tls.py` | Development CA and server certificate for gRPC/TLS |
| `fedmed/federated/client.py` | Hospital node (Flower `Client`) + per-node status port |
| `fedmed/federated/strategy.py` | Aggregation strategy: ciphertext sum, dropout handling, telemetry |
| `fedmed/federated/server.py` | Server entry point; refuses to start with a secret-key context |
| `fedmed/federated/events.py` | Append-only JSONL event log (single source of truth for a run) |
| `fedmed/api.py` | FastAPI: tails the event log over WebSocket, serves overlays and audit |
| `fedmed/audit.py` | Verifies the privacy and convergence claims against a finished run |
| `fedmed/baseline.py` | Centralized reference model (violates privacy on purpose, for comparison only) |
| `dashboard/` | React + Recharts monitor with an MRI overlay viewer |

## Node resilience

Flower returns failed clients separately from successful ones. `FedMedStrategy.aggregate_fit`
sums whichever ciphertexts arrived, records the missing hospitals, and passes the *actual*
`N = Σ nᵢ` of the survivors to the next round so decryption divides by the right number. A round
is only skipped if fewer than `federation.min_clients` nodes respond. A restarted hospital
reconnects as a new Flower client and receives the full current model, so no catch-up protocol
is needed. `fedmed federate --chaos 3:4` crashes hospital 3 mid-training in round 4 to prove it.

## Where the three "hospital ports" fit

Flower clients dial *out* to the server, so training traffic uses the single server port (8080).
Each hospital additionally serves a small HTTP `/status` endpoint on its own local port
(8101, 8102, 8103). The dashboard API polls those to show which nodes are reachable.
