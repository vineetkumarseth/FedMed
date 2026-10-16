# Privacy model and honest limitations

FedMed layers four protections. Each one answers a different question, and each has limits you
should understand before using this on real patient data.

## 1. Data never leaves the hospital (federated learning)

Raw scans, masks and per-scan predictions stay on the node. `fedmed audit` statically checks that
the server modules do not import `fedmed.data`. Federated learning alone is **not** a privacy
guarantee: model updates can leak training data (gradient inversion, membership inference),
which is why layers 2-4 exist.

## 2. The server cannot read the weights (CKKS homomorphic encryption)

* The server loads `keys/he_public.ctx` and refuses to start if that context contains a secret key.
* It only adds ciphertexts. It never sees an individual hospital's weights, nor the aggregate.
* `fedmed audit` confirms every round used `tenseal.ckks.v1` payloads and that the server never
  held plaintext weights.

Limits:

* **Single shared key.** All hospitals hold the same secret key (`keys/he_secret.ctx`). Anyone who
  steals it from one hospital can decrypt every update. Production systems should use threshold or
  multi-party HE so that no single party can decrypt.
* **CKKS is approximate.** Decrypted values carry ~1e-7 relative error. Tests bound it
  (`tests/test_he.py`); it is far below the noise of SGD.
* The hospitals themselves see the decrypted global model (that is the point: they train on it).
  A malicious *hospital* is outside this threat model.
* Sample counts `nᵢ` and metrics (Dice, loss) are sent in the clear.

## 3. Transport security (TLS)

gRPC runs over TLS with a private CA (`fedmed gen-certs`). Clients verify the server certificate.
It is **server-authenticated TLS, not mutual TLS**; Flower's classic API does not authenticate
clients. Use your organisation's PKI in production and add client authentication.

## 4. Differential privacy

Each hospital clips its update to norm `C` and adds `N(0, (σC)²)` noise **with fresh OS entropy**
(not a seeded generator, which would make the noise reproducible and defeat the purpose).
`fedmed/privacy/dp.py` includes a Rényi-DP accountant for the Gaussian mechanism.

What the guarantee means: **silo-level, per-update** DP. It bounds how much any one hospital's
transmitted update reveals about whether a given hospital's data influenced it.

Limits, which matter a lot here:

* With only **3 silos** and ~10⁵ parameters, noise strong enough for a *meaningful* ε (≲ 10)
  destroys accuracy. The default `noise_multiplier: 0.002` is chosen so training still works, and
  yields an ε in the 10⁵-10⁶ range. **That is not a formal privacy guarantee**; the dashboard labels
  it "demo-grade" for this reason.
* ε is computed with zero subsampling amplification and no composition tricks beyond RDP.
* Run `scripts/privacy_sweep.py` to see the actual utility-versus-ε curve for your data.
* DP does not defend against a malicious hospital or a compromised secret key.

## What "model inversion protection" can honestly be claimed

Layered: the server cannot see updates (HE), outsiders cannot read the wire (TLS), and the
transmitted update is clipped and noised (DP). Treat the DP part as *risk reduction* at the
default settings, and as a *guarantee* only after you have chosen σ, C and the number of rounds
to reach an ε you can defend, and accepted the accuracy cost.

## Not in scope

Secure aggregation by pairwise masking, malicious-client robustness (Byzantine-resilient
aggregation), client authentication, audit logging of key access, and regulatory review
(HIPAA/GDPR). This is an engineering reference implementation, not a certified medical system.
