import React from "react";
import { epsilonVerdict, fmtEpsilon } from "../lib/format.js";

export default function Privacy({ run, fit, ev }) {
  const s = run.start;
  if (!s) {
    return (
      <>
        <div className="cell__head"><h2 id="h-priv">Privacy ledger</h2><p>Appears once a run starts.</p></div>
      </>
    );
  }
  const eps = ev?.epsilon ?? fit?.epsilon ?? null;
  const verdict = s.dp.enabled ? epsilonVerdict(eps) : null;
  const clippedShare = fit ? fit.clients.filter((c) => c.clipped).length : null;
  const nts = fit?.noise_to_signal;

  const rows = [
    ["Transport", s.tls ? "gRPC over TLS: server identity is verified with the consortium CA" : "No TLS (development only)", s.tls],
    ["Aggregation", s.he.enabled
      ? `CKKS homomorphic encryption (N=${s.he.poly_modulus_degree}). Each update is ${s.he.ciphertexts_per_update} ciphertexts; the server holds the public key only and cannot decrypt.`
      : "Plaintext averaging: the server can read every hospital’s weights", s.he.enabled],
    ["Differential privacy", s.dp.enabled
      ? `Updates clipped to ‖Δ‖ ≤ ${s.dp.clip_norm}, Gaussian noise σ = ${s.dp.noise_multiplier}×clip`
      : "Off", s.dp.enabled],
  ];

  return (
    <>
      <div className="cell__head">
        <h2 id="h-priv">Privacy ledger</h2>
        <p>What protects the weights while they leave a hospital.</p>
      </div>
      <dl className="ledger">
        {rows.map(([k, v, on]) => (
          <div key={k} className="ledger__row">
            <dt><span className={`state ${on ? "state--on" : "state--off"}`}>{on ? "On" : "Off"}</span>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {s.dp.enabled && (
        <div className="eps">
          <div className="eps__value">ε = {fmtEpsilon(eps)}</div>
          <div className="eps__note">
            after {ev?.round ?? 0} rounds at δ = {s.dp.delta}.
            {verdict && <span className={`verdict verdict--${verdict.tone}`}> {verdict.label}.</span>}
            {nts !== null && nts !== undefined && <> Injected noise is about {Math.round(nts * 100)}% of the size of each update.</>}
            {clippedShare !== null && <> {clippedShare} of {fit.clients.length} updates hit the clipping bound last round.</>}
          </div>
        </div>
      )}
    </>
  );
}
