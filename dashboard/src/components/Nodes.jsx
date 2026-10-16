import React from "react";
import { participation } from "../lib/runState.js";
import { hospitalMeta } from "../lib/palette.js";
import { fmt } from "../lib/format.js";

export default function Nodes({ run, ev, fit, liveNodes }) {
  const hospitals = Array.from({ length: run.start?.hospitals ?? 3 }, (_, i) => i + 1);
  const part = participation(run);

  return (
    <>
      <div className="cell__head">
        <h2 id="h-nodes">Hospital nodes</h2>
        <p>Each node trains on its own scans. Squares show which rounds it contributed to.</p>
      </div>
      <ul className="nodes">
        {hospitals.map((h) => {
          const meta = hospitalMeta(h);
          const c = fit?.clients.find((x) => x.hospital === h);
          const e = ev?.clients.find((x) => x.hospital === h);
          const live = liveNodes?.find((n) => n.hospital === h);
          const lastDropped = fit && !c;
          const state = live
            ? live.reachable ? live.state : "unreachable"
            : lastDropped ? "dropped last round" : c ? "contributed" : "waiting";
          const bad = state === "unreachable" || state === "dropped last round";
          return (
            <li key={h} className="node">
              <div className="node__top">
                <span className="node__dot" style={{ background: meta.color }} aria-hidden="true" />
                <div className="node__id">
                  <b>{meta.name}</b>
                  <span>{meta.scanner}{live?.reachable ? ` · :${live.port}` : ""}</span>
                </div>
                <span className={`pill ${bad ? "pill--bad" : "pill--ok"}`}>{bad ? "✕ " : "✓ "}{state}</span>
              </div>
              <div className="strip" role="img" aria-label={`Participation of ${meta.name} per round`}>
                {part[h].map((s, i) => (
                  <span key={i} className={`strip__cell is-${s}`} title={`Round ${i + 1}: ${s === "ok" ? "contributed" : s}`}
                        style={s === "ok" ? { background: meta.color, borderColor: meta.color } : undefined} />
                ))}
              </div>
              <dl className="node__stats">
                <div><dt>Scans</dt><dd>{c?.n ?? "–"}</dd></div>
                <div><dt>Local Dice</dt><dd>{fmt(e?.dice, 2)}</dd></div>
                <div><dt>Sent / round</dt><dd>{c ? `${c.payload_mb.toFixed(1)} MB` : "–"}</dd></div>
                <div><dt>Encrypt</dt><dd>{c ? `${(c.encrypt_ms / 1000).toFixed(1)} s` : "–"}</dd></div>
              </dl>
            </li>
          );
        })}
      </ul>
    </>
  );
}
