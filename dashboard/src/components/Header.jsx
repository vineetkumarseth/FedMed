import React from "react";
import { fmt } from "../lib/format.js";

export default function Header({ run, mode, baseline, ev }) {
  const total = run.start?.rounds ?? 0;
  const done = ev?.round ?? 0;
  const finished = Boolean(run.end);
  const base = baseline?.final_dice;
  const ratio = ev && base ? ev.dice / base : null;

  let headline;
  if (!run.start) headline = "Waiting for a federation to start.";
  else if (!ev) headline = "Round 1 is training on-site at each hospital.";
  else
    headline = (
      <>
        The shared model segments tumors with Dice <strong>{fmt(ev.dice, 2)}</strong>
        {ratio ? <> — <strong>{Math.round(ratio * 100)}%</strong> of what pooling all the scans would give</> : null}
        {finished ? <>, after {total} rounds.</> : <>, after round {done} of {total}.</>}
      </>
    );

  const modeLabel = { live: "Live", sample: "Sample run", connecting: "Connecting" }[mode];

  return (
    <header className="head">
      <div className="head__top">
        <div className="brand">
          <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="7" fill="var(--film)" />
            <circle cx="16" cy="16" r="9" fill="none" stroke="var(--paper)" strokeWidth="2" />
            <circle cx="19" cy="13" r="4" fill="var(--lesion)" />
          </svg>
          <span className="brand__name">FedMed</span>
        </div>
        <div className="head__meta">
          <span className="run-id">{run.runId ?? "no run"}</span>
          <span className={`badge badge--${mode}`} role="status">
            <span className="badge__dot" aria-hidden="true" />
            {modeLabel}
          </span>
        </div>
      </div>
      <h1 className="headline">{headline}</h1>
      <div className="rail" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={done}
           aria-label="Training rounds completed">
        {Array.from({ length: total }, (_, i) => (
          <span key={i} className={`rail__seg ${i < done ? "is-done" : i === done && !finished ? "is-active" : ""}`} />
        ))}
      </div>
      {mode === "sample" && (
        <p className="note">
          Showing a bundled sample run. Start <code>fedmed federate --keep-api</code> and this page switches to live data.
        </p>
      )}
    </header>
  );
}
