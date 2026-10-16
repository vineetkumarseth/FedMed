import React from "react";

export default function Audit({ audit }) {
  return (
    <>
      <div className="cell__head">
        <h2 id="h-audit">Federated audit</h2>
        <p>Checks run against the finished run, so the privacy claims are verified rather than asserted.</p>
      </div>
      {!audit ? (
        <p className="empty">Run <code>fedmed audit</code> after training to fill this in.</p>
      ) : (
        <>
          <p className={`audit__verdict ${audit.passed ? "is-pass" : "is-fail"}`}>
            {audit.passed ? "All required checks passed" : "A required check failed"}
          </p>
          <ul className="checks">
            {audit.checks.map((c) => (
              <li key={c.name} className="check">
                <span className={`check__mark ${c.passed ? "is-pass" : c.informational ? "is-info" : "is-fail"}`}
                      aria-label={c.passed ? "passed" : c.informational ? "informational" : "failed"}>
                  {c.passed ? "✓" : c.informational ? "i" : "✕"}
                </span>
                <div><b>{c.name}</b><span>{c.detail}</span></div>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}
