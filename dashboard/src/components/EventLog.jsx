import React, { useEffect, useRef } from "react";

export default function EventLog({ run }) {
  const box = useRef(null);
  useEffect(() => { if (box.current) box.current.scrollTop = box.current.scrollHeight; }, [run.logs.length]);
  return (
    <>
      <div className="cell__head">
        <h2 id="h-log">Event log</h2>
        <p>Server-side notices. Warnings mark nodes that dropped out mid-round.</p>
      </div>
      <ol className="log" ref={box} tabIndex={0} aria-label="Event log">
        {run.logs.length === 0 && <li className="log__empty">No events yet.</li>}
        {run.logs.map((l, i) => (
          <li key={i} className={`log__row log__row--${l.level}`}>
            <time>+{Math.max(0, Math.round(l.t - (run.firstT ?? l.t)))}s</time>
            <span>{l.level === "warn" ? "⚠ " : ""}{l.message}</span>
          </li>
        ))}
      </ol>
    </>
  );
}
