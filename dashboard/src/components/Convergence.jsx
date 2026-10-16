import React from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { roundSeries } from "../lib/runState.js";
import { hospitalMeta } from "../lib/palette.js";
import { fmt } from "../lib/format.js";

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tip">
      <div className="tip__title">Round {label}</div>
      {payload.filter((p) => p.value !== null && p.value !== undefined).map((p) => (
        <div key={p.dataKey} className="tip__row">
          <span className="swatch" style={{ background: p.color }} />
          <span>{p.name}</span>
          <b>{fmt(p.value)}</b>
        </div>
      ))}
    </div>
  );
}

const axis = { stroke: "var(--ink-3)", fontSize: 12, tickLine: false };

export default function Convergence({ run, baseline }) {
  const data = roundSeries(run);
  const hospitals = Array.from({ length: run.start?.hospitals ?? 3 }, (_, i) => i + 1);
  const base = baseline?.final_dice;
  const total = run.start?.rounds ?? 1;
  const ticks = Array.from({ length: total }, (_, i) => i + 1);

  return (
    <>
      <div className="cell__head">
        <h2 id="h-conv">Convergence</h2>
        <p>Validation Dice after each round, measured on each hospital’s own held-out scans.</p>
      </div>
      <div className="chart" role="img" aria-label="Line chart of Dice score by training round for the global model and each hospital">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 12, right: 24, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--rule)" vertical={false} />
            <XAxis dataKey="round" ticks={ticks} {...axis} axisLine={{ stroke: "var(--rule-strong)" }} />
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axis} axisLine={false} width={36} />
            <Tooltip content={<ChartTip />} cursor={{ stroke: "var(--rule-strong)" }} />
            {base ? (
              <ReferenceLine y={base} stroke="var(--ink-3)" strokeDasharray="5 4"
                label={{ value: `centralized ${fmt(base, 2)}`, position: "insideBottomRight", fill: "var(--ink-3)", fontSize: 12 }} />
            ) : null}
            {hospitals.map((h) => (
              <Line key={h} type="monotone" dataKey={`h${h}`} name={hospitalMeta(h).name}
                    stroke={hospitalMeta(h).color} strokeWidth={1.6} dot={{ r: 2.5 }} connectNulls={false} isAnimationActive={false} />
            ))}
            <Line type="monotone" dataKey="global" name="Global model" stroke="var(--ink)" strokeWidth={3}
                  dot={{ r: 3.5, fill: "var(--ink)" }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <ul className="legend">
        <li><span className="swatch swatch--line" style={{ background: "var(--ink)" }} />Global model</li>
        {hospitals.map((h) => (
          <li key={h}><span className="swatch swatch--line" style={{ background: hospitalMeta(h).color }} />{hospitalMeta(h).name}</li>
        ))}
        {base ? <li><span className="swatch swatch--dash" />Centralized baseline</li> : null}
      </ul>
      <div className="mini">
        <h3>Loss</h3>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={data} margin={{ top: 6, right: 24, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--rule)" vertical={false} />
            <XAxis dataKey="round" ticks={ticks} {...axis} axisLine={{ stroke: "var(--rule-strong)" }} />
            <YAxis {...axis} axisLine={false} width={36} tickFormatter={(v) => v.toFixed(1)} />
            <Tooltip content={<ChartTip />} cursor={{ stroke: "var(--rule-strong)" }} />
            <Line type="monotone" dataKey="trainLoss" name="Local training loss" stroke="var(--ink-3)" strokeWidth={1.6}
                  strokeDasharray="4 3" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="loss" name="Global validation loss" stroke="var(--ink)" strokeWidth={2}
                  dot={{ r: 2.5 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
