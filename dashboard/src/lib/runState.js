// Pure reducer: turns the stream of server events (see fedmed/federated/events.py) into view state.
export const initialRun = () => ({
  runId: null,
  start: null,
  fits: {},
  evals: {},
  skipped: [],
  logs: [],
  end: null,
  firstT: null,
});

export function reduceEvent(state, e) {
  if (e.type === "reset") return { ...initialRun(), runId: e.run_id };
  const s = { ...state, firstT: state.firstT ?? e.t };
  switch (e.type) {
    case "run_start":
      return { ...s, start: e, runId: e.run_id };
    case "fit_round":
      return { ...s, fits: { ...s.fits, [e.round]: e } };
    case "eval_round":
      return { ...s, evals: { ...s.evals, [e.round]: e } };
    case "round_skipped":
      return { ...s, skipped: [...s.skipped, e] };
    case "run_end":
      return { ...s, end: e };
    case "log":
      return { ...s, logs: [...s.logs.slice(-199), e] };
    default:
      return s;
  }
}

// One row per round, shaped for the charts.
export function roundSeries(run) {
  const total = run.start?.rounds ?? Math.max(0, ...Object.keys(run.evals).map(Number));
  const hospitals = run.start?.hospitals ?? 3;
  const rows = [];
  for (let r = 1; r <= total; r++) {
    const ev = run.evals[r];
    const fit = run.fits[r];
    const row = { round: r, global: ev?.dice ?? null, loss: ev?.loss ?? null };
    if (fit) {
      row.trainLoss = fit.clients.reduce((a, c) => a + c.train_loss * c.n, 0) / fit.total_examples;
    }
    for (let h = 1; h <= hospitals; h++) {
      row[`h${h}`] = ev?.clients.find((c) => c.hospital === h)?.dice ?? null;
    }
    rows.push(row);
  }
  return rows;
}

// Which hospitals contributed to each round (the participation strip).
export function participation(run) {
  const total = run.start?.rounds ?? 0;
  const hospitals = run.start?.hospitals ?? 3;
  const out = {};
  for (let h = 1; h <= hospitals; h++) {
    out[h] = [];
    for (let r = 1; r <= total; r++) {
      const fit = run.fits[r];
      if (!fit) out[h].push("pending");
      else out[h].push(fit.clients.some((c) => c.hospital === h) ? "ok" : "dropped");
    }
  }
  return out;
}

export function latestEval(run) {
  const rounds = Object.keys(run.evals).map(Number);
  return rounds.length ? run.evals[Math.max(...rounds)] : null;
}

export function latestFit(run) {
  const rounds = Object.keys(run.fits).map(Number);
  return rounds.length ? run.fits[Math.max(...rounds)] : null;
}
