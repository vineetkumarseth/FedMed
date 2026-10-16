export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

export const fmtEpsilon = (eps) => {
  if (eps === null || eps === undefined) return "not tracked";
  if (eps === 0) return "0";
  if (eps >= 1e4) {
    const [m, e] = eps.toExponential(1).split("e+");
    const sup = [..."" + Number(e)].map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]).join("");
    return `${m}×10${sup}`;
  }
  return eps.toFixed(eps < 10 ? 2 : 0);
};

export const epsilonVerdict = (eps) => {
  if (eps === null || eps === undefined) return null;
  if (eps <= 10) return { label: "meaningful", tone: "ok" };
  if (eps <= 100) return { label: "weak", tone: "warn" };
  return { label: "demo-grade, not a formal guarantee", tone: "warn" };
};

export const fmtDuration = (s) => {
  if (s === null || s === undefined) return "–";
  if (s < 90) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
};
