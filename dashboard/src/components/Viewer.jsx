import React, { useEffect, useMemo, useRef, useState } from "react";
import { fmt } from "../lib/format.js";

const SCALE = 9; // CSS pixels per voxel in each pane

function decode(b64, n) {
  const bin = atob(b64);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Draw one axial slice; `mask` (optional) is filled lightly and outlined along its boundary.
function drawPane(canvas, size, image, z, mask, color, errors) {
  const px = size * SCALE;
  canvas.width = px;
  canvas.height = px;
  const ctx = canvas.getContext("2d");
  const base = document.createElement("canvas");
  base.width = size; base.height = size;
  const bctx = base.getContext("2d");
  const img = bctx.createImageData(size, size);
  const off = z * size * size;
  for (let i = 0; i < size * size; i++) {
    const v = image[off + i];
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  bctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(base, 0, 0, px, px);
  if (!mask) return;

  const at = (m, y, x) => (y >= 0 && y < size && x >= 0 && x < size ? m[off + y * size + x] : 0);
  ctx.fillStyle = color + "55";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!at(mask, y, x)) continue;
      ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
      if (!at(mask, y - 1, x)) { ctx.moveTo(x * SCALE, y * SCALE); ctx.lineTo((x + 1) * SCALE, y * SCALE); }
      if (!at(mask, y + 1, x)) { ctx.moveTo(x * SCALE, (y + 1) * SCALE); ctx.lineTo((x + 1) * SCALE, (y + 1) * SCALE); }
      if (!at(mask, y, x - 1)) { ctx.moveTo(x * SCALE, y * SCALE); ctx.lineTo(x * SCALE, (y + 1) * SCALE); }
      if (!at(mask, y, x + 1)) { ctx.moveTo((x + 1) * SCALE, y * SCALE); ctx.lineTo((x + 1) * SCALE, (y + 1) * SCALE); }
    }
  }
  ctx.stroke();
  if (errors) { // mark prediction misses: amber = tumor the model missed
    ctx.fillStyle = "#FFB020AA";
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++)
      if (errors[off + y * size + x]) ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
  }
}

function Pane({ title, sub, size, image, z, mask, color, errors }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current && image) drawPane(ref.current, size, image, z, mask, color, errors); },
    [size, image, z, mask, color, errors]);
  return (
    <figure className="pane">
      <canvas ref={ref} className="pane__canvas" aria-label={`${title}, axial slice ${z}`} />
      <figcaption><b>{title}</b><span>{sub}</span></figcaption>
    </figure>
  );
}

export default function Viewer({ data }) {
  const [caseIdx, setCaseIdx] = useState(0);
  const [z, setZ] = useState(null);
  const size = data?.size ?? 32;
  const n = size ** 3;

  const cases = useMemo(() => (data?.cases ?? []).map((c) => {
    const image = decode(c.image, n), gt = decode(c.gt, n), pred = decode(c.pred, n);
    const missed = new Uint8Array(n);
    const area = new Array(size).fill(0);
    for (let i = 0; i < n; i++) {
      if (gt[i] && !pred[i]) missed[i] = 1;
      if (gt[i]) area[Math.floor(i / (size * size))]++;
    }
    return { ...c, imageV: image, gtV: gt, predV: pred, missed, area };
  }), [data, n, size]);

  useEffect(() => { setZ(null); setCaseIdx(0); }, [data]);

  if (!cases.length) {
    return (
      <>
        <div className="cell__head"><h2 id="h-view">Segmentation light-box</h2>
          <p>Available after a run finishes: <code>fedmed export</code> writes the overlays.</p></div>
      </>
    );
  }
  const c = cases[Math.min(caseIdx, cases.length - 1)];
  const slice = z ?? c.focus_slice;
  const maxArea = Math.max(...c.area, 1);

  return (
    <>
      <div className="cell__head">
        <h2 id="h-view">Segmentation light-box</h2>
        <p>Held-out scans, predicted by the final global model as decrypted at a hospital. Axial slices, FLAIR channel.</p>
      </div>
      <div className="tabs" role="tablist" aria-label="Test case">
        {cases.map((k, i) => (
          <button key={k.id} role="tab" aria-selected={i === caseIdx} className={`tab ${i === caseIdx ? "is-on" : ""}`}
                  onClick={() => { setCaseIdx(i); setZ(null); }}>
            {k.id}<span className="tab__dice">Dice {fmt(k.dice, 2)}</span>
          </button>
        ))}
      </div>
      <div className="lightbox">
        <Pane title="MRI" sub="input" size={size} image={c.imageV} z={slice} />
        <Pane title="Radiologist mask" sub="ground truth" size={size} image={c.imageV} z={slice} mask={c.gtV} color="#46E39A" />
        <Pane title="Federated model" sub="prediction · amber = missed" size={size} image={c.imageV} z={slice}
              mask={c.predV} color="#FF4D8D" errors={c.missed} />
      </div>
      <div className="scrub">
        <label htmlFor="slice">Slice {slice + 1} of {size}</label>
        <div className="scrub__track">
          <svg viewBox={`0 0 ${size} 10`} preserveAspectRatio="none" aria-hidden="true" className="scrub__area">
            {c.area.map((a, i) => (
              <rect key={i} x={i + 0.1} width="0.8" y={10 - (a / maxArea) * 10} height={(a / maxArea) * 10}
                    fill={i === slice ? "var(--lesion)" : "var(--ink-4)"} />
            ))}
          </svg>
          <input id="slice" type="range" min={0} max={size - 1} value={slice} onChange={(e) => setZ(Number(e.target.value))} />
        </div>
        <p className="scrub__hint">Bars show how much tumor each slice contains.</p>
      </div>
    </>
  );
}
