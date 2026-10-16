import { useEffect, useReducer, useRef, useState } from "react";
import { initialRun, reduceEvent } from "./runState.js";

const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const WS_URL = API.replace(/^http/, "ws") + "/ws";
const RETRY_MS = 5000;

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

/**
 * Connects to the FedMed API over WebSocket (history replay + live events).
 * If the API is not reachable, falls back to a bundled sample run so the UI is never empty,
 * and keeps retrying in the background.
 *
 * mode: "connecting" | "live" | "sample"
 */
export function useRun() {
  const [run, dispatch] = useReducer(reduceEvent, undefined, initialRun);
  const [mode, setMode] = useState("connecting");
  const [extras, setExtras] = useState({ segmentation: null, baseline: null, audit: null });
  const modeRef = useRef("connecting");
  const setBoth = (m) => { modeRef.current = m; setMode(m); };

  useEffect(() => {
    let socket, retryTimer, closed = false, gotLive = false;

    const loadExtras = async (base, demo) => {
      const get = async (live, sample) => {
        try { return await getJson(base ? `${API}/api/${live}` : `/demo/${sample}`); }
        catch { return null; }
      };
      const [segmentation, baseline, audit] = await Promise.all([
        get("segmentation", "segmentation.json"), get("baseline", "baseline.json"), get("audit", "audit.json"),
      ]);
      setExtras({ segmentation, baseline, audit });
    };

    const loadSample = async () => {
      if (gotLive) return;
      try {
        const text = await (await fetch("/demo/metrics.jsonl")).text();
        dispatch({ type: "reset", run_id: "sample-run" });
        text.split("\n").filter(Boolean).forEach((l) => dispatch(JSON.parse(l)));
        setBoth("sample");
        loadExtras(false);
      } catch { setBoth("sample"); }
    };

    const connect = () => {
      if (closed) return;
      try { socket = new WebSocket(WS_URL); } catch { loadSample(); retryTimer = setTimeout(connect, RETRY_MS); return; }
      let opened = false;
      socket.onopen = () => { opened = true; };
      socket.onmessage = (m) => {
        if (!gotLive) { gotLive = true; setBoth("live"); }
        const e = JSON.parse(m.data);
        dispatch(e);
        if (e.type === "reset" || e.type === "eval_round" || e.type === "run_end") loadExtras(true);
      };
      socket.onclose = () => {
        if (closed) return;
        if (!opened && modeRef.current !== "live") loadSample();
        retryTimer = setTimeout(connect, RETRY_MS);
      };
      socket.onerror = () => socket.close();
    };

    connect();
    return () => { closed = true; clearTimeout(retryTimer); socket && socket.close(); };
  }, []);

  return { run, mode, ...extras };
}

// Poll hospital node health (only meaningful against a live API).
export function useNodes(enabled) {
  const [nodes, setNodes] = useState(null);
  useEffect(() => {
    if (!enabled) { setNodes(null); return; }
    let stop = false;
    const tick = async () => {
      try { const n = await getJson(`${API}/api/nodes`); if (!stop) setNodes(n); } catch { if (!stop) setNodes(null); }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => { stop = true; clearInterval(id); };
  }, [enabled]);
  return nodes;
}
