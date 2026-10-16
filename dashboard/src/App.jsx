import React from "react";
import { useNodes, useRun } from "./lib/useRun.js";
import { latestEval, latestFit } from "./lib/runState.js";
import Header from "./components/Header.jsx";
import Convergence from "./components/Convergence.jsx";
import Nodes from "./components/Nodes.jsx";
import Privacy from "./components/Privacy.jsx";
import Viewer from "./components/Viewer.jsx";
import Audit from "./components/Audit.jsx";
import EventLog from "./components/EventLog.jsx";

export default function App() {
  const { run, mode, segmentation, baseline, audit } = useRun();
  const nodes = useNodes(mode === "live");
  const ev = latestEval(run);
  const fit = latestFit(run);

  return (
    <div className="page">
      <Header run={run} mode={mode} baseline={baseline} ev={ev} />
      <main className="grid">
        <section className="cell cell--chart" aria-labelledby="h-conv">
          <Convergence run={run} baseline={baseline} />
        </section>
        <section className="cell cell--nodes" aria-labelledby="h-nodes">
          <Nodes run={run} ev={ev} fit={fit} liveNodes={nodes} />
        </section>
        <section className="cell cell--privacy" aria-labelledby="h-priv">
          <Privacy run={run} fit={fit} ev={ev} />
        </section>
        <section className="cell cell--viewer" aria-labelledby="h-view">
          <Viewer data={segmentation} />
        </section>
        <section className="cell cell--audit" aria-labelledby="h-audit">
          <Audit audit={audit} />
        </section>
        <section className="cell cell--log" aria-labelledby="h-log">
          <EventLog run={run} />
        </section>
      </main>
      <footer className="foot">
        FedMed trains one model across hospitals without moving patient data. Weights travel encrypted;
        the server only ever adds ciphertexts.
      </footer>
    </div>
  );
}
