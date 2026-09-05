"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api, Claim, RunSummary, STATE_COLOUR, World } from "../lib/api";

const GRID = 20;

export default function Page() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string>("");
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    api.runs()
      .then((r) => { setRuns(r); if (r.length) setRunId(r[0].run_id); })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return (
      <div className="panel">
        <h2>No API</h2>
        <p className="muted">
          Start it with <span className="mono">uvicorn api.main:app --port 8000</span>,
          after producing a run with <span className="mono">aiciv run</span>.
        </p>
        <p className="mono muted">{err}</p>
      </div>
    );
  }

  if (!runs.length) return <div className="panel muted">Looking for runs…</div>;

  return (
    <>
      <div className="panel">
        <label className="muted">Run&nbsp;</label>
        <select value={runId} onChange={(e) => setRunId(e.target.value)}>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} — {r.policy} · {r.domain} · seed {r.seed} · {r.ticks} ticks
            </option>
          ))}
        </select>
      </div>
      {runId && <RunView key={runId} runId={runId} />}
    </>
  );
}

function RunView({ runId }: { runId: string }) {
  const [meta, setMeta] = useState<any>(null);
  const [claims, setClaims] = useState<Claim[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [replay, setReplay] = useState<any>(null);

  useEffect(() => {
    api.run(runId).then(setMeta).catch(() => {});
    api.claims(runId).then(setClaims).catch(() => {});
    api.metrics(runId).then(setMetrics).catch(() => setMetrics(null));
    api.replay(runId).then(setReplay).catch(() => {});
  }, [runId]);

  return (
    <>
      <Reproducibility meta={meta} replay={replay} />
      <div className="grid two">
        <WorldMap runId={runId} ticks={meta?.ticks ?? 0} />
        <Metrics metrics={metrics} />
      </div>
      <ClaimBoard claims={claims} />
    </>
  );
}

function Reproducibility({ meta, replay }: { meta: any; replay: any }) {
  if (!meta) return null;
  return (
    <div className="panel">
      <h2>Provenance</h2>
      <div className="stat">
        <div><span className="k">seed</span><span className="v mono">{meta.seed}</span></div>
        <div><span className="k">policy</span><span className="v">{meta.policy}</span></div>
        <div><span className="k">domain</span><span className="v">{meta.domain}</span></div>
        <div><span className="k">final state hash</span>
          <span className="v mono" style={{ fontSize: ".9rem" }}>
            {String(meta.final_state_hash ?? "").slice(0, 16)}
          </span>
        </div>
        <div><span className="k">replay</span>
          <span className="v" style={{ color: replay?.matched ? "#4f9d69" : "#c46a1f" }}>
            {replay ? (replay.matched ? "verified" : "DIVERGED") : "…"}
          </span>
        </div>
      </div>
      {replay && !replay.matched && (
        <p className="note" style={{ marginTop: ".75rem" }}>{replay.detail}</p>
      )}
      <p className="note" style={{ marginTop: ".75rem" }}>
        Replay re-executes the recorded decisions through a fresh engine and
        compares every tick. It needs no model even for a run driven by one:
        the decisions are data, and only the world is under test.
      </p>
    </div>
  );
}

function WorldMap({ runId, ticks }: { runId: string; ticks: number }) {
  const [tick, setTick] = useState(0);
  const [world, setWorld] = useState<World | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => { setTick(Math.floor(ticks * 0.75)); }, [ticks]);

  useEffect(() => {
    let live = true;
    api.world(runId, tick).then((w) => { if (live) setWorld(w); }).catch(() => {});
    return () => { live = false; };
  }, [runId, tick]);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !world) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const cell = el.width / GRID;

    ctx.fillStyle = "#12141a";
    ctx.fillRect(0, 0, el.width, el.height);
    ctx.strokeStyle = "#232735";
    for (let i = 0; i <= GRID; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, el.height);
      ctx.moveTo(0, i * cell); ctx.lineTo(el.width, i * cell);
      ctx.stroke();
    }

    // NOTE: no soil, no strata. The API withholds them and so does this: an
    // observer who could see the partition could leak it back into the
    // experiment through prompt design.
    ctx.fillStyle = "#2f5d43";
    for (const key of Object.keys(world.planted)) {
      const t = Number(key);
      ctx.fillRect((t % GRID) * cell + 2, Math.floor(t / GRID) * cell + 2,
        cell - 4, cell - 4);
    }

    ctx.fillStyle = "#3f7fb5";
    for (const t of world.channels) {
      ctx.fillRect((t % GRID) * cell + 4, Math.floor(t / GRID) * cell + 4,
        cell - 8, cell - 8);
    }

    ctx.fillStyle = "#e8c46a";
    for (const [id, pos] of Object.entries(world.agents)) {
      if (!pos || pos[0] == null) continue;
      ctx.beginPath();
      ctx.arc(pos[0] * cell + cell / 2, pos[1] * cell + cell / 2, cell * 0.3,
        0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#12141a";
      ctx.font = `${Math.floor(cell * 0.45)}px ui-monospace, monospace`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(id, pos[0] * cell + cell / 2, pos[1] * cell + cell / 2);
      ctx.fillStyle = "#e8c46a";
    }
  }, [world]);

  return (
    <div className="panel">
      <h2>World</h2>
      <canvas ref={canvas} width={520} height={520}
              style={{ width: "100%", height: "auto", borderRadius: 6 }} />
      <input type="range" min={0} max={Math.max(ticks - 1, 0)} value={tick}
             onChange={(e) => setTick(Number(e.target.value))} />
      <div className="muted">
        tick {tick} / {ticks} · <span style={{ color: "#e8c46a" }}>agents</span>
        {" · "}<span style={{ color: "#2f5d43" }}>growing</span>
        {" · "}<span style={{ color: "#3f7fb5" }}>channels</span>
        {" · "}{world?.channels.length ?? 0} built
      </div>
    </div>
  );
}

function Metrics({ metrics }: { metrics: any }) {
  if (!metrics) {
    return <div className="panel muted">No metrics recorded for this run.</div>;
  }
  const k = metrics.knowledge ?? {};
  const g = metrics.grading ?? {};
  const s = metrics.social ?? {};
  const c = metrics.collective ?? {};
  return (
    <div className="panel">
      <h2>What the civilization actually did</h2>
      <div className="stat" style={{ marginBottom: ".75rem" }}>
        <div><span className="k">banked</span><span className="v">{k.banked ?? 0}</span></div>
        <div><span className="k">generalized</span><span className="v">{k.generalized ?? 0}</span></div>
        <div><span className="k">refuted</span><span className="v">{k.refuted ?? 0}</span></div>
        <div><span className="k">false discoveries</span>
          <span className="v">{g.false_positives ?? 0}</span></div>
        <div><span className="k">regret</span>
          <span className="v">{c.collective_regret ?? "—"}</span></div>
      </div>
      <table>
        <tbody>
          <tr><td className="muted">independently corroborated</td>
              <td>{k.independently_corroborated ?? 0}</td></tr>
          <tr><td className="muted">diffusion only</td>
              <td>{k.diffusion_only ?? 0}</td></tr>
          <tr><td className="muted">teaching events</td>
              <td>{s.teach_events ?? 0}</td></tr>
          <tr><td className="muted">blind adoption</td>
              <td>{s.blind_adoption ?? 0}
                {s.blind_adoption_rate != null && ` (${s.blind_adoption_rate})`}</td></tr>
          <tr><td className="muted">knowledge resilience</td>
              <td>{metrics.resilience?.resilience ?? "—"}</td></tr>
        </tbody>
      </table>
      <p className="note" style={{ marginTop: ".75rem" }}>
        A claim confirmed only by taught replications counts as diffusion, not
        independent corroboration. The two are different results.
      </p>
    </div>
  );
}

function ClaimBoard({ claims }: { claims: Claim[] }) {
  const sorted = useMemo(
    () => [...claims].sort((a, b) => a.claim_id.localeCompare(b.claim_id)),
    [claims]);

  if (!sorted.length) {
    return (
      <div className="panel muted">
        No claims. That is a finding, not an error: the protocol is optional and
        agents may simply farm.
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Claim board</h2>
      <table>
        <thead>
          <tr>
            <th>claim</th><th>by</th><th>state</th><th>hypothesis</th>
            <th>effect</th><th>95% CI</th><th>p</th><th>n</th><th>replication</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((c) => {
            const v = c.verdicts?.supported ?? {};
            const fmt = (x: any, d = 3) =>
              typeof x === "number" ? x.toFixed(d) : "—";
            return (
              <tr key={c.claim_id}>
                <td className="mono">{c.claim_id}</td>
                <td>#{c.author}</td>
                <td>
                  <span className="tag"
                        style={{ background: STATE_COLOUR[c.state] ?? "#666" }}>
                    {c.state}
                  </span>
                </td>
                <td className="muted">{describe(c)}</td>
                <td>{fmt(v.effect)}</td>
                <td className="mono" style={{ fontSize: ".8rem" }}>
                  {v.ci_low != null ? `[${fmt(v.ci_low)}, ${fmt(v.ci_high)}]` : "—"}
                </td>
                <td>{v.p_value != null ? Number(v.p_value).toExponential(1) : "—"}</td>
                <td>{v.n_treat != null ? `${v.n_treat}/${v.n_ctrl}` : "—"}</td>
                <td className="muted">{c.replication_kind ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="note" style={{ marginTop: ".75rem" }}>
        Every verdict carries an effect, an interval and an n. A p-value on its
        own says whether a difference exists without saying whether it matters.
      </p>
    </div>
  );
}

function describe(c: Claim): string {
  const side = (o: Record<string, any>) =>
    Object.entries(o)
      .map(([k, v]) => `${k}=${v.op === "eq" ? v.value : JSON.stringify(v.values ?? v)}`)
      .join(" & ") || "anything";
  return `${side(c.spec)} > ${side(c.baseline)}`;
}
