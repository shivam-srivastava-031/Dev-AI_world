"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  api, Claim, Progress, Report, RunSummary, STATE_COLOUR, World,
} from "../lib/api";

const GRID = 20;

// A model-driven tick takes minutes, so polling faster than this only burns
// requests. Progress is cheap; everything else refetches only when the tick
// count has actually moved.
const POLL_MS = 15000;

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
              {r.run_id} — {r.policy} · {r.domain} · seed {r.seed} ·{" "}
              {r.ticks_recorded ?? r.ticks}/{r.ticks} ticks
              {r.ticks_recorded < r.ticks ? " · running" : ""}
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
  const [progress, setProgress] = useState<Progress | null>(null);
  const [report, setReport] = useState<Report | null>(null);

  // The poll loop. /progress is the only thing on a timer; it is a couple of
  // counts and costs almost nothing. Everything expensive is keyed off the
  // recorded tick count, so a run that has not advanced refetches nothing.
  useEffect(() => {
    let live = true;
    const poll = () =>
      api.progress(runId)
        .then((p) => { if (live) setProgress(p); })
        .catch(() => {});
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => { live = false; clearInterval(id); };
  }, [runId]);

  const recorded = progress?.ticks_recorded ?? 0;

  useEffect(() => {
    if (!progress) return;
    let live = true;
    const set = <T,>(f: (v: T) => void) => (v: T) => { if (live) f(v); };
    api.run(runId).then(set(setMeta)).catch(() => {});
    api.claims(runId).then(set(setClaims)).catch(() => {});
    api.report(runId).then(set(setReport)).catch(() => {});
    api.metrics(runId).then(set(setMetrics)).catch(() => setMetrics(null));
    api.replay(runId).then(set(setReplay)).catch(() => {});
    return () => { live = false; };
  }, [runId, recorded]);

  return (
    <>
      <LiveHeader progress={progress} report={report} />
      <Reproducibility meta={meta} replay={replay} progress={progress} />
      <div className="grid two">
        <WorldMap runId={runId} ticks={recorded || (meta?.ticks ?? 0)} />
        <LiveReport report={report} />
      </div>
      <Metrics metrics={metrics} />
      <ClaimBoard claims={claims} />
    </>
  );
}

function LiveHeader({ progress, report }:
                    { progress: Progress | null; report: Report | null }) {
  if (!progress) return null;
  const { ticks_recorded: done, ticks_requested: total, live } = progress;
  const pct = total ? (100 * done) / total : 0;
  return (
    <div className="panel">
      <h2 style={{ display: "flex", alignItems: "center", gap: ".6rem" }}>
        {live && (
          <span className="tag" style={{ background: "#8a3a3a" }}>LIVE</span>
        )}
        tick {done} / {total}
      </h2>
      <div style={{
        height: 8, borderRadius: 4, background: "#232735", overflow: "hidden",
        margin: ".5rem 0 .75rem",
      }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: live ? "#c9a227" : "#4f9d69",
        }} />
      </div>
      <div className="stat">
        <div><span className="k">trials</span>
             <span className="v">{progress.trials}</span></div>
        <div><span className="k">claims</span>
             <span className="v">{progress.claims}</span></div>
      </div>
      {report && (
        <p className="note" style={{ marginTop: ".75rem" }}>{report.headline}</p>
      )}
      {live && (
        <p className="note">
          Every number here is from the last checkpoint, not the current tick.
          A partial result is a partial result: it is labelled that way rather
          than extrapolated to what the run might end up showing.
        </p>
      )}
    </div>
  );
}

function Reproducibility({ meta, replay, progress }:
                         { meta: any; replay: any; progress: Progress | null }) {
  if (!meta) return null;
  const live = progress?.live ?? false;
  return (
    <div className="panel">
      <h2>Provenance</h2>
      <div className="stat">
        <div><span className="k">seed</span><span className="v mono">{meta.seed}</span></div>
        <div><span className="k">policy</span><span className="v">{meta.policy}</span></div>
        <div><span className="k">domain</span><span className="v">{meta.domain}</span></div>
        <div><span className="k">
          {live ? "state hash so far" : "final state hash"}
        </span>
          <span className="v mono" style={{ fontSize: ".9rem" }}>
            {String(meta.final_state_hash ?? "").slice(0, 16)}
          </span>
        </div>
        <div><span className="k">replay</span>
          <span className="v" style={{ color: replay?.matched ? "#4f9d69" : "#c46a1f" }}>
            {replay
              ? replay.matched
                // Never plain "verified" on a live run: what has been checked
                // is the ticks recorded so far, and saying otherwise would
                // claim a guarantee about ticks that have not happened yet.
                ? `verified to tick ${replay.ticks_checked}`
                : "DIVERGED"
              : "…"}
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
  // Follow the newest recorded tick until the observer scrubs back, then stay
  // where they put it. Yanking the slider out from under someone inspecting
  // tick 40 would make the timeline unusable on a live run.
  const [following, setFollowing] = useState(true);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (following) setTick(Math.max(ticks - 1, 0));
  }, [ticks, following]);

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
             onChange={(e) => {
               setFollowing(false);
               setTick(Number(e.target.value));
             }} />
      <div className="muted">
        tick {tick} / {ticks} · <span style={{ color: "#e8c46a" }}>agents</span>
        {" · "}<span style={{ color: "#2f5d43" }}>growing</span>
        {" · "}<span style={{ color: "#3f7fb5" }}>channels</span>
        {" · "}{world?.channels.length ?? 0} built
        {!following && (
          <>
            {" · "}
            <a onClick={() => setFollowing(true)}
               style={{ cursor: "pointer", color: "#c9a227" }}>
              follow latest
            </a>
          </>
        )}
      </div>
    </div>
  );
}

function pctText(x: number | null | undefined): string {
  // An unmeasured quantity is shown as a dash, never as 0%. The two mean
  // completely different things and the dashboard must not conflate them.
  return typeof x === "number" ? `${(100 * x).toFixed(0)}%` : "—";
}

function LiveReport({ report }: { report: Report | null }) {
  if (!report) return <div className="panel muted">Loading report…</div>;
  const a = report.actions ?? {};
  const l = report.llm ?? {};
  const e = report.exploration ?? {};
  const k = report.knowledge ?? {};

  const verbs: [string, number][] =
    Object.entries(a.by_verb ?? {}) as [string, number][];
  const maxVerb = Math.max(1, ...verbs.map(([, n]) => n));

  return (
    <div className="panel">
      <h2>How the days are being spent</h2>

      <table style={{ marginBottom: ".75rem" }}>
        <tbody>
          {verbs.map(([verb, n]) => (
            <tr key={verb}>
              <td className="mono" style={{ width: "9rem" }}>{verb}</td>
              <td style={{ width: "100%" }}>
                <div style={{
                  width: `${(100 * n) / maxVerb}%`, height: 10, borderRadius: 3,
                  background: verb === "PLANT" || verb === "HARVEST"
                    ? "#4f9d69" : verb === "NOOP" ? "#8a6d9c" : "#7a8fa6",
                }} />
              </td>
              <td className="mono" style={{ textAlign: "right" }}>{n}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="stat" style={{ marginBottom: ".75rem" }}>
        <div><span className="k">invalid actions</span>
             <span className="v">{pctText(a.invalid_action_rate)}</span></div>
        <div><span className="k">productive</span>
             <span className="v">{pctText(a.productive_action_rate)}</span></div>
        <div><span className="k">transport failures</span>
             <span className="v">{pctText(l.transport_failure_rate)}</span></div>
        <div><span className="k">median latency</span>
             <span className="v">
               {l.median_latency_ms != null
                 ? `${(l.median_latency_ms / 1000).toFixed(1)}s` : "—"}
             </span></div>
      </div>

      {a.rejections_by_code && (
        <>
          <div className="muted" style={{ marginBottom: ".3rem" }}>
            Why the world said no
          </div>
          <table style={{ marginBottom: ".75rem" }}>
            <tbody>
              {(Object.entries(a.rejections_by_code) as [string, number][])
                .map(([code, n]) => (
                <tr key={code}>
                  <td className="mono" style={{ fontSize: ".8rem" }}>{code}</td>
                  <td style={{ textAlign: "right" }}>{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <div className="stat">
        <div><span className="k">trials</span>
             <span className="v">{e.trials ?? 0}</span></div>
        <div><span className="k">distinct recipes</span>
             <span className="v">{e.distinct_recipes ?? 0}</span></div>
        <div><span className="k">claims</span>
             <span className="v">{k.claims_proposed ?? 0}</span></div>
        <div><span className="k">banked</span>
             <span className="v">{k.banked ?? 0}</span></div>
      </div>

      {l.degraded && (
        <p className="note" style={{ marginTop: ".75rem", color: "#c46a1f" }}>
          DEGRADED: calls are failing in transport. Those ticks are the model
          being silent, not the model choosing to do nothing.
        </p>
      )}
      <p className="note" style={{ marginTop: ".75rem" }}>
        The invalid-action rate is the world rejecting a proposal, and it is a
        measurement rather than a bug: it says how well the agents model the
        rules they were given.
      </p>
    </div>
  );
}

function Metrics({ metrics }: { metrics: any }) {
  if (!metrics) {
    return (
      <div className="panel muted">
        No civilization report yet. It is computed once, over the whole run, at
        the end — several of its measures (time-to-competence, resilience,
        collective regret) are not defined on a partial history and would be
        misleading if computed early. The panels above read the checkpoint and
        are live.
      </div>
    );
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
