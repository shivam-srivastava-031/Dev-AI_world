"use client";

// The task section: objectives an observer sets, and how far along the run is.
//
// Two kinds of task, and the whole design turns on keeping them apart:
//
//   observer -- set here, scored from the log, never shown to an agent. Adding
//     one cannot change what the run does, which is why it is safe to add one
//     to a run that is already going.
//
//   briefed  -- came from a brief fixed before the run started. If it carried
//     a sentence, that sentence was in the agents' prompt and the run is
//     STEERED, labelled so everywhere it appears: comparing a steered run with
//     an unsteered one without noticing is the mistake this feature makes
//     easiest to make. A briefed task with no sentence was scored but never
//     said, and does not steer anything.
//
// Nothing here computes progress. The backend decides the value, whether a
// percentage even applies, and the sentence that describes the state, so the
// dashboard and `aiciv tasks` can never disagree about how far along a run is.

import { useEffect, useMemo, useState } from "react";
import {
  Brief, ScoredTask, TaskBoard, TaskInput, TaskMetric, api,
} from "../lib/api";

const GREEN = "#4f9d69";
const AMBER = "#c9a227";
const GREY = "#4a5062";
const VIOLET = "#8a6d9c";

/** Percentage for display only; the fraction itself came from the API. */
function pct(x: number): string {
  return `${(100 * x).toFixed(0)}%`;
}

/** A value in the units the metric is actually in. Never guesses a missing
 *  one into existence: an unmeasured quantity is a dash, not a zero. */
function showValue(v: number | null, unit: string): string {
  if (v === null) return "—";
  if (unit === "fraction") return `${(100 * v).toFixed(1)}%`;
  if (unit === "kg") return `${v.toFixed(2)} kg`;
  return String(v);
}

export default function TaskSection({ runId, refreshKey }:
                                    { runId: string; refreshKey: number }) {
  const [board, setBoard] = useState<TaskBoard | null>(null);
  const [metrics, setMetrics] = useState<TaskMetric[]>([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.taskMetrics().then(setMetrics).catch(() => {});
  }, []);

  // Refetched whenever the run has actually advanced, on the same key the rest
  // of the dashboard uses. Scoring walks the action log, so polling it on its
  // own timer would re-read a run that had not moved.
  useEffect(() => {
    let alive = true;
    api.tasks(runId)
      .then((b) => { if (alive) { setBoard(b); setErr(""); } })
      .catch((e) => { if (alive) setErr(String(e)); });
    return () => { alive = false; };
  }, [runId, refreshKey]);

  const reload = () => api.tasks(runId).then(setBoard).catch(() => {});

  return (
    <>
      <div className="panel">
        <h2 style={{ display: "flex", alignItems: "center", gap: ".6rem" }}>
          Tasks
          {board?.steered && (
            <span className="tag" style={{ background: AMBER }}>STEERED</span>
          )}
          <span className="muted" style={{ fontWeight: 400, fontSize: ".85rem" }}>
            {board ? `${board.done} of ${board.count} met` : "…"}
          </span>
        </h2>

        {err && <p className="mono muted">{err}</p>}

        {board && board.count === 0 && (
          <p className="muted">
            No tasks on this run. A task is a metric, a floor or a ceiling, and
            a target — set one below and it is scored against every checkpoint
            from here on, including the ones already recorded.
          </p>
        )}

        {board && board.count > 0 && (
          <>
            {board.mean_progress !== null && (
              <p className="muted" style={{ marginTop: 0 }}>
                Mean progress across the counting tasks:{" "}
                <span className="mono">{pct(board.mean_progress)}</span>.
                Ceilings and rates are not averaged in — met or not met is not
                a number, and turning it into one would invent a figure.
              </p>
            )}
            <div style={{ display: "grid", gap: ".85rem", marginTop: ".75rem" }}>
              {board.tasks.map((t) => (
                <TaskRow key={t.task_id} task={t}
                         onDelete={() => api.deleteTask(runId, t.task_id)
                           .then(setBoard).catch((e) => setErr(String(e)))} />
              ))}
            </div>
          </>
        )}

        <AddTaskForm metrics={metrics} allowDirective={false}
                     onAdd={(t) => api.addTask(runId, t).then(setBoard)} />

        <p className="note" style={{ marginTop: ".75rem" }}>
          A task added here is scored from the log and the agents never see it,
          so setting one on a live run cannot change what that run does. To put
          a task in front of the agents you write a brief below and launch a
          run under it — that is a different experiment, and it is marked as one.
        </p>
      </div>

      <Briefs metrics={metrics} onChange={reload} />
    </>
  );
}

function TaskRow({ task, onDelete }:
                 { task: ScoredTask; onDelete: () => void }) {
  // Came from a brief, and separately: was actually said. A briefed task with
  // no directive was scored but never spoken, so it gets no STEERED-coloured
  // badge -- that badge has to mean the agents heard something.
  const briefed = task.source === "briefed";
  const spoken = briefed && task.directive.trim().length > 0;
  const bar = task.done ? GREEN : AMBER;
  const want = task.comparator === ">=" ? "at least" : "at most";

  return (
    <div style={{ borderTop: "1px solid var(--line)", paddingTop: ".7rem" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: ".5rem" }}>
        <strong>{task.title}</strong>
        {briefed && (
          <span className="tag" style={{ background: spoken ? AMBER : GREY }}>
            {spoken ? "BRIEFED" : "FROM BRIEF · UNSAID"}
          </span>
        )}
        <span className="muted" style={{ fontSize: ".8rem" }}>
          {task.metric_label} {want} {showValue(task.target, task.unit)}
        </span>
        <span style={{ flex: 1 }} />
        <a onClick={onDelete} className="muted"
           style={{ cursor: "pointer", fontSize: ".8rem" }}>remove</a>
      </div>

      {/* A bar only for a target that fills up. A ceiling gets a state pill:
          being at 30% of a 15% limit is not "half done", it is not done. */}
      {task.kind === "accumulate" && task.progress !== null ? (
        <div style={{ marginTop: ".4rem" }}>
          <div style={{
            height: 8, borderRadius: 4, background: "#232735", overflow: "hidden",
          }}>
            <div style={{ width: pct(task.progress), height: "100%",
                          background: bar }} />
          </div>
          <div className="muted" style={{ fontSize: ".8rem", marginTop: ".25rem" }}>
            <span className="mono">
              {showValue(task.value, task.unit)} / {showValue(task.target, task.unit)}
            </span>
            {" · "}{pct(task.progress)}{" · "}{task.detail}
          </div>
        </div>
      ) : (
        <div className="muted" style={{ fontSize: ".8rem", marginTop: ".4rem",
                                        display: "flex", alignItems: "center",
                                        gap: ".5rem" }}>
          <span className="tag" style={{
            background: !task.measurable ? GREY : task.done ? GREEN : VIOLET,
          }}>
            {!task.measurable ? "NOT YET MEASURABLE" : task.done ? "MET" : "NOT MET"}
          </span>
          <span className="mono">{showValue(task.value, task.unit)}</span>
          <span>{task.detail}</span>
        </div>
      )}

      {briefed && (
        <p className="muted" style={{ fontSize: ".8rem", marginTop: ".35rem",
                                      fontStyle: spoken ? "italic" : "normal" }}>
          {spoken
            ? <>told to the agents: “{task.directive}”</>
            : <>came from a brief, but with no sentence to say — the agents
                were not told this one</>}
        </p>
      )}
    </div>
  );
}

/** The form. `allowDirective` is what separates the two paths: a run already
 *  under way has no channel to its agents, so the field is not offered there
 *  rather than offered and ignored. */
function AddTaskForm({ metrics, allowDirective, onAdd }: {
  metrics: TaskMetric[];
  allowDirective: boolean;
  onAdd: (t: TaskInput) => Promise<unknown>;
}) {
  const [title, setTitle] = useState("");
  const [metric, setMetric] = useState("trials");
  const [comparator, setComparator] = useState(">=");
  const [target, setTarget] = useState("100");
  const [directive, setDirective] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const chosen = useMemo(
    () => metrics.find((m) => m.key === metric), [metrics, metric]);

  // Fractions are entered as percentages because that is how people say them,
  // and converted here. The API only ever sees the fraction, so the CLI and
  // the dashboard store the same number.
  const isFraction = chosen?.unit === "fraction";
  const asTarget = (raw: string) => {
    const n = Number(raw);
    return isFraction ? n / 100 : n;
  };

  const submit = async () => {
    const n = asTarget(target);
    if (!title.trim()) { setErr("give the task a title"); return; }
    if (!Number.isFinite(n) || n < 0) { setErr("target must be a number ≥ 0"); return; }
    setBusy(true); setErr("");
    try {
      await onAdd({ title: title.trim(), metric, comparator, target: n,
                    ...(allowDirective ? { directive: directive.trim() } : {}) });
      setTitle(""); setDirective("");
    } catch (e) {
      setErr(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ marginTop: "1rem", borderTop: "1px solid var(--line)",
                  paddingTop: ".85rem" }}>
      <div style={{ display: "flex", gap: ".5rem", flexWrap: "wrap",
                    alignItems: "center" }}>
        <input value={title} placeholder="what you want done"
               onChange={(e) => setTitle(e.target.value)}
               style={{ flex: "2 1 16rem", minWidth: "10rem" }} />
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {metrics.map((m) => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>
        <select value={comparator} onChange={(e) => setComparator(e.target.value)}>
          <option value=">=">at least</option>
          <option value="<=">at most</option>
        </select>
        <input value={target} onChange={(e) => setTarget(e.target.value)}
               inputMode="decimal" style={{ width: "6rem" }} />
        {isFraction && <span className="muted">%</span>}
        <button onClick={submit} disabled={busy}>
          {busy ? "adding…" : "Add task"}
        </button>
      </div>

      {allowDirective && (
        <input value={directive}
               placeholder="the sentence the agents will read (optional)"
               onChange={(e) => setDirective(e.target.value)}
               style={{ width: "100%", marginTop: ".5rem" }} />
      )}

      {chosen && (
        <p className="muted" style={{ fontSize: ".8rem", marginTop: ".4rem" }}>
          {chosen.describes}.{" "}
          {chosen.cumulative && comparator === ">="
            ? "This counts up, so it gets a real percentage."
            : "This is a level rather than a total, so it is reported met or "
              + "not met with the current value — a percentage of it would "
              + "not mean anything."}
        </p>
      )}
      {err && <p className="muted" style={{ color: "#c46a1f", fontSize: ".85rem" }}>{err}</p>}
    </div>
  );
}

function Briefs({ metrics, onChange }:
                { metrics: TaskMetric[]; onChange: () => void }) {
  const [briefs, setBriefs] = useState<Brief[]>([]);
  const [name, setName] = useState("");
  const [open, setOpen] = useState<string>("");
  const [err, setErr] = useState("");

  const reload = () => api.briefs().then(setBriefs).catch((e) => setErr(String(e)));
  useEffect(() => { reload(); }, []);

  return (
    <div className="panel">
      <h2>Briefs — tasks the agents are actually told</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        A brief is fixed before a run starts and its sentences go into the
        agents&apos; prompt. Once a run has used a brief it is closed: it is the
        record of what those agents were told, and editing it afterwards would
        leave the run&apos;s manifest describing something that no longer exists.
      </p>

      {err && <p className="mono muted">{err}</p>}

      {briefs.map((b) => (
        <div key={b.brief_id} style={{ borderTop: "1px solid var(--line)",
                                       paddingTop: ".7rem", marginTop: ".7rem" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: ".5rem",
                        flexWrap: "wrap" }}>
            <strong>{b.name}</strong>
            <span className="tag" style={{ background: b.closed ? GREY : GREEN }}>
              {b.closed ? "CLOSED" : "OPEN"}
            </span>
            <span className="muted" style={{ fontSize: ".8rem" }}>
              {b.tasks.length} task{b.tasks.length === 1 ? "" : "s"}
              {b.launched_runs.length > 0 &&
                ` · used by ${b.launched_runs.join(", ")}`}
            </span>
            <span style={{ flex: 1 }} />
            {!b.closed && (
              <>
                <a onClick={() => setOpen(open === b.brief_id ? "" : b.brief_id)}
                   style={{ cursor: "pointer", fontSize: ".8rem" }}>
                  {open === b.brief_id ? "done" : "add task"}
                </a>
                <a className="muted" style={{ cursor: "pointer", fontSize: ".8rem" }}
                   onClick={() => api.deleteBrief(b.brief_id)
                     .then(reload).catch((e) => setErr(String(e)))}>
                  delete
                </a>
              </>
            )}
          </div>

          <ul style={{ margin: ".4rem 0", paddingLeft: "1.1rem" }}>
            {b.tasks.map((t) => (
              <li key={t.task_id} className="muted" style={{ fontSize: ".85rem" }}>
                {t.title} — {t.metric}{" "}
                {t.comparator === ">=" ? "≥" : "≤"} {t.target}
                {t.directive
                  ? <> · told as: <em>“{t.directive}”</em></>
                  : <> · <span style={{ color: VIOLET }}>scored silently, not
                      said to the agents</span></>}
                {!b.closed && (
                  <>
                    {" · "}
                    <a style={{ cursor: "pointer" }}
                       onClick={() => api.deleteBriefTask(b.brief_id, t.task_id)
                         .then(reload).catch((e) => setErr(String(e)))}>
                      remove
                    </a>
                  </>
                )}
              </li>
            ))}
          </ul>

          {b.method_terms.length > 0 && (
            <p className="note" style={{ color: "#c46a1f" }}>
              This brief uses method vocabulary ({b.method_terms.join(", ")}).
              That is your call, but a run given it can no longer be compared
              against a rules_only run that was not — the scaffold experiment
              exists to withhold exactly these words.
            </p>
          )}

          <p className="muted mono" style={{ fontSize: ".8rem" }}>
            {b.command}
          </p>

          {open === b.brief_id && (
            <AddTaskForm metrics={metrics} allowDirective
                         onAdd={(t) => api.addBriefTask(b.brief_id, t)
                           .then(() => { reload(); onChange(); })} />
          )}
        </div>
      ))}

      <div style={{ display: "flex", gap: ".5rem", marginTop: "1rem",
                    borderTop: "1px solid var(--line)", paddingTop: ".85rem" }}>
        <input value={name} placeholder="name a new brief"
               onChange={(e) => setName(e.target.value)}
               style={{ flex: "1 1 14rem" }} />
        <button
          onClick={() => {
            if (!name.trim()) return;
            api.createBrief(name.trim())
              .then(() => { setName(""); reload(); })
              .catch((e) => setErr(String(e.message ?? e)));
          }}>
          New brief
        </button>
      </div>

      <p className="note" style={{ marginTop: ".75rem" }}>
        A brief only reaches agents that read a prompt. Launch it with{" "}
        <span className="mono">--policy llm</span>; a scripted policy never
        sees it, and the run would be scored against tasks nobody was told.
      </p>
    </div>
  );
}
