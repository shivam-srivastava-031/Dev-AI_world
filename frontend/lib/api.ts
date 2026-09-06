// Thin typed client.
//
// The dashboard reads. The one exception is the task board -- an observer sets
// objectives here -- and those writes go to the board file, never to a run. It
// still computes nothing the backend has already decided: a task's percentage,
// whether a percentage even applies, and the sentence describing it all come
// down scored, so the UI cannot disagree with the CLI about how far along
// something is.

const BASE = process.env.NEXT_PUBLIC_API ?? "/api";

export type RunSummary = {
  run_id: string; arm: string; seed: number; ticks: number;
  agents: number; policy: string; domain: string;
  final_state_hash: string; started_at: string;
  status: string; ticks_recorded: number;
  // True when the agents were briefed. Not cosmetic: a steered run and an
  // unsteered one are different experiments and must not be read as one series.
  steered: boolean; tasks: number;
};

export type TaskMetric = {
  key: string; label: string; unit: "count" | "fraction" | "kg";
  cumulative: boolean; describes: string;
};

export type ScoredTask = {
  task_id: string; title: string; metric: string; metric_label: string;
  comparator: ">=" | "<="; target: number; directive: string;
  source: "observer" | "briefed"; created_at: string; brief_id: string;
  // "accumulate" has a real fraction in `progress`. "threshold" does not, and
  // `progress` is null there: a ceiling is met or it is not.
  kind: "accumulate" | "threshold";
  unit: "count" | "fraction" | "kg";
  value: number | null;
  progress: number | null;
  done: boolean;
  measurable: boolean;
  detail: string;
};

export type TaskBoard = {
  run_id: string;
  tasks: ScoredTask[];
  count: number; done: number; unmeasurable: number;
  mean_progress: number | null;
  // `steered` is narrower than `briefed_count`: a briefed task with no
  // directive text was scored but never spoken, so it does not steer the run.
  steered: boolean; briefed_count: number; spoken_count: number;
};

export type Brief = {
  brief_id: string; name: string; created_at: string;
  tasks: ScoredTask[]; launched_runs: string[];
  directives: string[]; closed: boolean; command: string;
  // Method vocabulary found in the directives. Reported, never blocked: it
  // says the run can no longer be compared with one that was not given it.
  method_terms: string[];
};

export type TaskInput = {
  title: string; metric: string; comparator: string;
  target: number; directive?: string;
};

export type Progress = {
  run_id: string;
  ticks_recorded: number;
  ticks_requested: number;
  status: string;
  // Derived from the recorded ticks, not the status column: a checkpoint
  // written by an older build says "complete" however far in it actually is.
  live: boolean;
  trials: number;
  claims: number;
};

export type Report = {
  run_id: string;
  ticks_recorded: number; ticks_requested: number;
  complete: boolean; headline: string;
  policy: string; domain: string; seed: number;
  actions: Record<string, any>;
  llm: Record<string, any>;
  exploration: Record<string, any>;
  knowledge: Record<string, any>;
  survival: Record<string, any>;
  goals: Record<string, any>;
};

export type Claim = {
  claim_id: string; claim_version: number; parent_claim_id: string | null;
  author: number; claim_type: string; state: string;
  spec: Record<string, any>; baseline: Record<string, any>;
  direction: string; min_delta: number;
  created_tick: number; registered_tick: number | null;
  decided_tick: number | null;
  replication_kind: string | null; replicator: number | null;
  verdicts: Record<string, any>;
};

export type World = {
  tick: number;
  agents: Record<string, [number, number]>;
  planted: Record<string, { agent_id: number; tick: number }>;
  channels: number[];
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

// The API refuses a bad task with a sentence explaining why -- "a directive
// only means something in a brief", say. Surfacing that sentence is the whole
// point of parsing the error body: a bare 422 would send the observer to the
// server log to find out what they did.
async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const j = await res.json();
      if (typeof j?.detail === "string") detail = j.detail;
      else if (Array.isArray(j?.detail)) detail = j.detail.map(
        (d: any) => d.msg ?? String(d)).join("; ");
    } catch { /* a non-JSON error body: the status is all there is */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  runs: () => get<RunSummary[]>("/runs"),
  run: (id: string) => get<any>(`/runs/${id}`),
  claims: (id: string) => get<Claim[]>(`/runs/${id}/claims`),
  world: (id: string, tick: number) => get<World>(`/runs/${id}/world?tick=${tick}`),
  metrics: (id: string) => get<any>(`/runs/${id}/metrics`),
  agents: (id: string) => get<any[]>(`/runs/${id}/agents`),
  replay: (id: string) => get<any>(`/runs/${id}/replay`),
  report: (id: string) => get<Report>(`/runs/${id}/report`),
  progress: (id: string) => get<Progress>(`/runs/${id}/progress`),

  taskMetrics: () => get<TaskMetric[]>("/tasks/metrics"),
  tasks: (id: string) => get<TaskBoard>(`/runs/${id}/tasks`),
  addTask: (id: string, t: TaskInput) =>
    send<TaskBoard>(`/runs/${id}/tasks`, "POST", t),
  deleteTask: (id: string, taskId: string) =>
    send<TaskBoard>(`/runs/${id}/tasks/${taskId}`, "DELETE"),

  briefs: () => get<Brief[]>("/briefs"),
  createBrief: (name: string) => send<Brief>("/briefs", "POST", { name }),
  deleteBrief: (id: string) => send<any>(`/briefs/${id}`, "DELETE"),
  addBriefTask: (id: string, t: TaskInput) =>
    send<any>(`/briefs/${id}/tasks`, "POST", t),
  deleteBriefTask: (id: string, taskId: string) =>
    send<any>(`/briefs/${id}/tasks/${taskId}`, "DELETE"),
};

// Claim lifecycle colours. Refuted is deliberately not red-as-failure: a
// refuted claim is the verification system working, not a mistake.
export const STATE_COLOUR: Record<string, string> = {
  proposed: "#8b8b8b",
  registered: "#7a8fa6",
  testing: "#c9a227",
  supported: "#4f9d69",
  confirmed: "#2f7d4f",
  generalized: "#1f6f4a",
  contested: "#c46a1f",
  refuted: "#8a6d9c",
  withdrawn: "#666666",
};
