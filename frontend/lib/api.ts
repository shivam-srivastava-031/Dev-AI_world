// Thin typed client. The dashboard is a read layer: it never posts, and it
// never computes anything the backend has already decided.

const BASE = process.env.NEXT_PUBLIC_API ?? "/api";

export type RunSummary = {
  run_id: string; arm: string; seed: number; ticks: number;
  agents: number; policy: string; domain: string;
  final_state_hash: string; started_at: string;
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

export const api = {
  runs: () => get<RunSummary[]>("/runs"),
  run: (id: string) => get<any>(`/runs/${id}`),
  claims: (id: string) => get<Claim[]>(`/runs/${id}/claims`),
  world: (id: string, tick: number) => get<World>(`/runs/${id}/world?tick=${tick}`),
  metrics: (id: string) => get<any>(`/runs/${id}/metrics`),
  agents: (id: string) => get<any[]>(`/runs/${id}/agents`),
  replay: (id: string) => get<any>(`/runs/${id}/replay`),
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
