"""Run persistence: one SQLite file per run, and exact replay from it.

What makes replay possible is a short list, and nothing outside it is needed:

  1. the seed and the config hash -- world generation is a pure function of them
  2. the ordered action log -- the complete decision stream
  3. the LLM call log, when there was one -- the complete nondeterminism stream
  4. no sequential RNG state exists at all, because every draw is key-derived

That fourth point is why a checkpoint can be loaded into a fresh process with
zero hidden state. There is no generator to restore.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from dataclasses import asdict
from typing import Any

from ..hashing import canonical_json, stable_json

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, arm TEXT, seed INTEGER NOT NULL,
  ticks INTEGER, agents INTEGER, policy TEXT, domain TEXT, domain_version TEXT,
  config_json TEXT, config_hash TEXT, manifest_json TEXT,
  final_state_hash TEXT, started_at TEXT, status TEXT
);

CREATE TABLE IF NOT EXISTS ticks (
  run_id TEXT, tick INTEGER, state_hash TEXT,
  PRIMARY KEY (run_id, tick)
);

CREATE TABLE IF NOT EXISTS actions (
  action_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT, tick INTEGER, seq_in_tick INTEGER, agent_id INTEGER,
  proposal_json TEXT NOT NULL,
  valid INTEGER, rejection_code TEXT, rejection_detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_actions_tick ON actions(run_id, tick, seq_in_tick);

CREATE TABLE IF NOT EXISTS trials (
  run_id TEXT, trial_id INTEGER, agent_id INTEGER, tile_id INTEGER,
  planted_tick INTEGER, harvested_tick INTEGER, params_json TEXT,
  soil_band INTEGER, skill_at_plant REAL, yield_kg REAL, crop_health TEXT,
  stratum TEXT, true_mu REAL, signature TEXT,
  PRIMARY KEY (run_id, trial_id)
);

CREATE TABLE IF NOT EXISTS claims (
  run_id TEXT, claim_id TEXT, claim_version INTEGER, parent_claim_id TEXT,
  author INTEGER, claim_type TEXT, state TEXT, spec_json TEXT,
  baseline_json TEXT, direction TEXT, min_delta REAL,
  created_tick INTEGER, registered_tick INTEGER, decided_tick INTEGER,
  replication_kind TEXT, replicator INTEGER, verdicts_json TEXT,
  PRIMARY KEY (run_id, claim_id)
);

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT, tick INTEGER, kind TEXT, agent_id INTEGER, payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_tick ON events(run_id, tick);

CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT PRIMARY KEY, report_json TEXT
);

CREATE TABLE IF NOT EXISTS storage_accounting (
  run_id TEXT, category TEXT, bytes INTEGER,
  PRIMARY KEY (run_id, category)
);
"""


class RunStore:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- writing ------------------------------------------------------------

    def save_run(self, run_id: str, engine, *, manifest: dict[str, Any],
                 metrics: dict[str, Any] | None = None) -> None:
        cfg = engine.config
        hashes = engine.hash_sequence()
        c = self.conn

        # Clear EVERY table for this run_id, not just `runs`. Re-saving the
        # same id previously left the old ticks and actions behind, so a
        # 900-tick run replayed as 1400 ticks against a log that was half
        # someone else's. Silent, and exactly the kind of thing that makes a
        # reproducibility claim worthless.
        for table in ("runs", "ticks", "actions", "trials", "claims",
                      "events", "metrics", "storage_accounting"):
            c.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        c.execute(
            "INSERT INTO runs (run_id, arm, seed, ticks, agents, policy, domain,"
            " domain_version, config_json, config_hash, manifest_json,"
            " final_state_hash, started_at, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),'complete')",
            (run_id, cfg.arm, cfg.seed, cfg.ticks, cfg.n_agents, cfg.policy,
             engine.domain.name, engine.domain.version,
             stable_json(asdict(cfg)), cfg.hash,
             json.dumps(manifest, sort_keys=True, default=str),
             hashes[-1] if hashes else None))

        c.executemany(
            "INSERT OR REPLACE INTO ticks (run_id, tick, state_hash) VALUES (?,?,?)",
            [(run_id, rec.tick, rec.state_hash) for rec in engine.history])

        rows = []
        for rec in engine.history:
            rejected = {a: r for a, r in rec.rejections}
            for seq, (agent, proposal) in enumerate(rec.proposals):
                rej = rejected.get(agent)
                rows.append((
                    run_id, rec.tick, seq, agent,
                    # stable_json, not canonical_json: this payload is read
                    # back by replay, and canonical form loses float types.
                    stable_json({"verb": proposal.verb.value,
                                 "params": proposal.params,
                                 "message": proposal.message,
                                 "rationale": proposal.rationale}),
                    0 if rej else 1,
                    rej.code.value if rej else None,
                    rej.detail if rej else None))
        c.executemany(
            "INSERT INTO actions (run_id, tick, seq_in_tick, agent_id,"
            " proposal_json, valid, rejection_code, rejection_detail)"
            " VALUES (?,?,?,?,?,?,?,?)", rows)

        c.executemany(
            "INSERT OR REPLACE INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run_id, int(t.trial_id), int(t.agent_id), int(t.tile_id),
              t.planted_tick, t.harvested_tick, stable_json(t.params),
              t.soil_band, t.skill_at_plant, t.yield_kg, t.crop_health,
              t.stratum, t.true_mu, t.signature) for t in engine.trials])

        c.executemany(
            "INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run_id, str(r.claim.claim_id), r.claim.claim_version,
              str(r.claim.parent_claim_id) if r.claim.parent_claim_id else None,
              int(r.claim.author), r.claim.claim_type.value, r.state.value,
              stable_json(r.claim.spec.conditions),
              stable_json(r.claim.baseline.conditions),
              r.claim.predicted_direction, r.claim.predicted_min_delta,
              r.claim.created_tick, r.registered_tick, r.decided_tick,
              r.replication_kind, r.replicator,
              json.dumps(r.verdicts, sort_keys=True, default=str))
             for r in engine.kb])

        c.executemany(
            "INSERT INTO events (run_id, tick, kind, agent_id, payload_json)"
            " VALUES (?,?,?,?,?)",
            [(run_id, e.tick, e.kind, e.agent_id, stable_json(e.payload))
             for e in engine.events])

        if metrics is not None:
            c.execute("INSERT OR REPLACE INTO metrics VALUES (?,?)",
                      (run_id, json.dumps(metrics, sort_keys=True, default=str)))

        self._account(run_id, engine)
        c.commit()

    def _account(self, run_id: str, engine) -> None:
        """Storage accounting, instrumented from day one.

        The 5 GB world limit is not enforced in v0.1, but measuring it later
        would mean reconstructing history that was never recorded.
        """
        cats = {
            "actions": sum(len(rec.proposals) for rec in engine.history) * 200,
            "trials": len(engine.trials) * 260,
            "events": len(engine.events) * 120,
            "claims": len(engine.kb) * 900,
        }
        self.conn.executemany(
            "INSERT OR REPLACE INTO storage_accounting VALUES (?,?,?)",
            [(run_id, k, v) for k, v in sorted(cats.items())])

    # -- reading ------------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT run_id, arm, seed, ticks, agents, policy, domain,"
            " final_state_hash, started_at FROM runs ORDER BY started_at DESC")]

    def run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?",
                                (run_id,)).fetchone()
        return dict(row) if row else None

    def actions(self, run_id: str, tick: int | None = None) -> list[dict[str, Any]]:
        if tick is None:
            cur = self.conn.execute(
                "SELECT * FROM actions WHERE run_id = ? ORDER BY tick, seq_in_tick",
                (run_id,))
        else:
            cur = self.conn.execute(
                "SELECT * FROM actions WHERE run_id = ? AND tick = ?"
                " ORDER BY seq_in_tick", (run_id, tick))
        return [dict(r) for r in cur]

    def trials(self, run_id: str, limit: int = 5000) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM trials WHERE run_id = ? ORDER BY trial_id LIMIT ?",
            (run_id, limit))]

    def claims(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM claims WHERE run_id = ? ORDER BY claim_id", (run_id,))]

    def events(self, run_id: str, kind: str | None = None,
               limit: int = 2000) -> list[dict[str, Any]]:
        if kind:
            cur = self.conn.execute(
                "SELECT * FROM events WHERE run_id = ? AND kind = ?"
                " ORDER BY event_id LIMIT ?", (run_id, kind, limit))
        else:
            cur = self.conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY event_id LIMIT ?",
                (run_id, limit))
        return [dict(r) for r in cur]

    def state_hashes(self, run_id: str) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT state_hash FROM ticks WHERE run_id = ? ORDER BY tick",
            (run_id,))]

    def metrics(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT report_json FROM metrics WHERE run_id = ?",
                                (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def storage(self, run_id: str) -> dict[str, int]:
        return {r[0]: r[1] for r in self.conn.execute(
            "SELECT category, bytes FROM storage_accounting WHERE run_id = ?",
            (run_id,))}

    def close(self) -> None:
        self.conn.close()
