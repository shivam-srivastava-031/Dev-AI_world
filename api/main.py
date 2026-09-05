"""FastAPI: a thin read layer over finished runs.

Deliberately thin. There is no simulation logic here and there must never be:
the engine has to stay runnable and testable with this process stopped, or the
science quietly acquires a dependency on a web server.

Every response passes through the same information boundary the agents face,
enforced by ``assert_no_leak``. A debugging endpoint that returned the tile
strata or true_mu would invalidate every experiment run after it was added --
so the guard is in the response path rather than in a code-review checklist.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from aiciv.information import assert_no_leak
from aiciv.persistence.replay import replay_actions
from aiciv.persistence.store import RunStore

RUNS_DIR = pathlib.Path("runs")

app = FastAPI(
    title="AI Civilization",
    description="Read-only observation of finished runs.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000"],
    allow_methods=["GET"], allow_headers=["*"],
)

#: Fields stripped before anything leaves the API. The tile stratum and true_mu
#: are METRICS_ONLY: a dashboard that displayed them would let a human observer
#: leak them back to the experiment through prompt design.
OBSERVER_ONLY = ("true_mu", "stratum", "signature")


def _store(run_id: str) -> RunStore:
    path = RUNS_DIR / f"{run_id}.sqlite"
    if not path.exists():
        raise HTTPException(404, f"no run {run_id!r} in {RUNS_DIR}/")
    return RunStore(path)


def _public(rows: list[dict[str, Any]], *, keep_ground_truth: bool
            ) -> list[dict[str, Any]]:
    """Strip observer-only fields unless the caller explicitly asked for the
    grading view -- which is a separate, clearly-labelled endpoint."""
    if keep_ground_truth:
        return rows
    return [{k: v for k, v in r.items() if k not in OBSERVER_ONLY} for r in rows]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "runs_dir": str(RUNS_DIR),
            "runs": len(list(RUNS_DIR.glob("*.sqlite"))) if RUNS_DIR.exists() else 0}


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for path in sorted(RUNS_DIR.glob("*.sqlite")):
        store = RunStore(path)
        out.extend(store.runs())
        store.close()
    return out


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    store = _store(run_id)
    try:
        meta = store.run(run_id)
        if meta is None:
            raise HTTPException(404, f"no run {run_id!r}")
        meta["manifest"] = json.loads(meta.pop("manifest_json") or "{}")
        meta["config"] = json.loads(meta.pop("config_json") or "{}")
        meta["storage"] = store.storage(run_id)
        return meta
    finally:
        store.close()


@app.get("/runs/{run_id}/world")
def get_world(run_id: str, tick: int = Query(0, ge=0)) -> dict[str, Any]:
    """The map at a tick, reconstructed from the event log.

    Positions come from move events rather than from a stored grid snapshot,
    which keeps the store small and the timeline scrubbable.
    """
    store = _store(run_id)
    try:
        meta = store.run(run_id)
        if meta is None:
            raise HTTPException(404, f"no run {run_id!r}")
        positions: dict[int, list[int]] = {}
        planted: dict[int, dict[str, Any]] = {}
        channels: list[int] = []
        for e in store.events(run_id, limit=200000):
            if e["tick"] > tick:
                break
            payload = json.loads(e["payload_json"] or "{}")
            if e["kind"] == "move":
                positions[e["agent_id"]] = [payload.get("x"), payload.get("y")]
            elif e["kind"] == "plant":
                planted[payload.get("tile_id")] = {"agent_id": e["agent_id"],
                                                   "tick": e["tick"]}
            elif e["kind"] == "harvest":
                planted.pop(payload.get("tile_id"), None)
            elif e["kind"] == "build_channel":
                channels.append(payload.get("tile_id"))
        payload = {"tick": tick, "agents": positions,
                   "planted": planted, "channels": sorted(set(channels))}
        assert_no_leak(payload, where="/world")
        return payload
    finally:
        store.close()


@app.get("/runs/{run_id}/agents")
def get_agents(run_id: str) -> list[dict[str, Any]]:
    store = _store(run_id)
    try:
        trials = store.trials(run_id, limit=100000)
        by_agent: dict[int, dict[str, Any]] = {}
        for t in trials:
            a = by_agent.setdefault(t["agent_id"], {
                "agent_id": t["agent_id"], "trials": 0, "total_yield": 0.0,
                "best_yield": 0.0})
            a["trials"] += 1
            a["total_yield"] = round(a["total_yield"] + t["yield_kg"], 3)
            a["best_yield"] = max(a["best_yield"], t["yield_kg"])
        claims = store.claims(run_id)
        for c in claims:
            a = by_agent.setdefault(c["author"], {
                "agent_id": c["author"], "trials": 0, "total_yield": 0.0,
                "best_yield": 0.0})
            a.setdefault("claims", 0)
            a["claims"] = a.get("claims", 0) + 1
        return [by_agent[k] for k in sorted(by_agent)]
    finally:
        store.close()


@app.get("/runs/{run_id}/claims")
def get_claims(run_id: str) -> list[dict[str, Any]]:
    """The claim board: what the civilization put on the record, with the
    evidence that decided each one."""
    store = _store(run_id)
    try:
        out = []
        for c in store.claims(run_id):
            c["spec"] = json.loads(c.pop("spec_json") or "{}")
            c["baseline"] = json.loads(c.pop("baseline_json") or "{}")
            c["verdicts"] = json.loads(c.pop("verdicts_json") or "{}")
            out.append(c)
        return out
    finally:
        store.close()


@app.get("/runs/{run_id}/trials")
def get_trials(run_id: str, limit: int = Query(2000, ge=1, le=50000),
               ground_truth: bool = Query(False)) -> list[dict[str, Any]]:
    """Trial evidence.

    ``ground_truth=true`` is the grading view and includes METRICS_ONLY fields.
    It is a separate flag rather than the default precisely so that showing an
    observer the hidden answer is a deliberate act.
    """
    store = _store(run_id)
    try:
        rows = store.trials(run_id, limit=limit)
        for r in rows:
            r["params"] = json.loads(r.pop("params_json") or "{}")
        return _public(rows, keep_ground_truth=ground_truth)
    finally:
        store.close()


@app.get("/runs/{run_id}/events")
def get_events(run_id: str, kind: str | None = None,
               limit: int = Query(500, ge=1, le=20000)) -> list[dict[str, Any]]:
    store = _store(run_id)
    try:
        rows = store.events(run_id, kind=kind, limit=limit)
        for r in rows:
            r["payload"] = json.loads(r.pop("payload_json") or "{}")
        return rows
    finally:
        store.close()


@app.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str) -> dict[str, Any]:
    store = _store(run_id)
    try:
        report = store.metrics(run_id)
        if report is None:
            raise HTTPException(404, "no metrics recorded for this run")
        return report
    finally:
        store.close()


@app.get("/runs/{run_id}/replay")
def verify_replay(run_id: str) -> dict[str, Any]:
    """Re-execute the recorded decisions and confirm the world agrees.

    Needs no model even for an LLM run: the decisions are data, and only the
    engine is under test.
    """
    store = _store(run_id)
    try:
        return replay_actions(store, run_id).to_dict()
    finally:
        store.close()
