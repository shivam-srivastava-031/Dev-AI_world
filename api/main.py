"""FastAPI: a thin read layer over finished runs.

Deliberately thin. There is no simulation logic here and there must never be:
the engine has to stay runnable and testable with this process stopped, or the
science quietly acquires a dependency on a web server.

Every response passes through the same information boundary the agents face,
enforced by ``assert_no_leak``. A debugging endpoint that returned the tile
strata or true_mu would invalidate every experiment run after it was added --
so the guard is in the response path rather than in a code-review checklist.

One exception to "read layer", and it is narrow on purpose: the task board.
An observer sets tasks here, and those writes go to ``tasks/board.json`` --
never to a run's SQLite file, which is still opened read-only everywhere and
still has no writer but the engine. A task cannot alter a run; the only way an
observer reaches the agents is a brief, which is fixed before the run starts,
recorded in that run's manifest, and marked steered wherever it is shown.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aiciv.experiment.report import report as build_report
from aiciv.information import assert_no_leak, scan_banned
from aiciv.persistence.replay import replay_actions
from aiciv.persistence.store import RunStore
from aiciv.tasks.board import DEFAULT_BOARD, BoardError, TaskBoard, new_id
from aiciv.tasks.spec import COMPARATORS, Task, catalogue, score_board

RUNS_DIR = pathlib.Path("runs")
BOARD_PATH = DEFAULT_BOARD

app = FastAPI(
    title="AI Civilization",
    description="Read-only observation of finished runs.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000"],
    # POST and DELETE reach the task board and nothing else. Run data has no
    # write route at all, from any method.
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
)

#: Fields stripped before anything leaves the API. The tile stratum and true_mu
#: are METRICS_ONLY: a dashboard that displayed them would let a human observer
#: leak them back to the experiment through prompt design.
OBSERVER_ONLY = ("true_mu", "stratum", "signature")


def _board() -> TaskBoard:
    """Loaded per request rather than held open.

    The board is a small JSON file that a person edits through a form; reading
    it fresh each time costs nothing and means the API never serves a stale
    copy after the file was changed by ``aiciv run --brief`` or by hand.
    """
    try:
        return TaskBoard(BOARD_PATH)
    except BoardError as e:
        raise HTTPException(500, str(e)) from e


def _store(run_id: str) -> RunStore:
    path = RUNS_DIR / f"{run_id}.sqlite"
    if not path.exists():
        raise HTTPException(404, f"no run {run_id!r} in {RUNS_DIR}/")
    # Read-only, always. A run measured in hours is worth watching while it
    # happens, and the observer must never take a write lock on the file the
    # engine is still writing to. This also makes "no simulation logic here"
    # structural rather than a convention.
    return RunStore(path, read_only=True)


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
        store = RunStore(path, read_only=True)
        try:
            out.extend(store.runs())
        except sqlite3.DatabaseError:
            # A run whose first checkpoint has not landed yet is a file with
            # no tables. Skipping it beats failing the whole listing.
            pass
        finally:
            store.close()
    # Runs live in separate files, so per-file ordering says nothing once they
    # are pooled. Newest first, then unfinished runs to the top: the run worth
    # watching is the one still moving. Two stable sorts rather than one key,
    # because the date is a string and cannot be reversed inside a tuple.
    out.sort(key=lambda r: r["started_at"] or "", reverse=True)
    out.sort(key=lambda r: r["ticks_recorded"] >= (r["ticks"] or 0))
    # Steering, carried on the listing itself. A steered run has to be
    # identifiable before it is opened -- the picker is where runs get compared
    # against each other, and that is exactly where the difference matters.
    board = _board()
    for r in out:
        r["steered"] = board.is_steered(r["run_id"])
        r["tasks"] = len(board.run_tasks(r["run_id"]))
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


@app.get("/runs/{run_id}/report")
def get_report(run_id: str, ground_truth: bool = Query(False)) -> dict[str, Any]:
    """The run report, readable from a partial checkpoint.

    Separate from /metrics: that one returns the full civilization report,
    which only exists once a run has finished. This is computed from whatever
    has been written so far, and labels itself PARTIAL when that is what it is,
    so a run in its twentieth hour can be watched rather than guessed at.

    The report is written for a terminal operator, who is allowed the grading
    view. The API is the observer boundary, so the stratum breakdown comes out
    unless it is asked for deliberately -- the same rule /trials follows.
    """
    store = _store(run_id)
    try:
        out = build_report(store, run_id)
        if "error" in out:
            raise HTTPException(404, out["error"])
        if not ground_truth:
            out["exploration"].pop("trials_by_stratum", None)
            assert_no_leak(out, where="/report")
        return out
    finally:
        store.close()


@app.get("/runs/{run_id}/progress")
def get_progress(run_id: str) -> dict[str, Any]:
    """Cheap enough to poll: just how far along the run is.

    The dashboard polls this on a timer and only refetches the expensive views
    when the tick count has actually moved. At four minutes a tick, polling the
    full report instead would be almost entirely wasted work.
    """
    store = _store(run_id)
    try:
        meta = store.run(run_id)
        if meta is None:
            raise HTTPException(404, f"no run {run_id!r}")
        recorded = len(store.state_hashes(run_id))
        return {
            "run_id": run_id,
            "ticks_recorded": recorded,
            "ticks_requested": meta["ticks"],
            "status": meta["status"],
            # Derived from the data rather than from the status column: a run
            # killed between checkpoints leaves the column stale, and a
            # checkpoint written by an older build says 'complete' regardless.
            "live": recorded < (meta["ticks"] or 0),
            "trials": len(store.trials(run_id, limit=200000)),
            "claims": len(store.claims(run_id)),
        }
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


# --- the task board --------------------------------------------------------
#
# The only write surface in this file. Everything below writes to
# tasks/board.json and nothing else; the run stores stay read-only.


class TaskIn(BaseModel):
    """A task as the dashboard sends it.

    ``source`` is NOT accepted from the client. Whether the agents were told
    about a task is decided by which endpoint it arrived at -- a brief, before
    the run, or a run, after it started -- and never by a field a caller could
    set. A task that lied about that would silently turn a steered run into an
    apparently free one.
    """

    title: str = Field(min_length=1, max_length=120)
    metric: str
    comparator: str
    target: float = Field(ge=0)
    #: Only meaningful on a brief: the sentence the agents actually read.
    directive: str = Field(default="", max_length=400)


class BriefIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


def _task_from(body: TaskIn) -> Task:
    if body.comparator not in COMPARATORS:
        raise HTTPException(422, f"comparator must be one of {COMPARATORS}")
    try:
        return Task(task_id=new_id("tsk"), title=body.title.strip(),
                    metric=body.metric, comparator=body.comparator,
                    target=float(body.target),
                    directive=body.directive.strip())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/tasks/metrics")
def task_metrics() -> list[dict[str, Any]]:
    """The measurable quantities a task may be built from.

    Everything here is readable from a partial checkpoint. Nothing that only
    exists in the end-of-run report is offered, because a task that cannot be
    scored until the run ends cannot be monitored while it runs.
    """
    return catalogue()


@app.get("/runs/{run_id}/tasks")
def get_tasks(run_id: str) -> dict[str, Any]:
    """The run's board, scored against the latest checkpoint."""
    store = _store(run_id)
    try:
        out = score_board(_board().run_tasks(run_id), store, run_id)
        assert_no_leak(out, where="/tasks")
        return out
    finally:
        store.close()


@app.post("/runs/{run_id}/tasks", status_code=201)
def add_task(run_id: str, body: TaskIn) -> dict[str, Any]:
    """Set an observer task on a run.

    The run is already under way or already finished, so nothing added here can
    reach the agents -- which is exactly why this is safe to do on a live run
    and why a directive is ignored on this route.
    """
    _store(run_id).close()          # 404 for a run that does not exist
    board = _board()
    task = _task_from(body)
    if task.directive:
        raise HTTPException(
            422, "a directive only means something in a brief, which is fixed "
                 "before the run starts. This run is already going, so its "
                 "agents cannot be told anything now.")
    try:
        board.add_run_task(run_id, task)
    except BoardError as e:
        raise HTTPException(400, str(e)) from e
    return get_tasks(run_id)


@app.delete("/runs/{run_id}/tasks/{task_id}")
def delete_task(run_id: str, task_id: str) -> dict[str, Any]:
    try:
        _board().remove_run_task(run_id, task_id)
    except BoardError as e:
        raise HTTPException(404, str(e)) from e
    return get_tasks(run_id)


@app.get("/briefs")
def list_briefs() -> list[dict[str, Any]]:
    """Briefs: task sets written to be put in front of the agents.

    A brief with a non-empty ``launched_runs`` is closed. It is the record of
    what those agents were told, so it cannot be edited afterwards -- an edit
    would leave the run's manifest describing a brief that no longer exists in
    that form.
    """
    board = _board()
    out = []
    for brief in board.briefs():
        directives = board.directives(brief["brief_id"])
        out.append({
            **brief,
            "directives": list(directives),
            "closed": bool(brief["launched_runs"]),
            "command": (f"aiciv run --policy llm --brief {brief['brief_id']}"),
            # Reported, never blocked. What an observer tells the agents is
            # theirs to decide; whether it hands them method vocabulary the
            # rules_only scaffold withholds is something they should be able
            # to see before they spend twenty hours on the run.
            "method_terms": sorted({t for d in directives
                                    for t in scan_banned(d)}),
        })
    return out


@app.post("/briefs", status_code=201)
def create_brief(body: BriefIn) -> dict[str, Any]:
    try:
        return _board().create_brief(body.name)
    except BoardError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/briefs/{brief_id}")
def delete_brief(brief_id: str) -> dict[str, Any]:
    board = _board()
    try:
        if board.brief(brief_id)["launched_runs"]:
            raise HTTPException(
                409, "this brief has been used by a run and is the record of "
                     "what those agents were told; it cannot be deleted")
        board.delete_brief(brief_id)
    except BoardError as e:
        raise HTTPException(404, str(e)) from e
    return {"deleted": brief_id}


@app.post("/briefs/{brief_id}/tasks", status_code=201)
def add_brief_task(brief_id: str, body: TaskIn) -> dict[str, Any]:
    """Add a task to a brief, and with it the sentence the agents will read.

    A task with no directive is still allowed: the observer may want the target
    scored without saying it out loud. That is the one case where a run is
    steered on some tasks and not others, and the brief lists both so the
    difference stays visible.
    """
    board = _board()
    try:
        board.add_brief_task(brief_id, _task_from(body))
    except BoardError as e:
        raise HTTPException(409, str(e)) from e
    return {"brief": board.brief(brief_id),
            "directives": list(board.directives(brief_id))}


@app.delete("/briefs/{brief_id}/tasks/{task_id}")
def delete_brief_task(brief_id: str, task_id: str) -> dict[str, Any]:
    board = _board()
    try:
        board.remove_brief_task(brief_id, task_id)
    except BoardError as e:
        raise HTTPException(409, str(e)) from e
    return {"brief": board.brief(brief_id),
            "directives": list(board.directives(brief_id))}
