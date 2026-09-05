"""Phase 1 (deferred) and Phase 10: persistence, replay, and the API boundary."""

from __future__ import annotations

import json

import pytest

import aiciv.agents.policies  # noqa: F401  registers every policy
from aiciv.config import RunConfig
from aiciv.hashing import canonical_json, stable_json
from aiciv.persistence.replay import replay_actions
from aiciv.persistence.store import RunStore
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine

TICKS = 300


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    e = Engine(RunConfig(seed=42, ticks=TICKS, n_agents=5,
                         policy="scripted_factorial"), SyntheticDomain())
    e.run()
    store = RunStore(tmp / "r.sqlite")
    store.save_run("r1", e, manifest={"seed": 42})
    return store, e


# --- serialisation: the bug that broke replay ----------------------------

def test_canonical_json_is_lossy_and_stable_json_is_not():
    """The distinction that cost a silent replay divergence.

    canonical_json stringifies floats, which is exactly what makes a hash
    stable across BLAS versions -- and exactly what makes it wrong for anything
    read back. Persisting an action payload with it turned min_delta=0.3 into
    "0.300", so on replay the claim was rejected as having a non-positive delta
    and the world diverged 167 ticks in.
    """
    payload = {"min_delta": 0.3, "spacing": 4, "name": "clover"}

    lossy = json.loads(canonical_json(payload))
    assert lossy["min_delta"] == "0.300"          # a string, by design

    round_tripped = json.loads(stable_json(payload))
    assert round_tripped == payload
    assert isinstance(round_tripped["min_delta"], float)


def test_stable_json_is_still_deterministic():
    a = {"b": 1, "a": [3, 2], "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": [3, 2], "b": 1}
    assert stable_json(a) == stable_json(b)


# --- the store ------------------------------------------------------------

def test_run_metadata_round_trips(saved):
    store, engine = saved
    meta = store.run("r1")
    assert meta["seed"] == 42
    assert meta["ticks"] == TICKS
    assert meta["domain"] == "synthetic"
    assert meta["final_state_hash"] == engine.hash_sequence()[-1]

    cfg = json.loads(meta["config_json"])
    assert cfg["seed"] == 42 and cfg["policy"] == "scripted_factorial"


def test_every_action_is_logged(saved):
    store, engine = saved
    expected = sum(len(rec.proposals) for rec in engine.history)
    assert len(store.actions("r1")) == expected


def test_actions_are_ordered_by_tick_then_agent(saved):
    store, _ = saved
    rows = store.actions("r1")
    keys = [(r["tick"], r["seq_in_tick"]) for r in rows]
    assert keys == sorted(keys)


def test_rejections_are_recorded_with_their_code(saved):
    store, engine = saved
    rejected = [r for r in store.actions("r1") if not r["valid"]]
    expected = sum(len(rec.rejections) for rec in engine.history)
    assert len(rejected) == expected
    for r in rejected:
        assert r["rejection_code"]


def test_trials_and_signatures_survive_the_round_trip(saved):
    store, engine = saved
    rows = store.trials("r1", limit=100000)
    assert len(rows) == len(engine.trials)
    by_id = {int(t.trial_id): t for t in engine.trials}
    for r in rows:
        original = by_id[r["trial_id"]]
        assert r["signature"] == original.signature
        assert json.loads(r["params_json"]) == original.params


def test_storage_is_accounted_from_day_one(saved):
    """The 5 GB limit is not enforced in v0.1, but measuring it later would
    mean reconstructing history that was never recorded."""
    store, _ = saved
    acct = store.storage("r1")
    assert set(acct) == {"actions", "trials", "events", "claims"}
    assert all(v >= 0 for v in acct.values())


# --- replay ---------------------------------------------------------------

def test_replay_reproduces_every_state_hash(saved):
    """Proves the WORLD ENGINE is deterministic. Needs no model even for an LLM
    run: the decisions are data and only the engine is under test."""
    store, engine = saved
    result = replay_actions(store, "r1")
    assert result.matched, result.detail
    assert result.ticks_checked == TICKS
    assert result.replayed_hash == engine.hash_sequence()[-1]


def test_replay_reports_where_it_diverged_not_merely_that_it_did(tmp_path):
    """A corrupted log must name the tick. "Something diverged somewhere" is
    not a useful thing to tell anyone.

    Tampering happens on its OWN copy of the database. Mutating the shared
    fixture would leave a corrupted log behind for whichever tests happened to
    run afterwards -- an order-dependent failure that is miserable to track
    down and has nothing to do with the code under test.
    """
    e = Engine(RunConfig(seed=42, ticks=120, n_agents=5,
                         policy="scripted_factorial"), SyntheticDomain())
    e.run()
    store = RunStore(tmp_path / "tampered.sqlite")
    store.save_run("r1", e, manifest={})

    store.conn.execute(
        "UPDATE actions SET proposal_json = ? WHERE tick = 50 AND agent_id = 0",
        (stable_json({"verb": "REST", "params": {}, "message": "",
                      "rationale": "tampered"}),))
    store.conn.commit()

    result = replay_actions(store, "r1")
    assert not result.matched
    assert result.first_divergence is not None
    assert result.first_divergence >= 50
    assert "diverged at tick" in result.detail
    store.close()


def test_replay_of_a_missing_run_is_reported_not_raised(saved):
    store, _ = saved
    result = replay_actions(store, "nope")
    assert not result.matched and "no such run" in result.detail


# --- the API inherits the information boundary ---------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory, saved):
    from fastapi.testclient import TestClient

    import api.main as apimod

    store, engine = saved
    runs_dir = tmp_path_factory.mktemp("apiruns")
    api_store = RunStore(runs_dir / "demo.sqlite")
    api_store.save_run("demo", engine, manifest={"seed": 42},
                       metrics={"knowledge": {"banked": 0}})
    api_store.close()
    apimod.RUNS_DIR = runs_dir
    return TestClient(apimod.app)


def test_api_lists_and_describes_runs(client):
    assert client.get("/health").json()["ok"]
    assert [r["run_id"] for r in client.get("/runs").json()] == ["demo"]
    meta = client.get("/runs/demo").json()
    assert meta["seed"] == 42 and "storage" in meta


def test_api_hides_ground_truth_by_default(client):
    """A dashboard that displayed the strata or true_mu would let a human
    observer leak them back into the experiment through prompt design."""
    rows = client.get("/runs/demo/trials?limit=5").json()
    assert rows
    for r in rows:
        for banned in ("true_mu", "stratum", "signature"):
            assert banned not in r


def test_api_grading_view_is_an_explicit_opt_in(client):
    rows = client.get("/runs/demo/trials?limit=5&ground_truth=true").json()
    assert rows and "true_mu" in rows[0] and "stratum" in rows[0]


def test_api_world_view_leaks_nothing(client):
    world = client.get("/runs/demo/world?tick=100").json()
    flat = repr(world)
    for banned in ("stratum", "true_mu", "holdout", "signature"):
        assert banned not in flat


def test_api_can_verify_a_replay(client):
    assert client.get("/runs/demo/replay").json()["matched"]


def test_api_404s_on_an_unknown_run(client):
    assert client.get("/runs/ghost").status_code == 404


def test_api_contains_no_simulation_logic():
    """The engine must stay runnable with this process stopped, or the science
    quietly acquires a dependency on a web server."""
    import ast
    import pathlib

    src = pathlib.Path("api/main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = [m for m in imported
                 if m.startswith(("aiciv.world.engine", "aiciv.world.executor",
                                  "aiciv.agents.policies"))]
    assert not forbidden, f"api imports simulation machinery: {forbidden}"


def test_resaving_a_run_id_replaces_it_completely(tmp_path):
    """A re-saved run must not inherit its predecessor's log.

    Only the `runs` row was being cleared, so saving a shorter run over a
    longer one left the old ticks and actions behind: a 900-tick run then
    replayed as 1400 ticks against a log that was half someone else's. Silent,
    and exactly the kind of thing that makes a reproducibility claim worthless.
    """
    store = RunStore(tmp_path / "r.sqlite")

    long_run = Engine(RunConfig(seed=42, ticks=200, n_agents=5,
                                policy="scripted_factorial"), SyntheticDomain())
    long_run.run()
    store.save_run("same_id", long_run, manifest={})
    assert len(store.state_hashes("same_id")) == 200

    short_run = Engine(RunConfig(seed=42, ticks=80, n_agents=5,
                                 policy="scripted_factorial"), SyntheticDomain())
    short_run.run()
    store.save_run("same_id", short_run, manifest={})

    assert len(store.state_hashes("same_id")) == 80
    assert store.run("same_id")["ticks"] == 80
    assert len(store.actions("same_id")) == sum(
        len(rec.proposals) for rec in short_run.history)

    result = replay_actions(store, "same_id")
    assert result.matched and result.ticks_checked == 80
    store.close()
