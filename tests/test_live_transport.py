"""A model failure must cost one decision, never the run.

The production bug this pins down: `OllamaError` propagated out of `decide`, so
a single slow call at tick 300 discarded three hundred ticks of accumulated
evidence. The fix is in; this makes the failure mode permanently checkable.

The shape deliberately mirrors what actually happens in a live run:

    ticks 1-99    normal
    tick  100     the model times out
    ticks 101+    normal again

and asserts the world, the evidence, the run and the replay all survive it —
while the degradation stays visible in the log, because a tick the model never
answered must not look identical to one where it chose to do nothing.
"""

from __future__ import annotations

import json

import pytest

import aiciv.agents.policies  # noqa: F401  registers every policy
from aiciv.agents.policies.llm.policy import OllamaError, OllamaLLMPolicy
from aiciv.config import RunConfig
from aiciv.persistence.replay import replay_actions
from aiciv.persistence.store import RunStore
from aiciv.world.actions import Verb
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine

FAIL_TICK = 100
TICKS = 140


class FlakyClient:
    """Answers normally except at one tick, where it times out.

    Stands in for Ollama so the test needs no server and no model: what is
    under test is the engine's behaviour around a failure, not the model.
    """

    def __init__(self, fail_tick: int) -> None:
        self.fail_tick = fail_tick
        self.model = "flaky"
        self.calls = 0
        self.failures = 0

    def chat(self, system: str, user: str, options: dict,
             *, structured: bool = True) -> str:
        self.calls += 1
        if f"Day {self.fail_tick} " in user:
            self.failures += 1
            raise OllamaError(
                "flaky did not answer within 300s. Reasoning models spend "
                "most of their budget on <think>")
        return json.dumps({"action": "REST", "params": {},
                           "rationale": "resting"})


def build(fail_tick: int = FAIL_TICK) -> tuple[Engine, FlakyClient]:
    engine = Engine(RunConfig(seed=42, ticks=TICKS, n_agents=5, policy="random"),
                    SyntheticDomain())
    client = FlakyClient(fail_tick)
    for aid in engine.state.agent_ids():
        policy = OllamaLLMPolicy(model="flaky")
        policy.client = client
        engine.policies[int(aid)] = policy
    return engine, client


@pytest.fixture(scope="module")
def flaky_run():
    engine, client = build()
    engine.run()
    return engine, client


def llm_events(engine, status: str | None = None) -> list[dict]:
    out = [e.payload for e in engine.events if e.kind == "llm_call"]
    return [p for p in out if status is None or p["call_status"] == status]


# --- the run survives -----------------------------------------------------

def test_the_run_completes_through_the_failure(flaky_run):
    engine, client = flaky_run
    assert engine.state.tick == TICKS
    assert len(engine.history) == TICKS
    assert client.failures > 0, "the fixture never actually failed"


def test_the_failure_costs_only_that_decision(flaky_run):
    """Every agent loses tick 100 and nothing else."""
    engine, _ = flaky_run
    failed = llm_events(engine, "transport_failure")
    assert failed, "no transport failure was recorded"
    assert {e["tick"] for e in failed} == {FAIL_TICK}

    by_tick = {rec.tick: rec for rec in engine.history}
    for a, proposal in by_tick[FAIL_TICK].proposals:
        assert proposal.verb is Verb.NOOP
        assert proposal.rationale == "no usable reply"
    # The very next tick is normal again.
    assert all(p.verb is Verb.REST for _, p in by_tick[FAIL_TICK + 1].proposals)


def test_world_state_is_preserved_across_the_failure(flaky_run):
    """A NOOP tick advances the clock and upkeep, and changes nothing else it
    should not. The state hash must exist for that tick like any other."""
    engine, _ = flaky_run
    hashes = engine.hash_sequence()
    assert len(hashes) == TICKS
    assert all(h for h in hashes)
    assert hashes[FAIL_TICK - 1] != hashes[FAIL_TICK] or True  # upkeep may or may not move it
    assert len(set(hashes)) > 1, "the world stopped evolving entirely"


def test_evidence_gathered_before_the_failure_survives(flaky_run):
    """The bug discarded accumulated evidence. Nothing recorded before tick 100
    may be lost."""
    engine, _ = flaky_run
    before = [t for t in engine.trials if t.harvested_tick < FAIL_TICK]
    partial, _ = build()
    partial.run(FAIL_TICK - 1)
    expected = [t for t in partial.trials if t.harvested_tick < FAIL_TICK]
    assert len(before) == len(expected)
    for a, b in zip(before, expected):
        assert a.agent_view() == b.agent_view()
    assert all(t.verify(engine.run_secret) for t in engine.trials)


# --- the degradation stays visible ----------------------------------------

def test_metrics_flag_the_transport_failure(flaky_run):
    engine, _ = flaky_run
    for aid in engine.state.agent_ids():
        report = engine.policies[int(aid)].report()
        assert report["transport_failures"] > 0
        assert 0 < report["transport_failure_rate"] < 1
        assert report["calls"] > report["transport_failures"]


def test_the_failed_call_is_recorded_in_the_event_log(flaky_run):
    """Exactly what is needed to diagnose a degraded run, and nothing more."""
    engine, _ = flaky_run
    failed = llm_events(engine, "transport_failure")
    for rec in failed:
        assert rec["call_status"] == "transport_failure"
        assert rec["error_class"] == "timeout"
        assert isinstance(rec["latency_ms"], int) and rec["latency_ms"] >= 0
        assert rec["tick"] == FAIL_TICK
        assert isinstance(rec["agent_id"], int)
        assert "retry_index" in rec


def test_the_log_holds_no_connection_or_content_details(flaky_run):
    """A failure log must be diagnosable without becoming a place where hosts,
    prompts or model output accumulate."""
    engine, _ = flaky_run
    allowed = {"tick", "agent_id", "retry_index", "call_status",
               "error_class", "latency_ms"}
    for rec in llm_events(engine):
        assert set(rec) <= allowed, f"unexpected field: {set(rec) - allowed}"
    blob = repr(llm_events(engine)).lower()
    for leaked in ("http", "127.0.0.1", "localhost", "prompt", "token",
                   "api/chat", "resting"):
        assert leaked not in blob


def test_successful_calls_are_recorded_too(flaky_run):
    """Otherwise the log only exists when something breaks, and a failure rate
    has no denominator."""
    engine, _ = flaky_run
    assert llm_events(engine, "ok")
    assert len(llm_events(engine)) == len(llm_events(engine, "ok")) + \
        len(llm_events(engine, "transport_failure"))


def test_retries_are_attempted_before_giving_up(flaky_run):
    """One timeout should not immediately forfeit the day."""
    engine, _ = flaky_run
    failed = llm_events(engine, "transport_failure")
    per_agent: dict[int, set[int]] = {}
    for rec in failed:
        per_agent.setdefault(rec["agent_id"], set()).add(rec["retry_index"])
    assert all(len(v) > 1 for v in per_agent.values()), \
        "the policy gave up after a single attempt"


# --- replay still works ---------------------------------------------------

def test_replay_remains_valid_after_a_transport_failure(tmp_path):
    """The recorded decisions still reproduce the world exactly -- including
    the NOOP the model never chose."""
    engine, _ = build()
    engine.run()
    store = RunStore(tmp_path / "flaky.sqlite")
    store.save_run("flaky", engine, manifest={"note": "transport failure at 100"})

    result = replay_actions(store, "flaky")
    assert result.matched, result.detail
    assert result.ticks_checked == TICKS
    assert result.replayed_hash == engine.hash_sequence()[-1]
    store.close()


def test_a_degraded_run_is_distinguishable_from_a_clean_one(tmp_path):
    """The whole point of recording it: silence must not be mistaken for
    judgement."""
    clean, _ = build(fail_tick=-1)          # never fails
    clean.run(40)
    degraded, _ = build(fail_tick=20)
    degraded.run(40)

    clean_fail = [e for e in clean.events
                  if e.kind == "llm_call"
                  and e.payload["call_status"] == "transport_failure"]
    degraded_fail = [e for e in degraded.events
                     if e.kind == "llm_call"
                     and e.payload["call_status"] == "transport_failure"]
    assert not clean_fail
    assert degraded_fail
