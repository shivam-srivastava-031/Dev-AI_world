"""Phase 7 gate: prior probe, LLM policy, and replay without a network.

No test here contacts Ollama. That is the point of the call log: a run must be
replayable, and its analysis re-runnable, by someone who does not have the model.
"""

from __future__ import annotations

import json

import pytest

from aiciv.agents.policies.llm.cache import LLMCache, ReplayCacheMiss
from aiciv.agents.policies.llm.policy import ParseError, parse_action
from aiciv.agents.policies.llm.prompts import (
    BANNED_TERMS, SCAFFOLD_LEVELS, observation_block, system_prompt,
)
from aiciv.experiment.prior_probe import (
    build_prompts, credit_against_baseline, load, run_probe, save,
)
from aiciv.world.actions import Verb
from aiciv.world.domains.base import TileContext
from aiciv.world.domains.synthetic import SyntheticDomain

SPACE = {"spacing": (1, 2, 3, 4, 5), "water": (0, 1, 2, 3, 4),
         "companion": ("NONE", "CLOVER"), "plant_day": tuple(range(12))}


def rules_only_prompt() -> str:
    return system_prompt(name="Ada", width=20, height=20, param_space=SPACE,
                         schedule_axis="plant_day", others="#1",
                         scaffold="rules_only")


# --- the scaffold must not hand them the method --------------------------

def test_scaffold_rules_only_narrative_contains_no_method_terms():
    """The experiment's most sensitive surface.

    If "run controlled trials" appears in the prose, we have given the agents
    the scientific method and any later claim about them applying it is
    circular. Scanned on the NARRATIVE: the action signatures are a separate
    grant, documented by the test below rather than swept up here.
    """
    from aiciv.agents.policies.llm.prompts import narrative
    text = narrative(name="Ada", width=20, height=20, param_space=SPACE,
                     schedule_axis="plant_day", others="#1",
                     scaffold="rules_only").lower()
    found = [t for t in BANNED_TERMS if t in text]
    assert not found, f"rules_only narrative leaks method vocabulary: {found}"


def test_offering_the_protocol_grants_comparison_vocabulary():
    """An honest accounting of a prior we cannot avoid.

    An action called PROPOSE_CLAIM with a "baseline" parameter tells an agent
    that comparing against something is a thing one does. Offering the protocol
    therefore GRANTS that vocabulary. It is recorded in
    docs/agent_prior_knowledge.md, not hidden -- and withdrawing the protocol
    removes it entirely, which is the only way to measure what naming it costs.
    """
    kwargs = dict(name="Ada", width=20, height=20, param_space=SPACE,
                  schedule_axis="plant_day", others="#1", scaffold="rules_only")
    with_protocol = system_prompt(**kwargs, protocol_offered=True).lower()
    without = system_prompt(**kwargs, protocol_offered=False).lower()

    assert "baseline" in with_protocol       # the grant, stated plainly
    assert "baseline" not in without
    assert not [t for t in BANNED_TERMS if t in without], (
        "with the protocol withdrawn, NO method vocabulary may remain")


def test_rules_plus_method_is_the_contrast_arm():
    """It exists to quantify what handing them the method is worth, so it has
    to actually contain what rules_only withholds."""
    p = system_prompt(name="Ada", width=20, height=20, param_space=SPACE,
                      schedule_axis="plant_day", others="#1",
                      scaffold="rules_plus_method").lower()
    assert "one thing at a time" in p
    assert "compare" in p


def test_scaffold_levels_are_closed():
    assert SCAFFOLD_LEVELS == ("bare", "rules_only", "rules_plus_method")
    with pytest.raises(ValueError):
        system_prompt(name="A", width=1, height=1, param_space=SPACE,
                      schedule_axis="plant_day", others="", scaffold="nope")


def test_schedule_axis_is_not_offered_as_a_choice():
    assert "plant_day (one of" not in rules_only_prompt()


# --- prompts must be byte-identical for identical state ------------------

def test_prompt_is_deterministic():
    """If this varied, the call log would key on noise and replay would break."""
    first = rules_only_prompt()
    assert all(rules_only_prompt() == first for _ in range(50))


def test_observation_block_is_deterministic_and_leaks_nothing():
    import aiciv.agents.policies  # noqa: F401
    from aiciv.agents.observation import build_observation
    from aiciv.config import RunConfig
    from aiciv.world.engine import Engine

    e = Engine(RunConfig(seed=42, ticks=60, n_agents=5, policy="random"),
               SyntheticDomain())
    e.run()
    obs = build_observation(e.state, 0)
    first = observation_block(obs)
    assert all(observation_block(obs) == first for _ in range(20))
    for banned in ("stratum", "true_mu", "signature", "holdout"):
        assert banned not in first


# --- the parser is strict -------------------------------------------------

def test_parses_a_well_formed_action():
    p = parse_action(
        '{"action": "PLANT", "params": {"tile": [3, 4], "spacing": "2", '
        '"water": 1, "companion": "CLOVER"}, "rationale": "trying clover"}',
        SPACE)
    assert p.verb is Verb.PLANT
    assert p.params["tile"] == [3, 4]
    assert p.params["spacing"] == 2          # coerced to the declared type
    assert p.rationale


def test_strips_reasoning_traces():
    """Reasoning models emit <think> blocks. They are not the answer."""
    assert parse_action('<think>maybe clover</think>{"action": "REST"}',
                        SPACE).verb is Verb.REST


def test_extracts_json_from_surrounding_prose():
    assert parse_action('Sure!\n{"action": "EAT"}\nHope that helps.',
                        SPACE).verb is Verb.EAT


@pytest.mark.parametrize("bad", [
    "", "   ", "no json here",
    '{"action": "FLY"}',
    '{"action": "PLANT", "params": []}',
    '{"nope": 1}',
    '{"action": "PLANT", ',
])
def test_malformed_replies_raise_rather_than_guess(bad):
    """Guessing what a model meant would substitute our judgement for its
    behaviour, which is the thing being measured."""
    with pytest.raises(ParseError):
        parse_action(bad, SPACE)


# --- the call log is what makes a run replayable -------------------------

def test_replay_reads_the_log_and_never_the_network(tmp_path):
    path = tmp_path / "calls.sqlite"
    rec = LLMCache(path=path, mode="record", run_id="r1")
    opts = {"temperature": 0.7, "seed": 1}
    calls = {"n": 0}

    def fake(prompt):
        calls["n"] += 1
        return '{"action": "REST"}'

    for tick in range(5):
        rec.get_or_call(model="m", options=opts, prompt=f"p{tick}",
                        tick=tick, agent_id=0, retry_index=0, call=fake)
    assert calls["n"] == 5
    rec.close()

    def explode(prompt):
        raise AssertionError("replay reached the network")

    rep = LLMCache(path=path, mode="replay", run_id="r1")
    for tick in range(5):
        out = rep.get_or_call(model="m", options=opts, prompt=f"p{tick}",
                              tick=tick, agent_id=0, retry_index=0, call=explode)
        assert out == '{"action": "REST"}'
    assert rep.hits == 5
    rep.close()


def test_replay_cache_miss_raises_loudly(tmp_path):
    """A miss means the prompt changed, so the replay would be of a different
    experiment. Silently calling the model would produce a run nobody can
    reproduce."""
    rep = LLMCache(path=tmp_path / "c.sqlite", mode="replay")
    with pytest.raises(ReplayCacheMiss):
        rep.get_or_call(model="m", options={}, prompt="unseen", tick=0,
                        agent_id=0, retry_index=0, call=lambda p: "x")


def test_prompt_version_is_part_of_the_key(tmp_path):
    path = tmp_path / "c.sqlite"
    a = LLMCache(path=path, mode="record", prompt_version="p1")
    b = LLMCache(path=path, mode="record", prompt_version="p2")
    assert a.key("m", {}, "same") != b.key("m", {}, "same")


def test_model_and_options_are_part_of_the_key(tmp_path):
    c = LLMCache(path=tmp_path / "c.sqlite", mode="record")
    assert c.key("m1", {}, "p") != c.key("m2", {}, "p")
    assert c.key("m", {"seed": 1}, "p") != c.key("m", {"seed": 2}, "p")


def test_retries_are_distinct_cache_entries(tmp_path):
    cache = LLMCache(path=tmp_path / "c.sqlite", mode="record")
    seen = []
    cache.get_or_call(model="m", options={}, prompt="p", tick=0, agent_id=0,
                      retry_index=0, call=lambda p: seen.append(0) or "a")
    cache.get_or_call(model="m", options={}, prompt="p", tick=0, agent_id=0,
                      retry_index=1, call=lambda p: seen.append(1) or "b")
    assert len(seen) == 2


# --- the prior probe ------------------------------------------------------

def test_probe_detects_a_model_that_already_knows_the_answer():
    """The outcome this project must be able to report: the model named the
    optimum cold, so in-run 'discovery' of it is RECALL, not discovery."""
    d = SyntheticDomain()
    best, _ = d.true_optimum(TileContext(0, 2), 1.0)

    def omniscient(prompt: str) -> str:
        if "largest harvest" in prompt:
            choice = {k: v for k, v in best.items() if k != d.schedule_axis}
            return json.dumps({"choice": choice, "why": "clover fixes nitrogen"})
        if "Estimate the harvest" in prompt:
            return json.dumps({"estimates": [3.0] * 20})
        return ("I would change one thing at a time, hold the rest constant, "
                "and repeat each combination several times to average out the "
                "random variation from soil and weather.")

    base = run_probe(d, omniscient, model="fake")
    assert base.recall_ratio is not None and base.recall_ratio > 0.95
    assert base.controlled_comparison_is_latent

    credit = credit_against_baseline(base, {"banked": 3})
    assert credit["cold_named_optimum"]
    assert "RECALL" in credit["interpretation"]
    assert "APPLICATION" in credit["method_interpretation"]


def test_probe_detects_a_model_that_knows_nothing():
    d = SyntheticDomain()
    base = run_probe(d, lambda p: "no idea, I would plant things and see",
                     model="fake")
    assert base.recall_ratio is None          # nothing parseable was named
    assert not base.controlled_comparison_is_latent
    credit = credit_against_baseline(base, {"banked": 3})
    assert not credit["cold_named_optimum"]
    assert "did not volunteer" in credit["method_interpretation"]


def test_probe_scores_a_bad_guess_below_the_optimum():
    d = SyntheticDomain()

    def poor(prompt: str) -> str:
        if "largest harvest" in prompt:
            # A legal recipe, but the trap rather than the optimum.
            return json.dumps({"choice": {"companion": "THISTLE", "spacing": 1,
                                          "water": 0}})
        return "not sure"

    base = run_probe(d, poor, model="fake")
    assert base.recall_ratio is not None
    assert base.recall_ratio < 0.95
    assert not credit_against_baseline(base, {})["cold_named_optimum"]


def test_probe_prompts_do_not_lead_the_witness():
    """Task 3 asks how the model would proceed. If the question itself named
    controls or replication, a positive answer would prove nothing."""
    method = build_prompts(SyntheticDomain(), "rules_only")["method"].lower()
    for term in ("control", "replicate", "hypothesis", "one at a time",
                 "hold constant", "baseline"):
        assert term not in method


def test_probe_baseline_round_trips(tmp_path):
    d = SyntheticDomain()
    base = run_probe(d, lambda p: "nothing", model="fake")
    assert load(save(base, tmp_path)).probe_id == base.probe_id


def test_probe_id_is_stable_per_model_domain_and_scaffold():
    d = SyntheticDomain()
    a = run_probe(d, lambda p: "x", model="m1")
    b = run_probe(d, lambda p: "y", model="m1")
    c = run_probe(d, lambda p: "x", model="m2")
    assert a.probe_id == b.probe_id      # same configuration, same identity
    assert a.probe_id != c.probe_id      # different model, different baseline


@pytest.mark.parametrize("answer", [
    "I would change one thing at a time and hold the rest constant, repeating each a few times.",
    "Vary only one factor, keep everything else fixed, and repeat to average out luck.",
    "Try each combination several times and compare against a baseline.",
    "Controlled trials, replicated, versus a control.",
])
def test_method_detector_is_generous_across_phrasings(answer):
    """A MISSED marker understates latent knowledge, which inflates the credit
    given to emergence -- the one direction this project must not err in.

    An earlier fixed-substring list scored an obviously methodical answer as
    having no method at all, because it knew "one at a time" but not "one thing
    at a time".
    """
    d = SyntheticDomain()
    base = run_probe(d, lambda p: answer, model="fake")
    assert base.method_score > 0, f"no method detected in: {answer!r}"


def test_method_detector_does_not_fire_on_an_empty_answer():
    d = SyntheticDomain()
    base = run_probe(d, lambda p: "I would just plant things and hope.",
                     model="fake")
    assert not base.controlled_comparison_is_latent
