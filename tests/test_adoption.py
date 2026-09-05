"""Protocol adoption must be graded on the agent's information state.

The failure this guards against: dividing registered claims by something the
agent had no access to, so that an agent which never gathered comparable
evidence is scored as having "failed to adopt the scientific protocol" when it
simply never had a formulable comparison in front of it.
"""

from __future__ import annotations

import pytest

from aiciv.knowledge.claim import ClaimType, KnowledgeClaim, TechniqueSpec
from aiciv.ids import AgentId
from aiciv.metrics.adoption import (
    MIN_PER_ARM, VISIBLE_FIELDS, adoption_for_agent, adoption_report,
    claim_contrasts, opportunities, project,
)


def rows(companion: str, n: int, yields, *, agent=0, start=0, tid_from=1):
    return [
        {
            "trial_id": tid_from + i, "agent_id": agent, "tile_id": 10 + (i % 4),
            "planted_tick": start + i, "harvested_tick": start + i + 6,
            "spacing": 2, "plant_day": 3, "water": 2, "companion": companion,
            "soil_band": 2, "skill_at_plant": 0.2,
            "yield_kg": float(yields[i % len(yields)]), "crop_health": "healthy",
            # Fields the agent must never be scored on:
            "stratum": "holdout", "true_mu": 99.0, "signature": "deadbeef",
        }
        for i in range(n)
    ]


def a_claim(var, hi, lo, author=0):
    return KnowledgeClaim(
        claim_id=f"clm_{hi}_{lo}", claim_version=1, author=AgentId(author),
        claim_type=ClaimType.COMPARISON,
        spec=TechniqueSpec({var: {"op": "eq", "value": hi}}),
        baseline=TechniqueSpec({var: {"op": "eq", "value": lo}}),
        predicted_direction="increase", predicted_min_delta=0.3, created_tick=1)


# --- rule 1: only agent-visible information ------------------------------

def test_projection_strips_everything_hidden():
    r = project(rows("CLOVER", 1, [3.0]))[0]
    assert set(r) <= VISIBLE_FIELDS
    for banned in ("stratum", "true_mu", "signature"):
        assert banned not in r


def test_result_is_identical_with_and_without_hidden_fields():
    """The strongest form of the guarantee: hidden fields cannot influence the
    answer, because removing them changes nothing."""
    full = rows("CLOVER", 10, [4.0, 4.2, 3.8]) + \
        rows("NONE", 10, [3.0, 3.1, 2.9], tid_from=100)
    stripped = project(full)
    assert [o.key() for o in opportunities(full)] == \
           [o.key() for o in opportunities(stripped)]
    a = adoption_for_agent(full, [])
    b = adoption_for_agent(stripped, [])
    assert a.to_dict() == b.to_dict()


# --- rule 2: no evidence is not a failure --------------------------------

def test_agent_with_no_evidence_is_undefined_not_zero():
    """The central fairness point. Scoring this agent 0.0 would count 'never
    had the chance' as 'refused to do science'."""
    r = adoption_for_agent(rows("CLOVER", 2, [3.0]), [])
    assert r.raw_opportunities == 0
    assert r.salient_opportunities == 0
    assert r.formal_protocol_adoption is None
    assert r.matched_adoption is None


def test_agent_with_one_arm_only_has_no_opportunity():
    """Fifty trials of one condition and none of another is not a comparison."""
    r = adoption_for_agent(rows("CLOVER", 50, [4.0]), [])
    assert r.raw_opportunities == 0
    assert r.matched_adoption is None


def test_aggregate_excludes_agents_without_opportunity():
    have = rows("CLOVER", 10, [4.0, 4.3, 3.9]) + \
        rows("NONE", 10, [3.0, 2.8, 3.1], tid_from=100)
    havent = rows("CLOVER", 3, [4.0], agent=1, tid_from=200)
    rep = adoption_report({0: have, 1: havent}, {0: [], 1: []})
    assert rep["agents_with_opportunity"] == 1
    assert rep["agents_without_opportunity"] == 1
    # agent 1 must not drag the mean down to 0
    assert rep["per_agent"][1]["matched_adoption"] is None


# --- the denominator itself ----------------------------------------------

def test_opportunity_needs_enough_evidence_in_both_arms():
    just_under = rows("CLOVER", MIN_PER_ARM, [4.0]) + \
        rows("NONE", MIN_PER_ARM - 1, [3.0], tid_from=100)
    assert opportunities(just_under) == []

    enough = rows("CLOVER", MIN_PER_ARM, [4.0, 4.2]) + \
        rows("NONE", MIN_PER_ARM, [3.0, 3.2], tid_from=100)
    assert len(opportunities(enough)) == 1


def test_opportunity_records_when_it_arose():
    """An agent cannot be faulted for not registering a claim before it had the
    evidence, so the tick the opportunity appeared is part of the record."""
    o = opportunities(
        rows("CLOVER", 8, [4.0, 4.2], start=0) +
        rows("NONE", 8, [3.0, 3.1], start=100, tid_from=100))[0]
    assert o.arose_tick >= 100


def test_opportunity_is_oriented_by_the_agents_own_data():
    o = opportunities(
        rows("CLOVER", 8, [4.0, 4.2]) +
        rows("NONE", 8, [3.0, 3.1], tid_from=100))[0]
    assert (o.level_a, o.level_b) == ("CLOVER", "NONE")
    assert o.observed_delta > 0


def test_salience_uses_the_agents_observed_effect_not_ground_truth():
    """A contrast the agent could form but which looked like nothing to it is
    a weaker expectation than one that looked substantial."""
    flat = rows("CLOVER", 10, [3.0, 3.05, 2.95]) + \
        rows("NONE", 10, [3.0, 3.02, 2.98], tid_from=100)
    assert opportunities(flat)
    assert not any(o.salient for o in opportunities(flat))

    strong = rows("CLOVER", 10, [4.5, 4.6, 4.4]) + \
        rows("NONE", 10, [3.0, 3.1, 2.9], tid_from=100)
    assert all(o.salient for o in opportunities(strong))


# --- matching claims to opportunities ------------------------------------

def test_claim_contrasts_extracts_the_pair():
    assert claim_contrasts([a_claim("companion", "CLOVER", "NONE")]) == \
        {("companion", "CLOVER", "NONE")}


def test_matched_adoption_credits_only_the_contrast_actually_claimed():
    ev = rows("CLOVER", 10, [4.5, 4.6, 4.4]) + \
        rows("NONE", 10, [3.0, 3.1, 2.9], tid_from=100)

    none_registered = adoption_for_agent(ev, [])
    assert none_registered.salient_opportunities == 1
    assert none_registered.matched_adoption == 0.0     # had the chance, did not act

    right = adoption_for_agent(ev, [a_claim("companion", "CLOVER", "NONE")])
    assert right.matched_adoption == 1.0

    # A claim about something else does not count as addressing this one.
    wrong = adoption_for_agent(ev, [a_claim("spacing", 4, 2)])
    assert wrong.registered_claims == 1
    assert wrong.matched_adoption == 0.0


def test_formal_and_matched_rates_are_different_questions():
    """formal = did it register anything at all, per salient chance.
    matched = did it register a claim about THAT chance."""
    ev = rows("CLOVER", 10, [4.5, 4.6]) + rows("NONE", 10, [3.0, 3.1], tid_from=100)
    r = adoption_for_agent(ev, [a_claim("spacing", 4, 2)])
    assert r.formal_protocol_adoption == 1.0    # it did do science
    assert r.matched_adoption == 0.0            # just not about this contrast
