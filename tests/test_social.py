"""Phase 5 gate: language, teaching, trust, archetypes, civilization metrics."""

from __future__ import annotations

import pytest

import aiciv.agents.policies  # noqa: F401  registers every policy
from aiciv.civilization.specialization import analyse as spec_analyse
from aiciv.civilization.trust import BeliefStatus, Beliefs, Relationships
from aiciv.config import RunConfig
from aiciv.language.gate import LanguageGate, NoveltyLog, Verdict
from aiciv.metrics.definitions import (
    civilization_report, controllable_optimum, time_to_competence,
)
from aiciv.world.actions import RejectionCode
from aiciv.world.domains.synthetic import SyntheticDomain
from aiciv.world.engine import Engine


@pytest.fixture(scope="module")
def gate():
    return LanguageGate()


# =====================================================================
# The language bound: on LANGUAGE, not on vocabulary
# =====================================================================

@pytest.mark.parametrize("msg", [
    "clover4 beats none4 in my plots so far",
    "try water+nitrogen on the dry field",
    "spacing2 gave more than spacing5 for me",
])
def test_coined_shorthand_passes_and_is_recorded(gate, msg):
    """THE regression guard. A binary gate would reject `clover4` as an
    invented symbol system and delete the most interesting cultural signal in
    the run: terminology evolving inside English."""
    r = gate.check(msg)
    assert r.passed, f"{msg!r} was blocked"
    assert r.verdict is Verdict.NONSTANDARD
    assert r.coinages


def test_plain_english_is_accepted(gate):
    r = gate.check("clover gives more yield than none at wide spacing")
    assert r.verdict is Verdict.ACCEPTED and not r.coinages


def test_substitute_grammar_is_rejected(gate):
    """Coinage inside a sentence is fine. A private code with no English
    holding it together is not."""
    r = gate.check("zx4 qq7 vv2 zx9 kk1 pp3 rr8 tt5")
    assert not r.passed
    assert r.rejection.code is RejectionCode.E_LANG_SUBSTITUTE_GRAMMAR


def test_non_latin_script_is_rejected(gate):
    r = gate.check("беанс гоод")
    assert r.rejection.code is RejectionCode.E_LANG_NON_ASCII


def test_degenerate_repetition_is_rejected(gate):
    r = gate.check("yes yes yes yes yes yes yes yes yes yes")
    assert r.rejection.code is RejectionCode.E_LANG_DEGENERATE


def test_empty_and_overlong_are_rejected(gate):
    assert gate.check("   ").rejection.code is RejectionCode.E_LANG_EMPTY
    assert gate.check("word " * 500).rejection.code is RejectionCode.E_LANG_TOO_LONG


def test_gate_is_deterministic(gate):
    msg = "clover4 seems better than none at wide spacing"
    first = gate.check(msg)
    for _ in range(50):
        r = gate.check(msg)
        assert (r.verdict, r.oov_ratio, r.coinages) == \
               (first.verdict, first.oov_ratio, first.coinages)


def test_lexicon_is_bundled_and_hashed(gate):
    """Not a pip dependency: a wordlist that could change under us would
    silently change what counts as English between runs."""
    assert gate.path.name == "en_lexicon.txt"
    assert len(gate.hash) == 32
    assert LanguageGate().hash == gate.hash


def test_novelty_tracks_adoption_across_agents(gate):
    log = NoveltyLog()
    log.record(10, 0, gate.check("clover4 is better than none here"))
    log.record(20, 1, gate.check("i also see clover4 doing well"))
    log.record(30, 2, gate.check("nothing to report today"))
    rep = log.report()
    assert "clover4" in rep["adoption"]
    assert rep["adoption"]["clover4"] == [0, 1]
    assert rep["english_compliance"] == 1.0


def test_compliance_and_novelty_are_separate_metrics(gate):
    """Conflating them is how a gate ends up punishing invention."""
    log = NoveltyLog()
    log.record(1, 0, gate.check("clover4 and beans2 both beat none"))   # novel, compliant
    log.record(2, 1, gate.check("zx4 qq7 vv2 zx9 kk1 pp3 rr8"))         # rejected
    rep = log.report()
    assert rep["coined_terms"] >= 2
    assert rep["language_rejection_rate"] == 0.5


# =====================================================================
# Belief and trust
# =====================================================================

def test_teaching_transfers_hearsay_only():
    b = Beliefs()
    b.learn("clm_1", source=3, tick=10, world_status="confirmed")
    held = b.get("clm_1")
    assert held.status is BeliefStatus.HEARSAY
    assert held.source == 3 and held.personal_trials == 0


def test_only_personal_trials_upgrade_a_belief():
    b = Beliefs()
    b.learn("clm_1", source=3, tick=1, world_status="confirmed")
    for i in range(4):
        b.observe("clm_1", in_treat=True, yield_kg=5.0, tick=10 + i)
    assert b.get("clm_1").status is BeliefStatus.HEARSAY   # one arm only
    for i in range(4):
        b.observe("clm_1", in_treat=False, yield_kg=3.0, tick=20 + i)
    assert b.get("clm_1").status is BeliefStatus.PERSONALLY_CONFIRMED


def test_a_students_own_data_can_refute_what_it_was_told():
    b = Beliefs()
    b.learn("clm_1", source=3, tick=1, world_status="supported")
    for i in range(4):
        b.observe("clm_1", in_treat=True, yield_kg=2.0, tick=10 + i)
        b.observe("clm_1", in_treat=False, yield_kg=5.0, tick=10 + i)
    assert b.get("clm_1").status is BeliefStatus.PERSONALLY_REFUTED


def test_trust_moves_on_outcome_not_assertion():
    r = Relationships()
    r.record_teach(teacher=1, student=0)
    base = r.get(0, 1)
    r.settle(student=0, teacher=1, confirmed=True)
    assert r.get(0, 1) > base
    r2 = Relationships()
    r2.record_teach(teacher=1, student=0)
    r2.settle(student=0, teacher=1, confirmed=False)
    assert r2.get(0, 1) < base
    # Being misled costs more than being helped gains.
    assert (base - r2.get(0, 1)) > (r.get(0, 1) - base)


# =====================================================================
# Archetypes produce different civilizations
# =====================================================================

def run(policy: str, ticks: int = 1200, seed: int = 42) -> Engine:
    e = Engine(RunConfig(seed=seed, ticks=ticks, n_agents=5, policy=policy),
               SyntheticDomain())
    e.run()
    return e


@pytest.fixture(scope="module")
def honest():
    return run("honest")


@pytest.fixture(scope="module")
def liar():
    return run("liar")


@pytest.fixture(scope="module")
def copycat():
    return run("copycat")


def test_teaching_actually_happens(honest):
    assert honest.relationships.report()["teach_events"] > 0


def test_teaching_transfers_no_skill(honest):
    """Knowledge transfers; competence does not."""
    for a in honest.state.agent_ids():
        for b in honest.beliefs[int(a)].all():
            if b.source is not None:
                assert b.personal_trials >= 0        # earned, never granted
    # No agent's skill was raised by being taught; skill only accrues on harvest.
    harvests = {int(a): 0 for a in honest.state.agent_ids()}
    for t in honest.trials:
        harvests[int(t.agent_id)] += 1
    for a in honest.state.agent_ids():
        expected = min(1.0, harvests[int(a)] * 0.02)
        assert honest.state.agents[a].skill_farming == pytest.approx(expected, abs=1e-6)


def test_lying_buys_nothing(liar):
    """A liar who is RIGHT about the world should still be judged on evidence.

    Here the liar inflates what it claims to have found. The world refuses the
    overclaim -- while its students, testing for themselves, personally confirm
    the underlying fact. Truth and honesty, correctly separated.
    """
    banked = [r for r in liar.kb if r.state.value in ("confirmed", "generalized")]
    assert not banked, "an inflated claim was banked"
    assert all(r.state.value == "refuted" for r in liar.kb)

    confirmed_personally = sum(
        1 for a in liar.state.agent_ids()
        for b in liar.beliefs[int(a)].all()
        if b.status is BeliefStatus.PERSONALLY_CONFIRMED)
    assert confirmed_personally > 0, (
        "the liar's students should still learn the true underlying fact")


def test_copycat_shows_blind_adoption(honest, copycat):
    """Adopting on hearsay without testing is the failure mode that makes a
    civilization confidently wrong. It has to be visible.

    The comparison is against the honest arm rather than an absolute
    threshold: what distinguishes the archetypes is that a copycat holds
    beliefs it never checked and an honest agent does not.
    """
    cc = civilization_report(copycat, copycat.domain)["social"]
    hh = civilization_report(honest, honest.domain)["social"]
    assert cc["blind_adoption"] > 0
    assert hh["blind_adoption"] == 0
    assert cc["blind_adoption_rate"] > hh["blind_adoption_rate"]


def test_what_the_copycat_arm_does_and_does_not_change(honest, copycat):
    """Pins down the archetype difference precisely, including the null part.

    DOES differ: a copycat acts on what it was told before testing it. An
    honest agent does not. That is `blind_adoption`, and it is the whole point
    of the archetype.

    Does NOT differ: how much public knowledge each banks. With the current
    timing both finish exploring before anyone teaches them, so copying costs
    them nothing. That is a real finding about WHEN teaching lands in a run,
    and asserting the convenient inequality instead would have buried it.

    Also does not differ in the intuitive direction: raw "untested beliefs"
    is not a discriminator either, because planting what you were told
    generates trials for it. Only acting-before-testing separates them.
    """
    cc = civilization_report(copycat, copycat.domain)
    hh = civilization_report(honest, honest.domain)

    assert cc["social"]["blind_adoption"] > 0
    assert hh["social"]["blind_adoption"] == 0

    # Recorded, not asserted as an inequality: the arms bank comparably here.
    banked = (cc["knowledge"]["banked"], hh["knowledge"]["banked"])
    assert all(b >= 0 for b in banked)
    assert cc["grading"]["false_positives"] == 0, (
        "copying spread a false claim, which would be a different and much "
        "more interesting result than the one this test documents")


# =====================================================================
# Civilization metrics
# =====================================================================

def test_false_discovery_rate_is_graded_against_ground_truth(honest):
    g = civilization_report(honest, honest.domain)["grading"]
    assert g["banked"] > 0
    assert g["false_discovery_rate"] == 0.0
    assert all(row["true_effect"] > 0 for row in g["graded"])


def test_competence_measures_knowledge_not_practice():
    """Both sides are evaluated at a reference skill and averaged over planting
    day, so an agent is scored on what it chose, not on how strong it got or
    when it happened to act."""
    d = SyntheticDomain()
    opt = controllable_optimum(d)
    assert 3.0 < opt < 5.0

    random_run = run("random", ticks=600)
    good_run = run("scripted_factorial", ticks=1200)
    r_reach = sum(1 for a in random_run.state.agent_ids()
                  if time_to_competence(d, random_run.trials, int(a)) is not None)
    g_reach = sum(1 for a in good_run.state.agent_ids()
                  if time_to_competence(d, good_run.trials, int(a)) is not None)
    assert g_reach > r_reach


def test_regret_orders_the_policies(honest):
    d = SyntheticDomain()
    rnd = civilization_report(run("random", ticks=600), d)["collective"]
    hon = civilization_report(honest, d)["collective"]
    assert hon["collective_regret"] < rnd["collective_regret"]


def test_resilience_reports_what_survives_losing_one_agent(honest):
    r = civilization_report(honest, honest.domain)["resilience"]
    assert r["banked"] > 0
    assert 0.0 <= r["resilience"] <= 1.0
    assert r["most_knowledgeable"] is not None


# =====================================================================
# Specialization must not mistake terrain for society
# =====================================================================

def test_specialization_null_homogeneous():
    """Identical policies, identical circumstances: no roles."""
    actions = {a: ["PLANT", "HARVEST", "MOVE", "REST"] * 5 for a in range(5)}
    env = {a: {"mean_soil_in_reach": 2.0, "reachable_tiles": 40,
               "opportunities": 40, "skill": 0.5} for a in range(5)}
    assert not spec_analyse(actions, env).detected


def test_specialization_null_heterogeneous_terrain():
    """THE important one. Identical policies on VARIED terrain must still show
    no roles: an agent beside fertile soil farms more, and calling that a
    social role is exactly the confound this controls for."""
    actions, env = {}, {}
    for a in range(5):
        # behaviour driven purely by how much arable land is in reach
        plants = 10 + 4 * a
        actions[a] = ["PLANT"] * plants + ["MOVE"] * (30 - plants) + ["HARVEST"] * 10
        env[a] = {"mean_soil_in_reach": 1.0 + 0.5 * a,
                  "reachable_tiles": 20 + 4 * a,
                  "opportunities": 20 + 4 * a, "skill": 0.5}
    rep = spec_analyse(actions, env)
    assert not rep.detected, (
        f"terrain advantage was mistaken for a social role "
        f"(index {rep.index:.3f})")
    assert rep.explained_variance > 0.5
