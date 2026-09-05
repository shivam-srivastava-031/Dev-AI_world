"""The action protocol and its rejection taxonomy.

A rejection code without a hint is a silent failure: the agent learns nothing
and we never find out why it kept doing the wrong thing.
"""

from __future__ import annotations

from aiciv.world.actions import (
    CORE_VERBS, HINTS, MESSAGE_VERBS, STAGE_OF, ActionProposal, Rejection,
    RejectionCode, Stage, Verb,
)


def test_every_rejection_code_has_a_hint():
    missing = [c.value for c in RejectionCode if c not in HINTS]
    assert not missing, f"codes without a hint: {missing}"


def test_every_rejection_code_has_a_stage():
    assert all(c in STAGE_OF for c in RejectionCode)
    assert set(STAGE_OF.values()) == set(Stage)


def test_hints_are_written_for_the_agent_not_the_developer():
    """Hints reach a language model, so they must be sentences, not tokens."""
    for code, hint in HINTS.items():
        assert hint and hint[0].isupper(), f"{code.value}: not a sentence"
        assert hint.rstrip().endswith((".", "!")), f"{code.value}: no terminator"
        assert "E_" not in hint, f"{code.value}: leaks an internal code"
        assert len(hint.split()) >= 4, f"{code.value}: too terse to teach anything"


def test_rejection_feedback_shape():
    r = Rejection(RejectionCode.E_KB_TRIAL_PRE_REGISTRATION, "trial 4 < tick 12")
    fb = r.as_feedback()
    assert set(fb) == {"code", "detail", "hint"}
    assert r.stage is Stage.EPISTEMIC
    assert "Register first" in fb["hint"]


def test_core_verbs_do_not_require_the_knowledge_base():
    """An agent that ignores the protocol entirely must still be able to live.

    The protocol is an affordance, not a mandate (docs/protocol.md 2.1).
    """
    kb_verbs = {Verb.PROPOSE_CLAIM, Verb.REGISTER_CLAIM, Verb.SUBMIT_TRIALS,
                Verb.CHALLENGE_CLAIM, Verb.WITHDRAW_CLAIM}
    assert not (CORE_VERBS & kb_verbs)
    for essential in (Verb.PLANT, Verb.HARVEST, Verb.EAT, Verb.MOVE, Verb.SAY):
        assert essential in CORE_VERBS


def test_message_verbs_are_the_language_gated_ones():
    assert MESSAGE_VERBS == {Verb.SAY, Verb.ASK, Verb.TEACH, Verb.GOAL_SET}


def test_proposal_is_inert():
    """A proposal is a request. Constructing one changes nothing."""
    p = ActionProposal(Verb.PLANT, {"tile": [4, 7], "spacing": 4},
                       rationale="clover seemed to help here")
    assert p.verb is Verb.PLANT and p.params["spacing"] == 4
    assert p.rationale                      # logged, never scored
