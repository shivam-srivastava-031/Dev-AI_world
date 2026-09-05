"""The action protocol: the only write channel from an agent into the world.

A policy returns an ActionProposal. The world validates it, then executes or
rejects it. A rejection carries a code and a *hint*, both of which are written
into the agent's memory so an agent learns the world's rules by hitting them
rather than by being told them in the prompt.

Design note: a rejected action still consumes the tick. Without that, an LLM
gets unbounded retries, wall-clock cost explodes, and the tick loop stops being
bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verb(str, Enum):
    # --- physical --------------------------------------------------------
    MOVE = "MOVE"
    INSPECT_TILE = "INSPECT_TILE"
    PLANT = "PLANT"
    TEND = "TEND"
    HARVEST = "HARVEST"
    BUILD_CHANNEL = "BUILD_CHANNEL"
    EAT = "EAT"
    REST = "REST"
    # --- communication ---------------------------------------------------
    SAY = "SAY"
    ASK = "ASK"
    TEACH = "TEACH"
    GOAL_SET = "GOAL_SET"
    # --- epistemic (all OPTIONAL affordances; nothing requires their use) --
    PREDICT = "PREDICT"
    PROPOSE_CLAIM = "PROPOSE_CLAIM"
    REGISTER_CLAIM = "REGISTER_CLAIM"
    SUBMIT_TRIALS = "SUBMIT_TRIALS"
    CHALLENGE_CLAIM = "CHALLENGE_CLAIM"
    WITHDRAW_CLAIM = "WITHDRAW_CLAIM"
    # --- null ------------------------------------------------------------
    NOOP = "NOOP"


#: Verbs an agent may use without ever engaging the knowledge base. An agent
#: that only ever uses these is a perfectly valid inhabitant; the shadow
#: verifier measures it regardless (see docs/protocol.md section 2.2).
CORE_VERBS = frozenset({
    Verb.MOVE, Verb.INSPECT_TILE, Verb.PLANT, Verb.TEND, Verb.HARVEST,
    Verb.EAT, Verb.REST, Verb.SAY, Verb.ASK, Verb.GOAL_SET, Verb.NOOP,
})

#: Verbs carrying free-text that must pass the language gate.
MESSAGE_VERBS = frozenset({Verb.SAY, Verb.ASK, Verb.TEACH, Verb.GOAL_SET})


class Stage(str, Enum):
    """Validation stages, applied in this order. First failure wins."""
    SCHEMA = "schema"
    BOUNDS = "bounds"
    SPATIAL = "spatial"
    RESOURCE = "resource"
    LANGUAGE = "language"
    EPISTEMIC = "epistemic"


class RejectionCode(str, Enum):
    # --- 1 SCHEMA ---------------------------------------------------------
    E_SCHEMA_MALFORMED_JSON = "E_SCHEMA_MALFORMED_JSON"
    E_SCHEMA_UNKNOWN_ACTION = "E_SCHEMA_UNKNOWN_ACTION"
    E_SCHEMA_MISSING_PARAM = "E_SCHEMA_MISSING_PARAM"
    E_SCHEMA_TYPE = "E_SCHEMA_TYPE"
    E_SCHEMA_EXTRA_PARAM = "E_SCHEMA_EXTRA_PARAM"
    # --- 2 BOUNDS ---------------------------------------------------------
    E_BOUNDS_TILE_OUT_OF_GRID = "E_BOUNDS_TILE_OUT_OF_GRID"
    E_BOUNDS_PARAM_RANGE = "E_BOUNDS_PARAM_RANGE"
    E_BOUNDS_DELTA_NONPOSITIVE = "E_BOUNDS_DELTA_NONPOSITIVE"
    # --- 3 SPATIAL --------------------------------------------------------
    E_SPATIAL_NOT_ADJACENT = "E_SPATIAL_NOT_ADJACENT"
    E_SPATIAL_TILE_OCCUPIED = "E_SPATIAL_TILE_OCCUPIED"
    E_SPATIAL_TILE_NOT_ARABLE = "E_SPATIAL_TILE_NOT_ARABLE"
    E_SPATIAL_TARGET_TOO_FAR = "E_SPATIAL_TARGET_TOO_FAR"
    # --- 4 RESOURCE / STATE ------------------------------------------------
    E_RES_INSUFFICIENT_WATER = "E_RES_INSUFFICIENT_WATER"
    E_RES_NO_SEEDS = "E_RES_NO_SEEDS"
    E_RES_TILE_ALREADY_PLANTED = "E_RES_TILE_ALREADY_PLANTED"
    E_RES_TILE_FALLOW = "E_RES_TILE_FALLOW"
    E_CAP_NOT_HELD = "E_CAP_NOT_HELD"
    E_CAP_INSUFFICIENT = "E_CAP_INSUFFICIENT"
    E_CAP_ALREADY_BUILT = "E_CAP_ALREADY_BUILT"
    E_RES_TILE_NOT_PLANTED = "E_RES_TILE_NOT_PLANTED"
    E_RES_CROP_NOT_MATURE = "E_RES_CROP_NOT_MATURE"
    E_RES_ENERGY = "E_RES_ENERGY"
    E_RES_TOO_MANY_PLOTS = "E_RES_TOO_MANY_PLOTS"
    E_RES_NO_FOOD = "E_RES_NO_FOOD"
    E_STATE_ALREADY_ACTED = "E_STATE_ALREADY_ACTED"
    # --- 5 LANGUAGE --------------------------------------------------------
    E_LANG_NON_ASCII = "E_LANG_NON_ASCII"
    E_LANG_SUBSTITUTE_GRAMMAR = "E_LANG_SUBSTITUTE_GRAMMAR"
    E_LANG_DEGENERATE = "E_LANG_DEGENERATE"
    E_LANG_EMPTY = "E_LANG_EMPTY"
    E_LANG_TOO_LONG = "E_LANG_TOO_LONG"
    # --- 6 SOCIAL / EPISTEMIC ----------------------------------------------
    E_KB_CLAIM_NOT_FOUND = "E_KB_CLAIM_NOT_FOUND"
    E_KB_CLAIM_NOT_YOURS = "E_KB_CLAIM_NOT_YOURS"
    E_KB_CLAIM_WRONG_STATE = "E_KB_CLAIM_WRONG_STATE"
    E_KB_TOO_MANY_OPEN_CLAIMS = "E_KB_TOO_MANY_OPEN_CLAIMS"
    E_KB_INSUFFICIENT_REPUTATION = "E_KB_INSUFFICIENT_REPUTATION"
    E_KB_SPEC_FROZEN = "E_KB_SPEC_FROZEN"
    E_KB_MISSING_BASELINE = "E_KB_MISSING_BASELINE"
    E_KB_TRIAL_NOT_FOUND = "E_KB_TRIAL_NOT_FOUND"
    E_KB_TRIAL_NOT_YOURS = "E_KB_TRIAL_NOT_YOURS"
    E_KB_TRIAL_PRE_REGISTRATION = "E_KB_TRIAL_PRE_REGISTRATION"
    E_KB_TRIAL_ALREADY_USED = "E_KB_TRIAL_ALREADY_USED"
    E_KB_POST_TREATMENT_COVARIATE = "E_KB_POST_TREATMENT_COVARIATE"
    E_KB_SELF_REPLICATION = "E_KB_SELF_REPLICATION"
    E_TRIAL_SIGNATURE_INVALID = "E_TRIAL_SIGNATURE_INVALID"
    E_TEACH_SELF = "E_TEACH_SELF"
    E_TEACH_UNKNOWN_CLAIM = "E_TEACH_UNKNOWN_CLAIM"
    E_TEACH_DISABLED = "E_TEACH_DISABLED"
    E_PREDICT_NO_PENDING_TRIAL = "E_PREDICT_NO_PENDING_TRIAL"


STAGE_OF: dict[RejectionCode, Stage] = {}
for _code in RejectionCode:
    _n = _code.value
    if _n.startswith("E_SCHEMA"):
        STAGE_OF[_code] = Stage.SCHEMA
    elif _n.startswith("E_BOUNDS"):
        STAGE_OF[_code] = Stage.BOUNDS
    elif _n.startswith("E_SPATIAL"):
        STAGE_OF[_code] = Stage.SPATIAL
    elif _n.startswith(("E_RES", "E_STATE")):
        STAGE_OF[_code] = Stage.RESOURCE
    elif _n.startswith("E_LANG"):
        STAGE_OF[_code] = Stage.LANGUAGE
    else:
        STAGE_OF[_code] = Stage.EPISTEMIC


#: Every code needs a hint. The hint is what teaches an agent the rule; a code
#: without one is a silent failure. test_every_rejection_code_has_a_hint guards.
HINTS: dict[RejectionCode, str] = {
    RejectionCode.E_SCHEMA_MALFORMED_JSON: "Your reply was not valid JSON. Return a single JSON object.",
    RejectionCode.E_SCHEMA_UNKNOWN_ACTION: "That is not an action I know. Use one of the listed actions.",
    RejectionCode.E_SCHEMA_MISSING_PARAM: "That action needs a parameter you did not supply.",
    RejectionCode.E_SCHEMA_TYPE: "A parameter had the wrong type.",
    RejectionCode.E_SCHEMA_EXTRA_PARAM: "You supplied a parameter that action does not take.",
    RejectionCode.E_BOUNDS_TILE_OUT_OF_GRID: "That position is outside the world.",
    RejectionCode.E_BOUNDS_PARAM_RANGE: "A parameter was outside its allowed values.",
    RejectionCode.E_BOUNDS_DELTA_NONPOSITIVE: "A predicted difference must be greater than zero.",
    RejectionCode.E_SPATIAL_NOT_ADJACENT: "You can only act on your own tile or one touching it.",
    RejectionCode.E_SPATIAL_TILE_OCCUPIED: "Another agent got to that tile first this tick.",
    RejectionCode.E_SPATIAL_TILE_NOT_ARABLE: "Nothing will grow on that tile.",
    RejectionCode.E_SPATIAL_TARGET_TOO_FAR: "That agent is too far away to hear you.",
    RejectionCode.E_RES_INSUFFICIENT_WATER: "You do not carry enough water for that.",
    RejectionCode.E_RES_NO_SEEDS: "You have no seeds.",
    RejectionCode.E_RES_TILE_ALREADY_PLANTED: "Something is already growing there.",
    RejectionCode.E_RES_TILE_FALLOW: "That ground was worked too recently and needs to rest.",
    RejectionCode.E_CAP_NOT_HELD: "You do not know how to build that yet.",
    RejectionCode.E_CAP_INSUFFICIENT: "You know how, but you are not ready to do it yet.",
    RejectionCode.E_CAP_ALREADY_BUILT: "There is already one of those here.",
    RejectionCode.E_RES_TILE_NOT_PLANTED: "Nothing is growing there.",
    RejectionCode.E_RES_CROP_NOT_MATURE: "That crop is not ready to harvest yet.",
    RejectionCode.E_RES_ENERGY: "You are too tired. Rest first.",
    RejectionCode.E_RES_TOO_MANY_PLOTS: "You are already tending as many plots as you can.",
    RejectionCode.E_RES_NO_FOOD: "You have no food to eat.",
    RejectionCode.E_STATE_ALREADY_ACTED: "You have already acted this tick.",
    RejectionCode.E_LANG_NON_ASCII: "Write in English using ordinary letters.",
    RejectionCode.E_LANG_SUBSTITUTE_GRAMMAR: "That did not read as English. Shorthand is fine, but the sentence around it must be English.",
    RejectionCode.E_LANG_DEGENERATE: "That message repeated itself without saying anything.",
    RejectionCode.E_LANG_EMPTY: "The message was empty.",
    RejectionCode.E_LANG_TOO_LONG: "That message was too long.",
    RejectionCode.E_KB_CLAIM_NOT_FOUND: "There is no claim with that id.",
    RejectionCode.E_KB_CLAIM_NOT_YOURS: "Only the author can do that to a claim.",
    RejectionCode.E_KB_CLAIM_WRONG_STATE: "That claim is not in a state where that is possible.",
    RejectionCode.E_KB_TOO_MANY_OPEN_CLAIMS: "You already have as many open claims as you may.",
    RejectionCode.E_KB_INSUFFICIENT_REPUTATION: "You do not have enough standing to register a claim.",
    RejectionCode.E_KB_SPEC_FROZEN: "A registered claim cannot be edited. Register a new one instead.",
    RejectionCode.E_KB_MISSING_BASELINE: "A claim needs something to be compared against.",
    RejectionCode.E_KB_TRIAL_NOT_FOUND: "There is no trial with that id.",
    RejectionCode.E_KB_TRIAL_NOT_YOURS: "You can only submit trials you performed.",
    RejectionCode.E_KB_TRIAL_PRE_REGISTRATION: "That trial happened before you registered the claim, so it cannot support it. Register first, then gather evidence.",
    RejectionCode.E_KB_TRIAL_ALREADY_USED: "That trial has already been counted for this claim.",
    RejectionCode.E_KB_POST_TREATMENT_COVARIATE: "You cannot adjust for something your own action changed.",
    RejectionCode.E_KB_SELF_REPLICATION: "Your own trials cannot replicate your own claim.",
    RejectionCode.E_TRIAL_SIGNATURE_INVALID: "That trial record does not match what the world recorded.",
    RejectionCode.E_TEACH_SELF: "You cannot teach yourself.",
    RejectionCode.E_TEACH_UNKNOWN_CLAIM: "You do not know that claim, so you cannot teach it.",
    RejectionCode.E_TEACH_DISABLED: "Teaching is not possible here.",
    RejectionCode.E_PREDICT_NO_PENDING_TRIAL: "There is nothing pending for you to predict.",
}


@dataclass(frozen=True)
class ActionProposal:
    """What a policy returns. Purely a request; it changes nothing by itself."""

    verb: Verb
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""          # PRIVATE. Logged, never scored, never affects the world.
    message: str = ""            # free text for MESSAGE_VERBS


@dataclass(frozen=True)
class Rejection:
    code: RejectionCode
    detail: str = ""

    @property
    def stage(self) -> Stage:
        return STAGE_OF[self.code]

    @property
    def hint(self) -> str:
        return HINTS[self.code]

    def as_feedback(self) -> dict[str, str]:
        """Exactly what gets written into agent memory and shown next tick."""
        return {"code": self.code.value, "detail": self.detail, "hint": self.hint}


@dataclass(frozen=True)
class Accepted:
    proposal: ActionProposal
