"""The information boundary, as data rather than convention.

Every datum in the system carries a class. Tests walk this registry and assert
that nothing HIDDEN or METRICS_ONLY reaches an Observation, a memory record, a
prompt, or an API response. Without this, a debugging endpoint added six months
from now quietly invalidates every experiment run after it.
"""

from __future__ import annotations

from enum import Enum


class InfoClass(str, Enum):
    PUBLIC = "public"                     # in every Observation
    AGENT_OBSERVABLE = "agent_observable" # reachable via an action
    AGENT_INFERABLE = "agent_inferable"   # derivable from observations alone
    HIDDEN = "hidden"                     # world-internal, never exposed
    METRICS_ONLY = "metrics_only"         # ground truth, for grading only
    PRIVATE = "private"                   # belongs to exactly one agent


#: Anything an agent must never see, in any channel.
FORBIDDEN_TO_AGENTS = frozenset({InfoClass.HIDDEN, InfoClass.METRICS_ONLY})

#: Field-name -> class. Keys are the canonical names used across the schema,
#: Observation objects and domain outcomes.
FIELD_CLASS: dict[str, InfoClass] = {
    # --- what an agent legitimately perceives -------------------------------
    "agent_id": InfoClass.PUBLIC,
    "tick": InfoClass.PUBLIC,
    "day_of_cycle": InfoClass.PUBLIC,
    "x": InfoClass.PUBLIC,
    "y": InfoClass.PUBLIC,
    "food": InfoClass.PUBLIC,
    "energy": InfoClass.PUBLIC,
    "water_stock": InfoClass.PUBLIC,
    "seeds": InfoClass.PUBLIC,
    "skill_farming": InfoClass.PUBLIC,
    "reputation": InfoClass.PUBLIC,
    "observed_yield": InfoClass.PUBLIC,
    "crop_health": InfoClass.PUBLIC,
    "crop_stage": InfoClass.PUBLIC,
    "last_action_result": InfoClass.PUBLIC,
    "rejection_code": InfoClass.PUBLIC,
    "spacing": InfoClass.PUBLIC,
    "plant_day": InfoClass.PUBLIC,
    "water": InfoClass.PUBLIC,
    "companion": InfoClass.PUBLIC,
    "tile_id": InfoClass.PUBLIC,
    "trial_id": InfoClass.PUBLIC,
    "claim_id": InfoClass.PUBLIC,
    "claim_state": InfoClass.PUBLIC,

    "soil_band": InfoClass.AGENT_OBSERVABLE,   # via INSPECT_TILE

    # --- world internals ----------------------------------------------------
    "tile_stratum": InfoClass.HIDDEN,          # discovery / confirmation / holdout
    "run_secret": InfoClass.HIDDEN,
    "signature": InfoClass.HIDDEN,
    "noise_draw": InfoClass.HIDDEN,
    "synergy_table": InfoClass.HIDDEN,
    "water_opt": InfoClass.HIDDEN,

    # --- ground truth, for grading only -------------------------------------
    "true_mu": InfoClass.METRICS_ONLY,
    "true_optimum": InfoClass.METRICS_ONLY,
    "true_delta": InfoClass.METRICS_ONLY,
    "prior_baseline": InfoClass.METRICS_ONLY,

    # --- one agent's alone ---------------------------------------------------
    "memory": InfoClass.PRIVATE,
    "belief": InfoClass.PRIVATE,
    "goals": InfoClass.PRIVATE,
    "rationale": InfoClass.PRIVATE,
}

#: Method vocabulary that must not appear in the rules_only prompt.
#:
#: This lives here rather than in the prompt module because it is a statement
#: about the information boundary, not about prose. If "run controlled trials"
#: reaches an agent, we have handed over the scientific method and any later
#: claim about them applying it is circular -- the same class of mistake as
#: leaking a stratum, and it belongs in the same file.
#:
#: The list is deliberately blunt: it is easier to keep a word out than to
#: argue about whether a particular sentence implies it.
BANNED_TERMS = (
    "hypothesis", "hypotheses", "experiment", "experimental", "control group",
    "baseline", "replicate", "replication", "confound", "variable",
    "significance", "significant", "sample size", "p-value", "scientific",
    "science", "systematic", "methodical", "isolate", "hold constant",
    "compare", "comparison", "evidence", "test",
)


def scan_banned(text: str) -> list[str]:
    """Which banned method terms a piece of text contains.

    Used two ways, and the difference matters. Against OUR prompt templates it
    is an assertion: a test fails if any term appears. Against an operator's
    brief it is a report -- what an observer tells the agents is theirs to
    decide, but a directive that says "run a controlled experiment" has handed
    over the very thing rules_only withholds, and that run can no longer be
    compared with one that was not given it. So it is surfaced at launch, in
    the manifest and in the dashboard, rather than silently accepted or
    silently refused.
    """
    low = text.lower()
    return sorted({t for t in BANNED_TERMS if t in low})


#: Substrings that must never appear as keys in anything handed to a policy.
#: Cheap, blunt, and catches the realistic failure mode: someone adds a debug
#: field called "true_mu_for_logging" and nobody notices for three months.
LEAK_MARKERS = ("true_mu", "true_optimum", "true_delta", "synergy",
                "stratum", "run_secret", "signature", "noise_draw")


def classify(field: str) -> InfoClass | None:
    return FIELD_CLASS.get(field)


def assert_no_leak(payload: object, *, where: str) -> None:
    """Recursively assert a structure carries nothing forbidden to agents.

    Used on every Observation before it reaches a policy, and in API tests.
    """
    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                key = str(k)
                low = key.lower()
                for marker in LEAK_MARKERS:
                    if marker in low:
                        raise AssertionError(
                            f"information leak at {where}{path}.{key}: "
                            f"matches forbidden marker {marker!r}"
                        )
                if classify(key) in FORBIDDEN_TO_AGENTS:
                    raise AssertionError(
                        f"information leak at {where}{path}.{key}: "
                        f"classified {classify(key).value}"
                    )
                walk(v, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(payload, "")
