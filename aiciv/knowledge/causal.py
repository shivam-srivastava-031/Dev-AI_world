"""Causal covariate classification, enforced rather than documented.

The trap this module exists to prevent:

    treatment -> agent gets a better harvest -> agent's skill rises

``skill_at_plant`` is PRE_TREATMENT for a *single* trial -- it is stamped before
the seed goes in. But across a claim's whole trial sequence it is
treatment-affected, because earlier treatment trials raised it. Adjusting for it
in a regression therefore opens a path that can absorb the very effect being
measured, biasing a real effect toward zero.

The rule: for CAUSAL_EFFECT claims, skill is handled by BLOCKING (compare within
skill strata) or by within-agent randomisation -- never by regression
adjustment. COMPARISON claims may adjust for it, and are labelled associational
in the report.
"""

from __future__ import annotations

from enum import Enum

from .claim import ClaimType


class CovariateClass(str, Enum):
    PRE_TREATMENT = "pre_treatment"    # fixed before the action is chosen
    POST_TREATMENT = "post_treatment"  # changed by the treatment
    MEDIATOR = "mediator"              # on the causal path
    OUTCOME = "outcome"                # the measurand
    TREATMENT = "treatment"            # the thing being varied


#: Every field a trial row carries must appear here.
#: test_covariate_classification_is_total walks the Trial dataclass and fails
#: if a new field is added without a classification.
COVARIATE_CLASS: dict[str, CovariateClass] = {
    # --- fixed before the agent chooses anything ---------------------------
    "tile_id": CovariateClass.PRE_TREATMENT,
    "soil_band": CovariateClass.PRE_TREATMENT,
    "planted_tick": CovariateClass.PRE_TREATMENT,
    "plant_day": CovariateClass.PRE_TREATMENT,
    "stratum": CovariateClass.PRE_TREATMENT,
    "agent_id": CovariateClass.PRE_TREATMENT,

    # --- the treatment itself ----------------------------------------------
    "spacing": CovariateClass.TREATMENT,
    "water": CovariateClass.TREATMENT,
    "companion": CovariateClass.TREATMENT,
    "density": CovariateClass.TREATMENT,
    "irrigation": CovariateClass.TREATMENT,
    "nitrogen": CovariateClass.TREATMENT,
    "params": CovariateClass.TREATMENT,

    # --- affected by the treatment -----------------------------------------
    # Pre-treatment for one trial, treatment-affected across a sequence. See
    # the module docstring; this is the whole reason the module exists.
    "skill_at_plant": CovariateClass.POST_TREATMENT,
    "harvested_tick": CovariateClass.POST_TREATMENT,

    # --- on the causal path -------------------------------------------------
    "crop_health": CovariateClass.MEDIATOR,

    # --- the measurand ------------------------------------------------------
    "yield_kg": CovariateClass.OUTCOME,

    # --- bookkeeping, never a covariate -------------------------------------
    "trial_id": CovariateClass.PRE_TREATMENT,
    "signature": CovariateClass.PRE_TREATMENT,
    "true_mu": CovariateClass.OUTCOME,
}

#: What a CAUSAL_EFFECT claim is allowed to adjust for.
ADMISSIBLE_FOR_CAUSAL = frozenset({CovariateClass.PRE_TREATMENT})

#: Covariates the verifier uses by default for an associational COMPARISON.
DEFAULT_COMPARISON_COVARIATES = ("soil_band", "water", "plant_day", "skill_at_plant")

#: Covariates admissible when the claim is causal. Note the absence of skill.
DEFAULT_CAUSAL_COVARIATES = ("soil_band", "plant_day")


#: Domain parameters not listed explicitly are TREATMENT: they are the thing
#: the agent varied. The schedule axis is the exception -- it is fixed by when
#: the agent acted, so it is pre-treatment.
SCHEDULE_AXES = frozenset({"plant_day"})


def classify(field: str) -> CovariateClass | None:
    known = COVARIATE_CLASS.get(field)
    if known is not None:
        return known
    if field in SCHEDULE_AXES:
        return CovariateClass.PRE_TREATMENT
    return None


def classify_for_domain(field: str, domain) -> CovariateClass | None:
    """Classification that knows the domain's own parameter names."""
    known = classify(field)
    if known is not None:
        return known
    if field in getattr(domain, "param_space", {}):
        return (CovariateClass.PRE_TREATMENT
                if field == getattr(domain, "schedule_axis", "")
                else CovariateClass.TREATMENT)
    return None


def validate_adjustment_set(
    claim_type: ClaimType, adjustment_set: tuple[str, ...]
) -> str | None:
    """Return a rejection detail, or None if the adjustment set is admissible.

    Called at PROPOSE_CLAIM time so an agent learns the rule from a hint rather
    than discovering months later that its causal estimates were biased.
    """
    for name in adjustment_set:
        cls = classify(name)
        if cls is None:
            return f"unknown covariate {name!r}"
        if cls is CovariateClass.OUTCOME:
            return f"{name!r} is the outcome and cannot be adjusted for"
        if cls is CovariateClass.TREATMENT:
            return f"{name!r} is the treatment and cannot be adjusted for"
        if claim_type is ClaimType.CAUSAL_EFFECT and cls not in ADMISSIBLE_FOR_CAUSAL:
            return (
                f"{name!r} is {cls.value}: adjusting for something your own "
                f"action changed would absorb the effect you are measuring. "
                f"Compare within levels of it instead"
            )
    return None


def block_by_skill(rows: list[dict], n_blocks: int = 3) -> dict[int, list[dict]]:
    """Partition trials into skill strata, for blocking rather than adjustment.

    Blocks are fixed cut points, not quantiles of the observed data: quantiles
    would be a function of the treatment assignment and reintroduce exactly the
    dependence blocking is meant to remove.
    """
    edges = [i / n_blocks for i in range(1, n_blocks)]
    out: dict[int, list[dict]] = {i: [] for i in range(n_blocks)}
    for r in rows:
        s = float(r.get("skill_at_plant", 0.0))
        b = sum(1 for e in edges if s >= e)
        out[b].append(r)
    return out
