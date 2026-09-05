"""Protocol adoption, graded on the AGENT's information state.

`registered_claims / something` is only a fair rate if the denominator counts
occasions when the agent could reasonably have formed a hypothesis. Dividing by
ticks, or by trials, or by what the world knows to be true, punishes an agent
for never having encountered comparable evidence -- which is not a failure to
adopt the scientific protocol, it is simply not having had the chance.

So an OPPORTUNITY is defined as: this agent, from its OWN observed trials, held
at least `min_per_arm` results in each of two levels of one parameter. That is
the point at which a comparison became formulable *to it*.

Two rules this module obeys, and which its tests enforce:

  1. It reads ONLY agent-visible fields. Never `stratum`, never `true_mu`,
     never the shadow verifier's view. Rows are projected before use, so a
     caller that passes full trial rows gets the same answer as one that
     passes an agent's own memory.
  2. An agent with no opportunities has an UNDEFINED adoption rate, not zero.
     Nothing to adopt is not a failure to adopt.

This keeps the metric aligned with the project's central rule: we grade
behaviour using the agent's information state, not ours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

#: Parameters an agent could plausibly form a contrast about. These are the
#: PUBLIC action parameters, the same ones it chose when planting.
OPPORTUNITY_VARIABLES = ("companion", "spacing", "water")

#: Fields this module is permitted to read. Anything else is a boundary breach.
VISIBLE_FIELDS = frozenset({
    "trial_id", "agent_id", "tile_id", "planted_tick", "harvested_tick",
    "spacing", "plant_day", "water", "companion", "soil_band",
    "skill_at_plant", "yield_kg", "crop_health",
})

MIN_PER_ARM = 6        # enough to form an impression; well under verification n
SALIENCE_D = 0.5       # observed standardised difference, from the agent's data


def project(rows: Iterable[dict]) -> list[dict]:
    """Strip every field an agent could not have seen.

    Called on the way in so this module cannot accidentally depend on hidden
    state even if a caller hands it raw world rows.
    """
    return [{k: v for k, v in r.items() if k in VISIBLE_FIELDS} for r in rows]


@dataclass(frozen=True)
class Opportunity:
    """A comparison this agent had the evidence to formulate."""

    agent_id: int
    variable: str
    level_a: Any
    level_b: Any
    n_a: int
    n_b: int
    observed_delta: float      # a's mean minus b's, in the agent's own data
    observed_d: float          # standardised, likewise
    arose_tick: int            # when both arms first reached min_per_arm

    @property
    def salient(self) -> bool:
        """Did it *look* like something worth claiming, to this agent?

        Judged on the agent's own observed effect size. A contrast it could
        formulate but which looked like nothing is a weaker expectation than
        one that looked substantial.
        """
        return abs(self.observed_d) >= SALIENCE_D

    def key(self) -> tuple:
        return (self.variable, self.level_a, self.level_b)


def _stats(vals: Sequence[float]) -> tuple[float, float]:
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(max(var, 0.0))


def opportunities(
    rows: Iterable[dict],
    *,
    min_per_arm: int = MIN_PER_ARM,
    variables: Sequence[str] = OPPORTUNITY_VARIABLES,
) -> list[Opportunity]:
    """Every contrast this agent had enough of its own evidence to form.

    Ordered pairs: (A vs B) and (B vs A) are the same opportunity, so only the
    pair with the larger observed mean is emitted -- an agent forms the
    hypothesis in the direction the data suggests.
    """
    rows = project(rows)
    if not rows:
        return []
    agent = int(rows[0].get("agent_id", -1))
    ordered = sorted(rows, key=lambda r: (int(r["harvested_tick"]),
                                          int(r["trial_id"])))
    out: list[Opportunity] = []

    for var in variables:
        by_level: dict[Any, list[dict]] = {}
        for r in ordered:
            if var in r:
                by_level.setdefault(r[var], []).append(r)

        levels = sorted(by_level, key=str)
        for i, a in enumerate(levels):
            for b in levels[i + 1:]:
                ra, rb = by_level[a], by_level[b]
                if len(ra) < min_per_arm or len(rb) < min_per_arm:
                    continue
                # The opportunity arose when BOTH arms reached the threshold.
                arose = max(int(ra[min_per_arm - 1]["harvested_tick"]),
                            int(rb[min_per_arm - 1]["harvested_tick"]))
                ya = [float(r["yield_kg"]) for r in ra]
                yb = [float(r["yield_kg"]) for r in rb]
                ma, sa = _stats(ya)
                mb, sb = _stats(yb)
                pooled = math.sqrt((sa ** 2 + sb ** 2) / 2) or 1e-9
                hi, lo = (a, b) if ma >= mb else (b, a)
                delta = abs(ma - mb)
                out.append(Opportunity(
                    agent_id=agent, variable=var, level_a=hi, level_b=lo,
                    n_a=len(ra) if hi == a else len(rb),
                    n_b=len(rb) if hi == a else len(ra),
                    observed_delta=delta, observed_d=delta / pooled,
                    arose_tick=arose,
                ))
    return out


def claim_contrasts(claims: Iterable[Any]) -> set[tuple]:
    """What contrasts did this agent actually register claims about?

    A claim counts as addressing (variable, hi, lo) when its spec and baseline
    pin that variable to those two levels.
    """
    out: set[tuple] = set()
    for c in claims:
        spec = getattr(c, "spec", None)
        base = getattr(c, "baseline", None)
        if spec is None or base is None:
            continue
        for var, cond in spec.conditions.items():
            other = base.conditions.get(var)
            if other is None or cond.get("op") != "eq" or other.get("op") != "eq":
                continue
            out.add((var, cond["value"], other["value"]))
    return out


@dataclass
class AdoptionReport:
    agent_id: int
    raw_opportunities: int
    salient_opportunities: int
    registered_claims: int
    addressed_salient: int
    formal_protocol_adoption: float | None
    matched_adoption: float | None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "raw_opportunities": self.raw_opportunities,
            "salient_opportunities": self.salient_opportunities,
            "registered_claims": self.registered_claims,
            "addressed_salient": self.addressed_salient,
            "formal_protocol_adoption": self.formal_protocol_adoption,
            "matched_adoption": self.matched_adoption,
            **self.detail,
        }


def adoption_for_agent(
    rows: Iterable[dict],
    registered: Sequence[Any],
    *,
    min_per_arm: int = MIN_PER_ARM,
) -> AdoptionReport:
    """Adoption for one agent, using only what that agent could observe."""
    rows = list(rows)
    agent = int(rows[0]["agent_id"]) if rows else -1
    opps = opportunities(rows, min_per_arm=min_per_arm)
    salient = [o for o in opps if o.salient]
    addressed = claim_contrasts(registered)

    hit = sum(1 for o in salient if o.key() in addressed)

    # Undefined, not zero. An agent that never had a formulable comparison
    # cannot be scored on whether it acted on one.
    formal = (len(registered) / len(salient)) if salient else None
    matched = (hit / len(salient)) if salient else None

    return AdoptionReport(
        agent_id=agent,
        raw_opportunities=len(opps),
        salient_opportunities=len(salient),
        registered_claims=len(registered),
        addressed_salient=hit,
        formal_protocol_adoption=None if formal is None else round(formal, 4),
        matched_adoption=None if matched is None else round(matched, 4),
        detail={"min_per_arm": min_per_arm, "salience_d": SALIENCE_D},
    )


def adoption_report(
    rows_by_agent: dict[int, list[dict]],
    claims_by_agent: dict[int, list[Any]],
    *,
    min_per_arm: int = MIN_PER_ARM,
) -> dict:
    """Whole-run adoption. Agents with no opportunity are excluded from the
    aggregate rather than counted as zero."""
    per_agent = {
        a: adoption_for_agent(rows, claims_by_agent.get(a, []),
                              min_per_arm=min_per_arm)
        for a, rows in sorted(rows_by_agent.items())
    }
    scored = [r for r in per_agent.values() if r.matched_adoption is not None]
    return {
        "per_agent": {a: r.to_dict() for a, r in per_agent.items()},
        "agents_with_opportunity": len(scored),
        "agents_without_opportunity": len(per_agent) - len(scored),
        "mean_matched_adoption": (
            round(sum(r.matched_adoption for r in scored) / len(scored), 4)
            if scored else None
        ),
        "total_salient_opportunities": sum(
            r.salient_opportunities for r in per_agent.values()),
        "total_registered": sum(r.registered_claims for r in per_agent.values()),
    }
