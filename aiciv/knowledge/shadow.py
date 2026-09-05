"""The shadow verifier: measurement that does not depend on agent cooperation.

The scientific protocol is an affordance, not a mandate (docs/protocol.md 2.2).
An agent may farm for four hundred ticks and never file a claim. If measurement
depended on agents filing claims, "the civilization discovered nothing" and "the
civilization did not use our paperwork" would be indistinguishable.

So this runs continuously over world-recorded trials regardless, asking: given
the evidence that physically exists in this world, what WOULD pass each stage?

It is held to the same standard it measures. It does not data-dredge: candidates
are generated from an early slice of trials and tested on a later, disjoint
slice, which is the retrospective analogue of pre-registration. Without that
split it would find "discoveries" everywhere and the comparison would be
meaningless.

KNOWN LIMITATION -- read before quoting protocol_adoption_rate
--------------------------------------------------------------
This scanner only forms MARGINAL single-variable hypotheses ("BEANS beats
NONE", averaged over every spacing). It cannot see the companion x spacing
interaction at all, which is the actual structure of the synthetic domain.

So a claim an agent makes ("BEANS at spacing 2 beats NONE at spacing 2") is
strictly RICHER than anything this scanner can propose. The two counts are
therefore not like-for-like, and `protocol_adoption_rate` should currently be
read as "did the civilization bank roughly as much as the raw evidence
supports", not as a clean efficiency ratio.

Making it like-for-like means scanning conditioned hypotheses -- every
(variable, level, conditioning-set) triple -- which multiplies the candidate
count and needs a stricter correction. That is deferred, and until it lands
this ratio is reported with the caveat attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..spec import VerificationSpec
from .analysis import TestResult, benjamini_hochberg, welch
from .claim import TechniqueSpec

#: Parameters the shadow verifier will form hypotheses about.
SCANNED = ("companion", "spacing", "water")


@dataclass
class ShadowFinding:
    variable: str
    spec: TechniqueSpec
    baseline: TechniqueSpec
    discovery: TestResult
    confirmation: TestResult | None = None
    stage: str = "none"                   # none | supported | confirmed
    detail: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (f"{self.spec.describe()} > {self.baseline.describe()} "
                f"[{self.stage}] effect={self.discovery.effect:+.3f} "
                f"p={self.discovery.p_value:.4g}")


def _eq(var: str, value: Any) -> TechniqueSpec:
    return TechniqueSpec({var: {"op": "eq", "value": value}})


def _split(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Chronological split. Early trials generate hypotheses, late trials test
    them, and no trial is ever used for both."""
    ordered = sorted(rows, key=lambda r: (int(r["harvested_tick"]),
                                          int(r["trial_id"])))
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]


class ShadowVerifier:
    def __init__(self, spec: VerificationSpec) -> None:
        self.spec = spec

    def scan(self, rows: Sequence[dict]) -> list[ShadowFinding]:
        """What the evidence in this world would support, agents notwithstanding."""
        early, late = _split(rows)
        if len(early) < 2 * self.spec.min_trials_per_group:
            return []

        candidates: list[ShadowFinding] = []
        for var in SCANNED:
            levels = sorted({r[var] for r in early if var in r}, key=str)
            for i, a in enumerate(levels):
                for b in levels:
                    if a == b:
                        continue
                    f = self._test_pair(var, a, b, early)
                    if f is not None:
                        candidates.append(f)

        if not candidates:
            return []

        # Correct across everything we looked at, not per-hypothesis: scanning
        # every level pair is exactly the multiplicity problem BH exists for.
        flags = benjamini_hochberg([c.discovery.p_value for c in candidates],
                                   self.spec.alpha)
        supported = [c for c, ok in zip(candidates, flags) if ok]
        for f in supported:
            f.stage = "supported"
            self._confirm(f, late)
        return supported

    def _test_pair(self, var: str, a: Any, b: Any,
                   rows: Sequence[dict]) -> ShadowFinding | None:
        treat = [r for r in rows if r.get(var) == a]
        ctrl = [r for r in rows if r.get(var) == b]
        n = self.spec.min_trials_per_group
        if len(treat) < n or len(ctrl) < n:
            return None
        res = welch([float(r["yield_kg"]) for r in treat],
                    [float(r["yield_kg"]) for r in ctrl])
        if res.effect <= 0 or res.hedges_g < self.spec.min_effect_size:
            return None
        return ShadowFinding(var, _eq(var, a), _eq(var, b), res)

    def _confirm(self, finding: ShadowFinding, late: Sequence[dict]) -> None:
        var = finding.variable
        a = finding.spec.conditions[var]["value"]
        b = finding.baseline.conditions[var]["value"]
        treat = [r for r in late if r.get(var) == a]
        ctrl = [r for r in late if r.get(var) == b]
        n = self.spec.min_trials_per_group
        if len(treat) < n or len(ctrl) < n:
            finding.detail["confirmation"] = "insufficient_late_evidence"
            return
        res = welch([float(r["yield_kg"]) for r in treat],
                    [float(r["yield_kg"]) for r in ctrl])
        finding.confirmation = res
        if res.p_value < self.spec.confirmation_alpha and res.effect > 0:
            finding.stage = "confirmed"


def adoption_metrics(agent_confirmed: int, shadow_confirmed: int) -> dict[str, Any]:
    """How much of what was discoverable did the civilization actually bank?

    A low ratio is a real finding, not a bug: it means the evidence existed and
    the agents did not turn it into public knowledge.
    """
    return {
        "shadow_confirmed": shadow_confirmed,
        "agent_confirmed": agent_confirmed,
        "protocol_adoption_rate": (
            round(agent_confirmed / shadow_confirmed, 4)
            if shadow_confirmed else None
        ),
    }
