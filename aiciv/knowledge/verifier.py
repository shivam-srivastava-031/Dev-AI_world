"""The Verifier: the only thing in the system that may declare something true.

Authority is enforced structurally, not by convention. ``KnowledgeBase.set_state``
demands a ``VerifierToken``, and a token can only be constructed inside
``Verifier.__init__``. No policy, no agent, no action handler can reach one.

Evidence admissibility, in order -- each rule closes a specific attack:

  signature valid        an agent with write access to the DB still cannot forge
  harvested after        pre-registration; kills post-hoc cherry-picking
    registration
  correct stratum        the stage decides which tiles count
  not already consumed   a trial cannot be spent twice on the same claim
  author != replicator   an author cannot self-replicate (CONFIRMED stage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

from ..ids import AgentId, ClaimId
from ..spec import VerificationSpec
from .analysis import TestResult, ancova, benjamini_hochberg, welch
from .causal import (
    DEFAULT_CAUSAL_COVARIATES, DEFAULT_COMPARISON_COVARIATES, block_by_skill,
)
from .claim import ClaimRecord, ClaimState, ClaimType, KnowledgeClaim
from .replication import (
    CommunicationLog, ReplicationKind, classify_replication,
)

STAGE_SUPPORTED = "supported"
STAGE_CONFIRMED = "confirmed"
STAGE_GENERALIZED = "generalized"

STAGE_STRATUM = {
    STAGE_SUPPORTED: "discovery",
    STAGE_CONFIRMED: "confirmation",
    STAGE_GENERALIZED: "holdout",
}


class VerifierToken:
    """Proof that a state change came from the Verifier.

    Construction is guarded: ``Verifier.__init__`` flips the class-level latch
    for exactly one instantiation. Anything else raises, so a policy that
    imports this module and tries ``VerifierToken()`` gets an error rather than
    write access to the knowledge base.
    """

    _armed = False

    def __init__(self) -> None:
        if not VerifierToken._armed:
            raise PermissionError(
                "VerifierToken may only be constructed by the Verifier; "
                "claim state is not agent-writable"
            )
        VerifierToken._armed = False


@dataclass
class Verdict:
    stage: str
    passed: bool
    reason: str
    result: TestResult | None = None
    n_treat: int = 0
    n_ctrl: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"stage": self.stage, "passed": self.passed, "reason": self.reason,
             "n_treat": self.n_treat, "n_ctrl": self.n_ctrl, **self.detail}
        if self.result is not None:
            d |= {
                "effect": round(self.result.effect, 4),
                "ci_low": round(self.result.ci_low, 4),
                "ci_high": round(self.result.ci_high, 4),
                "p_value": float(f"{self.result.p_value:.4g}"),
                "hedges_g": round(self.result.hedges_g, 4),
                "test": self.result.test_name,
            }
        return d


# --- covariate assembly ---------------------------------------------------

def _covariate_matrix(rows: Sequence[dict], names: Sequence[str]) -> np.ndarray:
    """Numeric design columns. plant_day becomes sin/cos: it is cyclic, and
    treating day 11 and day 0 as 11 apart would be wrong."""
    cols: list[np.ndarray] = []
    for name in names:
        if name == "plant_day":
            d = np.array([float(r["plant_day"]) for r in rows])
            cols.append(np.sin(2 * np.pi * d / 12))
            cols.append(np.cos(2 * np.pi * d / 12))
        else:
            cols.append(np.array([float(r[name]) for r in rows]))
    if not cols:
        return np.empty((len(rows), 0))
    return np.column_stack(cols)


def _yields(rows: Sequence[dict]) -> np.ndarray:
    return np.array([float(r["yield_kg"]) for r in rows])


# --- per-type analyses ----------------------------------------------------

def analyse(
    claim: KnowledgeClaim,
    treat: Sequence[dict],
    ctrl: Sequence[dict],
    spec: VerificationSpec,
) -> TestResult:
    """Run the analysis the claim's TYPE binds it to.

    A CAUSAL_EFFECT cannot be settled by the COMPARISON path just because the
    numbers came out nicely: it uses a restricted covariate set that excludes
    treatment-affected variables, and blocks on skill instead of adjusting.
    """
    ct = claim.claim_type

    if ct is ClaimType.OBSERVATION:
        return _analyse_observation(claim, treat, ctrl)
    if ct is ClaimType.INTERACTION:
        return _analyse_interaction(claim, treat, ctrl)
    if ct is ClaimType.CAUSAL_EFFECT:
        return _analyse_causal(claim, treat, ctrl, spec)
    # COMPARISON, OPTIMAL_RANGE and GENERALIZATION all reduce to a two-group
    # contrast; OPTIMAL_RANGE additionally checks adjacency in the verifier.
    return _analyse_comparison(claim, treat, ctrl, spec)


def _orient(claim: KnowledgeClaim, a: np.ndarray, b: np.ndarray):
    """A 'decrease' claim is the same test with the arms swapped."""
    return (b, a) if claim.predicted_direction == "decrease" else (a, b)


def _analyse_comparison(claim, treat, ctrl, spec) -> TestResult:
    yt, yc = _yields(treat), _yields(ctrl)
    a, b = _orient(claim, yt, yc)
    if spec.balance_mode == "off":
        return welch(a, b)

    names = claim.adjustment_set or DEFAULT_COMPARISON_COVARIATES
    names = tuple(n for n in names if all(n in r for r in list(treat) + list(ctrl)))
    rows = list(treat) + list(ctrl)
    if claim.predicted_direction == "decrease":
        rows = list(ctrl) + list(treat)
    y = np.concatenate([a, b])
    t = np.array([1] * len(a) + [0] * len(b))
    return ancova(y, t, _covariate_matrix(rows, names))


def _analyse_causal(claim, treat, ctrl, spec) -> TestResult:
    """Blocked on skill, adjusted only for genuinely pre-treatment covariates.

    Blocking rather than adjustment is the whole point: skill is downstream of
    treatment across a trial sequence, so conditioning on it in the regression
    can absorb the effect being estimated.
    """
    names = tuple(claim.adjustment_set) or DEFAULT_CAUSAL_COVARIATES
    names = tuple(n for n in names if all(n in r for r in list(treat) + list(ctrl)))

    blocks_t = block_by_skill(list(treat))
    blocks_c = block_by_skill(list(ctrl))

    usable = [b for b in blocks_t
              if len(blocks_t[b]) >= 2 and len(blocks_c.get(b, [])) >= 2]
    if not usable:
        # Too little overlap to block; fall back to the unadjusted contrast and
        # say so, rather than silently adjusting for a post-treatment variable.
        yt, yc = _orient(claim, _yields(treat), _yields(ctrl))
        r = welch(yt, yc)
        return TestResult(**{**r.__dict__, "test_name": "welch_unblocked",
                             "detail": {"reason": "no overlapping skill blocks"}})

    rows_t = [r for b in usable for r in blocks_t[b]]
    rows_c = [r for b in usable for r in blocks_c[b]]
    a, b_ = _orient(claim, _yields(rows_t), _yields(rows_c))
    rows = rows_t + rows_c
    if claim.predicted_direction == "decrease":
        rows = rows_c + rows_t

    block_id = np.array([
        float(next(k for k, v in (blocks_t if r in rows_t else blocks_c).items()
                   if r in v))
        for r in rows
    ])
    cov = _covariate_matrix(rows, names)
    cov = np.column_stack([cov, block_id]) if cov.size else block_id.reshape(-1, 1)
    y = np.concatenate([a, b_])
    t = np.array([1] * len(a) + [0] * len(b_))
    res = ancova(y, t, cov)
    return TestResult(**{**res.__dict__, "test_name": "ancova_blocked"})


def _analyse_observation(claim, treat, ctrl) -> TestResult:
    """Omnibus F: does the outcome vary across levels at all?"""
    var = claim.spec.variables[0] if claim.spec.variables else None
    rows = list(treat) + list(ctrl)
    if var is None or len(rows) < 4:
        return TestResult(0.0, -np.inf, np.inf, 1.0, 0.0, len(treat), len(ctrl),
                          0.0, "anova", detail={"error": "no variable"})
    groups: dict[Any, list[float]] = {}
    for r in rows:
        groups.setdefault(r[var], []).append(float(r["yield_kg"]))
    usable = [np.array(v) for v in groups.values() if len(v) >= 2]
    if len(usable) < 2:
        return TestResult(0.0, -np.inf, np.inf, 1.0, 0.0, len(treat), len(ctrl),
                          0.0, "anova", detail={"error": "too few levels"})
    f, p = stats.f_oneway(*usable)
    spread = float(max(np.mean(g) for g in usable) - min(np.mean(g) for g in usable))
    return TestResult(spread, 0.0, spread, float(p), 0.0, len(treat), len(ctrl),
                      float(len(rows) - len(usable)), "anova",
                      detail={"f": float(f), "levels": len(usable)})


def _analyse_interaction(claim, treat, ctrl) -> TestResult:
    """Fit y ~ a + b + a:b and test the interaction term.

    Requires the spec to constrain exactly two variables, and all four cells to
    be populated -- an interaction claim asserted from three cells is not an
    interaction claim.
    """
    vars_ = claim.spec.variables
    rows = list(treat) + list(ctrl)
    if len(vars_) < 2 or len(rows) < 8:
        return TestResult(0.0, -np.inf, np.inf, 1.0, 0.0, len(treat), len(ctrl),
                          0.0, "interaction",
                          detail={"error": "needs two variables and 8+ trials"})
    va, vb = vars_[0], vars_[1]
    a = np.array([float(_num(r[va])) for r in rows])
    b = np.array([float(_num(r[vb])) for r in rows])
    a = (a - a.mean())
    b = (b - b.mean())
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return TestResult(0.0, -np.inf, np.inf, 1.0, 0.0, len(treat), len(ctrl),
                          0.0, "interaction", detail={"error": "no variation"})
    y = _yields(rows)
    # ancova() tests column 1, so the interaction goes there and the main
    # effects become covariates.
    res = ancova(y, a * b, np.column_stack([a, b]))
    return TestResult(**{**res.__dict__, "test_name": "interaction"})


def _num(v: Any) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return float(abs(hash(str(v))) % 997)      # stable categorical coding


# --- the verifier ---------------------------------------------------------

class Verifier:
    """Runs a verification round at the end of every tick."""

    def __init__(self, spec: VerificationSpec, log: CommunicationLog | None = None):
        self.spec = spec
        self.log = log or CommunicationLog()
        VerifierToken._armed = True
        self.token = VerifierToken()

    # -- evidence selection -------------------------------------------------

    def eligible(
        self,
        record: ClaimRecord,
        rows: Sequence[dict],
        stage: str,
        run_secret: bytes,
        trials_by_id: dict[int, Any],
    ) -> tuple[list[dict], list[dict]]:
        """Admissible treatment and control rows for one stage."""
        claim = record.claim
        stratum = STAGE_STRATUM[stage]
        consumed = record.consumed.setdefault(stage, set())
        reg = record.registered_tick

        treat: list[dict] = []
        ctrl: list[dict] = []
        for r in rows:
            if r["stratum"] != stratum:
                continue
            if reg is None or r["harvested_tick"] <= reg:
                continue                       # pre-registration rule
            if int(r["trial_id"]) in consumed:
                continue
            trial = trials_by_id.get(int(r["trial_id"]))
            if trial is None or not trial.verify(run_secret):
                continue                       # forged or unknown
            if claim.spec.matches(r):
                treat.append(r)
            elif claim.baseline.matches(r):
                ctrl.append(r)
        return treat, ctrl

    # -- one stage ----------------------------------------------------------

    def evaluate_stage(
        self, record: ClaimRecord, treat: list[dict], ctrl: list[dict],
        stage: str, alpha: float,
    ) -> Verdict:
        spec = self.spec
        n_t, n_c = len(treat), len(ctrl)

        if n_t < spec.min_trials_per_group or n_c < spec.min_trials_per_group:
            return Verdict(stage, False, "insufficient_evidence", None, n_t, n_c)

        tiles = {r["tile_id"] for r in treat} | {r["tile_id"] for r in ctrl}
        if len(tiles) < spec.min_tiles:
            return Verdict(stage, False, "insufficient_tile_diversity", None, n_t, n_c,
                           {"tiles": len(tiles)})

        if stage == STAGE_CONFIRMED:
            author = int(record.claim.author)
            replicators = {int(r["agent_id"]) for r in treat} - {author}
            if not replicators:
                return Verdict(stage, False, "author_cannot_self_replicate",
                               None, n_t, n_c)
            # Diversity is counted across the claim's WHOLE evidence base, not
            # the confirmation arm alone: the author supplied the discovery
            # stratum and the replicator the confirmation stratum, so one
            # replicator already means two independent hands on the evidence.
            contributors = replicators | {author}
            if len(contributors) < spec.min_agents:
                return Verdict(stage, False, "insufficient_agent_diversity",
                               None, n_t, n_c, {"contributors": len(contributors)})

        result = analyse(record.claim, treat, ctrl, spec)

        if result.p_value >= alpha:
            return Verdict(stage, False, "not_significant", result, n_t, n_c)
        if record.claim.claim_type is not ClaimType.OBSERVATION:
            if result.hedges_g < spec.min_effect_size:
                return Verdict(stage, False, "effect_too_small", result, n_t, n_c)
            if result.effect < 0.6 * record.claim.predicted_min_delta:
                return Verdict(stage, False, "below_claimed_delta", result, n_t, n_c)

        return Verdict(stage, True, "passed", result, n_t, n_c)

    # -- a full round -------------------------------------------------------

    def run_round(
        self, kb, rows_by_claim, tick: int, run_secret: bytes,
        trials_by_id: dict[int, Any],
    ) -> list[tuple[ClaimId, ClaimState, Verdict]]:
        """Advance every open claim by at most one stage.

        Candidate verdicts are collected first and their p-values corrected
        together with Benjamini-Hochberg: five agents registering hypotheses
        freely WILL generate false positives at a raw alpha, and correcting
        per-claim in isolation would not account for that.
        """
        candidates: list[tuple[ClaimRecord, str, Verdict, ClaimState]] = []

        for record in kb.open_records():
            claim = record.claim
            if record.state is ClaimState.REGISTERED:
                kb.set_state(claim.claim_id, ClaimState.TESTING, self.token, tick)

            stage, alpha, nxt = self._next_stage(record)
            if stage is None:
                continue

            rows = rows_by_claim(record)
            treat, ctrl = self.eligible(record, rows, stage, run_secret, trials_by_id)
            verdict = self.evaluate_stage(record, treat, ctrl, stage, alpha)

            if verdict.passed:
                candidates.append((record, stage, verdict, nxt))
            else:
                record.verdicts[stage] = verdict.to_dict()
                self._maybe_refute(kb, record, treat, ctrl, tick)

        if candidates:
            flags = benjamini_hochberg(
                [c[2].result.p_value for c in candidates], self.spec.alpha)
            for (record, stage, verdict, nxt), ok in zip(candidates, flags):
                if not ok:
                    verdict.passed = False
                    verdict.reason = "failed_multiplicity_correction"
                    record.verdicts[stage] = verdict.to_dict()
                    continue
                verdict.detail["p_adjusted_pass"] = True
                record.verdicts[stage] = verdict.to_dict()
                record.consumed.setdefault(stage, set()).update(
                    int(r["trial_id"]) for r in
                    self.eligible(record, rows_by_claim(record), stage,
                                  run_secret, trials_by_id)[0])
                if stage == STAGE_CONFIRMED:
                    self._attach_replication(record, rows_by_claim(record))
                kb.set_state(record.claim.claim_id, nxt, self.token, tick)

        return [(c[0].claim.claim_id, c[3], c[2]) for c in candidates]

    def _next_stage(self, record: ClaimRecord):
        if record.state in (ClaimState.TESTING, ClaimState.CONTESTED):
            return STAGE_SUPPORTED, self.spec.alpha, ClaimState.SUPPORTED
        if record.state is ClaimState.SUPPORTED:
            return STAGE_CONFIRMED, self.spec.confirmation_alpha, ClaimState.CONFIRMED
        if record.state is ClaimState.CONFIRMED:
            return STAGE_GENERALIZED, self.spec.holdout_alpha, ClaimState.GENERALIZED
        return None, None, None

    def _maybe_refute(self, kb, record, treat, ctrl, tick) -> None:
        total = len(record.consumed.get(STAGE_SUPPORTED, set())) + len(treat) + len(ctrl)
        if total <= self.spec.max_trials:
            return
        if record.state in (ClaimState.TESTING, ClaimState.CONTESTED):
            kb.set_state(record.claim.claim_id, ClaimState.REFUTED, self.token, tick)
            record.notes.append(
                f"refuted at tick {tick}: {total} trials without reaching "
                f"significance (budget {self.spec.max_trials})")

    def _attach_replication(self, record: ClaimRecord, rows) -> None:
        claim = record.claim
        others = sorted({int(r["agent_id"]) for r in rows
                         if claim.spec.matches(r)} - {int(claim.author)})
        if not others:
            return
        rep = others[0]
        first = min(int(r["harvested_tick"]) for r in rows
                    if claim.spec.matches(r) and int(r["agent_id"]) == rep)
        kind = classify_replication(
            author=claim.author, replicator=AgentId(rep),
            claim_id=claim.claim_id, first_matching_trial_tick=first, log=self.log)
        record.replicator = rep
        record.replication_kind = kind.value if kind else None
