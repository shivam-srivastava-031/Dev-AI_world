"""Civilization-level metrics.

This is the only package permitted to import a domain, because grading a
discovery requires ground truth. That import is exactly what lets us say
"false discovery rate" rather than "claims we happened to believe".

Every number here is designed so a model narrating progress cannot move it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from ..civilization.trust import BeliefStatus
from ..knowledge.claim import ClaimState
from ..knowledge.replication import INDEPENDENT_KINDS, ReplicationKind
from ..world.domains.base import Domain, TileContext

BANKED = (ClaimState.CONFIRMED, ClaimState.GENERALIZED)
ESTABLISHED = (ClaimState.SUPPORTED, *BANKED, ClaimState.CONTESTED)

COMPETENCE_WINDOW = 10        # consecutive trials
COMPETENCE_FRACTION = 0.85    # of the best mu reachable


# --- ground truth (METRICS_ONLY) ------------------------------------------

def true_effect(domain: Domain, spec, baseline, *, soil: int = 2,
                skill: float = 0.5) -> float:
    """The claim's REAL effect, averaged over every recipe in each region.

    Uses the hidden function. This is how a "verified discovery" gets graded
    true positive or false positive instead of merely counted.
    """
    def region_mean(s) -> float:
        vals = [domain.true_mu(r, TileContext(0, soil), skill)
                for r in domain.enumerate_recipes() if s.matches(r)]
        return float(np.mean(vals)) if vals else 0.0
    return region_mean(spec) - region_mean(baseline)


def grade_claims(domain: Domain, kb, *, tolerance: float = 0.5) -> dict[str, Any]:
    """Split banked claims into true and false positives against ground truth."""
    tp, fp, rows = 0, 0, []
    for r in kb:
        if r.state not in BANKED:
            continue
        te = true_effect(domain, r.claim.spec, r.claim.baseline)
        ok = te >= tolerance * r.claim.predicted_min_delta
        tp, fp = (tp + 1, fp) if ok else (tp, fp + 1)
        rows.append({
            "claim_id": str(r.claim.claim_id),
            "state": r.state.value,
            "true_effect": round(te, 4),
            "claimed_min_delta": r.claim.predicted_min_delta,
            "true_positive": ok,
        })
    banked = tp + fp
    return {
        "banked": banked,
        "true_positives": tp,
        "false_positives": fp,
        # A headline number, never a footnote.
        "false_discovery_rate": round(fp / banked, 4) if banked else None,
        "graded": rows,
    }


# --- knowledge -------------------------------------------------------------

def knowledge_metrics(kb) -> dict[str, Any]:
    counts = kb.counts()
    by_hash: dict[str, set[int]] = defaultdict(set)
    for r in kb:
        by_hash[r.claim.spec_hash].add(int(r.claim.author))
    rediscovered = sum(1 for authors in by_hash.values() if len(authors) > 1)

    profile = Counter()
    independent = 0
    for r in kb:
        if r.state in BANKED and r.replication_kind:
            profile[r.replication_kind] += 1
            if ReplicationKind(r.replication_kind) in INDEPENDENT_KINDS:
                independent += 1
    banked = sum(counts[s.value] for s in BANKED)
    return {
        "claims_total": len(kb),
        "by_state": {k: v for k, v in counts.items() if v},
        "established": sum(counts[s.value] for s in ESTABLISHED),
        "banked": banked,
        "generalized": counts[ClaimState.GENERALIZED.value],
        "refuted": counts[ClaimState.REFUTED.value],
        "distinct_techniques": len(by_hash),
        "independently_rediscovered": rediscovered,
        "confirmation_independence_profile": dict(profile),
        # A claim confirmed only by taught replications shows DIFFUSION, which
        # is a different result from independent corroboration.
        "independently_corroborated": independent,
        "diffusion_only": banked - independent,
    }


# --- social ----------------------------------------------------------------

def social_metrics(engine) -> dict[str, Any]:
    rel = engine.relationships.report()
    statuses: Counter = Counter()
    blind = 0
    taught_total = 0
    for a in engine.state.agent_ids():
        for b in engine.beliefs[int(a)].all():
            statuses[b.status.value] += 1
            if b.source is not None:
                taught_total += 1
                # Adopted on hearsay and never tested: the failure mode that
                # makes a civilization confidently wrong.
                if b.status is BeliefStatus.HEARSAY and b.adopted_tick is not None:
                    blind += 1
    return {
        **rel,
        "beliefs_by_status": dict(statuses),
        "taught_beliefs": taught_total,
        "blind_adoption": blind,
        "blind_adoption_rate": (round(blind / taught_total, 4)
                                if taught_total else None),
        "correction_events": rel["misleading_teach_events"],
    }


# --- competence and regret -------------------------------------------------

REFERENCE_SKILL = 1.0     # both sides of every comparison are evaluated here


def controlled_axes(domain: Domain) -> tuple[str, ...]:
    """Parameters an agent genuinely chooses.

    The schedule axis is excluded: it is set by WHEN the agent happens to
    act, so scoring against an optimum that includes it measures scheduling
    luck rather than knowledge.
    """
    return tuple(n for n in sorted(domain.param_space)
                 if n != domain.schedule_axis)


def _day_averaged_mu(domain: Domain, base: dict, soil: int,
                     tile_id: int = 0) -> float:
    days = domain.param_space[domain.schedule_axis].values
    return float(np.mean([
        domain.true_mu({**base, domain.schedule_axis: d},
                       TileContext(tile_id, soil), REFERENCE_SKILL)
        for d in days
    ]))


def controllable_optimum(domain: Domain, soil: int = 2) -> float:
    """Best mu over the parameters an agent actually controls, averaged over
    the one it does not."""
    axes = controlled_axes(domain)
    best = -np.inf
    seen: set[tuple] = set()
    for r in domain.enumerate_recipes():
        key = tuple(r[k] for k in axes)
        if key in seen:
            continue
        seen.add(key)
        best = max(best, _day_averaged_mu(domain, {k: r[k] for k in axes}, soil))
    return float(best)


def choice_mu(domain: Domain, trial: Any, *, soil: int | None = None) -> float:
    """The trial's CHOICE re-evaluated at a reference skill, averaged over day.

    Two things are stripped out so this measures knowledge rather than luck or
    practice: the agent's skill (both sides use REFERENCE_SKILL) and the
    planting day (averaged over, because the agent did not pick it). What is
    left is: did this agent know what to plant and how to water it.
    """
    base = {k: v for k, v in trial.params.items()
            if k != domain.schedule_axis}
    band = trial.soil_band if soil is None else soil
    return _day_averaged_mu(domain, base, band, int(trial.tile_id))


def time_to_competence(domain: Domain, trials: Sequence[Any], agent_id: int,
                       *, soil: int = 2) -> int | None:
    """First tick where the trailing K consecutive trials average >= 85% of the
    best recipe, judged on TRUE mu at a fixed reference skill.

    True mu rather than observed yield: measurement noise is not the agent's
    doing, and a run of luck must not read as competence. Sustained over a
    window, so one good harvest does not qualify either. And at a reference
    skill, so this measures knowledge rather than practice.
    """
    own = sorted((t for t in trials if int(t.agent_id) == agent_id),
                 key=lambda t: (t.harvested_tick, t.trial_id))
    if len(own) < COMPETENCE_WINDOW:
        return None
    best_mu = controllable_optimum(domain, soil)
    target = COMPETENCE_FRACTION * best_mu
    window: list[float] = []
    for t in own:
        window.append(choice_mu(domain, t, soil=soil))
        if len(window) > COMPETENCE_WINDOW:
            window.pop(0)
        if len(window) == COMPETENCE_WINDOW and sum(window) / COMPETENCE_WINDOW >= target:
            return int(t.harvested_tick)
    return None


def collective_regret(domain: Domain, trials: Sequence[Any], *, soil: int = 2,
                      last_n: int = 100) -> float:
    """How far the civilization's recipe CHOICES sit from the true optimum.

    Also evaluated at the reference skill: regret should fall because agents
    learned something, not because they got stronger at doing the wrong thing.
    """
    if not trials:
        return float("nan")
    best_mu = controllable_optimum(domain, soil)
    recent = sorted(trials, key=lambda t: t.harvested_tick)[-last_n:]
    achieved = float(np.mean([choice_mu(domain, t, soil=soil) for t in recent]))
    return round(best_mu - achieved, 4)


# --- resilience -------------------------------------------------------------

def knowledge_resilience(engine) -> dict[str, Any]:
    """Counterfactually remove the most knowledgeable agent. What survives?

    A civilization whose knowledge lives in one head is one accident away from
    losing it. Computed on a finished run: a claim survives if someone other
    than the ablated agent authored it, replicated it, or personally confirmed
    it.
    """
    banked = [r for r in engine.kb if r.state in BANKED]
    if not banked:
        return {"banked": 0, "surviving": 0, "resilience": None,
                "most_knowledgeable": None}

    authored = Counter(int(r.claim.author) for r in banked)
    if not authored:
        return {"banked": len(banked), "surviving": len(banked), "resilience": 1.0,
                "most_knowledgeable": None}
    victim = authored.most_common(1)[0][0]

    surviving = 0
    for r in banked:
        holders = {int(r.claim.author)}
        if r.replicator is not None:
            holders.add(int(r.replicator))
        for a in engine.state.agent_ids():
            b = engine.beliefs[int(a)].get(str(r.claim.claim_id))
            if b is not None and b.status is BeliefStatus.PERSONALLY_CONFIRMED:
                holders.add(int(a))
        if holders - {victim}:
            surviving += 1

    return {
        "banked": len(banked),
        "most_knowledgeable": victim,
        "surviving": surviving,
        "resilience": round(surviving / len(banked), 4),
        "concentration": round(authored.most_common(1)[0][1] / len(banked), 4),
    }


# --- the whole report -------------------------------------------------------

def civilization_report(engine, domain: Domain) -> dict[str, Any]:
    trials = engine.trials
    agents = [int(a) for a in engine.state.agent_ids()]

    t2c = {a: time_to_competence(domain, trials, a) for a in agents}
    reached = [v for v in t2c.values() if v is not None]

    actions_by_agent: dict[int, list[str]] = defaultdict(list)
    for rec in engine.history:
        for a, proposal in rec.proposals:
            actions_by_agent[a].append(proposal.verb.value)

    from ..civilization.specialization import analyse, environment_for
    spec = analyse(actions_by_agent,
                   {a: environment_for(engine.state, a) for a in agents})

    return {
        "knowledge": knowledge_metrics(engine.kb),
        "grading": grade_claims(domain, engine.kb),
        "social": social_metrics(engine),
        "language": engine.novelty.report(),
        "competence": {
            "time_to_competence": t2c,
            "agents_reaching_competence": len(reached),
            "t2c_first": min(reached) if reached else None,
            "t2c_median": (int(np.median(reached)) if reached else None),
            "t2c_last": max(reached) if reached else None,
        },
        "collective": {
            "trials": len(trials),
            "collective_regret": collective_regret(domain, trials),
            "total_yield": round(sum(float(t.yield_kg) for t in trials), 2),
        },
        "resilience": knowledge_resilience(engine),
        "specialization": spec.to_dict(),
    }
