"""The run report: exactly the quantities worth looking at first.

Readable from a partial checkpoint, so a run measured in hours can be inspected
while it is still going rather than only once it ends.

Three things this deliberately does NOT do:

  * It does not fill in a missing number with a plausible one. An unmeasurable
    quantity reports None, because a fabricated zero is worse than a gap.
  * It does not describe an agent that never had the chance to do something as
    having failed to do it.
  * It does not soften a null. If nothing was banked, that is the headline.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ..persistence.store import RunStore


def _pct(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def action_profile(store: RunStore, run_id: str) -> dict[str, Any]:
    """What the agents spent their days on, and how often the world said no."""
    rows = store.actions(run_id)
    if not rows:
        return {"actions": 0}

    verbs = Counter()
    rejected = Counter()
    for r in rows:
        verbs[json.loads(r["proposal_json"])["verb"]] += 1
        if not r["valid"]:
            rejected[r["rejection_code"]] += 1

    total = len(rows)
    invalid = sum(rejected.values())
    productive = verbs["PLANT"] + verbs["HARVEST"]
    return {
        "actions": total,
        "by_verb": dict(verbs.most_common()),
        "invalid_actions": invalid,
        "invalid_action_rate": _pct(invalid, total),
        "rejections_by_code": dict(rejected.most_common()),
        # Time actually spent farming rather than looking around. A run where
        # this is near zero produces no evidence and can measure nothing.
        "productive_action_rate": _pct(productive, total),
        "inspect_rate": _pct(verbs["INSPECT_TILE"], total),
        "noop_rate": _pct(verbs["NOOP"], total),
    }


def llm_health(store: RunStore, run_id: str) -> dict[str, Any]:
    """Whether the model was actually answering.

    A run with a high transport-failure rate is DEGRADED: those ticks are NOOPs
    the model never chose, and reporting them as decisions would attribute its
    silence to its judgement.
    """
    calls = [json.loads(e["payload_json"])
             for e in store.events(run_id, kind="llm_call", limit=200000)]
    if not calls:
        return {"llm_calls": 0, "note": "not a model-driven run"}

    status = Counter(c["call_status"] for c in calls)
    errors = Counter(c["error_class"] for c in calls if c.get("error_class"))
    lat = sorted(c["latency_ms"] for c in calls if c.get("latency_ms") is not None)
    total = len(calls)
    return {
        "llm_calls": total,
        "by_status": dict(status),
        "transport_failure_rate": _pct(status.get("transport_failure", 0), total),
        "parse_failure_rate": _pct(status.get("parse_failure", 0), total),
        "error_classes": dict(errors),
        "median_latency_ms": lat[len(lat) // 2] if lat else None,
        "degraded": status.get("transport_failure", 0) > 0.02 * total,
    }


def exploration(store: RunStore, run_id: str) -> dict[str, Any]:
    """How much of the parameter space was actually visited.

    An agent that plants the same thing forever generates evidence about one
    cell and can discover nothing, however many trials it runs.
    """
    trials = store.trials(run_id, limit=200000)
    if not trials:
        return {"trials": 0, "distinct_recipes": 0,
                "note": "no evidence was produced"}

    recipes = Counter()
    per_agent: dict[int, int] = {}
    tiles = set()
    for t in trials:
        params = json.loads(t["params_json"])
        recipes[tuple(sorted(params.items()))] += 1
        per_agent[t["agent_id"]] = per_agent.get(t["agent_id"], 0) + 1
        tiles.add(t["tile_id"])

    strata = Counter(t["stratum"] for t in trials)
    return {
        "trials": len(trials),
        "trials_per_agent": dict(sorted(per_agent.items())),
        "distinct_recipes": len(recipes),
        "distinct_tiles": len(tiles),
        "most_repeated_recipe_share": _pct(recipes.most_common(1)[0][1],
                                           len(trials)),
        "trials_by_stratum": dict(strata),
        # A claim needs min_trials_per_group in each of two arms. Below that,
        # nothing can reach SUPPORTED no matter how good the agents are.
        "evidence_sufficient_for_a_claim": len(trials) >= 32,
    }


def knowledge(store: RunStore, run_id: str) -> dict[str, Any]:
    claims = store.claims(run_id)
    states = Counter(c["state"] for c in claims)
    return {
        "claims_proposed": len(claims),
        "by_state": dict(states),
        "supported": states.get("supported", 0),
        "confirmed": states.get("confirmed", 0),
        "generalized": states.get("generalized", 0),
        "refuted": states.get("refuted", 0),
        "banked": states.get("confirmed", 0) + states.get("generalized", 0),
        "replication_kinds": dict(Counter(
            c["replication_kind"] for c in claims if c["replication_kind"])),
    }


def survival(store: RunStore, run_id: str) -> dict[str, Any]:
    """Hunger is the only strong selection pressure in this world, so whether
    agents stayed fed says what they were optimising under."""
    events = store.events(run_id, kind="hunger", limit=200000)
    per_agent = Counter(e["agent_id"] for e in events)
    return {
        "hunger_events": len(events),
        "hunger_by_agent": dict(sorted(per_agent.items())),
        "agents_that_went_hungry": len(per_agent),
    }


def goals(store: RunStore, run_id: str) -> dict[str, Any]:
    """Goals are self-authored, never scored. Diversity is a signal about the
    population, not a target anyone was steered toward."""
    events = store.events(run_id, kind="goal_set", limit=200000)
    texts = [json.loads(e["payload_json"]).get("text", "") for e in events]
    distinct = {t.strip().lower() for t in texts if t.strip()}
    return {
        "goals_set": len(texts),
        "distinct_goals": len(distinct),
        "goal_diversity": _pct(len(distinct), len(texts)) if texts else None,
        "sample": sorted(distinct)[:5],
    }


def report(store: RunStore, run_id: str) -> dict[str, Any]:
    meta = store.run(run_id)
    if meta is None:
        return {"error": f"no run {run_id!r}"}

    out = {
        "run_id": run_id,
        "ticks_recorded": len(store.state_hashes(run_id)),
        "ticks_requested": meta["ticks"],
        "policy": meta["policy"],
        "domain": meta["domain"],
        "seed": meta["seed"],
        "manifest": json.loads(meta["manifest_json"] or "{}"),
        "actions": action_profile(store, run_id),
        "llm": llm_health(store, run_id),
        "exploration": exploration(store, run_id),
        "knowledge": knowledge(store, run_id),
        "survival": survival(store, run_id),
        "goals": goals(store, run_id),
    }
    out["complete"] = out["ticks_recorded"] >= out["ticks_requested"]
    out["headline"] = _headline(out)
    return out


def _headline(r: dict[str, Any]) -> str:
    """One sentence, stated plainly, including when the answer is nothing."""
    ex, kn, ll = r["exploration"], r["knowledge"], r["llm"]
    partial = "" if r["complete"] else (
        f" (PARTIAL: {r['ticks_recorded']}/{r['ticks_requested']} ticks)")

    if ll.get("degraded"):
        return (f"DEGRADED RUN{partial}: {ll['transport_failure_rate']:.1%} of "
                f"calls failed in transport, so those ticks are silence rather "
                f"than judgement")
    if ex.get("trials", 0) == 0:
        return (f"NO EVIDENCE{partial}: the agents produced no completed trials, "
                f"so nothing could be verified either way")
    if kn["banked"] == 0:
        return (f"NOTHING BANKED{partial}: {ex['trials']} trials and "
                f"{kn['claims_proposed']} claims, none of which survived "
                f"verification")
    return (f"{kn['banked']} claim(s) banked from {ex['trials']} trials"
            f"{partial}")
