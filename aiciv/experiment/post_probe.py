"""The post-run probe: did the world teach the model anything?

The prior probe asks a cold model what it knows. This asks the SAME three
questions after a run, with that agent's own accumulated evidence in front of
it. The pair is what separates four things a single number cannot:

    RECALL       the cold model already knew it            (prior high)
    APPLICATION  it used a latent capability in-world      (prior high, in-run high)
    LEARNING     the world changed what it knows           (prior low, post high)
    CUMULATIVE   knowledge banked and reused by others     (verified claims)

Two rules keep the comparison fair, and both are the point rather than
housekeeping:

  * The QUESTIONS are byte-identical to the prior probe. A post-run probe that
    asked easier questions would manufacture learning.
  * The evidence handed to the model is only what THAT AGENT actually observed.
    Feeding it the run's pooled trials would measure our summary, not its
    experience.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..world.domains.base import Domain, TileContext
from .prior_probe import (
    PriorBaseline, _extract_json, _score_method, build_prompts,
    held_out_recipes, strip_reasoning,
)


@dataclass
class PostBaseline:
    """Same shape as PriorBaseline, so the two can be diffed field by field."""

    agent_id: int
    model: str
    domain: str
    trials_seen: int
    named_recipe: dict[str, Any] | None
    named_recipe_mu: float | None
    optimum_mu: float
    recall_ratio: float | None
    prediction_mae: float | None
    prediction_baseline_mae: float | None
    method_markers: dict[str, bool] = field(default_factory=dict)
    method_score: float = 0.0
    raw: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evidence_block(rows: Sequence[dict], domain: Domain,
                   *, max_cells: int = 30) -> str:
    """What this agent actually saw, aggregated the way the engine does it.

    Aggregated rather than listed: sixty noisy harvests are not something a 7B
    model can average in its head, and asking it to would measure arithmetic
    rather than knowledge. Only this agent's own rows are ever passed in.
    """
    if not rows:
        return "You have grown nothing yet."
    axes = [n for n in sorted(domain.param_space) if n != domain.schedule_axis]
    cells: dict[tuple, list[float]] = {}
    for r in rows:
        key = tuple(r.get(a) for a in axes)
        cells.setdefault(key, []).append(float(r["yield_kg"]))

    ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))[:max_cells]
    lines = ["What you grew, and what it yielded on average:"]
    for key, ys in sorted(ranked, key=lambda kv: str(kv[0])):
        desc = ", ".join(f"{a}={v}" for a, v in zip(axes, key))
        lines.append(f"  {desc}: {len(ys)} plantings, average "
                     f"{sum(ys) / len(ys):.2f}, best {max(ys):.2f}")
    return "\n".join(lines)


def run_post_probe(domain: Domain, ask, *, model: str, agent_id: int,
                   rows: Sequence[dict], soil: int = 2,
                   reference_skill: float = 1.0) -> PostBaseline:
    """``ask`` maps a prompt to a completion, as in the prior probe."""
    from ..metrics.definitions import controllable_optimum

    # Byte-identical questions, with the agent's own evidence prepended.
    prompts = build_prompts(domain, "rules_only")
    evidence = evidence_block(rows, domain)
    answers = {k: (ask(f"{evidence}\n\n{p}") or "") for k, p in prompts.items()}

    tile = TileContext(0, soil)
    best_mu = controllable_optimum(domain, soil)

    named = _extract_json(answers["name_best"])
    choice = (named or {}).get("choice") if named else None
    named_mu = None
    if isinstance(choice, dict):
        days = domain.param_space[domain.schedule_axis].values
        recipes = [{**choice, domain.schedule_axis: d} for d in days]
        if all(domain.validate_recipe(r) is None for r in recipes):
            named_mu = sum(domain.true_mu(r, tile, reference_skill)
                           for r in recipes) / len(recipes)

    held = held_out_recipes(domain)
    truths = [domain.true_mu(r, tile, reference_skill) for r in held]
    mean_truth = sum(truths) / len(truths)
    baseline_mae = sum(abs(t - mean_truth) for t in truths) / len(truths)

    est_obj = _extract_json(answers["predict"])
    mae = None
    if est_obj and isinstance(est_obj.get("estimates"), list):
        est = est_obj["estimates"][:len(truths)]
        if len(est) == len(truths):
            try:
                mae = sum(abs(float(e) - t) for e, t in zip(est, truths)) / len(truths)
            except (TypeError, ValueError):
                mae = None

    markers, score = _score_method(answers["method"])

    return PostBaseline(
        agent_id=agent_id, model=model, domain=domain.name,
        trials_seen=len(rows),
        named_recipe=choice if isinstance(choice, dict) else None,
        named_recipe_mu=round(named_mu, 4) if named_mu is not None else None,
        optimum_mu=round(best_mu, 4),
        recall_ratio=(round(named_mu / best_mu, 4)
                      if named_mu is not None and best_mu else None),
        prediction_mae=round(mae, 4) if mae is not None else None,
        prediction_baseline_mae=round(baseline_mae, 4),
        method_markers=markers, method_score=score,
        raw={k: strip_reasoning(v)[:1500] for k, v in answers.items()},
    )


def classify_learning(prior: PriorBaseline, post: PostBaseline,
                      *, in_run: dict[str, Any] | None = None) -> dict[str, Any]:
    """Separate recall from application from learning from cumulative knowledge.

    Every verdict below is hedged the way the evidence requires. The probe
    measures what a model SAYS, so a flat post-run score means it did not
    articulate a change -- not that nothing changed.
    """
    in_run = in_run or {}
    pr, po = prior.recall_ratio, post.recall_ratio
    moved = (None if pr is None or po is None else round(po - pr, 4))

    pred_prior = prior.prediction_mae
    pred_post = post.prediction_mae
    pred_moved = (None if pred_prior is None or pred_post is None
                  else round(pred_prior - pred_post, 4))

    if pr is not None and pr > 0.95:
        knowledge = ("RECALL: the cold model already named the optimum, so "
                     "naming it again proves nothing")
    elif moved is not None and moved > 0.15:
        knowledge = ("LEARNING: the model named a materially better recipe "
                     "after farming than before")
    elif moved is not None and moved < -0.15:
        knowledge = ("REGRESSION: it named a worse recipe after farming than "
                     "before, which is a real and reportable outcome")
    elif moved is not None:
        knowledge = ("NO ARTICULATED CHANGE: what it names is about the same. "
                     "The probe measures what it SAYS, so this does not "
                     "establish that nothing was learned")
    else:
        knowledge = "UNMEASURABLE: one of the two probes produced no usable answer"

    if prior.controlled_comparison_is_latent:
        method = ("APPLICATION at best: the cold model already volunteered "
                  "controlled comparison, so in-run use is not its emergence")
    elif post.method_score > prior.method_score:
        method = ("the model described more method after farming than before; "
                  "suggestive, and a lower bound either way")
    else:
        method = "no increase in articulated method"

    banked = in_run.get("banked")
    cumulative = (
        "CUMULATIVE: knowledge survived verification and entered the public "
        "record" if banked else
        "NOT CUMULATIVE: nothing survived verification, so whatever the agents "
        "learned stayed private to them")

    return {
        "prior_recall": pr, "post_recall": po, "recall_delta": moved,
        "prior_prediction_mae": pred_prior, "post_prediction_mae": pred_post,
        "prediction_improvement": pred_moved,
        "prior_method_score": prior.method_score,
        "post_method_score": post.method_score,
        "trials_seen": post.trials_seen,
        "banked_claims": banked,
        "knowledge_verdict": knowledge,
        "method_verdict": method,
        "cumulative_verdict": cumulative,
        "caveat": ("Both probes measure what the model SAYS when asked. That "
                   "is a lower bound on what it knows, and therefore an upper "
                   "bound on the credit that may be assigned to emergence."),
    }
