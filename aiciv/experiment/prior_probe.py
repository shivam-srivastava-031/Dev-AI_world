"""The prior probe: making LATENT knowledge measurable.

An LLM agent cannot be a cognitive blank slate. A 7B instruction-tuned model
already knows what a plant is, what "more" means, and what a fair test is. We
did not put that there and cannot take it out.

So instead of pretending, we measure it. The probe runs the model COLD -- same
scaffolding, zero experience, no memory, no trials -- and asks three things:

  1. Name the best parameter combination you can.       (scored vs ground truth)
  2. Predict yields for held-out combinations.          (MAE and calibration)
  3. Unprompted: how would you find out?                (method vocabulary)

The result is a prior baseline, stored per (model, domain, prompt version) and
referenced from every run manifest. **In-run discoveries are credited only
against it.** If the cold model already names the optimum, the discovery is
RECALL and the report must say so.

The probe measures what a model will SAY when asked directly, which is a lower
bound on latent knowledge and therefore an UPPER bound on how much credit
emergence can be assigned. It is reported as a bound, never a point estimate.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..hashing import blake2b_hex
from ..world.domains.base import Domain, TileContext

#: Method vocabulary. Spontaneous use in task 3 means controlled comparison is
#: LATENT for this model -- it did not have to be discovered in-world.
#:
#: Patterns, not fixed substrings, and deliberately GENEROUS. A missed marker
#: understates latent knowledge, which inflates the credit assigned to
#: emergence -- the one direction this project must not err in. Over-detecting
#: costs us a conservative report; under-detecting costs us a false claim.
#: An earlier substring list missed "one thing at a time" because it only knew
#: "one at a time", and scored an obviously methodical answer as having no
#: method at all.
METHOD_PATTERNS: dict[str, tuple[str, ...]] = {
    "control": (
        r"one (thing|factor|variable|change|at a time)",
        r"hold(ing)? (the )?(rest|others?|everything|them)? ?constant",
        r"keep(ing)? (the )?(rest|others?|everything)( else)? (the same|fixed|constant)",
        r"chang(e|ing) (only )?one",
        r"vary(ing)? (only )?one",
        r"everything else (the same|fixed|unchanged)",
        r"\bcontrolled?\b",
    ),
    "replication": (
        r"repeat", r"replicat", r"several times", r"multiple (trials|times)",
        r"more than once", r"average (over|out)", r"a few times", r"many times",
    ),
    "comparison": (
        r"compar", r"versus", r"\bvs\b", r"against (a|the|another)", r"baseline",
        r"side by side",
    ),
    "confounding": (
        r"confound", r"other factors", r"soil", r"weather", r"noise",
        r"random variation", r"by chance", r"luck",
    ),
    "sample_size": (
        r"sample size", r"enough (trials|data|results)", r"how many",
        r"large enough", r"n *= *\d+",
    ),
}

#: Kept for callers that want the flat vocabulary rather than the patterns.
METHOD_MARKERS = {k: v for k, v in METHOD_PATTERNS.items()}


@dataclass
class PriorBaseline:
    probe_id: str
    model: str
    domain: str
    domain_version: str
    scaffold: str
    prompt_version: str
    named_recipe: dict[str, Any] | None
    named_recipe_mu: float | None
    optimum_mu: float
    recall_ratio: float | None          # named / optimum, at reference skill
    prediction_mae: float | None
    prediction_baseline_mae: float | None
    method_markers: dict[str, bool] = field(default_factory=dict)
    method_score: float = 0.0
    raw: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def controlled_comparison_is_latent(self) -> bool:
        """Did the cold model volunteer the method, unprompted?

        If it did, `protocol_adoption_rate` measures APPLICATION of a latent
        capability and must be labelled that way. It never measures the
        emergence of the method itself.
        """
        return bool(self.method_markers.get("control")
                    and self.method_markers.get("replication"))


def _score_method(text: str) -> tuple[dict[str, bool], float]:
    # Scored on the ANSWER, never on the reasoning trace: a model that muses
    # "should I control for soil?" and then proposes nothing has not proposed
    # a method.
    low = strip_reasoning(text or "").lower()
    found = {k: any(re.search(p, low) for p in patterns)
             for k, patterns in METHOD_PATTERNS.items()}
    return found, round(sum(found.values()) / len(found), 4)


#: Reasoning models wrap their working in these. The trace is not the answer,
#: and scoring it would credit a model for method vocabulary it used while
#: thinking rather than method it proposed.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks, including an unterminated one.

    An unterminated block means the token budget ran out mid-thought, so there
    is no answer at all -- returning the trace would be worse than returning
    nothing, because the scorer would treat deliberation as a reply.
    """
    if not text:
        return ""
    return OPEN_THINK_RE.sub("", THINK_RE.sub("", text)).strip()


def _extract_json(text: str) -> dict | None:
    text = strip_reasoning(text)
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def held_out_recipes(domain: Domain, n: int = 20) -> list[dict]:
    """A deterministic spread across the parameter space."""
    recipes = domain.enumerate_recipes()
    step = max(1, len(recipes) // n)
    return [recipes[i * step] for i in range(min(n, len(recipes)))]


def build_prompts(domain: Domain, scaffold: str) -> dict[str, str]:
    from ..agents.policies.llm.prompts import param_help

    space = {k: v.values for k, v in domain.param_space.items()}
    help_text = param_help(space, domain.schedule_axis)
    held = held_out_recipes(domain)

    return {
        "name_best": (
            f"You are about to farm in a world where each planting is described "
            f"by: {help_text}.\n"
            f"Without having grown anything yet, name the single combination you "
            f"would expect to give the largest harvest.\n"
            f'Reply as JSON: {{"choice": {{...}}, "why": "..."}}'
        ),
        "predict": (
            f"Each planting is described by: {help_text}.\n"
            f"Estimate the harvest in tonnes per hectare for each of these, as "
            f"best you can:\n" +
            "\n".join(f"{i}: {json.dumps({k: r[k] for k in sorted(r)})}"
                      for i, r in enumerate(held)) +
            '\nReply as JSON: {"estimates": [<number>, ...]}'
        ),
        "method": (
            f"Each planting is described by: {help_text}.\n"
            f"You want to end up growing as much as possible. "
            f"How would you go about working out which choices are best? "
            f"Answer in a few sentences."
        ),
    }


def run_probe(domain: Domain, ask, *, model: str, scaffold: str = "rules_only",
              prompt_version: str = "p1", soil: int = 2,
              reference_skill: float = 1.0) -> PriorBaseline:
    """``ask`` maps a prompt to a completion. Injected so the probe can be
    exercised without a model server."""
    prompts = build_prompts(domain, scaffold)
    answers = {k: (ask(p) or "") for k, p in prompts.items()}

    from ..metrics.definitions import controllable_optimum

    tile = TileContext(0, soil)
    # Controllable optimum, and day-averaged scoring below: the model is
    # asked to name what it would CHOOSE, so grading it against an optimum
    # that includes the sowing day would penalise it for something it was
    # never asked about.
    best_mu = controllable_optimum(domain, soil)

    # 1. Did it name a good combination cold?
    named = _extract_json(answers["name_best"])
    choice = (named or {}).get("choice") if named else None
    named_mu = None
    if isinstance(choice, dict):
        days = domain.param_space[domain.schedule_axis].values
        recipes = [{**choice, domain.schedule_axis: d} for d in days]
        if all(domain.validate_recipe(r) is None for r in recipes):
            named_mu = sum(domain.true_mu(r, tile, reference_skill)
                           for r in recipes) / len(recipes)

    # 2. How well does it predict, against the flat "guess the mean" baseline?
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

    # 3. Did it volunteer the method?
    markers, score = _score_method(answers["method"])

    truncated = [k for k, v in answers.items()
                 if "<think>" in (v or "") and "</think>" not in (v or "")]

    probe_id = blake2b_hex(model, domain.name, domain.version, scaffold,
                           prompt_version)
    return PriorBaseline(
        probe_id=probe_id, model=model, domain=domain.name,
        domain_version=domain.version, scaffold=scaffold,
        prompt_version=prompt_version,
        named_recipe=choice if isinstance(choice, dict) else None,
        named_recipe_mu=round(named_mu, 4) if named_mu is not None else None,
        optimum_mu=round(best_mu, 4),
        recall_ratio=(round(named_mu / best_mu, 4)
                      if named_mu is not None and best_mu else None),
        prediction_mae=round(mae, 4) if mae is not None else None,
        prediction_baseline_mae=round(baseline_mae, 4),
        method_markers=markers, method_score=score,
        raw={k: v[:1500] for k, v in answers.items()},
        note=("The probe measures what the model SAYS when asked directly. "
              "That is a lower bound on latent knowledge, and therefore an "
              "upper bound on the credit emergence may be assigned."
              + (f" TRUNCATED (ran out of tokens mid-thought): {truncated}. "
                 f"These tasks have no answer and their scores are not a "
                 f"finding about the model." if truncated else "")),
    )


def save(baseline: PriorBaseline, directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"prior_{baseline.probe_id[:12]}.json"
    path.write_text(json.dumps(baseline.to_dict(), indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def load(path: pathlib.Path) -> PriorBaseline:
    return PriorBaseline(**json.loads(path.read_text(encoding="utf-8")))


def credit_against_baseline(baseline: PriorBaseline,
                            in_run: dict[str, Any]) -> dict[str, Any]:
    """Discount what the model already knew before the run began.

    A discovery count without its baseline is not a result.
    """
    recall = baseline.recall_ratio
    return {
        "prior_probe_id": baseline.probe_id,
        "cold_recall_ratio": recall,
        "cold_named_optimum": bool(recall is not None and recall > 0.95),
        "controlled_comparison_is_latent":
            baseline.controlled_comparison_is_latent,
        "method_score_cold": baseline.method_score,
        "banked_in_run": in_run.get("banked"),
        "interpretation": (
            "RECALL: the model named the optimum before farming anything; "
            "in-run discovery of it is not evidence of discovery"
            if recall is not None and recall > 0.95 else
            "in-run discovery exceeded what the cold model could name"
        ),
        "method_interpretation": (
            "APPLICATION of a latent capability, not its emergence"
            if baseline.controlled_comparison_is_latent else
            "the cold model did not volunteer controlled comparison"
        ),
    }
