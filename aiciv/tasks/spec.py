"""Tasks: an objective an observer sets, and how far along it is.

A task is a metric, a comparator and a target -- nothing more. It is scored
against whatever the run has written so far, so a task on a live run reads from
the last checkpoint like every other panel.

Two rules this file exists to keep:

  * A task never invents a number. If the quantity it names has not been
    measured yet, the value is None and the progress is None -- not zero. "No
    trials yet" and "0% of the way to 200 trials" look the same only by
    accident, and they stop looking the same the moment a run is degraded.

  * A task only shows a percentage when a percentage means something. A target
    that counts up ("run 200 trials") has a real fraction. A target that is a
    ceiling or a rate ("keep invalid actions under 15%") does not: being at 30%
    is not "half done", it is not done. Those report met/not-met with the
    current value, and the dashboard draws them differently.

Scoring is pure and read-only. Nothing here writes to a run.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..persistence.store import RunStore

#: Comparators an observer may choose. Deliberately two: a floor and a ceiling.
#: Equality on a live counter is a task that passes for one checkpoint and then
#: silently un-passes, which is not a useful thing to be able to express.
COMPARATORS = (">=", "<=")

#: Where a task came from, and the separation is the whole point of it.
#:
#: ``observer`` -- set in the dashboard, scored from the log, and NEVER shown to
#: an agent. The run is unsteered; the task is a question asked of it after the
#: fact, and adding one cannot change what the run does.
#:
#: ``briefed`` -- the task came from a brief fixed before the run started. If it
#: carries directive text, that text was in the agents' prompt and the run is
#: STEERED: the agents were told what to aim at, which is a different experiment
#: from one where they were not. A briefed task with no directive is scored but
#: was never spoken, and does not steer anything. Steering is marked in the
#: manifest and in the dashboard, because comparing a steered run against an
#: unsteered one without noticing is the most misleading thing this could enable.
SOURCES = ("observer", "briefed")


@dataclass(frozen=True)
class Metric:
    """One measurable quantity, with the honesty rules attached to it."""

    key: str
    label: str
    #: True when the quantity only ever counts up over a run. Only a cumulative
    #: metric with a floor target has a meaningful "percent complete".
    cumulative: bool
    unit: str                      # "count" | "fraction" | "kg"
    describes: str

    def kind(self, comparator: str) -> str:
        return ("accumulate" if self.cumulative and comparator == ">="
                else "threshold")


#: The catalogue, in the order the dashboard offers it. Everything here is
#: readable from a partial checkpoint; nothing that exists only in the
#: end-of-run civilization report is offered, because a task nobody can score
#: until the run ends cannot be monitored while it runs.
METRICS: tuple[Metric, ...] = (
    Metric("ticks", "ticks recorded", True, "count",
           "days of world time actually written to the log"),
    Metric("trials", "completed trials", True, "count",
           "plant-to-harvest cycles that produced a measured yield"),
    Metric("distinct_recipes", "distinct recipes", True, "count",
           "unique parameter combinations planted at least once"),
    Metric("distinct_tiles", "distinct tiles used", True, "count",
           "tiles planted at least once"),
    Metric("plants", "successful plantings", True, "count",
           "PLANT actions the world accepted"),
    Metric("harvests", "harvests", True, "count",
           "HARVEST actions the world accepted"),
    Metric("channels_built", "channels built", True, "count",
           "BUILD_CHANNEL actions the world accepted"),
    Metric("inspections", "tile inspections", True, "count",
           "INSPECT_TILE actions the world accepted"),
    Metric("claims_proposed", "claims proposed", True, "count",
           "claims put on the record in any state"),
    Metric("claims_registered", "claims registered", True, "count",
           "claims registered before their evidence was gathered"),
    Metric("claims_banked", "claims banked", True, "count",
           "claims that reached confirmed or generalized"),
    Metric("claims_confirmed", "claims confirmed", True, "count",
           "claims confirmed by a replication"),
    Metric("claims_generalized", "claims generalized", True, "count",
           "claims that held outside the conditions they were found in"),
    Metric("claims_refuted", "claims refuted", True, "count",
           "claims the evidence went against; verification working, not a fault"),
    Metric("teach_actions", "teaching acts", True, "count",
           "TEACH actions the world accepted"),
    Metric("messages_sent", "messages sent", True, "count",
           "SAY and ASK actions the world accepted"),
    Metric("goals_set", "goals set", True, "count",
           "GOAL_SET actions; goals are self-authored and never scored"),
    Metric("distinct_goals", "distinct goals", True, "count",
           "unique goal texts, case-folded"),
    Metric("hunger_events", "hunger events", True, "count",
           "days an agent went short of food"),
    Metric("productive_action_rate", "productive action rate", False, "fraction",
           "share of all actions that were PLANT or HARVEST"),
    Metric("invalid_action_rate", "invalid action rate", False, "fraction",
           "share of proposals the world rejected"),
    Metric("noop_rate", "NOOP rate", False, "fraction",
           "share of actions that were NOOP, including model silence"),
    Metric("inspect_rate", "inspect rate", False, "fraction",
           "share of actions spent looking rather than acting"),
    Metric("most_repeated_recipe_share", "most repeated recipe share", False,
           "fraction",
           "share of trials spent on the single most repeated recipe"),
    Metric("best_yield", "best yield", False, "kg",
           "the largest single harvest so far"),
    Metric("mean_yield", "mean yield", False, "kg",
           "the mean yield across every completed trial"),
)

BY_KEY: dict[str, Metric] = {m.key: m for m in METRICS}


def catalogue() -> list[dict[str, Any]]:
    """The metric list, for the dashboard's task form."""
    return [{"key": m.key, "label": m.label, "unit": m.unit,
             "cumulative": m.cumulative, "describes": m.describes}
            for m in METRICS]


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    metric: str
    comparator: str
    target: float
    #: The sentence the agents were given, when they were given one. Empty for
    #: an observer task, and it is empty because they were never told.
    directive: str = ""
    source: str = "observer"
    created_at: str = ""
    brief_id: str = ""

    def __post_init__(self) -> None:
        if self.metric not in BY_KEY:
            raise ValueError(f"unknown metric {self.metric!r}")
        if self.comparator not in COMPARATORS:
            raise ValueError(f"comparator must be one of {COMPARATORS}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if self.target < 0:
            raise ValueError("target must not be negative")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "title": self.title,
                "metric": self.metric, "comparator": self.comparator,
                "target": self.target, "directive": self.directive,
                "source": self.source, "created_at": self.created_at,
                "brief_id": self.brief_id}


def _pct(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def snapshot(store: RunStore, run_id: str) -> dict[str, float | None]:
    """Every metric in the catalogue, computed once from the checkpoint.

    Computed together rather than per task: a board of ten tasks would
    otherwise walk the action log ten times, and the dashboard polls.
    """
    actions = store.actions(run_id)
    trials = store.trials(run_id, limit=200000)
    claims = store.claims(run_id)

    verbs: Counter = Counter()
    valid_verbs: Counter = Counter()
    invalid = 0
    for r in actions:
        verb = json.loads(r["proposal_json"])["verb"]
        verbs[verb] += 1
        if r["valid"]:
            valid_verbs[verb] += 1
        else:
            invalid += 1
    n_actions = len(actions)

    recipes: Counter = Counter()
    tiles = set()
    yields: list[float] = []
    for t in trials:
        recipes[tuple(sorted(json.loads(t["params_json"]).items()))] += 1
        tiles.add(t["tile_id"])
        if t["yield_kg"] is not None:
            yields.append(float(t["yield_kg"]))

    states = Counter(c["state"] for c in claims)

    goal_texts = [json.loads(e["payload_json"]).get("text", "")
                  for e in store.events(run_id, kind="goal_set", limit=200000)]
    distinct_goals = {t.strip().lower() for t in goal_texts if t.strip()}

    return {
        "ticks": len(store.state_hashes(run_id)),
        "trials": len(trials),
        "distinct_recipes": len(recipes),
        "distinct_tiles": len(tiles),
        "plants": valid_verbs["PLANT"],
        "harvests": valid_verbs["HARVEST"],
        "channels_built": valid_verbs["BUILD_CHANNEL"],
        "inspections": valid_verbs["INSPECT_TILE"],
        "claims_proposed": len(claims),
        "claims_registered": sum(1 for c in claims
                                 if c["registered_tick"] is not None),
        "claims_banked": states["confirmed"] + states["generalized"],
        "claims_confirmed": states["confirmed"],
        "claims_generalized": states["generalized"],
        "claims_refuted": states["refuted"],
        "teach_actions": valid_verbs["TEACH"],
        "messages_sent": valid_verbs["SAY"] + valid_verbs["ASK"],
        "goals_set": len(goal_texts),
        "distinct_goals": len(distinct_goals),
        "hunger_events": len(store.events(run_id, kind="hunger", limit=200000)),
        # Rates are None, never 0.0, when there is nothing to take a rate of.
        "productive_action_rate": _pct(verbs["PLANT"] + verbs["HARVEST"],
                                       n_actions),
        "invalid_action_rate": _pct(invalid, n_actions),
        "noop_rate": _pct(verbs["NOOP"], n_actions),
        "inspect_rate": _pct(verbs["INSPECT_TILE"], n_actions),
        "most_repeated_recipe_share": (
            _pct(recipes.most_common(1)[0][1], len(trials)) if recipes else None),
        "best_yield": round(max(yields), 3) if yields else None,
        "mean_yield": round(sum(yields) / len(yields), 3) if yields else None,
    }


def _satisfied(value: float, comparator: str, target: float) -> bool:
    return value >= target if comparator == ">=" else value <= target


def _fmt(value: float, unit: str) -> str:
    if unit == "fraction":
        return f"{100 * value:.1f}%"
    if unit == "kg":
        return f"{value:.2f} kg"
    return f"{value:g}"


def _detail(metric: Metric, task: Task, value: float, done: bool) -> str:
    want = "at least" if task.comparator == ">=" else "at most"
    now = _fmt(value, metric.unit)
    goal = _fmt(task.target, metric.unit)
    if done:
        return f"met: {metric.label} is {now}, {want} {goal} was asked for"
    if task.comparator == ">=":
        return (f"{now} of {goal} {metric.label}; "
                f"{_fmt(task.target - value, metric.unit)} still to go")
    return f"not met: {metric.label} is {now}, {want} {goal} was asked for"


def score(task: Task, values: dict[str, float | None]) -> dict[str, Any]:
    """One task against one snapshot.

    ``progress`` is a fraction only for a counting metric with a floor target.
    Everywhere else it is None and ``kind`` says why, so the dashboard cannot
    accidentally draw a filling bar for something that does not fill up.
    """
    metric = BY_KEY[task.metric]
    kind = metric.kind(task.comparator)
    value = values.get(task.metric)

    if value is None:
        return {**task.to_dict(), "kind": kind, "unit": metric.unit,
                "metric_label": metric.label, "value": None,
                "progress": None, "done": False, "measurable": False,
                "detail": (f"not measurable yet: {metric.label} is the "
                           f"{metric.describes}, and this run has produced "
                           f"none so far")}

    done = _satisfied(value, task.comparator, task.target)
    progress: float | None = None
    if kind == "accumulate":
        # A zero target is already met by definition; dividing by it would be
        # the one place this file could produce a nonsense number.
        progress = 1.0 if task.target <= 0 else min(1.0, value / task.target)

    return {**task.to_dict(), "kind": kind, "unit": metric.unit,
            "metric_label": metric.label, "value": value,
            "progress": progress, "done": done, "measurable": True,
            "detail": _detail(metric, task, value, done)}


def score_board(tasks: list[Task], store: RunStore, run_id: str
                ) -> dict[str, Any]:
    """A whole board, plus the one honest summary line over it.

    The summary counts tasks met, and separately reports the mean progress of
    the counting tasks only. Averaging a met/not-met in as 0 or 1 would invent
    a number, so it is not averaged in.
    """
    values = snapshot(store, run_id)
    scored = [score(t, values) for t in tasks]
    fractions = [s["progress"] for s in scored if s["progress"] is not None]
    briefed = [s for s in scored if s["source"] == "briefed"]
    # Steering is whether a sentence actually reached the prompt, not whether a
    # brief was attached. A brief may carry tasks with no directive text -- the
    # observer wanting the target scored without saying it out loud -- and that
    # run is not steered. Conflating the two would put a STEERED badge on a run
    # whose agents were told nothing.
    said = [s for s in briefed if s["directive"].strip()]
    return {
        "run_id": run_id,
        "tasks": scored,
        "count": len(scored),
        "done": sum(1 for s in scored if s["done"]),
        "unmeasurable": sum(1 for s in scored if not s["measurable"]),
        # None, not 0, when no task on the board has a meaningful fraction.
        "mean_progress": (round(sum(fractions) / len(fractions), 4)
                          if fractions else None),
        # The steering flag. A run that was told something is not an unsteered
        # run and must not be pooled with one.
        "steered": bool(said),
        "briefed_count": len(briefed),
        "spoken_count": len(said),
    }
