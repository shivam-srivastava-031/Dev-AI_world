"""Experiment runner: arms, paired seeds, and honest comparison.

The design is a factorial over policy x teaching, run on the SAME seeds in
every arm. Pairing matters more than it might look: worlds differ enormously
between seeds, and an unpaired comparison spends most of its power on that
variation rather than on the thing being tested.

  B - A  isolates whether teaching helps
  C - A  isolates whether the model adds anything over a scripted searcher

Comparisons are reported as an effect with a bootstrap interval. A bare
p-value would say whether a difference exists without saying whether it
matters, and at these sample sizes that is the less useful half.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..config import RunConfig
from ..metrics.adoption import adoption_report
from ..metrics.definitions import civilization_report
from ..world.domains.agronomy import AgronomyDomain
from ..world.domains.base import Domain
from ..world.domains.synthetic import SyntheticDomain

DOMAINS: dict[str, Callable[..., Domain]] = {
    "synthetic": SyntheticDomain,
    "agronomy": AgronomyDomain,
}


@dataclass
class Arm:
    name: str
    policy: str
    teaching: bool = True
    water_hint: bool = True
    domain: str = "synthetic"
    scaffold: str = "rules_only"
    model: str | None = None

    def config(self, seed: int, ticks: int, n_agents: int) -> RunConfig:
        return RunConfig(
            seed=seed, ticks=ticks, n_agents=n_agents, policy=self.policy,
            domain=self.domain, water_hint=self.water_hint,
            teaching_enabled=self.teaching, scaffold_level=self.scaffold,
            model=self.model or "assistant:latest", arm=self.name,
        )


#: The 2x2 the plan calls for, plus the floor and the ablations.
DEFAULT_ARMS = [
    Arm("random", "random"),
    Arm("scripted_greedy", "scripted_greedy"),
    Arm("scripted_factorial", "scripted_factorial"),
    Arm("social_no_teaching", "honest", teaching=False),
    Arm("social_teaching", "honest", teaching=True),
    Arm("copycat", "copycat"),
    Arm("liar", "liar"),
]


@dataclass
class RunResult:
    arm: str
    seed: int
    metrics: dict[str, Any]
    adoption: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)

    def scalar(self, path: str) -> float | None:
        """Pull a nested metric by dotted path, e.g. 'knowledge.banked'."""
        node: Any = self.metrics
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, (int, float)) else None


def run_one(arm: Arm, seed: int, *, ticks: int, n_agents: int) -> RunResult:
    from ..world.engine import Engine
    import aiciv.agents.policies  # noqa: F401  registers every policy

    cfg = arm.config(seed, ticks, n_agents)
    domain = DOMAINS[arm.domain](water_hint=arm.water_hint)
    engine = Engine(cfg, domain)
    engine.run()

    rows_by_agent: dict[int, list[dict]] = {}
    for row in engine.rows():
        rows_by_agent.setdefault(int(row["agent_id"]), []).append(row)
    claims_by_agent: dict[int, list] = {}
    for r in engine.kb:
        if r.registered_tick is not None:
            claims_by_agent.setdefault(int(r.claim.author), []).append(r.claim)

    return RunResult(
        arm=arm.name, seed=seed,
        metrics=civilization_report(engine, domain),
        adoption=adoption_report(rows_by_agent, claims_by_agent),
        manifest={
            "config_hash": cfg.hash,
            "verification_spec_hash": domain.verification.hash,
            "domain": domain.name, "domain_version": domain.version,
            "ticks": ticks, "agents": n_agents, "teaching": arm.teaching,
            "final_state_hash": engine.hash_sequence()[-1]
            if engine.hash_sequence() else None,
        },
    )


# --- comparison -----------------------------------------------------------

def paired_bootstrap(a: Sequence[float], b: Sequence[float], *,
                     reps: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Bootstrap the PAIRED difference a - b over shared seeds.

    Pairing is the point: the same world appears in both arms, so world-to-world
    variation cancels instead of swamping the effect.
    """
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None]
    if len(pairs) < 2:
        return {"n_pairs": len(pairs), "effect": None, "ci": None,
                "note": "too few paired seeds to compare"}

    diffs = np.array([x - y for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_pairs": len(pairs),
        "effect": round(float(diffs.mean()), 4),
        "ci": [round(float(lo), 4), round(float(hi), 4)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "per_seed": [round(float(d), 4) for d in diffs],
    }


PRIMARY_METRICS = (
    "knowledge.banked",
    "knowledge.generalized",
    "knowledge.independently_corroborated",
    "grading.false_positives",
    "collective.collective_regret",
    "competence.agents_reaching_competence",
    "social.blind_adoption",
    "resilience.resilience",
)


def compare(results: list[RunResult], arm_a: str, arm_b: str,
            metrics: Sequence[str] = PRIMARY_METRICS) -> dict[str, Any]:
    by_seed_a = {r.seed: r for r in results if r.arm == arm_a}
    by_seed_b = {r.seed: r for r in results if r.arm == arm_b}
    shared = sorted(set(by_seed_a) & set(by_seed_b))

    out: dict[str, Any] = {"arm_a": arm_a, "arm_b": arm_b, "seeds": shared,
                           "metrics": {}}
    for m in metrics:
        a = [by_seed_a[s].scalar(m) for s in shared]
        b = [by_seed_b[s].scalar(m) for s in shared]
        out["metrics"][m] = paired_bootstrap(a, b, seed=hash(m) % 10_000)
    return out


@dataclass
class Experiment:
    name: str
    arms: list[Arm] = field(default_factory=lambda: list(DEFAULT_ARMS))
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    ticks: int = 1200
    n_agents: int = 5
    out_dir: pathlib.Path = pathlib.Path("runs")

    def run(self, *, progress=None) -> list[RunResult]:
        results: list[RunResult] = []
        for arm in self.arms:
            for seed in self.seeds:
                if progress:
                    progress(arm.name, seed)
                results.append(run_one(arm, seed, ticks=self.ticks,
                                       n_agents=self.n_agents))
        return results

    def report(self, results: list[RunResult]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for arm in self.arms:
            rows = [r for r in results if r.arm == arm.name]
            if not rows:
                continue
            summary[arm.name] = {
                m: _describe([r.scalar(m) for r in rows]) for m in PRIMARY_METRICS
            }

        names = [a.name for a in self.arms]
        contrasts = {}
        # The two contrasts the design exists to produce.
        for a, b in (("social_teaching", "social_no_teaching"),
                     ("scripted_factorial", "scripted_greedy"),
                     ("scripted_factorial", "random")):
            if a in names and b in names:
                contrasts[f"{a}_vs_{b}"] = compare(results, a, b)

        return {
            "experiment": self.name,
            "seeds": self.seeds,
            "ticks": self.ticks,
            "arms": {a.name: asdict(a) for a in self.arms},
            "summary": summary,
            "contrasts": contrasts,
        }

    def save(self, report: dict[str, Any]) -> pathlib.Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"experiment_{self.name}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
        return path


def _describe(values: list[float | None]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(float(np.mean(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "min": round(float(min(vals)), 4),
        "max": round(float(max(vals)), 4),
    }
