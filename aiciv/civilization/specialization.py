"""Emergent specialization -- detected, never modeled.

There is no `role` field anywhere in this project. If we gave an agent a role,
we would have built the institution rather than observed one. So specialization
is measured after the fact, from behaviour.

The trap this module exists to avoid: an agent that happens to live beside
fertile soil farms more, and naive clustering calls that a social role. It is
not a role, it is a location. So behaviour is regressed on the environment each
agent actually faced, and the specialization index is computed from the
RESIDUALS -- the part of what an agent did that its circumstances do not
explain.

Two null tests keep it honest, and the second is the one that matters:
identical policies on homogeneous terrain must show no roles, and identical
policies on HETEROGENEOUS terrain must also show no roles.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

#: Behavioural dimensions. Proportions of an agent's own actions, so an agent
#: that simply acted more often does not look like a specialist.
BEHAVIOURS = ("PLANT", "HARVEST", "MOVE", "REST", "EAT", "TEACH", "SAY",
              "PROPOSE_CLAIM", "REGISTER_CLAIM", "CHALLENGE_CLAIM",
              "INSPECT_TILE", "PREDICT")

#: What an agent's circumstances explain. Controlling for these is the whole
#: point; without them the index measures terrain, not society.
ENVIRONMENT = ("mean_soil_in_reach", "reachable_tiles", "opportunities", "skill")


@dataclass
class SpecializationReport:
    index: float                      # 0 = fully explained by environment
    per_agent_residual: dict[int, float] = field(default_factory=dict)
    explained_variance: float = 0.0
    n_agents: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        """A deliberately conservative threshold. Calling a role that is really
        a patch of good soil is the expensive mistake here."""
        return self.index >= 0.35

    def to_dict(self) -> dict:
        return {
            "specialization_index": round(self.index, 4),
            "roles_detected": self.detected,
            "explained_by_environment": round(self.explained_variance, 4),
            "n_agents": self.n_agents,
            "per_agent_residual": {a: round(v, 4)
                                   for a, v in sorted(self.per_agent_residual.items())},
            **self.detail,
        }


def behaviour_matrix(actions_by_agent: dict[int, list[str]]) -> tuple[list[int], np.ndarray]:
    agents = sorted(actions_by_agent)
    rows = []
    for a in agents:
        counts = Counter(actions_by_agent[a])
        total = max(sum(counts.values()), 1)
        rows.append([counts.get(b, 0) / total for b in BEHAVIOURS])
    return agents, np.asarray(rows, dtype=float)


def environment_matrix(env_by_agent: dict[int, dict], agents: list[int]) -> np.ndarray:
    return np.asarray(
        [[float(env_by_agent.get(a, {}).get(k, 0.0)) for k in ENVIRONMENT]
         for a in agents], dtype=float)


def analyse(
    actions_by_agent: dict[int, list[str]],
    env_by_agent: dict[int, dict],
) -> SpecializationReport:
    """Specialization index = residual dispersion / raw dispersion.

    1.0 means the environment explains none of the behavioural spread;
    0.0 means it explains all of it.
    """
    agents, B = behaviour_matrix(actions_by_agent)
    n = len(agents)
    if n < 3:
        return SpecializationReport(0.0, {}, 0.0, n,
                                    {"note": "too few agents to separate role from place"})

    E = environment_matrix(env_by_agent, agents)
    # Keep only environment columns that actually vary; a constant column is
    # collinear with the intercept and makes the design rank-deficient.
    keep = [i for i in range(E.shape[1]) if np.ptp(E[:, i]) > 1e-12]
    X = np.column_stack([np.ones(n)] + [E[:, i] for i in keep])

    raw_var = float(np.sum(np.var(B, axis=0)))
    if raw_var <= 1e-12:
        return SpecializationReport(0.0, {a: 0.0 for a in agents}, 1.0, n,
                                    {"note": "all agents behaved identically"})

    # With few agents the design can saturate; if so the residual is zero by
    # construction and would falsely read as "environment explains everything".
    if X.shape[1] >= n:
        X = X[:, :max(1, n - 1)]

    beta, *_ = np.linalg.lstsq(X, B, rcond=None)
    resid = B - X @ beta
    resid_var = float(np.sum(np.var(resid, axis=0)))

    index = max(0.0, min(1.0, resid_var / raw_var))
    per_agent = {a: float(np.linalg.norm(resid[i])) for i, a in enumerate(agents)}
    return SpecializationReport(
        index=index,
        per_agent_residual=per_agent,
        explained_variance=1.0 - index,
        n_agents=n,
        detail={"environment_terms": [ENVIRONMENT[i] for i in keep],
                "raw_dispersion": round(raw_var, 5)},
    )


def environment_for(state, agent_id: int, *, radius: int = 3) -> dict:
    """The circumstances one agent actually faced, from world state.

    METRICS-side: this reads the grid directly, which is allowed here because
    aiciv.metrics and aiciv.civilization grade behaviour and are not policies.
    """
    a = state.agents[agent_id]
    soils, reachable = [], 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = a.x + dx, a.y + dy
            if not state.grid.in_bounds(x, y):
                continue
            t = state.grid.at(x, y)
            if t.arable:
                reachable += 1
                soils.append(t.soil_band)
    return {
        "mean_soil_in_reach": (sum(soils) / len(soils)) if soils else 0.0,
        "reachable_tiles": reachable,
        "opportunities": reachable,
        "skill": a.skill_farming,
    }
