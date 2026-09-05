"""Domain interface: the hidden mechanics of a world.

A Domain owns the function from a farming recipe to a yield. Agents never see
it -- ``aiciv.agents.*`` is forbidden from importing this package, and a test
enforces that. Only ``aiciv.metrics.*`` may import a domain, because grading a
discovery requires ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ...spec import VerificationSpec


@dataclass(frozen=True)
class ParamSpec:
    """A parameter an agent may vary. This part IS public."""

    name: str
    kind: str                      # "int" | "categorical"
    values: tuple[Any, ...]
    description: str = ""


@dataclass(frozen=True)
class TileContext:
    """Everything about a tile that feeds the yield function."""

    tile_id: int
    soil_band: int                 # AGENT_OBSERVABLE via INSPECT_TILE


@dataclass(frozen=True)
class Outcome:
    """What the world hands back after a harvest.

    ``observed_yield`` is what the agent sees. ``true_mu`` is METRICS_ONLY and
    must never reach an Observation, a memory record or a prompt.
    """

    observed_yield: float
    crop_health: str               # deterministic, non-noisy water side-channel
    true_mu: float = field(repr=False)


class Domain(ABC):
    name: str
    version: str
    verification: VerificationSpec

    @property
    @abstractmethod
    def param_space(self) -> Mapping[str, ParamSpec]:
        """Public: what an agent is allowed to vary. Not what any of it does."""

    @abstractmethod
    def true_mu(self, recipe: Mapping[str, Any], tile: TileContext, skill: float) -> float:
        """HIDDEN + METRICS_ONLY: the noiseless expected yield."""

    @abstractmethod
    def crop_health(self, recipe: Mapping[str, Any]) -> str:
        """PUBLIC: a deterministic, noiseless categorical hint."""

    def evaluate(
        self,
        recipe: Mapping[str, Any],
        tile: TileContext,
        skill: float,
        rng: np.random.Generator,
    ) -> Outcome:
        mu = self.true_mu(recipe, tile, skill)
        observed = max(0.0, mu + rng.normal(0.0, self.sigma))
        return Outcome(
            observed_yield=observed,
            crop_health=self.crop_health(recipe),
            true_mu=mu,
        )

    @property
    @abstractmethod
    def sigma(self) -> float:
        """Observation noise standard deviation."""

    # ---- METRICS_ONLY -----------------------------------------------------

    def enumerate_recipes(self) -> Sequence[dict[str, Any]]:
        """Every point in the agent-controllable parameter space."""
        import itertools

        names = list(self.param_space)
        grids = [self.param_space[n].values for n in names]
        return [dict(zip(names, combo)) for combo in itertools.product(*grids)]

    def true_optimum(self, tile: TileContext, skill: float) -> tuple[dict[str, Any], float]:
        """METRICS_ONLY: brute-force best recipe and its mu on a given tile."""
        best, best_mu = None, -np.inf
        for r in self.enumerate_recipes():
            mu = self.true_mu(r, tile, skill)
            if mu > best_mu:
                best, best_mu = r, mu
        assert best is not None
        return best, best_mu

    def validate_recipe(self, recipe: Mapping[str, Any]) -> str | None:
        """Return a rejection detail string, or None if the recipe is legal."""
        for name, spec in self.param_space.items():
            if name not in recipe:
                return f"missing parameter '{name}'"
            if recipe[name] not in spec.values:
                return (
                    f"parameter '{name}'={recipe[name]!r} outside "
                    f"allowed values {list(spec.values)}"
                )
        extra = set(recipe) - set(self.param_space)
        if extra:
            return f"unknown parameters: {sorted(extra)}"
        return None
