"""The Policy interface. ScriptedGreedy, RandomPolicy and OllamaLLMPolicy are
interchangeable behind it, which is what makes the control arms meaningful.

PolicyContext deliberately does NOT expose WorldState. An Observation is the
only read channel; test_policy_context_has_no_world_ref enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..world.actions import ActionProposal


@dataclass
class PolicyContext:
    agent_id: int
    tick: int
    rng: np.random.Generator
    config: Any
    param_space: dict[str, tuple]      # PUBLIC: what may be varied, not what it does
    notes: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    name: str

    def reset(self, ctx: PolicyContext) -> None: ...

    def decide(self, obs, ctx: PolicyContext) -> ActionProposal: ...

    def observe_result(self, result: dict, ctx: PolicyContext) -> None: ...


POLICY_REGISTRY: dict[str, type] = {}


def register(cls):
    POLICY_REGISTRY[cls.name] = cls
    return cls


def build_policy(name: str):
    if name not in POLICY_REGISTRY:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name]()
