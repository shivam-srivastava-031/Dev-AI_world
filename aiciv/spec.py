"""Shared value objects with no dependencies on world, agents or knowledge."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .hashing import blake2b_hex


@dataclass(frozen=True)
class VerificationSpec:
    """Per-domain statistical acceptance criteria.

    ``min_trials_per_group`` is NEVER hand-chosen. It is the output of the power
    analysis in tests/test_calibration.py, which requires power > 0.8 for the
    domain's target effect and < 0.3 for a null. A domain that has not been
    through that calibration has no business carrying a VerificationSpec.
    """

    min_trials_per_group: int
    min_agents: int
    min_tiles: int
    alpha: float                  # SUPPORTED stage
    confirmation_alpha: float     # CONFIRMED stage (independent replication)
    holdout_alpha: float          # GENERALIZED stage
    min_effect_size: float        # Hedges' g floor
    correction: str               # "bh" | "bonferroni" | "none"
    max_trials: int               # budget before a claim is REFUTED
    balance_mode: str             # "adjust" | "gate" | "off"
    calibrated_by: str = "uncalibrated"

    def __post_init__(self) -> None:
        for name in ("alpha", "confirmation_alpha", "holdout_alpha"):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0,1), got {v}")
        if self.balance_mode not in ("adjust", "gate", "off"):
            raise ValueError(f"bad balance_mode: {self.balance_mode}")
        if self.correction not in ("bh", "bonferroni", "none"):
            raise ValueError(f"bad correction: {self.correction}")

    @property
    def hash(self) -> str:
        return blake2b_hex(asdict(self))
