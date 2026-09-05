"""Run configuration. Frozen once a run starts; its hash goes in the manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .hashing import blake2b_hex

# --- calibrated constants (docs/statistics.md) ---------------------------
MAX_CONCURRENT_PLOTS = 4     # 1 plot => only ~5 claims affordable per run
MATURATION_TICKS = 6
FALLOW_TICKS = 10           # a harvested tile rests; forces rotation so
                            # ordinary farming samples every stratum
SEED_COST = 1
WATER_CARRY_MAX = 8
ENERGY_MAX = 10.0
ENERGY_PER_ACTION = 1.0
ENERGY_PER_REST = 4.0
FOOD_PER_TICK = 0.5
SURVIVAL_FLOOR = 0.5         # basic survival is GUARANTEED in v0.1
SKILL_PER_HARVEST = 0.02

# --- opt-in protocol economics -------------------------------------------
REPUTATION_START = 10.0
REPUTATION_MAX = 10.0
REGISTER_COST = 2.0
CONFIRM_REFUND = 3.0
REPUTATION_REGEN = 0.5
REPUTATION_REGEN_EVERY = 10   # without regen the KB deadlocks around tick 150
MAX_OPEN_CLAIMS = 2

MEMORY_SUMMARY_EVERY = 30
CHECKPOINT_EVERY = 25


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    ticks: int = 400
    n_agents: int = 5
    width: int = 20
    height: int = 20

    domain: str = "synthetic"
    water_hint: bool = True

    policy: str = "random"
    scaffold_level: str = "rules_only"
    model: str = "assistant:latest"

    teaching_enabled: bool = True
    max_concurrent_plots: int = MAX_CONCURRENT_PLOTS
    maturation_ticks: int = MATURATION_TICKS
    fallow_ticks: int = FALLOW_TICKS
    rejection_costs_tick: bool = True

    arm: str = "default"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scaffold_level not in ("bare", "rules_only", "rules_plus_method"):
            raise ValueError(f"bad scaffold_level: {self.scaffold_level}")
        if self.n_agents < 1:
            raise ValueError("need at least one agent")
        if self.max_concurrent_plots < 1:
            raise ValueError("need at least one plot")

    @property
    def hash(self) -> str:
        return blake2b_hex(asdict(self))
