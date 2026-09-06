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
    #: Threads for the DECIDE phase. Safe at any value because every agent
    #: decides against the SAME frozen snapshot and results are reassembled
    #: in agent_id order -- the property the phase separation was designed
    #: for. 1 keeps scripted runs single-threaded; raise it for LLM arms,
    #: where a decision costs ~25s and the run is entirely latency-bound.
    decide_workers: int = 1

    arm: str = "default"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    #: The observer's brief, when the run was launched under one. Empty means
    #: unsteered: nobody told these agents what to aim at.
    #:
    #: Both fields are part of the config and therefore part of ``hash``, on
    #: purpose. Two runs with the same seed and different briefs are two
    #: different experiments, and if they shared a config hash the manifest
    #: would claim they were the same one.
    brief_id: str = ""
    directives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scaffold_level not in ("bare", "rules_only", "rules_plus_method"):
            raise ValueError(f"bad scaffold_level: {self.scaffold_level}")
        # Coerce rather than reject: a config round-tripped through JSON comes
        # back with a list, and a list and a tuple of the same strings must not
        # produce two different config hashes for the same run.
        if not isinstance(self.directives, tuple):
            object.__setattr__(self, "directives", tuple(self.directives))
        if any(not isinstance(d, str) for d in self.directives):
            raise ValueError("directives must be strings")
        if self.n_agents < 1:
            raise ValueError("need at least one agent")
        if self.max_concurrent_plots < 1:
            raise ValueError("need at least one plot")

    #: Config fields that are omitted from the hash when they are empty.
    #:
    #: Steering is ADDED to the identity of a run rather than folded into it.
    #: Hashing `brief_id=""` and `directives=()` alongside everything else
    #: changed the hash of every unsteered config, so every run recorded before
    #: briefs existed failed its own replay with "config hash mismatch" -- a
    #: divergence that never happened, reported by the one check that exists to
    #: tell you when a real one did.
    #:
    #: A steered run still hashes differently from an unsteered one, which is
    #: the property that matters: two runs with the same seed and different
    #: briefs are two different experiments and must not share an identity.
    _OMITTED_WHEN_EMPTY = ("brief_id", "directives")

    @property
    def hash(self) -> str:
        fields = asdict(self)
        for name in self._OMITTED_WHEN_EMPTY:
            if not fields[name]:
                del fields[name]
        return blake2b_hex(fields)
