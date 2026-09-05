"""Trial records: world-authored evidence.

An agent NEVER submits trial data. It performs actions; the world records what
happened and signs the row. When an agent wants a trial to support a claim it
submits a trial_id integer, and the verifier re-reads the row and re-checks the
signature. This is what makes fabrication structurally impossible rather than
merely discouraged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..hashing import q, sign_row, verify_row
from ..ids import AgentId, TileId, TrialId

#: Fields an agent may see about its own trial. Everything else -- notably the
#: tile stratum and the signature -- is HIDDEN and must be projected away
#: before anything reaches an Observation.
AGENT_VISIBLE_FIELDS = (
    "trial_id", "tile_id", "planted_tick", "harvested_tick",
    "spacing", "plant_day", "water", "companion",
    "soil_band", "skill_at_plant", "yield_kg", "crop_health",
)


@dataclass(frozen=True)
class Trial:
    trial_id: TrialId
    agent_id: AgentId
    tile_id: TileId
    planted_tick: int
    harvested_tick: int
    spacing: int
    plant_day: int
    water: int
    companion: str
    soil_band: int
    skill_at_plant: float        # PRE_TREATMENT for this trial
    yield_kg: float              # what the agent observed
    crop_health: str
    stratum: str                 # HIDDEN: decides which stage this counts for
    true_mu: float = field(repr=False, default=0.0)   # METRICS_ONLY
    signature: str = ""

    def row(self) -> dict[str, Any]:
        """The signed payload. Excludes true_mu: ground truth is not evidence."""
        d = asdict(self)
        d.pop("true_mu", None)
        return d

    def signed(self, secret: bytes) -> "Trial":
        return Trial(**{**asdict(self), "signature": sign_row(secret, self.row())})

    def verify(self, secret: bytes) -> bool:
        return verify_row(secret, self.row())

    def agent_view(self) -> dict[str, Any]:
        """Projection safe to place in an Observation or a memory record."""
        d = asdict(self)
        return {k: d[k] for k in AGENT_VISIBLE_FIELDS}

    def matches(self, conditions: dict[str, Any]) -> bool:
        """Does this trial fall inside a claim's parameter region?

        Conditions are {param: {"op": ..., ...}}; an absent param is unconstrained.
        """
        d = asdict(self)
        for name, cond in conditions.items():
            if name not in d:
                return False
            v = d[name]
            op = cond.get("op")
            if op == "eq" and v != cond["value"]:
                return False
            if op == "in" and v not in cond["values"]:
                return False
            if op == "range" and not (cond["lo"] <= v <= cond["hi"]):
                return False
            if op not in ("eq", "in", "range"):
                return False
        return True


def make_trial(
    *,
    trial_id: int,
    agent_id: int,
    tile_id: int,
    planted_tick: int,
    harvested_tick: int,
    recipe: dict[str, Any],
    soil_band: int,
    skill_at_plant: float,
    yield_kg: float,
    crop_health: str,
    stratum: str,
    true_mu: float,
    secret: bytes,
) -> Trial:
    """Construct and sign a trial. The ONLY sanctioned way to create evidence."""
    t = Trial(
        trial_id=TrialId(trial_id),
        agent_id=AgentId(agent_id),
        tile_id=TileId(tile_id),
        planted_tick=planted_tick,
        harvested_tick=harvested_tick,
        spacing=int(recipe["spacing"]),
        plant_day=int(recipe["plant_day"]),
        water=int(recipe["water"]),
        companion=str(recipe["companion"]),
        soil_band=int(soil_band),
        skill_at_plant=q(skill_at_plant),
        yield_kg=q(yield_kg),
        crop_health=crop_health,
        stratum=stratum,
        true_mu=q(true_mu),
    )
    return t.signed(secret)
