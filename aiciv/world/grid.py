"""The grid: tiles, soil, and the hidden three-way stratum partition.

The stratum is the mechanism behind SUPPORTED -> CONFIRMED -> GENERALIZED. It is
HIDDEN: it never appears in an Observation, and agents are deliberately NOT
blocked from planting on holdout tiles, because blocking would leak the
partition through rejection codes. The stratum only decides which trials count
toward which verification stage.

Partition is hash-interleaved rather than spatial. A spatial partition would
mean an agent farming near home only ever produces discovery-stratum trials,
which would make confirmation unreachable for anyone who does not wander.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..ids import TileId, tile_xy
from ..rng import derive_unit

WIDTH = 20
HEIGHT = 20
N_SOIL_BANDS = 5


class Stratum(str, Enum):
    DISCOVERY = "discovery"
    CONFIRMATION = "confirmation"
    HOLDOUT = "holdout"


#: Fractions must sum to 1.0. Confirmation needs enough mass that a claim can
#: actually reach CONFIRMED; holdout stays small because it is only spent once
#: per claim.
STRATUM_FRACTIONS: dict[Stratum, float] = {
    Stratum.DISCOVERY: 0.70,
    Stratum.CONFIRMATION: 0.20,
    Stratum.HOLDOUT: 0.10,
}


class Terrain(str, Enum):
    ARABLE = "arable"
    WATER = "water"
    ROCK = "rock"


@dataclass(frozen=True)
class Tile:
    tile_id: TileId
    x: int
    y: int
    terrain: Terrain
    soil_band: int          # AGENT_OBSERVABLE via INSPECT_TILE
    stratum: Stratum        # HIDDEN

    @property
    def arable(self) -> bool:
        return self.terrain is Terrain.ARABLE


@dataclass(frozen=True)
class Grid:
    width: int
    height: int
    tiles: tuple[Tile, ...]

    def __post_init__(self) -> None:
        if len(self.tiles) != self.width * self.height:
            raise ValueError("tile count does not match grid dimensions")

    def at(self, x: int, y: int) -> Tile:
        if not self.in_bounds(x, y):
            raise IndexError(f"({x},{y}) out of bounds")
        return self.tiles[y * self.width + x]

    def by_id(self, tile_id: TileId) -> Tile:
        return self.tiles[int(tile_id)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, x: int, y: int) -> list[Tile]:
        """Chebyshev-1 neighbourhood, in deterministic scan order."""
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if self.in_bounds(x + dx, y + dy):
                    out.append(self.at(x + dx, y + dy))
        return out

    def arable_tiles(self) -> list[Tile]:
        return [t for t in self.tiles if t.arable]

    def stratum_counts(self) -> dict[Stratum, int]:
        counts = {s: 0 for s in Stratum}
        for t in self.arable_tiles():
            counts[t.stratum] += 1
        return counts


def _assign_stratum(seed: int, tile_id: int) -> Stratum:
    """Hash-interleaved assignment. Stable for a given (seed, tile_id)."""
    u = derive_unit(seed, "stratum", tile_id)
    acc = 0.0
    for stratum, frac in STRATUM_FRACTIONS.items():
        acc += frac
        if u < acc:
            return stratum
    return Stratum.HOLDOUT


def generate_world(seed: int, *, width: int = WIDTH, height: int = HEIGHT) -> Grid:
    """Pure function of (seed, width, height). No global RNG, no wall clock.

    Terrain is mostly arable with a little water and rock so that MOVE and
    PLANT have real spatial constraints; soil banding is spatially smooth so a
    soil gradient exists to be discovered, rather than being white noise.
    """
    tiles: list[Tile] = []
    for y in range(height):
        for x in range(width):
            tid = TileId(y * width + x)

            terr_u = derive_unit(seed, "terrain", tid)
            if terr_u < 0.06:
                terrain = Terrain.WATER
            elif terr_u < 0.11:
                terrain = Terrain.ROCK
            else:
                terrain = Terrain.ARABLE

            # Smooth-ish soil: a low-frequency deterministic field plus jitter,
            # so neighbouring tiles correlate and a gradient is learnable.
            base = (x / max(width - 1, 1)) * 0.6 + (y / max(height - 1, 1)) * 0.4
            jitter = derive_unit(seed, "soil", tid) * 0.5 - 0.25
            band = int(min(N_SOIL_BANDS - 1, max(0, round((base + jitter) * (N_SOIL_BANDS - 1)))))

            tiles.append(Tile(
                tile_id=tid, x=x, y=y, terrain=terrain,
                soil_band=band, stratum=_assign_stratum(seed, tid),
            ))
    return Grid(width=width, height=height, tiles=tuple(tiles))
