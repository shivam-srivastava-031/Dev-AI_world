"""Typed identifiers. NewType keeps them free at runtime but catches the
classic bug of passing a tile_id where an agent_id belongs."""

from __future__ import annotations

from typing import NewType

AgentId = NewType("AgentId", int)
TileId = NewType("TileId", int)
TrialId = NewType("TrialId", int)
ClaimId = NewType("ClaimId", str)
RunId = NewType("RunId", str)


def claim_id(n: int) -> ClaimId:
    return ClaimId(f"clm_{n:05d}")


def tile_id(x: int, y: int, width: int) -> TileId:
    return TileId(y * width + x)


def tile_xy(t: TileId, width: int) -> tuple[int, int]:
    return (int(t) % width, int(t) // width)
