"""Determinism substrate: key-derived RNG, stable hashing, no global randomness."""

from __future__ import annotations

import ast
import pathlib

import pytest

from aiciv.hashing import (
    blake2b_hex, blake2b_int, canonical_json, fmt_float, q, sign_row, verify_row,
)
from aiciv.rng import derive_int, derive_rng, derive_seed, derive_unit
from aiciv.world.grid import generate_world

PKG = pathlib.Path(__file__).resolve().parents[1] / "aiciv"


# --- hashing -------------------------------------------------------------

def test_float_quantisation_is_stable_and_signless():
    assert q(1.0005) == 1.001
    assert q(-0.0) == 0.0
    assert fmt_float(2 / 3) == "0.667"
    assert fmt_float(1e-9) == "0.000"          # never scientific notation


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"s": {3, 1, 2}}) == canonical_json({"s": [1, 2, 3]})


def test_canonical_json_absorbs_float_drift():
    """A trailing-bit difference from a different BLAS must not change a hash."""
    assert canonical_json({"y": 0.1 + 0.2}) == canonical_json({"y": 0.3})


def test_hash_separator_prevents_concatenation_collisions():
    assert blake2b_hex("ab", "c") != blake2b_hex("a", "bc")


def test_signature_detects_tampering():
    secret = b"run-secret"
    row = {"trial_id": 4, "yield_kg": 3.210, "agent_id": 1}
    row["signature"] = sign_row(secret, row)
    assert verify_row(secret, row)

    forged = dict(row, yield_kg=9.999)          # the realistic attack
    assert not verify_row(secret, forged)
    assert not verify_row(b"other-secret", row)
    assert not verify_row(secret, {k: v for k, v in row.items() if k != "signature"})


# --- key-derived RNG ------------------------------------------------------

def test_derive_rng_is_reproducible():
    assert derive_rng(42, "yield", 1, 2).normal() == derive_rng(42, "yield", 1, 2).normal()


def test_derive_rng_order_independence():
    """The property the whole engine rests on: consumption order cannot matter.

    Drawing key A then key B must give each key the same value as drawing B
    then A. This is why checkpoints carry no RNG state and why adding a feature
    that draws an extra number cannot shift anyone else's draws.
    """
    keys = [("yield", 1, 2), ("yield", 3, 4), ("water", 9, 9), ("noise", 0, 0)]
    forward = {k: derive_rng(42, *k).normal() for k in keys}
    backward = {k: derive_rng(42, *k).normal() for k in reversed(keys)}
    assert forward == backward


def test_different_keys_and_seeds_diverge():
    assert derive_seed(42, "a") != derive_seed(42, "b")
    assert derive_seed(42, "a") != derive_seed(43, "a")
    assert derive_rng(42, "yield", 1).normal() != derive_rng(42, "yield", 2).normal()


def test_derive_int_and_unit_ranges():
    for i in range(200):
        assert 0 <= derive_int(42, "x", i, lo=0, hi=10) < 10
        assert 0.0 <= derive_unit(42, "u", i) < 1.0


def test_derive_rng_requires_a_purpose_key():
    with pytest.raises(ValueError):
        derive_rng(42)


# --- the AST scan ---------------------------------------------------------

def test_no_global_rng_usage():
    """Nothing outside rng.py may touch module-level randomness.

    A single ``np.random.normal()`` anywhere else silently destroys replay, and
    it is the kind of line that gets added in a hurry and never noticed.
    """
    offenders: list[str] = []

    def dotted_name(node: ast.AST) -> str:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    for path in PKG.rglob("*.py"):
        if path.name == "rng.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # Only CALLS matter. A bare ``np.random.Generator`` in a type
            # annotation is not a source of randomness.
            if not isinstance(node, ast.Call):
                continue
            dotted = dotted_name(node.func)
            if dotted.startswith(("random.", "np.random.", "numpy.random.")):
                # default_rng is the sanctioned constructor, seeded explicitly
                # by rng.derive_rng; nothing else may reach module-level state.
                if not dotted.endswith("default_rng"):
                    rel = path.relative_to(PKG.parent)
                    offenders.append(f"{rel}:{node.lineno} {dotted}()")
    assert not offenders, "global RNG use outside rng.py:\n" + "\n".join(offenders)


# --- world generation ------------------------------------------------------

def test_world_generation_is_a_pure_function_of_seed():
    a, b = generate_world(42), generate_world(42)
    assert a.tiles == b.tiles
    assert generate_world(43).tiles != a.tiles


def test_stratum_partition_matches_target_fractions():
    counts = generate_world(42).stratum_counts()
    total = sum(counts.values())
    fracs = {s.value: v / total for s, v in counts.items()}
    assert fracs["discovery"] == pytest.approx(0.70, abs=0.06)
    assert fracs["confirmation"] == pytest.approx(0.20, abs=0.06)
    assert fracs["holdout"] == pytest.approx(0.10, abs=0.06)


def test_strata_are_interleaved_not_spatial():
    """A homebody agent must still be able to reach the confirmation stage.

    Spatial blocks would mean an agent farming near its start only ever produces
    discovery-stratum trials, making CONFIRMED unreachable without wandering.
    """
    grid = generate_world(42)
    lacking = 0
    regions = 0
    for cy in range(2, 18, 3):
        for cx in range(2, 18, 3):
            regions += 1
            local = {grid.at(x, y).stratum
                     for x in range(cx - 2, cx + 3)
                     for y in range(cy - 2, cy + 3)}
            if len(local) < 3:
                lacking += 1
    assert lacking / regions < 0.10
