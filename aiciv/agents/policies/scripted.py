"""Scripted control policies. Neither is given the hidden yield function.

These are the arms that make an LLM result interpretable. Without them,
"the LLM found the optimum" says nothing about whether reasoning was involved.

  ScriptedGreedy    a competent but naive experimenter. Optimises one variable
                    at a time -- the most natural thing to do, and exactly what
                    this landscape is built to punish. Converges on
                    (spacing=3, MARIGOLD) and stops.

  ScriptedFactorial sweeps companion x spacing as a full factorial. Finds the
                    interaction, at a known and measurable trial cost.

Together they bracket any LLM policy. Scoring like Greedy means no advantage
over naive method; scoring like Factorial at similar cost means brute force.
Only beating Factorial's trials_to_first_confirmed indicates inference.

Neither imports the domain. Both learn water from the PUBLIC crop_health
signal, exactly as any agent would.
"""

from __future__ import annotations

from collections import defaultdict

from ...world.actions import ActionProposal, Verb
from ..policy import PolicyContext, register

COMPANIONS = ("NONE", "CLOVER", "BEANS", "MARIGOLD", "THISTLE")
SPACINGS = (1, 2, 3, 4, 5)

REPLICATES = 6          # cell-mean SE ~0.20 at sigma 0.50; 4 was too noisy to
                        # rank 25 cells and both policies picked the wrong one
WATER_PROBES = 3        # crop_health is deterministic, so this converges fast
POST_REG_TARGET = 20    # per arm, comfortably above min_trials_per_group


class _ScriptedBase:
    """Farming loop, water control, and per-agent bookkeeping.

    The policy object persists across ticks, so it doubles as that agent's
    memory. Nothing here touches world state.
    """

    name = "scripted_base"

    def __init__(self) -> None:
        self.water_pref: dict[int, int] = {s: 2 for s in SPACINGS}
        self.water_done: dict[int, bool] = {s: False for s in SPACINGS}
        self.water_tries: dict[int, int] = {s: 0 for s in SPACINGS}
        self.results: dict[tuple[str, int], list[float]] = defaultdict(list)
        self.post_reg: dict[tuple[str, int], int] = defaultdict(int)
        self.pending: dict[int, tuple[str, int]] = {}
        self.seen: set = set()
        self.claim_id: str | None = None
        self.registered = False
        self.registered_tick: int | None = None
        self.trials_run = 0

    def reset(self, ctx: PolicyContext) -> None:
        pass

    def observe_result(self, result: dict, ctx: PolicyContext) -> None:
        pass

    # -- water: a feedback controller on a PUBLIC, noiseless signal ---------

    def _ingest(self, obs) -> None:
        """Fold finished harvests into memory.

        crop_health is deterministic and public, so steering water by it is
        honest -- and it reveals nothing about the companion x spacing
        interaction, which is the part that has to be discovered.
        """
        for h in obs.recent_harvests:
            key = (h.get("tile_id"), h.get("trial_id"))
            if key in self.seen:
                continue
            cond = self.pending.pop(h.get("tile_id"), None)
            if cond is None:
                continue
            self.seen.add(key)
            companion, spacing = cond
            self.trials_run += 1

            health = h.get("crop_health")
            if health == "wilted":
                self.water_pref[spacing] = min(4, self.water_pref[spacing] + 1)
            elif health == "waterlogged":
                self.water_pref[spacing] = max(0, self.water_pref[spacing] - 1)
            elif health == "healthy":
                self.water_done[spacing] = True

            if not self.water_done[spacing]:
                self.water_tries[spacing] += 1
                if self.water_tries[spacing] >= WATER_PROBES:
                    self.water_done[spacing] = True
                continue      # calibration trials are not evidence about cells

            self.results[(companion, spacing)].append(float(h.get("yield_kg", 0.0)))
            if self.registered:
                self.post_reg[(companion, spacing)] += 1

    # -- the loop ----------------------------------------------------------

    def decide(self, obs, ctx: PolicyContext) -> ActionProposal:
        self._ingest(obs)

        ready = [t for t in obs.nearby_tiles if t["crop_ready"] and t["mine"]]
        if ready:
            return ActionProposal(Verb.HARVEST, {"tile": list(ready[0]["tile"])})

        if obs.energy < 2.0:
            return ActionProposal(Verb.REST)
        if obs.food >= 1.0 and obs.energy < 6.0:
            return ActionProposal(Verb.EAT)

        act = self._epistemic_action(obs, ctx)
        if act is not None:
            return act

        free = [t for t in obs.nearby_tiles
                if t["terrain"] == "arable" and not t["planted"]]
        if free and obs.seeds > 0 and len(obs.plots) < ctx.config.max_concurrent_plots:
            companion, spacing = self._next_condition(ctx)
            tile = free[0]["tile"]
            self.pending[tile[1] * 20 + tile[0]] = (companion, spacing)
            return ActionProposal(Verb.PLANT, {
                "tile": list(tile), "spacing": spacing,
                "water": min(obs.water_stock, self.water_pref[spacing]),
                "companion": companion,
            })

        waiting = [p for p in obs.my_plots if p["ready"]]
        if waiting:
            step = self._step_toward(obs, waiting[0]["tile"])
            if step is not None:
                return ActionProposal(Verb.MOVE, {"tile": step})

        arable = [t["tile"] for t in obs.nearby_tiles
                  if t["terrain"] == "arable" and t["tile"] != [obs.x, obs.y]]
        if arable:
            return ActionProposal(
                Verb.MOVE, {"tile": list(arable[int(ctx.rng.integers(0, len(arable)))])})
        return ActionProposal(Verb.NOOP)

    @staticmethod
    def _step_toward(obs, target):
        tx, ty = target
        best, best_d = None, None
        for t in obs.nearby_tiles:
            x, y = t["tile"]
            if t["terrain"] != "arable" or [x, y] == [obs.x, obs.y]:
                continue
            d = max(abs(tx - x), abs(ty - y))
            if best_d is None or d < best_d:
                best, best_d = [x, y], d
        if best is None or best_d >= max(abs(tx - obs.x), abs(ty - obs.y)):
            return None
        return best

    # -- calibration then exploration --------------------------------------

    def _water_phase(self) -> tuple[str, int] | None:
        """Fix water per spacing BEFORE measuring cells.

        Learning water during the factorial poisons the early cells: a cell
        measured at the wrong water looks bad for a reason that has nothing to
        do with its companion.
        """
        for s in SPACINGS:
            if not self.water_done[s]:
                return ("NONE", s)
        return None

    def mean(self, companion: str, spacing: int) -> float | None:
        vals = self.results.get((companion, spacing))
        return sum(vals) / len(vals) if vals else None

    def _next_condition(self, ctx) -> tuple[str, int]:
        raise NotImplementedError

    def _epistemic_action(self, obs, ctx):
        return None

    # -- claim machinery, shared so the arms differ only in SEARCH ----------

    def _register_pending(self, obs):
        for c in obs.my_claims:
            if c["state"] == "proposed":
                self.claim_id = c["claim_id"]
                self.registered = True
                self.post_reg.clear()
                return ActionProposal(Verb.REGISTER_CLAIM,
                                      {"claim_id": c["claim_id"]})
        return None

    def _propose(self, companion: str, spacing: int):
        return ActionProposal(
            Verb.PROPOSE_CLAIM,
            {
                "spec": {"companion": {"op": "eq", "value": companion},
                         "spacing": {"op": "eq", "value": spacing}},
                "baseline": {"companion": {"op": "eq", "value": "NONE"},
                             "spacing": {"op": "eq", "value": spacing}},
                "direction": "increase",
                "min_delta": 0.3,
                "claim_type": "comparison",
            },
            message=f"{companion} at spacing {spacing} gives more than no "
                    f"companion at the same spacing",
        )


@register
class ScriptedGreedy(_ScriptedBase):
    """One factor at a time. Provably trapped by this landscape.

    Sweeps spacing with no companion, fixes the best, then sweeps companions at
    that spacing. Every single-variable move from the result is worse, so it
    stops -- never trying the two-variable move that would escape.
    """

    name = "scripted_greedy"

    def __init__(self) -> None:
        super().__init__()
        self.stage = "water"
        self.best_spacing: int | None = None
        self.best_companion: str | None = None

    def _next_condition(self, ctx):
        w = self._water_phase()
        if w is not None:
            return w

        if self.best_spacing is None:
            for s in SPACINGS:
                if len(self.results[("NONE", s)]) < REPLICATES:
                    return ("NONE", s)
            self.best_spacing = max(
                SPACINGS, key=lambda s: self.mean("NONE", s) or -1e9)

        if self.best_companion is None:
            for c in COMPANIONS:
                if len(self.results[(c, self.best_spacing)]) < REPLICATES:
                    return (c, self.best_spacing)
            self.best_companion = max(
                COMPANIONS, key=lambda c: self.mean(c, self.best_spacing) or -1e9)

        # Converged. Gather evidence for the claim it can actually make.
        c, s = self.best_companion, self.best_spacing
        if c == "NONE":
            return ("NONE", s)
        return (c, s) if self.post_reg[(c, s)] <= self.post_reg[("NONE", s)] \
            else ("NONE", s)

    def _epistemic_action(self, obs, ctx):
        reg = self._register_pending(obs)
        if reg is not None:
            return reg
        if self.registered or self.claim_id is not None:
            return None
        if self.best_companion in (None, "NONE"):
            return None
        if len(self.results[(self.best_companion, self.best_spacing)]) < REPLICATES:
            return None
        self.claim_id = "pending"
        return self._propose(self.best_companion, self.best_spacing)


@register
class ScriptedFactorial(_ScriptedBase):
    """Full companion x spacing factorial, then a pre-registered claim.

    Expensive and unsubtle, but it does see the interaction. Its trial cost is
    the yardstick an LLM must beat to be doing more than brute force.
    """

    name = "scripted_factorial"

    def __init__(self) -> None:
        super().__init__()
        self.cells = [(c, s) for s in SPACINGS for c in COMPANIONS]
        self.best: tuple[str, int] | None = None

    def _next_condition(self, ctx):
        w = self._water_phase()
        if w is not None:
            return w

        for cell in self.cells:
            if len(self.results[cell]) < REPLICATES:
                return cell

        if self.best is None:
            c, s = max(self.cells, key=lambda cs: self.mean(*cs) or -1e9)
            self.best = (c, s)

        c, s = self.best
        if c == "NONE":
            return (c, s)
        # Strict alternation on POST-REGISTRATION counts. Balancing on all-time
        # counts leaves one arm short after registration and the claim stalls
        # forever on insufficient_evidence.
        return (c, s) if self.post_reg[(c, s)] <= self.post_reg[("NONE", s)] \
            else ("NONE", s)

    def _epistemic_action(self, obs, ctx):
        reg = self._register_pending(obs)
        if reg is not None:
            return reg
        if self.registered or self.claim_id is not None or self.best is None:
            return None
        c, s = self.best
        if c == "NONE":
            return None
        self.claim_id = "pending"
        return self._propose(c, s)
