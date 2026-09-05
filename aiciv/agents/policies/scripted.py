"""Scripted control policies. Neither is given the hidden yield function.

These are the arms that make an LLM result interpretable. Without them, "the
LLM found the optimum" says nothing about whether reasoning was involved.

  ScriptedGreedy    a competent but naive experimenter. Optimises one variable
                    at a time -- the most natural thing to do, and exactly what
                    a deceptive landscape punishes.

  ScriptedFactorial crosses the domain's two search axes as a full factorial.
                    Finds interactions, at a known and measurable trial cost.

Together they bracket any LLM policy. Scoring like Greedy means no advantage
over naive method; scoring like Factorial at similar cost means brute force.
Only beating Factorial's trials_to_first_confirmed indicates inference.

Neither imports a domain. Both read the PUBLIC metadata the domain advertises
through PolicyContext -- which parameters exist, which two are worth crossing,
which one the crop_health signal speaks to -- and learn everything else from
outcomes. That metadata names parameters an agent can already see; it says
nothing about what any setting does or where the optimum lies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...world.actions import ActionProposal, Verb
from ..policy import PolicyContext, register

REPLICATES = 6          # cell-mean SE ~0.20 at sigma 0.50; 4 was too noisy to
                        # rank 25 cells and both policies picked the wrong one
WATER_PROBES = 3        # crop_health is deterministic, so this converges fast


class _ScriptedBase:
    """Farming loop, calibration control, and per-agent bookkeeping.

    The policy object persists across ticks, so it doubles as that agent's
    memory. Nothing here touches world state.
    """

    name = "scripted_base"

    def __init__(self) -> None:
        self.calib_pref: dict[Any, Any] = {}
        self.calib_done: dict[Any, bool] = {}
        self.calib_tries: dict[Any, int] = defaultdict(int)
        self.results: dict[tuple, list[float]] = defaultdict(list)
        self.post_reg: dict[tuple, int] = defaultdict(int)
        self.pending: dict[int, tuple] = {}
        self.seen: set = set()
        self.claim_id: str | None = None
        self.registered = False
        self.trials_run = 0
        self._axes: tuple[str, str] | None = None

    # -- domain metadata, read once from the context -----------------------

    def _setup(self, ctx: PolicyContext) -> None:
        if self._axes is not None:
            return
        axes = tuple(ctx.sweep_axes) if ctx.sweep_axes else ()
        if len(axes) < 2:
            free = [n for n in sorted(ctx.param_space) if n != ctx.schedule_axis]
            axes = tuple(free[:2])
        self._axes = (axes[0], axes[1])       # (compared, held constant)
        self.calib_axis = ctx.calibration_axis
        calib_vals = ctx.param_space.get(self.calib_axis, ())
        mid = calib_vals[len(calib_vals) // 2] if calib_vals else None
        for level in ctx.param_space[self._axes[1]]:
            self.calib_pref[level] = mid
            self.calib_done[level] = not calib_vals
        self.other_axes = [n for n in sorted(ctx.param_space)
                           if n not in (*self._axes, self.calib_axis,
                                        ctx.schedule_axis)]

    @property
    def compared(self) -> str:
        return self._axes[0]

    @property
    def context(self) -> str:
        return self._axes[1]

    def reset(self, ctx: PolicyContext) -> None:
        pass

    def observe_result(self, result: dict, ctx: PolicyContext) -> None:
        pass

    # -- calibration: a feedback controller on a PUBLIC, noiseless signal ---

    def _ingest(self, obs, ctx) -> None:
        """Fold finished harvests into memory.

        crop_health is deterministic and public, so steering the calibration
        axis by it is honest -- and it says nothing about the interaction
        between the two search axes, which is the part to be discovered.
        """
        vals = ctx.param_space.get(self.calib_axis, ())
        for h in obs.recent_harvests:
            key = (h.get("tile_id"), h.get("trial_id"))
            if key in self.seen:
                continue
            cell = self.pending.pop(h.get("tile_id"), None)
            if cell is None:
                continue
            self.seen.add(key)
            self.trials_run += 1
            ctx_level = cell[1]

            health = h.get("crop_health")
            if vals and health in ("wilted", "waterlogged"):
                cur = self.calib_pref.get(ctx_level, vals[len(vals) // 2])
                i = vals.index(cur) if cur in vals else len(vals) // 2
                i = min(len(vals) - 1, i + 1) if health == "wilted" else max(0, i - 1)
                self.calib_pref[ctx_level] = vals[i]
            elif health == "healthy":
                self.calib_done[ctx_level] = True

            if not self.calib_done.get(ctx_level, True):
                self.calib_tries[ctx_level] += 1
                if self.calib_tries[ctx_level] >= WATER_PROBES:
                    self.calib_done[ctx_level] = True
                continue      # calibration trials are not evidence about cells

            self.results[cell].append(float(h.get("yield_kg", 0.0)))
            if self.registered:
                self.post_reg[cell] += 1

    # -- the loop ----------------------------------------------------------

    def decide(self, obs, ctx: PolicyContext) -> ActionProposal:
        self._setup(ctx)
        self._ingest(obs, ctx)

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
                if t["terrain"] == "arable" and not t["planted"]
                and not t.get("fallow")]
        if free and obs.seeds > 0 and len(obs.plots) < ctx.config.max_concurrent_plots:
            cell = self._next_condition(ctx)
            tile = free[0]["tile"]
            self.pending[tile[1] * 20 + tile[0]] = cell
            return ActionProposal(Verb.PLANT, self._plant_params(tile, cell, ctx))

        waiting = [p for p in obs.my_plots if p["ready"]]
        if waiting:
            step = self._step_toward(obs, waiting[0]["tile"])
            if step is not None:
                return ActionProposal(Verb.MOVE, {"tile": step})

        arable = [t["tile"] for t in obs.nearby_tiles
                  if t["terrain"] == "arable" and t["tile"] != [obs.x, obs.y]]
        if arable:
            return ActionProposal(
                Verb.MOVE,
                {"tile": list(arable[int(ctx.rng.integers(0, len(arable)))])})
        return ActionProposal(Verb.NOOP)

    def _plant_params(self, tile, cell, ctx) -> dict:
        params: dict[str, Any] = {"tile": list(tile)}
        params[self.compared], params[self.context] = cell
        if self.calib_axis:
            vals = ctx.param_space[self.calib_axis]
            params[self.calib_axis] = self.calib_pref.get(
                cell[1], vals[len(vals) // 2])
        for name in self.other_axes:
            vals = ctx.param_space[name]
            params[name] = vals[len(vals) // 2]
        return params

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

    def _calibration_phase(self, ctx) -> tuple | None:
        """Fix the calibration axis BEFORE measuring cells.

        Learning it during the factorial poisons early cells: a cell measured
        at the wrong setting looks bad for a reason unrelated to its treatment.
        """
        reference = ctx.param_space[self.compared][0]
        for level in ctx.param_space[self.context]:
            if not self.calib_done.get(level, True):
                return (reference, level)
        return None

    def mean(self, cell: tuple) -> float | None:
        vals = self.results.get(cell)
        return sum(vals) / len(vals) if vals else None

    def _next_condition(self, ctx) -> tuple:
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

    def _propose(self, cell: tuple, ctx):
        """Compare a level of the search axis against the domain's reference
        level, holding the context axis fixed."""
        treat, context = cell
        reference = ctx.param_space[self.compared][0]
        return ActionProposal(
            Verb.PROPOSE_CLAIM,
            {
                "spec": {self.compared: {"op": "eq", "value": treat},
                         self.context: {"op": "eq", "value": context}},
                "baseline": {self.compared: {"op": "eq", "value": reference},
                             self.context: {"op": "eq", "value": context}},
                "direction": "increase",
                "min_delta": 0.3,
                "claim_type": "comparison",
            },
            message=f"{self.compared} {treat} gives more than {reference} "
                    f"when {self.context} is {context}",
        )


@register
class ScriptedGreedy(_ScriptedBase):
    """One factor at a time. Trapped by a deceptive landscape.

    Sweeps the context axis at the reference treatment, fixes the best, then
    sweeps treatments there. Every single-variable move from the result is
    worse, so it stops -- without trying the two-variable move that escapes.
    """

    name = "scripted_greedy"

    def __init__(self) -> None:
        super().__init__()
        self.best_context = None
        self.best_treat = None

    def _next_condition(self, ctx):
        c = self._calibration_phase(ctx)
        if c is not None:
            return c

        reference = ctx.param_space[self.compared][0]
        contexts = ctx.param_space[self.context]
        treats = ctx.param_space[self.compared]

        if self.best_context is None:
            for level in contexts:
                if len(self.results[(reference, level)]) < REPLICATES:
                    return (reference, level)
            self.best_context = max(
                contexts, key=lambda L: self.mean((reference, L)) or -1e9)

        if self.best_treat is None:
            for t in treats:
                if len(self.results[(t, self.best_context)]) < REPLICATES:
                    return (t, self.best_context)
            self.best_treat = max(
                treats, key=lambda t: self.mean((t, self.best_context)) or -1e9)

        cell = (self.best_treat, self.best_context)
        base = (reference, self.best_context)
        if self.best_treat == reference:
            return cell
        return cell if self.post_reg[cell] <= self.post_reg[base] else base

    def _epistemic_action(self, obs, ctx):
        reg = self._register_pending(obs)
        if reg is not None:
            return reg
        if self.registered or self.claim_id is not None:
            return None
        if self.best_treat is None or \
                self.best_treat == ctx.param_space[self.compared][0]:
            return None
        if len(self.results[(self.best_treat, self.best_context)]) < REPLICATES:
            return None
        self.claim_id = "pending"
        return self._propose((self.best_treat, self.best_context), ctx)


@register
class ScriptedFactorial(_ScriptedBase):
    """Full factorial over the domain's two search axes, then a claim.

    Expensive and unsubtle, but it sees interactions. Its trial cost is the
    yardstick an LLM must beat to be doing more than brute force.
    """

    name = "scripted_factorial"

    def __init__(self) -> None:
        super().__init__()
        self.best: tuple | None = None

    def _cells(self, ctx):
        return [(t, c) for c in ctx.param_space[self.context]
                for t in ctx.param_space[self.compared]]

    def _next_condition(self, ctx):
        c = self._calibration_phase(ctx)
        if c is not None:
            return c

        for cell in self._cells(ctx):
            if len(self.results[cell]) < REPLICATES:
                return cell

        if self.best is None:
            self.best = max(self._cells(ctx), key=lambda c: self.mean(c) or -1e9)

        treat, context = self.best
        reference = ctx.param_space[self.compared][0]
        if treat == reference:
            return self.best
        # Strict alternation on POST-REGISTRATION counts. Balancing on all-time
        # counts leaves one arm short after registration and the claim stalls
        # forever on insufficient_evidence.
        base = (reference, context)
        return self.best if self.post_reg[self.best] <= self.post_reg[base] else base

    def _epistemic_action(self, obs, ctx):
        reg = self._register_pending(obs)
        if reg is not None:
            return reg
        if self.registered or self.claim_id is not None or self.best is None:
            return None
        if self.best[0] == ctx.param_space[self.compared][0]:
            return None
        self.claim_id = "pending"
        return self._propose(self.best, ctx)
