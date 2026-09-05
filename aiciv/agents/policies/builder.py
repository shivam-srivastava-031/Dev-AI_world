"""The Builder: the archetype that closes WORLD -> CIVILIZATION -> WORLD.

Every other policy treats the world as fixed and gets better at measuring it.
This one turns confirmed knowledge into a thing that exists, and that thing
changes what everyone can attempt.

The sequence it demonstrates end to end:

    sweep the water axis
      -> a confirmed claim about water
      -> compiled into a Procedure
      -> plus practice, an irrigation Capability
      -> BUILD_CHANNEL puts an Artifact in the world
      -> the artifact raises water available nearby
      -> recipes that were previously unaffordable become attemptable
      -> including for agents who never learned any of it

That last line is what makes it technology rather than a private skill.
"""

from __future__ import annotations

from ...world.actions import ActionProposal, Verb
from ..policy import PolicyContext, register
from .archetypes import _Social

REPLICATES = 6


@register
class Builder(_Social):
    """Sweeps the water axis, claims it, then builds on what it learned."""

    name = "builder"
    teaches = True

    def __init__(self) -> None:
        super().__init__()
        self.built = False
        self.water_best = None

    # -- search the axis that leads somewhere ------------------------------

    def _next_condition(self, ctx):
        c = self._calibration_phase(ctx)
        if c is not None:
            return c
        # Sweep the calibration axis itself rather than the usual pair: a claim
        # about water is what compiles into an irrigation procedure.
        vals = ctx.param_space.get(self.calib_axis, ())
        reference = ctx.param_space[self.compared][0]
        context = ctx.param_space[self.context][len(
            ctx.param_space[self.context]) // 2]
        for v in vals:
            if len(self.results[(reference, context, v)]) < REPLICATES:
                self._forced_calib = v
                return (reference, context)
        if self.water_best is None and vals:
            self.water_best = max(
                vals, key=lambda v: self.mean((reference, context, v)) or -1e9)

        # Alternate treatment and control on POST-REGISTRATION counts. Planting
        # only the winning level leaves the baseline arm empty, and the claim
        # dies of insufficient_evidence with n=61/0 -- plenty of data, all of it
        # on one side of the comparison it registered.
        if vals and self.water_best is not None:
            treat = (reference, context, self.water_best)
            base = (reference, context, vals[0])
            self._forced_calib = (
                self.water_best if self.post_reg[treat] <= self.post_reg[base]
                else vals[0])
        else:
            self._forced_calib = self.water_best
        return (reference, context)

    def _plant_params(self, tile, cell, ctx) -> dict:
        params = super()._plant_params(tile, cell, ctx)
        forced = getattr(self, "_forced_calib", None)
        if forced is not None and self.calib_axis:
            params[self.calib_axis] = forced
        return params

    def _ingest(self, obs, ctx) -> None:
        """Key results by the calibration level too, so the water sweep is
        actually measurable rather than averaged away."""
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
            level = getattr(self, "_planted_calib", {}).pop(h.get("tile_id"), None)
            ctx_level = cell[1]
            self.calib_done[ctx_level] = True
            if level is None:
                continue
            self.results[(cell[0], cell[1], level)].append(
                float(h.get("yield_kg", 0.0)))
            if self.registered:
                self.post_reg[(cell[0], cell[1], level)] += 1

    def decide(self, obs, ctx: PolicyContext) -> ActionProposal:
        proposal = super().decide(obs, ctx)

        # Wait rather than repeatedly asking for water it cannot carry. Without
        # this the Builder spends the run on rejected PLANTs -- 1167 of them in
        # a 1400-tick run -- and never gathers the evidence for its own claim.
        # Scarcity is the point of irrigation, but an agent has to be able to
        # bootstrap out of it.
        if proposal.verb is Verb.PLANT and self.calib_axis in proposal.params:
            wanted = proposal.params[self.calib_axis]
            if isinstance(wanted, int) and obs.water_stock < wanted:
                return ActionProposal(Verb.REST)
            store = getattr(self, "_planted_calib", None)
            if store is None:
                store = self._planted_calib = {}
            tile = proposal.params["tile"]
            store[tile[1] * 20 + tile[0]] = wanted
        return proposal

    # -- claim the water axis, then build on it ----------------------------

    def _epistemic_action(self, obs, ctx):
        reg = self._register_pending(obs)
        if reg is not None:
            return reg

        build = self._build_action(obs, ctx)
        if build is not None:
            return build

        if self.registered or self.claim_id is not None:
            return None
        if self.water_best is None:
            return None
        vals = ctx.param_space.get(self.calib_axis, ())
        if not vals or self.water_best == vals[0]:
            return None

        self.claim_id = "pending"
        return ActionProposal(
            Verb.PROPOSE_CLAIM,
            {
                "spec": {self.calib_axis: {"op": "eq", "value": self.water_best}},
                "baseline": {self.calib_axis: {"op": "eq", "value": vals[0]}},
                "direction": "increase",
                "min_delta": 0.3,
                "claim_type": "comparison",
            },
            message=f"{self.calib_axis} {self.water_best} gives more than "
                    f"{vals[0]}",
        )

    def _build_action(self, obs, ctx):
        """Build once the world recognises the capability.

        Deliberately checks nothing itself: the validator owns whether this is
        allowed, so a Builder that tries too early simply gets a hint back.
        """
        if self.built:
            return None
        confirmed = [c for c in obs.my_claims
                     if c["state"] in ("confirmed", "generalized")]
        if not confirmed:
            return None
        here = [t for t in obs.nearby_tiles
                if t["terrain"] == "arable" and not t["planted"]]
        if not here:
            return None
        self.built = True
        return ActionProposal(Verb.BUILD_CHANNEL,
                              {"tile": list(here[0]["tile"])})
