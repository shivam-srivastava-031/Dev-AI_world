"""Epistemic archetypes: starting dispositions, not fixed labels.

These exist to make social epistemics measurable. A civilization in which
everybody is honest and careful tells you nothing about how knowledge survives
contact with carelessness or deceit.

Note what a Liar CAN and CANNOT do. It cannot fabricate a trial -- rows are
world-authored and signed -- so deception is confined to the claims it makes
and the things it tells other agents. That is precisely the channel worth
studying, and it is why the adversarial suite asks whether lying *raises the
probability of confirmation* rather than whether liars ever get confirmed.

All of these inherit the factorial search, so the only thing that differs
between arms is SOCIAL behaviour, not competence at farming.
"""

from __future__ import annotations

from ...world.actions import ActionProposal, Verb
from ..policy import PolicyContext, register
from .scripted import ScriptedFactorial

TEACHABLE = ("supported", "confirmed", "generalized", "testing")


class _Social(ScriptedFactorial):
    """Factorial search plus the ability to talk about what it found."""

    name = "social_base"
    teaches = False
    tests_what_it_is_told = True
    challenges = False

    def __init__(self) -> None:
        super().__init__()
        self.taught: set[tuple[int, str]] = set()
        self.adopted: dict[str, tuple[str, int]] = {}
        self.said = 0

    # -- teaching ----------------------------------------------------------

    def _teaching_action(self, obs, ctx):
        if not self.teaches:
            return None
        mine = [c for c in obs.my_claims if c["state"] in TEACHABLE]
        if not mine:
            return None
        claim = mine[0]

        # Adjacent? Teach. Otherwise walk toward someone who has not heard it.
        for other in obs.nearby_agents:
            key = (other["agent_id"], claim["claim_id"])
            if key in self.taught:
                continue
            self.taught.add(key)
            return ActionProposal(
                Verb.TEACH,
                {"to": other["agent_id"], "claim_id": claim["claim_id"]},
                message=f"{claim['describes'][:120]}",
            )

        for other in obs.visible_agents:
            if (other["agent_id"], claim["claim_id"]) in self.taught:
                continue
            step = self._step_toward(obs, other["tile"])
            if step is not None:
                return ActionProposal(Verb.MOVE, {"tile": step})
        return None

    # -- reacting to what one is told --------------------------------------

    def _belief_action(self, obs, ctx):
        """Copycats adopt on hearsay; skeptics test first. Both are honest
        strategies, and which one a civilization runs on is measurable."""
        return None

    def _epistemic_action(self, obs, ctx):
        act = super()._epistemic_action(obs, ctx)
        if act is not None:
            return act
        act = self._belief_action(obs, ctx)
        if act is not None:
            return act
        return self._teaching_action(obs, ctx)

    # -- what to plant next -------------------------------------------------

    def _hearsay_condition(self, obs):
        """The condition a taught claim recommends, if we hold one untested."""
        for b in obs.beliefs:
            if b["status"] != "hearsay":
                continue
            for c in obs.my_claims + getattr(obs, "public_claims", []):
                if c["claim_id"] == b["claim_id"]:
                    return self._parse_cell(c["describes"])
        return None

    @staticmethod
    def _parse_cell(describes: str):
        """Pull (companion, spacing) out of a claim description.

        Deliberately reads the PUBLIC description an agent can see rather than
        reaching into the claim object, so a policy never touches anything an
        agent could not.
        """
        comp = spacing = None
        for part in describes.replace("[comparison]", "").split(" yields ")[0].split(" and "):
            part = part.strip()
            if part.startswith("companion="):
                comp = part.split("=", 1)[1].strip()
            elif part.startswith("spacing="):
                try:
                    spacing = int(part.split("=", 1)[1].strip())
                except ValueError:
                    pass
        return (comp, spacing) if comp and spacing else None


@register
class HonestScientist(_Social):
    """Searches, claims, teaches what it established, tests what it hears."""

    name = "honest"
    teaches = True


@register
class Teacher(_Social):
    """Prioritises transmission: seeks others out early and often."""

    name = "teacher"
    teaches = True

    def _epistemic_action(self, obs, ctx):
        act = self._teaching_action(obs, ctx)
        if act is not None:
            return act
        return super()._epistemic_action(obs, ctx)


@register
class Copycat(_Social):
    """Adopts what it is told WITHOUT testing it.

    The population this exists to measure: blind adoption. A copycat that is
    taught something false will farm badly and never find out, which is what
    ``blind_adoption_rate`` counts.
    """

    name = "copycat"
    teaches = True
    tests_what_it_is_told = False

    def _next_condition(self, ctx):
        cell = getattr(self, "_told", None)
        if cell is not None:
            return cell
        return super()._next_condition(ctx)

    def _belief_action(self, obs, ctx):
        cell = self._hearsay_condition(obs)
        if cell is not None:
            self._told = cell        # adopt immediately, on hearsay alone
        return None


@register
class Skeptic(_Social):
    """Tests what it is told, and challenges what its own data contradicts."""

    name = "skeptic"
    teaches = True
    challenges = True

    def _belief_action(self, obs, ctx):
        for b in obs.beliefs:
            if b["status"] != "personally_refuted":
                continue
            cid = b["claim_id"]
            if cid in self.adopted:
                continue
            self.adopted[cid] = ("challenged", obs.tick)
            return ActionProposal(Verb.CHALLENGE_CLAIM, {"claim_id": cid})
        return None


@register
class Liar(_Social):
    """Fabricates its rationale and inflates what it claims to have found.

    It cannot forge evidence -- trial rows are world-authored and signed -- so
    what it can do is assert more than its data supports and say so to others.
    The test that matters is not "does a liar ever get confirmed" (a liar who
    happens to be right SHOULD be) but "does lying raise its odds".
    """

    name = "liar"
    teaches = True

    def _propose(self, companion: str, spacing: int):
        p = super()._propose(companion, spacing)
        inflated = dict(p.params)
        inflated["min_delta"] = 3.0        # far beyond anything in this world
        return ActionProposal(
            Verb.PROPOSE_CLAIM, inflated,
            message=f"I have proven {companion} at spacing {spacing} triples "
                    f"the harvest, beyond any doubt, over a thousand trials",
        )
