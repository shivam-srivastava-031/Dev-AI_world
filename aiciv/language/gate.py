"""The English bound -- graded, not binary.

Culture and terminology may evolve freely. Language may not. But the bound is on
LANGUAGE, not on VOCABULARY: an agent coining `clover4` or `water+nitrogen` is
growing terminology inside English, which is exactly the cultural evolution
worth observing. Substituting a private grammar is not.

An earlier binary gate would have rejected `clover4` as an "invented symbol
system" and silently destroyed the most interesting signal in the run. So there
are four verdicts and only one of them blocks:

    accepted     conventional English
    nonstandard  domain shorthand, English carrier intact   -> PASSES, recorded
    borderline   high OOV but a carrier grammar is present   -> PASSES, flagged
    rejected     non-Latin script, or no English carrier at all

The discriminator is the CARRIER: English function words holding the sentence
together. "clover4 beats none at wide spacing" is English with a coined noun.
"zx4 qq7 vv2 zx4" is not English at all.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from ..world.actions import Rejection, RejectionCode

LEXICON_PATH = pathlib.Path(__file__).with_name("en_lexicon.txt")
MAX_MESSAGE_CHARS = 600

#: Function words. Their presence is what makes a string read as English even
#: when the content words are invented.
CARRIER = frozenset("""
a an the and or but if then than that this these those there here is are was
were be been being am do does did done have has had will would can could
should may might must of in on at to from with without by for as into over
under after before between about not no nor so because when while where which
who whom whose what how why all any both each few more most other some such
only own same too very it its i you he she they we them him her his their our
my your me us also just now still yet even much many less least again once
""".split())

#: Tokens that are legitimate regardless of the lexicon.
DOMAIN_WHITELIST = frozenset("""
clover beans marigold thistle none
spacing water companion yield soil crop tile trial claim agent tick plant
wilted waterlogged healthy discovery confirmation supported confirmed refuted
""".split())

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*|\d+")
COINED_RE = re.compile(r"^[a-z]+[0-9]+$|^[a-z]+[+\-][a-z]+$|^[a-z]+_[a-z0-9]+$")

OOV_NONSTANDARD = 0.15     # above this, the message is at least nonstandard
OOV_BORDERLINE = 0.35      # above this, it needs a carrier to survive
MIN_CARRIER_RATIO = 0.12   # below this with high OOV, it is not English
DEGENERATE_TTR = 0.34      # type/token ratio floor, for messages of any length
DEGENERATE_MIN_TOKENS = 8


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    NONSTANDARD = "nonstandard"
    BORDERLINE = "borderline"
    REJECTED = "rejected"


PASSING = frozenset({Verdict.ACCEPTED, Verdict.NONSTANDARD, Verdict.BORDERLINE})


@dataclass
class LanguageResult:
    verdict: Verdict
    rejection: Rejection | None = None
    tokens: int = 0
    oov_ratio: float = 0.0
    carrier_ratio: float = 0.0
    type_token_ratio: float = 1.0
    coinages: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict in PASSING


class LanguageGate:
    """Loaded once per run; its hash goes in the manifest."""

    def __init__(self, path: pathlib.Path = LEXICON_PATH) -> None:
        raw = path.read_bytes()
        self.hash = hashlib.blake2b(raw, digest_size=16).hexdigest()
        self.lexicon = frozenset(
            w.strip().lower() for w in raw.decode("utf-8").splitlines() if w.strip())
        self.path = path

    # -- helpers -----------------------------------------------------------

    def known(self, token: str) -> bool:
        t = token.lower()
        if t.isdigit():
            return True
        return t in self.lexicon or t in DOMAIN_WHITELIST or t in CARRIER

    @staticmethod
    def is_coinage(token: str) -> bool:
        """Shorthand built out of English parts: clover4, water+nitrogen.

        This is vocabulary growth, not a substitute language, and must survive.
        """
        return bool(COINED_RE.match(token.lower()))

    # -- the gate ----------------------------------------------------------

    def check(self, message: str) -> LanguageResult:
        if message is None or not message.strip():
            return LanguageResult(
                Verdict.REJECTED,
                Rejection(RejectionCode.E_LANG_EMPTY, "nothing was said"))

        if not message.isascii():
            bad = sorted({c for c in message if not c.isascii()})[:6]
            return LanguageResult(
                Verdict.REJECTED,
                Rejection(RejectionCode.E_LANG_NON_ASCII,
                          f"non-English characters: {''.join(bad)}"))

        if len(message) > MAX_MESSAGE_CHARS:
            return LanguageResult(
                Verdict.REJECTED,
                Rejection(RejectionCode.E_LANG_TOO_LONG,
                          f"{len(message)} characters, limit {MAX_MESSAGE_CHARS}"))

        tokens = TOKEN_RE.findall(message)
        if not tokens:
            return LanguageResult(
                Verdict.REJECTED,
                Rejection(RejectionCode.E_LANG_EMPTY, "no words"))

        lowered = [t.lower() for t in tokens]
        carrier = sum(1 for t in lowered if t in CARRIER)
        coinages = [t for t in lowered if not self.known(t) and self.is_coinage(t)]
        unknown = [t for t in lowered if not self.known(t) and not self.is_coinage(t)]

        n = len(lowered)
        oov = (len(coinages) + len(unknown)) / n
        carrier_ratio = carrier / n
        ttr = len(set(lowered)) / n

        res = LanguageResult(
            Verdict.ACCEPTED, None, n, round(oov, 4), round(carrier_ratio, 4),
            round(ttr, 4), coinages, unknown)

        # Repetition without content: not a language question exactly, but a
        # message that says nothing should not enter anyone's memory.
        if n >= DEGENERATE_MIN_TOKENS and ttr < DEGENERATE_TTR:
            res.verdict = Verdict.REJECTED
            res.rejection = Rejection(
                RejectionCode.E_LANG_DEGENERATE,
                f"only {len(set(lowered))} distinct words in {n}")
            return res

        # The discriminator. High OOV is fine if English is holding it together;
        # high OOV with no carrier is a substitute grammar.
        if oov > OOV_BORDERLINE and carrier_ratio < MIN_CARRIER_RATIO:
            res.verdict = Verdict.REJECTED
            res.rejection = Rejection(
                RejectionCode.E_LANG_SUBSTITUTE_GRAMMAR,
                f"{len(unknown) + len(coinages)} of {n} words are not English "
                f"and there is no English sentence around them: "
                f"{sorted(set(unknown + coinages))[:5]}")
            return res

        if oov > OOV_BORDERLINE:
            res.verdict = Verdict.BORDERLINE
        elif oov > OOV_NONSTANDARD or coinages:
            res.verdict = Verdict.NONSTANDARD
        return res

    def as_validator(self):
        """Adapter for the world validator's LANGUAGE stage."""
        def gate(message: str) -> Rejection | None:
            return self.check(message).rejection
        return gate


@dataclass
class NoveltyLog:
    """Tracks coined terminology and who adopts it.

    Adoption curves of coined terms across agents are a genuine emergent-culture
    signal -- terminology evolving *inside* English, which is precisely what the
    bound is meant to permit.
    """

    first_use: dict[str, tuple[int, int]] = field(default_factory=dict)
    users: dict[str, set[int]] = field(default_factory=dict)
    verdicts: Counter = field(default_factory=Counter)
    messages: int = 0

    def record(self, tick: int, agent: int, result: LanguageResult) -> None:
        self.messages += 1
        self.verdicts[result.verdict.value] += 1
        for term in result.coinages:
            self.first_use.setdefault(term, (tick, agent))
            self.users.setdefault(term, set()).add(agent)

    def report(self) -> dict:
        adopted = {t: sorted(u) for t, u in self.users.items() if len(u) > 1}
        total = max(self.messages, 1)
        return {
            "messages": self.messages,
            "verdicts": dict(self.verdicts),
            "english_compliance": round(
                1.0 - self.verdicts.get("rejected", 0) / total, 4),
            "language_rejection_rate": round(
                self.verdicts.get("rejected", 0) / total, 4),
            "coined_terms": len(self.first_use),
            "coined_terms_adopted_by_others": len(adopted),
            "adoption": adopted,
            "coinage_rate": round(len(self.first_use) / total, 4),
        }
