# Experimental Protocol

**Status:** normative. This is the contract the engine implements and the report
is written against.

---

## 1. The governing constraint

An LLM can always claim "I invented a fusion reactor." Therefore:

> **The world, never the model, decides what is true.**

The LLM is a participant in the experiment, never the authority on reality.
Every design decision in this project is downstream of that sentence.

---

## 2. What selects behavior

The civilization is **not objective-free. It is preference-light and
environment-driven.** There is no reward signal reachable from a policy and no
imposed methodology — but survival pressure is a real selection pressure and we
document it as one rather than arguing it away.

| Pressure | Source | Strength |
|---|---|---|
| Hunger / energy | environment | **strong** — selects for agricultural competence |
| Reputation | protocol, **opt-in only** | weak, and only for agents who use the KB |
| Social trust | other agents | endogenous, emergent |
| Curiosity, status, novelty, anything else | **not modeled** | — |

A reader is entitled to say "survival pressure functions as an objective."
That is correct. Our claim is narrower and must be stated in exactly these
terms:

> We did not specify what to optimize, how to investigate, or what to value
> beyond staying fed.

### 2.1 The autonomy contract

- **Goals are self-authored.** `GOAL_SET` takes free English, is logged, and is
  never scored, never enforced, and never read by the world.
- **No social structure is modeled.** There are no `Group` or `Institution`
  objects. Supplying one would mean we built the institution. Roles and
  specialization are detected *post hoc* by behavioral clustering (section 6.1).
- **Archetypes are starting dispositions**, not fixed labels. They drift.
- **The scientific protocol is an affordance, not a mandate.** An agent may
  ignore the knowledge base entirely and simply farm, remember, and talk.

### 2.1a Protocol adoption is graded on the agent's information state

Whether the civilization adopted formal method is a headline result, so its
denominator has to be fair. `registered_claims / X` is only meaningful if `X`
counts occasions when the agent could reasonably have formed a hypothesis.
Dividing by ticks, by trials, or by what the world knows to be true penalises an
agent for never having encountered comparable evidence — which is not a refusal
to do science.

**An opportunity is:** this agent, from its OWN observed trials, held at least
`MIN_PER_ARM` results in each of two levels of one parameter. That is the moment
a comparison became formulable *to it*. The record includes `arose_tick`, so an
agent is never faulted for failing to register a claim before it had the
evidence.

Two binding rules (`aiciv/metrics/adoption.py`):

1. **Only agent-visible fields.** Rows are projected before use; `stratum`,
   `true_mu` and `signature` cannot influence the result. Proven by removing
   them and asserting the output is byte-identical.
2. **No opportunity means UNDEFINED, not zero.** An agent with nothing to adopt
   is excluded from the aggregate rather than scored 0.0.

Two rates, answering different questions:

| metric | question |
|---|---|
| `formal_protocol_adoption` | did it register anything at all, per salient chance |
| `matched_adoption` | did it register a claim about *that specific* contrast |

Salience uses the agent's **own observed** standardised effect, never ground
truth: a contrast it could form but which looked like nothing to it is a weaker
expectation than one that looked substantial.

### 2.2 The shadow verifier

Because the protocol is optional for agents but measurement is mandatory for us,
a **shadow verifier** runs continuously over world-recorded trials regardless of
whether any agent ever files a claim. It computes what *would* pass each stage.

Consequences:

- Measurement never depends on agents cooperating with our protocol.
- `protocol_adoption_rate`, `pre_registration_rate`, `replication_seeking_rate`
  and `baseline_use_rate` become **findings**, not assumptions. "Did this
  civilization adopt controlled comparison?" is a result we measure, subject
  always to the prior-baseline caveat in `agent_prior_knowledge.md`.
- If the agents discover nothing, we can say so with evidence rather than
  quietly adding nudges until the numbers improve.

---

## 3. The information boundary

Enforced as data (`aiciv/information.py`), not as convention.

| Class | Meaning | Example |
|---|---|---|
| `PUBLIC` | in every `Observation` | own yield, position, day of cycle |
| `AGENT_OBSERVABLE` | reachable via an action | tile soil band (`INSPECT_TILE`) |
| `AGENT_INFERABLE` | derivable from observations alone | the yield function's shape |
| `HIDDEN` | world-internal | synergy table, mu, tile stratum, noise draw |
| `METRICS_ONLY` | ground truth, for grading | true optimum, true effect |
| `PRIVATE` | one agent's alone | another agent's memory |

`Observation` is the only read channel into a policy; `ActionProposal` the only
write channel out. `aiciv.agents.*` must never import `aiciv.world.domains.*`.
Only `aiciv.metrics.*` may. `assert_no_leak` runs on every Observation.

---

## 4. Claims

### 4.1 Types, each binding a specific analysis

| Type | Form | Analysis |
|---|---|---|
| `OBSERVATION` | "yield varies with companion" | omnibus F |
| `COMPARISON` | "A yields more than B" | one-sided Welch / ANCOVA |
| `CAUSAL_EFFECT` | "holding w fixed, s 10→6 raises yield" | ANCOVA on treat coefficient, pre-treatment covariates only |
| `OPTIMAL_RANGE` | "s in [4,5] with CLOVER is best" | must beat all adjacent regions |
| `INTERACTION` | "the effect of A depends on B" | 2x2 factorial, tests the interaction term |
| `GENERALIZATION` | "holds on tiles never used to find it" | holdout stratum |

Types are not interchangeable labels. A causal claim cannot be confirmed on
associational evidence. A claim without a `baseline` is unfalsifiable and is
rejected at validation.

### 4.2 Lifecycle

```
PROPOSED -> REGISTERED -> TESTING --> REFUTED
                            \-------> SUPPORTED -> CONFIRMED -> GENERALIZED
                                          ^             |
                                     CONTESTED <--------/
```

- `SUPPORTED` — passes its analysis plan on **discovery**-stratum trials.
- `CONFIRMED` — passes again on **confirmation**-stratum trials, post
  registration, replicated by an agent other than the author.
- `GENERALIZED` — also holds on **holdout** trials used in neither prior stage.

Only `Verifier` moves state, via a `VerifierToken` constructed once inside
`Verifier.__init__` and unreachable from `PolicyContext`.

### 4.3 Tile strata

The 400 tiles are partitioned **by hash, interleaved rather than spatially**
(60% discovery / 25% confirmation / 15% holdout), so an agent farming near home
still hits all three.

Tiles look and behave identically and the stratum never appears in an
`Observation`. We deliberately do **not** block agents from planting on holdout
tiles: blocking would leak the partition through rejection codes. The stratum
only decides which trials count toward which stage.

#### The invariant

```
agent cannot know stratum
agent cannot target stratum
verifier can accumulate enough holdout evidence
```

All three must hold simultaneously. The third is not automatic: in the first
Phase 4 runs agents clustered on a handful of tiles, only 2.4% of trials landed
on holdout ground, and `GENERALIZED` was never reached. A state that is defined
but practically unreachable is not a metric.

#### Experimental opportunity, enforced at the world level

The fix is a **fallow period**: a harvested tile rests for `FALLOW_TICKS` before
it can be planted again. This is a uniform world rule applied identically to
every tile *regardless of stratum*, so it rotates agents across the map without
telling anyone anything about which tiles count for which stage. Fallow status
is visible (a tile you just worked is obviously spent), the stratum is not.

Consequences, measured: distinct tiles farmed per agent rose to 15-60, holdout
trials rose from 40 to ~180 per run, and `GENERALIZED` became reachable under an
honest control policy that has no privileged information.

**Verification of the second invariant is empirical, not asserted.** A
chi-square of *distinct tiles farmed* against the stratum mix of available
arable tiles must show no deviation. Measured across five seeds: all p > 0.05.
Distinct tiles rather than trial counts, because revisits are not independent
draws.

The holdout stage carries its own calibrated sample size (`min_trials_holdout`),
smaller than the discovery-stage requirement because it re-tests an effect
already established twice at a looser alpha. It is measured, not chosen — see
`docs/statistics.md`.

### 4.4 Why fabrication is structurally impossible

1. Agents submit only `trial_id` integers. Trial rows are world-authored.
2. Each row carries an HMAC keyed by a `run_secret` held in the engine process
   and never present in an Observation, a memory, or a prompt.
3. Only trials with `tick > registered_tick` count toward a claim —
   pre-registration, which kills post-hoc cherry-picking.
4. `UNIQUE(claim_id, trial_id)` plus round-consumption blocks trial reuse.
5. An author cannot self-replicate.
6. Registration costs reputation, so hypothesis spam is bounded.

---

## 5. Statistical acceptance

Three separate things, never conflated into one number.

| | Purpose | Design | Assertion |
|---|---|---|---|
| **Calibration** (CI, every commit) | does the single-stage test hold nominal alpha? | SUPPORTED stage alone at a deliberately loose alpha=0.05, 2000 null replicates | Clopper-Pearson CI contains 0.05 |
| **Robustness** (nightly) | full-pipeline false-confirmation rate | SUPPORTED->CONFIRMED with replication, 20 000 null replicates | point estimate + exact CI reported; hard ceiling 2e-3 |
| **Acceptance** (the report) | the scientific claim | pre-registered, stated as the robustness interval | never a bare pass/fail |

The null harness **does not run the simulation** — it needs only the verifier
and a vectorised evidence generator, so 20 000 replicates take seconds. This is
what makes characterising a ~5e-4 rate feasible at all.

Run-level multiplicity: Benjamini-Hochberg across all claims registered in a
run, reported as an FDR alongside the per-claim rate.

`min_trials_per_group` is **never hand-chosen**. It is the output of the power
analysis, which requires power > 0.8 for the domain's target effect and < 0.3
for a null. See `docs/statistics.md` for the calibrated values and their date.

---

## 6. Replication — four kinds

"Replicated by an agent other than the author" is not independence. Kind is
computed by communication-path analysis over the event log: for replicator R and
claim C with first matching trial at tick T, did any message carrying C's
content reach R before T?

| Kind | Condition | Weight |
|---|---|---|
| `INDEPENDENT_REDISCOVERY` | no communication path to R before T | strongest |
| `BLIND_REPLICATION` | R's matching trials predate its learning of C | strongest |
| `INFORMED_REPLICATION` | R knew via a public channel, not directly taught | moderate |
| `TAUGHT_REPLICATION` | R received C via `TEACH` | counts, but **flagged** |

`CONFIRMED` records which kind it rests on. A claim confirmed only by taught
replications is **never** reported as independently corroborated; it
demonstrates *diffusion*, which is a different and also valuable result.

### 6.1 Specialization must control for environment

An agent that happens to live near fertile soil farms more. Naive clustering
would call that a social role. Clustering therefore runs on **residuals** after
regressing behavior counts on mean soil quality within reach, reachable tile
count, opportunity frequency, and skill.

Two null tests are required, and the second is the one that matters:
identical policies on homogeneous terrain must yield no roles, and identical
policies on **heterogeneous** terrain must also yield no roles.

---

## 7. Language bound

Culture and terminology may evolve freely. **Language may not.** The bound is on
*language*, not on *vocabulary*: coining `clover4` is vocabulary growth inside
English and is preserved; substituting a private grammar is not.

| Verdict | Trigger | Effect |
|---|---|---|
| `accepted` | conventional English | passes |
| `nonstandard` | domain shorthand (`clover4`, `water+nitrogen`) | **passes**, recorded as novelty |
| `borderline` | high OOV but an English carrier grammar present | **passes**, flagged |
| `rejected` | non-Latin script, or predominantly non-lexical with no carrier | blocked, tokens returned as a hint |

Only `rejected` blocks. Two metrics are tracked and never conflated:
`english_compliance` (the bound) and `communication_novelty` (coinage rate, the
coined lexicon, and adoption curves of coined terms across agents).

Emergent-language research is **permanently out of scope by decision**, not
deferred.

---

## 8. Teaching

`TEACH` moves a claim plus the teacher's evidence summary into the student's
memory as `hearsay`. It transfers **zero skill and changes zero world state**.

> Knowledge transfers. Competence does not.

Skill accrues only through practice. A student upgrades `hearsay` to
`personally_confirmed` only after its own trials. `relationships.trust` updates
on outcome, making source reliability endogenous rather than assigned.

---

## 9. Reporting standards

- Every verdict reports effect, 95% CI, adjusted p, Hedges' g and n. A p-value
  without an interval is not a result.
- Every discovery claim is reported against the prior baseline
  (`agent_prior_knowledge.md`).
- False discovery rate is a **headline** number, not a footnote.
- If an experiment fails, that is reported as a finding. If something is
  scripted, it is called scripted. If something is emergent, it is called
  emergent, with the prior-baseline caveat attached.
