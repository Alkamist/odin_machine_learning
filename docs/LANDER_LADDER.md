# The Lander Ladder: Rungs 1 and 3, and the Determinism Fix

Session log, 2026-07-21. Executes the sequencing proposed at the end of
`LEARNING_SIGNAL.md`. Everything marked MEASURED is data from this session; everything
else is reasoning about it.

Headline: **the value bootstrap (rung 1) failed in every reward regime tested**, a
methodology defect invalidated the harness before any of it could be measured, and the one
positive result is potential-based shaping, which trades land rate for a halved time-to-land
against an honest objective.

## 0. The harness was measuring the CPU scheduler (fixed)

This came first and everything else depends on it.

MEASURED: the same binary at the same seed, run twice alone in sequence, produced
completely different trajectories, diverging from episode 1. Quantified over 6 sim-minutes:

| | 12 repeats of seed 1 | 12 different seeds |
| --- | --- | --- |
| land rate range | 35.7% - 70.6% | 13.3% - 71.4% |
| land rate sd | **12.6 pts** | 17.1 pts |

Within-seed noise was ~74% of between-seed spread, so **the seed controlled almost
nothing** and paired-by-seed analysis — the methodology every result in
`CONTINUAL_LEARNING.md` rests on — bought nearly nothing.

Cause: the worker is a free-running thread and `catch_up`'s tolerance was hardcoded to
`decision_period`. Which mailbox snapshot the worker latched, and how many train/refine
steps it completed per sim-second, both depended on wall-clock. This is the
cadence-oscillation bug at `CONTINUAL_LEARNING.md:92-96` ("incidental, unfixed") compounded
by thread timing; it is not incidental.

The prior docs' results were sound *at the time* — they used a lockstep
`boot`/`drive`/`shutdown` harness. The commit "Rewrite the agent as a runtime-sized
asynchronous package" removed that path, silently invalidating the methodology.

**Fix: `Agent.pacing_tolerance`.** `PACING_ASYNC :: DECISION_PERIOD` reproduces the old
behavior; `PACING_PINNED :: f64(0)` makes the main thread wait for the worker to drain its
credit on the exact snapshot. Deliberately *not* a second harness — the same thread and the
same `_step` run either way, only the pacing contract tightens. A separate lockstep harness
is what made the old one structurally blind to the async-only wall-stuck failure.

MEASURED: 9 runs of one seed (8 in parallel plus 1 alone) spanning 3.6x - 5.4x realtime
produced **bit-identical** output.

Use pinned for comparisons, sweep the tolerance for latency robustness, run loose for
concurrency validation.

### Consequence: every prior sweep number is suspect

MEASURED, same code and seeds, 6 sim-min, 12 seeds:

| | land rate | timeouts |
| --- | --- | --- |
| async (12-way parallel) | 38.7% | 37.6% |
| pinned | **64.1%** | **5%** |

Under 12-way load the agent ran at 3.3-3.9x realtime versus 5.3-7.3x alone, so the worker
silently dropped roughly half its training steps per sim-second. **Sweep width was a hidden
independent variable**, single runs were never comparable to sweeps, and the PESSIMISM
tables in `CONTINUOUS_TDMPC.md` are among the results affected.

### `catch_up` was also burning a second core

It spun on `thread.yield()`, which on Windows returns immediately when nothing else is
runnable. Under pinned pacing the main thread waits ~95% of every step, so each run used two
cores — one computing, one spinning. Now bounded spin then sleep (`CATCH_UP_SPINS` 64,
`CATCH_UP_SLEEP` 50us). Determinism verified preserved against a pre-fix 6-minute run.

## 1. The agent had no concept of an episode ending (fixed)

Three defects, all invisible while nothing read Q:

- `reward()` reported `dead=true` only for failures. A successful landing ends the episode
  but reported `dead=false`, so `_train_value` bootstrapped `Q(landed_state)` — and landed
  states never appear as training inputs (`_remember` stores `a.previous`;
  `_forget_episode` clears `has_previous` at the boundary). Q there was unconstrained
  extrapolation.
- `_rollout` kept a landed particle `alive`, re-collecting `LANDING_BONUS` every remaining
  horizon step — up to +800 for a state the model may have hallucinated.
- Terminal transitions were recorded only when the terminal sensor happened to fall on a
  decision boundary, roughly one time in three.

**Fix:** `Score_Proc` returns `(reward, done, failed)`; `buffer_terminal`;
`end_episode(a, sensor)` feeding `_close_episode`; no-bootstrap targets in `_train_value`
and `_observed_return`; `_rollout` splits "stop the particle" from "apply the penalty".
`DEATH_PENALTY` in `_rollout` is now discounted, which it was not — it disagreed with
`_observed_return`.

Two subtleties worth not re-deriving:

- **`terminal` comes from the score proc's `done`, never from the harness ending the
  episode.** A lander `Timeout` is time-limit truncation and must still bootstrap. Getting
  this backwards silently breaks truncation.
- If the terminal sensor *does* land on a decision boundary, `_decide` records it and then
  `_close_episode` would record a second, spurious zero-delta `terminal -> terminal`
  transition. `_decide` now forgets the episode immediately after recording a terminal.

MEASURED effect (12 seeds, 6 sim-min): control 64.1% -> **67.3%** with sd 12.9 -> 10.1;
bootstrap 33.2% -> 42.2%. Real, worth having, did not change any verdict below.

## 2. Rung 1: the trust-gated value bootstrap failed

Implemented as a terminal bootstrap in `_rollout`, weighted by `discount * trust`, with
trust derived from the `_value_fit` correlation. MEASURED, 12 seeds, 6 sim-min, paired:

| | control | bootstrap |
| --- | --- | --- |
| land rate (per-seed mean) | **67.3%** | 42.2% |
| land rate (pooled) | 165/246 | 284/656 |
| between-seed sd | 10.1 | 25.2 |
| time-to-land | 17.3s | **7.2s** |
| crash rate | **13%** | 48.6% |

Paired difference 25.1 points in favor of control, t=3.45, 10 of 12 seeds.

**But the dawdling fix worked**: time-to-land halved and timeouts vanished. And seed 3 read
**86.7% at 6.1s** under bootstrap versus 63.6% at 17.1s under control — when it avoids the
failure below, the bootstrapped agent is far better at the actual task.

### Why it failed: the reward made suicide rational

The reward is negative everywhere, so `Q(flying) ~ 50 * r`. MEASURED: at spawn altitude
`r ~ -2.4`, giving `Q ~ -120`, against `DEATH_PENALTY = -40`. **Crashing was worth +80 over
continuing to fly.** MEASURED: median crash at 3.7s, at x ~ +208 — flying into the ground
without approaching the pad.

The myopic planner masked it. A 20-step sum is `-2.4 * 16.6 ~ -40`, almost exactly balanced
against the death penalty, which is also why the control still crashes 13%. Widening the
horizon exposed a reward bug that a short horizon had been hiding.

MEASURED `DEATH_PENALTY` sweep (12 seeds, bootstrap on) — the treadmill in both directions:

| penalty | land rate | timeouts | crashes |
| --- | --- | --- | --- |
| 40 | 33.2% | 22 | 309 |
| 150 | 7.4% | 96 | 106 |
| 400 | 3.1% | 115 | 96 |

Small penalty gives suicide, large penalty gives refusal to touch down. **No value escapes.**

## 3. Rung 3: the reward ladder

MEASURED, 6 seeds x 3 sim-min per config (goal arms extended to 10 sim-min):

| reward | arm | land rate | time-to-land | value_fit | dominant failure |
| --- | --- | --- | --- | --- | --- |
| dense | myopic | **64.1%** | 17.4s | 0.95 | dawdles |
| dense | bootstrap | 32.9% | 6.5s | 0.92 | suicide basin |
| no-POS | both | 0% | - | 0.75 | **hovering** (30/37 timeouts) |
| goal, g=.98 | both | 0% | - | 0.99 | blind planner, instant crashes |
| goal, g=.995 | myopic | ~0% | - | 0.97 | 3 landings in 936 episodes |
| goal, g=.995 | bootstrap | 3.1% | 3.7s | 0.66 | exploration wall |
| potential(pos) | both | 0% | - | 0.99 | dives (385/391 crashes) |
| potential(full) | myopic | 50.9% | **10.4s** | 0.53 | - |
| potential(full) | bootstrap | 16.8% | 11.9s | 0.69 | - |
| potential(full), g=.995 | myopic | 47.5% | 9.5s | 0.34 | - |

Four findings:

**`POS_WEIGHT` was carrying the entire task.** Strip it and hovering becomes *optimal* —
VEL/TILT/SPIN all reward being still and upright, and the bonus is only reachable through
crash risk. The dense reward decomposes as POS = "go to the pad" (the task, and the descent
solution), VEL/TILT/SPIN = "be gentle" (landing quality, already implied by the landing
predicate), bonus/death = the goal. **No subset of the current terms specifies the task
without also encoding the solution.**

**Goal-only hits the exploration wall**, exactly as `LEARNING_SIGNAL.md` predicted verbatim.
At 10 sim-min with g=0.995 and the bootstrap, 5 of 6 seeds land 0-2 times. Seed 1 finds
landings at episode 24 and reaches 14 total, but last-10 is 2/10 — it exploits a discovery
without converging.

**Potential-based shaping** (Ng, Harada & Russell 1999) is implemented as a game-supplied
`Potential_Proc` that the *agent* applies as `g*P(s') - P(s)`, so the policy-invariant
telescoping form is structurally guaranteed and a game cannot write non-invariant shaping.
`P(terminal)` is forced to 0, which is required for invariance in episodic tasks.

**Choosing P matters more than the theory suggests.** With P = position only, land rate was
**0%**. Telescoping collapses the planner's horizon return to `g^H*P(s_H) - P(s_0)`, and
`P(s_0)` is identical across candidates, so the planner ranks purely by the potential at the
end of the horizon. Position-only P says "get low fast" with nothing penalizing speed, so it
dives. Reusing the **entire** dense term as the potential gives 50.9% at 10.4s — half the
dense baseline's time-to-land, against a goal-only objective that does not encode the
solution. By `LEARNING_SIGNAL.md`'s own standard that is better progress at a lower number.

## 4. The trust signal was the wrong choice (my error)

I argued in-session that `_value_fit` correlation was a better trust signal than the
ensemble disagreement `CONTINUAL_LEARNING.md` proposes, on the grounds that "self-consistency
is exactly the license to bootstrap." **The data refutes this.** MEASURED: under dense
reward Q sits at fit 0.92-0.95 while the bootstrap is actively harmful.

Pearson correlation is invariant to scale and offset, so a Q that ranks states perfectly but
is mis-scaled by 3x passes the gate and still wrecks the planner — which is precisely the
2026-07-20 failure in `CONTINUOUS_TDMPC.md` (magnitude ~100+ swamping the horizon sum). I
chose a metric structurally blind to the known failure mode.

**The gate should measure calibration — RMSE against observed returns — not correlation.**
`_value_fit` already computes both series (`predictions` and `returns[:samples]`); the
change is a few lines. This is the cheapest untested hypothesis and the obvious next thing.

Note also that `value_fit` **collapses under potential shaping** (0.34-0.69), because
shaping makes the return nearly constant at `-P(s_0)` and removes the spread the correlation
needs. Whatever replaces it must behave sensibly in that regime too.

## 5. State of the tree

Committed as `f521305`: pacing tolerance, episode termination, trust-gated bootstrap.

Uncommitted (91 insertions, 20 deletions across 5 files):
- `agent/agent.odin` — `Potential_Proc`, `create(potential=)`, `_potential`,
  `_shaped_reward`, the `catch_up` spin fix, `#config` knobs for `PLAN_DISCOUNT` and
  `DEATH_PENALTY`.
- `agent/planner.odin`, `agent/worker.odin` — apply the shaping at the three places reward
  is computed.
- `lander/lander.odin` — `REWARD_MODE` ladder (`DENSE`/`NO_POS`/`GOAL`/`POTENTIAL_POS`/
  `POTENTIAL`) and `potential`.
- `lander/headless/main.odin` — wires the potential, reports `value_trust`.

Cruft to remove once the experiments settle: the `#config` on `DEATH_PENALTY` (its sweep is
done), and probably the `REWARD_POTENTIAL_POS` mode, which exists only to document the
0% result above.

**`cartpole/` is untouched and still runs async and unpinned**, so its test sweep still
carries the noise that made lander's numbers meaningless. Switching it to `PACING_PINNED` is
worth doing but will move its numbers, so it needs a deliberate re-baseline rather than
being folded into another change.

## 6. Method notes

I burned a 10-minute machine-saturating run on a 72-config grid with no early-out, which
violates the testing constraints in `GENERAL_LEARNER.md`. What worked instead: a staged
screen at 6 seeds x 3 sim-min with concurrency capped around 10, extending only the arms
whose verdict was ambiguous. That found the same answers in ~2 minutes per batch.

Also: read the *shape* of a screen before believing its summary. The goal-arm rows read 0%
at 3 sim-min, but the landings in `goalg_boot` were clustered at episodes 24, 25, 27 of 28 —
a learning curve starting, not noise. The 10-minute follow-up was what showed it still never
converges.

## 7. Where to pick up

1. **Re-gate the bootstrap on calibration rather than correlation** (§4). Cheap, falsifiable,
   and directly tests the one thing known to be wrong. If it fails, rung 1 is genuinely dead
   on lander and that should be recorded as a negative result rather than retried.
2. **If it fails, bank `REWARD_POTENTIAL` + the myopic planner** as lander's configuration:
   50.9% at 10.4s, honest objective, no suicide basin, dawdling halved. Then delete the
   bootstrap or leave it gated at zero with this document explaining why.
3. **Either way, exploration is the next real problem** — it is what rung 3 isolated, and
   `LEARNING_SIGNAL.md` sections on step 4/5 of the wavedash spec and on temporal abstraction
   are the design notes for it. Do not attempt it before the harness questions above are
   settled, since it cannot be measured on a noisy one.
