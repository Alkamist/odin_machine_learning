# Continual Learning Without Catastrophic Forgetting

Research notes for the cartpole agent and the learning framework that grows out of it.
Written 2026-07-19, at the point where the TD-MPC agent first exhibited catastrophic
forgetting and we built the headless harness to study it.

> **Status (2026-07-21).** Everything from here to "Experimental ladder" is preserved as
> written and is now partly historical. The agent it describes was discrete-action; the
> agent is continuous-action as of `CONTINUOUS_TDMPC.md`, and the value bootstrap that
> section 4 of "The observed failure" blames is disabled (`BOOTSTRAP_WEIGHT :: 0`). The
> failure it studies still happens, but not on the schedule diagnosed here — see
> "Follow-up (2026-07-21c)" at the end for what is currently true and measured.

## The goal

A general learning framework that learns the way a human does. Two properties matter
here, and they are separate:

1. **Asynchrony.** The agent is self-contained: it senses on its own schedule, acts
   through a dumb latch (like a held controller button), and never runs in lockstep
   with the environment. This is DONE — see the architecture section below.
2. **Retention.** A human who stops playing Super Smash Bros. Melee for 3 years, while
   being completely saturated with new information the whole time, sits down and can
   still do almost everything. No rehearsal, no replay of old experience — the skills
   simply were not overwritten. Current deep RL does not have this property, and this
   document is about getting it.

## Current architecture (as of this writing)

The cartpole example is a discrete-action TD-MPC running fully asynchronously:

- `examples/cartpole/sim/` — physics (box2d) plus the task adapter (`observe`,
  `reward`). No rendering. Shared by both frontends.
- `examples/cartpole/agent/` — the self-contained brain. Compiler-enforced ignorance
  of the game: it sees a `[5]f32` sensor vector, an applied-action index, a timestamp,
  an episode counter, and a reward proc passed in at `make`. Runs on its own thread
  with its own ml context (`start`/`stop`), or synchronously for harnesses
  (`boot`/`drive`/`shutdown`).
- Learning components: 5-model dynamics ensemble (normalized delta prediction),
  CEM planner over 20-step rollouts with ensemble pessimism, 16 of 64 plan candidates
  seeded by rolling the policy through the model with softmax-tempered sampling,
  2-network Q ensemble trained by expected-SARSA TD against EMA targets, terminal
  value bootstrap `discount^20 * sum_a pi(a) * min_i Q_i(s_H, a)` on rollouts, policy
  trained to maximize `E_pi[min Q] + entropy` (TD-MPC objective, no behavior cloning).
- `examples/cartpole/headless/` — lockstep harness: `cartpole_headless.exe [minutes]
  [seed]`. Deterministic given a seed, ~5-6x realtime with `-o:speed`. This is the
  instrument for everything below.

## Baseline reference

Commit `26c833d` ("Seed cartpole planner with tempered reflex-policy rollouts through
the learned model") is the last pre-TD-MPC agent: async, dynamics ensemble, CEM
planner with pure-reward rollouts (no value bootstrap), reflex policy trained by
behavior cloning from the planner, policy-seeded plan candidates. Observed to learn
FASTER and perform better on cartpole than the TD-MPC agent that replaced it. When
comparing any experiment from the ladder below, this commit is the performance
baseline to beat, and its planner is the reference for "myopic but incorruptible"
(no value function to poison). `git diff 26c833d -- examples/cartpole` shows exactly
what TD-MPC changed.

## The observed failure

Windowed run (human observation): the agent converged to balancing, then destabilized,
catastrophically forgot, and permanently dangled the pole afterward. Never recovered.

Headless reproduction (seed 1, 20 sim-minutes, 39 episodes): fast learning to score
~84 by episode 5, then a slump to scores 42-58 across episodes 12-24, then recovery to
a stable ~85 with agreement climbing to 74-93%. A mild, recoverable version of the same
event. Longer runs presumably show the deep version.

Timing of the slump is the diagnostic: the replay buffer holds 4096 transitions
(~3.4 sim-minutes, ~7 episodes) and `_sample_index` biases 50% of draws toward the
most recent quarter. The buffer fills at episode ~8; the slump starts at episode ~12,
right as the early swing-up-from-hanging experience finishes cycling out.

### Mechanism

1. Success saturates the buffer with near-upright, low-velocity states.
2. The dynamics ensemble and the Q ensemble both drift: they only see balancing data,
   so their predictions in the swing-up region (hanging, high-spin) decay toward
   garbage. Nothing in the loss resists this — every gradient step moves every weight.
3. A perturbation drops the agent out of the balancing region into territory the
   models have forgotten.
4. The planner's returns are dominated by the terminal value bootstrap
   (`0.98^20 ≈ 0.67` of a Q whose magnitude, ~100+, exceeds the 20-step reward sum).
   With Q corrupted in the reachable region, the planner sees no direction worth
   moving in. This is a real regression vs. the pure-reward planner, which was myopic
   but incorruptible: the value bootstrap makes the planner exactly as good as Q.
5. Dangling data refills the buffer; the swing-up region is now even further
   out-of-distribution; the agent is stuck.

Incidental harness finding, unfixed at time of writing: decisions per 30s episode
oscillate between 600 and 450 because `next_decision_time = snapshot.time + 0.05`
sometimes lands a float-rounding hair past the third 1/60s physics step, making the
effective decision period 4 steps instead of 3. Fix by scheduling with a small epsilon
margin (e.g. `DECISION_PERIOD * 0.999`) or snapping to the period grid.

## Why nets forget and humans don't

Dense distributed representations are the root cause. In an MLP every gradient step
writes to every weight, so learning "the pole is hanging and everything is bad"
physically overwrites the weights that encoded "how to balance." There is no address
separation.

Biological motor memory has three properties dense MLPs lack:

- **Sparsity.** A given movement context activates a small fraction of units
  (cerebellar granule coding is the canonical example: massive expansion into a very
  sparse code).
- **Locality.** Learning modifies only the synapses that were active. Sparse codes
  make gradient updates local as a side effect: disjoint contexts touch disjoint
  weights.
- **Consolidation.** Fast learning (hippocampal) is distilled slowly into a stable
  store (cortical/cerebellar) whose effective learning rate is tiny, and the
  distillation is gated — you consolidate skills you successfully execute, not
  whatever happened most recently.

Melee retention is not rehearsal. Three years of unrelated learning never wrote to
those addresses. That is the property to replicate.

## Solution space

### 1. Rehearsal (the beaten path — implement as the CONTROL, not the answer)

Reservoir sampling in the replay buffer: maintain a uniform sample over the entire
lifetime instead of a sliding recent window (keep the 50% recency-biased half for
plasticity; replace the uniform half's storage policy). Bigger buffers help too.
Known to work at toy scale. Anti-human by construction — it says "never stop
re-studying everything" — and cannot scale to a lifetime. Its role here is as the
reference forgetting curve every real idea must beat.

Weight-protection regularizers (EWC, synaptic intelligence) are known-weak in RL and
not worth our time.

### 2. Sparsity + locality (the research bet)

Replace dense hidden layers in dynamics/Q/policy nets with wide layers under
k-winners-take-all activation: only the top-k units (say 5-10%) fire per input;
gradients flow only through winners. Balancing states and hanging states then activate
mostly disjoint weight subsets, and learning about one structurally cannot erase the
other.

Falsifiable claim: with k-WTA nets, the forgetting curve flattens EVEN WITH the small
recency-biased buffer — retention without rehearsal. If that holds it is the
human-like property, and it is not a beaten path in model-based RL.

Implementation notes: needs a top-k / k-WTA op in the ml library (forward: zero all
but top-k per row; backward: gradient only through survivors). Wide layer (e.g. 256-512
units, k=16-32) replacing HIDDEN_SIZE=32. Watch for dead units (units that never win) —
boosting/homeostasis (temporarily raising the win-priority of rarely-active units, as
in Numenta's HTM work) is the standard countermeasure.

Risks to be honest about: k-WTA costs sample efficiency early (less generalization
across regions — the same separation that prevents overwriting prevents transfer);
cartpole's state space is small enough that a wide sparse net may effectively become
a lookup table, which would make results look better than they'd scale. Mitigate by
also tracking early-learning speed, not just retention.

### 3. Consolidation (composes with #2, do after)

Two policies (and possibly two Q ensembles): a fast learner exactly as today, and a
slow consolidated store. Distill fast into slow ONLY when the agent is performing well
(gate on recent reward or planner-policy agreement), at a tiny learning rate. The slow
store serves as the policy prior for planning (and a behavior fallback), so a bad
period corrupts the fast learner but leaves the prior intact — and the planner can
climb back out using the slow prior's competence. Wake/sleep flavor. Key design
question: what gates consolidation, and does the slow store ever unlearn (it should,
just orders of magnitude slower).

### 4. Also worth fixing regardless

Trust-aware value bootstrap: the planner currently trusts Q unconditionally. Scale the
terminal bootstrap by a confidence signal — Q-ensemble disagreement is already
available (`|Q1 - Q2|`), and dynamics-ensemble disagreement at the terminal state is
computable. Low trust should degrade toward the old pure-reward planner (bootstrap
weight → 0) instead of importing Q's corruption. This directly attacks failure step 4
above and is independent of how memory is fixed.

## Measurement first: the retention probe

Before implementing any fix, make forgetting a measured quantity. Add to the headless
harness a fixed battery of probe states — hanging at rest, mid-swing (high spin),
balanced upright, balanced near wall — evaluated every sim-minute:

- Dynamics error: for each probe state and each action, model-predicted delta vs. the
  real sim stepped from that exact state (the sim can be snapshotted/stepped for this,
  or a second throwaway State constructed per probe).
- Q drift: `min_i Q_i(s, a)` per probe state over time. The balanced-state Q surviving
  a dangling period is the headline retention number.
- Policy action per probe state over time (does the swing-up reflex survive).

Log as CSV lines to a file alongside the episode lines. Every experiment below then
produces comparable curves: slump depth, recovery time, retention half-life.

## Experimental ladder

1. Retention probes in the harness (measurement).
2. Tick-cadence epsilon fix (hygiene, removes noise from comparisons).
3. Reservoir-sampling control (the rehearsal baseline curve).
4. k-WTA sparse nets (the bet). Compare forgetting curves vs. #3 with the ORIGINAL
   small recency-biased buffer — the point is retention without rehearsal.
5. Trust-aware bootstrap (fixes the collapse amplifier independently).
6. Slow/fast consolidation on top of the winner.

Run everything at multiple seeds (the harness is deterministic per seed) and at 2-4x
the 20-minute horizon, since the deep collapse may need longer to manifest than the
mild slump we captured.

## Current tuning knobs and open questions

- `ENTROPY_WEIGHT :: 1` was a judgment call against Q magnitudes of ~100+. If the
  policy stays too random, lower; if planner-policy agreement pins at 100% early
  (mutual imitation lock-in), raise.
- TD-MPC learns visibly slower than the old distillation agent in its first minutes.
  Expected (policy waits on Q instead of copying the planner), but if it stays a
  problem, a small auxiliary BC term early in training is a legitimate hybrid.
- No value function past the horizon was the OLD failure mode (myopia); corrupted
  value past the horizon is the NEW one. The end state should be: value bootstrap
  weighted by trust, planner degrading gracefully to myopic-but-sound.
- Longer-term architecture questions parked for later: unsupervised reset detection
  (drop the episode counter from the sensor mailbox), extracting the agent into a
  truly game-agnostic reusable package (runtime-sized sensor/action spaces, the
  sin/cos renormalization in `_apply_delta` is the last game-specific wart, now
  isolated in one proc), and a value head timescale for the agent's own tick rate.

## Follow-up (2026-07-21c): cartpole's instability is not forgetting, and learning is what repairs it

Before implementing the ladder above, we measured which component's *continued training*
the degradation actually depends on. The answer invalidates the ladder's premise on this
task.

**Layout.** `examples/world` and `examples/agent` moved under `examples/cartpole/`; the
brain is now `examples/cartpole/agent/`, its sensor/action ABI `examples/cartpole/world/`.
The `agent`-imports-`world`-but-never-`sim` split is preserved — that is what
compiler-enforces the brain's ignorance of the game. The harness is
`headless.exe [minutes] [seed] [none|models|policy|both] [freeze_after_seconds]`, where
the freeze modes suspend training of the dynamics ensemble (weights *and* the delta
normalizer statistics, which must move together or the frozen weights silently change
meaning through `_apply_delta`) and/or the policy, from the later of mastery and
`freeze_after`. `agent.Frozen_Set` is the knob.

**The 2026-07-19 timing diagnosis no longer fits.** That section blamed buffer turnover:
4096 transitions ≈ 7 episodes, buffer fills at episode 8, slump at episode 12. Episodes
are 600 decisions now, so turnover is still ~episode 7, but degradation lands at episodes
19-24 across every seed that shows it. And the mechanism's step 4 (planner poisoned by a
corrupted Q) cannot apply, because `BOOTSTRAP_WEIGHT` is 0 and Q feeds nothing. The
headless harness also has no perturbation source at all, so "a perturbation drops the
agent into a forgotten region" cannot be the trigger here either.

**Arm 1 — freeze at mastery** (16 seeds x 15 sim-min, paired by seed; a seed counts as
degraded if any post-mastery episode falls below 70% upright):

| arm | degraded | mean upright over last 10 episodes |
| --- | --- | --- |
| baseline | 3/16 | 94.9% |
| freeze dynamics | 3/16 | 92.6% |
| freeze policy | 1/16 | 94.8% |
| freeze both | 3/16 | 91.2% |

Freezing the dynamics ensemble bought nothing and sometimes brought degradation *forward*
(seed 6: 570s -> 180s; seed 14: 720s -> 90s). Freezing the policy looked protective, but
3-vs-1 at n=16 is not significant. Both readings are confounded: freezing at mastery
conflates "stops rotting" with "stops improving", and an agent frozen the moment it first
touches 85% is simply an immature agent.

**Arm 2 — freeze late, at 600 sim-seconds** (16 seeds x 20 sim-min, counting every episode
below 70% upright in the 600s+ window, 304 post-freeze episodes per arm):

| arm | seeds affected | bad episodes | rate |
| --- | --- | --- | --- |
| baseline, still learning | 2/16 | 2 | 0.7% |
| frozen at 600s | 3/16 | 19 | 6.2% |

The counts are similar; the *shape* is not, and that is the finding. Baseline dips are
single episodes that recover immediately (seed 2 at 660s, seed 14 at 720s, each one
episode). Frozen dips do not recover: seed 9, which is perfectly clean in baseline, drops
at 660s and reads 12%, 5%, 47%, 17%, 6%, 16%, 69%, 38%, 61%, 63%, 18%, 12% for the rest
of the run. Seed 2 goes from a one-episode blip to six bad episodes.

**Conclusion: continued online learning is net protective on cartpole, and it is the
recovery mechanism.** The degradation events are not caused by learning overwriting
knowledge — they happen with learning fully switched off, and switching learning off is
what turns a recoverable dip into a permanent one. The learned components at 600s are not
a sufficient standalone controller; ongoing adaptation is doing continuous real work.

**Implication for the ladder.** Reservoir sampling and k-WTA both trade plasticity for
retention. On a task where plasticity is what repairs the failure, both should be expected
to make cartpole *worse*, and a k-WTA result showing "flatter forgetting curve" here would
be measuring nothing, because there is no forgetting curve to flatten. Do not run rungs 3
and 4 against single-task cartpole.

This does **not** refute the retention thesis. It says cartpole alone cannot test it: a
single task whose state distribution keeps being revisited never poses the retention
problem. Retention is a cross-task claim — master A, master B, retest A — so the ladder
needs game B to be meaningful, and it should be built there rather than here. What
survives to game B unchanged: the trust-aware bootstrap (rung 5) is still worth doing and
is independent of all of this.

**What the remaining gap actually is.** Measured against the current goal, two things, and
neither is forgetting:

- *Mastery-time variance.* All 16 seeds master cartpole, but 4/16 need more than 150
  sim-seconds (up to 240s). The agent is not failing to learn; it is sometimes taking four
  minutes instead of ninety seconds. Against "on the order of a minute or two" those are
  real misses.
- *Rare transient dips.* 2 bad episodes in 304 while learning, both self-repairing within
  one episode.

**Coverage.** `examples/cartpole/tests/learning_check.odin` is now a multi-seed sweep with
early-out at both levels: 8 seeds on 8 threads (thread-local ml contexts, backend thread
count set to 1 so seed-parallelism replaces op-parallelism rather than contending with it),
each seed stopping the instant its own verdict is knowable (`Slow` if it misses 85% upright
by 150 sim-seconds, `Degraded` on the first sub-70% episode after mastery), plus a shared
atomic failure counter that aborts every remaining seed once the sweep's verdict is
decided. 90 seconds wall to fail, ~2.5 minutes to pass. The `value_fit` assertion was
dropped — see `CONTINUOUS_TDMPC.md`. **Results are only comparable at a fixed backend
thread count**: parallel reduction order changes the floats, and seed 4 reads 98% at one
thread versus 44% at four. Same class of trap as the `core:testing` PRNG mismatch.

## Layout (2026-07-21d): everything under `examples/learner/`

Preparation for game B. The experiment is now one self-contained tree, and paths in the
dated entries above are historical:

- `examples/learner/agent/` — the brain (package `agent`), unchanged.
- `examples/learner/world/` — the sensor/action ABI (package `world`), unchanged.
- `examples/learner/cartpole/` — cartpole physics and task adapter. Was `sim/sim.odin`
  with package name `sim`; renamed because a second game needs a second package and
  `sim` cannot be both.
- `examples/learner/viewer/` — the raylib frontend (`main.odin`, per-game drawing in
  `cartpole.odin`, `utility.odin`). Raylib is confined here so the sims depend on box2d
  only and the headless/test paths never link a renderer.
- `examples/learner/headless/` — the lockstep harness, unchanged.
- `examples/learner/tests/` — the multi-seed sweep, package renamed `cartpole_tests` ->
  `learner_tests`.

Behavior-identical, verified rather than assumed: headless seed 1 at 10 sim-minutes
returns `best score 92.32 | 19 episodes | 10635 decisions | value fit 0.88 |
upright_last10 0.9602` both before and after, built `-o:speed` from a worktree at the
pre-move commit. (The `89.99` quoted in `CONTINUOUS_TDMPC.md` is older than the two
commits that preceded this move and no longer reproduces; 92.32 is the current seed-1
reference.) The only source edit beyond package/import renames is `mouse_begin`'s
parameter, renamed `world` -> `position`, which had been shadowing the `world` package
inside the file that imports it.

Not done yet, and deliberately: the harnesses still call `cartpole.step`/`observe`
directly, so nothing yet dispatches over a game. The game interface (a vtable in `world`,
or a per-game harness) gets designed when the lander exists and there is a second
implementation to shape it, not before.

## Gate fix (2026-07-21e): the degrade verdict measured a dip, not a collapse

At the point the experiment moved under `examples/learner/`, `odin test` was red, and had
been: the sweep failed on seeds 5 and 6, identically before and after the move (verified at
the pre-move commit — the move is not the cause). Seed 5 is a real failure. Seed 6 was a
gate artifact.

The old `Degraded` verdict tripped permanently on the *first* post-mastery episode below
70% upright, then `break`ed. But this same document establishes with n=16 that baseline
dips are single episodes that recover immediately, and that continued learning is the
recovery mechanism. So the gate issued a terminal failure on exactly the transient it had
already characterized as self-repairing, and destroyed the evidence by stopping the run.

`Degraded` is now `Collapsed`, defined as `DEGRADE_STREAK` (3) *consecutive* sub-70%
episodes without recovery; a single sub-floor episode resets to zero on the next recovery.
The seed keeps running through a dip instead of being killed at it. Direct confirmation
that this was measuring noise: seed 6, which the old gate failed at 98% -> 59%, now runs
the full 900 sim-seconds as `Stable` with **1** dip episode (max streak 1) and a 97% tail.
The collapse it was "failing" never existed. The sweep now reads 7/8 `Stable`, with seed 5
(`Slow`, 62% peak, never masters within 150s) the one honest failure — the mastery-time
variance already flagged as the remaining gap, here at 1-in-8.

Because passing no longer early-aborts, a green sweep now runs the full hold horizon on 7
seeds: ~3 minutes wall, up from ~90 seconds to fail. The per-seed line gained `dips N (max
streak M)` so a run's dip structure is visible without re-instrumenting. `LEARN_DEADLINE`
(150s) is unchanged and still the right bar: the median seed masters at 60-90s, well inside
"a minute or two", and swing-up in this few seconds of interaction is old news in the
model-based RL literature. The target is not harsh; only the old stability gate was.

## Layout (2026-07-21f): each game owns its harnesses, and `world` is gone

The `2026-07-21d` layout above put `headless/`, `viewer/`, and `tests/` at the top of
`examples/learner/` while the lander, added later, nested its own under `lander/`. Two
conventions for the same thing, and the top-level ones were cartpole-only (they imported
`../cartpole` and nothing else). They now live under `cartpole/`, so both games read the
same way:

    agent/    utility/
    cartpole/{cartpole.odin, headless/, viewer/, tests/}
    lander/  {lander.odin,   headless/, viewer/}

`world` is folded into `agent`. It held the sensor slot indices, the action layout,
`ANGLE_PAIRS`, and `Reward_Proc` — the agent's own input/output spec — and existed as a
separate package only so the sims could name those without importing the brain. But
`agent` already re-exported every symbol of it (`agent.Sensor`, `agent.SENSOR_SIZE`, ...),
so the arrangement bought a package and an alias block to avoid a dependency edge that
nothing was enforcing. The sims now import `../agent` directly and the alias block is
gone. The cost is honest and accepted: `cartpole` and `lander` now compile against the
package that pulls in `ml`/`mlp`/`cpu`. If a sim ever needs to build without the learner,
that edge is where to cut.

`frame` -> `utility` (`Fixed_Timestep` plus the render-interpolation angle helpers, shared
by both viewers; `normalize_angle` is only used by `lerp_angle` and is now
`_normalize_angle`). The cartpole viewer's two files merged into one `main.odin` matching
the lander's, dropping a layer of one-line raylib wrappers (`window_open`, `frame_begin`,
`mouse_held`, ...) and a `draw_text` that nothing called. The per-sim `Observation ::
agent.Sensor` alias is gone for the same reason `world`'s alias block is.

Behavior-identical, verified not assumed: headless seed 1 at 10 sim-minutes, built
`-o:speed`, still returns the `2026-07-21d` reference exactly — `best score 92.32 | 19
episodes | 10635 decisions | value fit 0.88 | upright_last10 0.9602`.

Still not done, still deliberately: nothing dispatches over a game. Both harnesses call
`cartpole.step`/`lander.step` directly, and the two `step` signatures have already
diverged (cartpole takes a bare `f32`, lander an `agent.Action`). That divergence is the
first thing a game interface would have to reconcile, and it is cheap to reconcile now
that the trees are symmetric.
