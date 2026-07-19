# Continual Learning Without Catastrophic Forgetting

Research notes for the cartpole agent and the learning framework that grows out of it.
Written 2026-07-19, at the point where the TD-MPC agent first exhibited catastrophic
forgetting and we built the headless harness to study it.

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
