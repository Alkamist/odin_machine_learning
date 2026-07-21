# Continuous TD-MPC and the Analog Action Interface

Plan of record for pivoting the cartpole agent from discrete single-latch control to
the canonical continuous TD-MPC control formulation over a factored, human-comfortable
action space. Written 2026-07-20, before any code, as the agreed scope for the work that
precedes the continual-learning experiments in `CONTINUAL_LEARNING.md`.

## Why

Two goals, one immediate and one long-range, both point the same way:

1. **The retention experiments need a standard substrate.** The pre-TD-MPC agent solved
   cartpole, which means cartpole was too easy to be the whole story. The plan
   (`CONTINUAL_LEARNING.md`) is now a cross-task forgetting test: master game A, master a
   genuinely hard game B, return to A and measure what survived. If A and B run on a
   nonstandard discrete agent, every result is ambiguous — is it about retention, or
   about our odd action space? Getting onto the published continuous formulation first
   removes that confound.

2. **The agent should use the interface a human uses.** A person plays a huge variety of
   games with the same mouse and keyboard. The action space is not "one of N
   mutually-exclusive buttons" — it is a set of *independent channels* (each key its own
   on/off; the mouse a continuous 2-D axis) that combine freely. We want the agent to
   learn *that* interface, per game, choosing the most human-comfortable control for each
   game and having the bot learn the same one.

The current agent's discrete single action-per-tick latch is the simplification we made
for cartpole. It is the wrong foundation for both goals, and we fix it before building
game B.

## Decision and scope

"Real TD-MPC" has two layers. We adopt the first now and defer the second.

**Adopt now — the continuous-control formulation (Level 1):**
- Continuous, factored action space (below).
- A `Q(s, a)` critic that takes the action as an input; expectations over the policy are
  *sampled*, not enumerated.
- A CEM/MPPI planner refining a Gaussian over continuous action sequences, seeded by a
  policy prior (as today).
- Keep the observation-space dynamics ensemble (interpretable, and its disagreement gives
  us both planning pessimism and the trust signal the roadmap wants).
- Keep the hand-given true reward proc.

**Defer — the latent world model (Level 2):**
Full TD-MPC2 also replaces raw-observation prediction with a learned encoder + latent
dynamics + *learned* reward + latent Q, plus SimNorm and two-hot value regression. We
hold this off, and not for effort reasons: the retention thesis is a claim about *a
representation decaying*, and a learned encoder adds another dense representation that
forgets, muddying the clean "the swing-up model rotted" analysis. We also already have
the true reward, so learning it buys nothing here. The latent model earns its complexity
when we move to pixels / arbitrary games we cannot hand-instrument — a real future, but
not cartpole or lander. The TD-MPC2 numerical tricks (two-hot value, SimNorm) are scaling
aids for high-dim multi-task settings; we add them only if a task demands them, not
speculatively.

## The factored action space

An action is a vector of *independent channels*, not a categorical over their product.
Each channel is one of:

- **Binary** — a key: on/off, learned as an independent Bernoulli (sigmoid head).
- **Analog** — an axis (e.g. mouse-x): continuous in a bounded range, learned as a
  Gaussian (tanh-squashed mean, per-channel std).

The human and the agent share this interface exactly, as they already share the discrete
latch today: the frontend maps held keys to binary channels and mouse motion to analog
channels; the agent emits the same vector. This kills the permutation explosion — "thrust
+ left" is two channels on, never a distinct enumerated action — and generalizes to any
game's control scheme.

Where a game is comfortably analog, prefer analog. Cartpole becomes a single analog axis
(mouse-x → cart command). Lunar lander becomes analog throttle + analog rotation. The rule
is: find the most human-comfortable interface for the game, then have the bot learn it.

## What changes in the agent, and what does not

Generalizes cleanly:
- **Dynamics encoding** (`_encode`, `MODEL_INPUT` in `agent.odin`): action becomes
  `[binary flags…, analog axes…]` concatenated, instead of a one-hot. Model still predicts
  observation-space sensor deltas.
- **Policy head**: per-binary sigmoid (Bernoulli) and per-analog mean (Gaussian); entropy
  is the sum over channels. Replaces the softmax-over-`ACTION_COUNT` head.
- **Human input**: becomes simpler — keys map straight to their channels, mouse to its
  axis. The A/D press/release combo-collapsing in `human_action` (`cartpole.odin:55`) goes
  away.
- **CEM planner**: per-channel elite statistics — Gaussian mean/std for analog, Bernoulli
  frequency for binary — replacing the discrete elite-count scheme. This is closer to the
  canonical TD-MPC planner than what we hand-rolled.

Stays / deferred:
- Observation-space dynamics ensemble (5 models), true reward proc, EMA target critics,
  the async architecture, the buffer.
- The `_apply_delta` sin/cos renormalization wart (`agent.odin:661-671`) is cartpole's
  observation-manifold prior. It survives for cartpole but must become a static property
  of the shared sensor layout (passed at `make`) once game B shares the space — tracked in
  `CONTINUAL_LEARNING.md`, not solved here.

## The critic rewrite (the load-bearing change)

Today `Q(s) → [ACTION_COUNT]` and every value computation enumerates discrete actions.
None of these survive a factored/continuous space; each becomes a `Q(s, a)` call with the
policy expectation estimated by sampling K actions and averaging:

- Expected-SARSA target `Σ_a π(a)·min_i Q_i` (`_train_value`, ~`agent.odin:392-401`) →
  sample `a' ~ π(·|s')`, target `r + γ·min_i Q_i(s', a')`.
- Policy objective `E_π[min Q]` as `Σ_a π(a)·(−Q)` (`_train_policy`, ~`agent.odin:467-480`)
  → maximize `min_i Q_i(s, a)` for `a ~ π(·|s)` (reparameterized) plus entropy.
- Terminal bootstrap `Σ_a π(a)·Q` on rollouts (`_rollout`, ~`agent.odin:759-773`) →
  sample `a ~ π` at the terminal state, bootstrap with `min_i Q_i(s_H, a)`.

This is the standard actor-critic / SAC / TD-MPC move. It is the main work and the main
risk of the pivot; the rest is mechanical. Low-dim tasks keep the MSE critic — no two-hot
regression unless a task needs it.

## Build sequence

1. **Analog cartpole, human-playable (this first).** Refactor cartpole's action to a
   single analog axis, drive it from the mouse in the windowed frontend, and confirm by
   hand that it is *possible and feels good* before committing to the agent rewrite. The
   agent path gets a temporary discrete→analog bridge (map its 3 actions to −1/0/+1) so the
   build keeps running; the real continuous agent comes next. Perturbation (currently the
   mouse) relocates to a click/key or becomes automated.
2. **Continuous TD-MPC agent.** Implement the factored action space, `Q(s, a)` critic,
   Gaussian policy head, and CEM-Gaussian planner. Validation gate: continuous
   mouse-cartpole learns at least as well as the discrete key-cartpole baseline — same
   task, so a clean apples-to-apples check that the formulation is sound.
3. **Lunar lander (game B)** with its own comfortable analog interface (throttle +
   rotation), built against the now-continuous agent.
4. **Continual-learning experiments.** One shared agent, one shared sensor + action space
   spanning both games, no game-ID (context inferred from observation), per
   `CONTINUAL_LEARNING.md`. This is where the retention ladder (probes → reservoir control
   → k-WTA → trust-aware bootstrap → consolidation) runs.

## Validation bar

`odin check` and `odin test tests` (plus `-define:ML_CPU_POISON=true` and
`-microarch:x86-64-v3`) on every touched package. New ops go through `tests/cases/` so
gradient check and parity are automatic. The load-bearing gate for the pivot is behavioral:
continuous mouse-cartpole must match the discrete baseline on the same task before we move
to lunar lander.

## Open decisions

- **Mouse-cartpole control mapping.** Target-velocity (mouse-x offset → cart velocity,
  the honest analog of the current force-toward-target-speed physics) vs. target-position
  servo (cart seeks the cursor under a force-limited P controller — more intuitive, risks
  trivializing if the servo is too stiff). Decide by feel in task 1; keep it one constant.
- **Gaussian policy std.** Fixed schedule vs. learned per-channel std. Start simple
  (fixed/annealed), revisit if exploration stalls.
- **K, the action-sample count** for the critic expectations. Start at 1–2 (SAC uses 1);
  raise if the value estimates are too noisy.
- **Perturbation input** for cartpole once the mouse is the control (click-to-shove, a
  key, or automated random shoves — the last is best for repeatable retention probes).

## Outcome (2026-07-20): validation gate met, value function taken off the critical path

The continuous rewrite (Level 1) landed and worked, but its *first* form learned
markedly slower than the discrete pre-TD-MPC baseline. Diagnosis: the pivot changed more
than the action space — it also put the learned value function on the critical path of
both downstream consumers, and Q is the slowest thing in the system to learn:

- The CEM planner's return was dominated by the terminal Q bootstrap
  (`0.98^20 ≈ 0.67` of a Q whose magnitude ~100+ exceeds the horizon reward sum), so the
  planner was only as good as a from-scratch Q.
- The policy was trained by the deterministic policy gradient through Q
  (`maximize min_i Q_i`), which points nowhere while Q is garbage.

The fix restores the discrete baseline's learning recipe *within* the continuous
formulation — the reward is analytic and the 5-D dynamics model learns in a few hundred
frames, so neither consumer needs Q to be fast:

- **Pure-reward planner.** The terminal bootstrap in `_rollout` is gated behind
  `BOOTSTRAP_WEIGHT :: f32(0)` (compile-time `when`, zero cost when off). The planner
  scores rollouts on the true reward through the dynamics ensemble — myopic but
  incorruptible.
- **Behavior-cloning policy.** `_train_policy` regresses the policy toward the planner's
  own actions stored in the buffer (`tanh(mean) → analog action`,
  `sigmoid(logit) → binary bit`), trained only on planner-authored transitions (new
  `Transition.planned` flag, tracked via `Agent.previous_planned`, so random warmup
  actions are excluded). This is a supervised target available from the first
  post-warmup decision.
- **Q kept as background substrate.** `_train_value` still runs; Q simply feeds nothing
  yet. It is ready for the trust-aware bootstrap (§4 of `CONTINUAL_LEARNING.md`) and the
  retention experiments, which are the reason it exists at all on this task.

Result (headless, `-o:speed`, seeds 1-4, 3 sim-min): episode 1 (~525 decisions) already
scores 44-69, versus the old TD-MPC needing ~5 episodes to reach ~84. Fast learning
restored; the "continuous cartpole learns at least as well as the discrete baseline"
gate is met.

Coverage: `examples/cartpole/tests/` (package `cartpole_tests`, kept out of the main
`tests` suite so box2d stays unlinked there) runs the real sim+agent headless for
90 sim-seconds at seeds 1-2 and asserts best score ≥ 45 (~30s wall). Run with
`odin test examples/cartpole/tests -o:speed`.

Standing implication for the roadmap: on cartpole the value function does no work. It is
insurance for harder games (where the horizon can't span the reward) and the load-bearing
object the retention thesis is about. The clean next step when Q should re-enter planning
is the trust-aware bootstrap — scale `BOOTSTRAP_WEIGHT` by ensemble agreement — rather
than the current all-or-nothing constant, so a from-scratch or corrupted Q degrades
gracefully to the pure-reward planner instead of poisoning it.

## Follow-up (2026-07-21b): the score metric was gameable, pessimism was hurting

Three findings, all from replacing single-seed spot checks with paired multi-seed sweeps.
Every claim below is n>=12 with seeds matched across variants.

**The score metric rewarded spinning.** `score = integral of |pole_angle| dt` over a 30s
episode. A freely spinning pole averages `|angle| = pi/2` and scores **~47**; a balanced
one averages `pi` and scores **~94**. The learning check's `best score >= 45` gate was
therefore satisfied by a pole spinning and never balancing — observed directly at seed 7,
episode 1: `score 43.91 | upright 11%`. `sim` now tracks `upright_time` (time with
`cos(angle) < -0.9`, i.e. within ~26 degrees of inverted) and the gate is
`best_upright >= 50%`. The separation is unambiguous and ungameable: spinning reads ~14%,
balancing reads 90%+.

**`PESSIMISM :: 1` was costing early learning.** The planner subtracted the ensemble's
return standard deviation from every candidate's score. Early in training the models
disagree everywhere, so an unnormalized penalty on 20-step *accumulated* return swamps the
reward term and makes CEM select for "where the models already agree" — i.e. visited
states — instead of for reward. That penalizes precisely the swing-up exploration the task
needs. Paired by seed, `PESSIMISM=0` minus `PESSIMISM=1`:

| metric | 24 seeds x 2 sim-min | 12 seeds x 6 sim-min |
| --- | --- | --- |
| best score | +7.38 (t=2.86) | +0.67 (t=3.78) |
| best upright | +14.88 pts (t=2.74) | +0.92 pts (t=3.53) |
| upright last 10 | +12.46 pts (t=2.90) | +5.50 pts (t=1.54) |

Median best-upright at 2 sim-min moves 48% -> 88%. The effect is large early and mostly
gone by 6 minutes (both converge to ~97%), which is why it hid from single long runs.
`PESSIMISM` is now 0 and the deviation computation sits behind a compile-time
`when PESSIMISM > 0`, mirroring `BOOTSTRAP_WEIGHT`. The machinery is kept because ensemble
disagreement is still wanted as the *trust signal* gating the value bootstrap — a targeted
use, not a blanket penalty on every rollout return.

**The test harness and the headless harness were not running the same experiment.**
`core:testing` installs a Xoshiro256 generator (`runner.odin:155-158`) while `main` uses
the runtime default, so `rand.reset(seed)` seeded different PRNGs and identical seeds
produced different trajectories — seed 1 read 93% upright headless and 27% under
`odin test`. Both harnesses now install `rand.default_random_generator` over an explicit
`rand.create(seed)` state, and seed 1 reproduces `best score 89.99` in both. Any
comparison made across the two harnesses before this fix is void.

**Method note.** Wall time scales with total sim-time, so many-short-runs is both cheaper
and better-powered than few-long-runs for early-learning questions: 24 seeds x 2 sim-min
is less work than 12 x 6 and doubles the sample. Variants must be compared **paired by
seed** — the unpaired test on the same data read t~1.8 (inconclusive) where the paired test
read t~2.9, because per-seed difficulty variance dwarfs the treatment effect.

**Still open.** About a third of seeds fail to reach balancing within 2 sim-min (q1 = 45%
upright, min 16%), so "learns reliably and fast" is not yet true. `value_fit` went
*negative* on some seeds (-0.52, 0.22 at two axes) while sitting at 0.85-0.96 on others:
Q is not reliably sane, which matters before any trust-aware bootstrap reads it. And the
wall-stuck failure mode did not reproduce in 36 headless seeds (0 early-ended episodes) —
plausibly because it lives in the *async* windowed path (`start`/`sense`/`act`), which the
lockstep `drive` harness structurally cannot exercise.

## Follow-up (2026-07-21): Q verified, sensor layout parameterized

Three cleanups, all behavior-preserving (headless seed 1 scores are unchanged to the
decimal), taken before starting game B:

- **Q is no longer unmeasured.** Because `BOOTSTRAP_WEIGHT` is 0 and the policy is
  behavior-cloned, nothing downstream depended on the critic, so nothing would have
  noticed it rotting. `agent.value_fit` walks the buffer in chronological order (episode
  boundaries detected by `s + Δ ≈ s'`), accumulates the observed discounted return over a
  150-step window (`γ^150 ≈ 0.05`, terminating early on `dead` with the same
  `-DEATH_PENALTY` the critic is trained against), and reports the Pearson correlation
  against `min_i Q_i(s, a)`. Measured: **0.76 after one episode, 0.84-0.87 thereafter**.
  Q was in fact learning a sensible function the whole time; it was simply unread. The
  learning check now asserts fit ≥ 0.5, so the trust-aware bootstrap will be debugging one
  new thing rather than two.
- **The sin/cos renormalization is now data, not code.** `_apply_delta`'s hardcoded
  `state[2]/state[3]` wart is replaced by `[]Angle_Pair` passed to `agent.make` and copied
  into a fixed `MAX_ANGLE_PAIRS` array, with bounds asserted at construction. Cartpole
  declares its own layout in `sim.ANGLE_PAIRS`. This was the item blocking a second game
  sharing the observation space.
- **The two "agreement" signals are disambiguated.** The exposed one was planner↔policy
  action match, which saturates at 100% by episode 2 and is not a trust signal; it is now
  `policy_match`. The name `agreement` is left free for the ensemble-disagreement quantity
  in `_rollout`, which is what the trust-aware bootstrap will actually consume.
