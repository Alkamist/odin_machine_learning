# Where the Learning Signal Comes From

Design conversation, 2026-07-21, prompted by the lander dawdling and by the question
"what exactly do you give a model if you want an exact task but no hand-crafted reward?"

This is an argument, not a result. Three things in it are measured (marked MEASURED); the
rest is reasoning that should be treated as a hypothesis with a falsifiable shape. It sets
the framing for the work in `CONTINUOUS_TDMPC.md` and `CONTINUAL_LEARNING.md` rather than
replacing either.

## The precipitating evidence

MEASURED (headless, `-o:speed`, 3 sim-min): lander lands but dawdles. Seed 1 landed 4 of 6
with episodes running 18-30s against a 30s limit; seed 2 landed 4 of 8 with 3 crashes and
1 timeout. Descent is an exponential approach that never commits.

This is not a bug. `POS_WEIGHT * height^2 + VEL_WEIGHT * velocity_y^2` has a closed-form
greedy optimum — descend at a speed proportional to remaining altitude. The agent is
executing the reward function exactly as written. Adding a descent-rate term would fix
lander and make the project worse, because it entrenches the dependency that has to go.

The same reading applied to cartpole is sharper. `cartpole.odin`'s `reward` contains:

    energy       := ENERGY_SCALE * spin * spin + 0.5 * (1 - cos_angle)
    energy_error := energy - 1
    reward       -= ENERGY_WEIGHT * energy_error * energy_error

That is the Astrom-Furuta energy-pumping swing-up law, hand-written, driving pole energy to
a target of 1. The agent does not discover swing-up. Swing-up is in the reward, and the
agent runs MPC against a learned plant to track a hand-written Lyapunov function. Fast and
respectable engineering; not "learns generally."

MEASURED: lander's `value_fit` reads 0.97 (311 samples) after 3 sim-min. The critic is
excellent and is consumed by nothing — `BOOTSTRAP_WEIGHT` has been 0 since 2026-07-20.

## A reward function does two jobs

1. **Specifies the goal** — what counts as success.
2. **Supplies the learning gradient** — the dense local "warmer/colder".

"Win the game" does job 1 perfectly. It is not vague or underspecified; it is a complete
specification of Melee. What it does not do is job 2.

Hand-crafted shaping is job 2 masquerading as job 1. That is why it is corrosive: to supply
the gradient you must already know the answer, so the agent's ceiling becomes the author's
understanding.

## Why "get control right first, strip the reward later" cannot work

This was the standing plan and it is wrong, for a reason worth keeping.

Shaping does not merely accelerate learning. It changes *which machine gets built*. A dense
near-Lyapunov reward makes a short-horizon planner over a learned plant sufficient — so
that is what got built, and it works. The components sparse reward requires (a load-bearing
value function, real exploration) never had to exist, because nothing was broken without
them.

`BOOTSTRAP_WEIGHT :: 0` is that fact as an artifact. The critic has been trained, measured,
and read by nothing for the life of the project. The dense reward did not sit alongside the
value function; it crowded it out.

Shaping is not a scaffold you remove later. It is a load path. Remove it and there is no
other one.

## The bandwidth argument

Win/lose is roughly one bit per five-minute game. Fitting a policy of real capacity from
one bit per five minutes takes an astronomical number of games. That is approximately what
the large results did — OpenAI Five on the order of tens of thousands of simulated years
and with substantial hand-shaping; AlphaStar reaching top-tier play only after bootstrapping
from a large corpus of human replays. (Figures from memory, order-of-magnitude only, and
not load-bearing — the shape of the argument is what matters.)

That is the brute-force fix, and `GENERAL_LEARNER.md` forbids it: realtime, serial, online,
minutes not millennia. So pure sparse reward is ruled out here on **information-theoretic
grounds, not algorithmic ones**. There are not enough bits per second in the objective, and
massive parallelism is disallowed.

Which says exactly where the bits must come from instead. The sensory stream is millions of
bits per second and it is free.

## What the agent gets given

- **The sparse true objective.** Win/lose, land/crash. Never shaped. It is small and its
  only job is to *select among* behaviors, not to construct them.
- **A dense self-supervised objective** — predict your own next observation. Task-free,
  always on, reward-independent, and the source of essentially all the bits. Already
  present as the dynamics ensemble, and the healthiest part of the system precisely
  because it never depended on reward.
- **A value function.** Sparse-to-dense propagation over time is the entire job description
  of TD learning. There is no other legitimate mechanism, which is why the critic has to
  stop being decorative.
- **A task-free exploration drive** — model disagreement, novelty, or empowerment. Generates
  the coverage the value function needs. Known failure mode: raw novelty produces an agent
  that does fascinating nothing (the noisy-TV problem), so the sparse objective stays as
  the leash.
- **Demonstrations.** No human derives wavedashing from "win the game." `GENERAL_LEARNER.md`
  already lists learning by observation; for Melee it is load-bearing, not optional.

## Goals, not rewards, as the instruction channel

From the conversation: "my reward is trying to enjoy life and not dying, everything in
between I make up myself."

The thing made up in between is a **goal**, and a goal is not a reward — it is a predicate
over states. "Land on the pad." "Get them off-stage." "Learn to wavedash."

A goal-conditioned value function is the machinery that turns a goal predicate into dense
control. This is the concrete form of the "magic" item in `GENERAL_LEARNER.md`: you do not
instruct a human by handing them a reward function, you name a goal state and their own
machinery supplies the gradient. Goals are also relabelable — every trajectory is a
successful demonstration of reaching wherever it actually ended up — which manufactures
dense supervision out of failure at zero shaping cost.

## The wavedash spec

The target capability, stated as a story: someone accidentally wavedashes, thinks "that
might be useful," and adopts it *even though it will make them worse for a while*.

Decomposed, that is six requirements:

1. **The accident.** Prediction error, not reward. The model said "land and stop," the world
   said "slide." Dense and free.
2. **Noticing.** Tag the surprise as interesting rather than noise, and attribute it to the
   input sequence that caused it.
3. **Reproduction.** Deliberately make it happen again, with no reward for doing so. Novelty
   as a goal.
4. **Evaluation.** Imagine where *having the capability* beats current options — reasoning
   about the technique, not measuring win rate after the fact.
5. **Accepting regression.** Take known short-term loss to acquire the skill.
6. **Transmission.** The discovery propagates through a population rather than being
   re-derived by each individual.

Where the current plan lands: 1 solidly, 2 partially, 3 well, and then it stops. 4 and 5
are not in it, and they are the ones that constitute the intelligence being asked for.

## The two holes

**Step 4 needs temporal abstraction.** To ask "is wavedashing useful," the wavedash must
exist as a *unit* with preconditions and effects, not as twelve frames of stick input. The
planner imagines 20 primitive steps at 50ms — 1.0 sim-second — and structurally cannot
represent a technique, let alone evaluate one strategically.

**Step 5 needs the agent to value its own learning progress.** Every RL objective maximizes
expected return; a policy that knowingly takes worse return to acquire a skill is
off-objective unless the objective includes the *derivative* of competence rather than
competence. That formulation exists (learning-progress intrinsic motivation, Oudeyer;
compression progress, Schmidhuber) and is the right variant specifically because it fixes
the noisy-TV failure — an agent maximizing improvement rate leaves the static screen as
soon as it stops improving. Named and formalized; unsolved at any scale that matters.

## The unification

"Thousands of hours, not thousands of years" and "discover and adopt a technique" look like
two asks. They are one.

Human sample efficiency does not come from a better gradient. It comes from **not searching
in frame-space**. A pro thinks "shield-drop into up-air," not forty inputs. Search happens
over a compositional vocabulary of techniques, collapsing effective planning depth by one to
two orders of magnitude — and that vocabulary is exactly what step 4 needs in order to
evaluate a new entry against.

So hierarchy is the single missing piece under both demands. It is also sitting underneath
the lander problem: lander dawdles because a 1-second primitive-action horizon cannot
represent "commit to a descent." Same disease, small enough to measure in minutes.

## Step 6: population and culture

The wavedash was found once, across thousands of players over months, and then transmitted.
Any given player almost certainly learned it from someone else. "A single agent, alone,
discovers wavedashing" is therefore a *harder* bar than human-level — it asks one agent to
do what a culture did.

Two consequences, one deflationary and one generative:

- **Deflationary.** Target the individual-scale version. Every player constantly discovers
  small personal tech; that is the real individual capability, and it is a far better
  testbed because it is observable in an afternoon.
- **Generative.** A *population* of agents — communicating, competing, refining — is the
  natural home for the full-strength version. Competition supplies the nonstationary
  opponent distribution that keeps the sparse objective informative; communication supplies
  the transmission channel that lets one agent's discovery become the population's
  vocabulary. That vocabulary is the same object hierarchy needs (§ The unification), which
  suggests population and skill-discovery are two views of one mechanism rather than
  separate features.

Explicitly parked. It presumes single-agent steps 1-5 work, and it re-admits parallelism —
which has to be justified as *cultural transmission between lifelong learners* rather than
smuggling back the tens-of-thousands-of-years brute force the whole framing rejects. That
distinction is the thing to keep honest if this is ever picked up.

## Honest caveats

- **The field's record on hierarchy is poor.** Options and hierarchical RL have been the
  obvious next thing for twenty-five years and have mostly underdelivered; discovered
  skills tend to be degenerate or need hand-specification, which is the shaping treadmill
  wearing a different hat. The diagnosis here is defensible. A recipe is not claimed.
- **Intrinsic motivation is an excellent way to build an agent that does nothing.** The
  sparse objective is not optional as a leash.
- **This document argues; it does not measure.** Only the three MEASURED lines are evidence.
  Everything else earns its place by producing a falsifiable next experiment or gets cut.

## What this changes about the plan

The lander ladder from `CONTINUOUS_TDMPC.md`'s open items is not replaced — it is
re-purposed. It is the *substrate*, not the destination. Steps 1-3 are prerequisites: there
is no building learning-progress exploration on a system with no goal channel and a critic
at zero weight, and no discovering skills in a system that only plans one second ahead.

Sequencing, revised:

1. **Fix lander's dawdling under the current dense reward** — trust-gated value bootstrap
   (rung 5 of `CONTINUAL_LEARNING.md`), which now has something to win rather than a
   hand-tuned reward to lose to. Clean paired A/B. The trust signal should probably be the
   already-computed `_value_fit` correlation rather than ensemble disagreement, since
   disagreement is a proxy for what that statistic directly is.
2. **Plan at a coarser imagined timestep than the action timestep.** Buys a ~10s horizon at
   identical compute, needs no Q at all, and — the actual reason — creates the place where
   skills will eventually live. Do it now rather than retrofitting.
3. **Make lander's reward a goal predicate**, not a shaped scalar. Landed / on-pad /
   upright / slow. Strip `POS_WEIGHT`, `VEL_WEIGHT`, `TILT_WEIGHT`, `SPIN_WEIGHT` one at a
   time, dense reward as the paired control at every rung, and find where it breaks.
   Whatever breaks is the real research problem, now isolated.
4. Only then are steps 4-5 of the wavedash spec experiments that can actually be run.

The gate for "this project made real progress" is: **lander solved from a goal-only reward.**
At that point the competence is in the weights for the first time, and the cross-task
retention experiment becomes sharp — today it would mostly measure a dynamics model that
transfers between two box2d games anyway, because the task-specific knowledge is sitting in
a hand-written reward proc that cannot be forgotten.

## What to avoid

Building a lander that works for reasons that do not extend. That is what happened with
cartpole, and the bill was a value function sitting dead at zero weight for months while
the thing that actually solved the task was a control law typed in by hand.
