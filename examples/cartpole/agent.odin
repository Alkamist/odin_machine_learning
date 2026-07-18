package main

import "core:math"
import "core:math/rand"
import "core:slice"

import ml "../../"
import    "../../networks/mlp"

// A model-based agent that learns to swing up and balance the pole from
// scratch, in realtime, with no pre-training.
//
// Model-free RL is hopeless on this budget: 500 frames is ~166 decisions, and
// each one yields a single scalar of reward. Instead we learn the *dynamics*
// - given a state and an action, what is the next state - which extracts a
// dense 5-dimensional regression target from every transition. A few hundred
// transitions is plenty to fit a system this small. Choosing actions is then a
// search problem rather than a learning problem: roll candidate action
// sequences through the learned model and keep the best one (CEM/MPC).

ACTION_COUNT :: len(Action)
OBS_SIZE     :: 5
MODEL_INPUT  :: OBS_SIZE + ACTION_COUNT
HIDDEN_SIZE  :: 96

// The agent commits to an action for this many physics frames. Holding actions
// shrinks the search space and makes the sampled sequences coherent enough to
// pump energy into the pole, and it lets the model predict a 3-frame step,
// which is 3x cheaper to roll out and accumulates 3x less error per horizon.
ACTION_REPEAT :: 3

PLAN_HORIZON :: 40 // compiled maximum; see tuning.plan_horizon
PLAN_SAMPLES :: 192
PLAN_ELITES  :: 8
PLAN_ITERS   :: 3

BUFFER_CAPACITY  :: 4096
TRAIN_BATCH_SIZE :: 64
TRAIN_MINIMUM    :: 8
TRAIN_STEPS      :: 32
LEARNING_RATE    :: 3e-3

// Decisions of uniformly random actions before planning takes over. The model
// is worthless until it has seen the state space, and random exploration
// covers it faster than a plan drawn from a model that knows nothing.
WARMUP_DECISIONS :: 24

// Observations are normalized into roughly [-1, 1] so a single network scale
// works for every component.
X_SCALE :: f32(CART_LIMIT)
V_SCALE :: f32(CART_SPEED)
W_SCALE :: f32(8)

// Normalized pole energy: 1.0 is exactly the energy needed to stand upright.
// For a uniform rod pivoted at one end, E/(m*g*L) = L*w^2/(6*g) + (1-cos)/2.
ENERGY_SCALE :: f32(POLE_SIZE.y) / (6.0 * GRAVITY)

Observation :: [OBS_SIZE]f32

Transition :: struct {
	observation: Observation,
	action:      Action,
	delta:       Observation,
}

Agent :: struct {
	model: mlp.Mlp,
	opt:   ml.Optimizer,

	buffer:       [BUFFER_CAPACITY]Transition,
	buffer_count: int,
	buffer_next:  int,

	// Running statistics of the deltas, used to standardize the regression
	// targets. Without this the network spends its capacity on the components
	// that happen to have the largest raw magnitude.
	delta_mean:     Observation,
	delta_sq_mean:  Observation,
	delta_samples:  int,

	// Per-timestep action distribution, carried across decisions so each plan
	// starts from the previous one shifted forward by a step.
	plan: [PLAN_HORIZON][ACTION_COUNT]f32,

	action:       Action,
	hold:         int,
	previous:     Observation,
	has_previous: bool,
	decisions:    int,
}

agent_make :: proc(allocator := context.allocator) -> (agent: Agent) {
	agent.model = mlp.make(MODEL_INPUT, HIDDEN_SIZE, OBS_SIZE, allocator=allocator)
	agent_forget_episode(&agent)
	return
}

agent_destroy :: proc(agent: Agent) {
	mlp.destroy(agent.model)
}

// Called on episode reset. The model and the replay buffer survive - that is
// the whole point - but anything tied to the trajectory does not.
agent_forget_episode :: proc(agent: ^Agent) {
	agent.has_previous = false
	agent.hold         = 0
	agent.action       = .None
	for h in 0 ..< tuning.plan_horizon {
		for a in 0 ..< ACTION_COUNT {
			agent.plan[h][a] = 1.0 / f32(ACTION_COUNT)
		}
	}
}

// Drives one physics frame: records the transition that just completed, trains
// on it, and re-plans, but only on frames where the held action expires.
agent_step :: proc(agent: ^Agent, state: State) -> Action {
	if agent.hold > 0 {
		agent.hold -= 1
		return agent.action
	}

	observation := observe(state)

	if agent.has_previous {
		transition := Transition {
			observation = agent.previous,
			action      = agent.action,
		}
		for i in 0 ..< OBS_SIZE {
			transition.delta[i] = observation[i] - agent.previous[i]
		}
		agent_remember(agent, transition)
	}

	agent_train(agent)

	if agent.decisions < tuning.warmup {
		agent.action = Action(rand.int_max(ACTION_COUNT))
	}
	else {
		agent.action = agent_plan(agent, observation)
	}

	agent.previous     = observation
	agent.has_previous = true
	agent.hold         = ACTION_REPEAT - 1
	agent.decisions   += 1

	return agent.action
}

agent_remember :: proc(agent: ^Agent, transition: Transition) {
	agent.buffer[agent.buffer_next] = transition
	agent.buffer_next = (agent.buffer_next + 1) % BUFFER_CAPACITY
	if agent.buffer_count < BUFFER_CAPACITY {
		agent.buffer_count += 1
	}

	agent.delta_samples += 1
	rate := 1.0 / f32(agent.delta_samples)
	for i in 0 ..< OBS_SIZE {
		d := transition.delta[i]
		agent.delta_mean[i]    += (d     - agent.delta_mean[i])    * rate
		agent.delta_sq_mean[i] += (d * d - agent.delta_sq_mean[i]) * rate
	}
}

// Half the batch comes from the whole buffer and half from the most recent
// quarter of it. The states the agent most needs accuracy in - near the top,
// where it is trying to balance - are also the ones it has only just started
// visiting, and uniform sampling drowns them in the swing-up data.
@(require_results)
agent_sample_index :: proc(agent: ^Agent) -> int {
	if rand.float32() < 0.5 {
		return rand.int_max(agent.buffer_count)
	}

	recent := max(agent.buffer_count / 4, 1)
	age    := rand.int_max(recent)
	return (agent.buffer_next - 1 - age + BUFFER_CAPACITY) % BUFFER_CAPACITY
}

@(require_results)
agent_delta_deviation :: proc(agent: ^Agent, i: int) -> f32 {
	variance := agent.delta_sq_mean[i] - agent.delta_mean[i] * agent.delta_mean[i]
	return max(math.sqrt(max(variance, 0)), 1e-4)
}

@(require_results)
observe :: proc(state: State) -> (observation: Observation) {
	position := cart_position(state)
	velocity := cart_velocity(state)
	angle    := pole_angle(state)
	spin     := pole_spin(state)

	observation[0] = position / X_SCALE
	observation[1] = velocity / V_SCALE
	observation[2] = math.sin(angle)
	observation[3] = math.cos(angle)
	observation[4] = spin / W_SCALE
	return
}

encode :: proc(observation: Observation, action: Action, dst: []f32) {
	for i in 0 ..< OBS_SIZE {
		dst[i] = observation[i]
	}
	for i in 0 ..< ACTION_COUNT {
		dst[OBS_SIZE + i] = 0
	}
	dst[OBS_SIZE + int(action)] = 1
}

agent_train :: proc(agent: ^Agent) {
	// Sampling with replacement, so training starts as soon as there is
	// anything at all to train on rather than idling until a full batch has
	// accumulated. Those first few dozen frames are a large slice of the
	// budget the agent has to get this right in.
	if agent.buffer_count < TRAIN_MINIMUM {
		return
	}

	inputs  := make([]f32, TRAIN_BATCH_SIZE * MODEL_INPUT, context.temp_allocator)
	targets := make([]f32, TRAIN_BATCH_SIZE * OBS_SIZE,    context.temp_allocator)

	for _ in 0 ..< tuning.train_steps {
		ml.clear(training=true)

		for b in 0 ..< TRAIN_BATCH_SIZE {
			transition := agent.buffer[agent_sample_index(agent)]

			encode(transition.observation, transition.action, inputs[b * MODEL_INPUT:][:MODEL_INPUT])

			for i in 0 ..< OBS_SIZE {
				targets[b * OBS_SIZE + i] = (transition.delta[i] - agent.delta_mean[i]) / agent_delta_deviation(agent, i)
			}
		}

		x          := ml.reshape(ml.tensor(inputs),  {TRAIN_BATCH_SIZE, MODEL_INPUT})
		y          := ml.reshape(ml.tensor(targets), {TRAIN_BATCH_SIZE, OBS_SIZE})
		prediction := mlp.forward(agent.model, x)
		loss       := ml.mean(ml.mean_squared_error(prediction, y))

		ml.backward(loss)

		if ml.optimizer_step(&agent.opt, period=1, learning_rate=tuning.learning_rate) {
			mlp.update(agent.opt, agent.model)
		}
	}
}

// Cross-entropy method over per-timestep action distributions: sample a batch
// of action sequences, roll them all through the learned model at once, keep
// the best few, and refit the distribution toward them.
@(require_results)
agent_plan :: proc(agent: ^Agent, observation: Observation) -> Action {
	sequences := make([][PLAN_HORIZON]Action, tuning.plan_samples, context.temp_allocator)
	returns   := make([]f32,                  tuning.plan_samples, context.temp_allocator)
	order     := make([]int,                  tuning.plan_samples, context.temp_allocator)

	// Warm start: the plan from the previous decision is still mostly valid,
	// just one step stale.
	for h in 0 ..< tuning.plan_horizon - 1 {
		agent.plan[h] = agent.plan[h + 1]
	}
	for a in 0 ..< ACTION_COUNT {
		agent.plan[tuning.plan_horizon - 1][a] = 1.0 / f32(ACTION_COUNT)
	}

	for _ in 0 ..< tuning.plan_iters {
		for n in 0 ..< tuning.plan_samples {
			for h in 0 ..< tuning.plan_horizon {
				sequences[n][h] = sample_action(agent.plan[h])
			}
			returns[n] = 0
		}

		agent_rollout(agent, observation, sequences, returns)

		// Selection sort of just the elites: cheaper than ordering every
		// sample when only the top few matter.
		for i in 0 ..< tuning.plan_samples {
			order[i] = i
		}
		for e in 0 ..< tuning.plan_elites {
			best := e
			for i in e + 1 ..< tuning.plan_samples {
				if returns[order[i]] > returns[order[best]] {
					best = i
				}
			}
			slice.swap(order, e, best)
		}

		for h in 0 ..< tuning.plan_horizon {
			counts: [ACTION_COUNT]f32
			for e in 0 ..< tuning.plan_elites {
				counts[int(sequences[order[e]][h])] += 1.0 / f32(tuning.plan_elites)
			}
			// Momentum keeps a single unlucky batch from collapsing the plan,
			// and the floor keeps every action reachable on the next iteration.
			for a in 0 ..< ACTION_COUNT {
				agent.plan[h][a] = 0.5 * agent.plan[h][a] + 0.5 * counts[a]
				agent.plan[h][a] = max(agent.plan[h][a], 0.02)
			}
		}
	}

	best := 0
	for a in 1 ..< ACTION_COUNT {
		if agent.plan[0][a] > agent.plan[0][best] {
			best = a
		}
	}
	return Action(best)
}

// Rolls every candidate sequence through the model as one batch, one horizon
// step at a time, accumulating shaped reward.
agent_rollout :: proc(agent: ^Agent, observation: Observation, sequences: [][PLAN_HORIZON]Action, returns: []f32) {
	ml.clear()

	states := make([]Observation, tuning.plan_samples, context.temp_allocator)
	alive  := make([]bool,        tuning.plan_samples, context.temp_allocator)
	inputs := make([]f32,         tuning.plan_samples * MODEL_INPUT, context.temp_allocator)
	deltas := make([]f32,         tuning.plan_samples * OBS_SIZE,    context.temp_allocator)

	for n in 0 ..< tuning.plan_samples {
		states[n] = observation
		alive[n]  = true
	}

	discount := f32(1)

	for h in 0 ..< tuning.plan_horizon {
		for n in 0 ..< tuning.plan_samples {
			encode(states[n], sequences[n][h], inputs[n * MODEL_INPUT:][:MODEL_INPUT])
		}

		x          := ml.reshape(ml.tensor(inputs), {tuning.plan_samples, MODEL_INPUT})
		prediction := mlp.forward(agent.model, x)
		ml.get_data(prediction, deltas)

		for n in 0 ..< tuning.plan_samples {
			if !alive[n] {
				continue
			}

			for i in 0 ..< OBS_SIZE {
				states[n][i] += deltas[n * OBS_SIZE + i] * agent_delta_deviation(agent, i) + agent.delta_mean[i]
			}

			// sin/cos drift off the unit circle as errors accumulate; pulling
			// them back keeps the rolled-out angle meaningful.
			length := math.sqrt(states[n][2] * states[n][2] + states[n][3] * states[n][3])
			if length > 1e-4 {
				states[n][2] /= length
				states[n][3] /= length
			}

			reward, dead := plan_reward(states[n])
			returns[n]   += discount * reward

			if dead {
				// Hitting a wall ends the episode, so a plan that runs into one
				// forfeits everything it would have earned afterwards.
				returns[n] -= 40
				alive[n]    = false
			}
		}

		discount *= tuning.discount
	}
}

// The reward the planner optimizes, which is deliberately *not* the score the
// game reports. Integrated |angle| is greedy: from a dead hang there is no
// sequence within the horizon that increases it, so a planner maximizing it
// just jitters. The energy term fixes that - it rewards pumping the pole
// toward exactly the energy required to stand up, which is reachable inside
// one second and which leads to the upright state that the real score wants.
@(require_results)
plan_reward :: proc(state: Observation) -> (reward: f32, dead: bool) {
	cos_angle := state[3]
	spin      := state[4] * W_SCALE

	upright := -cos_angle // 1 upright, -1 hanging
	energy  := ENERGY_SCALE * spin * spin + 0.5 * (1 - cos_angle)

	energy_error := energy - 1

	reward = tuning.upright_weight * upright
	reward -= tuning.energy_weight * energy_error * energy_error
	reward -= tuning.center_weight * state[0] * state[0]

	// Once the pole is near the top, stop rewarding it for carrying spin so
	// the agent settles into a balance instead of pinwheeling through upright.
	if upright > 0 {
		reward -= tuning.spin_weight * upright * state[4] * state[4]
	}

	// Balancing needs the cart to keep moving under the pole, so it drifts, and
	// a cart that drifts into a wall ends the episode. A plain quadratic pull
	// toward the middle is too soft to stop that: it loses to the upright term
	// right up until the wall is one step away, at which point the only way out
	// is a reversal violent enough to drop the pole. This barrier ramps up from
	// half-track out, so the agent buys its way back to the middle early, while
	// the correction is still gentle enough to balance through.
	barrier := max(abs(state[0]) - tuning.barrier_onset, 0)
	reward  -= tuning.barrier_weight * barrier * barrier

	dead = abs(state[0]) > 0.9
	return
}

@(require_results)
sample_action :: proc(probabilities: [ACTION_COUNT]f32) -> Action {
	total: f32
	for p in probabilities {
		total += p
	}

	target := rand.float32() * total
	sum:    f32

	for a in 0 ..< ACTION_COUNT {
		sum += probabilities[a]
		if target <= sum {
			return Action(a)
		}
	}
	return Action(ACTION_COUNT - 1)
}
