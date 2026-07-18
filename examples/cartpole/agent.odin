package main

import "core:math"
import "core:math/rand"
import "core:slice"

import ml "../../"
import    "../../networks/mlp"

ACTION_COUNT :: len(Action)
OBS_SIZE     :: 5
MODEL_INPUT  :: OBS_SIZE + ACTION_COUNT
HIDDEN_SIZE  :: 32

ACTION_REPEAT :: 3

ENSEMBLE_SIZE :: 5

PLAN_HORIZON  :: 20
PLAN_SAMPLES  :: 64
PLAN_ELITES   :: 8
PLAN_DISCOUNT :: f32(0.98)

PESSIMISM :: f32(1)

BUFFER_CAPACITY  :: 4096
TRAIN_BATCH_SIZE :: 64
TRAIN_MINIMUM    :: 8
TRAIN_STEPS      :: 24
LEARNING_RATE    :: 3e-3

WARMUP_DECISIONS :: 24

UPRIGHT_WEIGHT :: f32(3)
ENERGY_WEIGHT  :: f32(3)
CENTER_WEIGHT  :: f32(3)
SPIN_WEIGHT    :: f32(2)
BARRIER_ONSET  :: f32(0.5)
BARRIER_WEIGHT :: f32(20)

X_SCALE :: f32(CART_LIMIT)
V_SCALE :: f32(CART_SPEED)
W_SCALE :: f32(8)

ENERGY_SCALE :: f32(POLE_SIZE.y) / (6.0 * GRAVITY)

Observation :: [OBS_SIZE]f32

Transition :: struct {
	observation: Observation,
	action:      Action,
	delta:       Observation,
}

Agent :: struct {
	models: [ENSEMBLE_SIZE]mlp.Mlp,
	opts:   [ENSEMBLE_SIZE]ml.Optimizer,

	buffer:       [BUFFER_CAPACITY]Transition,
	buffer_count: int,
	buffer_next:  int,

	delta_mean:     Observation,
	delta_sq_mean:  Observation,
	delta_samples:  int,

	plan: [PLAN_HORIZON][ACTION_COUNT]f32,

	action:       Action,
	hold:         int,
	train_credit: f32,
	previous:     Observation,
	has_previous: bool,
	decisions:    int,
}

agent_make :: proc(allocator := context.allocator) -> (agent: Agent) {
	for m in 0 ..< ENSEMBLE_SIZE {
		agent.models[m] = mlp.make(MODEL_INPUT, HIDDEN_SIZE, OBS_SIZE, allocator=allocator)
	}
	agent_forget_episode(&agent)
	return
}

agent_destroy :: proc(agent: Agent) {
	for m in 0 ..< ENSEMBLE_SIZE {
		mlp.destroy(agent.models[m])
	}
}

agent_forget_episode :: proc(agent: ^Agent) {
	agent.has_previous = false
	agent.hold         = 0
	agent.action       = .None
	for h in 0 ..< PLAN_HORIZON {
		for a in 0 ..< ACTION_COUNT {
			agent.plan[h][a] = 1.0 / f32(ACTION_COUNT)
		}
	}
}

agent_step :: proc(agent: ^Agent, state: State) -> Action {
	agent.train_credit += f32(TRAIN_STEPS) / f32(ACTION_REPEAT)

	steps := int(agent.train_credit)
	agent.train_credit -= f32(steps)
	agent_train(agent, steps)

	if agent.hold > 0 {
		agent.hold -= 1
		if agent.decisions >= WARMUP_DECISIONS {
			agent_plan_refine(agent, agent.previous)
		}
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

	if agent.decisions < WARMUP_DECISIONS {
		agent.action = Action(rand.int_max(ACTION_COUNT))
	}
	else {
		agent_plan_shift(agent)
		agent_plan_refine(agent, observation)
		agent.action = agent_plan_action(agent)
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

agent_train :: proc(agent: ^Agent, steps: int) {
	if agent.buffer_count < TRAIN_MINIMUM {
		return
	}

	inputs  := make([]f32, TRAIN_BATCH_SIZE * MODEL_INPUT, context.temp_allocator)
	targets := make([]f32, TRAIN_BATCH_SIZE * OBS_SIZE,    context.temp_allocator)

	for _ in 0 ..< steps {
		ml.clear(training=true)

		total: ml.Tensor

		for m in 0 ..< ENSEMBLE_SIZE {
			for b in 0 ..< TRAIN_BATCH_SIZE {
				transition := agent.buffer[agent_sample_index(agent)]

				encode(transition.observation, transition.action, inputs[b * MODEL_INPUT:][:MODEL_INPUT])

				for i in 0 ..< OBS_SIZE {
					targets[b * OBS_SIZE + i] = (transition.delta[i] - agent.delta_mean[i]) / agent_delta_deviation(agent, i)
				}
			}

			x          := ml.reshape(ml.tensor(inputs),  {TRAIN_BATCH_SIZE, MODEL_INPUT})
			y          := ml.reshape(ml.tensor(targets), {TRAIN_BATCH_SIZE, OBS_SIZE})
			prediction := mlp.forward(agent.models[m], x)
			loss       := ml.mean(ml.mean_squared_error(prediction, y))

			total = loss if m == 0 else ml.add(total, loss)
		}

		ml.backward(total)

		for m in 0 ..< ENSEMBLE_SIZE {
			if ml.optimizer_step(&agent.opts[m], period=1, learning_rate=LEARNING_RATE) {
				mlp.update(agent.opts[m], agent.models[m])
			}
		}
	}
}

agent_plan_shift :: proc(agent: ^Agent) {
	for h in 0 ..< PLAN_HORIZON - 1 {
		agent.plan[h] = agent.plan[h + 1]
	}
	for a in 0 ..< ACTION_COUNT {
		agent.plan[PLAN_HORIZON - 1][a] = 1.0 / f32(ACTION_COUNT)
	}
}

@(require_results)
agent_plan_action :: proc(agent: ^Agent) -> Action {
	best := 0
	for a in 1 ..< ACTION_COUNT {
		if agent.plan[0][a] > agent.plan[0][best] {
			best = a
		}
	}
	return Action(best)
}

agent_plan_refine :: proc(agent: ^Agent, observation: Observation) {
	sequences := make([][PLAN_HORIZON]Action, PLAN_SAMPLES, context.temp_allocator)
	returns   := make([]f32,                  PLAN_SAMPLES, context.temp_allocator)
	order     := make([]int,                  PLAN_SAMPLES, context.temp_allocator)

	for n in 0 ..< PLAN_SAMPLES {
		for h in 0 ..< PLAN_HORIZON {
			sequences[n][h] = sample_action(agent.plan[h])
		}
	}

	agent_rollout(agent, observation, sequences, returns)

	for i in 0 ..< PLAN_SAMPLES {
		order[i] = i
	}
	for e in 0 ..< PLAN_ELITES {
		best := e
		for i in e + 1 ..< PLAN_SAMPLES {
			if returns[order[i]] > returns[order[best]] {
				best = i
			}
		}
		slice.swap(order, e, best)
	}

	for h in 0 ..< PLAN_HORIZON {
		counts: [ACTION_COUNT]f32
		for e in 0 ..< PLAN_ELITES {
			counts[int(sequences[order[e]][h])] += 1.0 / f32(PLAN_ELITES)
		}
		for a in 0 ..< ACTION_COUNT {
			agent.plan[h][a] = 0.5 * agent.plan[h][a] + 0.5 * counts[a]
			agent.plan[h][a] = max(agent.plan[h][a], 0.02)
		}
	}
}

agent_rollout :: proc(agent: ^Agent, observation: Observation, sequences: [][PLAN_HORIZON]Action, returns: []f32) {
	PARTICLES :: PLAN_SAMPLES * ENSEMBLE_SIZE

	states := make([]Observation, PARTICLES,                  context.temp_allocator)
	alive  := make([]bool,        PARTICLES,                  context.temp_allocator)
	scores := make([]f32,         PARTICLES,                  context.temp_allocator)
	inputs := make([]f32,         PLAN_SAMPLES * MODEL_INPUT, context.temp_allocator)
	deltas := make([]f32,         PLAN_SAMPLES * OBS_SIZE,    context.temp_allocator)

	for p in 0 ..< PARTICLES {
		states[p] = observation
		alive[p]  = true
		scores[p] = 0
	}

	discount := f32(1)

	for h in 0 ..< PLAN_HORIZON {
		ml.clear()

		for m in 0 ..< ENSEMBLE_SIZE {
			for n in 0 ..< PLAN_SAMPLES {
				encode(states[n * ENSEMBLE_SIZE + m], sequences[n][h], inputs[n * MODEL_INPUT:][:MODEL_INPUT])
			}

			x          := ml.reshape(ml.tensor(inputs), {PLAN_SAMPLES, MODEL_INPUT})
			prediction := mlp.forward(agent.models[m], x)
			ml.get_data(prediction, deltas)

			for n in 0 ..< PLAN_SAMPLES {
				p := n * ENSEMBLE_SIZE + m
				if !alive[p] {
					continue
				}

				for i in 0 ..< OBS_SIZE {
					states[p][i] += deltas[n * OBS_SIZE + i] * agent_delta_deviation(agent, i) + agent.delta_mean[i]
				}

				length := math.sqrt(states[p][2] * states[p][2] + states[p][3] * states[p][3])
				if length > 1e-4 {
					states[p][2] /= length
					states[p][3] /= length
				}

				reward, dead := plan_reward(states[p])
				scores[p]    += discount * reward

				if dead {
					scores[p] -= 40
					alive[p]   = false
				}
			}
		}

		discount *= PLAN_DISCOUNT
	}

	for n in 0 ..< PLAN_SAMPLES {
		mean:    f32
		sq_mean: f32

		for m in 0 ..< ENSEMBLE_SIZE {
			score   := scores[n * ENSEMBLE_SIZE + m]
			mean    += score          / f32(ENSEMBLE_SIZE)
			sq_mean += score * score  / f32(ENSEMBLE_SIZE)
		}

		deviation := math.sqrt(max(sq_mean - mean * mean, 0))
		returns[n] = mean - PESSIMISM * deviation
	}
}

@(require_results)
plan_reward :: proc(state: Observation) -> (reward: f32, dead: bool) {
	cos_angle := state[3]
	spin      := state[4] * W_SCALE

	upright := -cos_angle // 1 upright, -1 hanging
	energy  := ENERGY_SCALE * spin * spin + 0.5 * (1 - cos_angle)

	energy_error := energy - 1

	reward = UPRIGHT_WEIGHT * upright
	reward -= ENERGY_WEIGHT * energy_error * energy_error
	reward -= CENTER_WEIGHT * state[0] * state[0]

	if upright > 0 {
		reward -= SPIN_WEIGHT * upright * state[4] * state[4]
	}

	barrier := max(abs(state[0]) - BARRIER_ONSET, 0)
	reward  -= BARRIER_WEIGHT * barrier * barrier

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
