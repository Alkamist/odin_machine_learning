package agent

import "base:builtin"

import "core:math"
import "core:math/rand"
import "core:slice"
import "core:sync"
import "core:thread"
import "core:time"

import ml  "../../../"
import     "../../../networks/mlp"
import cpu "../../../backends/cpu"

SENSOR_SIZE  :: 5
ACTION_COUNT :: 3

Reward_Proc :: proc(sensor: [SENSOR_SIZE]f32) -> (reward: f32, dead: bool)

MODEL_INPUT :: SENSOR_SIZE + ACTION_COUNT
HIDDEN_SIZE :: 32

ENSEMBLE_SIZE :: 5

PLAN_HORIZON  :: 20
PLAN_SAMPLES  :: 64
PLAN_ELITES   :: 8
PLAN_DISCOUNT :: f32(0.98)

POLICY_SEED_SAMPLES :: 16
POLICY_TEMPERATURE  :: f32(1)

PESSIMISM :: f32(1)

BUFFER_CAPACITY  :: 4096
TRAIN_BATCH_SIZE :: 64
TRAIN_MINIMUM    :: 8
TRAIN_STEPS      :: 24
LEARNING_RATE    :: 3e-3

POLICY_CAPACITY :: 4096
POLICY_MINIMUM  :: 64
POLICY_RATE     :: 1e-3
AGREEMENT_RATE  :: f32(0.01)

WARMUP_DECISIONS :: 24

DECISION_PERIOD      :: f64(0.05)
REFINES_PER_DECISION :: 3
REFINE_INTERVAL      :: TRAIN_STEPS / REFINES_PER_DECISION

CONTEXT_SIZE :: 1024 * 1024 * 256

Sensor :: [SENSOR_SIZE]f32

Transition :: struct {
	sensor: Sensor,
	action: int,
	delta:  Sensor,
}

Policy_Sample :: struct {
	sensor: Sensor,
	action: int,
}

Snapshot :: struct {
	valid:   bool,
	time:    f64,
	sensor:  Sensor,
	applied: int,
	episode: u64,
}

Agent :: struct {
	reward: Reward_Proc,

	worker:  ^thread.Thread,
	running: bool,

	mailbox_mutex: sync.Mutex,
	mailbox:       Snapshot,

	latch:          int,
	decisions:      int,
	agreement_bits: u32,

	models: [ENSEMBLE_SIZE]mlp.Mlp,
	opts:   [ENSEMBLE_SIZE]ml.Optimizer,

	policy:     mlp.Mlp,
	policy_opt: ml.Optimizer,

	buffer:       [BUFFER_CAPACITY]Transition,
	buffer_count: int,
	buffer_next:  int,

	policy_buffer:       [POLICY_CAPACITY]Policy_Sample,
	policy_buffer_count: int,
	policy_buffer_next:  int,

	agreement: f32,

	delta_mean:    Sensor,
	delta_sq_mean: Sensor,
	delta_samples: int,

	plan: [PLAN_HORIZON][ACTION_COUNT]f32,

	previous:           Sensor,
	has_previous:       bool,
	last_episode:       u64,
	next_decision_time: f64,
	train_credit:       int,
	refine_budget:      int,
}

make :: proc(reward: Reward_Proc, allocator := context.allocator) -> (a: ^Agent) {
	a = new(Agent, allocator)
	a.reward = reward
	return
}

destroy :: proc(a: ^Agent, allocator := context.allocator) {
	free(a, allocator)
}

start :: proc(a: ^Agent) {
	sync.atomic_store(&a.running, true)
	a.worker = thread.create_and_start_with_poly_data(a, _run)
}

stop :: proc(a: ^Agent) {
	sync.atomic_store(&a.running, false)
	thread.join(a.worker)
	thread.destroy(a.worker)
}

sense :: proc(a: ^Agent, time: f64, sensor: [SENSOR_SIZE]f32, applied: int, episode: u64) {
	sync.mutex_lock(&a.mailbox_mutex)
	a.mailbox = {valid=true, time=time, sensor=sensor, applied=applied, episode=episode}
	sync.mutex_unlock(&a.mailbox_mutex)
}

@(require_results)
act :: proc(a: ^Agent) -> int {
	return sync.atomic_load(&a.latch)
}

decisions :: proc(a: ^Agent) -> int {
	return sync.atomic_load(&a.decisions)
}

@(require_results)
agreement :: proc(a: ^Agent) -> f32 {
	return transmute(f32)sync.atomic_load(&a.agreement_bits)
}

_run :: proc(a: ^Agent) {
	ctx := cpu.context_create(CONTEXT_SIZE)
	defer cpu.context_destroy(ctx)

	ml.context_scope(ctx)

	for m in 0 ..< ENSEMBLE_SIZE {
		a.models[m] = mlp.make(MODEL_INPUT, HIDDEN_SIZE, SENSOR_SIZE)
		a.opts[m]   = ml.optimizer_make(learning_rate=LEARNING_RATE)
	}
	a.policy     = mlp.make(SENSOR_SIZE, HIDDEN_SIZE, ACTION_COUNT)
	a.policy_opt = ml.optimizer_make(learning_rate=POLICY_RATE)

	defer {
		for m in 0 ..< ENSEMBLE_SIZE {
			ml.optimizer_destroy(&a.opts[m])
			mlp.destroy(a.models[m])
		}
		ml.optimizer_destroy(&a.policy_opt)
		mlp.destroy(a.policy)
	}

	_forget_episode(a)

	for sync.atomic_load(&a.running) {
		defer free_all(context.temp_allocator)

		sync.mutex_lock(&a.mailbox_mutex)
		snapshot := a.mailbox
		sync.mutex_unlock(&a.mailbox_mutex)

		if !snapshot.valid {
			time.sleep(1 * time.Millisecond)
			continue
		}

		if snapshot.episode != a.last_episode {
			_forget_episode(a)
			a.last_episode = snapshot.episode
		}

		if snapshot.time >= a.next_decision_time {
			_decide(a, snapshot)
			a.next_decision_time = snapshot.time + DECISION_PERIOD
		}
		else if a.train_credit > 0 || a.refine_budget > 0 {
			if a.train_credit > 0 {
				_train(a)
				_train_policy(a)
				a.train_credit -= 1
			}
			if a.refine_budget > 0 && a.train_credit % REFINE_INTERVAL == 0 {
				if a.has_previous && a.decisions >= WARMUP_DECISIONS {
					_plan_refine(a, a.previous)
				}
				a.refine_budget -= 1
			}
		}
		else {
			time.sleep(500 * time.Microsecond)
		}
	}
}

_forget_episode :: proc(a: ^Agent) {
	a.has_previous = false
	for h in 0 ..< PLAN_HORIZON {
		for action_index in 0 ..< ACTION_COUNT {
			a.plan[h][action_index] = 1.0 / f32(ACTION_COUNT)
		}
	}
}

_decide :: proc(a: ^Agent, snapshot: Snapshot) {
	if a.has_previous {
		transition := Transition{sensor=a.previous, action=snapshot.applied}
		for i in 0 ..< SENSOR_SIZE {
			transition.delta[i] = snapshot.sensor[i] - a.previous[i]
		}
		_remember(a, transition)
	}

	action: int
	if a.decisions < WARMUP_DECISIONS {
		action = rand.int_max(ACTION_COUNT)
	}
	else {
		_plan_shift(a)
		_plan_refine(a, snapshot.sensor)
		action = _plan_action(a)

		if a.policy_buffer_count >= POLICY_MINIMUM {
			match := f32(0)
			if _policy_action(a, snapshot.sensor) == action {
				match = 1
			}
			a.agreement += (match - a.agreement) * AGREEMENT_RATE
			sync.atomic_store(&a.agreement_bits, transmute(u32)a.agreement)
		}

		_remember_policy(a, snapshot.sensor, action)
	}

	sync.atomic_store(&a.latch, action)

	a.previous      = snapshot.sensor
	a.has_previous  = true
	a.train_credit += TRAIN_STEPS
	a.refine_budget = REFINES_PER_DECISION - 1

	sync.atomic_store(&a.decisions, a.decisions + 1)
}

_remember :: proc(a: ^Agent, transition: Transition) {
	a.buffer[a.buffer_next] = transition
	a.buffer_next = (a.buffer_next + 1) % BUFFER_CAPACITY
	if a.buffer_count < BUFFER_CAPACITY {
		a.buffer_count += 1
	}

	a.delta_samples += 1
	rate := 1.0 / f32(a.delta_samples)
	for i in 0 ..< SENSOR_SIZE {
		d := transition.delta[i]
		a.delta_mean[i]    += (d     - a.delta_mean[i])    * rate
		a.delta_sq_mean[i] += (d * d - a.delta_sq_mean[i]) * rate
	}
}

_remember_policy :: proc(a: ^Agent, sensor: Sensor, action: int) {
	a.policy_buffer[a.policy_buffer_next] = {sensor=sensor, action=action}
	a.policy_buffer_next = (a.policy_buffer_next + 1) % POLICY_CAPACITY
	if a.policy_buffer_count < POLICY_CAPACITY {
		a.policy_buffer_count += 1
	}
}

@(require_results)
_policy_action :: proc(a: ^Agent, sensor: Sensor) -> int {
	input  := builtin.make([]f32, SENSOR_SIZE,  context.temp_allocator)
	logits := builtin.make([]f32, ACTION_COUNT, context.temp_allocator)

	for i in 0 ..< SENSOR_SIZE {
		input[i] = sensor[i]
	}

	ml.clear()

	x      := ml.tensor(input, []int{1, SENSOR_SIZE})
	output := mlp.forward(a.policy, x)
	ml.get_data(output, logits)

	best := 0
	for action_index in 1 ..< ACTION_COUNT {
		if logits[action_index] > logits[best] {
			best = action_index
		}
	}
	return best
}

_train_policy :: proc(a: ^Agent) {
	if a.policy_buffer_count < POLICY_MINIMUM {
		return
	}

	inputs  := builtin.make([]f32, TRAIN_BATCH_SIZE * SENSOR_SIZE, context.temp_allocator)
	targets := builtin.make([]int, TRAIN_BATCH_SIZE,               context.temp_allocator)

	ml.clear(training=true)

	for b in 0 ..< TRAIN_BATCH_SIZE {
		sample := a.policy_buffer[rand.int_max(a.policy_buffer_count)]

		for i in 0 ..< SENSOR_SIZE {
			inputs[b * SENSOR_SIZE + i] = sample.sensor[i]
		}
		targets[b] = sample.action
	}

	x      := ml.tensor(inputs, []int{TRAIN_BATCH_SIZE, SENSOR_SIZE})
	logits := mlp.forward(a.policy, x)
	loss   := ml.mean(ml.cross_entropy(logits, targets))

	ml.backward(loss)

	if ml.optimizer_step(&a.policy_opt) {
		mlp.update(&a.policy_opt, a.policy)
	}
}

@(require_results)
_sample_index :: proc(a: ^Agent) -> int {
	if rand.float32() < 0.5 {
		return rand.int_max(a.buffer_count)
	}

	recent := max(a.buffer_count / 4, 1)
	age    := rand.int_max(recent)
	return (a.buffer_next - 1 - age + BUFFER_CAPACITY) % BUFFER_CAPACITY
}

@(require_results)
_delta_deviation :: proc(a: ^Agent, i: int) -> f32 {
	variance := a.delta_sq_mean[i] - a.delta_mean[i] * a.delta_mean[i]
	return max(math.sqrt(max(variance, 0)), 1e-4)
}

_encode :: proc(sensor: Sensor, action: int, dst: []f32) {
	for i in 0 ..< SENSOR_SIZE {
		dst[i] = sensor[i]
	}
	for i in 0 ..< ACTION_COUNT {
		dst[SENSOR_SIZE + i] = 0
	}
	dst[SENSOR_SIZE + action] = 1
}

_train :: proc(a: ^Agent) {
	if a.buffer_count < TRAIN_MINIMUM {
		return
	}

	inputs  := builtin.make([]f32, TRAIN_BATCH_SIZE * MODEL_INPUT, context.temp_allocator)
	targets := builtin.make([]f32, TRAIN_BATCH_SIZE * SENSOR_SIZE, context.temp_allocator)

	ml.clear(training=true)

	total: ml.Tensor

	for m in 0 ..< ENSEMBLE_SIZE {
		for b in 0 ..< TRAIN_BATCH_SIZE {
			transition := a.buffer[_sample_index(a)]

			_encode(transition.sensor, transition.action, inputs[b * MODEL_INPUT:][:MODEL_INPUT])

			for i in 0 ..< SENSOR_SIZE {
				targets[b * SENSOR_SIZE + i] = (transition.delta[i] - a.delta_mean[i]) / _delta_deviation(a, i)
			}
		}

		x          := ml.tensor(inputs,  []int{TRAIN_BATCH_SIZE, MODEL_INPUT})
		y          := ml.tensor(targets, []int{TRAIN_BATCH_SIZE, SENSOR_SIZE})
		prediction := mlp.forward(a.models[m], x)
		loss       := ml.mean(ml.mean_squared_error(prediction, y))

		total = loss if m == 0 else ml.add(total, loss)
	}

	ml.backward(total)

	for m in 0 ..< ENSEMBLE_SIZE {
		if ml.optimizer_step(&a.opts[m]) {
			mlp.update(&a.opts[m], a.models[m])
		}
	}
}

_plan_shift :: proc(a: ^Agent) {
	for h in 0 ..< PLAN_HORIZON - 1 {
		a.plan[h] = a.plan[h + 1]
	}
	for action_index in 0 ..< ACTION_COUNT {
		a.plan[PLAN_HORIZON - 1][action_index] = 1.0 / f32(ACTION_COUNT)
	}
}

@(require_results)
_plan_action :: proc(a: ^Agent) -> int {
	best := 0
	for action_index in 1 ..< ACTION_COUNT {
		if a.plan[0][action_index] > a.plan[0][best] {
			best = action_index
		}
	}
	return best
}

_plan_refine :: proc(a: ^Agent, sensor: Sensor) {
	sequences := builtin.make([][PLAN_HORIZON]int, PLAN_SAMPLES, context.temp_allocator)
	returns   := builtin.make([]f32,               PLAN_SAMPLES, context.temp_allocator)
	order     := builtin.make([]int,               PLAN_SAMPLES, context.temp_allocator)

	seeded := 0
	if a.policy_buffer_count >= POLICY_MINIMUM {
		seeded = POLICY_SEED_SAMPLES
		_policy_seed(a, sensor, sequences[:seeded])
	}

	for n in seeded ..< PLAN_SAMPLES {
		for h in 0 ..< PLAN_HORIZON {
			sequences[n][h] = _sample_action(a.plan[h])
		}
	}

	_rollout(a, sensor, sequences, returns)

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
			counts[sequences[order[e]][h]] += 1.0 / f32(PLAN_ELITES)
		}
		for action_index in 0 ..< ACTION_COUNT {
			a.plan[h][action_index] = 0.5 * a.plan[h][action_index] + 0.5 * counts[action_index]
			a.plan[h][action_index] = max(a.plan[h][action_index], 0.02)
		}
	}
}

_policy_seed :: proc(a: ^Agent, sensor: Sensor, sequences: [][PLAN_HORIZON]int) {
	count := len(sequences)

	states       := builtin.make([]Sensor, count,                context.temp_allocator)
	inputs       := builtin.make([]f32,    count * MODEL_INPUT,  context.temp_allocator)
	observations := builtin.make([]f32,    count * SENSOR_SIZE,  context.temp_allocator)
	logits       := builtin.make([]f32,    count * ACTION_COUNT, context.temp_allocator)
	deltas       := builtin.make([]f32,    count * SENSOR_SIZE,  context.temp_allocator)

	for p in 0 ..< count {
		states[p] = sensor
	}

	for h in 0 ..< PLAN_HORIZON {
		ml.clear()

		for p in 0 ..< count {
			for i in 0 ..< SENSOR_SIZE {
				observations[p * SENSOR_SIZE + i] = states[p][i]
			}
		}

		x := ml.tensor(observations, []int{count, SENSOR_SIZE})
		ml.get_data(mlp.forward(a.policy, x), logits)

		for p in 0 ..< count {
			action := _sample_logits(logits[p * ACTION_COUNT:][:ACTION_COUNT])
			sequences[p][h] = action
			_encode(states[p], action, inputs[p * MODEL_INPUT:][:MODEL_INPUT])
		}

		model := rand.int_max(ENSEMBLE_SIZE)
		y     := ml.tensor(inputs, []int{count, MODEL_INPUT})
		ml.get_data(mlp.forward(a.models[model], y), deltas)

		for p in 0 ..< count {
			_apply_delta(a, &states[p], deltas[p * SENSOR_SIZE:][:SENSOR_SIZE])
		}
	}
}

_apply_delta :: proc(a: ^Agent, state: ^Sensor, deltas: []f32) {
	for i in 0 ..< SENSOR_SIZE {
		state[i] += deltas[i] * _delta_deviation(a, i) + a.delta_mean[i]
	}

	length := math.sqrt(state[2] * state[2] + state[3] * state[3])
	if length > 1e-4 {
		state[2] /= length
		state[3] /= length
	}
}

@(require_results)
_sample_logits :: proc(logits: []f32) -> int {
	highest := logits[0]
	for i in 1 ..< ACTION_COUNT {
		highest = max(highest, logits[i])
	}

	probabilities: [ACTION_COUNT]f32
	for i in 0 ..< ACTION_COUNT {
		probabilities[i] = math.exp((logits[i] - highest) / POLICY_TEMPERATURE)
	}
	return _sample_action(probabilities)
}

_rollout :: proc(a: ^Agent, sensor: Sensor, sequences: [][PLAN_HORIZON]int, returns: []f32) {
	PARTICLES :: PLAN_SAMPLES * ENSEMBLE_SIZE

	states := builtin.make([]Sensor, PARTICLES,                  context.temp_allocator)
	alive  := builtin.make([]bool,   PARTICLES,                  context.temp_allocator)
	scores := builtin.make([]f32,    PARTICLES,                  context.temp_allocator)
	inputs := builtin.make([]f32,    PLAN_SAMPLES * MODEL_INPUT, context.temp_allocator)
	deltas := builtin.make([]f32,    PLAN_SAMPLES * SENSOR_SIZE, context.temp_allocator)

	for p in 0 ..< PARTICLES {
		states[p] = sensor
		alive[p]  = true
		scores[p] = 0
	}

	discount := f32(1)

	for h in 0 ..< PLAN_HORIZON {
		ml.clear()

		for m in 0 ..< ENSEMBLE_SIZE {
			for n in 0 ..< PLAN_SAMPLES {
				_encode(states[n * ENSEMBLE_SIZE + m], sequences[n][h], inputs[n * MODEL_INPUT:][:MODEL_INPUT])
			}

			x          := ml.tensor(inputs, []int{PLAN_SAMPLES, MODEL_INPUT})
			prediction := mlp.forward(a.models[m], x)
			ml.get_data(prediction, deltas)

			for n in 0 ..< PLAN_SAMPLES {
				p := n * ENSEMBLE_SIZE + m
				if !alive[p] {
					continue
				}

				_apply_delta(a, &states[p], deltas[n * SENSOR_SIZE:][:SENSOR_SIZE])

				reward, dead := a.reward(states[p])
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
_sample_action :: proc(probabilities: [ACTION_COUNT]f32) -> int {
	total: f32
	for p in probabilities {
		total += p
	}

	target := rand.float32() * total
	sum:    f32

	for action_index in 0 ..< ACTION_COUNT {
		sum += probabilities[action_index]
		if target <= sum {
			return action_index
		}
	}
	return ACTION_COUNT - 1
}
