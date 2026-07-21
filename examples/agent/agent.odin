package agent

import "base:builtin"

import "core:math"
import "core:math/rand"
import "core:slice"
import "core:sync"
import "core:thread"
import "core:time"

import ml  "../../"
import     "../../networks/mlp"
import cpu "../../backends/cpu"

import "../world"

SENSOR_SIZE  :: world.SENSOR_SIZE
BINARY_COUNT :: world.BINARY_COUNT
ANALOG_COUNT :: world.ANALOG_COUNT
ACTION_DIM   :: world.ACTION_DIM

Sensor :: world.Sensor
Action :: world.Action

Reward_Proc :: world.Reward_Proc

MODEL_INPUT :: SENSOR_SIZE + ACTION_DIM
POLICY_OUT  :: BINARY_COUNT + ANALOG_COUNT
HIDDEN_SIZE :: 32

ENSEMBLE_SIZE :: 5

PLAN_HORIZON  :: 20
PLAN_SAMPLES  :: 64
PLAN_ELITES   :: 8
PLAN_DISCOUNT :: f32(0.98)

PLAN_INIT_STD :: f32(0.5)
PLAN_MIN_STD  :: f32(0.05)

POLICY_SEED_SAMPLES :: 16

ANALOG_STD :: f32(0.3)

PESSIMISM :: f32(0)

BOOTSTRAP_WEIGHT :: f32(0)

VALUE_ENSEMBLE :: 2
TAU            :: f32(0.01)
DEATH_PENALTY  :: f32(40)

BUFFER_CAPACITY  :: 4096
TRAIN_BATCH_SIZE :: 64
TRAIN_MINIMUM    :: 8
TRAIN_STEPS      :: 24
TRAIN_BACKLOG    :: TRAIN_STEPS * 2
LEARNING_RATE    :: 3e-3

POLICY_RATE       :: 1e-3
POLICY_MATCH_RATE :: f32(0.01)

VALUE_FIT_HORIZON :: 150
VALUE_FIT_SAMPLES :: 512
VALUE_FIT_EPSILON :: f32(1e-3)

WARMUP_DECISIONS :: 24

DECISION_PERIOD      :: f64(0.05)
REFINES_PER_DECISION :: 3
REFINE_INTERVAL      :: TRAIN_STEPS / REFINES_PER_DECISION

CONTEXT_SIZE :: 1024 * 1024 * 256

Transition :: struct {
	sensor:  Sensor,
	action:  Action,
	delta:   Sensor,
	reward:  f32,
	dead:    bool,
	planned: bool,
}

Snapshot :: struct {
	valid:   bool,
	time:    f64,
	sensor:  Sensor,
	applied: Action,
	episode: u64,
}

Plan_Step :: struct {
	analog_mean: [ANALOG_COUNT]f32,
	analog_std:  [ANALOG_COUNT]f32,
	binary_prob: [BINARY_COUNT]f32,
}

Agent :: struct {
	reward: Reward_Proc,

	worker:  ^thread.Thread,
	running: bool,

	mailbox_mutex: sync.Mutex,
	mailbox:       Snapshot,

	action_mutex:      sync.Mutex,
	latch:             Action,
	decisions:         int,
	policy_match_bits: u32,

	models: [ENSEMBLE_SIZE]mlp.Mlp,
	opts:   [ENSEMBLE_SIZE]ml.Optimizer,

	policy:     mlp.Mlp,
	policy_opt: ml.Optimizer,

	values:        [VALUE_ENSEMBLE]mlp.Mlp,
	value_targets: [VALUE_ENSEMBLE]mlp.Mlp,
	value_opts:    [VALUE_ENSEMBLE]ml.Optimizer,

	buffer:       [BUFFER_CAPACITY]Transition,
	buffer_count: int,
	buffer_next:  int,

	policy_match: f32,

	delta_mean:    Sensor,
	delta_sq_mean: Sensor,
	delta_samples: int,

	plan: [PLAN_HORIZON]Plan_Step,

	previous:           Sensor,
	previous_planned:   bool,
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

sense :: proc(a: ^Agent, time: f64, sensor: [SENSOR_SIZE]f32, applied: Action, episode: u64) {
	sync.mutex_lock(&a.mailbox_mutex)
	a.mailbox = {valid=true, time=time, sensor=sensor, applied=applied, episode=episode}
	sync.mutex_unlock(&a.mailbox_mutex)
}

@(require_results)
act :: proc(a: ^Agent) -> (action: Action) {
	sync.mutex_lock(&a.action_mutex)
	action = a.latch
	sync.mutex_unlock(&a.action_mutex)
	return
}

decisions :: proc(a: ^Agent) -> int {
	return sync.atomic_load(&a.decisions)
}

@(require_results)
policy_match :: proc(a: ^Agent) -> f32 {
	return transmute(f32)sync.atomic_load(&a.policy_match_bits)
}

boot :: proc(a: ^Agent) {
	for m in 0 ..< ENSEMBLE_SIZE {
		a.models[m] = mlp.make(MODEL_INPUT, HIDDEN_SIZE, SENSOR_SIZE)
		a.opts[m]   = ml.optimizer_make(learning_rate=LEARNING_RATE)
	}
	a.policy     = mlp.make(SENSOR_SIZE, HIDDEN_SIZE, POLICY_OUT)
	a.policy_opt = ml.optimizer_make(learning_rate=POLICY_RATE)

	for v in 0 ..< VALUE_ENSEMBLE {
		a.values[v]        = mlp.make(SENSOR_SIZE + ACTION_DIM, HIDDEN_SIZE, 1)
		a.value_targets[v] = mlp.make(SENSOR_SIZE + ACTION_DIM, HIDDEN_SIZE, 1)
		a.value_opts[v]    = ml.optimizer_make(learning_rate=LEARNING_RATE)
		mlp.copy(a.value_targets[v], a.values[v])
	}

	_forget_episode(a)
}

shutdown :: proc(a: ^Agent) {
	for m in 0 ..< ENSEMBLE_SIZE {
		ml.optimizer_destroy(&a.opts[m])
		mlp.destroy(a.models[m])
	}
	ml.optimizer_destroy(&a.policy_opt)
	mlp.destroy(a.policy)
	for v in 0 ..< VALUE_ENSEMBLE {
		ml.optimizer_destroy(&a.value_opts[v])
		mlp.destroy(a.values[v])
		mlp.destroy(a.value_targets[v])
	}
}

_step_brain :: proc(a: ^Agent, snapshot: Snapshot) -> (idle: bool) {
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
			_train_value(a)
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
		idle = true
	}

	return
}

drive :: proc(a: ^Agent, time: f64, sensor: [SENSOR_SIZE]f32, applied: Action, episode: u64) {
	snapshot := Snapshot{valid=true, time=time, sensor=sensor, applied=applied, episode=episode}
	for {
		idle := _step_brain(a, snapshot)
		free_all(context.temp_allocator)
		if idle {
			break
		}
	}
}

_run :: proc(a: ^Agent) {
	ctx := cpu.context_create(CONTEXT_SIZE)
	defer cpu.context_destroy(ctx)

	ml.context_scope(ctx)

	boot(a)
	defer shutdown(a)

	for sync.atomic_load(&a.running) {
		free_all(context.temp_allocator)

		sync.mutex_lock(&a.mailbox_mutex)
		snapshot := a.mailbox
		sync.mutex_unlock(&a.mailbox_mutex)

		if !snapshot.valid {
			time.sleep(1 * time.Millisecond)
			continue
		}

		if _step_brain(a, snapshot) {
			time.sleep(500 * time.Microsecond)
		}
	}
}

_forget_episode :: proc(a: ^Agent) {
	a.has_previous = false
	for h in 0 ..< PLAN_HORIZON {
		a.plan[h] = _plan_init_step()
	}
}

@(require_results)
_plan_init_step :: proc() -> (step: Plan_Step) {
	for k in 0 ..< ANALOG_COUNT {
		step.analog_mean[k] = 0
		step.analog_std[k]  = PLAN_INIT_STD
	}
	for j := 0; j < BINARY_COUNT; j += 1 {
		step.binary_prob[j] = 0.5
	}
	return
}

_decide :: proc(a: ^Agent, snapshot: Snapshot) {
	if a.has_previous {
		transition_reward, transition_dead := a.reward(snapshot.sensor)
		transition := Transition{sensor=a.previous, action=snapshot.applied, reward=transition_reward, dead=transition_dead, planned=a.previous_planned}
		for i in 0 ..< SENSOR_SIZE {
			transition.delta[i] = snapshot.sensor[i] - a.previous[i]
		}
		_remember(a, transition)
	}

	action: Action
	used_planner := a.decisions >= WARMUP_DECISIONS
	if a.decisions < WARMUP_DECISIONS {
		for j := 0; j < BINARY_COUNT; j += 1 {
			action[j] = 1 if rand.float32() < 0.5 else 0
		}
		for k in 0 ..< ANALOG_COUNT {
			action[BINARY_COUNT + k] = rand.float32() * 2 - 1
		}
	}
	else {
		_plan_shift(a)
		_plan_refine(a, snapshot.sensor)
		action = _plan_action(a)

		policy_action := _policy_mean(a, snapshot.sensor)
		match := f32(1)
		for j := 0; j < BINARY_COUNT; j += 1 {
			if action[j] != policy_action[j] {
				match = 0
			}
		}
		for k in 0 ..< ANALOG_COUNT {
			if abs(action[BINARY_COUNT + k] - policy_action[BINARY_COUNT + k]) > 0.25 {
				match = 0
			}
		}
		a.policy_match += (match - a.policy_match) * POLICY_MATCH_RATE
		sync.atomic_store(&a.policy_match_bits, transmute(u32)a.policy_match)
	}

	sync.mutex_lock(&a.action_mutex)
	a.latch = action
	sync.mutex_unlock(&a.action_mutex)

	a.previous         = snapshot.sensor
	a.previous_planned = used_planner
	a.has_previous     = true
	a.train_credit     = min(a.train_credit + TRAIN_STEPS, TRAIN_BACKLOG)
	a.refine_budget    = REFINES_PER_DECISION - 1

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

@(require_results)
_sigmoid :: proc(x: f32) -> f32 {
	return 1.0 / (1.0 + math.exp(-x))
}

@(require_results)
_randn :: proc() -> f32 {
	return rand.float32_normal(0, 1)
}

@(require_results)
_sample_row :: proc(row: []f32) -> (action: Action) {
	for j := 0; j < BINARY_COUNT; j += 1 {
		action[j] = 1 if rand.float32() < _sigmoid(row[j]) else 0
	}
	for k in 0 ..< ANALOG_COUNT {
		mean := row[BINARY_COUNT + k]
		action[BINARY_COUNT + k] = math.tanh(mean + ANALOG_STD * _randn())
	}
	return
}

@(require_results)
_mean_row :: proc(row: []f32) -> (action: Action) {
	for j := 0; j < BINARY_COUNT; j += 1 {
		action[j] = 1 if _sigmoid(row[j]) > 0.5 else 0
	}
	for k in 0 ..< ANALOG_COUNT {
		action[BINARY_COUNT + k] = math.tanh(row[BINARY_COUNT + k])
	}
	return
}

@(require_results)
_policy_sample :: proc(a: ^Agent, sensor: Sensor) -> Action {
	return _sample_row(_policy_row(a, sensor))
}

@(require_results)
_policy_mean :: proc(a: ^Agent, sensor: Sensor) -> Action {
	return _mean_row(_policy_row(a, sensor))
}

@(require_results)
_policy_row :: proc(a: ^Agent, sensor: Sensor) -> []f32 {
	input := builtin.make([]f32, SENSOR_SIZE, context.temp_allocator)
	row   := builtin.make([]f32, POLICY_OUT,  context.temp_allocator)

	for i in 0 ..< SENSOR_SIZE {
		input[i] = sensor[i]
	}

	if ml.pass() {
		x      := ml.tensor(input, []int{1, SENSOR_SIZE})
		output := mlp.forward(a.policy, x)
		ml.get_data(output, row)
	}
	return row
}

_train_value :: proc(a: ^Agent) {
	if a.buffer_count < TRAIN_MINIMUM {
		return
	}

	states     := builtin.make([]f32,    TRAIN_BATCH_SIZE * SENSOR_SIZE, context.temp_allocator)
	successors := builtin.make([]f32,    TRAIN_BATCH_SIZE * SENSOR_SIZE, context.temp_allocator)
	actions    := builtin.make([]Action, TRAIN_BATCH_SIZE,               context.temp_allocator)
	rewards    := builtin.make([]f32,    TRAIN_BATCH_SIZE,               context.temp_allocator)
	deaths     := builtin.make([]bool,   TRAIN_BATCH_SIZE,               context.temp_allocator)

	for b in 0 ..< TRAIN_BATCH_SIZE {
		transition := a.buffer[_sample_index(a)]
		for i in 0 ..< SENSOR_SIZE {
			states[b * SENSOR_SIZE + i]     = transition.sensor[i]
			successors[b * SENSOR_SIZE + i] = transition.sensor[i] + transition.delta[i]
		}
		actions[b] = transition.action
		rewards[b] = transition.reward
		deaths[b]  = transition.dead
	}

	CRITIC_INPUT :: SENSOR_SIZE + ACTION_DIM

	successor_rows := builtin.make([]f32, TRAIN_BATCH_SIZE * POLICY_OUT,   context.temp_allocator)
	critic_next    := builtin.make([]f32, TRAIN_BATCH_SIZE * CRITIC_INPUT, context.temp_allocator)
	targets        := builtin.make([]f32, TRAIN_BATCH_SIZE,               context.temp_allocator)

	target_q: [VALUE_ENSEMBLE][]f32
	for v in 0 ..< VALUE_ENSEMBLE {
		target_q[v] = builtin.make([]f32, TRAIN_BATCH_SIZE, context.temp_allocator)
	}

	if ml.pass() {
		successor_tensor := ml.tensor(successors, []int{TRAIN_BATCH_SIZE, SENSOR_SIZE})
		ml.get_data(mlp.forward(a.policy, successor_tensor), successor_rows)

		for b in 0 ..< TRAIN_BATCH_SIZE {
			action := _sample_row(successor_rows[b * POLICY_OUT:][:POLICY_OUT])
			base   := b * CRITIC_INPUT
			for i in 0 ..< SENSOR_SIZE {
				critic_next[base + i] = successors[b * SENSOR_SIZE + i]
			}
			for d in 0 ..< ACTION_DIM {
				critic_next[base + SENSOR_SIZE + d] = action[d]
			}
		}

		critic_next_tensor := ml.tensor(critic_next, []int{TRAIN_BATCH_SIZE, CRITIC_INPUT})
		for v in 0 ..< VALUE_ENSEMBLE {
			ml.get_data(mlp.forward(a.value_targets[v], critic_next_tensor), target_q[v])
		}
	}

	for b in 0 ..< TRAIN_BATCH_SIZE {
		if deaths[b] {
			targets[b] = rewards[b] - DEATH_PENALTY
			continue
		}

		q := target_q[0][b]
		for v in 1 ..< VALUE_ENSEMBLE {
			q = min(q, target_q[v][b])
		}
		targets[b] = rewards[b] + PLAN_DISCOUNT * q
	}

	critic_now := builtin.make([]f32, TRAIN_BATCH_SIZE * CRITIC_INPUT, context.temp_allocator)
	for b in 0 ..< TRAIN_BATCH_SIZE {
		base := b * CRITIC_INPUT
		for i in 0 ..< SENSOR_SIZE {
			critic_now[base + i] = states[b * SENSOR_SIZE + i]
		}
		for d in 0 ..< ACTION_DIM {
			critic_now[base + SENSOR_SIZE + d] = actions[b][d]
		}
	}

	if ml.pass(training=true) {
		x        := ml.tensor(critic_now, []int{TRAIN_BATCH_SIZE, CRITIC_INPUT})
		y        := ml.tensor(targets,    []int{TRAIN_BATCH_SIZE})
		total: ml.Tensor

		for v in 0 ..< VALUE_ENSEMBLE {
			q_values   := mlp.forward(a.values[v], x)
			prediction := ml.reshape(q_values, []int{TRAIN_BATCH_SIZE})
			loss       := ml.mean(ml.mean_squared_error(prediction, y))

			total = loss if v == 0 else ml.add(total, loss)
		}

		ml.backward(total)

		for v in 0 ..< VALUE_ENSEMBLE {
			if ml.optimizer_step(&a.value_opts[v]) {
				mlp.update(&a.value_opts[v], a.values[v])
			}
		}

		for v in 0 ..< VALUE_ENSEMBLE {
			for layer, layer_index in a.values[v].layers {
				ml.lerp_assign(a.value_targets[v].layers[layer_index].weight, layer.weight, TAU)
				ml.lerp_assign(a.value_targets[v].layers[layer_index].bias,   layer.bias,   TAU)
			}
		}
	}
}

@(require_results)
_buffer_index :: proc(a: ^Agent, position: int) -> int {
	return (a.buffer_next - a.buffer_count + position + BUFFER_CAPACITY) % BUFFER_CAPACITY
}

@(require_results)
_continues :: proc(previous, next: Transition) -> bool {
	for i in 0 ..< SENSOR_SIZE {
		if abs(previous.sensor[i] + previous.delta[i] - next.sensor[i]) > VALUE_FIT_EPSILON {
			return false
		}
	}
	return true
}

@(require_results)
_observed_return :: proc(a: ^Agent, position: int) -> (observed: f32, ok: bool) {
	discount := f32(1)
	cursor   := position

	for _ in 0 ..< VALUE_FIT_HORIZON {
		transition := a.buffer[_buffer_index(a, cursor)]
		observed   += discount * transition.reward

		if transition.dead {
			observed -= discount * DEATH_PENALTY
			return observed, true
		}

		discount *= PLAN_DISCOUNT
		cursor   += 1

		if cursor >= a.buffer_count {
			return 0, false
		}
		if !_continues(transition, a.buffer[_buffer_index(a, cursor)]) {
			return 0, false
		}
	}

	return observed, true
}

@(require_results)
_correlation :: proc(x, y: []f32) -> f32 {
	count := f32(len(x))

	mean_x, mean_y: f32
	for i in 0 ..< len(x) {
		mean_x += x[i] / count
		mean_y += y[i] / count
	}

	covariance, variance_x, variance_y: f32
	for i in 0 ..< len(x) {
		difference_x := x[i] - mean_x
		difference_y := y[i] - mean_y

		covariance += difference_x * difference_y
		variance_x += difference_x * difference_x
		variance_y += difference_y * difference_y
	}

	spread := math.sqrt(variance_x * variance_y)
	if spread < 1e-12 {
		return 0
	}
	return covariance / spread
}

@(require_results)
value_fit :: proc(a: ^Agent) -> (correlation: f32, samples: int) {
	CRITIC_INPUT :: SENSOR_SIZE + ACTION_DIM

	stride := max((a.buffer_count + VALUE_FIT_SAMPLES - 1) / VALUE_FIT_SAMPLES, 1)

	inputs  := builtin.make([]f32, VALUE_FIT_SAMPLES * CRITIC_INPUT, context.temp_allocator)
	returns := builtin.make([]f32, VALUE_FIT_SAMPLES,                context.temp_allocator)

	for position := 0; position < a.buffer_count && samples < VALUE_FIT_SAMPLES; position += stride {
		observed, complete := _observed_return(a, position)
		if !complete {
			continue
		}

		transition := a.buffer[_buffer_index(a, position)]
		_encode(transition.sensor, transition.action, inputs[samples * CRITIC_INPUT:][:CRITIC_INPUT])
		returns[samples] = observed
		samples += 1
	}

	if samples < 2 {
		samples = 0
		return
	}

	estimates: [VALUE_ENSEMBLE][]f32
	for v in 0 ..< VALUE_ENSEMBLE {
		estimates[v] = builtin.make([]f32, samples, context.temp_allocator)
	}

	if ml.pass() {
		x := ml.tensor(inputs[:samples * CRITIC_INPUT], []int{samples, CRITIC_INPUT})
		for v in 0 ..< VALUE_ENSEMBLE {
			ml.get_data(mlp.forward(a.values[v], x), estimates[v])
		}
	}

	predictions := builtin.make([]f32, samples, context.temp_allocator)
	for s in 0 ..< samples {
		q := estimates[0][s]
		for v in 1 ..< VALUE_ENSEMBLE {
			q = min(q, estimates[v][s])
		}
		predictions[s] = q
	}

	correlation = _correlation(predictions, returns[:samples])
	return
}

_train_policy :: proc(a: ^Agent) {
	if a.buffer_count < TRAIN_MINIMUM {
		return
	}

	states         := builtin.make([]f32, TRAIN_BATCH_SIZE * SENSOR_SIZE,  context.temp_allocator)
	analog_targets := builtin.make([]f32, TRAIN_BATCH_SIZE * ANALOG_COUNT, context.temp_allocator)
	binary_targets := builtin.make([]f32, TRAIN_BATCH_SIZE * BINARY_COUNT, context.temp_allocator)

	count    := 0
	attempts := 0
	for count < TRAIN_BATCH_SIZE && attempts < TRAIN_BATCH_SIZE * 8 {
		attempts += 1
		transition := a.buffer[_sample_index(a)]
		if !transition.planned {
			continue
		}
		for i in 0 ..< SENSOR_SIZE {
			states[count * SENSOR_SIZE + i] = transition.sensor[i]
		}
		for j := 0; j < BINARY_COUNT; j += 1 {
			binary_targets[count * BINARY_COUNT + j] = transition.action[j]
		}
		for k in 0 ..< ANALOG_COUNT {
			analog_targets[count * ANALOG_COUNT + k] = transition.action[BINARY_COUNT + k]
		}
		count += 1
	}

	if count == 0 {
		return
	}

	if ml.pass(training=true) {
		state_tensor := ml.tensor(states[:count * SENSOR_SIZE], []int{count, SENSOR_SIZE})
		policy_out   := mlp.forward(a.policy, state_tensor)

		means      := ml.slice_trailing(policy_out, BINARY_COUNT, POLICY_OUT)
		analog_tgt := ml.tensor(analog_targets[:count * ANALOG_COUNT], []int{count, ANALOG_COUNT})
		loss       := ml.mean(ml.mean_squared_error(ml.tanh(means), analog_tgt))

		when BINARY_COUNT > 0 {
			logits     := ml.slice_trailing(policy_out, 0, BINARY_COUNT)
			binary_tgt := ml.tensor(binary_targets[:count * BINARY_COUNT], []int{count, BINARY_COUNT})
			loss = ml.add(loss, ml.mean(ml.mean_squared_error(ml.sigmoid(logits), binary_tgt)))
		}

		ml.backward(loss)

		if ml.optimizer_step(&a.policy_opt) {
			mlp.update(&a.policy_opt, a.policy)
		}
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

_encode :: proc(sensor: Sensor, action: Action, dst: []f32) {
	for i in 0 ..< SENSOR_SIZE {
		dst[i] = sensor[i]
	}
	for d in 0 ..< ACTION_DIM {
		dst[SENSOR_SIZE + d] = action[d]
	}
}

_train :: proc(a: ^Agent) {
	if a.buffer_count < TRAIN_MINIMUM {
		return
	}

	inputs  := builtin.make([]f32, TRAIN_BATCH_SIZE * MODEL_INPUT, context.temp_allocator)
	targets := builtin.make([]f32, TRAIN_BATCH_SIZE * SENSOR_SIZE, context.temp_allocator)

	if ml.pass(training=true) {
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
}

_plan_shift :: proc(a: ^Agent) {
	for h in 0 ..< PLAN_HORIZON - 1 {
		a.plan[h] = a.plan[h + 1]
	}
	a.plan[PLAN_HORIZON - 1] = _plan_init_step()
}

@(require_results)
_plan_action :: proc(a: ^Agent) -> (action: Action) {
	for j := 0; j < BINARY_COUNT; j += 1 {
		action[j] = 1 if a.plan[0].binary_prob[j] > 0.5 else 0
	}
	for k in 0 ..< ANALOG_COUNT {
		action[BINARY_COUNT + k] = clamp(a.plan[0].analog_mean[k], -1, 1)
	}
	return
}

@(require_results)
_sample_plan_step :: proc(step: Plan_Step) -> (action: Action) {
	for j := 0; j < BINARY_COUNT; j += 1 {
		action[j] = 1 if rand.float32() < step.binary_prob[j] else 0
	}
	for k in 0 ..< ANALOG_COUNT {
		action[BINARY_COUNT + k] = clamp(step.analog_mean[k] + step.analog_std[k] * _randn(), -1, 1)
	}
	return
}

_plan_refine :: proc(a: ^Agent, sensor: Sensor) {
	sequences := builtin.make([][PLAN_HORIZON]Action, PLAN_SAMPLES, context.temp_allocator)
	returns   := builtin.make([]f32,                  PLAN_SAMPLES, context.temp_allocator)
	order     := builtin.make([]int,                  PLAN_SAMPLES, context.temp_allocator)

	seeded := POLICY_SEED_SAMPLES
	_policy_seed(a, sensor, sequences[:seeded])

	for n in seeded ..< PLAN_SAMPLES {
		for h in 0 ..< PLAN_HORIZON {
			sequences[n][h] = _sample_plan_step(a.plan[h])
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
		for k in 0 ..< ANALOG_COUNT {
			mean: f32
			for e in 0 ..< PLAN_ELITES {
				mean += sequences[order[e]][h][BINARY_COUNT + k] / f32(PLAN_ELITES)
			}
			variance: f32
			for e in 0 ..< PLAN_ELITES {
				diff := sequences[order[e]][h][BINARY_COUNT + k] - mean
				variance += diff * diff / f32(PLAN_ELITES)
			}
			std := max(math.sqrt(variance), PLAN_MIN_STD)

			a.plan[h].analog_mean[k] = 0.5 * a.plan[h].analog_mean[k] + 0.5 * mean
			a.plan[h].analog_std[k]  = 0.5 * a.plan[h].analog_std[k]  + 0.5 * std
		}
		for j := 0; j < BINARY_COUNT; j += 1 {
			rate: f32
			for e in 0 ..< PLAN_ELITES {
				rate += sequences[order[e]][h][j] / f32(PLAN_ELITES)
			}
			a.plan[h].binary_prob[j] = 0.5 * a.plan[h].binary_prob[j] + 0.5 * rate
		}
	}
}

_policy_seed :: proc(a: ^Agent, sensor: Sensor, sequences: [][PLAN_HORIZON]Action) {
	count := len(sequences)

	states       := builtin.make([]Sensor, count,               context.temp_allocator)
	inputs       := builtin.make([]f32,    count * MODEL_INPUT,  context.temp_allocator)
	observations := builtin.make([]f32,    count * SENSOR_SIZE,  context.temp_allocator)
	rows         := builtin.make([]f32,    count * POLICY_OUT,   context.temp_allocator)
	deltas       := builtin.make([]f32,    count * SENSOR_SIZE,  context.temp_allocator)

	for p in 0 ..< count {
		states[p] = sensor
	}

	for h in 0 ..< PLAN_HORIZON {
		if ml.pass() {
			for p in 0 ..< count {
				for i in 0 ..< SENSOR_SIZE {
					observations[p * SENSOR_SIZE + i] = states[p][i]
				}
			}

			x := ml.tensor(observations, []int{count, SENSOR_SIZE})
			ml.get_data(mlp.forward(a.policy, x), rows)

			for p in 0 ..< count {
				action := _sample_row(rows[p * POLICY_OUT:][:POLICY_OUT])
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
}

_apply_delta :: proc(a: ^Agent, state: ^Sensor, deltas: []f32) {
	for i in 0 ..< SENSOR_SIZE {
		state[i] += deltas[i] * _delta_deviation(a, i) + a.delta_mean[i]
	}

	for pair in world.ANGLE_PAIRS {
		length := math.sqrt(state[pair.sin] * state[pair.sin] + state[pair.cos] * state[pair.cos])
		if length > 1e-4 {
			state[pair.sin] /= length
			state[pair.cos] /= length
		}
	}
}

_rollout :: proc(a: ^Agent, sensor: Sensor, sequences: [][PLAN_HORIZON]Action, returns: []f32) {
	PARTICLES    :: PLAN_SAMPLES * ENSEMBLE_SIZE
	CRITIC_INPUT :: SENSOR_SIZE + ACTION_DIM

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
		if ml.pass() {
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
						scores[p] -= DEATH_PENALTY
						alive[p]   = false
					}
				}
			}
		}

		discount *= PLAN_DISCOUNT
	}

	when BOOTSTRAP_WEIGHT > 0 {
		terminal_states := builtin.make([]f32, PARTICLES * SENSOR_SIZE,  context.temp_allocator)
		terminal_rows   := builtin.make([]f32, PARTICLES * POLICY_OUT,   context.temp_allocator)
		critic_inputs   := builtin.make([]f32, PARTICLES * CRITIC_INPUT, context.temp_allocator)

		terminal_q: [VALUE_ENSEMBLE][]f32
		for v in 0 ..< VALUE_ENSEMBLE {
			terminal_q[v] = builtin.make([]f32, PARTICLES, context.temp_allocator)
		}

		for p in 0 ..< PARTICLES {
			for i in 0 ..< SENSOR_SIZE {
				terminal_states[p * SENSOR_SIZE + i] = states[p][i]
			}
		}

		if ml.pass() {
			terminal_input := ml.tensor(terminal_states, []int{PARTICLES, SENSOR_SIZE})
			ml.get_data(mlp.forward(a.policy, terminal_input), terminal_rows)

			for p in 0 ..< PARTICLES {
				action := _sample_row(terminal_rows[p * POLICY_OUT:][:POLICY_OUT])
				base   := p * CRITIC_INPUT
				for i in 0 ..< SENSOR_SIZE {
					critic_inputs[base + i] = states[p][i]
				}
				for d in 0 ..< ACTION_DIM {
					critic_inputs[base + SENSOR_SIZE + d] = action[d]
				}
			}

			critic_tensor := ml.tensor(critic_inputs, []int{PARTICLES, CRITIC_INPUT})
			for v in 0 ..< VALUE_ENSEMBLE {
				ml.get_data(mlp.forward(a.values[v], critic_tensor), terminal_q[v])
			}
		}

		for p in 0 ..< PARTICLES {
			if !alive[p] {
				continue
			}

			bootstrap := terminal_q[0][p]
			for v in 1 ..< VALUE_ENSEMBLE {
				bootstrap = min(bootstrap, terminal_q[v][p])
			}
			scores[p] += BOOTSTRAP_WEIGHT * discount * bootstrap
		}
	}

	for n in 0 ..< PLAN_SAMPLES {
		mean: f32
		for m in 0 ..< ENSEMBLE_SIZE {
			mean += scores[n * ENSEMBLE_SIZE + m] / f32(ENSEMBLE_SIZE)
		}
		returns[n] = mean

		when PESSIMISM > 0 {
			sq_mean: f32
			for m in 0 ..< ENSEMBLE_SIZE {
				score   := scores[n * ENSEMBLE_SIZE + m]
				sq_mean += score * score / f32(ENSEMBLE_SIZE)
			}
			returns[n] -= PESSIMISM * math.sqrt(max(sq_mean - mean * mean, 0))
		}
	}
}
