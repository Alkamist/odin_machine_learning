package agent

import "base:runtime"

import "core:mem"
import "core:sync"
import "core:thread"

import ml "../../../"
import    "../../../networks/mlp"

Score_Proc     :: proc(sensor: []f32) -> (reward: f32, done: bool, failed: bool)
Normalize_Proc :: proc(sensor: []f32)

HIDDEN_SIZE   :: 32
ENSEMBLE_SIZE :: 5

PLAN_HORIZON  :: 20
PLAN_SAMPLES  :: 64
PLAN_ELITES   :: 8
PLAN_DISCOUNT :: f32(0.98)

PLAN_INIT_STD :: f32(0.5)
PLAN_MIN_STD  :: f32(0.05)

POLICY_SEED_SAMPLES :: 16

ACTION_STD :: f32(0.3)

VALUE_ENSEMBLE :: 2
TAU            :: f32(0.01)
DEATH_PENALTY  :: f32(#config(LEARNER_DEATH_PENALTY, 40))

BOOTSTRAP_ENABLED     :: #config(LEARNER_BOOTSTRAP, true)
BOOTSTRAP_TRUST_FLOOR :: f32(0.3)
BOOTSTRAP_TRUST_PEAK  :: f32(0.7)
BOOTSTRAP_MIN_SAMPLES :: 64

BUFFER_CAPACITY  :: 4096
TRAIN_BATCH_SIZE :: 64
TRAIN_MINIMUM    :: 8
TRAIN_STEPS      :: 24
TRAIN_BACKLOG    :: TRAIN_STEPS * 2
LEARNING_RATE    :: 3e-3

POLICY_RATE       :: 1e-3
POLICY_MATCH_RATE :: f32(0.01)
POLICY_MATCH_SLOP :: f32(0.25)

VALUE_FIT_HORIZON  :: 150
VALUE_FIT_SAMPLES  :: 512
VALUE_FIT_EPSILON  :: f32(1e-3)
VALUE_FIT_INTERVAL :: 200

WARMUP_DECISIONS :: 24

DECISION_PERIOD      :: f64(0.05)
REFINES_PER_DECISION :: 3
REFINE_INTERVAL      :: TRAIN_STEPS / REFINES_PER_DECISION

PACING_ASYNC  :: DECISION_PERIOD
PACING_PINNED :: f64(0)

CONTEXT_SIZE :: 1024 * 1024 * 256

Frozen :: enum {
	Models,
	Policy,
}

Frozen_Set :: bit_set[Frozen; u32]

Stats :: struct {
	decisions:      int,
	policy_match:   f32,
	value_fit:      f32,
	value_trust:    f32,
	fit_samples:    int,
	processed_time: f64,
}

Agent :: struct {
	sensor_count: int,
	action_count: int,

	score:     Score_Proc,
	normalize: Normalize_Proc,

	decision_period:  f64,
	pacing_tolerance: f64,
	compute_threads:  int,

	allocator:        mem.Allocator,
	random_generator: runtime.Random_Generator,

	worker:  ^thread.Thread,
	running: bool,

	frozen_bits: u32,
	episode:     u64,

	mailbox_mutex: sync.Mutex,
	mailbox:       Snapshot,

	action_mutex: sync.Mutex,
	latch:        []f32,

	ending_mutex:  sync.Mutex,
	ending:        bool,
	ending_sensor: []f32,

	published_decisions:      int,
	published_policy_match:   u32,
	published_value_fit:      u32,
	published_value_trust:    u32,
	published_fit_samples:    int,
	published_processed_time: u64,

	models: [ENSEMBLE_SIZE]mlp.Mlp,
	opts:   [ENSEMBLE_SIZE]ml.Optimizer,

	policy:     mlp.Mlp,
	policy_opt: ml.Optimizer,

	values:        [VALUE_ENSEMBLE]mlp.Mlp,
	value_targets: [VALUE_ENSEMBLE]mlp.Mlp,
	value_opts:    [VALUE_ENSEMBLE]ml.Optimizer,

	buffer_sensors:  []f32,
	buffer_actions:  []f32,
	buffer_deltas:   []f32,
	buffer_rewards:  []f32,
	buffer_dead:     []bool,
	buffer_terminal: []bool,
	buffer_planned:  []bool,
	buffer_count:    int,
	buffer_next:     int,

	delta_mean:    []f32,
	delta_sq_mean: []f32,
	delta_samples: int,

	plan_mean: []f32,
	plan_std:  []f32,

	snapshot:           Snapshot,
	previous:           []f32,
	ending_snapshot:    []f32,
	previous_planned:   bool,
	has_previous:       bool,
	last_episode:       u64,
	next_decision_time: f64,
	train_credit:       int,
	refine_budget:      int,
	decisions:          int,
	policy_match:       f32,
	value_trust:        f32,
}

@(require_results)
create :: proc(
	sensor_count, action_count: int,
	score:                      Score_Proc,
	normalize:                  Normalize_Proc = nil,
	decision_period          := DECISION_PERIOD,
	pacing_tolerance         := PACING_ASYNC,
	compute_threads          := 1,
	allocator                := context.allocator,
	loc                      := #caller_location,
) -> (a: ^Agent) {
	assert(sensor_count > 0, "an agent needs at least one sensor", loc=loc)
	assert(action_count > 0, "an agent needs at least one action", loc=loc)
	assert(score != nil, "an agent needs a score procedure", loc=loc)
	assert(decision_period > 0, "the decision period must be positive", loc=loc)
	assert(pacing_tolerance >= 0, "the pacing tolerance cannot be negative", loc=loc)
	assert(compute_threads > 0, "an agent needs at least one compute thread", loc=loc)

	a  = new(Agent, allocator=allocator, loc=loc)
	a^ = {
		sensor_count     = sensor_count,
		action_count     = action_count,
		score            = score,
		normalize        = normalize,
		decision_period  = decision_period,
		pacing_tolerance = pacing_tolerance,
		compute_threads  = compute_threads,
		allocator        = allocator,
		random_generator = context.random_generator,
		running          = true,

		latch           = make([]f32, action_count, allocator=allocator, loc=loc),
		previous        = make([]f32, sensor_count, allocator=allocator, loc=loc),
		ending_sensor   = make([]f32, sensor_count, allocator=allocator, loc=loc),
		ending_snapshot = make([]f32, sensor_count, allocator=allocator, loc=loc),

		mailbox  = {
			sensor  = make([]f32, sensor_count, allocator=allocator, loc=loc),
			applied = make([]f32, action_count, allocator=allocator, loc=loc),
		},
		snapshot = {
			sensor  = make([]f32, sensor_count, allocator=allocator, loc=loc),
			applied = make([]f32, action_count, allocator=allocator, loc=loc),
		},

		buffer_sensors  = make([]f32,  BUFFER_CAPACITY * sensor_count, allocator=allocator, loc=loc),
		buffer_actions  = make([]f32,  BUFFER_CAPACITY * action_count, allocator=allocator, loc=loc),
		buffer_deltas   = make([]f32,  BUFFER_CAPACITY * sensor_count, allocator=allocator, loc=loc),
		buffer_rewards  = make([]f32,  BUFFER_CAPACITY,                allocator=allocator, loc=loc),
		buffer_dead     = make([]bool, BUFFER_CAPACITY,                allocator=allocator, loc=loc),
		buffer_terminal = make([]bool, BUFFER_CAPACITY,                allocator=allocator, loc=loc),
		buffer_planned  = make([]bool, BUFFER_CAPACITY,                allocator=allocator, loc=loc),

		delta_mean    = make([]f32, sensor_count, allocator=allocator, loc=loc),
		delta_sq_mean = make([]f32, sensor_count, allocator=allocator, loc=loc),

		plan_mean = make([]f32, PLAN_HORIZON * action_count, allocator=allocator, loc=loc),
		plan_std  = make([]f32, PLAN_HORIZON * action_count, allocator=allocator, loc=loc),
	}

	a.worker = thread.create_and_start_with_poly_data(a, _run)
	return
}

destroy :: proc(a: ^Agent) {
	sync.atomic_store(&a.running, false)
	thread.join(a.worker)
	thread.destroy(a.worker)

	delete(a.latch,           allocator=a.allocator)
	delete(a.previous,        allocator=a.allocator)
	delete(a.ending_sensor,   allocator=a.allocator)
	delete(a.ending_snapshot, allocator=a.allocator)

	delete(a.mailbox.sensor,   allocator=a.allocator)
	delete(a.mailbox.applied,  allocator=a.allocator)
	delete(a.snapshot.sensor,  allocator=a.allocator)
	delete(a.snapshot.applied, allocator=a.allocator)

	delete(a.buffer_sensors,  allocator=a.allocator)
	delete(a.buffer_actions,  allocator=a.allocator)
	delete(a.buffer_deltas,   allocator=a.allocator)
	delete(a.buffer_rewards,  allocator=a.allocator)
	delete(a.buffer_dead,     allocator=a.allocator)
	delete(a.buffer_terminal, allocator=a.allocator)
	delete(a.buffer_planned,  allocator=a.allocator)

	delete(a.delta_mean,    allocator=a.allocator)
	delete(a.delta_sq_mean, allocator=a.allocator)

	delete(a.plan_mean, allocator=a.allocator)
	delete(a.plan_std,  allocator=a.allocator)

	free(a, a.allocator)
}

observe :: proc(a: ^Agent, time: f64, sensor: []f32, applied: []f32 = nil, loc := #caller_location) {
	assert(len(sensor) == a.sensor_count, "sensor length must match the agent's sensor count", loc=loc)
	assert(applied == nil || len(applied) == a.action_count, "applied action length must match the agent's action count", loc=loc)

	sync.mutex_lock(&a.mailbox_mutex)
	defer sync.mutex_unlock(&a.mailbox_mutex)

	a.mailbox.valid       = true
	a.mailbox.time        = time
	a.mailbox.episode     = sync.atomic_load(&a.episode)
	a.mailbox.has_applied = applied != nil

	copy(a.mailbox.sensor, sensor)
	if applied != nil {
		copy(a.mailbox.applied, applied)
	}
}

act :: proc(a: ^Agent, action: []f32, loc := #caller_location) {
	assert(len(action) == a.action_count, "action length must match the agent's action count", loc=loc)

	sync.mutex_lock(&a.action_mutex)
	copy(action, a.latch)
	sync.mutex_unlock(&a.action_mutex)
}

end_episode :: proc(a: ^Agent, sensor: []f32, loc := #caller_location) {
	assert(len(sensor) == a.sensor_count, "sensor length must match the agent's sensor count", loc=loc)

	sync.mutex_lock(&a.ending_mutex)
	copy(a.ending_sensor, sensor)
	a.ending = true
	sync.mutex_unlock(&a.ending_mutex)

	sync.atomic_add(&a.episode, 1)
}

freeze :: proc(a: ^Agent, set: Frozen_Set) {
	sync.atomic_store(&a.frozen_bits, transmute(u32)set)
}

@(require_results)
stats :: proc(a: ^Agent) -> Stats {
	return {
		decisions      = sync.atomic_load(&a.published_decisions),
		policy_match   = transmute(f32)sync.atomic_load(&a.published_policy_match),
		value_fit      = transmute(f32)sync.atomic_load(&a.published_value_fit),
		value_trust    = transmute(f32)sync.atomic_load(&a.published_value_trust),
		fit_samples    = sync.atomic_load(&a.published_fit_samples),
		processed_time = transmute(f64)sync.atomic_load(&a.published_processed_time),
	}
}

catch_up :: proc(a: ^Agent, time: f64) {
	for sync.atomic_load(&a.running) {
		if transmute(f64)sync.atomic_load(&a.published_processed_time) >= time - a.pacing_tolerance {
			return
		}
		thread.yield()
	}
}

@(require_results)
_frozen :: proc(a: ^Agent) -> Frozen_Set {
	return transmute(Frozen_Set)sync.atomic_load(&a.frozen_bits)
}
