package agent

import "core:math/rand"
import "core:sync"
import "core:time"

import ml  "../../../"
import     "../../../networks/mlp"
import cpu "../../../backends/cpu"

EMPTY_SLEEP :: 1 * time.Millisecond
IDLE_SLEEP  :: 500 * time.Microsecond

Snapshot :: struct {
	valid:       bool,
	has_applied: bool,
	time:        f64,
	episode:     u64,
	sensor:      []f32,
	applied:     []f32,
}

_run :: proc(a: ^Agent) {
	context.random_generator = a.random_generator

	ctx := cpu.context_create(CONTEXT_SIZE)
	defer cpu.context_destroy(ctx)

	ml.context_scope(ctx)

	cpu.set_thread_count(a.compute_threads)

	_boot(a)
	defer _teardown(a)

	for sync.atomic_load(&a.running) {
		free_all(context.temp_allocator)

		sync.mutex_lock(&a.mailbox_mutex)
		_snapshot_receive(&a.snapshot, a.mailbox)
		sync.mutex_unlock(&a.mailbox_mutex)

		if !a.snapshot.valid {
			time.sleep(EMPTY_SLEEP)
			continue
		}

		if !_step(a) {
			continue
		}

		processed := transmute(f64)sync.atomic_load(&a.published_processed_time)
		sync.atomic_store(&a.published_processed_time, transmute(u64)a.snapshot.time)

		if a.snapshot.time <= processed {
			time.sleep(IDLE_SLEEP)
		}
	}
}

_snapshot_receive :: proc(dst: ^Snapshot, src: Snapshot) {
	dst.valid       = src.valid
	dst.has_applied = src.has_applied
	dst.time        = src.time
	dst.episode     = src.episode

	copy(dst.sensor,  src.sensor)
	copy(dst.applied, src.applied)
}

_boot :: proc(a: ^Agent) {
	encoded := _encoded_size(a)

	for m in 0 ..< ENSEMBLE_SIZE {
		a.models[m] = mlp.make(encoded, HIDDEN_SIZE, a.sensor_count)
		a.opts[m]   = ml.optimizer_make(learning_rate=LEARNING_RATE)
	}

	a.policy     = mlp.make(a.sensor_count, HIDDEN_SIZE, a.action_count)
	a.policy_opt = ml.optimizer_make(learning_rate=POLICY_RATE)

	for v in 0 ..< VALUE_ENSEMBLE {
		a.values[v]        = mlp.make(encoded, HIDDEN_SIZE, 1)
		a.value_targets[v] = mlp.make(encoded, HIDDEN_SIZE, 1)
		a.value_opts[v]    = ml.optimizer_make(learning_rate=LEARNING_RATE)
		mlp.copy(a.value_targets[v], a.values[v])
	}

	_forget_episode(a)
}

_teardown :: proc(a: ^Agent) {
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

_forget_episode :: proc(a: ^Agent) {
	a.has_previous = false
	_plan_reset(a)
}

@(require_results)
_step :: proc(a: ^Agent) -> (idle: bool) {
	if a.snapshot.episode != a.last_episode {
		_forget_episode(a)
		a.last_episode = a.snapshot.episode
	}

	if a.snapshot.time >= a.next_decision_time {
		_decide(a)
		a.next_decision_time = a.snapshot.time + a.decision_period
	}
	else if a.train_credit > 0 || a.refine_budget > 0 {
		if a.train_credit > 0 {
			_train_models(a)
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

_decide :: proc(a: ^Agent) {
	sensor := a.snapshot.sensor

	if a.has_previous {
		applied := a.snapshot.has_applied ? a.snapshot.applied : a.latch
		reward, dead := a.score(sensor)
		_remember(a, a.previous, applied, sensor, reward, dead, a.previous_planned)
	}

	action       := make([]f32, a.action_count, allocator=context.temp_allocator)
	used_planner := a.decisions >= WARMUP_DECISIONS

	if used_planner {
		_plan_shift(a)
		_plan_refine(a, sensor)
		_plan_action(a, action)

		expected := make([]f32, a.action_count, allocator=context.temp_allocator)
		_policy_mean(a, sensor, expected)

		match := f32(1)
		for k in 0 ..< a.action_count {
			if abs(action[k] - expected[k]) > POLICY_MATCH_SLOP {
				match = 0
			}
		}

		a.policy_match += (match - a.policy_match) * POLICY_MATCH_RATE
		sync.atomic_store(&a.published_policy_match, transmute(u32)a.policy_match)
	}
	else {
		for k in 0 ..< a.action_count {
			action[k] = rand.float32() * 2 - 1
		}
	}

	sync.mutex_lock(&a.action_mutex)
	copy(a.latch, action)
	sync.mutex_unlock(&a.action_mutex)

	copy(a.previous, sensor)
	a.previous_planned = used_planner
	a.has_previous     = true
	a.train_credit     = min(a.train_credit + TRAIN_STEPS, TRAIN_BACKLOG)
	a.refine_budget    = REFINES_PER_DECISION - 1
	a.decisions       += 1

	sync.atomic_store(&a.published_decisions, a.decisions)

	if a.decisions % VALUE_FIT_INTERVAL == 0 {
		correlation, samples := _value_fit(a)
		sync.atomic_store(&a.published_value_fit,   transmute(u32)correlation)
		sync.atomic_store(&a.published_fit_samples, samples)
	}
}
