package agent

import "core:math"
import "core:math/rand"
import "core:slice"

import ml "../../../"
import    "../../../networks/mlp"

@(require_results)
_plan_slot :: proc(a: ^Agent, plan: []f32, h: int) -> []f32 {
	return plan[h * a.action_count:][:a.action_count]
}

@(require_results)
_sequence_slot :: proc(a: ^Agent, sequences: []f32, n, h: int) -> []f32 {
	return sequences[(n * PLAN_HORIZON + h) * a.action_count:][:a.action_count]
}

_plan_reset :: proc(a: ^Agent) {
	for k in 0 ..< len(a.plan_mean) {
		a.plan_mean[k] = 0
		a.plan_std[k]  = PLAN_INIT_STD
	}
}

_plan_shift :: proc(a: ^Agent) {
	copy(a.plan_mean, a.plan_mean[a.action_count:])
	copy(a.plan_std,  a.plan_std[a.action_count:])

	last_mean := _plan_slot(a, a.plan_mean, PLAN_HORIZON - 1)
	last_std  := _plan_slot(a, a.plan_std,  PLAN_HORIZON - 1)
	for k in 0 ..< a.action_count {
		last_mean[k] = 0
		last_std[k]  = PLAN_INIT_STD
	}
}

_plan_action :: proc(a: ^Agent, action: []f32) {
	for k in 0 ..< a.action_count {
		action[k] = clamp(a.plan_mean[k], -1, 1)
	}
}

_sample_plan_slot :: proc(a: ^Agent, h: int, action: []f32) {
	mean := _plan_slot(a, a.plan_mean, h)
	std  := _plan_slot(a, a.plan_std,  h)
	for k in 0 ..< a.action_count {
		action[k] = clamp(mean[k] + std[k] * rand.float32_normal(0, 1), -1, 1)
	}
}

_plan_refine :: proc(a: ^Agent, sensor: []f32) {
	sequences := make([]f32, PLAN_SAMPLES * PLAN_HORIZON * a.action_count, allocator=context.temp_allocator)
	returns   := make([]f32, PLAN_SAMPLES,                                 allocator=context.temp_allocator)
	order     := make([]int, PLAN_SAMPLES,                                 allocator=context.temp_allocator)

	_policy_seed(a, sensor, sequences, POLICY_SEED_SAMPLES)

	for n in POLICY_SEED_SAMPLES ..< PLAN_SAMPLES {
		for h in 0 ..< PLAN_HORIZON {
			_sample_plan_slot(a, h, _sequence_slot(a, sequences, n, h))
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
		mean_slot := _plan_slot(a, a.plan_mean, h)
		std_slot  := _plan_slot(a, a.plan_std,  h)

		for k in 0 ..< a.action_count {
			mean: f32
			for e in 0 ..< PLAN_ELITES {
				mean += _sequence_slot(a, sequences, order[e], h)[k] / f32(PLAN_ELITES)
			}

			variance: f32
			for e in 0 ..< PLAN_ELITES {
				difference := _sequence_slot(a, sequences, order[e], h)[k] - mean
				variance   += difference * difference / f32(PLAN_ELITES)
			}
			std := max(math.sqrt(variance), PLAN_MIN_STD)

			mean_slot[k] = 0.5 * mean_slot[k] + 0.5 * mean
			std_slot[k]  = 0.5 * std_slot[k]  + 0.5 * std
		}
	}
}

_policy_seed :: proc(a: ^Agent, sensor: []f32, sequences: []f32, count: int) {
	input_size := _encoded_size(a)

	states := make([]f32, count * a.sensor_count, allocator=context.temp_allocator)
	inputs := make([]f32, count * input_size,     allocator=context.temp_allocator)
	rows   := make([]f32, count * a.action_count, allocator=context.temp_allocator)
	deltas := make([]f32, count * a.sensor_count, allocator=context.temp_allocator)

	for p in 0 ..< count {
		copy(states[p * a.sensor_count:][:a.sensor_count], sensor)
	}

	for h in 0 ..< PLAN_HORIZON {
		if ml.pass() {
			x := ml.tensor(states, []int{count, a.sensor_count})
			ml.get_data(mlp.forward(a.policy, x), rows)

			for p in 0 ..< count {
				action := _sequence_slot(a, sequences, p, h)
				_sample_row(a, rows[p * a.action_count:][:a.action_count], action)
				_encode(a, states[p * a.sensor_count:][:a.sensor_count], action, inputs[p * input_size:][:input_size])
			}

			model := rand.int_max(ENSEMBLE_SIZE)
			y     := ml.tensor(inputs, []int{count, input_size})
			ml.get_data(mlp.forward(a.models[model], y), deltas)

			for p in 0 ..< count {
				_apply_delta(a, states[p * a.sensor_count:][:a.sensor_count], deltas[p * a.sensor_count:][:a.sensor_count])
			}
		}
	}
}

_bootstrap :: proc(a: ^Agent, states: []f32, alive: []bool, scores: []f32, weight: f32) {
	PARTICLES :: PLAN_SAMPLES * ENSEMBLE_SIZE

	input_size := _encoded_size(a)

	rows   := make([]f32, PARTICLES * a.action_count, allocator=context.temp_allocator)
	inputs := make([]f32, PARTICLES * input_size,     allocator=context.temp_allocator)
	action := make([]f32, a.action_count,             allocator=context.temp_allocator)

	estimates: [VALUE_ENSEMBLE][]f32
	for v in 0 ..< VALUE_ENSEMBLE {
		estimates[v] = make([]f32, PARTICLES, allocator=context.temp_allocator)
	}

	if ml.pass() {
		x := ml.tensor(states, []int{PARTICLES, a.sensor_count})
		ml.get_data(mlp.forward(a.policy, x), rows)

		for p in 0 ..< PARTICLES {
			_mean_row(a, rows[p * a.action_count:][:a.action_count], action)
			_encode(a, states[p * a.sensor_count:][:a.sensor_count], action, inputs[p * input_size:][:input_size])
		}

		y := ml.tensor(inputs, []int{PARTICLES, input_size})
		for v in 0 ..< VALUE_ENSEMBLE {
			ml.get_data(mlp.forward(a.values[v], y), estimates[v])
		}
	}

	for p in 0 ..< PARTICLES {
		if !alive[p] {
			continue
		}

		q := estimates[0][p]
		for v in 1 ..< VALUE_ENSEMBLE {
			q = min(q, estimates[v][p])
		}
		scores[p] += weight * q
	}
}

_rollout :: proc(a: ^Agent, sensor: []f32, sequences: []f32, returns: []f32) {
	PARTICLES :: PLAN_SAMPLES * ENSEMBLE_SIZE

	input_size := _encoded_size(a)

	states := make([]f32,  PARTICLES * a.sensor_count,    allocator=context.temp_allocator)
	alive  := make([]bool, PARTICLES,                     allocator=context.temp_allocator)
	scores := make([]f32,  PARTICLES,                     allocator=context.temp_allocator)
	inputs := make([]f32,  PLAN_SAMPLES * input_size,     allocator=context.temp_allocator)
	deltas := make([]f32,  PLAN_SAMPLES * a.sensor_count, allocator=context.temp_allocator)

	for p in 0 ..< PARTICLES {
		copy(states[p * a.sensor_count:][:a.sensor_count], sensor)
		alive[p] = true
	}

	discount := f32(1)

	for h in 0 ..< PLAN_HORIZON {
		if ml.pass() {
			for m in 0 ..< ENSEMBLE_SIZE {
				for n in 0 ..< PLAN_SAMPLES {
					particle := n * ENSEMBLE_SIZE + m
					_encode(a, states[particle * a.sensor_count:][:a.sensor_count], _sequence_slot(a, sequences, n, h), inputs[n * input_size:][:input_size])
				}

				x          := ml.tensor(inputs, []int{PLAN_SAMPLES, input_size})
				prediction := mlp.forward(a.models[m], x)
				ml.get_data(prediction, deltas)

				for n in 0 ..< PLAN_SAMPLES {
					particle := n * ENSEMBLE_SIZE + m
					if !alive[particle] {
						continue
					}

					state := states[particle * a.sensor_count:][:a.sensor_count]
					_apply_delta(a, state, deltas[n * a.sensor_count:][:a.sensor_count])

					reward, done, failed := a.score(state)
					scores[particle]     += discount * reward

					if failed {
						scores[particle] -= discount * DEATH_PENALTY
					}
					if done {
						alive[particle] = false
					}
				}
			}
		}

		discount *= PLAN_DISCOUNT
	}

	when BOOTSTRAP_ENABLED {
		if a.value_trust > 0 {
			_bootstrap(a, states, alive, scores, discount * a.value_trust)
		}
	}

	for n in 0 ..< PLAN_SAMPLES {
		mean: f32
		for m in 0 ..< ENSEMBLE_SIZE {
			mean += scores[n * ENSEMBLE_SIZE + m] / f32(ENSEMBLE_SIZE)
		}
		returns[n] = mean
	}
}
