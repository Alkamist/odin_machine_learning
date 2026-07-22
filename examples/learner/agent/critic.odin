package agent

import "core:math"

import ml "../../../"
import    "../../../networks/mlp"

_train_value :: proc(a: ^Agent) {
	if a.buffer_count < TRAIN_MINIMUM {
		return
	}

	critic_input := _encoded_size(a)

	successors := make([]f32,  TRAIN_BATCH_SIZE * a.sensor_count, allocator=context.temp_allocator)
	critic_now := make([]f32,  TRAIN_BATCH_SIZE * critic_input,   allocator=context.temp_allocator)
	rewards    := make([]f32,  TRAIN_BATCH_SIZE,                  allocator=context.temp_allocator)
	deaths     := make([]bool, TRAIN_BATCH_SIZE,                  allocator=context.temp_allocator)
	terminals  := make([]bool, TRAIN_BATCH_SIZE,                  allocator=context.temp_allocator)

	for b in 0 ..< TRAIN_BATCH_SIZE {
		index  := _sample_index(a)
		sensor := _buffer_sensor(a, index)
		delta  := _buffer_delta(a, index)

		successor := successors[b * a.sensor_count:][:a.sensor_count]
		for i in 0 ..< a.sensor_count {
			successor[i] = sensor[i] + delta[i]
		}

		_encode(a, sensor, _buffer_action(a, index), critic_now[b * critic_input:][:critic_input])

		rewards[b]   = a.buffer_rewards[index]
		deaths[b]    = a.buffer_dead[index]
		terminals[b] = a.buffer_terminal[index]
	}

	successor_rows := make([]f32, TRAIN_BATCH_SIZE * a.action_count, allocator=context.temp_allocator)
	critic_next    := make([]f32, TRAIN_BATCH_SIZE * critic_input,   allocator=context.temp_allocator)
	targets        := make([]f32, TRAIN_BATCH_SIZE,                  allocator=context.temp_allocator)
	action         := make([]f32, a.action_count,                    allocator=context.temp_allocator)

	target_q: [VALUE_ENSEMBLE][]f32
	for v in 0 ..< VALUE_ENSEMBLE {
		target_q[v] = make([]f32, TRAIN_BATCH_SIZE, allocator=context.temp_allocator)
	}

	if ml.pass() {
		successor_tensor := ml.tensor(successors, []int{TRAIN_BATCH_SIZE, a.sensor_count})
		ml.get_data(mlp.forward(a.policy, successor_tensor), successor_rows)

		for b in 0 ..< TRAIN_BATCH_SIZE {
			_sample_row(a, successor_rows[b * a.action_count:][:a.action_count], action)
			_encode(a, successors[b * a.sensor_count:][:a.sensor_count], action, critic_next[b * critic_input:][:critic_input])
		}

		critic_next_tensor := ml.tensor(critic_next, []int{TRAIN_BATCH_SIZE, critic_input})
		for v in 0 ..< VALUE_ENSEMBLE {
			ml.get_data(mlp.forward(a.value_targets[v], critic_next_tensor), target_q[v])
		}
	}

	for b in 0 ..< TRAIN_BATCH_SIZE {
		if deaths[b] {
			targets[b] = rewards[b] - DEATH_PENALTY
			continue
		}
		if terminals[b] {
			targets[b] = rewards[b]
			continue
		}

		q := target_q[0][b]
		for v in 1 ..< VALUE_ENSEMBLE {
			q = min(q, target_q[v][b])
		}
		targets[b] = rewards[b] + PLAN_DISCOUNT * q
	}

	if ml.pass(training=true) {
		x := ml.tensor(critic_now, []int{TRAIN_BATCH_SIZE, critic_input})
		y := ml.tensor(targets,    []int{TRAIN_BATCH_SIZE})

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
_continues :: proc(a: ^Agent, previous, next: int) -> bool {
	sensor := _buffer_sensor(a, previous)
	delta  := _buffer_delta(a, previous)
	follow := _buffer_sensor(a, next)

	for i in 0 ..< a.sensor_count {
		if abs(sensor[i] + delta[i] - follow[i]) > VALUE_FIT_EPSILON {
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
		index    := _oldest_index(a, cursor)
		observed += discount * a.buffer_rewards[index]

		if a.buffer_dead[index] {
			observed -= discount * DEATH_PENALTY
			return observed, true
		}
		if a.buffer_terminal[index] {
			return observed, true
		}

		discount *= PLAN_DISCOUNT
		cursor   += 1

		if cursor >= a.buffer_count {
			return 0, false
		}
		if !_continues(a, index, _oldest_index(a, cursor)) {
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
_trust :: proc(correlation: f32, samples: int) -> f32 {
	if samples < BOOTSTRAP_MIN_SAMPLES {
		return 0
	}
	return clamp((correlation - BOOTSTRAP_TRUST_FLOOR) / (BOOTSTRAP_TRUST_PEAK - BOOTSTRAP_TRUST_FLOOR), 0, 1)
}

@(require_results)
_value_fit :: proc(a: ^Agent) -> (correlation: f32, samples: int) {
	critic_input := _encoded_size(a)
	stride       := max((a.buffer_count + VALUE_FIT_SAMPLES - 1) / VALUE_FIT_SAMPLES, 1)

	inputs  := make([]f32, VALUE_FIT_SAMPLES * critic_input, allocator=context.temp_allocator)
	returns := make([]f32, VALUE_FIT_SAMPLES,                allocator=context.temp_allocator)

	for position := 0; position < a.buffer_count && samples < VALUE_FIT_SAMPLES; position += stride {
		observed, complete := _observed_return(a, position)
		if !complete {
			continue
		}

		index := _oldest_index(a, position)
		_encode(a, _buffer_sensor(a, index), _buffer_action(a, index), inputs[samples * critic_input:][:critic_input])
		returns[samples] = observed
		samples += 1
	}

	if samples < 2 {
		samples = 0
		return
	}

	estimates: [VALUE_ENSEMBLE][]f32
	for v in 0 ..< VALUE_ENSEMBLE {
		estimates[v] = make([]f32, samples, allocator=context.temp_allocator)
	}

	if ml.pass() {
		x := ml.tensor(inputs[:samples * critic_input], []int{samples, critic_input})
		for v in 0 ..< VALUE_ENSEMBLE {
			ml.get_data(mlp.forward(a.values[v], x), estimates[v])
		}
	}

	predictions := make([]f32, samples, allocator=context.temp_allocator)
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
