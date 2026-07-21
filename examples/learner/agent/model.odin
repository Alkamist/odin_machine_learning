package agent

import ml "../../../"
import    "../../../networks/mlp"

@(require_results)
_encoded_size :: proc(a: ^Agent) -> int {
	return a.sensor_count + a.action_count
}

_encode :: proc(a: ^Agent, sensor, action, dst: []f32) {
	copy(dst, sensor)
	copy(dst[a.sensor_count:], action)
}

_apply_delta :: proc(a: ^Agent, state, deltas: []f32) {
	for i in 0 ..< a.sensor_count {
		state[i] += deltas[i] * _delta_deviation(a, i) + a.delta_mean[i]
	}
	if a.normalize != nil {
		a.normalize(state)
	}
}

_train_models :: proc(a: ^Agent) {
	if a.buffer_count < TRAIN_MINIMUM || .Models in _frozen(a) {
		return
	}

	input_size := _encoded_size(a)

	inputs  := make([]f32, TRAIN_BATCH_SIZE * input_size,     allocator=context.temp_allocator)
	targets := make([]f32, TRAIN_BATCH_SIZE * a.sensor_count, allocator=context.temp_allocator)

	if ml.pass(training=true) {
		total: ml.Tensor

		for m in 0 ..< ENSEMBLE_SIZE {
			for b in 0 ..< TRAIN_BATCH_SIZE {
				index := _sample_index(a)
				_encode(a, _buffer_sensor(a, index), _buffer_action(a, index), inputs[b * input_size:][:input_size])

				delta := _buffer_delta(a, index)
				for i in 0 ..< a.sensor_count {
					targets[b * a.sensor_count + i] = (delta[i] - a.delta_mean[i]) / _delta_deviation(a, i)
				}
			}

			x          := ml.tensor(inputs,  []int{TRAIN_BATCH_SIZE, input_size})
			y          := ml.tensor(targets, []int{TRAIN_BATCH_SIZE, a.sensor_count})
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
