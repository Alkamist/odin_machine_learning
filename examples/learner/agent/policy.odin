package agent

import "core:math"
import "core:math/rand"

import ml "../../../"
import    "../../../networks/mlp"

_sample_row :: proc(a: ^Agent, row, action: []f32) {
	for k in 0 ..< a.action_count {
		action[k] = math.tanh(row[k] + ACTION_STD * rand.float32_normal(0, 1))
	}
}

_mean_row :: proc(a: ^Agent, row, action: []f32) {
	for k in 0 ..< a.action_count {
		action[k] = math.tanh(row[k])
	}
}

@(require_results)
_policy_row :: proc(a: ^Agent, sensor: []f32) -> []f32 {
	row := make([]f32, a.action_count, allocator=context.temp_allocator)

	if ml.pass() {
		x      := ml.tensor(sensor, []int{1, a.sensor_count})
		output := mlp.forward(a.policy, x)
		ml.get_data(output, row)
	}
	return row
}

_policy_mean :: proc(a: ^Agent, sensor, action: []f32) {
	_mean_row(a, _policy_row(a, sensor), action)
}

_train_policy :: proc(a: ^Agent) {
	if a.buffer_count < TRAIN_MINIMUM || .Policy in _frozen(a) {
		return
	}

	states  := make([]f32, TRAIN_BATCH_SIZE * a.sensor_count, allocator=context.temp_allocator)
	targets := make([]f32, TRAIN_BATCH_SIZE * a.action_count, allocator=context.temp_allocator)

	count    := 0
	attempts := 0
	for count < TRAIN_BATCH_SIZE && attempts < TRAIN_BATCH_SIZE * 8 {
		attempts += 1

		index := _sample_index(a)
		if !a.buffer_planned[index] {
			continue
		}

		copy(states[count * a.sensor_count:][:a.sensor_count],  _buffer_sensor(a, index))
		copy(targets[count * a.action_count:][:a.action_count], _buffer_action(a, index))
		count += 1
	}

	if count == 0 {
		return
	}

	if ml.pass(training=true) {
		x          := ml.tensor(states[:count * a.sensor_count],  []int{count, a.sensor_count})
		y          := ml.tensor(targets[:count * a.action_count], []int{count, a.action_count})
		prediction := ml.tanh(mlp.forward(a.policy, x))
		loss       := ml.mean(ml.mean_squared_error(prediction, y))

		ml.backward(loss)

		if ml.optimizer_step(&a.policy_opt) {
			mlp.update(&a.policy_opt, a.policy)
		}
	}
}
