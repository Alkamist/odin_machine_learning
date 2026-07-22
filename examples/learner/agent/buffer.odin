package agent

import "core:math"
import "core:math/rand"

@(require_results)
_buffer_sensor :: proc(a: ^Agent, index: int) -> []f32 {
	return a.buffer_sensors[index * a.sensor_count:][:a.sensor_count]
}

@(require_results)
_buffer_action :: proc(a: ^Agent, index: int) -> []f32 {
	return a.buffer_actions[index * a.action_count:][:a.action_count]
}

@(require_results)
_buffer_delta :: proc(a: ^Agent, index: int) -> []f32 {
	return a.buffer_deltas[index * a.sensor_count:][:a.sensor_count]
}

_remember :: proc(a: ^Agent, sensor, action, successor: []f32, reward: f32, dead, terminal, planned: bool) {
	index := a.buffer_next

	copy(_buffer_sensor(a, index), sensor)
	copy(_buffer_action(a, index), action)

	delta := _buffer_delta(a, index)
	for i in 0 ..< a.sensor_count {
		delta[i] = successor[i] - sensor[i]
	}

	a.buffer_rewards[index]  = reward
	a.buffer_dead[index]     = dead
	a.buffer_terminal[index] = terminal
	a.buffer_planned[index]  = planned

	a.buffer_next = (a.buffer_next + 1) % BUFFER_CAPACITY
	if a.buffer_count < BUFFER_CAPACITY {
		a.buffer_count += 1
	}

	if .Models in _frozen(a) {
		return
	}

	a.delta_samples += 1
	rate            := 1.0 / f32(a.delta_samples)
	for i in 0 ..< a.sensor_count {
		d := delta[i]
		a.delta_mean[i]    += (d     - a.delta_mean[i])    * rate
		a.delta_sq_mean[i] += (d * d - a.delta_sq_mean[i]) * rate
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
_oldest_index :: proc(a: ^Agent, position: int) -> int {
	return (a.buffer_next - a.buffer_count + position + BUFFER_CAPACITY) % BUFFER_CAPACITY
}

@(require_results)
_delta_deviation :: proc(a: ^Agent, i: int) -> f32 {
	variance := a.delta_sq_mean[i] - a.delta_mean[i] * a.delta_mean[i]
	return max(math.sqrt(max(variance, 0)), 1e-4)
}
