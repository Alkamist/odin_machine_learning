package frame

import "core:time"

MAX_STEPS :: 8

Fixed_Timestep :: struct {
	is_looping:    bool,
	accumulator:   f32,
	interpolation: f32,
	previous_tick: time.Tick,
}

@(require_results)
fixed_timestep :: proc(step: ^Fixed_Timestep, fixed_delta: f32) -> bool {
	if !step.is_looping {
		current_tick := time.tick_now()
		if step.previous_tick._nsec != 0 {
			elapsed          := f32(time.duration_seconds(time.tick_diff(step.previous_tick, current_tick)))
			step.accumulator += min(elapsed, MAX_STEPS * fixed_delta)
		}
		step.previous_tick = current_tick
	}

	if step.accumulator < fixed_delta {
		step.is_looping    = false
		step.interpolation = step.accumulator / fixed_delta
		return false
	}

	step.accumulator -= fixed_delta
	step.is_looping   = true
	return true
}

@(require_results)
normalize_angle :: proc(angle: f32) -> f32 {
	result := angle
	for result >  180.0 {
		result -= 360.0
	}
	for result < -180.0 {
		result += 360.0
	}
	return result
}

@(require_results)
lerp_angle :: proc(from, to, t: f32) -> f32 {
	return from + normalize_angle(to - from) * t
}
