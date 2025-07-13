package utility

import "core:time"
import "core:math"
import "core:math/rand"
import "core:slice"


// These are basic utility functions used throughout the examples.


Rolling_f32 :: struct($N: int) where N > 0 {
	offset: int,
	len:    int,
	values: [N]f32,
}

rolling_value :: proc(rolling: Rolling_f32($N), #any_int lookback: int) -> f32 {
	assert(lookback >= 0 && lookback < rolling.len)

	index := (rolling.offset - 1 - lookback + N) % N

	return rolling.values[index]
}

rolling_append :: proc(rolling: ^Rolling_f32($N), value: f32) {
	rolling.values[rolling.offset] = value

	rolling.offset += 1
	if rolling.offset >= N {
		rolling.offset = 0
	}

	rolling.len = min(rolling.len + 1, N)
}

rolling_average :: proc(rolling: Rolling_f32($N)) -> (res: f32) {
	if rolling.len == 0 do return

	for i in 0 ..< rolling.len {
		res += rolling_value(rolling, i)
	}

	res /= f32(rolling.len)

	return
}

rolling_std :: proc(rolling: Rolling_f32($N)) -> (res: f32) {
	average := rolling_average(rolling)

	sum: f32
	for i in 0 ..< rolling.len {
		diff := rolling_value(rolling, i) - average
		sum  += diff * diff
	}

	res = math.sqrt(sum / f32(rolling.len))

	return
}

Fixed_Timestep :: struct {
	is_looping:    bool,
	accumulator:   f32,
	interpolation: f32,
	previous_tick: time.Tick,
}

@(require_results)
fixed_timestep :: proc(step: ^Fixed_Timestep, fixed_delta: f32) -> bool {
	if !step.is_looping {
		if step.previous_tick._nsec == 0 {
			step.previous_tick = time.tick_now()
		}
		current_tick := time.tick_now()
		step.accumulator += f32(time.duration_seconds(time.tick_diff(step.previous_tick, current_tick)))
		step.previous_tick = current_tick
	}

	step.is_looping = step.accumulator >= fixed_delta

	if step.is_looping {
		step.accumulator -= fixed_delta
	}

	step.interpolation = step.accumulator / fixed_delta

	return step.is_looping
}

@(require_results)
rotate :: proc(v: [2]f32, angle: f32) -> [2]f32 {
	c := math.cos(angle)
	s := math.sin(angle)
	return {
		v.x * c - v.y * s,
		v.x * s + v.y * c,
	}
}

@(require_results)
normalize_angle :: proc(angle: f32) -> f32 {
	result := angle
	for result >  180.0 do result -= 360.0
	for result < -180.0 do result += 360.0
	return result
}

@(require_results)
lerp_angle :: proc(from, to, t: f32) -> f32 {
	return from + normalize_angle(to - from) * t
}

@(require_results)
linear_learning_rate :: proc(maximum, minimum: f32, step, decay: int) -> f32 {
	if step >= decay {
		return minimum
	}

	decay_factor := f32(step) / f32(decay)
	return maximum * (1 - decay_factor) + minimum * decay_factor
}

@(require_results)
cosine_annealing_learning_rate :: proc(maximum, minimum: f32, step, decay: int) -> f32 {
	if step >= decay {
		return minimum
	}

	decay_ratio := f32(step) / f32(decay)
	assert(0 <= decay_ratio && decay_ratio <= 1)

	coeff := 0.5 * (1.0 + math.cos(math.PI * decay_ratio))

	return minimum + coeff * (maximum - minimum)
}

@(require_results)
sample_probability_distribution :: proc(probabilities: []f32) -> int {
	r := rand.float32()

	cumulative_probability: f32
	for prob, i in probabilities {
		cumulative_probability += prob
		if r <= cumulative_probability {
			return i
		}
	}

	return len(probabilities) - 1
}

@(require_results)
sample_top_p :: proc(logits: []f32, p: f32 = 0.9, temperature: f32 = 1.0) -> int {
	Item :: struct {
		index:       int,
		logit:       f32,
		probability: f32,
	}

	items := make([]Item, len(logits), context.temp_allocator)

	max_logit := min(f32)
	for i in 0 ..< len(logits) {
		logit := logits[i]
		if temperature != 1.0 {
			logit /= temperature
		}
		if logit > max_logit {
			max_logit = logit
		}
		items[i] = {i, logit, 0}
	}

	sum: f32
	for i in 0 ..< len(items) {
		items[i].probability = math.exp(items[i].logit - max_logit)
		sum += items[i].probability
	}
	for i in 0 ..< len(items) {
		items[i].probability /= sum
	}

	slice.sort_by(items, proc(a, b: Item) -> bool {
		return a.probability > b.probability
	})

	if p > 0 && p < 1.0 {
		cumulative: f32
		for i in 0 ..< len(items) {
			cumulative += items[i].probability
			if cumulative >= p {
				items = items[:i + 1]
				break
			}
		}
	}

	sum = 0.0
	for i in 0 ..< len(items) {
		sum += items[i].probability
	}
	for i in 0 ..< len(items) {
		items[i].probability /= sum
	}

	r := rand.float32()

	cumulative: f32
	for i in 0 ..< len(items) {
		cumulative += items[i].probability
		if r <= cumulative {
			return items[i].index
		}
	}

	return items[0].index
}