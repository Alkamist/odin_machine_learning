package machine_learning

import "base:builtin"
import "base:intrinsics"

import "core:mem"
import "core:math"

Optimizer_State :: struct {
	m: Backend_Buffer,
	v: Backend_Buffer,
}

Optimizer_Kind :: enum {
	Adam_W,
}

Optimizer :: struct {
	kind: Optimizer_Kind,

	learning_rate:      f32,
	beta1:              f32,
	beta2:              f32,
	epsilon:            f32,
	weight_decay:       f32,
	accumulation_steps: int,

	iteration:            u64,
	accumulation_counter: int,

	bias_correction1: f32,
	bias_correction2: f32,

	backend: ^Backend,
	state:   map[Backend_Buffer]Optimizer_State,
}

@(require_results)
optimizer_make :: proc(
	learning_rate:      f32 = 0.001,
	beta1:              f32 = 0.9,
	beta2:              f32 = 0.999,
	epsilon:            f32 = 1e-8,
	weight_decay:       f32 = 0,
	accumulation_steps: int = 1,
	kind:               Optimizer_Kind = .Adam_W,
	loc := #caller_location,
) -> (opt: Optimizer) {
	assert(accumulation_steps >= 1, "accumulation_steps must be at least 1", loc=loc)
	opt = {
		kind               = kind,
		learning_rate      = learning_rate,
		beta1              = beta1,
		beta2              = beta2,
		epsilon            = epsilon,
		weight_decay       = weight_decay,
		accumulation_steps = accumulation_steps,
	}
	return
}

@(require_results)
_optimizer_state :: proc(opt: ^Optimizer, t: Tensor, loc := #caller_location) -> Optimizer_State {
	key := t.buffers[.Data]
	if existing, ok := opt.state[key]; ok {
		return existing
	}

	if opt.state == nil {
		opt.state = builtin.make(map[Backend_Buffer]Optimizer_State)
	}
	opt.backend = t.backend

	byte_count := _data_byte_count(.F32, t.count)
	byte_count = (byte_count + 3) & ~int(3)

	state := Optimizer_State{
		m = t.backend.buffer_alloc(byte_count, .Gradient, true, loc),
		v = t.backend.buffer_alloc(byte_count, .Gradient, true, loc),
	}
	opt.state[key] = state
	return state
}

@(require_results)
_optimizer_state_lookup :: proc(opt: ^Optimizer, t: Tensor) -> (Optimizer_State, bool) {
	if opt == nil {
		return {}, false
	}
	state, ok := opt.state[t.buffers[.Data]]
	return state, ok
}

optimizer_destroy :: proc(opt: ^Optimizer, loc := #caller_location) {
	if opt.backend == nil {
		return
	}
	for _, state in opt.state {
		opt.backend.buffer_free(state.m, loc)
		opt.backend.buffer_free(state.v, loc)
	}
	builtin.delete(opt.state)
	opt.state   = nil
	opt.backend = nil
}

@(require_results)
optimizer_step :: proc(opt: ^Optimizer) -> bool {
	opt.accumulation_counter += 1
	if opt.accumulation_counter < opt.accumulation_steps {
		return false
	}
	opt.accumulation_counter = 0

	opt.iteration += 1

	opt.bias_correction1 = 1 - math.pow(opt.beta1, f32(opt.iteration))
	opt.bias_correction2 = 1 - math.pow(opt.beta2, f32(opt.iteration))

	return true
}

registry_step :: proc(opt: ^Optimizer, r: ^Registry, max_grad_norm: f32 = 0, loc := #caller_location) -> (stepped: bool) {
	if !optimizer_step(opt) {
		return false
	}
	if max_grad_norm > 0 {
		clip_gradient_norm(r, max_grad_norm, loc=loc)
	}
	registry_update(opt, r, loc=loc)
	return true
}

update :: proc(opt: ^Optimizer, t: Tensor, loc := #caller_location) {
	assert(opt.iteration > 0, "update called before optimizer_step; gate updates with `if optimizer_step(&opt) { ... }`", loc=loc)
	state := _optimizer_state(opt, t, loc)
	t.backend.update(opt^, t, state.m, state.v, loc)
}

clip_gradient_norm :: proc(r: ^Registry, max_norm: f32, loc := #caller_location) -> (norm: f32) {
	trainable_count := 0
	for parameter in r.parameters {
		if .Train in parameter.flags {
			trainable_count += 1
		}
	}
	if trainable_count == 0 {
		return
	}

	ctx     := current_context(loc=loc)
	backend := ctx.backend

	if ctx.grad_norm_accumulator == (Backend_Buffer{}) {
		ctx.grad_norm_accumulator = backend.buffer_alloc(size_of(f64), .Gradient, true, loc)
	}
	accumulator := ctx.grad_norm_accumulator

	zero_bytes: [size_of(f64)]byte
	backend.buffer_set(accumulator, zero_bytes[:], loc)

	for parameter in r.parameters {
		if .Train not_in parameter.flags {
			continue
		}
		assert(parameter.tensor.buffers[.Gradient] != Backend_Buffer{}, "clip_gradient_norm requires parameters with a gradient buffer", loc=loc)
		backend.buffer_sq_sum_accumulate(parameter.tensor.buffers[.Gradient], parameter.tensor.count, accumulator, loc)
	}

	total_sq: [1]f64
	backend.buffer_get(accumulator, mem.slice_to_bytes(total_sq[:]), loc)

	norm = f32(math.sqrt(total_sq[0]))
	if norm <= max_norm || norm == 0 {
		return
	}

	scale := max_norm / norm
	for parameter in r.parameters {
		if .Train not_in parameter.flags {
			continue
		}
		backend.buffer_scale(parameter.tensor.buffers[.Gradient], parameter.tensor.count, scale, loc)
	}
	return
}

@(require_results)
linear_schedule :: proc(step, total_steps: int, start_value, end_value: f32) -> f32 {
	if total_steps <= 0 {
		return end_value
	}
	t := builtin.clamp(f32(step) / f32(total_steps), 0, 1)
	return start_value + (end_value - start_value) * t
}

@(require_results)
warmup_cosine_schedule :: proc(step, total_steps, warmup_steps: int, peak_value, final_value: f32) -> f32 {
	if warmup_steps > 0 && step < warmup_steps {
		return peak_value * f32(step) / f32(warmup_steps)
	}
	if total_steps <= warmup_steps {
		return final_value
	}
	t := builtin.clamp(f32(step - warmup_steps) / f32(total_steps - warmup_steps), 0, 1)
	return final_value + (peak_value - final_value) * 0.5 * (1 + math.cos(math.PI * t))
}

