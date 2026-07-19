package ml_tests

import "core:mem"
import "core:math"
import "core:testing"

import ml    "../"
import cpu   "../backends/cpu"
import cases "cases"

ADAM_SIZE  :: 8
ADAM_STEPS :: 12

ADAM_LR  :: f32(0.01)
ADAM_B1  :: f32(0.9)
ADAM_B2  :: f32(0.999)
ADAM_EPS :: f32(1e-8)
ADAM_WD  :: f32(0.1)

_adam_grad :: cases.adam_grad

@(test)
test_adam_update :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	param := ml.alloc(.F32, {ADAM_SIZE}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)

	init_w: [ADAM_SIZE]f32
	for i in 0 ..< ADAM_SIZE {
		init_w[i] = f32(i) * 0.1 - 0.35
	}
	ml.set_data(param, init_w[:])

	ref_w := init_w
	ref_m: [ADAM_SIZE]f32
	ref_v: [ADAM_SIZE]f32

	opt := ml.optimizer_make(learning_rate=ADAM_LR, beta1=ADAM_B1, beta2=ADAM_B2, epsilon=ADAM_EPS, weight_decay=ADAM_WD)
	grad: [ADAM_SIZE]f32
	for step in 1 ..= ADAM_STEPS {
		for i in 0 ..< ADAM_SIZE {
			grad[i] = _adam_grad(step, i)
		}
		ml.set_bytes(param, .Gradient, mem.slice_to_bytes(grad[:]))

		stepped := ml.optimizer_step(&opt)
		testing.expect(t, stepped, "optimizer_step should fire every step by default")
		ml.update(&opt, param)

		bc1 := 1 - math.pow(ADAM_B1, f32(step))
		bc2 := 1 - math.pow(ADAM_B2, f32(step))
		for i in 0 ..< ADAM_SIZE {
			g := grad[i]
			ref_m[i] = ADAM_B1 * ref_m[i] + (1 - ADAM_B1) * g
			ref_v[i] = ADAM_B2 * ref_v[i] + (1 - ADAM_B2) * g * g
			m_hat := ref_m[i] / bc1
			v_hat := ref_v[i] / bc2
			ref_w[i] = ref_w[i] * (1 - ADAM_LR * ADAM_WD) - ADAM_LR * m_hat / (math.sqrt(v_hat) + ADAM_EPS)
		}
	}

	got: [ADAM_SIZE]f32
	ml.get_data(param, got[:])

	for i in 0 ..< ADAM_SIZE {
		a := f64(ref_w[i])
		b := f64(got[i])
		denom := max(max(abs(a), abs(b)), 1e-4)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= 1e-4,
			"adam elem %d ref=%.7g got=%.7g rel_err=%.4g", i, a, b, rel)
	}

	ml.optimizer_destroy(&opt)
	ml.destroy(param)
}

ADAM_ACCUM_STEPS   :: 2
ADAM_ACCUM_WINDOWS :: 3

@(test)
test_adam_accumulation :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	param := ml.alloc(.F32, {ADAM_SIZE}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)

	init_w: [ADAM_SIZE]f32
	for i in 0 ..< ADAM_SIZE {
		init_w[i] = f32(i) * 0.1 - 0.35
	}
	ml.set_data(param, init_w[:])

	ref_w := init_w
	ref_m: [ADAM_SIZE]f32
	ref_v: [ADAM_SIZE]f32

	opt := ml.optimizer_make(learning_rate=ADAM_LR, beta1=ADAM_B1, beta2=ADAM_B2, epsilon=ADAM_EPS, weight_decay=ADAM_WD, accumulation_steps=ADAM_ACCUM_STEPS)
	accum: [ADAM_SIZE]f32
	for micro in 1 ..= ADAM_ACCUM_STEPS * ADAM_ACCUM_WINDOWS {
		for i in 0 ..< ADAM_SIZE {
			accum[i] += _adam_grad(micro, i)
		}
		ml.set_bytes(param, .Gradient, mem.slice_to_bytes(accum[:]))

		stepped := ml.optimizer_step(&opt)
		testing.expect_value(t, stepped, micro % ADAM_ACCUM_STEPS == 0)
		if !stepped {
			continue
		}
		ml.update(&opt, param)

		window := micro / ADAM_ACCUM_STEPS
		bc1 := 1 - math.pow(ADAM_B1, f32(window))
		bc2 := 1 - math.pow(ADAM_B2, f32(window))
		for i in 0 ..< ADAM_SIZE {
			g := accum[i]
			ref_m[i] = ADAM_B1 * ref_m[i] + (1 - ADAM_B1) * g
			ref_v[i] = ADAM_B2 * ref_v[i] + (1 - ADAM_B2) * g * g
			m_hat := ref_m[i] / bc1
			v_hat := ref_v[i] / bc2
			ref_w[i] = ref_w[i] * (1 - ADAM_LR * ADAM_WD) - ADAM_LR * m_hat / (math.sqrt(v_hat) + ADAM_EPS)
			accum[i] = 0
		}

		zeroed: [ADAM_SIZE]f32
		ml.get_gradient(param, zeroed[:])
		for i in 0 ..< ADAM_SIZE {
			testing.expectf(t, zeroed[i] == 0, "gradient elem %d should be zeroed by update, got %v", i, zeroed[i])
		}
	}

	testing.expect_value(t, opt.iteration, u64(ADAM_ACCUM_WINDOWS))

	got: [ADAM_SIZE]f32
	ml.get_data(param, got[:])
	for i in 0 ..< ADAM_SIZE {
		a := f64(ref_w[i])
		b := f64(got[i])
		denom := max(max(abs(a), abs(b)), 1e-4)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= 1e-4,
			"adam accum elem %d ref=%.7g got=%.7g rel_err=%.4g", i, a, b, rel)
	}

	ml.optimizer_destroy(&opt)
	ml.destroy(param)
}
