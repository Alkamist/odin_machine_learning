package ml_tests

import "core:mem"
import "core:math"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

ADAM_SIZE  :: 8
ADAM_STEPS :: 12

ADAM_LR  :: f32(0.01)
ADAM_B1  :: f32(0.9)
ADAM_B2  :: f32(0.999)
ADAM_EPS :: f32(1e-8)
ADAM_WD  :: f32(0.1)

_adam_grad :: proc(step, index: int) -> f32 {
	return (f32((step * 7 + index * 3) % 11) - 5) * 0.03
}

@(test)
test_adam_update :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	ml.context_begin(ctx)

	param := ml.make(.F32, {ADAM_SIZE})

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
	ml.context_end()
	cpu.context_destroy(ctx)
}
