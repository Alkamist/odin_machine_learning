package ml_tests

import "core:math/rand"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"
import     "../networks/mlp"

DETERMINISM_STEPS       :: 3
DETERMINISM_BATCH       :: 8
DETERMINISM_PARAM_COUNT :: 58

_seeded_train_run :: proc(seed: u64, params_out: []f32) {
	state := rand.create(seed)
	context.random_generator = rand.default_random_generator(&state)

	ctx := cpu.context_create(4 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	model := mlp.make(4, 8, 2)
	defer mlp.destroy(model)

	opt := ml.optimizer_make()
	defer ml.optimizer_destroy(&opt)

	input_data:  [DETERMINISM_BATCH * 4]f32
	target_data: [DETERMINISM_BATCH * 2]f32

	for _ in 0 ..< DETERMINISM_STEPS {
		for i in 0 ..< len(input_data) {
			input_data[i] = rand.float32_normal(0, 1)
		}
		for i in 0 ..< len(target_data) {
			target_data[i] = rand.float32_normal(0, 1)
		}

		ml.pass_begin(training=true)
		x      := ml.tensor(input_data[:], []int{DETERMINISM_BATCH, 4})
		target := ml.tensor(target_data[:], []int{DETERMINISM_BATCH, 2})

		prediction := mlp.forward(model, ml.dropout(x, 0.25))
		loss       := ml.mean(ml.mean_squared_error(prediction, target))
		ml.backward(loss)

		if ml.optimizer_step(&opt) {
			mlp.update(&opt, model)
		}
	}

	ml.registry_read(&model.params, params_out)
}

@(test)
test_determinism :: proc(t: ^testing.T) {
	first:  [DETERMINISM_PARAM_COUNT]f32
	second: [DETERMINISM_PARAM_COUNT]f32
	other:  [DETERMINISM_PARAM_COUNT]f32

	_seeded_train_run(42, first[:])
	_seeded_train_run(42, second[:])
	_seeded_train_run(7, other[:])

	for i in 0 ..< DETERMINISM_PARAM_COUNT {
		testing.expectf(t, transmute(u32)first[i] == transmute(u32)second[i], "same seed must give bit-identical parameters, index %v: %v vs %v", i, first[i], second[i])
	}

	differs := false
	for i in 0 ..< DETERMINISM_PARAM_COUNT {
		if transmute(u32)other[i] != transmute(u32)first[i] {
			differs = true
			break
		}
	}
	testing.expect(t, differs, "a different seed must change the trained parameters")
}
