package ml_tests

import "core:mem"
import "core:os"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

CHECKPOINT_TEST_PATH  :: "test_checkpoint_roundtrip.safetensors"
CHECKPOINT_TEST_SIZE  :: 8
CHECKPOINT_TEST_STEPS :: 5

@(test)
test_checkpoint_optimizer_iteration_roundtrip :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	ml.context_begin(ctx)

	param := ml.make(.F32, {CHECKPOINT_TEST_SIZE})
	init_w: [CHECKPOINT_TEST_SIZE]f32
	for i in 0 ..< CHECKPOINT_TEST_SIZE {
		init_w[i] = f32(i) * 0.25 - 1
	}
	ml.set_data(param, init_w[:])

	opt := ml.optimizer_make(learning_rate=ADAM_LR, beta1=ADAM_B1, beta2=ADAM_B2, epsilon=ADAM_EPS, weight_decay=ADAM_WD)
	grad: [CHECKPOINT_TEST_SIZE]f32
	for step in 1 ..= CHECKPOINT_TEST_STEPS {
		for i in 0 ..< CHECKPOINT_TEST_SIZE {
			grad[i] = _adam_grad(step, i)
		}
		ml.set_bytes(param, .Gradient, mem.slice_to_bytes(grad[:]))
		_ = ml.optimizer_step(&opt)
		ml.update(&opt, param)
	}
	testing.expect_value(t, opt.iteration, u64(CHECKPOINT_TEST_STEPS))

	params := []ml.Parameter{{name="weight", tensor=param}}
	metadata: map[string]string
	saved := ml.checkpoint_save(CHECKPOINT_TEST_PATH, params, &opt, metadata)
	testing.expect(t, saved, "checkpoint_save should succeed")
	defer os.remove(CHECKPOINT_TEST_PATH)

	restored_param := ml.make(.F32, {CHECKPOINT_TEST_SIZE})
	restored_params := []ml.Parameter{{name="weight", tensor=restored_param}}
	restored_opt := ml.optimizer_make(learning_rate=ADAM_LR, beta1=ADAM_B1, beta2=ADAM_B2, epsilon=ADAM_EPS, weight_decay=ADAM_WD)
	loaded_metadata, loaded := ml.checkpoint_load(CHECKPOINT_TEST_PATH, restored_params, &restored_opt)
	testing.expect(t, loaded, "checkpoint_load should succeed")
	defer ml.checkpoint_metadata_destroy(loaded_metadata)

	testing.expect_value(t, restored_opt.iteration, u64(CHECKPOINT_TEST_STEPS))

	saved_w:    [CHECKPOINT_TEST_SIZE]f32
	restored_w: [CHECKPOINT_TEST_SIZE]f32
	ml.get_data(param, saved_w[:])
	ml.get_data(restored_param, restored_w[:])
	for i in 0 ..< CHECKPOINT_TEST_SIZE {
		testing.expect_value(t, restored_w[i], saved_w[i])
	}

	stepped := ml.optimizer_step(&restored_opt)
	testing.expect(t, stepped, "optimizer_step should fire every step by default")
	testing.expect_value(t, restored_opt.iteration, u64(CHECKPOINT_TEST_STEPS + 1))

	ml.optimizer_destroy(&opt)
	ml.optimizer_destroy(&restored_opt)
	ml.destroy(param)
	ml.destroy(restored_param)
	ml.context_end()
	cpu.context_destroy(ctx)
}
