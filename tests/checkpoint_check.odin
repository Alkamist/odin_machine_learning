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
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	param := ml.alloc(.F32, {CHECKPOINT_TEST_SIZE}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	registry: ml.Registry
	ml.parameter_register(&registry, "", "weight", param, init=ml.Init_None{}, flags=ml.PARAMETER_DEFAULT_FLAGS + {.Owned})
	metadata: map[string]string
	saved := ml.checkpoint_save(CHECKPOINT_TEST_PATH, &registry, &opt, metadata)
	testing.expect_value(t, saved, ml.Checkpoint_Error.None)
	defer os.remove(CHECKPOINT_TEST_PATH)

	restored_param := ml.alloc(.F32, {CHECKPOINT_TEST_SIZE}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	restored_registry: ml.Registry
	ml.parameter_register(&restored_registry, "", "weight", restored_param, init=ml.Init_None{}, flags=ml.PARAMETER_DEFAULT_FLAGS + {.Owned})
	restored_opt := ml.optimizer_make(learning_rate=ADAM_LR, beta1=ADAM_B1, beta2=ADAM_B2, epsilon=ADAM_EPS, weight_decay=ADAM_WD)
	loaded_metadata, load_err := ml.checkpoint_load(CHECKPOINT_TEST_PATH, &restored_registry, &restored_opt)
	testing.expect_value(t, load_err, ml.Checkpoint_Error.None)
	defer ml.checkpoint_metadata_destroy(loaded_metadata)

	missing_metadata, missing_err := ml.checkpoint_load("does_not_exist_checkpoint.safetensors", &restored_registry, &restored_opt)
	testing.expect_value(t, missing_err, ml.Checkpoint_Error.Not_Found)
	ml.checkpoint_metadata_destroy(missing_metadata)

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
	ml.registry_destroy(&registry)
	ml.registry_destroy(&restored_registry)
}
