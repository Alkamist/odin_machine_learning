package ml_tests

import "core:mem"
import "core:os"
import "core:strings"
import "core:testing"

import ml    "../"
import cpu   "../backends/cpu"
import gemma "../networks/gemma"

GEMMA_LORA_TEST_PATH :: "test_gemma_lora_roundtrip.safetensors"

_tiny_gemma_config :: proc(allocator := context.allocator) -> (cfg: gemma.Config) {
	cfg = {
		num_hidden_layers           = 2,
		hidden_size                 = 8,
		intermediate_size           = 16,
		num_attention_heads         = 2,
		num_key_value_heads         = 1,
		head_dim_sliding            = 4,
		head_dim_full               = 4,
		vocab_size                  = 32,
		max_position_embeddings     = 16,
		sliding_window              = 4,
		hidden_size_per_layer_input = 4,
		num_kv_shared_layers        = 0,
		rope_base_sliding           = 10000,
		rope_base_full              = 10000,
		rope_fraction_full          = 1,
		rms_norm_eps                = 1e-6,
		final_logit_softcapping     = 0,
		tie_word_embeddings         = true,
	}
	cfg.layer_types = make([]gemma.Layer_Type, cfg.num_hidden_layers, allocator)
	cfg.layer_types[0] = .Sliding
	cfg.layer_types[1] = .Full
	return
}

@(test)
test_gemma_lora_checkpoint_roundtrip :: proc(t: ^testing.T) {
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	context.allocator = mem.tracking_allocator(&track)

	{
		ctx := cpu.context_create(64 * 1024 * 1024)
		ml.context_begin(ctx)

		cfg := _tiny_gemma_config()
		lora_cfg := gemma.LoRA_Config{rank=2, alpha=4, targets={.Q, .V}}
		model := gemma.make(cfg, dtype=.F32, for_training=true, lora_cfg=lora_cfg)
		gemma.randomize(model)

		params := make([dynamic]ml.Parameter)
		gemma.parameters(model, &params)
		testing.expectf(t, len(params) == 8, "expected 8 LoRA-only parameters under QLoRA, got %d", len(params))
		for p in params {
			testing.expectf(t, strings.contains(p.name, "lora_"), "expected LoRA-only parameter, got %q", p.name)
		}

		ml.clear(training=true)
		logits := gemma.forward(model, []int{1, 2, 3})
		loss   := ml.mean(ml.cross_entropy(logits, []int{2, 3, 4}))
		ml.backward(loss)

		opt: ml.Optimizer
		stepped := ml.optimizer_step(&opt, period=1, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0)
		testing.expect(t, stepped, "optimizer_step should fire with period=1")
		gemma.update(&opt, model)
		ml.clear()

		metadata: map[string]string
		saved := ml.checkpoint_save(GEMMA_LORA_TEST_PATH, params[:], &opt, metadata)
		testing.expect(t, saved, "checkpoint_save should succeed")
		defer os.remove(GEMMA_LORA_TEST_PATH)

		restored := gemma.make(cfg, dtype=.F32, for_training=true, lora_cfg=lora_cfg)
		restored_params := make([dynamic]ml.Parameter)
		gemma.parameters(restored, &restored_params)

		restored_opt: ml.Optimizer
		loaded_metadata, loaded := ml.checkpoint_load(GEMMA_LORA_TEST_PATH, restored_params[:], &restored_opt)
		testing.expect(t, loaded, "checkpoint_load should succeed")
		testing.expect_value(t, restored_opt.iteration, u64(1))

		total := ml.parameters_len(params[:])
		saved_values    := make([]f32, total)
		restored_values := make([]f32, total)
		ml.parameters_read(params[:], saved_values)
		ml.parameters_read(restored_params[:], restored_values)
		for i in 0 ..< total {
			testing.expect_value(t, restored_values[i], saved_values[i])
		}

		delete(saved_values)
		delete(restored_values)
		ml.checkpoint_metadata_destroy(loaded_metadata)
		for p in params {
			delete(p.name)
		}
		delete(params)
		for p in restored_params {
			delete(p.name)
		}
		delete(restored_params)
		ml.optimizer_destroy(&opt)
		ml.optimizer_destroy(&restored_opt)
		gemma.destroy(model)
		gemma.destroy(restored)
		gemma.config_destroy(cfg)
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	testing.expectf(t, len(track.allocation_map) == 0, "expected no leaks, got %d live allocations", len(track.allocation_map))
	testing.expectf(t, len(track.bad_free_array) == 0, "expected no bad frees, got %d", len(track.bad_free_array))

	mem.tracking_allocator_clear(&track)
	context.allocator = track.backing
	mem.tracking_allocator_destroy(&track)
}
