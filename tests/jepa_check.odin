package ml_tests

import "core:math"
import "core:mem"
import "core:os"
import "core:testing"

import ml   "../"
import cpu  "../backends/cpu"
import jepa "../networks/jepa"

JEPA_TEST_PATH :: "test_jepa_roundtrip.safetensors"

@(test)
test_jepa_checkpoint_roundtrip :: proc(t: ^testing.T) {
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	context.allocator = mem.tracking_allocator(&track)

	{
		ctx := cpu.context_create(8 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		cfg := jepa.DEFAULT_CONFIG
		cfg.state_size  = 3
		cfg.action_size = 2
		cfg.hidden_size = 8
		cfg.latent_size = 4

		model := jepa.make(cfg)

		testing.expect(t, ml.has_gradient(model.encoder.layers[0].weight), "encoder weights should carry gradient buffers")
		testing.expect(t, !ml.has_gradient(model.target_encoder.layers[0].weight), "target encoder weights should not carry gradient buffers")
		testing.expect_value(t, ml.registry_element_count(&model.target_encoder.params), 0)
		testing.expect_value(t, ml.registry_element_count(&model.target_encoder.params, flags=ml.Parameter_Flags{.Checkpoint}), ml.registry_element_count(&model.encoder.params))

		batch_size  := 4
		states      := make([]f32, batch_size * cfg.state_size)
		actions     := make([]f32, batch_size * cfg.action_size)
		next_states := make([]f32, batch_size * cfg.state_size)
		for &v, i in states {
			v = f32(i) * 0.1 - 0.5
		}
		for &v, i in actions {
			v = f32(i % 2)
		}
		for &v, i in next_states {
			v = f32(i) * 0.05 - 0.3
		}

		batch   := jepa.Batch{batch_size=batch_size, states=states, actions=actions, next_states=next_states}
		metrics := jepa.train_step(model, batch)
		testing.expect(t, !math.is_nan(metrics.loss), "train_step loss should be finite")

		opt := ml.optimizer_make(learning_rate=1e-3)
		stepped := ml.optimizer_step(&opt)
		testing.expect(t, stepped, "optimizer_step should fire every step by default")
		jepa.update(&opt, model)
		jepa.ema_update(model, 0.99)
		ml.clear()

		saved := jepa.save(model, JEPA_TEST_PATH, opt=&opt, iteration=7, decoder_iteration=3)
		testing.expect(t, saved, "jepa.save should succeed")
		defer os.remove(JEPA_TEST_PATH)

		restored_opt: ml.Optimizer
		restored, iteration, decoder_iteration, load_err := jepa.load(cfg, JEPA_TEST_PATH, opt=&restored_opt)
		testing.expect_value(t, load_err, ml.Checkpoint_Error.None)
		testing.expect_value(t, iteration, u64(7))
		testing.expect_value(t, decoder_iteration, u64(3))
		testing.expect_value(t, restored_opt.iteration, u64(1))

		gathered:          ml.Registry
		restored_gathered: ml.Registry
		jepa.parameters(model, &gathered)
		jepa.decoder_parameters(model, &gathered)
		jepa.parameters(restored, &restored_gathered)
		jepa.decoder_parameters(restored, &restored_gathered)
		testing.expect_value(t, len(restored_gathered.parameters), len(gathered.parameters))

		for p, i in gathered.parameters {
			q := restored_gathered.parameters[i]
			testing.expect_value(t, q.name, p.name)
			saved_values    := make([]f32, p.tensor.count)
			restored_values := make([]f32, q.tensor.count)
			ml.get_data(p.tensor, saved_values)
			ml.get_data(q.tensor, restored_values)
			for j in 0 ..< p.tensor.count {
				testing.expect_value(t, restored_values[j], saved_values[j])
			}
			delete(saved_values)
			delete(restored_values)
		}

		ml.registry_destroy(&gathered)
		ml.registry_destroy(&restored_gathered)
		delete(states)
		delete(actions)
		delete(next_states)
		ml.optimizer_destroy(&opt)
		ml.optimizer_destroy(&restored_opt)
		jepa.destroy(model)
		jepa.destroy(restored)
	}

	testing.expectf(t, len(track.allocation_map) == 0, "expected no leaks, got %d live allocations", len(track.allocation_map))
	testing.expectf(t, len(track.bad_free_array) == 0, "expected no bad frees, got %d", len(track.bad_free_array))

	mem.tracking_allocator_clear(&track)
	context.allocator = track.backing
	mem.tracking_allocator_destroy(&track)
}
