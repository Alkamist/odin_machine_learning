package ml_tests

import "core:mem"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"
import     "../networks/mlp"

@(test)
test_registry_mlp_names :: proc(t: ^testing.T) {
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	context.allocator = mem.tracking_allocator(&track)

	{
		ctx := cpu.context_create(1 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		model := mlp.make(4, 8, 2)

		gathered: ml.Registry
		mlp.parameters(model, &gathered, prefix="encoder")

		expected := []string{
			"encoder.0.weight",
			"encoder.0.bias",
			"encoder.1.weight",
			"encoder.1.bias",
		}
		testing.expectf(t, len(gathered.parameters) == len(expected), "expected %d parameters, got %d", len(expected), len(gathered.parameters))
		for name, i in expected {
			testing.expectf(t, gathered.parameters[i].name == name, "parameter %d expected %q, got %q", i, name, gathered.parameters[i].name)
		}

		count := ml.registry_element_count(&model.params)
		testing.expectf(t, count == 58, "expected parameter count 58, got %d", count)

		ml.registry_destroy(&gathered)

		mlp.destroy(model)
	}

	testing.expectf(t, len(track.allocation_map) == 0, "expected no leaks, got %d live allocations", len(track.allocation_map))
	testing.expectf(t, len(track.bad_free_array) == 0, "expected no bad frees, got %d", len(track.bad_free_array))

	mem.tracking_allocator_clear(&track)
	context.allocator = track.backing
	mem.tracking_allocator_destroy(&track)
}
