package ml_tests

import "core:mem"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

@(test)
test_i32_tensor_round_trip :: proc(t: ^testing.T) {
	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, context.allocator)
	context.allocator = mem.tracking_allocator(&track)

	{
		ctx := cpu.context_create(1 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		ml.pass_begin(training=false)

		flat := ml.tensor([]i32{5, -3, 7, 11})
		testing.expect(t, flat.type == .I32, "flat tensor must be I32")
		testing.expect(t, flat.rank == 1 && flat.shape[0] == 4, "flat tensor shape must be [4]")

		got: [4]i32
		ml.get_data(flat, got[:])
		expected := [4]i32{5, -3, 7, 11}
		for i in 0 ..< 4 {
			testing.expectf(t, got[i] == expected[i], "flat[%v] expected %v got %v", i, expected[i], got[i])
		}

		ml.set_data(flat, []i32{1, 2, 3, 4})
		ml.get_data(flat, got[:])
		for i in 0 ..< 4 {
			testing.expectf(t, got[i] == i32(i + 1), "overwritten flat[%v] expected %v got %v", i, i + 1, got[i])
		}

		shaped := ml.tensor([]i32{10, 20, 30, 40, 50, 60}, []int{2, 3})
		testing.expect(t, shaped.rank == 2 && shaped.shape[0] == 2 && shaped.shape[1] == 3, "shaped tensor must be [2,3]")

		got6: [6]i32
		ml.get_data(shaped, got6[:])
		shaped_expected := [6]i32{10, 20, 30, 40, 50, 60}
		for i in 0 ..< 6 {
			testing.expectf(t, got6[i] == shaped_expected[i], "shaped[%v] expected %v got %v", i, shaped_expected[i], got6[i])
		}
	}

	testing.expectf(t, len(track.allocation_map) == 0, "expected no leaks, got %d live allocations", len(track.allocation_map))
	testing.expectf(t, len(track.bad_free_array) == 0, "expected no bad frees, got %d", len(track.bad_free_array))

	mem.tracking_allocator_clear(&track)
	context.allocator = track.backing
	mem.tracking_allocator_destroy(&track)
}

@(test)
test_i32_tensor_has_no_gradient :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	ml.pass_begin(training=true)

	ints := ml.zeros(.I32, []int{4})
	testing.expect(t, !ml.has_gradient(ints), "I32 tensor must not have a gradient buffer under training")

	floats := ml.zeros(.F32, []int{4})
	testing.expect(t, ml.has_gradient(floats), "F32 tensor must have a gradient buffer under training")
}

@(test)
test_i32_select_tensor_matches_ints :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	ml.pass_begin(training=false)

	input := ml.zeros(.F32, []int{4, 3})
	ml.set_data(input, []f32{
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11,
	})

	out_ints := ml.select(input, []int{2, 0, 2, 1})

	indices := ml.tensor([]i32{2, 0, 2, 1})
	out_tensor := ml.select(input, indices)

	a: [12]f32
	b: [12]f32
	ml.get_data(out_ints, a[:])
	ml.get_data(out_tensor, b[:])
	for i in 0 ..< 12 {
		testing.expectf(t, a[i] == b[i], "select output[%v] mismatch: ints=%v tensor=%v", i, a[i], b[i])
	}
}

@(test)
test_i32_cross_entropy_tensor_matches_ints :: proc(t: ^testing.T) {
	logits_data := []f32{
		1.0, 2.0, 0.5, -1.0,
		0.3, 0.1, 2.5, 1.2,
		-0.5, 0.7, 1.1, 3.0,
	}
	targets := []int{2, 0, 3}

	loss_ints:   f32
	loss_tensor: f32
	grads_ints:   [12]f32
	grads_tensor: [12]f32

	{
		ctx := cpu.context_create(1 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		ml.pass_begin(training=true)
		logits := ml.zeros(.F32, []int{3, 4})
		ml.set_data(logits, logits_data)
		loss := ml.mean(ml.cross_entropy(logits, targets))
		ml.backward(loss)

		out: [1]f32
		ml.get_data(loss, out[:])
		loss_ints = out[0]
		ml.get_gradient(logits, grads_ints[:])
	}

	{
		ctx := cpu.context_create(1 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		ml.pass_begin(training=true)
		logits := ml.zeros(.F32, []int{3, 4})
		ml.set_data(logits, logits_data)
		target_tensor := ml.tensor([]i32{2, 0, 3})
		loss := ml.mean(ml.cross_entropy(logits, target_tensor))
		ml.backward(loss)

		out: [1]f32
		ml.get_data(loss, out[:])
		loss_tensor = out[0]
		ml.get_gradient(logits, grads_tensor[:])
	}

	testing.expectf(t, loss_ints == loss_tensor, "cross_entropy loss mismatch: ints=%v tensor=%v", loss_ints, loss_tensor)
	for i in 0 ..< 12 {
		testing.expectf(t, grads_ints[i] == grads_tensor[i], "cross_entropy grad[%v] mismatch: ints=%v tensor=%v", i, grads_ints[i], grads_tensor[i])
	}
}
