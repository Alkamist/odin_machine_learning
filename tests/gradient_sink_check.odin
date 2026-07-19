package ml_tests

import "core:mem"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

@(test)
test_backward_skips_gradient_sinks :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	ml.clear(training=true)

	shape := [1]int{4}
	x := ml.zeros(.F32, shape[:])
	ml.set_data(x, []f32{1, 2, 3, 4})

	frozen := ml.scratch(.F32, shape[:])
	ml.set_data(frozen, []f32{5, 6, 7, 8})

	summed := ml.add(x, frozen)
	scaled := ml.mul(summed, frozen)
	loss   := ml.mean(scaled)
	ml.backward(loss)

	x_grad: [4]f32
	ml.get_bytes(x, .Gradient, mem.slice_to_bytes(x_grad[:]))
	for i in 0 ..< 4 {
		testing.expectf(t, x_grad[i] != 0, "x gradient[%v] should be nonzero after backward through sinks", i)
	}
}
