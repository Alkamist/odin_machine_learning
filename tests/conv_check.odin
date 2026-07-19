package ml_tests

import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

@(test)
test_conv2d_forward :: proc(t: ^testing.T) {
	ctx := cpu.context_create(4 * 1024 * 1024)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	ml.clear()

	input_data  := [9]f32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	weight_data := [4]f32{1, 2, 3, 4}

	input  := ml.tensor(input_data[:], []int{1, 3, 3, 1})
	weight := ml.tensor(weight_data[:], []int{1, 2, 2, 1})

	output := ml.conv2d(input, weight, stride=1, padding=0)

	testing.expect(t, output.rank == 4, "conv2d output must be rank 4")
	expected_shape := [4]int{1, 2, 2, 1}
	for d in 0 ..< 4 {
		testing.expectf(t, output.shape[d] == expected_shape[d], "conv2d output dim %d: got %d want %d", d, output.shape[d], expected_shape[d])
	}

	got: [4]f32
	ml.get_data(output, got[:])

	expected := [4]f32{37, 47, 67, 77}
	for i in 0 ..< 4 {
		testing.expectf(t, abs(got[i] - expected[i]) < 1e-4, "conv2d output %d: got %v want %v", i, got[i], expected[i])
	}
}

@(test)
test_conv2d_bias_shape :: proc(t: ^testing.T) {
	ctx := cpu.context_create(4 * 1024 * 1024)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	ml.clear()

	input  := ml.zeros(.F32, {2, 5, 5, 3})
	weight := ml.zeros(.F32, {4, 2, 2, 3})
	bias   := ml.zeros(.F32, {4})

	output := ml.conv2d(input, weight, bias=bias, stride=1, padding=0)

	expected_shape := [4]int{2, 4, 4, 4}
	testing.expect(t, output.rank == 4, "conv2d+bias output must be rank 4")
	for d in 0 ..< 4 {
		testing.expectf(t, output.shape[d] == expected_shape[d], "conv2d+bias output dim %d: got %d want %d", d, output.shape[d], expected_shape[d])
	}
}

@(test)
test_conv1d_shape :: proc(t: ^testing.T) {
	ctx := cpu.context_create(4 * 1024 * 1024)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	ml.clear()

	input  := ml.zeros(.F32, {1, 6, 3})
	weight := ml.zeros(.F32, {4, 2, 3})

	output := ml.conv1d(input, weight, stride=1, padding=0)

	expected_shape := [3]int{1, 5, 4}
	testing.expect(t, output.rank == 3, "conv1d output must be rank 3")
	for d in 0 ..< 3 {
		testing.expectf(t, output.shape[d] == expected_shape[d], "conv1d output dim %d: got %d want %d", d, output.shape[d], expected_shape[d])
	}
}
