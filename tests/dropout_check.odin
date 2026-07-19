package ml_tests

import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

@(test)
test_dropout :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	DROPOUT_SIZE :: 1024
	RATE         :: f32(0.5)

	input_data: [DROPOUT_SIZE]f32
	for i in 0 ..< DROPOUT_SIZE {
		input_data[i] = f32(i % 7) + 1
	}

	ml.clear(training=true)
	x := ml.tensor(input_data[:])
	y := ml.dropout(x, RATE)

	output: [DROPOUT_SIZE]f32
	ml.get_data(y, output[:])

	scale   := 1 / (1 - RATE)
	dropped := 0
	for i in 0 ..< DROPOUT_SIZE {
		if output[i] == 0 {
			dropped += 1
		} else {
			testing.expectf(t, output[i] == input_data[i] * scale, "kept element %v should be scaled by %v, got %v from %v", i, scale, output[i], input_data[i])
		}
	}
	testing.expectf(t, dropped > DROPOUT_SIZE / 4 && dropped < DROPOUT_SIZE * 3 / 4, "dropout rate 0.5 should drop roughly half, dropped %v of %v", dropped, DROPOUT_SIZE)

	ml.clear()
	x_infer := ml.tensor(input_data[:])
	y_infer := ml.dropout(x_infer, RATE)
	testing.expect(t, y_infer.buffers[.Data] == x_infer.buffers[.Data], "dropout must be identity outside training")
}
