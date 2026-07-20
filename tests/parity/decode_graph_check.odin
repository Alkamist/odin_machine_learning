package ml_parity_tests

import "core:testing"

import ml   "../.."
import cuda "../../backends/cuda"

DECODE_GRAPH_ROWS  :: 8
DECODE_GRAPH_COLS  :: 4
DECODE_GRAPH_STEPS :: 6

@(test)
test_decode_graph_select :: proc(t: ^testing.T) {
	if !_cuda_ready(t, "decode graph select test") {
		return
	}

	ctx := cuda.context_create(decode_graph=true)
	previous := ml.context_begin(ctx)

	table_data: [DECODE_GRAPH_ROWS * DECODE_GRAPH_COLS]f32
	for i in 0 ..< len(table_data) {
		table_data[i] = f32(i) * 0.25 - 3.0
	}

	table := ml.alloc(.F32, {DECODE_GRAPH_ROWS, DECODE_GRAPH_COLS}, persistent=true, buffers={.Data})
	ml.set_data(table, table_data[:])

	for step in 0 ..< DECODE_GRAPH_STEPS {
		ml.pass_begin(training=false)

		indices := [2]int{step % DECODE_GRAPH_ROWS, (step * 3 + 1) % DECODE_GRAPH_ROWS}
		output  := ml.select(table, indices[:])

		got: [2 * DECODE_GRAPH_COLS]f32
		ml.get_data(output, got[:])

		for row, r in indices {
			for c in 0 ..< DECODE_GRAPH_COLS {
				expected := table_data[row * DECODE_GRAPH_COLS + c]
				actual   := got[r * DECODE_GRAPH_COLS + c]
				testing.expectf(t, transmute(u32)actual == transmute(u32)expected,
					"step %v row %v col %v: expected %v got %v", step, r, c, expected, actual)
			}
		}
	}

	cuda.enable_decode_graph(false)
	ml.pass_begin(training=false)
	ml.destroy(table)
	ml.context_end(previous)
	cuda.context_destroy(ctx)
	cuda.device_destroy()
}
