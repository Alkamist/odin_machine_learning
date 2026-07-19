package ml_tests

import "core:mem"
import "core:thread"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

PARALLEL_THREADS  :: 4
PARALLEL_BATCH    :: 8
PARALLEL_IN_DIM   :: 4
PARALLEL_HIDDEN   :: 8
PARALLEL_OUT_DIM  :: 3
PARALLEL_OUT_SIZE :: PARALLEL_BATCH * PARALLEL_OUT_DIM
PARALLEL_CTX_SIZE :: 4 * 1024 * 1024

Parallel_Model :: struct {
	w1: ml.Tensor,
	b1: ml.Tensor,
	w2: ml.Tensor,
	b2: ml.Tensor,
}

Parallel_Payload :: struct {
	model:     ^Parallel_Model,
	input:     []f32,
	result:    []f32,
	allocator: mem.Allocator,
}

_parallel_fill :: proc(t: ml.Tensor, count: int) {
	data := make([]f32, count)
	defer delete(data)
	for i in 0 ..< count {
		data[i] = f32((i * 13 + 7) % 23) * 0.01
	}
	ml.set_data(t, data)
}

_parallel_model_build :: proc() -> (model: Parallel_Model) {
	model.w1 = ml.alloc(.F32, {PARALLEL_HIDDEN, PARALLEL_IN_DIM}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	model.b1 = ml.alloc(.F32, {PARALLEL_HIDDEN}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	model.w2 = ml.alloc(.F32, {PARALLEL_OUT_DIM, PARALLEL_HIDDEN}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	model.b2 = ml.alloc(.F32, {PARALLEL_OUT_DIM}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)

	_parallel_fill(model.w1, PARALLEL_HIDDEN * PARALLEL_IN_DIM)
	_parallel_fill(model.b1, PARALLEL_HIDDEN)
	_parallel_fill(model.w2, PARALLEL_OUT_DIM * PARALLEL_HIDDEN)
	_parallel_fill(model.b2, PARALLEL_OUT_DIM)

	return
}

_parallel_model_destroy :: proc(model: Parallel_Model) {
	ml.destroy(model.w1)
	ml.destroy(model.b1)
	ml.destroy(model.w2)
	ml.destroy(model.b2)
}

_parallel_forward :: proc(model: ^Parallel_Model, input: []f32, result: []f32) {
	ml.clear()
	x      := ml.tensor(input, []int{PARALLEL_BATCH, PARALLEL_IN_DIM})
	hidden := ml.relu(ml.linear(x, model.w1, bias=model.b1))
	output := ml.linear(hidden, model.w2, bias=model.b2)
	ml.get_data(output, result)
}

_parallel_worker :: proc(payload: ^Parallel_Payload) {
	context.allocator = payload.allocator
	ctx := cpu.context_create(PARALLEL_CTX_SIZE)
	previous := ml.context_begin(ctx)
	_parallel_forward(payload.model, payload.input, payload.result)
	ml.context_end(previous)
	cpu.context_destroy(ctx)
}

@(test)
test_parallel_inference :: proc(t: ^testing.T) {
	cpu.set_thread_count(4)
	defer cpu.set_thread_count(1)

	input: [PARALLEL_BATCH * PARALLEL_IN_DIM]f32
	for i in 0 ..< len(input) {
		input[i] = f32((i * 7 + 3) % 17) * 0.05 - 0.4
	}

	main_ctx := cpu.context_create(PARALLEL_CTX_SIZE)

	previous := ml.context_begin(main_ctx)
	model := _parallel_model_build()
	reference: [PARALLEL_OUT_SIZE]f32
	_parallel_forward(&model, input[:], reference[:])
	ml.context_end(previous)

	results:  [PARALLEL_THREADS][PARALLEL_OUT_SIZE]f32
	payloads: [PARALLEL_THREADS]Parallel_Payload
	threads:  [PARALLEL_THREADS]^thread.Thread

	for i in 0 ..< PARALLEL_THREADS {
		payloads[i] = {model=&model, input=input[:], result=results[i][:], allocator=context.allocator}
		threads[i]  = thread.create_and_start_with_poly_data(&payloads[i], _parallel_worker)
	}

	for i in 0 ..< PARALLEL_THREADS {
		thread.join(threads[i])
		thread.destroy(threads[i])
	}

	for i in 0 ..< PARALLEL_THREADS {
		for e in 0 ..< PARALLEL_OUT_SIZE {
			testing.expectf(t, transmute(u32)results[i][e] == transmute(u32)reference[e],
				"thread %v output must be bit-identical to reference, index %v: %v vs %v",
				i, e, results[i][e], reference[e])
		}
	}

	previous = ml.context_begin(main_ctx)
	_parallel_model_destroy(model)
	ml.context_end(previous)
	cpu.context_destroy(main_ctx)
}
