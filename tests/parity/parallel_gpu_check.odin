package ml_parity_tests

import "core:mem"
import "core:thread"
import "core:testing"

import ml   "../.."
import cuda "../../backends/cuda"

GPU_PARALLEL_THREADS  :: 4
GPU_PARALLEL_BATCH    :: 8
GPU_PARALLEL_IN_DIM   :: 4
GPU_PARALLEL_HIDDEN   :: 8
GPU_PARALLEL_OUT_DIM  :: 3
GPU_PARALLEL_OUT_SIZE :: GPU_PARALLEL_BATCH * GPU_PARALLEL_OUT_DIM

Gpu_Parallel_Model :: struct {
	w1: ml.Tensor,
	b1: ml.Tensor,
	w2: ml.Tensor,
	b2: ml.Tensor,
}

Gpu_Parallel_Payload :: struct {
	model:     ^Gpu_Parallel_Model,
	input:     []f32,
	result:    []f32,
	allocator: mem.Allocator,
}

_gpu_parallel_fill :: proc(t: ml.Tensor, count: int) {
	data := make([]f32, count)
	defer delete(data)
	for i in 0 ..< count {
		data[i] = f32((i * 13 + 7) % 23) * 0.01
	}
	ml.set_data(t, data)
}

_gpu_parallel_model_build :: proc() -> (model: Gpu_Parallel_Model) {
	model.w1 = ml.alloc(.F32, {GPU_PARALLEL_HIDDEN, GPU_PARALLEL_IN_DIM}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	model.b1 = ml.alloc(.F32, {GPU_PARALLEL_HIDDEN}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	model.w2 = ml.alloc(.F32, {GPU_PARALLEL_OUT_DIM, GPU_PARALLEL_HIDDEN}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	model.b2 = ml.alloc(.F32, {GPU_PARALLEL_OUT_DIM}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)

	_gpu_parallel_fill(model.w1, GPU_PARALLEL_HIDDEN * GPU_PARALLEL_IN_DIM)
	_gpu_parallel_fill(model.b1, GPU_PARALLEL_HIDDEN)
	_gpu_parallel_fill(model.w2, GPU_PARALLEL_OUT_DIM * GPU_PARALLEL_HIDDEN)
	_gpu_parallel_fill(model.b2, GPU_PARALLEL_OUT_DIM)

	return
}

_gpu_parallel_model_destroy :: proc(model: Gpu_Parallel_Model) {
	ml.destroy(model.w1)
	ml.destroy(model.b1)
	ml.destroy(model.w2)
	ml.destroy(model.b2)
}

_gpu_parallel_forward :: proc(model: ^Gpu_Parallel_Model, input: []f32, result: []f32) {
	ml.clear()
	x      := ml.tensor(input, []int{GPU_PARALLEL_BATCH, GPU_PARALLEL_IN_DIM})
	hidden := ml.relu(ml.linear(x, model.w1, bias=model.b1))
	output := ml.linear(hidden, model.w2, bias=model.b2)
	ml.get_data(output, result)
}

_gpu_parallel_worker :: proc(payload: ^Gpu_Parallel_Payload) {
	context.allocator = payload.allocator
	ctx := cuda.context_create()
	previous := ml.context_begin(ctx)
	_gpu_parallel_forward(payload.model, payload.input, payload.result)
	ml.context_end(previous)
	cuda.context_destroy(ctx)
	cuda.device_destroy()
}

@(test)
test_parallel_gpu_inference :: proc(t: ^testing.T) {
	if !_cuda_ready(t, "parallel GPU inference test") {
		return
	}

	input: [GPU_PARALLEL_BATCH * GPU_PARALLEL_IN_DIM]f32
	for i in 0 ..< len(input) {
		input[i] = f32((i * 7 + 3) % 17) * 0.05 - 0.4
	}

	main_ctx := cuda.context_create()

	previous := ml.context_begin(main_ctx)
	model := _gpu_parallel_model_build()
	reference: [GPU_PARALLEL_OUT_SIZE]f32
	_gpu_parallel_forward(&model, input[:], reference[:])
	ml.context_end(previous)

	results:  [GPU_PARALLEL_THREADS][GPU_PARALLEL_OUT_SIZE]f32
	payloads: [GPU_PARALLEL_THREADS]Gpu_Parallel_Payload
	threads:  [GPU_PARALLEL_THREADS]^thread.Thread

	for i in 0 ..< GPU_PARALLEL_THREADS {
		payloads[i] = {model=&model, input=input[:], result=results[i][:], allocator=context.allocator}
		threads[i]  = thread.create_and_start_with_poly_data(&payloads[i], _gpu_parallel_worker)
	}

	for i in 0 ..< GPU_PARALLEL_THREADS {
		thread.join(threads[i])
		thread.destroy(threads[i])
	}

	for i in 0 ..< GPU_PARALLEL_THREADS {
		for e in 0 ..< GPU_PARALLEL_OUT_SIZE {
			testing.expectf(t, transmute(u32)results[i][e] == transmute(u32)reference[e],
				"thread %v output must be bit-identical to reference, index %v: %v vs %v",
				i, e, results[i][e], reference[e])
		}
	}

	previous = ml.context_begin(main_ctx)
	_gpu_parallel_model_destroy(model)
	ml.context_end(previous)
	cuda.context_destroy(main_ctx)
	cuda.device_destroy()
}
