package ml_parity_tests

import "core:fmt"
import "core:mem"
import "core:log"
import "core:testing"
import "core:math/rand"

import ml      "../.."
import cpu     "../../backends/cpu"
import cuda    "../../backends/cuda"
import cudadrv "../../backends/cuda/bindings/cuda"
import cases   "../cases"

CPU_CTX_SIZE :: 8 * 1024 * 1024

PARITY_TOL :: f64(1e-4)
REL_FLOOR  :: f64(1e-3)

_cuda_available :: proc() -> bool {
	if cudadrv.Init(0) != .SUCCESS {
		return false
	}
	count: i32
	if cudadrv.DeviceGetCount(&count) != .SUCCESS {
		return false
	}
	return count > 0
}

_output_count :: proc(tc: cases.Op_Test, inputs_data: [][]f32) -> int {
	ml.clear(training=true)

	tensors: [cases.MAX_INPUTS]ml.Tensor
	n := len(inputs_data)
	for i in 0 ..< n {
		shape     := tc.inputs[i].shape
		tensors[i] = ml.zeros(.F32, shape[:tc.inputs[i].rank])
		ml.set_data(tensors[i], inputs_data[i])
	}

	output := tc.run(tensors[:n])
	return ml.len(output)
}

_parity_eval :: proc(tc: cases.Op_Test, inputs_data: [][]f32, w: []f32, do_backward: bool, out_host: []f32, grads_host: [][]f32) {
	ml.clear(training=true)

	tensors: [cases.MAX_INPUTS]ml.Tensor
	n := len(inputs_data)
	for i in 0 ..< n {
		shape     := tc.inputs[i].shape
		tensors[i] = ml.zeros(.F32, shape[:tc.inputs[i].rank])
		ml.set_data(tensors[i], inputs_data[i])
	}

	output := tc.run(tensors[:n])
	count  := ml.len(output)
	ml.get_data(output, out_host[:count])

	if do_backward {
		out_shape := output.shape
		weights   := ml.zeros(.F32, out_shape[:output.rank])
		ml.set_data(weights, w[:count])
		weighted := ml.mul(output, weights)
		ml.backward(weighted)
		for i in 0 ..< n {
			if tc.inputs[i].check {
				ml.get_gradient(tensors[i], grads_host[i])
			}
		}
	}
}

_compare :: proc(t: ^testing.T, tc: cases.Op_Test, label: string, cpu_vals, cuda_vals: []f32) {
	for e in 0 ..< len(cpu_vals) {
		a := f64(cpu_vals[e])
		b := f64(cuda_vals[e])
		denom := max(max(abs(a), abs(b)), REL_FLOOR)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= PARITY_TOL,
			"%s: %s elem %d cpu=%.6g cuda=%.6g rel_err=%.4g (tol=%.3g)",
			tc.name, label, e, a, b, rel, PARITY_TOL)
	}
}

_run_parity :: proc(t: ^testing.T, tc: cases.Op_Test, cpu_ctx, cuda_ctx: ^ml.Context, do_backward: bool) {
	state := rand.create(tc.seed)
	context.random_generator = rand.default_random_generator(&state)

	n := tc.input_count

	inputs_data: [cases.MAX_INPUTS][]f32
	cpu_grads:   [cases.MAX_INPUTS][]f32
	cuda_grads:  [cases.MAX_INPUTS][]f32
	for i in 0 ..< n {
		shape         := tc.inputs[i].shape
		element_count := ml.shape_element_count(shape[:tc.inputs[i].rank])
		inputs_data[i] = make([]f32, element_count)
		cpu_grads[i]   = make([]f32, element_count)
		cuda_grads[i]  = make([]f32, element_count)
	}
	defer for i in 0 ..< n {
		delete(inputs_data[i])
		delete(cpu_grads[i])
		delete(cuda_grads[i])
	}

	tc.prepare(inputs_data[:n])

	ml.context_begin(cpu_ctx)
	output_count := _output_count(tc, inputs_data[:n])
	ml.context_end()

	w := make([]f32, output_count)
	defer delete(w)
	for i in 0 ..< output_count {
		magnitude := rand.float32_range(0.5, 1.5)
		w[i]       = rand.float32() < 0.5 ? -magnitude : magnitude
	}

	cpu_out  := make([]f32, output_count)
	cuda_out := make([]f32, output_count)
	defer delete(cpu_out)
	defer delete(cuda_out)

	ml.context_begin(cpu_ctx)
	_parity_eval(tc, inputs_data[:n], w, do_backward, cpu_out, cpu_grads[:n])
	ml.context_end()

	ml.context_begin(cuda_ctx)
	_parity_eval(tc, inputs_data[:n], w, do_backward, cuda_out, cuda_grads[:n])
	ml.context_end()

	_compare(t, tc, "output", cpu_out, cuda_out)

	if do_backward {
		for i in 0 ..< n {
			if !tc.inputs[i].check {
				continue
			}
			label := fmt.tprintf("grad[input %d]", i)
			_compare(t, tc, label, cpu_grads[i], cuda_grads[i])
		}
	}
}

@(test)
test_cpu_cuda_parity :: proc(t: ^testing.T) {
	if !_cuda_available() {
		log.info("CUDA device not available; skipping CPU-vs-CUDA parity tests")
		return
	}

	cpu_ctx  := cpu.context_create(CPU_CTX_SIZE)
	cuda_ctx := cuda.context_create()

	forward_ops  := cuda_ctx.backend.forward_ops
	backward_ops := cuda_ctx.backend.backward_ops

	for tc in cases.get() {
		if tc.kind not_in forward_ops {
			log.infof("parity: skipping %s (op not in CUDA forward_ops)", tc.name)
			continue
		}
		do_backward := tc.kind in backward_ops
		if !do_backward {
			log.infof("parity: %s forward-only (op not in CUDA backward_ops)", tc.name)
		}
		_run_parity(t, tc, cpu_ctx, cuda_ctx, do_backward)
	}

	adam_cpu_w,  adam_cpu_m,  adam_cpu_v:  [ADAM_SIZE]f32
	adam_cuda_w, adam_cuda_m, adam_cuda_v: [ADAM_SIZE]f32

	ml.context_begin(cpu_ctx)
	_run_adam(adam_cpu_w[:], adam_cpu_m[:], adam_cpu_v[:])
	ml.context_end()

	ml.context_begin(cuda_ctx)
	_run_adam(adam_cuda_w[:], adam_cuda_m[:], adam_cuda_v[:])
	ml.context_end()

	_adam_compare(t, "param",    adam_cpu_w[:],  adam_cuda_w[:])
	_adam_compare(t, "moment_m", adam_cpu_m[:],  adam_cuda_m[:])
	_adam_compare(t, "moment_v", adam_cpu_v[:],  adam_cuda_v[:])

	cpu.context_destroy(cpu_ctx)
	cuda.context_destroy(cuda_ctx)
	cuda.device_destroy()
}

ADAM_SIZE  :: 8
ADAM_STEPS :: 12

_adam_grad :: proc(step, index: int) -> f32 {
	return (f32((step * 7 + index * 3) % 11) - 5) * 0.03
}

_run_adam :: proc(w_out, m_out, v_out: []f32, loc := #caller_location) {
	size  := len(w_out)
	shape := [1]int{size}
	param := ml.make(.F32, shape[:])

	init_w := make([]f32, size)
	defer delete(init_w)
	for i in 0 ..< size {
		init_w[i] = f32(i) * 0.1 - 0.35
	}
	ml.set_data(param, init_w)

	grad := make([]f32, size)
	defer delete(grad)

	opt: ml.Optimizer
	for step in 1 ..= ADAM_STEPS {
		for i in 0 ..< size {
			grad[i] = _adam_grad(step, i)
		}
		ml.set_bytes(param, .Gradient, mem.slice_to_bytes(grad))
		if ml.optimizer_step(&opt, period=1, learning_rate=0.01, weight_decay=0.1) {
			ml.update(&opt, param)
		}
	}

	ml.get_data(param, w_out)

	state, ok := ml._optimizer_state_lookup(&opt, param)
	assert(ok, "optimizer state must exist after updates", loc=loc)

	m_bytes := make([]byte, size * 4)
	v_bytes := make([]byte, size * 4)
	defer delete(m_bytes)
	defer delete(v_bytes)
	param.backend.buffer_get(state.m, m_bytes, loc)
	param.backend.buffer_get(state.v, v_bytes, loc)
	copy(m_out, mem.slice_data_cast([]f32, m_bytes))
	copy(v_out, mem.slice_data_cast([]f32, v_bytes))

	ml.optimizer_destroy(&opt)
	ml.destroy(param)
}

_adam_compare :: proc(t: ^testing.T, label: string, cpu_vals, cuda_vals: []f32) {
	for i in 0 ..< len(cpu_vals) {
		a := f64(cpu_vals[i])
		b := f64(cuda_vals[i])
		denom := max(max(abs(a), abs(b)), REL_FLOOR)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= PARITY_TOL,
			"adam %s elem %d cpu=%.6g cuda=%.6g rel_err=%.4g (tol=%.3g)",
			label, i, a, b, rel, PARITY_TOL)
	}
}
