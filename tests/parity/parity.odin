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

ML_REQUIRE_CUDA :: #config(ML_REQUIRE_CUDA, false)

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

_cuda_ready :: proc(t: ^testing.T, what: string) -> bool {
	if _cuda_available() {
		return true
	}
	when ML_REQUIRE_CUDA {
		testing.expectf(t, false, "ML_REQUIRE_CUDA is set but no CUDA device is available; %s cannot run", what)
	} else {
		log.warnf("============ skipped: %s (no CUDA device available) ============", what)
	}
	return false
}

_output_count :: proc(tc: cases.Op_Test, inputs_data: [][]f32) -> int {
	ml.pass_begin(training=true)

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
	ml.pass_begin(training=true)

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

		flat       := ml.reshape(weighted, {1, count})
		sum_weight := ml.scratch(.F32, {1, count})
		ml.fill_value(sum_weight, 1.0 / f32(count))
		loss := ml.linear(flat, sum_weight)
		ml.backward(loss)
		for i in 0 ..< n {
			if tc.inputs[i].check {
				ml.get_gradient(tensors[i], grads_host[i])
			}
		}
	}
}

_compare :: proc(t: ^testing.T, tc: cases.Op_Test, label: string, cpu_vals, cuda_vals: []f32) {
	tol := tc.parity_tol > 0 ? tc.parity_tol : PARITY_TOL
	for e in 0 ..< len(cpu_vals) {
		a := f64(cpu_vals[e])
		b := f64(cuda_vals[e])
		denom := max(max(abs(a), abs(b)), REL_FLOOR)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= tol,
			"%s: %s elem %d cpu=%.6g cuda=%.6g rel_err=%.4g (tol=%.3g)",
			tc.name, label, e, a, b, rel, tol)
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

	previous := ml.context_begin(cpu_ctx)
	output_count := _output_count(tc, inputs_data[:n])
	ml.context_end(previous)

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

	previous = ml.context_begin(cpu_ctx)
	_parity_eval(tc, inputs_data[:n], w, do_backward, cpu_out, cpu_grads[:n])
	ml.context_end(previous)

	previous = ml.context_begin(cuda_ctx)
	_parity_eval(tc, inputs_data[:n], w, do_backward, cuda_out, cuda_grads[:n])
	ml.context_end(previous)

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
	if !_cuda_ready(t, "CPU-vs-CUDA parity tests") {
		return
	}

	cpu_ctx  := cpu.context_create(CPU_CTX_SIZE)
	cuda_ctx := cuda.context_create()

	forward_ops  := cuda_ctx.backend.forward_ops
	backward_ops := cuda_ctx.backend.backward_ops

	for tc in cases.get() {
		if !testing.expectf(t, tc.kind in forward_ops, "%s has a cases-registry entry but %v is not in CUDA forward_ops — every registry op must stay parity-covered", tc.name, tc.kind) {
			continue
		}
		do_backward := tc.kind in backward_ops
		if !do_backward {
			log.infof("%s forward-only (op not in CUDA backward_ops)", tc.name)
		}
		_run_parity(t, tc, cpu_ctx, cuda_ctx, do_backward)
	}

	adam_cpu_w,  adam_cpu_m,  adam_cpu_v:  [ADAM_SIZE]f32
	adam_cuda_w, adam_cuda_m, adam_cuda_v: [ADAM_SIZE]f32

	previous := ml.context_begin(cpu_ctx)
	_run_adam(adam_cpu_w[:], adam_cpu_m[:], adam_cpu_v[:])
	ml.context_end(previous)

	previous = ml.context_begin(cuda_ctx)
	_run_adam(adam_cuda_w[:], adam_cuda_m[:], adam_cuda_v[:])
	ml.context_end(previous)

	_adam_compare(t, "param",    adam_cpu_w[:],  adam_cuda_w[:])
	_adam_compare(t, "moment_m", adam_cpu_m[:],  adam_cuda_m[:])
	_adam_compare(t, "moment_v", adam_cpu_v[:],  adam_cuda_v[:])

	clip_cpu_grad  := make([]f32, CLIP_TOTAL)
	clip_cuda_grad := make([]f32, CLIP_TOTAL)
	defer delete(clip_cpu_grad)
	defer delete(clip_cuda_grad)
	clip_cpu_norm, clip_cuda_norm: f32

	previous = ml.context_begin(cpu_ctx)
	clip_cpu_norm = _run_clip(clip_cpu_grad)
	ml.context_end(previous)

	previous = ml.context_begin(cuda_ctx)
	clip_cuda_norm = _run_clip(clip_cuda_grad)
	ml.context_end(previous)

	_adam_compare(t, "clip_grad", clip_cpu_grad, clip_cuda_grad)
	{
		a := f64(clip_cpu_norm)
		b := f64(clip_cuda_norm)
		denom := max(max(abs(a), abs(b)), REL_FLOOR)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= PARITY_TOL, "clip norm cpu=%.7g cuda=%.7g rel_err=%.4g (tol=%.3g)", a, b, rel, PARITY_TOL)
	}

	cpu.context_destroy(cpu_ctx)
	cuda.context_destroy(cuda_ctx)
	cuda.device_destroy()
}

CLIP_MAX_NORM :: f32(1.0)
CLIP_TOTAL    :: 288

_clip_grad :: proc(index: int) -> f32 {
	return (f32((index * 13 + 7) % 23) - 11) * 0.05
}

_run_clip :: proc(grads_out: []f32, loc := #caller_location) -> f32 {
	sizes := [3]int{64, 128, 96}
	n := len(sizes)

	tensors := make([]ml.Tensor, n)
	defer delete(tensors)
	r: ml.Registry

	offset := 0
	for size, i in sizes {
		shape := [1]int{size}
		tensors[i] = ml.alloc(.F32, shape[:], persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
		grad := make([]f32, size)
		for j in 0 ..< size {
			grad[j] = _clip_grad(offset + j)
		}
		ml.set_bytes(tensors[i], .Gradient, mem.slice_to_bytes(grad))
		delete(grad)
		ml.parameter_register(&r, "", "", tensors[i], init=ml.Init_None{}, flags=ml.PARAMETER_DEFAULT_FLAGS + {.Owned})
		offset += size
	}

	norm := ml.clip_gradient_norm(&r, CLIP_MAX_NORM)

	offset = 0
	for size, i in sizes {
		ml.get_gradient(tensors[i], grads_out[offset:offset + size])
		offset += size
	}

	ml.registry_destroy(&r)
	return norm
}

@(test)
test_cuda_lifecycle :: proc(t: ^testing.T) {
	if !_cuda_ready(t, "CUDA lifecycle test") {
		return
	}

	for cycle in 0 ..< 2 {
		ctx := cuda.context_create(fast_math=cycle == 0)
		defer cuda.device_destroy()
		defer cuda.context_destroy(ctx)
		ml.context_scope(ctx)

		param := ml.alloc(.F32, {4}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
		src := [4]f32{1, 2, 3, 4}
		ml.set_data(param, src[:])
		got: [4]f32
		ml.get_data(param, got[:])
		for i in 0 ..< 4 {
			testing.expectf(t, got[i] == src[i], "lifecycle cycle %d elem %d got=%v want=%v", cycle, i, got[i], src[i])
		}

		a := ml.tensor([]f32{1, 2, 3, 4})
		b := ml.tensor([]f32{5, 6, 7, 8})
		c := ml.mul(a, b)
		product: [4]f32
		ml.get_data(c, product[:])
		for i in 0 ..< 4 {
			want := src[i] * (src[i] + 4)
			testing.expectf(t, product[i] == want, "lifecycle cycle %d mul elem %d got=%v want=%v", cycle, i, product[i], want)
		}
		ml.pass_begin()

		ml.destroy(param)
	}
}

ADAM_SIZE  :: 8
ADAM_STEPS :: 12

_adam_grad :: cases.adam_grad

_run_adam :: proc(w_out, m_out, v_out: []f32, loc := #caller_location) {
	size  := len(w_out)
	shape := [1]int{size}
	param := ml.alloc(.F32, shape[:], persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)

	init_w := make([]f32, size)
	defer delete(init_w)
	for i in 0 ..< size {
		init_w[i] = f32(i) * 0.1 - 0.35
	}
	ml.set_data(param, init_w)

	grad := make([]f32, size)
	defer delete(grad)

	opt := ml.optimizer_make(learning_rate=0.01, weight_decay=0.1)
	for step in 1 ..= ADAM_STEPS {
		for i in 0 ..< size {
			grad[i] = _adam_grad(step, i)
		}
		ml.set_bytes(param, .Gradient, mem.slice_to_bytes(grad))
		if ml.optimizer_step(&opt) {
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
		testing.expectf(t,
			rel <= PARITY_TOL,
			"adam %s elem %d cpu=%.6g cuda=%.6g rel_err=%.4g (tol=%.3g)",
			label, i, a, b, rel, PARITY_TOL,
		)
	}
}
