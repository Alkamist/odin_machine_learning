package ml_parity_tests

import "core:fmt"
import "core:math"
import "core:mem"
import "core:log"
import "core:testing"

import ml   "../.."
import cpu  "../../backends/cpu"
import cuda "../../backends/cuda"

BF16P_TOL       :: f64(2e-2)
BF16P_REL_FLOOR :: f64(1e-2)
BF16P_MAX_INPUTS :: 3

Bf16p_Spec :: struct {
	dims: [ml.MAX_TENSOR_RANK]int,
	rank: int,
}

Bf16p_Case :: struct {
	name:        string,
	kind:        ml.Operation_Kind,
	input_count: int,
	inputs:      [BF16P_MAX_INPUTS]Bf16p_Spec,
	run:         proc(t: []ml.Tensor) -> ml.Tensor,
	out_f32:     bool,
	backward:    bool,
}

_bf16p_sp :: proc(dims: ..int) -> (s: Bf16p_Spec) {
	s.rank = len(dims)
	for d, i in dims {
		s.dims[i] = d
	}
	return
}

_bf16p_fill :: proc(dst: []f32, seed: int) {
	for i in 0 ..< len(dst) {
		dst[i] = 0.4 * math.sin(f32(i) * 0.31 + f32(seed) * 1.7)
	}
}

_bf16p_make :: proc(shape: []int, src: []f32) -> ml.Tensor {
	t   := ml.zeros(.Bf16, shape)
	buf := make([]ml.Bf16, len(src))
	defer delete(buf)
	for v, i in src {
		buf[i] = ml.bf16_from_f32(v)
	}
	ml.set_bytes(t, .Data, mem.slice_to_bytes(buf))
	return t
}

_bf16p_read_grad :: proc(t: ml.Tensor, dst: []f32) {
	ml.get_bytes(t, .Gradient, mem.slice_to_bytes(dst))
}

_bf16p_read :: proc(t: ml.Tensor, dst: []f32) {
	if t.type == .F32 {
		ml.get_data(t, dst)
		return
	}
	buf := make([]ml.Bf16, t.count)
	defer delete(buf)
	ml.get_bytes(t, .Data, mem.slice_to_bytes(buf))
	for v, i in buf {
		dst[i] = ml.bf16_to_f32(v)
	}
}

_bf16p_compare :: proc(t: ^testing.T, name, label: string, cpu_vals, cuda_vals: []f32, tol: f64) {
	for i in 0 ..< len(cpu_vals) {
		a     := f64(cpu_vals[i])
		b     := f64(cuda_vals[i])
		denom := max(max(abs(a), abs(b)), BF16P_REL_FLOOR)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= tol,
			"%s: %s elem %d cpu=%.6g cuda=%.6g rel_err=%.4g (tol=%.3g)",
			name, label, i, a, b, rel, tol)
	}
}

_bf16p_forward_eval :: proc(tc: Bf16p_Case, inputs: [][]f32, out: ^[]f32) {
	ml.clear(training=false)
	tensors: [BF16P_MAX_INPUTS]ml.Tensor
	for i in 0 ..< tc.input_count {
		dims := tc.inputs[i].dims
		tensors[i] = _bf16p_make(dims[:tc.inputs[i].rank], inputs[i])
	}
	output := tc.run(tensors[:tc.input_count])
	if out^ == nil {
		out^ = make([]f32, output.count)
	}
	_bf16p_read(output, out^)
}

_bf16p_backward_eval :: proc(tc: Bf16p_Case, inputs: [][]f32, weights: []f32, grads: [][]f32) {
	ml.clear(training=true)
	tensors: [BF16P_MAX_INPUTS]ml.Tensor
	for i in 0 ..< tc.input_count {
		dims := tc.inputs[i].dims
		tensors[i] = _bf16p_make(dims[:tc.inputs[i].rank], inputs[i])
	}
	output := tc.run(tensors[:tc.input_count])
	count  := output.count

	output_f32 := output.type == .F32 ? output : ml.cast_to(output, .F32)

	out_shape := output_f32.shape
	w := ml.zeros(.F32, out_shape[:output_f32.rank])
	ml.set_data(w, weights[:count])
	weighted := ml.mul(output_f32, w)
	flat     := ml.reshape(weighted, {1, count})
	ones     := ml.scratch(.F32, {1, count})
	ml.fill_value(ones, 1.0 / f32(count))
	loss := ml.linear(flat, ones)
	ml.backward(loss)

	for i in 0 ..< tc.input_count {
		_bf16p_read_grad(tensors[i], grads[i])
	}
}

_bf16p_run :: proc(t: ^testing.T, tc: Bf16p_Case, cpu_ctx, cuda_ctx: ^ml.Context, do_backward: bool) {
	inputs: [BF16P_MAX_INPUTS][]f32
	cpu_grads:  [BF16P_MAX_INPUTS][]f32
	cuda_grads: [BF16P_MAX_INPUTS][]f32
	for i in 0 ..< tc.input_count {
		dims  := tc.inputs[i].dims
		count := ml.shape_element_count(dims[:tc.inputs[i].rank])
		inputs[i]     = make([]f32, count)
		cpu_grads[i]  = make([]f32, count)
		cuda_grads[i] = make([]f32, count)
		_bf16p_fill(inputs[i], i + 1)
	}
	defer for i in 0 ..< tc.input_count {
		delete(inputs[i])
		delete(cpu_grads[i])
		delete(cuda_grads[i])
	}

	cpu_out:  []f32
	cuda_out: []f32
	defer delete(cpu_out)
	defer delete(cuda_out)

	previous := ml.context_begin(cpu_ctx)
	_bf16p_forward_eval(tc, inputs[:tc.input_count], &cpu_out)
	ml.context_end(previous)

	previous = ml.context_begin(cuda_ctx)
	_bf16p_forward_eval(tc, inputs[:tc.input_count], &cuda_out)
	ml.context_end(previous)

	_bf16p_compare(t, tc.name, "output", cpu_out, cuda_out, BF16P_TOL)

	if do_backward {
		weights := make([]f32, len(cpu_out))
		defer delete(weights)
		for i in 0 ..< len(weights) {
			weights[i] = math.cos(f32(i) * 0.53) + 0.5
		}

		previous = ml.context_begin(cpu_ctx)
		_bf16p_backward_eval(tc, inputs[:tc.input_count], weights, cpu_grads[:tc.input_count])
		ml.context_end(previous)

		previous = ml.context_begin(cuda_ctx)
		_bf16p_backward_eval(tc, inputs[:tc.input_count], weights, cuda_grads[:tc.input_count])
		ml.context_end(previous)

		for i in 0 ..< tc.input_count {
			label := fmt.tprintf("grad[input %d]", i)
			_bf16p_compare(t, tc.name, label, cpu_grads[i], cuda_grads[i], BF16P_TOL)
		}
	}
}

_bf16p_run_add        :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.add(t[0], t[1]) }
_bf16p_run_mul        :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.mul(t[0], t[1]) }
_bf16p_run_gelu       :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.gelu(t[0]) }
_bf16p_run_silu       :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.silu(t[0]) }
_bf16p_run_tanh       :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.tanh(t[0]) }
_bf16p_run_gelu_mul   :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.gelu_mul(t[0], t[1]) }
_bf16p_run_rmsnorm    :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.rmsnorm(t[0], t[1]) }
_bf16p_run_rope       :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.rope(t[0], 2) }
_bf16p_run_select     :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.select(t[0], []int{2, 0, 3, 1}) }
_bf16p_run_slice_tr   :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.slice_trailing(t[0], 2, 6) }
_bf16p_run_slice_ld   :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.slice_leading(t[0], 1, 4) }
_bf16p_run_cast       :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.cast_to(t[0], .F32) }
_bf16p_run_attention  :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.attention(t[0], t[1], t[2], 2) }

_bf16p_case_storage: [16]Bf16p_Case
_bf16p_case_count:   int
_bf16p_case_built:   bool

_bf16p_cases :: proc() -> []Bf16p_Case {
	if _bf16p_case_built {
		return _bf16p_case_storage[:_bf16p_case_count]
	}
	list := []Bf16p_Case{
		{name="add",            kind=.Add,            input_count=2, inputs={0=_bf16p_sp(3, 8), 1=_bf16p_sp(8)},                       run=_bf16p_run_add,                backward=true},
		{name="mul",            kind=.Mul,            input_count=2, inputs={0=_bf16p_sp(3, 8), 1=_bf16p_sp(8)},                       run=_bf16p_run_mul,                backward=true},
		{name="gelu",           kind=.Gelu,           input_count=1, inputs={0=_bf16p_sp(3, 8)},                                       run=_bf16p_run_gelu,               backward=true},
		{name="silu",           kind=.Silu,           input_count=1, inputs={0=_bf16p_sp(3, 8)},                                       run=_bf16p_run_silu,               backward=true},
		{name="tanh",           kind=.Tanh,           input_count=1, inputs={0=_bf16p_sp(3, 8)},                                       run=_bf16p_run_tanh,               backward=true},
		{name="gelu_mul",       kind=.Gelu_Mul,       input_count=2, inputs={0=_bf16p_sp(3, 8), 1=_bf16p_sp(3, 8)},                    run=_bf16p_run_gelu_mul},
		{name="rmsnorm",        kind=.Rmsnorm,        input_count=2, inputs={0=_bf16p_sp(3, 8), 1=_bf16p_sp(8)},                       run=_bf16p_run_rmsnorm,            backward=true},
		{name="rope",           kind=.Rope,           input_count=1, inputs={0=_bf16p_sp(3, 8)},                                       run=_bf16p_run_rope,               backward=true},
		{name="select",         kind=.Select,         input_count=1, inputs={0=_bf16p_sp(4, 8)},                                       run=_bf16p_run_select,             backward=true},
		{name="slice_trailing", kind=.Slice_Trailing, input_count=1, inputs={0=_bf16p_sp(3, 8)},                                       run=_bf16p_run_slice_tr,           backward=true},
		{name="slice_leading",  kind=.Slice_Leading,  input_count=1, inputs={0=_bf16p_sp(5, 8)},                                       run=_bf16p_run_slice_ld,           backward=true},
		{name="cast",           kind=.Cast,           input_count=1, inputs={0=_bf16p_sp(4, 5)},                                       run=_bf16p_run_cast, out_f32=true, backward=true},
		{name="attention",      kind=.Attention,      input_count=3, inputs={0=_bf16p_sp(4, 8), 1=_bf16p_sp(4, 8), 2=_bf16p_sp(4, 8)}, run=_bf16p_run_attention,          backward=true},
	}
	for c, i in list {
		_bf16p_case_storage[i] = c
	}
	_bf16p_case_count = len(list)
	_bf16p_case_built = true
	return _bf16p_case_storage[:_bf16p_case_count]
}

@(test)
test_cpu_cuda_bf16_parity :: proc(t: ^testing.T) {
	if !_cuda_ready(t, "Bf16 parity tests") {
		return
	}

	cpu_ctx  := cpu.context_create(CPU_CTX_SIZE)
	cuda_ctx := cuda.context_create()
	defer {
		cpu.context_destroy(cpu_ctx)
		cuda.context_destroy(cuda_ctx)
		cuda.device_destroy()
	}

	cuda_forward  := cuda_ctx.backend.forward_ops
	cuda_backward := cuda_ctx.backend.backward_ops
	cpu_backward  := cpu_ctx.backend.backward_ops

	for tc in _bf16p_cases() {
		if tc.kind not_in cuda_forward {
			log.infof("bf16 parity: skipping %s (op not in CUDA forward_ops)", tc.name)
			continue
		}
		do_backward := tc.backward && tc.kind in cuda_backward && tc.kind in cpu_backward
		_bf16p_run(t, tc, cpu_ctx, cuda_ctx, do_backward)
	}

	_fusedgpu_check_attention_cache(t, cuda_ctx)
	_fusedgpu_check_rmsnorm_rope_write_cache(t, cuda_ctx)
	_fusedgpu_check_gate_up_geglu(t, cuda_ctx)
	_fusedgpu_check_quant_backward_dx(t, cuda_ctx, "linear_q4_k", false)
	_fusedgpu_check_quant_backward_dx(t, cuda_ctx, "linear_q6_k", true)
}
