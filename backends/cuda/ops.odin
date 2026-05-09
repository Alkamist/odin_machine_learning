package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"

import "bindings/cuda"
import "bindings/cublas"

import ml "../../"

// Per-op cached pipelines. Built lazily on first use, lifetime is the device.
_add_pipeline:             ^Pipeline
_add_bf16_pipeline:        ^Pipeline
_add_back_a_pipeline:      ^Pipeline
_add_back_b_pipeline:      ^Pipeline
_add_back_a_bf16_pipeline: ^Pipeline
_add_back_b_bf16_pipeline: ^Pipeline
_quantize_q8_1_pipeline:     ^Pipeline
_linear_q4_k_mmvq_pipeline: ^Pipeline
_linear_q4_k_gate_up_geglu_bf16_pipeline: ^Pipeline
_linear_q6_k_mmvq_pipeline: ^Pipeline

_mul_pipeline:           ^Pipeline
_mul_bf16_pipeline:      ^Pipeline
_gelu_mul_bf16_pipeline: ^Pipeline
_gelu_mul_f32_pipeline:  ^Pipeline
_tanh_pipeline:          ^Pipeline
_tanh_bf16_pipeline:     ^Pipeline
_cast_bf16_to_f32_pipeline: ^Pipeline
_cast_f32_to_bf16_pipeline: ^Pipeline
_rmsnorm_pipeline:       ^Pipeline
_rmsnorm_bf16_pipeline:  ^Pipeline
_add_rmsnorm_bf16_pipeline: ^Pipeline
_add_rmsnorm_f32_pipeline:  ^Pipeline
_rmsnorm_rope_bf16_pipeline:       ^Pipeline
_rmsnorm_rope_f32_pipeline:        ^Pipeline
_rmsnorm_rope_cache_bf16_pipeline: ^Pipeline
_rope_pipeline:          ^Pipeline
_rope_bf16_pipeline:     ^Pipeline
_attention_bf16_pipeline:                ^Pipeline
_attention_cache_bf16_pipeline:          ^Pipeline
_attention_cache_vec_bf16_d256_pipeline: ^Pipeline
_attention_cache_vec_bf16_d512_pipeline: ^Pipeline
_cache_write_bf16_pipeline:              ^Pipeline
_select_f32_pipeline:           ^Pipeline
_select_bf16_pipeline:          ^Pipeline
_slice_trailing_f32_pipeline:   ^Pipeline
_slice_trailing_bf16_pipeline:  ^Pipeline

@(require_results)
data :: #force_inline proc(t: ml.Tensor) -> Gpu_Buffer {
	return transmute(Gpu_Buffer)t.buffers[.Data]
}
@(require_results)
gradient :: #force_inline proc(t: ml.Tensor) -> Gpu_Buffer {
	return transmute(Gpu_Buffer)t.buffers[.Gradient]
}

@(require_results)
_div_up :: #force_inline proc(a, b: int) -> u32 {
	return u32((a + b - 1) / b)
}

// Lazy-compile + dispatch the cache_write kernel. Source is bf16; cache is
// also bf16 packed-pairs.
_dispatch_cache_write :: proc(src_type: ml.Data_Type, grid: u32, args: []rawptr, loc: runtime.Source_Code_Location) {
	fmt.assertf(src_type == .Bf16, "cache_write: unsupported src dtype %v", src_type, loc=loc)
	if _cache_write_bf16_pipeline == nil {
		_cache_write_bf16_pipeline = _compile_pipeline(CACHE_WRITE_BF16_SRC, "cache_write_bf16.cu", "cache_write_bf16")
	}
	_dispatch(_cache_write_bf16_pipeline, grid, 1, 1, 256, 1, 1, 0, args, loc)
}

// Quantize a single contiguous bf16 OR fp32 row to Q8_1 blocks. The result
// is the activation-pool allocation containing q8_byte_count = blocks * 36
// bytes. Picks the bf16-input or fp32-input kernel based on `input_type`.
// Caller is responsible for the `q8_1_cache` lookup/insert.
_emit_quantize_q8_1 :: proc(gctx: ^Context, xp: cuda.DevicePtr, input_size: int, input_type: ml.Data_Type, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	xp := xp
	fmt.assertf(input_type == .Bf16, "quantize_q8_1: unsupported input dtype %v", input_type, loc=loc)
	q8_block_count := input_size / Q8_1_BLOCK_ELEMS
	q8_byte_count  := q8_block_count * Q8_1_BLOCK_BYTES
	q8 := _activation_alloc(gctx, u64(q8_byte_count), loc)
	K  := i32(input_size)
	args := [?]rawptr{ &xp, &q8, &K }

	if _quantize_q8_1_pipeline == nil {
		_quantize_q8_1_pipeline = _compile_pipeline(QUANTIZE_Q8_1_BF16_SRC, "quantize_q8_1_bf16.cu", "quantize_q8_1_bf16")
	}
	_dispatch(_quantize_q8_1_pipeline,
		_div_up(input_size, 256), 1, 1,
		256, 1, 1,
		0, args[:], loc)
	return q8
}

// Lazily upload `cache_position` for the current forward to the per-context
// device buffer. Position-bearing kernels (rmsnorm_rope, rope, attention_cache)
// take a `const int* pos_ptr` and read from this stable buffer instead of
// receiving the value as a baked-in scalar arg. Net effect: the captured graph
// references identical pointers across decode steps for these kernels, so
// `cuGraphExecUpdate` finds nothing to patch on the kernel-arg side. Only the
// K/V cache memcpy nodes still vary (their dst depends on cache_position %
// t_capacity); fusing rope+set_rows would close that gap too.
_emit_position_upload :: proc(gctx: ^Context, value: int, loc: runtime.Source_Code_Location) {
	if gctx.position_written_this_forward { return }
	(^i32)(gctx.position_pinned)^ = i32(value)
	cuda.check(cuda.MemcpyHtoDAsync(gctx.position_dev, gctx.position_pinned, 4, gctx.stream), loc=loc)
	gctx.position_written_this_forward = true
}

// Lazily (re)allocate the shift scratch used by sliding K/V cache writes.
// Sized to the largest preserved-rows × kv_size byte count seen so far. The
// pointer is stable across forwards once allocated, so the captured graph's
// memcpy nodes reference a fixed device address; cuGraphExecUpdate only has
// to patch the size if it changes.
_ensure_shift_scratch :: proc(gctx: ^Context, byte_count: u64, loc: runtime.Source_Code_Location) {
	if gctx.shift_scratch_size >= byte_count { return }
	if gctx.shift_scratch_dev != 0 {
		cuda.check(cuda.MemFree(gctx.shift_scratch_dev), loc=loc)
	}
	cuda.check(cuda.MemAlloc(&gctx.shift_scratch_dev, uint(byte_count)), loc=loc)
	gctx.shift_scratch_size = byte_count
}

// Replace cuda.odin's stub. Routes by op variant.
_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	#partial switch _ in op.variant {
	case ml.Add:          _add_forward(op, loc)
	case ml.Mul:          _mul_forward(op, loc)
	case ml.Gelu_Mul:     _gelu_mul_forward(op, loc)
	case ml.Tanh:         _tanh_forward(op, loc)
	case ml.Cast:         _cast_forward(op, loc)
	case ml.Linear:       _linear_forward(op, loc)
	case ml.Linear_Q4_K:               _linear_q4_k_forward(op, loc)
	case ml.Linear_Q4_K_Gate_Up_Geglu: _linear_q4_k_gate_up_geglu_forward(op, loc)
	case ml.Linear_Q6_K:               _linear_q6_k_forward(op, loc)
	case ml.Rmsnorm:      _rmsnorm_forward(op, loc)
	case ml.Add_Rmsnorm:  _add_rmsnorm_forward(op, loc)
	case ml.Rmsnorm_Rope:    _rmsnorm_rope_forward(op, loc)
	case ml.Rmsnorm_Rope_Write_Cache: _rmsnorm_rope_write_cache_forward(op, loc)
	case ml.Rope:            _rope_forward(op, loc)
	case ml.Attention:       _attention_forward(op, loc)
	case ml.Attention_Cache: _attention_cache_forward(op, loc)
	case ml.Select:          _select_forward(op, loc)
	case ml.Slice_Trailing:  _slice_trailing_forward(op, loc)
	case: fmt.panicf("cuda backend: forward not implemented for op variant %T", op.variant, loc=loc)
	}
}

_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	#partial switch _ in op.variant {
	case ml.Add:    _add_backward(op, loc)
	case ml.Linear: _linear_backward(op, loc)
	case:           fmt.panicf("cuda backend: backward not implemented for op variant %T", op.variant, loc=loc)
	}
}

_update :: proc(opt: ml.Optimizer, t: ml.Tensor, loc: runtime.Source_Code_Location) {
	fmt.panicf("cuda backend: update not implemented", loc=loc)
}

// ----- Linear ----------------------------------------------------------------
//
// Forward:   output[m, n] = sum_k input[m, k] * weight[n, k]   (= input @ weight^T)
// Backward:  dx[m, k]    += sum_n grad_out[m, n] * weight[n, k] (= dy @ weight)
//            dw[n, k]    += sum_m grad_out[m, n] * input[m, k]  (= dy^T @ input)
//
// All tensors are row-major. cuBLAS is column-major, so each row-major matrix
// `R[X, Y]` is treated as a column-major matrix `R^T[Y, X]` with leading dim Y.
// The substitution math is folded into the GemmEx args below.
//
// On Ampere with bf16 inputs and CUBLAS_COMPUTE_32F, GemmEx selects a tensor
// core algorithm automatically (no per-call algo override needed).

_linear_dtype :: proc(t: ml.Data_Type, loc: runtime.Source_Code_Location) -> cublas.DataType {
	#partial switch t {
	case .Bf16: return .R_16BF
	case .F32:  return .R_32F
	case:       fmt.panicf("linear: unsupported dtype %v", t, loc=loc)
	}
	return .R_32F
}

_linear_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear).weight
	output      := op.output
	output_size := i32(weight.shape[0])
	input_size  := i32(weight.shape[1])
	count       := i32(ml.len(input) / int(input_size))

	gctx := _gctx(loc)
	w_dt := _linear_dtype(weight.type, loc)
	x_dt := _linear_dtype(input.type,  loc)
	y_dt := _linear_dtype(output.type, loc)

	alpha: f32 = 1.0
	beta:  f32 = 0.0

	w_ptr := data(weight).ptr
	x_ptr := data(input).ptr
	y_ptr := data(output).ptr

	cublas.check(cublas.GemmEx(
		gctx.cublas_handle,
		.T,  .N,
		output_size, count, input_size,
		&alpha,
		rawptr(uintptr(w_ptr)), w_dt, input_size,
		rawptr(uintptr(x_ptr)), x_dt, input_size,
		&beta,
		rawptr(uintptr(y_ptr)), y_dt, output_size,
		._32F,
		.DEFAULT,
	), loc=loc)
}

_linear_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear).weight
	output      := op.output
	output_size := i32(weight.shape[0])
	input_size  := i32(weight.shape[1])
	count       := i32(ml.len(input) / int(input_size))

	gctx := _gctx(loc)
	dt   := _linear_dtype(input.type, loc)

	alpha: f32 = 1.0
	beta:  f32 = 1.0  // accumulate into existing gradients (`+=`)

	w_ptr  := data    (weight).ptr
	x_ptr  := data    (input ).ptr
	dy_ptr := gradient(output).ptr
	dx_ptr := gradient(input ).ptr
	dw_ptr := gradient(weight).ptr

	// dx = dy @ weight, accumulated into grad_input.
	cublas.check(cublas.GemmEx(
		gctx.cublas_handle,
		.N, .N,
		input_size, count, output_size,
		&alpha,
		rawptr(uintptr(w_ptr )), dt, input_size,
		rawptr(uintptr(dy_ptr)), dt, output_size,
		&beta,
		rawptr(uintptr(dx_ptr)), dt, input_size,
		._32F,
		.DEFAULT,
	), loc=loc)

	// dw = input^T @ dy (computed in column-major as input @ dy^T), accumulated.
	cublas.check(cublas.GemmEx(
		gctx.cublas_handle,
		.N, .T,
		input_size, output_size, count,
		&alpha,
		rawptr(uintptr(x_ptr )), dt, input_size,
		rawptr(uintptr(dy_ptr)), dt, output_size,
		&beta,
		rawptr(uintptr(dw_ptr)), dt, input_size,
		._32F,
		.DEFAULT,
	), loc=loc)
}

// ----- Linear_Q4_K (decode / M=1) ---------------------------------------------
//
// Pre-quantize the BF16 input row to Q8_1, then mmvq against the Q4_K weight.
// Mirrors the vulkan two-stage dispatch so the on-device byte layout is bit
// identical (same Q4_K weight buffers round-trip on either backend).

_linear_q4_k_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q4_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "linear_q4_k: K must be a multiple of 256, got %v", input_size, loc=loc)
	fmt.assertf(count == 1, "cuda linear_q4_k currently supports M=1 (decode); got M=%v", count, loc=loc)
	fmt.assertf(output.type == .Bf16, "cuda linear_q4_k requires Bf16 output (got %v)", output.type, loc=loc)

	if _linear_q4_k_mmvq_pipeline == nil {
		_linear_q4_k_mmvq_pipeline = _compile_pipeline(LINEAR_Q4_K_MMVQ_SRC, "linear_q4_k_mmvq.cu", "linear_q4_k_mmvq")
	}

	gctx := _gctx(loc)

	xp := data(input).ptr

	q8: cuda.DevicePtr
	if cached, ok := gctx.q8_1_cache[xp]; ok {
		q8 = cached
	} else {
		q8 = _emit_quantize_q8_1(gctx, xp, input_size, input.type, loc)
		gctx.q8_1_cache[xp] = q8
	}

	wp := data(weight).ptr
	yp := data(output).ptr  // fp32 output, written directly by mmvq.
	M  := i32(count)
	K  := i32(input_size)
	N  := i32(output_size)

	mmvq_args := [?]rawptr{ &q8, &wp, &yp, &M, &K, &N }
	_dispatch(_linear_q4_k_mmvq_pipeline,
		u32(output_size), 1, 1,    // ROWS_PER_BLOCK=1: one block per output row
		32, 4, 1,                    // 4 warps × 32 lanes
		0, mmvq_args[:], loc)
}

// Fused FFN front-half (gate + up + GEGLU) over Q4_K weights, decode (M=1).
// Mirrors `_linear_q4_k_forward` but consumes two weight tensors and emits
// `gelu(gate) * up` directly, eliminating one Q4_K dispatch and the
// downstream `gelu_mul_bf16` per layer.
_linear_q4_k_gate_up_geglu_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	v           := op.variant.(ml.Linear_Q4_K_Gate_Up_Geglu)
	w_gate      := v.w_gate
	w_up        := v.w_up
	output      := op.output
	output_size := w_gate.shape[0]
	input_size  := w_gate.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "linear_q4_k_gate_up_geglu: K must be a multiple of 256, got %v", input_size, loc=loc)
	fmt.assertf(count == 1, "cuda linear_q4_k_gate_up_geglu requires M=1 (decode); got M=%v", count, loc=loc)
	fmt.assertf(output.type == .Bf16, "cuda linear_q4_k_gate_up_geglu requires Bf16 output (got %v)", output.type, loc=loc)

	if _linear_q4_k_gate_up_geglu_bf16_pipeline == nil {
		_linear_q4_k_gate_up_geglu_bf16_pipeline = _compile_pipeline(LINEAR_Q4_K_GATE_UP_GEGLU_BF16_SRC, "linear_q4_k_gate_up_geglu_bf16.cu", "linear_q4_k_gate_up_geglu_bf16")
	}

	gctx := _gctx(loc)

	xp := data(input).ptr

	q8: cuda.DevicePtr
	if cached, ok := gctx.q8_1_cache[xp]; ok {
		q8 = cached
	} else {
		q8 = _emit_quantize_q8_1(gctx, xp, input_size, input.type, loc)
		gctx.q8_1_cache[xp] = q8
	}

	wgp := data(w_gate).ptr
	wup := data(w_up).ptr
	yp  := data(output).ptr  // fp32 output, written directly by the fused kernel.
	M   := i32(count)
	K   := i32(input_size)
	N   := i32(output_size)

	mmvq_args := [?]rawptr{ &q8, &wgp, &wup, &yp, &M, &K, &N }
	_dispatch(_linear_q4_k_gate_up_geglu_bf16_pipeline,
		u32(output_size), 1, 1,
		32, 4, 1,
		0, mmvq_args[:], loc)
}

_add_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b

	#partial switch a.type {
	case .F32:
		if _add_pipeline == nil {
			_add_pipeline = _compile_pipeline(ADD_F32_SRC,           "add.cu",           "add_f32")
		}
		ap := data(a).ptr; bp := data(b).ptr; cp := data(output).ptr
		n   := i32(ml.len(a))
		n_b := i32(ml.len(b))
		args := [?]rawptr{ &ap, &bp, &cp, &n, &n_b }
		grid := _div_up(ml.len(a), ADD_LOCAL_SIZE)
		_dispatch(_add_pipeline, grid, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args[:], loc)

	case .Bf16:
		if _add_bf16_pipeline == nil {
			_add_bf16_pipeline = _compile_pipeline(ADD_BF16_SRC,          "add_bf16.cu",      "add_bf16")
		}
		ap := data(a).ptr; bp := data(b).ptr; cp := data(output).ptr
		pair_count := (ml.len(a) + 1) / 2
		n          := i32(ml.len(a))
		n_b        := i32(ml.len(b))
		pc         := i32(pair_count)
		args := [?]rawptr{ &ap, &bp, &cp, &n, &n_b, &pc }
		grid := _div_up(pair_count, ADD_LOCAL_SIZE)
		_dispatch(_add_bf16_pipeline, grid, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args[:], loc)

	case:
		fmt.panicf("add: unsupported dtype %v", a.type, loc=loc)
	}
}

_add_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b
	stride := ml.len(a) / ml.len(b)

	#partial switch a.type {
	case .F32:
		// da_a += dy
		if _add_back_a_pipeline == nil {
			_add_back_a_pipeline = _compile_pipeline(ADD_BACK_A_SRC,        "add_back_a.cu",    "add_back_a_f32")
		}
		dyp := gradient(output).ptr; dap := gradient(a).ptr
		n := i32(ml.len(a))
		args_a := [?]rawptr{ &dyp, &dap, &n }
		grid_a := _div_up(ml.len(a), ADD_LOCAL_SIZE)
		_dispatch(_add_back_a_pipeline, grid_a, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args_a[:], loc)

		// da_b += sum_stride dy
		if _add_back_b_pipeline == nil {
			_add_back_b_pipeline = _compile_pipeline(ADD_BACK_B_SRC,        "add_back_b.cu",    "add_back_b_f32")
		}
		dbp := gradient(b).ptr
		n_b := i32(ml.len(b)); st := i32(stride)
		args_b := [?]rawptr{ &dyp, &dbp, &n_b, &st }
		grid_b := _div_up(ml.len(b), ADD_LOCAL_SIZE)
		_dispatch(_add_back_b_pipeline, grid_b, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args_b[:], loc)

	case .Bf16:
		if _add_back_a_bf16_pipeline == nil {
			_add_back_a_bf16_pipeline = _compile_pipeline(ADD_BACK_A_BF16_SRC,   "add_back_a_bf16.cu", "add_back_a_bf16")
		}
		dyp := gradient(output).ptr; dap := gradient(a).ptr
		a_pairs := (ml.len(a) + 1) / 2
		n := i32(ml.len(a)); pc := i32(a_pairs)
		args_a := [?]rawptr{ &dyp, &dap, &n, &pc }
		grid_a := _div_up(a_pairs, ADD_LOCAL_SIZE)
		_dispatch(_add_back_a_bf16_pipeline, grid_a, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args_a[:], loc)

		if _add_back_b_bf16_pipeline == nil {
			_add_back_b_bf16_pipeline = _compile_pipeline(ADD_BACK_B_BF16_SRC,   "add_back_b_bf16.cu", "add_back_b_bf16")
		}
		dbp := gradient(b).ptr
		b_pairs := (ml.len(b) + 1) / 2
		n_b := i32(ml.len(b)); st := i32(stride); pcb := i32(b_pairs)
		args_b := [?]rawptr{ &dyp, &dbp, &n_b, &st, &pcb }
		grid_b := _div_up(b_pairs, ADD_LOCAL_SIZE)
		_dispatch(_add_back_b_bf16_pipeline, grid_b, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args_b[:], loc)

	case:
		fmt.panicf("add backward: unsupported dtype %v", a.type, loc=loc)
	}
}

// ----- Mul -------------------------------------------------------------------
_mul_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a := op.input
	b := op.variant.(ml.Mul).b
	c := op.output

	#partial switch a.type {
	case .F32:
		if _mul_pipeline == nil {
			_mul_pipeline = _compile_pipeline(MUL_F32_SRC,           "mul.cu",           "mul_f32")
		}
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		n   := i32(ml.len(a)); n_b := i32(ml.len(b))
		args := [?]rawptr{ &ap, &bp, &cp, &n, &n_b }
		_dispatch(_mul_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		if _mul_bf16_pipeline == nil {
			_mul_bf16_pipeline = _compile_pipeline(MUL_BF16_SRC,          "mul_bf16.cu",      "mul_bf16")
		}
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		pair_count := (ml.len(a) + 1) / 2
		n   := i32(ml.len(a)); n_b := i32(ml.len(b)); pc := i32(pair_count)
		args := [?]rawptr{ &ap, &bp, &cp, &n, &n_b, &pc }
		_dispatch(_mul_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("mul: unsupported dtype %v", a.type, loc=loc)
	}
}

// ----- Gelu_Mul (fused gelu(a) * b) ------------------------------------------
_gelu_mul_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a := op.input
	b := op.variant.(ml.Gelu_Mul).b
	c := op.output

	#partial switch a.type {
	case .Bf16:
		if _gelu_mul_bf16_pipeline == nil {
			_gelu_mul_bf16_pipeline = _compile_pipeline(GELU_MUL_BF16_SRC, "gelu_mul_bf16.cu", "gelu_mul_bf16")
		}
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		pair_count := (ml.len(a) + 1) / 2
		n   := i32(ml.len(a)); n_b := i32(ml.len(b)); pc := i32(pair_count)
		args := [?]rawptr{ &ap, &bp, &cp, &n, &n_b, &pc }
		_dispatch(_gelu_mul_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		if _gelu_mul_f32_pipeline == nil {
			_gelu_mul_f32_pipeline = _compile_pipeline(GELU_MUL_F32_SRC, "gelu_mul_f32.cu", "gelu_mul_f32")
		}
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		n   := i32(ml.len(a)); n_b := i32(ml.len(b))
		args := [?]rawptr{ &ap, &bp, &cp, &n, &n_b }
		_dispatch(_gelu_mul_f32_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("gelu_mul: unsupported dtype %v", a.type, loc=loc)
	}
}

// ----- Tanh ------------------------------------------------------------------
_tanh_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	#partial switch x.type {
	case .F32:
		if _tanh_pipeline == nil {
			_tanh_pipeline = _compile_pipeline(TANH_F32_SRC,          "tanh.cu",          "tanh_f32")
		}
		xp := data(x).ptr; yp := data(y).ptr
		n := i32(ml.len(x))
		args := [?]rawptr{ &xp, &yp, &n }
		_dispatch(_tanh_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		if _tanh_bf16_pipeline == nil {
			_tanh_bf16_pipeline = _compile_pipeline(TANH_BF16_SRC,         "tanh_bf16.cu",     "tanh_bf16")
		}
		xp := data(x).ptr; yp := data(y).ptr
		pair_count := (ml.len(x) + 1) / 2
		n := i32(ml.len(x)); pc := i32(pair_count)
		args := [?]rawptr{ &xp, &yp, &n, &pc }
		_dispatch(_tanh_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("tanh: unsupported dtype %v", x.type, loc=loc)
	}
}

// ----- Cast ------------------------------------------------------------------
_cast_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	if x.type == y.type {
		// Same-type cast = byte copy.
		bytes := uint(ml.len(x) * ml.data_type_size(x.type))
		gctx := _gctx(loc)
		cuda.check(cuda.MemcpyDtoDAsync(data(y).ptr, data(x).ptr, bytes, gctx.stream), loc=loc)
		return
	}

	pair_count := (ml.len(x) + 1) / 2
	n := i32(ml.len(x)); pc := i32(pair_count)
	xp := data(x).ptr; yp := data(y).ptr

	switch {
	case x.type == .Bf16 && y.type == .F32:
		if _cast_bf16_to_f32_pipeline == nil {
			_cast_bf16_to_f32_pipeline = _compile_pipeline(CAST_BF16_TO_F32_SRC,  "cast_bf16_to_f32.cu", "cast_bf16_to_f32")
		}
		args := [?]rawptr{ &xp, &yp, &n, &pc }
		_dispatch(_cast_bf16_to_f32_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case x.type == .F32 && y.type == .Bf16:
		if _cast_f32_to_bf16_pipeline == nil {
			_cast_f32_to_bf16_pipeline = _compile_pipeline(CAST_F32_TO_BF16_SRC,  "cast_f32_to_bf16.cu", "cast_f32_to_bf16")
		}
		args := [?]rawptr{ &xp, &yp, &n, &pc }
		_dispatch(_cast_f32_to_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("cast: unsupported (%v -> %v)", x.type, y.type, loc=loc)
	}
}

// ----- Rmsnorm ---------------------------------------------------------------
_rmsnorm_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x      := op.input
	y      := op.output
	v      := op.variant.(ml.Rmsnorm)
	size   := x.shape[x.rank - 1]
	count  := ml.len(x) / size

	xp := data(x).ptr; wp := data(v.weight).ptr; yp := data(y).ptr
	c   := i32(count); s := i32(size); eps := v.eps
	args := [?]rawptr{ &xp, &wp, &yp, &c, &s, &eps }

	#partial switch x.type {
	case .Bf16:
		fmt.assertf(size % 2 == 0, "rmsnorm bf16 requires even size (got %v)", size, loc=loc)
		if _rmsnorm_bf16_pipeline == nil {
			_rmsnorm_bf16_pipeline = _compile_pipeline(RMSNORM_BF16_SRC,      "rmsnorm_bf16.cu",  "rmsnorm_bf16")
		}
		_dispatch(_rmsnorm_bf16_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		fmt.assertf(size % 2 == 0, "rmsnorm f32 requires even size (got %v)", size, loc=loc)
		if _rmsnorm_pipeline == nil {
			_rmsnorm_pipeline = _compile_pipeline(RMSNORM_F32_SRC,       "rmsnorm.cu",       "rmsnorm_f32")
		}
		_dispatch(_rmsnorm_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("rmsnorm: unsupported dtype %v", x.type, loc=loc)
	}
}

// ----- Add_Rmsnorm -----------------------------------------------------------
_add_rmsnorm_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a   := op.input
	y   := op.output
	v   := op.variant.(ml.Add_Rmsnorm)

	size  := a.shape[a.rank - 1]
	count := ml.len(a) / size
	fmt.assertf(size % 2 == 0, "add_rmsnorm requires even trailing dim (got %v)", size, loc=loc)

	ap := data(a).ptr
	bp := data(v.b).ptr
	wp := data(v.weight).ptr
	rp := data(v.residual_out).ptr
	yp := data(y).ptr
	c   := i32(count); s := i32(size); eps := v.eps
	args := [?]rawptr{ &ap, &bp, &wp, &rp, &yp, &c, &s, &eps }

	#partial switch a.type {
	case .Bf16:
		if _add_rmsnorm_bf16_pipeline == nil {
			_add_rmsnorm_bf16_pipeline = _compile_pipeline(ADD_RMSNORM_BF16_SRC, "add_rmsnorm_bf16.cu", "add_rmsnorm_bf16")
		}
		_dispatch(_add_rmsnorm_bf16_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		if _add_rmsnorm_f32_pipeline == nil {
			_add_rmsnorm_f32_pipeline = _compile_pipeline(ADD_RMSNORM_F32_SRC, "add_rmsnorm_f32.cu", "add_rmsnorm_f32")
		}
		_dispatch(_add_rmsnorm_f32_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("add_rmsnorm: unsupported dtype %v", a.type, loc=loc)
	}
}

// ----- Rmsnorm_Rope ----------------------------------------------------------
_rmsnorm_rope_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Rmsnorm_Rope)

	token_count := x.shape[0]
	head_size   := x.shape[1] / v.head_count
	fmt.assertf(head_size % 2 == 0, "rmsnorm_rope requires even head_size (got %v)", head_size, loc=loc)

	gctx := _gctx(loc)
	_emit_position_upload(gctx, v.position_offset, loc)

	xp := data(x).ptr; wp := data(v.weight).ptr; yp := data(y).ptr
	tc := i32(token_count); hc := i32(v.head_count); hs := i32(head_size)
	eps := v.eps; base := v.base
	pos_dev := gctx.position_dev; rpc := i32(v.rotate_pair_count)
	args := [?]rawptr{ &xp, &wp, &yp, &tc, &hc, &hs, &eps, &base, &pos_dev, &rpc }

	#partial switch x.type {
	case .Bf16:
		if _rmsnorm_rope_bf16_pipeline == nil {
			_rmsnorm_rope_bf16_pipeline = _compile_pipeline(RMSNORM_ROPE_BF16_SRC, "rmsnorm_rope_bf16.cu", "rmsnorm_rope_bf16")
		}
		_dispatch(_rmsnorm_rope_bf16_pipeline, u32(token_count * v.head_count), 1, 1, 128, 1, 1, 0, args[:], loc)
	case .F32:
		if _rmsnorm_rope_f32_pipeline == nil {
			_rmsnorm_rope_f32_pipeline = _compile_pipeline(RMSNORM_ROPE_F32_SRC, "rmsnorm_rope_f32.cu", "rmsnorm_rope_f32")
		}
		_dispatch(_rmsnorm_rope_f32_pipeline, u32(token_count * v.head_count), 1, 1, 128, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("rmsnorm_rope: unsupported dtype %v", x.type, loc=loc)
	}
}

// ----- Rmsnorm_Rope_Write_Cache (bf16) --------------------------------------
//
// Fused rmsnorm + rope + K-cache write. Writes the rotated K row directly to
// `cache` at the slot for `cache_position` (matching `cache_write_bf16`'s
// formula); marks the cache pointer in `k_cache_written_this_forward` so the
// downstream `_attention_cache_forward` skips its redundant K cache_write.
_rmsnorm_rope_write_cache_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	v := op.variant.(ml.Rmsnorm_Rope_Write_Cache)

	token_count := x.shape[0]
	head_size   := x.shape[1] / v.head_count
	fmt.assertf(head_size % 2 == 0, "rmsnorm_rope_write_cache requires even head_size (got %v)", head_size, loc=loc)
	fmt.assertf(token_count <= v.cache_capacity, "rmsnorm_rope_write_cache token_count (%v) cannot exceed cache_capacity (%v)", token_count, v.cache_capacity, loc=loc)
	fmt.assertf(v.cache.type == .Bf16, "rmsnorm_rope_write_cache requires Bf16 cache (got %v)", v.cache.type, loc=loc)

	gctx := _gctx(loc)
	cp := data(v.cache).ptr

	// K-cache shift mirrors `_attention_cache_forward`'s logic but runs here
	// (must happen before the fused kernel writes new rows at slots
	// [cap-token_count, cap)). For non-sliding layers the cache is sized to
	// the full sequence so the shift never triggers.
	kv_size      := v.cache.shape[1]
	row_bytes    := uint(kv_size) * 2
	// The cache_write kernel writes new rows at slots [cap-tc, cap) once
	// pos >= cap-tc; the host's job here is to shift the still-valid prefix
	// back by exactly the number of rows being dropped, which is
	// min(before_count + tc - cap, tc) = clamp(excess, 0, tc). The naive
	// `shift_amount = excess` over-shifts (and underflows preserved_rows)
	// once pos exceeds cap, e.g. cache.length = 2*sliding_window.
	excess       := v.position_offset + token_count - v.cache_capacity
	shift_amount := min(max(excess, 0), token_count)
	if shift_amount > 0 && !(cp in gctx.k_cache_written_this_forward) {
		preserved_rows  := v.cache_capacity - shift_amount
		preserved_bytes := uint(preserved_rows) * row_bytes
		_ensure_shift_scratch(gctx, u64(preserved_bytes), loc)
		shift_src_offset := uint(shift_amount) * row_bytes
		cuda.check(cuda.MemcpyDtoDAsync(gctx.shift_scratch_dev, cp + cuda.DevicePtr(shift_src_offset), preserved_bytes, gctx.stream), loc=loc)
		cuda.check(cuda.MemcpyDtoDAsync(cp, gctx.shift_scratch_dev, preserved_bytes, gctx.stream), loc=loc)
	}

	_emit_position_upload(gctx, v.position_offset, loc)

	xp     := data(x).ptr
	wp     := data(v.weight).ptr
	tc     := i32(token_count); hc := i32(v.head_count); hs := i32(head_size)
	eps    := v.eps; base := v.base
	pos_dev := gctx.position_dev; rpc := i32(v.rotate_pair_count)
	cap    := i32(v.cache_capacity)
	args := [?]rawptr{ &xp, &wp, &cp, &tc, &hc, &hs, &eps, &base, &pos_dev, &rpc, &cap }

	fmt.assertf(x.type == .Bf16, "rmsnorm_rope_write_cache: unsupported input dtype %v", x.type, loc=loc)
	if _rmsnorm_rope_cache_bf16_pipeline == nil {
		_rmsnorm_rope_cache_bf16_pipeline = _compile_pipeline(RMSNORM_ROPE_CACHE_BF16_SRC, "rmsnorm_rope_cache_bf16.cu", "rmsnorm_rope_cache_bf16")
	}
	_dispatch(_rmsnorm_rope_cache_bf16_pipeline, u32(token_count * v.head_count), 1, 1, 128, 1, 1, 0, args[:], loc)

	// Tell `_attention_cache_forward` the K cache row is already populated.
	gctx.k_cache_written_this_forward[cp] = true
}

// ----- Rope ------------------------------------------------------------------
_rope_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Rope)
	token_count := x.shape[0]
	head_size   := x.shape[x.rank - 1] / v.head_count
	total_pairs := token_count * v.head_count * (head_size / 2)

	gctx := _gctx(loc)
	_emit_position_upload(gctx, v.position_offset, loc)

	xp := data(x).ptr; yp := data(y).ptr
	tc := i32(token_count); hc := i32(v.head_count); hs := i32(head_size)
	base := v.base; pos_dev := gctx.position_dev; rpc := i32(v.rotate_pair_count)
	args := [?]rawptr{ &xp, &yp, &tc, &hc, &hs, &base, &pos_dev, &rpc }

	#partial switch x.type {
	case .F32:
		if _rope_pipeline == nil {
			_rope_pipeline = _compile_pipeline(ROPE_F32_SRC,          "rope.cu",          "rope_f32")
		}
		_dispatch(_rope_pipeline, _div_up(total_pairs, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		fmt.assertf(head_size % 2 == 0, "rope bf16 requires even head_size (got %v)", head_size, loc=loc)
		if _rope_bf16_pipeline == nil {
			_rope_bf16_pipeline = _compile_pipeline(ROPE_BF16_SRC,         "rope_bf16.cu",     "rope_bf16")
		}
		_dispatch(_rope_bf16_pipeline, _div_up(total_pairs, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("rope: unsupported dtype %v", x.type, loc=loc)
	}
}

// ----- Linear_Q6_K (decode / M=1) --------------------------------------------
//
// Q6_K x q8_1 mmvq. Mirrors Q4_K's two-stage dispatch (quantize_q8_1 then
// mmvq) so the q8_1 input is shared with co-located Q4_K matmuls via the
// per-forward `q8_1_cache`.

_linear_q6_k_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q6_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "linear_q6_k: K must be a multiple of 256, got %v", input_size, loc=loc)
	fmt.assertf(count == 1, "cuda linear_q6_k currently supports M=1 (decode); got M=%v", count, loc=loc)
	fmt.assertf(output.type == .Bf16, "cuda linear_q6_k requires Bf16 output (got %v)", output.type, loc=loc)

	if _linear_q6_k_mmvq_pipeline == nil {
		_linear_q6_k_mmvq_pipeline = _compile_pipeline(LINEAR_Q6_K_MMVQ_SRC, "linear_q6_k_mmvq.cu", "linear_q6_k_mmvq")
	}

	gctx := _gctx(loc)

	xp := data(input).ptr

	q8: cuda.DevicePtr
	if cached, ok := gctx.q8_1_cache[xp]; ok {
		q8 = cached
	} else {
		q8 = _emit_quantize_q8_1(gctx, xp, input_size, input.type, loc)
		gctx.q8_1_cache[xp] = q8
	}

	wp := data(weight).ptr
	yp := data(output).ptr
	M  := i32(count)
	K  := i32(input_size)
	N  := i32(output_size)

	mmvq_args := [?]rawptr{ &q8, &wp, &yp, &M, &K, &N }
	_dispatch(_linear_q6_k_mmvq_pipeline,
		u32(output_size), 1, 1,
		32, 4, 1,
		0, mmvq_args[:], loc)
}

// ----- Attention (no cache) --------------------------------------------------
//
// One block per (head, q_token); 64 threads run an online-softmax FA2 forward
// against contiguous K/V buffers. Caps head_size at MAX_D=512.

_attention_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	q   := op.input
	o   := op.output
	v   := op.variant.(ml.Attention)
	k   := v.key
	val := v.value

	token_count := q.shape[0]
	q_size      := q.shape[1]
	kv_size     := k.shape[1]
	head_size   := q_size / v.n_q_heads
	fmt.assertf(head_size <= 512, "cuda attention caps head_size at 512 (got %v)", head_size, loc=loc)
	fmt.assertf(q.type == .Bf16,  "cuda attention only supports Bf16 (got %v)", q.type, loc=loc)
	fmt.assertf(head_size % 2 == 0, "cuda bf16 attention requires even head_size (got %v)", head_size, loc=loc)
	fmt.assertf(v.window == 0 || v.causal, "cuda attention window > 0 requires causal=true", loc=loc)

	if _attention_bf16_pipeline == nil {
		_attention_bf16_pipeline = _compile_pipeline(ATTENTION_BF16_SRC, "attention_bf16.cu", "attention_bf16")
	}

	qp := data(q).ptr; kp := data(k).ptr; vp := data(val).ptr; op_ptr := data(o).ptr
	lse_ptr := data(v.lse).ptr
	n_q_heads  := i32(v.n_q_heads)
	n_kv_heads := i32(v.n_kv_heads)
	hs := i32(head_size); tc := i32(token_count); qs := i32(q_size); kvs := i32(kv_size)
	causal := i32(v.causal ? 1 : 0); window := i32(v.window)

	args := [?]rawptr{
		&qp, &kp, &vp, &op_ptr, &lse_ptr,
		&n_q_heads, &n_kv_heads, &hs, &tc, &qs, &kvs, &causal, &window,
	}
	_dispatch(_attention_bf16_pipeline, u32(v.n_q_heads), u32(token_count), 1, 64, 1, 1, 0, args[:], loc)
}

// ----- Attention with cache --------------------------------------------------
//
// Linear cache layout (= ggml's): K/V live in [0..capacity) by seq position,
// slot 0 oldest, slot cap-1 newest. For sliding layers in steady state
// (cache_position + n_rows > capacity), the host shifts the cache contents
// back by `shift_amount` rows via two cuMemcpyDtoDAsync (through a per-
// context scratch) before cache_write writes the new rows at the end of the
// cache. For full layers and sliding layers in their first `capacity`
// tokens, no shift is needed; cache_write linear-appends at slot
// `cache_position`.

_attention_cache_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	q       := op.input
	o       := op.output
	v       := op.variant.(ml.Attention_Cache)
	k       := v.key
	val     := v.value
	k_cache := v.k_cache
	v_cache := v.v_cache

	token_count := q.shape[0]
	q_size      := q.shape[1]
	kv_size     := k.shape[1]
	head_size   := q_size / v.n_q_heads
	fmt.assertf(head_size <= 512, "cuda attention_with_cache caps head_size at 512 (got %v)", head_size, loc=loc)
	fmt.assertf(q.type == .Bf16, "cuda attention_with_cache requires Bf16 Q (got %v)", q.type, loc=loc)
	fmt.assertf(k_cache.type == .Bf16, "cuda attention_with_cache requires Bf16 K cache (got %v)", k_cache.type, loc=loc)
	fmt.assertf(head_size % 2 == 0, "cuda attention_with_cache requires even head_size (got %v)", head_size, loc=loc)

	gctx := _gctx(loc)

	capacity := k_cache.shape[0]

	fmt.assertf(token_count <= capacity, "cuda attention_with_cache: q_token_count (%v) cannot exceed cache capacity (%v); chunk multi-token prefill if needed", token_count, capacity, loc=loc)

	// Gemma's KV-shared layers reuse the source layer's k_cache/v_cache and
	// re-pass the same K/V tensors here; without dedup, each shared layer
	// would re-shift and re-write the cache and clobber the source's history.
	// K and V are tracked separately so the fused `Rmsnorm_Rope_Write_Cache`
	// op can mark only the K side as done.
	k_cache_ptr := data(k_cache).ptr
	v_cache_ptr := data(v_cache).ptr
	k_already_written := k_cache_ptr in gctx.k_cache_written_this_forward
	v_already_written := v_cache_ptr in gctx.v_cache_written_this_forward

	// Shift sliding-layer cache contents back by `shift_amount` rows when the
	// next write would overflow capacity. Per-cache: each side only shifts on
	// its own first write of this forward (the fused rope op may have already
	// shifted K).
	row_bytes  := uint(kv_size) * 2 // bf16 = 2 bytes per element
	shift_amount := 0
	if v.window > 0 {
		// See note in `_rmsnorm_rope_write_cache_forward`: clamp shift to
		// `token_count` so we don't over-shift (and underflow preserved_rows)
		// once cache_position exceeds capacity.
		excess := v.cache_position + token_count - capacity
		shift_amount = min(max(excess, 0), token_count)
	}

	if shift_amount > 0 {
		preserved_rows  := capacity - shift_amount
		preserved_bytes := uint(preserved_rows) * row_bytes
		_ensure_shift_scratch(gctx, u64(preserved_bytes), loc)

		shift_src_offset := uint(shift_amount) * row_bytes

		if !k_already_written {
			k_dst := k_cache_ptr
			cuda.check(cuda.MemcpyDtoDAsync(gctx.shift_scratch_dev, k_dst + cuda.DevicePtr(shift_src_offset), preserved_bytes, gctx.stream), loc=loc)
			cuda.check(cuda.MemcpyDtoDAsync(k_dst, gctx.shift_scratch_dev, preserved_bytes, gctx.stream), loc=loc)
		}
		if !v_already_written {
			v_dst := v_cache_ptr
			cuda.check(cuda.MemcpyDtoDAsync(gctx.shift_scratch_dev, v_dst + cuda.DevicePtr(shift_src_offset), preserved_bytes, gctx.stream), loc=loc)
			cuda.check(cuda.MemcpyDtoDAsync(v_dst, gctx.shift_scratch_dev, preserved_bytes, gctx.stream), loc=loc)
		}
	}

	_emit_position_upload(gctx, v.cache_position, loc)

	if !k_already_written || !v_already_written {
		pos_dev := gctx.position_dev
		nr  := i32(token_count)
		kvs := i32(kv_size)
		cap := i32(capacity)

		pairs_total := token_count * (kv_size / 2)
		grid        := _div_up(pairs_total, 256)

		if !k_already_written {
			k_src := data(k).ptr
			k_dst := k_cache_ptr
			k_args := [?]rawptr{ &k_src, &k_dst, &pos_dev, &nr, &kvs, &cap }
			_dispatch_cache_write(k.type, grid, k_args[:], loc)
		}
		if !v_already_written {
			v_src := data(val).ptr
			v_dst := v_cache_ptr
			v_args := [?]rawptr{ &v_src, &v_dst, &pos_dev, &nr, &kvs, &cap }
			_dispatch_cache_write(val.type, grid, v_args[:], loc)
		}
	}

	gctx.k_cache_written_this_forward[k_cache_ptr] = true
	gctx.v_cache_written_this_forward[v_cache_ptr] = true

	qp := data(q).ptr; kcp := data(k_cache).ptr; vcp := data(v_cache).ptr; op_ptr := data(o).ptr
	n_q_heads  := i32(v.n_q_heads)
	n_kv_heads := i32(v.n_kv_heads)
	hs := i32(head_size); qtc := i32(token_count); pos_dev := gctx.position_dev
	qs := i32(q_size); kvs := i32(kv_size); window := i32(v.window); cap := i32(capacity)

	args := [?]rawptr{
		&qp, &kcp, &vcp, &op_ptr,
		&n_q_heads, &n_kv_heads, &hs, &qtc, &pos_dev, &qs, &kvs, &window, &cap,
	}

	// ggml-style vec kernel with cooperative threads-per-K-dot and vectorized V
	// loads. Compiled per-D via NVRTC so per-D loops unroll at compile time.
	switch head_size {
	case 256:
		if _attention_cache_vec_bf16_d256_pipeline == nil {
			opts := [?]cstring{"-DD_HEAD=256"}
			_attention_cache_vec_bf16_d256_pipeline = _compile_pipeline(ATTENTION_CACHE_VEC_BF16_SRC, "attention_cache_vec_bf16_d256.cu", "attention_cache_vec_bf16", opts[:])
		}
		_dispatch(_attention_cache_vec_bf16_d256_pipeline, u32(v.n_q_heads), u32(token_count), 1, 32, 4, 1, 0, args[:], loc)
	case 512:
		if _attention_cache_vec_bf16_d512_pipeline == nil {
			opts := [?]cstring{"-DD_HEAD=512"}
			_attention_cache_vec_bf16_d512_pipeline = _compile_pipeline(ATTENTION_CACHE_VEC_BF16_SRC, "attention_cache_vec_bf16_d512.cu", "attention_cache_vec_bf16", opts[:])
		}
		_dispatch(_attention_cache_vec_bf16_d512_pipeline, u32(v.n_q_heads), u32(token_count), 1, 32, 4, 1, 0, args[:], loc)
	case:
		if _attention_cache_bf16_pipeline == nil {
			_attention_cache_bf16_pipeline = _compile_pipeline(ATTENTION_CACHE_BF16_SRC, "attention_cache_bf16.cu", "attention_cache_bf16")
		}
		_dispatch(_attention_cache_bf16_pipeline, u32(v.n_q_heads), u32(token_count), 1, 64, 1, 1, 0, args[:], loc)
	}
}

// ----- Select (embedding lookup) ---------------------------------------------
//
// `indices` is a CPU-side []int. We upload it to a per-call scratch buffer
// from the activation pool (recycles across clears, mirrors the vulkan
// path's transient indices buffer).

_upload_indices :: proc(gctx: ^Context, indices: []int, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	bytes := uint(builtin.len(indices) * size_of(u32))
	dev_ptr := _activation_alloc(gctx, u64(bytes), loc)

	// Stage to a temp host []u32 in temp_allocator, then async H2D.
	host := builtin.make([]u32, builtin.len(indices), context.temp_allocator)
	for v, i in indices { host[i] = u32(v) }
	cuda.check(cuda.MemcpyHtoDAsync(dev_ptr, raw_data(host), bytes, gctx.stream), loc=loc)
	return dev_ptr
}

_select_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x       := op.input
	y       := op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(y) / builtin.len(indices)

	gctx := _gctx(loc)
	idx_ptr := _upload_indices(gctx, indices, loc)

	xp := data(x).ptr; yp := data(y).ptr
	n_idx := i32(builtin.len(indices)); s := i32(size)

	#partial switch x.type {
	case .Bf16:
		fmt.assertf(size % 2 == 0, "cuda bf16 select requires even row size (got %v)", size, loc=loc)
		if _select_bf16_pipeline == nil {
			_select_bf16_pipeline = _compile_pipeline(SELECT_BF16_SRC, "select_bf16.cu", "select_bf16")
		}
		pair_count := size / 2
		args := [?]rawptr{ &xp, &idx_ptr, &yp, &n_idx, &s }
		_dispatch(_select_bf16_pipeline, _div_up(pair_count, 256), u32(builtin.len(indices)), 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		if _select_f32_pipeline == nil {
			_select_f32_pipeline = _compile_pipeline(SELECT_F32_SRC, "select.cu", "select_f32")
		}
		args := [?]rawptr{ &xp, &idx_ptr, &yp, &n_idx, &s }
		_dispatch(_select_f32_pipeline, _div_up(size, 256), u32(builtin.len(indices)), 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("select: unsupported dtype %v", x.type, loc=loc)
	}
}

// ----- Slice_Trailing --------------------------------------------------------
_slice_trailing_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Slice_Trailing)

	trailing     := x.shape[x.rank - 1]
	new_trailing := y.shape[y.rank - 1]
	leading      := ml.len(x) / trailing

	xp := data(x).ptr; yp := data(y).ptr
	ld := i32(leading); tr := i32(trailing); nt := i32(new_trailing); st := i32(v.start)

	#partial switch x.type {
	case .Bf16:
		if _slice_trailing_bf16_pipeline == nil {
			_slice_trailing_bf16_pipeline = _compile_pipeline(SLICE_TRAILING_BF16_SRC, "slice_trailing_bf16.cu", "slice_trailing_bf16")
		}
		pair_count := (leading * new_trailing + 1) / 2
		pc := i32(pair_count)
		args := [?]rawptr{ &xp, &yp, &ld, &tr, &nt, &st, &pc }
		_dispatch(_slice_trailing_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		if _slice_trailing_f32_pipeline == nil {
			_slice_trailing_f32_pipeline = _compile_pipeline(SLICE_TRAILING_F32_SRC, "slice_trailing.cu", "slice_trailing_f32")
		}
		args := [?]rawptr{ &xp, &yp, &ld, &tr, &nt, &st }
		_dispatch(_slice_trailing_f32_pipeline, _div_up(leading * new_trailing, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("slice_trailing: unsupported dtype %v", x.type, loc=loc)
	}
}
