package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"

import "bindings/cuda"
import "bindings/cublas"

import ml "../.."

























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

// CUDA caps grid dimensions Y and Z at 65535 (only X reaches 2^31-1). A kernel that maps a
// batch-sized count onto grid.y must clamp it here and grid-stride the rest inside the kernel, or
// the launch is rejected with INVALID_VALUE the moment the count crosses this line.
MAX_GRID_DIM_YZ :: 65535

_dispatch_cache_write :: proc(src_type: ml.Data_Type, grid: u32, args: []rawptr, loc: runtime.Source_Code_Location) {
	fmt.assertf(src_type == .Bf16, "unsupported src dtype %v", src_type, loc=loc)
	_cache_write_bf16_pipeline := _compile_pipeline(CACHE_WRITE_BF16_SRC, "cache_write_bf16.cu", "cache_write_bf16")
	_dispatch(_cache_write_bf16_pipeline, grid, 1, 1, 256, 1, 1, 0, args, loc)
}

_emit_quantize_q8_1 :: proc(gctx: ^Context, xp: cuda.DevicePtr, input_size: int, input_type: ml.Data_Type, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	xp := xp
	fmt.assertf(input_type == .Bf16, "unsupported input dtype %v", input_type, loc=loc)
	q8_block_count := input_size / Q8_1_BLOCK_ELEMS
	q8_byte_count  := q8_block_count * Q8_1_BLOCK_BYTES
	q8 := _activation_alloc(gctx, u64(q8_byte_count), loc)
	K  := i32(input_size)
	args := [?]rawptr{&xp, &q8, &K }

	_quantize_q8_1_pipeline := _compile_pipeline(QUANTIZE_Q8_1_BF16_SRC, "quantize_q8_1_bf16.cu", "quantize_q8_1_bf16")
	_dispatch(_quantize_q8_1_pipeline,
		_div_up(input_size, 256), 1, 1,
		256, 1, 1,
		0, args[:], loc)
	return q8
}

_emit_position_upload :: proc(gctx: ^Context, value: int, loc: runtime.Source_Code_Location) {
	if gctx.position_written_this_forward {
		fmt.assertf(value == gctx.position_value_this_forward,
			"conflicting positions in one forward pass: %v after %v (position_dev is shared per forward)",
			value, gctx.position_value_this_forward, loc=loc)
		return
	}
	(^i32)(gctx.position_pinned)^ = i32(value)
	cuda.check(cuda.MemcpyHtoDAsync(gctx.position_dev, gctx.position_pinned, 4, gctx.stream), loc=loc)
	gctx.position_written_this_forward = true
	gctx.position_value_this_forward   = value
}

_ensure_shift_scratch :: proc(gctx: ^Context, byte_count: u64, loc: runtime.Source_Code_Location) {
	if gctx.shift_scratch_size >= byte_count {
		return
	}
	if gctx.shift_scratch_dev != 0 {
		cuda.check(cuda.MemFree(gctx.shift_scratch_dev), loc=loc)
	}
	cuda.check(cuda.MemAlloc(&gctx.shift_scratch_dev, uint(byte_count)), loc=loc)
	gctx.shift_scratch_size = byte_count
}

_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	#partial switch _ in op.variant {
	case ml.Add:          _add_forward(op, loc)
	case ml.Mul:          _mul_forward(op, loc)
	case ml.Gelu_Mul:     _gelu_mul_forward(op, loc)
	case ml.Gelu:         _gelu_forward(op, loc)
	case ml.Silu:         _silu_forward(op, loc)
	case ml.Tanh:         _tanh_forward(op, loc)
	case ml.Exp:          _exp_forward(op, loc)
	case ml.Clamp:        _clamp_forward(op, loc)
	case ml.Min:          _min_forward(op, loc)
	case ml.Softmax:      _softmax_forward(op, loc)
	case ml.Entropy:      _entropy_forward(op, loc)
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
	case ml.Cross_Entropy:   _cross_entropy_forward(op, loc)
	case ml.Select:          _select_forward(op, loc)
	case ml.Slice_Trailing:  _slice_trailing_forward(op, loc)
	case ml.Slice_Leading:   _slice_leading_forward(op, loc)
	case ml.Sub:                _sub_forward(op, loc)
	case ml.Div:                _div_forward(op, loc)
	case ml.Max:                _max_forward(op, loc)
	case ml.Sqrt:               _sqrt_forward(op, loc)
	case ml.Relu:               _relu_forward(op, loc)
	case ml.Sigmoid:            _sigmoid_forward(op, loc)
	case ml.Mean:               _mean_forward(op, loc)
	case ml.Sum:                _sum_forward(op, loc)
	case ml.Max_Reduce:         _max_reduce_forward(op, loc)
	case ml.Im2col:             _im2col_forward(op, loc)
	case ml.Max_Pool2d:         _max_pool2d_forward(op, loc)
	case ml.Avg_Pool2d:         _avg_pool2d_forward(op, loc)
	case ml.Transpose:          _transpose_forward(op, loc)
	case ml.Slice:              _slice_forward(op, loc)
	case ml.Concat:             _concat_forward(op, loc)
	case ml.Layernorm:          _layernorm_forward(op, loc)
	case ml.Log_Softmax:        _log_softmax_forward(op, loc)
	case ml.Mean_Squared_Error: _mean_squared_error_forward(op, loc)
	case ml.Smooth_L1:          _smooth_l1_forward(op, loc)
	case ml.Batched_Matmul:     _batched_matmul_forward(op, loc)
	case ml.Permute:            _permute_forward(op, loc)
	case ml.Causal_Mask:        _causal_mask_forward(op, loc)
	case ml.Lerp_Assign:        _lerp_assign_forward(op, loc)
	case ml.Accumulate_Mean:    _accumulate_mean_forward(op, loc)
	case: fmt.panicf("forward not implemented for op variant %T", op.variant, loc=loc)
	}
}

_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	#partial switch _ in op.variant {
	case ml.Add:            _add_backward(op, loc)
	case ml.Mul:            _mul_backward(op, loc)
	case ml.Linear:         _linear_backward(op, loc)
	case ml.Linear_Q4_K:    _linear_q4_k_backward(op, loc)
	case ml.Linear_Q6_K:    _linear_q6_k_backward(op, loc)
	case ml.Silu:           _silu_backward(op, loc)
	case ml.Gelu:           _gelu_backward(op, loc)
	case ml.Tanh:           _tanh_backward(op, loc)
	case ml.Exp:            _exp_backward(op, loc)
	case ml.Clamp:          _clamp_backward(op, loc)
	case ml.Min:            _min_backward(op, loc)
	case ml.Softmax:        _softmax_backward(op, loc)
	case ml.Entropy:        _entropy_backward(op, loc)
	case ml.Select:         _select_backward(op, loc)
	case ml.Slice_Trailing: _slice_trailing_backward(op, loc)
	case ml.Slice_Leading:  _slice_leading_backward(op, loc)
	case ml.Rmsnorm:        _rmsnorm_backward(op, loc)
	case ml.Rope:           _rope_backward(op, loc)
	case ml.Attention:      _attention_backward(op, loc)
	case ml.Cross_Entropy:  _cross_entropy_backward(op, loc)
	case ml.Cast:           _cast_backward(op, loc)
	case ml.Sub:                _sub_backward(op, loc)
	case ml.Div:                _div_backward(op, loc)
	case ml.Max:                _max_backward(op, loc)
	case ml.Sqrt:               _sqrt_backward(op, loc)
	case ml.Relu:               _relu_backward(op, loc)
	case ml.Sigmoid:            _sigmoid_backward(op, loc)
	case ml.Mean:               _mean_backward(op, loc)
	case ml.Sum:                _sum_backward(op, loc)
	case ml.Max_Reduce:         _max_reduce_backward(op, loc)
	case ml.Im2col:             _im2col_backward(op, loc)
	case ml.Max_Pool2d:         _max_pool2d_backward(op, loc)
	case ml.Avg_Pool2d:         _avg_pool2d_backward(op, loc)
	case ml.Transpose:          _transpose_backward(op, loc)
	case ml.Slice:              _slice_backward(op, loc)
	case ml.Concat:             _concat_backward(op, loc)
	case ml.Layernorm:          _layernorm_backward(op, loc)
	case ml.Log_Softmax:        _log_softmax_backward(op, loc)
	case ml.Mean_Squared_Error: _mean_squared_error_backward(op, loc)
	case ml.Smooth_L1:          _smooth_l1_backward(op, loc)
	case ml.Batched_Matmul:     _batched_matmul_backward(op, loc)
	case ml.Permute:            _permute_backward(op, loc)
	case ml.Causal_Mask:        _causal_mask_backward(op, loc)
	case:                   fmt.panicf("backward not implemented for op variant %T", op.variant, loc=loc)
	}
}

_cast_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	if gradient(x).ptr == 0 { return }

	// With F32 grads, cast backward is dx += dy in f32 regardless of the
	// forward cast direction. The data-side conversion has no effect on
	// the gradient flow.
	_cast_back_f32_pipeline := _compile_pipeline(CAST_BACK_F32_SRC, "cast_back_f32.cu", "cast_back_f32")

	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	n   := i32(ml.len(x))
	args := [?]rawptr{&dyp, &dxp, &n}
	_dispatch(_cast_back_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_update :: proc(opt: ml.Optimizer, t: ml.Tensor, m_buf, v_buf: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	d := data(t).ptr
	g := gradient(t).ptr
	m := transmute(Gpu_Buffer)m_buf
	v := transmute(Gpu_Buffer)v_buf
	fmt.assertf(m.ptr != 0, "optimizer moment m is nil", loc=loc)
	fmt.assertf(v.ptr != 0, "optimizer moment v is nil", loc=loc)
	mp := m.ptr; vp := v.ptr

	n  := i32(t.count)
	b1 := opt.beta1; b2 := opt.beta2
	c1 := opt.bias_correction1; c2 := opt.bias_correction2
	lr := opt.learning_rate; wd := opt.weight_decay; eps := opt.epsilon

	args := [?]rawptr{&d, &g, &mp, &vp, &n, &b1, &b2, &c1, &c2, &lr, &wd, &eps}

	#partial switch t.type {
	case .F32:
		_adam_f32_pipeline := _compile_pipeline(ADAM_F32_SRC, "adam_f32.cu", "adam_f32")
		_dispatch(_adam_f32_pipeline, _div_up(t.count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_adam_bf16_pipeline := _compile_pipeline(ADAM_BF16_SRC, "adam_bf16.cu", "adam_bf16")
		_dispatch(_adam_bf16_pipeline, _div_up(t.count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", t.type, loc=loc)
	}
}

_sq_sum_accumulate :: proc(buffer: ml.Backend_Buffer, count: int, accumulator: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	b := transmute(Gpu_Buffer)buffer
	a := transmute(Gpu_Buffer)accumulator
	bp := b.ptr; ap := a.ptr; n := i32(count)
	args := [?]rawptr{&bp, &ap, &n}

	pipeline := _compile_pipeline(SQ_SUM_F32_SRC, "sq_sum_f32.cu", "sq_sum_f32")
	_dispatch(pipeline, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_scale :: proc(buffer: ml.Backend_Buffer, count: int, scale: f32, loc: runtime.Source_Code_Location) {
	b := transmute(Gpu_Buffer)buffer
	bp := b.ptr; n := i32(count); s := scale
	args := [?]rawptr{&bp, &n, &s}

	pipeline := _compile_pipeline(SCALE_F32_SRC, "scale_f32.cu", "scale_f32")
	_dispatch(pipeline, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_linear_dtype :: proc(t: ml.Data_Type, loc: runtime.Source_Code_Location) -> cublas.DataType {
	#partial switch t {
	case .Bf16: return .R_16BF
	case .F32:  return .R_32F
	case:       fmt.panicf("unsupported dtype %v", t, loc=loc)
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

	alpha := f32(1.0)
	beta  := f32(0.0)

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

	alpha := f32(1.0)
	beta  := f32(1.0)

	w_ptr  := data    (weight).ptr
	x_ptr  := data    (input ).ptr
	dy_ptr := gradient(output).ptr
	dx_ptr := gradient(input ).ptr
	dw_ptr := gradient(weight).ptr

	// Frozen weight (e.g. QLoRA base): no gradient buffer, skip dW GEMM.
	have_dw := dw_ptr != 0
	have_dx := dx_ptr != 0

	// Gradients are f32 always. For bf16 weights/inputs we cast f32 dy down
	// to bf16 once, then run two Tensor-Core GEMMs (bf16 x bf16 -> f32).
	x_dt := _linear_dtype(input.type, loc)
	w_dt := _linear_dtype(weight.type, loc)
	dy_for_gemm := dy_ptr
	dy_dt := cublas.DataType.R_32F

	if input.type == .Bf16 {
		dy_count   := int(count) * int(output_size)
		dy_bf_size := uintptr(dy_count) * 2
		dy_bf_ptr  := _activation_alloc(gctx, u64(dy_bf_size), loc)

		_cast_f32_to_bf16_pipeline := _compile_pipeline(CAST_F32_TO_BF16_SRC, "cast_f32_to_bf16.cu", "cast_f32_to_bf16")
		dy_p := dy_ptr
		bf_p := dy_bf_ptr
		nc   := i32(dy_count)
		pc   := i32((dy_count + 1) / 2)
		cast_args := [?]rawptr{&dy_p, &bf_p, &nc, &pc}
		_dispatch(_cast_f32_to_bf16_pipeline, _div_up((dy_count + 1) / 2, 256), 1, 1, 256, 1, 1, 0, cast_args[:], loc)

		dy_for_gemm = dy_bf_ptr
		dy_dt       = .R_16BF
	}

	if have_dx {
		cublas.check(cublas.GemmEx(
			gctx.cublas_handle,
			.N, .N,
			input_size, count, output_size,
			&alpha,
			rawptr(uintptr(w_ptr      )), w_dt,    input_size,
			rawptr(uintptr(dy_for_gemm)), dy_dt,   output_size,
			&beta,
			rawptr(uintptr(dx_ptr     )), .R_32F,  input_size,
			._32F,
			.DEFAULT,
		), loc=loc)
	}

	if have_dw {
		cublas.check(cublas.GemmEx(
			gctx.cublas_handle,
			.N, .T,
			input_size, output_size, count,
			&alpha,
			rawptr(uintptr(x_ptr      )), x_dt,    input_size,
			rawptr(uintptr(dy_for_gemm)), dy_dt,   output_size,
			&beta,
			rawptr(uintptr(dw_ptr     )), .R_32F,  input_size,
			._32F,
			.DEFAULT,
		), loc=loc)
	}
}

_linear_q4_k_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q4_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "K must be a multiple of 256, got %v", input_size, loc=loc)
	fmt.assertf(output.type == .Bf16, "linear_q4_k requires Bf16 output (got %v)", output.type, loc=loc)

	if count == 1 {
		// Decode-time fast path: per-element fused dequant+matmul.
		_linear_q4_k_forward_mmvq(op, loc)
		return
	}

	// Training / multi-token path: dequantize the whole weight to bf16
	// scratch, run a Tensor-Core bf16 GEMM. Scratch is freed after the
	// activation pool resets.
	gctx := _gctx(loc)
	w_bf := _dequantize_q4_k(gctx, data(weight).ptr, output_size, input_size, loc)
	xp   := data(input).ptr
	yp   := data(output).ptr

	alpha := f32(1.0)
	beta  := f32(0.0)
	cublas.check(cublas.GemmEx(
		gctx.cublas_handle,
		.T,  .N,
		i32(output_size), i32(count), i32(input_size),
		&alpha,
		rawptr(uintptr(w_bf)), .R_16BF, i32(input_size),
		rawptr(uintptr(xp)),   .R_16BF, i32(input_size),
		&beta,
		rawptr(uintptr(yp)),   .R_16BF, i32(output_size),
		._32F,
		.DEFAULT,
	), loc=loc)
}

_linear_q4_k_forward_mmvq :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q4_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]

	_linear_q4_k_mmvq_pipeline := _compile_pipeline(LINEAR_Q4_K_MMVQ_SRC, "linear_q4_k_mmvq.cu", "linear_q4_k_mmvq")

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
	M  := i32(1)
	K  := i32(input_size)
	N  := i32(output_size)

	mmvq_args := [?]rawptr{&q8, &wp, &yp, &M, &K, &N}
	_dispatch(_linear_q4_k_mmvq_pipeline,
		u32(output_size), 1, 1,
		32, 4, 1,
		0, mmvq_args[:], loc)
}

// Backward for Linear_Q4_K: only computes dx (Q4_K weight is frozen by
// design - that's the QLoRA recipe). Dequantizes W to bf16 scratch,
// casts f32 dy to bf16 scratch, runs a Tensor-Core bf16 GEMM with f32
// output for dx.
_linear_q4_k_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q4_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	dxp := gradient(input).ptr
	if dxp == 0 {
		// No gradient consumer downstream; nothing to do.
		return
	}

	gctx := _gctx(loc)

	// Dequantize W to bf16 scratch.
	w_bf := _dequantize_q4_k(gctx, data(weight).ptr, output_size, input_size, loc)

	// Cast f32 dy to bf16 scratch.
	dy_count   := count * output_size
	dy_bf_size := uintptr(dy_count) * 2
	dy_bf_ptr  := _activation_alloc(gctx, u64(dy_bf_size), loc)
	_cast_f32_to_bf16_pipeline := _compile_pipeline(CAST_F32_TO_BF16_SRC, "cast_f32_to_bf16.cu", "cast_f32_to_bf16")
	dy_p := gradient(output).ptr
	bf_p := dy_bf_ptr
	nc   := i32(dy_count)
	pc   := i32((dy_count + 1) / 2)
	cast_args := [?]rawptr{&dy_p, &bf_p, &nc, &pc}
	_dispatch(_cast_f32_to_bf16_pipeline, _div_up((dy_count + 1) / 2, 256), 1, 1, 256, 1, 1, 0, cast_args[:], loc)

	// dx (f32) = W^T (output_size x input_size, transposed) @ dy_bf
	// Forward: y = x @ W^T (W stored as [out, in])
	// dx = dy @ W (no transpose on W since y = x @ W^T means dx = dy @ (W^T)^T = dy @ W)
	alpha := f32(1.0)
	beta  := f32(1.0)
	cublas.check(cublas.GemmEx(
		gctx.cublas_handle,
		.N, .N,
		i32(input_size), i32(count), i32(output_size),
		&alpha,
		rawptr(uintptr(w_bf)),    .R_16BF, i32(input_size),
		rawptr(uintptr(dy_bf_ptr)), .R_16BF, i32(output_size),
		&beta,
		rawptr(uintptr(dxp)),     .R_32F,  i32(input_size),
		._32F,
		.DEFAULT,
	), loc=loc)
}

// Dequantize a Q4_K weight buffer to a bf16 scratch tensor.
//
// The result is cached in `gctx.dequant_cache` keyed by the source weight
// pointer, so a forward followed by a backward over the same weight only
// pays for one dequantization. The scratch lives in the activation pool
// and is invalidated at the next ml.clear() (which also clears the cache).
_dequantize_q4_k :: proc(gctx: ^Context, src: cuda.DevicePtr, output_size, input_size: int, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	if cached, ok := gctx.dequant_cache[src]; ok {
		return cached
	}
	src := src
	count := output_size * input_size
	fmt.assertf(count % ml.K_QUANT_BLOCK_SIZE == 0, "count %v not divisible by 256", count, loc=loc)
	num_blocks := count / ml.K_QUANT_BLOCK_SIZE

	dst := _activation_alloc(gctx, u64(count) * 2, loc)

	_dequantize_q4_k_to_bf16_pipeline := _compile_pipeline(DEQUANTIZE_Q4_K_TO_BF16_SRC, "dequantize_q4_k_to_bf16.cu", "dequantize_q4_k_to_bf16")
	total := i32(count)
	args := [?]rawptr{&src, &dst, &total}
	_dispatch(_dequantize_q4_k_to_bf16_pipeline, u32(num_blocks), 1, 1, 256, 1, 1, 0, args[:], loc)

	gctx.dequant_cache[src] = dst
	return dst
}

_linear_q4_k_gate_up_geglu_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	v           := op.variant.(ml.Linear_Q4_K_Gate_Up_Geglu)
	w_gate      := v.w_gate
	w_up        := v.w_up
	output      := op.output
	output_size := w_gate.shape[0]
	input_size  := w_gate.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "K must be a multiple of 256, got %v", input_size, loc=loc)
	fmt.assertf(count == 1, "linear_q4_k_gate_up_geglu requires M=1 (decode); got M=%v", count, loc=loc)
	fmt.assertf(output.type == .Bf16, "linear_q4_k_gate_up_geglu requires Bf16 output (got %v)", output.type, loc=loc)

	_linear_q4_k_gate_up_geglu_bf16_pipeline := _compile_pipeline(LINEAR_Q4_K_GATE_UP_GEGLU_BF16_SRC, "linear_q4_k_gate_up_geglu_bf16.cu", "linear_q4_k_gate_up_geglu_bf16")

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
	yp  := data(output).ptr
	M   := i32(count)
	K   := i32(input_size)
	N   := i32(output_size)

	mmvq_args := [?]rawptr{&q8, &wgp, &wup, &yp, &M, &K, &N}
	_dispatch(_linear_q4_k_gate_up_geglu_bf16_pipeline,
		u32(output_size), 1, 1,
		32, 4, 1,
		0, mmvq_args[:], loc,
	)
}

_add_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b

	#partial switch a.type {
	case .F32:
		_add_pipeline := _compile_pipeline(ADD_F32_SRC, "add.cu", "add_f32")
		ap := data(a).ptr; bp := data(b).ptr; cp := data(output).ptr
		n   := i32(ml.len(a))
		n_b := i32(ml.len(b))
		args := [?]rawptr{&ap, &bp, &cp, &n, &n_b}
		grid := _div_up(ml.len(a), ADD_LOCAL_SIZE)
		_dispatch(_add_pipeline, grid, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args[:], loc)

	case .Bf16:
		_add_bf16_pipeline := _compile_pipeline(ADD_BF16_SRC, "add_bf16.cu", "add_bf16")
		ap := data(a).ptr; bp := data(b).ptr; cp := data(output).ptr
		pair_count := (ml.len(a) + 1) / 2
		n          := i32(ml.len(a))
		n_b        := i32(ml.len(b))
		pc         := i32(pair_count)
		args := [?]rawptr{&ap, &bp, &cp, &n, &n_b, &pc}
		grid := _div_up(pair_count, ADD_LOCAL_SIZE)
		_dispatch(_add_bf16_pipeline, grid, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args[:], loc)

	case:
		fmt.panicf("unsupported dtype %v", a.type, loc=loc)
	}
}

_add_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b
	stride := ml.len(a) / ml.len(b)

	dyp := gradient(output).ptr
	dap := gradient(a).ptr
	dbp := gradient(b).ptr

	if dap != 0 {
		_add_back_a_pipeline := _compile_pipeline(ADD_BACK_A_SRC, "add_back_a.cu", "add_back_a_f32")
		n := i32(ml.len(a))
		args_a := [?]rawptr{&dyp, &dap, &n}
		grid_a := _div_up(ml.len(a), ADD_LOCAL_SIZE)
		_dispatch(_add_back_a_pipeline, grid_a, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args_a[:], loc)
	}

	if dbp != 0 {
		_add_back_b_pipeline := _compile_pipeline(ADD_BACK_B_SRC, "add_back_b.cu", "add_back_b_f32")
		n_b := i32(ml.len(b)); st := i32(stride)
		args_b := [?]rawptr{&dyp, &dbp, &n_b, &st}
		grid_b := _div_up(ml.len(b), ADD_LOCAL_SIZE)
		_dispatch(_add_back_b_pipeline, grid_b, 1, 1, ADD_LOCAL_SIZE, 1, 1, 0, args_b[:], loc)
	}
}

_mul_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a := op.input
	b := op.variant.(ml.Mul).b
	c := op.output

	#partial switch a.type {
	case .F32:
		_mul_pipeline := _compile_pipeline(MUL_F32_SRC, "mul.cu", "mul_f32")
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		n   := i32(ml.len(a)); n_b := i32(ml.len(b))
		args := [?]rawptr{&ap, &bp, &cp, &n, &n_b}
		_dispatch(_mul_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_mul_bf16_pipeline := _compile_pipeline(MUL_BF16_SRC, "mul_bf16.cu", "mul_bf16")
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		pair_count := (ml.len(a) + 1) / 2
		n   := i32(ml.len(a)); n_b := i32(ml.len(b)); pc := i32(pair_count)
		args := [?]rawptr{&ap, &bp, &cp, &n, &n_b, &pc}
		_dispatch(_mul_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", a.type, loc=loc)
	}
}

_gelu_mul_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a := op.input
	b := op.variant.(ml.Gelu_Mul).b
	c := op.output

	#partial switch a.type {
	case .Bf16:
		_gelu_mul_bf16_pipeline := _compile_pipeline(GELU_MUL_BF16_SRC, "gelu_mul_bf16.cu", "gelu_mul_bf16")
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		pair_count := (ml.len(a) + 1) / 2
		n   := i32(ml.len(a)); n_b := i32(ml.len(b)); pc := i32(pair_count)
		args := [?]rawptr{&ap, &bp, &cp, &n, &n_b, &pc}
		_dispatch(_gelu_mul_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		_gelu_mul_f32_pipeline := _compile_pipeline(GELU_MUL_F32_SRC, "gelu_mul_f32.cu", "gelu_mul_f32")
		ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
		n   := i32(ml.len(a)); n_b := i32(ml.len(b))
		args := [?]rawptr{&ap, &bp, &cp, &n, &n_b}
		_dispatch(_gelu_mul_f32_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", a.type, loc=loc)
	}
}

_tanh_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	#partial switch x.type {
	case .F32:
		_tanh_pipeline := _compile_pipeline(TANH_F32_SRC,          "tanh.cu",          "tanh_f32")
		xp := data(x).ptr; yp := data(y).ptr
		n := i32(ml.len(x))
		args := [?]rawptr{&xp, &yp, &n}
		_dispatch(_tanh_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_tanh_bf16_pipeline := _compile_pipeline(TANH_BF16_SRC,         "tanh_bf16.cu",     "tanh_bf16")
		xp := data(x).ptr; yp := data(y).ptr
		pair_count := (ml.len(x) + 1) / 2
		n := i32(ml.len(x)); pc := i32(pair_count)
		args := [?]rawptr{&xp, &yp, &n, &pc}
		_dispatch(_tanh_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_exp_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	#partial switch x.type {
	case .F32:
		_exp_pipeline := _compile_pipeline(EXP_F32_SRC, "exp.cu", "exp_f32")
		xp := data(x).ptr; yp := data(y).ptr
		n := i32(ml.len(x))
		args := [?]rawptr{&xp, &yp, &n}
		_dispatch(_exp_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_exp_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	if gradient(x).ptr == 0 { return }

	yp  := data(y).ptr
	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	n   := i32(ml.len(x))
	args := [?]rawptr{&yp, &dyp, &dxp, &n}

	#partial switch x.type {
	case .F32:
		_exp_back_f32_pipeline := _compile_pipeline(EXP_BACK_F32_SRC, "exp_back_f32.cu", "exp_back_f32")
		_dispatch(_exp_back_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_clamp_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Clamp)
	lo := v.min_val; hi := v.max_val

	#partial switch x.type {
	case .F32:
		_clamp_pipeline := _compile_pipeline(CLAMP_F32_SRC, "clamp.cu", "clamp_f32")
		xp := data(x).ptr; yp := data(y).ptr
		n := i32(ml.len(x))
		args := [?]rawptr{&xp, &yp, &lo, &hi, &n}
		_dispatch(_clamp_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_clamp_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Clamp)
	lo := v.min_val; hi := v.max_val

	if gradient(x).ptr == 0 { return }

	xp  := data(x).ptr
	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	n   := i32(ml.len(x))
	args := [?]rawptr{&xp, &dyp, &dxp, &lo, &hi, &n}

	#partial switch x.type {
	case .F32:
		_clamp_back_f32_pipeline := _compile_pipeline(CLAMP_BACK_F32_SRC, "clamp_back_f32.cu", "clamp_back_f32")
		_dispatch(_clamp_back_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_softmax_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	cols := x.shape[x.rank - 1]
	rows := ml.len(x) / cols

	#partial switch x.type {
	case .F32:
		_softmax_pipeline := _compile_pipeline(SOFTMAX_F32_SRC, "softmax.cu", "softmax_f32")
		xp := data(x).ptr; yp := data(y).ptr
		rr := i32(rows); cc := i32(cols)
		args := [?]rawptr{&xp, &yp, &rr, &cc}
		_dispatch(_softmax_pipeline, _div_up(rows, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_softmax_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	cols := x.shape[x.rank - 1]
	rows := ml.len(x) / cols

	if gradient(x).ptr == 0 { return }

	yp := data(y).ptr; dyp := gradient(y).ptr; dxp := gradient(x).ptr
	rr := i32(rows); cc := i32(cols)
	args := [?]rawptr{&yp, &dyp, &dxp, &rr, &cc}

	#partial switch x.type {
	case .F32:
		_softmax_back_f32_pipeline := _compile_pipeline(SOFTMAX_BACK_F32_SRC, "softmax_back_f32.cu", "softmax_back_f32")
		_dispatch(_softmax_back_f32_pipeline, _div_up(rows, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_entropy_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	p := op.input
	h := op.output
	cols := p.shape[p.rank - 1]
	rows := ml.len(p) / cols

	#partial switch p.type {
	case .F32:
		_entropy_pipeline := _compile_pipeline(ENTROPY_F32_SRC, "entropy.cu", "entropy_f32")
		pp := data(p).ptr; hp := data(h).ptr
		rr := i32(rows); cc := i32(cols)
		args := [?]rawptr{&pp, &hp, &rr, &cc}
		_dispatch(_entropy_pipeline, _div_up(rows, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", p.type, loc=loc)
	}
}

_entropy_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	p := op.input
	h := op.output
	cols := p.shape[p.rank - 1]
	rows := ml.len(p) / cols

	if gradient(p).ptr == 0 { return }

	pp := data(p).ptr; dyp := gradient(h).ptr; dpp := gradient(p).ptr
	rr := i32(rows); cc := i32(cols)
	args := [?]rawptr{&pp, &dyp, &dpp, &rr, &cc}

	#partial switch p.type {
	case .F32:
		_entropy_back_f32_pipeline := _compile_pipeline(ENTROPY_BACK_F32_SRC, "entropy_back_f32.cu", "entropy_back_f32")
		_dispatch(_entropy_back_f32_pipeline, _div_up(rows, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", p.type, loc=loc)
	}
}

_cast_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	if x.type == y.type {
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
		_cast_bf16_to_f32_pipeline := _compile_pipeline(CAST_BF16_TO_F32_SRC,  "cast_bf16_to_f32.cu", "cast_bf16_to_f32")
		args := [?]rawptr{&xp, &yp, &n, &pc}
		_dispatch(_cast_bf16_to_f32_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case x.type == .F32 && y.type == .Bf16:
		_cast_f32_to_bf16_pipeline := _compile_pipeline(CAST_F32_TO_BF16_SRC,  "cast_f32_to_bf16.cu", "cast_f32_to_bf16")
		args := [?]rawptr{&xp, &yp, &n, &pc}
		_dispatch(_cast_f32_to_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported (%v -> %v)", x.type, y.type, loc=loc)
	}
}

_rmsnorm_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x      := op.input
	y      := op.output
	v      := op.variant.(ml.Rmsnorm)
	size   := x.shape[x.rank - 1]
	count  := ml.len(x) / size

	xp := data(x).ptr; wp := data(v.weight).ptr; yp := data(y).ptr
	c   := i32(count); s := i32(size); eps := v.eps

	#partial switch x.type {
	case .Bf16:
		fmt.assertf(size % 2 == 0, "rmsnorm bf16 requires even size (got %v)", size, loc=loc)
		_rmsnorm_bf16_pipeline := _compile_pipeline(RMSNORM_BF16_SRC,      "rmsnorm_bf16.cu",  "rmsnorm_bf16")
		rstd_p := data(v.rstd).ptr
		args := [?]rawptr{&xp, &wp, &yp, &rstd_p, &c, &s, &eps}
		_dispatch(_rmsnorm_bf16_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		fmt.assertf(v.weight.type == .F32, "rmsnorm f32 requires F32 weight (got %v)", v.weight.type, loc=loc)
		_rmsnorm_pipeline := _compile_pipeline(RMSNORM_F32_SRC, "rmsnorm.cu", "rmsnorm_f32")
		rstd_p := data(v.rstd).ptr
		args := [?]rawptr{&xp, &wp, &yp, &rstd_p, &c, &s, &eps}
		_dispatch(_rmsnorm_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

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
	args := [?]rawptr{&ap, &bp, &wp, &rp, &yp, &c, &s, &eps}

	#partial switch a.type {
	case .Bf16:
		_add_rmsnorm_bf16_pipeline := _compile_pipeline(ADD_RMSNORM_BF16_SRC, "add_rmsnorm_bf16.cu", "add_rmsnorm_bf16")
		_dispatch(_add_rmsnorm_bf16_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		_add_rmsnorm_f32_pipeline := _compile_pipeline(ADD_RMSNORM_F32_SRC, "add_rmsnorm_f32.cu", "add_rmsnorm_f32")
		_dispatch(_add_rmsnorm_f32_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", a.type, loc=loc)
	}
}

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
	args := [?]rawptr{&xp, &wp, &yp, &tc, &hc, &hs, &eps, &base, &pos_dev, &rpc}

	#partial switch x.type {
	case .Bf16:
		_rmsnorm_rope_bf16_pipeline := _compile_pipeline(RMSNORM_ROPE_BF16_SRC, "rmsnorm_rope_bf16.cu", "rmsnorm_rope_bf16")
		_dispatch(_rmsnorm_rope_bf16_pipeline, u32(token_count * v.head_count), 1, 1, 128, 1, 1, 0, args[:], loc)
	case .F32:
		_rmsnorm_rope_f32_pipeline := _compile_pipeline(RMSNORM_ROPE_F32_SRC, "rmsnorm_rope_f32.cu", "rmsnorm_rope_f32")
		_dispatch(_rmsnorm_rope_f32_pipeline, u32(token_count * v.head_count), 1, 1, 128, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

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

	kv_size      := v.cache.shape[1]
	row_bytes    := uint(kv_size) * 2
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
	args := [?]rawptr{&xp, &wp, &cp, &tc, &hc, &hs, &eps, &base, &pos_dev, &rpc, &cap}

	fmt.assertf(x.type == .Bf16, "unsupported input dtype %v", x.type, loc=loc)
	_rmsnorm_rope_cache_bf16_pipeline := _compile_pipeline(RMSNORM_ROPE_CACHE_BF16_SRC, "rmsnorm_rope_cache_bf16.cu", "rmsnorm_rope_cache_bf16")
	_dispatch(_rmsnorm_rope_cache_bf16_pipeline, u32(token_count * v.head_count), 1, 1, 128, 1, 1, 0, args[:], loc)

	gctx.k_cache_written_this_forward[cp] = true
}

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
	args := [?]rawptr{&xp, &yp, &tc, &hc, &hs, &base, &pos_dev, &rpc}

	#partial switch x.type {
	case .F32:
		_rope_pipeline := _compile_pipeline(ROPE_F32_SRC,          "rope.cu",          "rope_f32")
		_dispatch(_rope_pipeline, _div_up(total_pairs, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		fmt.assertf(head_size % 2 == 0, "rope bf16 requires even head_size (got %v)", head_size, loc=loc)
		_rope_bf16_pipeline := _compile_pipeline(ROPE_BF16_SRC,         "rope_bf16.cu",     "rope_bf16")
		_dispatch(_rope_bf16_pipeline, _div_up(total_pairs, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_linear_q6_k_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q6_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "K must be a multiple of 256, got %v", input_size, loc=loc)
	fmt.assertf(output.type == .Bf16, "linear_q6_k requires Bf16 output (got %v)", output.type, loc=loc)

	if count == 1 {
		_linear_q6_k_forward_mmvq(op, loc)
		return
	}

	gctx := _gctx(loc)
	w_bf := _dequantize_q6_k(gctx, data(weight).ptr, output_size, input_size, loc)
	xp   := data(input).ptr
	yp   := data(output).ptr

	alpha := f32(1.0)
	beta  := f32(0.0)
	cublas.check(cublas.GemmEx(
		gctx.cublas_handle,
		.T,  .N,
		i32(output_size), i32(count), i32(input_size),
		&alpha,
		rawptr(uintptr(w_bf)), .R_16BF, i32(input_size),
		rawptr(uintptr(xp)),   .R_16BF, i32(input_size),
		&beta,
		rawptr(uintptr(yp)),   .R_16BF, i32(output_size),
		._32F,
		.DEFAULT,
	), loc=loc)
}

_linear_q6_k_forward_mmvq :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q6_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]

	_linear_q6_k_mmvq_pipeline := _compile_pipeline(LINEAR_Q6_K_MMVQ_SRC, "linear_q6_k_mmvq.cu", "linear_q6_k_mmvq")

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
	M  := i32(1)
	K  := i32(input_size)
	N  := i32(output_size)

	mmvq_args := [?]rawptr{&q8, &wp, &yp, &M, &K, &N}
	_dispatch(_linear_q6_k_mmvq_pipeline,
		u32(output_size), 1, 1,
		32, 4, 1,
		0, mmvq_args[:], loc)
}

_dequantize_q6_k :: proc(gctx: ^Context, src: cuda.DevicePtr, output_size, input_size: int, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	if cached, ok := gctx.dequant_cache[src]; ok {
		return cached
	}
	src := src
	count := output_size * input_size
	fmt.assertf(count % ml.K_QUANT_BLOCK_SIZE == 0, "count %v not divisible by 256", count, loc=loc)
	num_blocks := count / ml.K_QUANT_BLOCK_SIZE

	dst := _activation_alloc(gctx, u64(count) * 2, loc)

	_dequantize_q6_k_to_bf16_pipeline := _compile_pipeline(DEQUANTIZE_Q6_K_TO_BF16_SRC, "dequantize_q6_k_to_bf16.cu", "dequantize_q6_k_to_bf16")
	total := i32(count)
	args := [?]rawptr{&src, &dst, &total}
	_dispatch(_dequantize_q6_k_to_bf16_pipeline, u32(num_blocks), 1, 1, 256, 1, 1, 0, args[:], loc)

	gctx.dequant_cache[src] = dst
	return dst
}

_linear_q6_k_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	input       := op.input
	weight      := op.variant.(ml.Linear_Q6_K).weight
	output      := op.output
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	dxp := gradient(input).ptr
	if dxp == 0 {
		return
	}

	gctx := _gctx(loc)
	w_bf := _dequantize_q6_k(gctx, data(weight).ptr, output_size, input_size, loc)

	dy_count   := count * output_size
	dy_bf_size := uintptr(dy_count) * 2
	dy_bf_ptr  := _activation_alloc(gctx, u64(dy_bf_size), loc)
	_cast_f32_to_bf16_pipeline := _compile_pipeline(CAST_F32_TO_BF16_SRC, "cast_f32_to_bf16.cu", "cast_f32_to_bf16")
	dy_p := gradient(output).ptr
	bf_p := dy_bf_ptr
	nc   := i32(dy_count)
	pc   := i32((dy_count + 1) / 2)
	cast_args := [?]rawptr{&dy_p, &bf_p, &nc, &pc}
	_dispatch(_cast_f32_to_bf16_pipeline, _div_up((dy_count + 1) / 2, 256), 1, 1, 256, 1, 1, 0, cast_args[:], loc)

	alpha := f32(1.0)
	beta  := f32(1.0)
	cublas.check(cublas.GemmEx(
		gctx.cublas_handle,
		.N, .N,
		i32(input_size), i32(count), i32(output_size),
		&alpha,
		rawptr(uintptr(w_bf)),     .R_16BF, i32(input_size),
		rawptr(uintptr(dy_bf_ptr)), .R_16BF, i32(output_size),
		&beta,
		rawptr(uintptr(dxp)),      .R_32F,  i32(input_size),
		._32F,
		.DEFAULT,
	), loc=loc)
}

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
	fmt.assertf(head_size <= 512, "attention caps head_size at 512 (got %v)", head_size, loc=loc)
	fmt.assertf(v.window == 0 || v.causal, "attention window > 0 requires causal=true", loc=loc)

	qp := data(q).ptr; kp := data(k).ptr; vp := data(val).ptr; op_ptr := data(o).ptr
	n_q_heads  := i32(v.n_q_heads)
	n_kv_heads := i32(v.n_kv_heads)
	hs := i32(head_size); tc := i32(token_count); qs := i32(q_size); kvs := i32(kv_size)
	causal := i32(v.causal ? 1 : 0); window := i32(v.window)

	#partial switch q.type {
	case .Bf16:
		fmt.assertf(head_size % 2 == 0, "bf16 attention requires even head_size (got %v)", head_size, loc=loc)
		// Training mode: use the materialising train kernel so backward has
		// the softmax matrix. Must match _alloc_scratch's kernel choice.
		gctx := _gctx(loc)
		training := gctx.training
		if training {
			fmt.assertf(token_count <= 2048, "attention_train_bf16 caps token_count at 2048 (got %v)", token_count, loc=loc)
			_attention_train_bf16_pipeline := _compile_pipeline(ATTENTION_TRAIN_BF16_SRC, "attention_train_bf16.cu", "attention_train_bf16")
			sm_ptr := data(v.softmax_outputs).ptr
			args := [?]rawptr{
				&qp, &kp, &vp, &op_ptr, &sm_ptr,
				&n_q_heads, &n_kv_heads, &hs, &tc, &qs, &kvs, &causal, &window,
			}
			_dispatch(_attention_train_bf16_pipeline, u32(v.n_q_heads), u32(token_count), 1, 256, 1, 1, 0, args[:], loc)
		} else {
			_attention_bf16_pipeline := _compile_pipeline(ATTENTION_BF16_SRC, "attention_bf16.cu", "attention_bf16")
			lse_ptr := data(v.lse).ptr
			args := [?]rawptr{
				&qp, &kp, &vp, &op_ptr, &lse_ptr,
				&n_q_heads, &n_kv_heads, &hs, &tc, &qs, &kvs, &causal, &window,
			}
			_dispatch(_attention_bf16_pipeline, u32(v.n_q_heads), u32(token_count), 1, 64, 1, 1, 0, args[:], loc)
		}

	case .F32:
		fmt.assertf(token_count <= 2048, "attention_train_f32 caps token_count at 2048 (got %v)", token_count, loc=loc)
		_attention_train_f32_pipeline := _compile_pipeline(ATTENTION_TRAIN_F32_SRC, "attention_train_f32.cu", "attention_train_f32")
		sm_ptr := data(v.softmax_outputs).ptr
		args := [?]rawptr{
			&qp, &kp, &vp, &op_ptr, &sm_ptr,
			&n_q_heads, &n_kv_heads, &hs, &tc, &qs, &kvs, &causal, &window,
		}
		_dispatch(_attention_train_f32_pipeline, u32(v.n_q_heads), u32(token_count), 1, 256, 1, 1, 0, args[:], loc)

	case:
		fmt.panicf("unsupported dtype %v", q.type, loc=loc)
	}
}

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
	fmt.assertf(head_size <= 512, "attention_with_cache caps head_size at 512 (got %v)", head_size, loc=loc)
	fmt.assertf(q.type == .Bf16, "attention_with_cache requires Bf16 Q (got %v)", q.type, loc=loc)
	fmt.assertf(k_cache.type == .Bf16, "attention_with_cache requires Bf16 K cache (got %v)", k_cache.type, loc=loc)
	fmt.assertf(head_size % 2 == 0, "attention_with_cache requires even head_size (got %v)", head_size, loc=loc)

	gctx := _gctx(loc)

	capacity := k_cache.shape[0]

	fmt.assertf(token_count <= capacity, "q_token_count (%v) cannot exceed cache capacity (%v); chunk multi-token prefill if needed", token_count, capacity, loc=loc)

	k_cache_ptr := data(k_cache).ptr
	v_cache_ptr := data(v_cache).ptr
	k_already_written := k_cache_ptr in gctx.k_cache_written_this_forward
	v_already_written := v_cache_ptr in gctx.v_cache_written_this_forward

	row_bytes  := uint(kv_size) * 2
	shift_amount := 0
	if v.window > 0 {
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
			k_args := [?]rawptr{&k_src, &k_dst, &pos_dev, &nr, &kvs, &cap}
			_dispatch_cache_write(k.type, grid, k_args[:], loc)
		}
		if !v_already_written {
			v_src := data(val).ptr
			v_dst := v_cache_ptr
			v_args := [?]rawptr{&v_src, &v_dst, &pos_dev, &nr, &kvs, &cap}
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

	switch head_size {
	case 256:
		opts := [?]cstring{"-DD_HEAD=256"}
		_attention_cache_vec_bf16_d256_pipeline := _compile_pipeline(ATTENTION_CACHE_VEC_BF16_SRC, "attention_cache_vec_bf16_d256.cu", "attention_cache_vec_bf16", opts[:])
		_dispatch(_attention_cache_vec_bf16_d256_pipeline, u32(v.n_q_heads), u32(token_count), 1, 32, 4, 1, 0, args[:], loc)
	case 512:
		opts := [?]cstring{"-DD_HEAD=512"}
		_attention_cache_vec_bf16_d512_pipeline := _compile_pipeline(ATTENTION_CACHE_VEC_BF16_SRC, "attention_cache_vec_bf16_d512.cu", "attention_cache_vec_bf16", opts[:])
		_dispatch(_attention_cache_vec_bf16_d512_pipeline, u32(v.n_q_heads), u32(token_count), 1, 32, 4, 1, 0, args[:], loc)
	case:
		_attention_cache_bf16_pipeline := _compile_pipeline(ATTENTION_CACHE_BF16_SRC, "attention_cache_bf16.cu", "attention_cache_bf16")
		_dispatch(_attention_cache_bf16_pipeline, u32(v.n_q_heads), u32(token_count), 1, 64, 1, 1, 0, args[:], loc)
	}
}

_upload_indices :: proc(gctx: ^Context, indices: []int, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	bytes := uint(builtin.len(indices) * size_of(u32))
	dev_ptr := _activation_alloc(gctx, u64(bytes), loc)

	host := ([^]u32)(_pinned_staging_take(gctx, u64(bytes), loc))
	for v, i in indices {
		host[i] = u32(v)
	}
	cuda.check(cuda.MemcpyHtoDAsync(dev_ptr, host, bytes, gctx.stream), loc=loc)
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
		fmt.assertf(size % 2 == 0, "bf16 select requires even row size (got %v)", size, loc=loc)
		_select_bf16_pipeline := _compile_pipeline(SELECT_BF16_SRC, "select_bf16.cu", "select_bf16")
		pair_count := size / 2
		args := [?]rawptr{&xp, &idx_ptr, &yp, &n_idx, &s}
		_dispatch(_select_bf16_pipeline, _div_up(pair_count, 256), u32(min(builtin.len(indices), MAX_GRID_DIM_YZ)), 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		_select_f32_pipeline := _compile_pipeline(SELECT_F32_SRC, "select.cu", "select_f32")
		args := [?]rawptr{&xp, &idx_ptr, &yp, &n_idx, &s}
		_dispatch(_select_f32_pipeline, _div_up(size, 256), u32(min(builtin.len(indices), MAX_GRID_DIM_YZ)), 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

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
		_slice_trailing_bf16_pipeline := _compile_pipeline(SLICE_TRAILING_BF16_SRC, "slice_trailing_bf16.cu", "slice_trailing_bf16")
		pair_count := (leading * new_trailing + 1) / 2
		pc := i32(pair_count)
		args := [?]rawptr{&xp, &yp, &ld, &tr, &nt, &st, &pc}
		_dispatch(_slice_trailing_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .F32:
		_slice_trailing_f32_pipeline := _compile_pipeline(SLICE_TRAILING_F32_SRC, "slice_trailing.cu", "slice_trailing_f32")
		args := [?]rawptr{&xp, &yp, &ld, &tr, &nt, &st}
		_dispatch(_slice_trailing_f32_pipeline, _div_up(leading * new_trailing, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_silu_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	xp := data(x).ptr; yp := data(y).ptr
	n  := i32(ml.len(x))

	#partial switch x.type {
	case .F32:
		_silu_f32_pipeline := _compile_pipeline(SILU_F32_SRC, "silu_f32.cu", "silu_f32")
		args := [?]rawptr{&xp, &yp, &n}
		_dispatch(_silu_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_silu_bf16_pipeline := _compile_pipeline(SILU_BF16_SRC, "silu_bf16.cu", "silu_bf16")
		pair_count := (ml.len(x) + 1) / 2
		pc := i32(pair_count)
		args := [?]rawptr{&xp, &yp, &n, &pc}
		_dispatch(_silu_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_gelu_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	xp := data(x).ptr; yp := data(y).ptr
	n  := i32(ml.len(x))

	#partial switch x.type {
	case .F32:
		_gelu_f32_pipeline := _compile_pipeline(GELU_F32_SRC, "gelu_f32.cu", "gelu_f32")
		args := [?]rawptr{&xp, &yp, &n}
		_dispatch(_gelu_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_gelu_bf16_pipeline := _compile_pipeline(GELU_BF16_SRC, "gelu_bf16.cu", "gelu_bf16")
		pair_count := (ml.len(x) + 1) / 2
		pc := i32(pair_count)
		args := [?]rawptr{&xp, &yp, &n, &pc}
		_dispatch(_gelu_bf16_pipeline, _div_up(pair_count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_gelu_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	if gradient(x).ptr == 0 { return }

	xp  := data(x).ptr
	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	n   := i32(ml.len(x))
	args := [?]rawptr{&xp, &dyp, &dxp, &n}

	#partial switch x.type {
	case .F32:
		_gelu_back_f32_pipeline := _compile_pipeline(GELU_BACK_F32_SRC, "gelu_back_f32.cu", "gelu_back_f32")
		_dispatch(_gelu_back_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_gelu_back_bf16_pipeline := _compile_pipeline(GELU_BACK_BF16_SRC, "gelu_back_bf16.cu", "gelu_back_bf16")
		_dispatch(_gelu_back_bf16_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_tanh_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	if gradient(x).ptr == 0 { return }

	yp  := data(y).ptr
	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	n   := i32(ml.len(x))
	args := [?]rawptr{&yp, &dyp, &dxp, &n}

	#partial switch x.type {
	case .F32:
		_tanh_back_f32_pipeline := _compile_pipeline(TANH_BACK_F32_SRC, "tanh_back_f32.cu", "tanh_back_f32")
		_dispatch(_tanh_back_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_tanh_back_bf16_pipeline := _compile_pipeline(TANH_BACK_BF16_SRC, "tanh_back_bf16.cu", "tanh_back_bf16")
		_dispatch(_tanh_back_bf16_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_slice_trailing_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Slice_Trailing)

	trailing     := x.shape[x.rank - 1]
	new_trailing := y.shape[y.rank - 1]
	leading      := ml.len(x) / trailing
	total        := leading * new_trailing

	if gradient(x).ptr == 0 { return }

	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	ld := i32(leading); tr := i32(trailing); nt := i32(new_trailing); st := i32(v.start)
	args := [?]rawptr{&dyp, &dxp, &ld, &tr, &nt, &st}

	// F32 grads everywhere now -> bf16 and f32 paths are byte-identical.
	#partial switch x.type {
	case .F32:
		_slice_trailing_back_f32_pipeline := _compile_pipeline(SLICE_TRAILING_BACK_F32_SRC, "slice_trailing_back_f32.cu", "slice_trailing_back_f32")
		_dispatch(_slice_trailing_back_f32_pipeline, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_slice_trailing_back_bf16_pipeline := _compile_pipeline(SLICE_TRAILING_BACK_BF16_SRC, "slice_trailing_back_bf16.cu", "slice_trailing_back_bf16")
		_dispatch(_slice_trailing_back_bf16_pipeline, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_slice_leading_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Slice_Leading)

	leading  := x.shape[0]
	row_size := ml.len(x) / leading
	elem     := ml.data_type_size(x.type)
	bytes    := uint(y.shape[0] * row_size * elem)
	src_off  := uint(v.start * row_size * elem)

	gctx := _gctx(loc)
	cuda.check(cuda.MemcpyDtoDAsync(data(y).ptr, data(x).ptr + cuda.DevicePtr(src_off), bytes, gctx.stream), loc=loc)
}

_slice_leading_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Slice_Leading)

	if gradient(x).ptr == 0 { return }

	leading  := x.shape[0]
	row_size := ml.len(x) / leading
	count    := y.shape[0] * row_size
	offset   := v.start * row_size

	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	cnt := i32(count); off := i32(offset)
	args := [?]rawptr{&dyp, &dxp, &cnt, &off}

	_slice_leading_back_f32_pipeline := _compile_pipeline(SLICE_LEADING_BACK_F32_SRC, "slice_leading_back_f32.cu", "slice_leading_back_f32")
	_dispatch(_slice_leading_back_f32_pipeline, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_silu_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output

	if gradient(x).ptr == 0 { return }

	xp  := data(x).ptr
	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	n   := i32(ml.len(x))
	args := [?]rawptr{&xp, &dyp, &dxp, &n}

	#partial switch x.type {
	case .F32:
		_silu_back_f32_pipeline := _compile_pipeline(SILU_BACK_F32_SRC, "silu_back_f32.cu", "silu_back_f32")
		_dispatch(_silu_back_f32_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_silu_back_bf16_pipeline := _compile_pipeline(SILU_BACK_BF16_SRC, "silu_back_bf16.cu", "silu_back_bf16")
		_dispatch(_silu_back_bf16_pipeline, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_cross_entropy_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Cross_Entropy)
	fmt.assertf(x.type == .F32, "cross_entropy requires F32 (got %v)", x.type, loc=loc)

	_cross_entropy_f32_pipeline := _compile_pipeline(CROSS_ENTROPY_F32_SRC, "cross_entropy_f32.cu", "cross_entropy_f32")

	gctx := _gctx(loc)
	idx_ptr := _upload_indices(gctx, v.targets, loc)

	xp := data(x).ptr
	pp := data(v.probabilities).ptr
	yp := data(y).ptr
	class_size := i32(x.shape[x.rank - 1])
	sample_count := builtin.len(v.targets)

	args := [?]rawptr{&xp, &idx_ptr, &pp, &yp, &class_size}
	_dispatch(_cross_entropy_f32_pipeline, u32(sample_count), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_cross_entropy_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Cross_Entropy)

	if gradient(x).ptr == 0 { return }

	_cross_entropy_back_f32_pipeline := _compile_pipeline(CROSS_ENTROPY_BACK_F32_SRC, "cross_entropy_back_f32.cu", "cross_entropy_back_f32")

	gctx := _gctx(loc)
	idx_ptr := _upload_indices(gctx, v.targets, loc)

	pp  := data(v.probabilities).ptr
	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	class_size   := i32(x.shape[x.rank - 1])
	sample_count := i32(builtin.len(v.targets))
	total        := int(class_size) * int(sample_count)

	args := [?]rawptr{&pp, &idx_ptr, &dyp, &dxp, &sample_count, &class_size}
	_dispatch(_cross_entropy_back_f32_pipeline, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_mul_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a := op.input
	b := op.variant.(ml.Mul).b
	c := op.output

	ap  := data(a).ptr
	bp  := data(b).ptr
	dyp := gradient(c).ptr
	dap := gradient(a).ptr
	dbp := gradient(b).ptr
	n_a := i32(ml.len(a))
	n_b := i32(ml.len(b))

	// Skip the b-side accumulation when b has no gradient buffer (frozen
	// constant scalar). Saves the slow bf16 CAS path in the typical broadcast
	// case where many a-elements target one b cell.
	have_a_grad := dap != 0
	have_b_grad := dbp != 0

	args_a := [?]rawptr{&bp, &dyp, &dap, &n_a, &n_b}
	args_b := [?]rawptr{&ap, &dyp, &dbp, &n_a, &n_b}

	#partial switch a.type {
	case .F32:
		if have_a_grad {
			_mul_back_a_f32_pipeline := _compile_pipeline(MUL_BACK_A_F32_SRC, "mul_back_a_f32.cu", "mul_back_a_f32")
			_dispatch(_mul_back_a_f32_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args_a[:], loc)
		}
		if have_b_grad {
			_mul_back_b_f32_pipeline := _compile_pipeline(MUL_BACK_B_F32_SRC, "mul_back_b_f32.cu", "mul_back_b_f32")
			_dispatch(_mul_back_b_f32_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args_b[:], loc)
		}
	case .Bf16:
		if have_a_grad {
			_mul_back_a_bf16_pipeline := _compile_pipeline(MUL_BACK_A_BF16_SRC, "mul_back_a_bf16.cu", "mul_back_a_bf16")
			_dispatch(_mul_back_a_bf16_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args_a[:], loc)
		}
		if have_b_grad {
			_mul_back_b_bf16_pipeline := _compile_pipeline(MUL_BACK_B_BF16_SRC, "mul_back_b_bf16.cu", "mul_back_b_bf16")
			_dispatch(_mul_back_b_bf16_pipeline, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args_b[:], loc)
		}
	case:
		fmt.panicf("unsupported dtype %v", a.type, loc=loc)
	}
}

_select_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	w := op.input
	y := op.output
	indices := op.variant.(ml.Select).indices

	dwp := gradient(w).ptr
	if dwp == 0 {
		// Frozen embedding (e.g. QLoRA): no gradient to accumulate.
		return
	}

	gctx := _gctx(loc)
	idx_ptr := _upload_indices(gctx, indices, loc)

	dyp := gradient(y).ptr

	row_size := ml.len(y) / builtin.len(indices)
	n_idx    := i32(builtin.len(indices))
	rs       := i32(row_size)
	total    := builtin.len(indices) * row_size

	args := [?]rawptr{&dyp, &idx_ptr, &dwp, &n_idx, &rs}

	#partial switch w.type {
	case .F32:
		_select_back_f32_pipeline := _compile_pipeline(SELECT_BACK_F32_SRC, "select_back_f32.cu", "select_back_f32")
		_dispatch(_select_back_f32_pipeline, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		_select_back_bf16_pipeline := _compile_pipeline(SELECT_BACK_BF16_SRC, "select_back_bf16.cu", "select_back_bf16")
		_dispatch(_select_back_bf16_pipeline, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", w.type, loc=loc)
	}
}

_rmsnorm_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Rmsnorm)
	fmt.assertf(v.weight.type == x.type, "rmsnorm backward requires weight dtype to match input (x=%v, w=%v)", x.type, v.weight.type, loc=loc)

	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size

	xp     := data(x).ptr
	wp     := data(v.weight).ptr
	rstd_p := data(v.rstd).ptr
	dyp    := gradient(y).ptr
	dxp    := gradient(x).ptr
	dwp    := gradient(v.weight).ptr
	c      := i32(count); s := i32(size)

	if dxp == 0 && dwp == 0 { return }

	args := [?]rawptr{&xp, &wp, &rstd_p, &dyp, &dxp, &dwp, &c, &s}

	#partial switch x.type {
	case .F32:
		_rmsnorm_back_f32_pipeline := _compile_pipeline(RMSNORM_BACK_F32_SRC, "rmsnorm_back_f32.cu", "rmsnorm_back_f32")
		_dispatch(_rmsnorm_back_f32_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		fmt.assertf(size % 2 == 0, "rmsnorm backward bf16 requires even trailing dim (got %v)", size, loc=loc)
		_rmsnorm_back_bf16_pipeline := _compile_pipeline(RMSNORM_BACK_BF16_SRC, "rmsnorm_back_bf16.cu", "rmsnorm_back_bf16")
		_dispatch(_rmsnorm_back_bf16_pipeline, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_rope_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Rope)

	if gradient(x).ptr == 0 { return }

	gctx := _gctx(loc)
	_emit_position_upload(gctx, v.position_offset, loc)

	token_count := x.shape[0]
	head_size   := x.shape[x.rank - 1] / v.head_count
	total_pairs := token_count * v.head_count * (head_size / 2)

	dyp := gradient(y).ptr
	dxp := gradient(x).ptr
	tc := i32(token_count); hc := i32(v.head_count); hs := i32(head_size)
	base := v.base; pos_dev := gctx.position_dev; rpc := i32(v.rotate_pair_count)
	args := [?]rawptr{&dyp, &dxp, &tc, &hc, &hs, &base, &pos_dev, &rpc}

	#partial switch x.type {
	case .F32:
		_rope_back_f32_pipeline := _compile_pipeline(ROPE_BACK_F32_SRC, "rope_back_f32.cu", "rope_back_f32")
		_dispatch(_rope_back_f32_pipeline, _div_up(total_pairs, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case .Bf16:
		fmt.assertf(head_size % 2 == 0, "rope backward bf16 requires even head_size (got %v)", head_size, loc=loc)
		_rope_back_bf16_pipeline := _compile_pipeline(ROPE_BACK_BF16_SRC, "rope_back_bf16.cu", "rope_back_bf16")
		_dispatch(_rope_back_bf16_pipeline, _div_up(total_pairs, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	case:
		fmt.panicf("unsupported dtype %v", x.type, loc=loc)
	}
}

_attention_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	q   := op.input
	o   := op.output
	v   := op.variant.(ml.Attention)
	k   := v.key
	val := v.value

	token_count := q.shape[0]
	q_size      := q.shape[1]
	kv_size     := k.shape[1]
	head_size   := q_size / v.n_q_heads
	gqa         := v.n_q_heads / v.n_kv_heads

	qp  := data(q).ptr
	kp  := data(k).ptr
	vp  := data(val).ptr
	smp := data(v.softmax_outputs).ptr
	dyp := gradient(o).ptr
	dqp := gradient(q).ptr
	dkp := gradient(k).ptr
	dvp := gradient(val).ptr

	if dqp == 0 && dkp == 0 && dvp == 0 { return }
	fmt.assertf(dqp != 0 && dkp != 0 && dvp != 0, "attention backward requires dq, dk, and dv gradient buffers (partial-null unsupported)", loc=loc)

	n_q_heads  := i32(v.n_q_heads)
	n_kv_heads := i32(v.n_kv_heads)
	hs := i32(head_size); tc := i32(token_count); qs := i32(q_size); kvs := i32(kv_size)
	causal := i32(v.causal ? 1 : 0); window := i32(v.window)

	args := [?]rawptr{
		&qp, &kp, &vp, &smp, &dyp, &dqp, &dkp, &dvp,
		&n_q_heads, &n_kv_heads, &hs, &tc, &qs, &kvs, &causal, &window,
	}

	#partial switch q.type {
	case .F32:
		_attention_train_back_f32_pipeline := _compile_pipeline(ATTENTION_TRAIN_BACK_F32_SRC, "attention_train_back_f32.cu", "attention_train_back_f32")
		_dispatch(_attention_train_back_f32_pipeline,
			u32(v.n_kv_heads), u32(gqa), u32(token_count),
			256, 1, 1,
			0, args[:], loc,
		)
	case .Bf16:
		fmt.assertf(token_count <= 2048, "attention_train_back_bf16 caps token_count at 2048 (got %v)", token_count, loc=loc)
		fmt.assertf(head_size % 2 == 0, "attention backward bf16 requires even head_size (got %v)", head_size, loc=loc)
		_attention_train_back_bf16_pipeline := _compile_pipeline(ATTENTION_TRAIN_BACK_BF16_SRC, "attention_train_back_bf16.cu", "attention_train_back_bf16")
		_dispatch(_attention_train_back_bf16_pipeline,
			u32(v.n_kv_heads), u32(gqa), u32(token_count),
			256, 1, 1,
			0, args[:], loc,
		)
	case:
		fmt.panicf("unsupported dtype %v", q.type, loc=loc)
	}
}

LAYERNORM_EPS :: f32(1e-5)

_unary_ew_forward :: proc(op: ml.Operation, source_name, entry: cstring, opts: []cstring, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	fmt.assertf(x.type == .F32, "%s requires F32 (got %v)", entry, x.type, loc=loc)
	p := _compile_pipeline(ELEMENTWISE_UNARY_SRC, source_name, entry, opts)
	xp := data(x).ptr; yp := data(y).ptr; n := i32(ml.len(x))
	args := [?]rawptr{&xp, &yp, &n}
	_dispatch(p, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_unary_ew_backward :: proc(op: ml.Operation, source_name, entry: cstring, opts: []cstring, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	if gradient(x).ptr == 0 { return }
	fmt.assertf(x.type == .F32, "%s requires F32 (got %v)", entry, x.type, loc=loc)
	p := _compile_pipeline(ELEMENTWISE_UNARY_BACK_SRC, source_name, entry, opts)
	xp := data(x).ptr; yp := data(y).ptr; dyp := gradient(y).ptr; dxp := gradient(x).ptr; n := i32(ml.len(x))
	args := [?]rawptr{&xp, &yp, &dyp, &dxp, &n}
	_dispatch(p, _div_up(ml.len(x), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_relu_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=relu_f32", "-DOP_EXPR=(v<0.0f?0.0f:v)"}
	_unary_ew_forward(op, "relu.cu", "relu_f32", opts[:], loc)
}
_relu_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=relu_back_f32", "-DOP_DERIV=(xv>0.0f?1.0f:0.0f)"}
	_unary_ew_backward(op, "relu_back.cu", "relu_back_f32", opts[:], loc)
}

_sigmoid_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=sigmoid_f32", "-DOP_EXPR=(1.0f/(1.0f+expf(-v)))"}
	_unary_ew_forward(op, "sigmoid.cu", "sigmoid_f32", opts[:], loc)
}
_sigmoid_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=sigmoid_back_f32", "-DOP_DERIV=((1.0f/(1.0f+expf(-xv)))*(1.0f-(1.0f/(1.0f+expf(-xv)))))"}
	_unary_ew_backward(op, "sigmoid_back.cu", "sigmoid_back_f32", opts[:], loc)
}

_sqrt_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=sqrt_f32", "-DOP_EXPR=sqrtf(v)"}
	_unary_ew_forward(op, "sqrt.cu", "sqrt_f32", opts[:], loc)
}
_sqrt_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=sqrt_back_f32", "-DOP_DERIV=(yv>0.0f?0.5f/yv:0.0f)"}
	_unary_ew_backward(op, "sqrt_back.cu", "sqrt_back_f32", opts[:], loc)
}

_binary_ew_forward :: proc(op: ml.Operation, b: ml.Tensor, source_name, entry: cstring, opts: []cstring, loc: runtime.Source_Code_Location) {
	a := op.input
	c := op.output
	fmt.assertf(a.type == .F32, "%s requires F32 (got %v)", entry, a.type, loc=loc)
	p := _compile_pipeline(ELEMENTWISE_BINARY_SRC, source_name, entry, opts)
	ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
	n := i32(ml.len(a)); n_b := i32(ml.len(b))
	args := [?]rawptr{&ap, &bp, &cp, &n, &n_b}
	_dispatch(p, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_binary_ew_backward :: proc(
	op: ml.Operation, b: ml.Tensor,
	back_a_name, back_a_entry: cstring, back_a_opts: []cstring,
	back_b_name, back_b_entry: cstring, back_b_opts: []cstring,
	loc: runtime.Source_Code_Location,
) {
	a := op.input
	c := op.output
	fmt.assertf(a.type == .F32, "%s requires F32 (got %v)", back_a_entry, a.type, loc=loc)
	ap := data(a).ptr; bp := data(b).ptr
	dyp := gradient(c).ptr; dap := gradient(a).ptr; dbp := gradient(b).ptr
	n := i32(ml.len(a)); n_b := i32(ml.len(b)); stride := i32(ml.len(a) / ml.len(b))

	if dap != 0 {
		p := _compile_pipeline(ELEMENTWISE_BINARY_BACK_A_SRC, back_a_name, back_a_entry, back_a_opts)
		args := [?]rawptr{&ap, &bp, &dyp, &dap, &n, &n_b}
		_dispatch(p, _div_up(ml.len(a), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	}
	if dbp != 0 {
		p := _compile_pipeline(ELEMENTWISE_BINARY_BACK_B_SRC, back_b_name, back_b_entry, back_b_opts)
		args := [?]rawptr{&ap, &bp, &dyp, &dbp, &n_b, &stride}
		_dispatch(p, _div_up(ml.len(b), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
	}
}

_sub_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=sub_f32", "-DOP_EXPR=(av-bv)"}
	_binary_ew_forward(op, op.variant.(ml.Sub).b, "sub.cu", "sub_f32", opts[:], loc)
}
_sub_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a_opts := [?]cstring{"-DOP_NAME=sub_back_a_f32", "-DDA_EXPR=1.0f"}
	b_opts := [?]cstring{"-DOP_NAME=sub_back_b_f32", "-DDB_EXPR=-1.0f"}
	_binary_ew_backward(op, op.variant.(ml.Sub).b, "sub_back_a.cu", "sub_back_a_f32", a_opts[:], "sub_back_b.cu", "sub_back_b_f32", b_opts[:], loc)
}

_div_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=div_f32", "-DOP_EXPR=(av/bv)"}
	_binary_ew_forward(op, op.variant.(ml.Div).b, "div.cu", "div_f32", opts[:], loc)
}
_div_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a_opts := [?]cstring{"-DOP_NAME=div_back_a_f32", "-DDA_EXPR=(1.0f/bv)"}
	b_opts := [?]cstring{"-DOP_NAME=div_back_b_f32", "-DDB_EXPR=(-av/(bv*bv))"}
	_binary_ew_backward(op, op.variant.(ml.Div).b, "div_back_a.cu", "div_back_a_f32", a_opts[:], "div_back_b.cu", "div_back_b_f32", b_opts[:], loc)
}

_max_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=max_f32", "-DOP_EXPR=(av>bv?av:bv)"}
	_binary_ew_forward(op, op.variant.(ml.Max).b, "max.cu", "max_f32", opts[:], loc)
}
_max_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a_opts := [?]cstring{"-DOP_NAME=max_back_a_f32", "-DDA_EXPR=(av>=bv?1.0f:0.0f)"}
	b_opts := [?]cstring{"-DOP_NAME=max_back_b_f32", "-DDB_EXPR=(av>=bv?0.0f:1.0f)"}
	_binary_ew_backward(op, op.variant.(ml.Max).b, "max_back_a.cu", "max_back_a_f32", a_opts[:], "max_back_b.cu", "max_back_b_f32", b_opts[:], loc)
}

_min_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	opts := [?]cstring{"-DOP_NAME=min_f32", "-DOP_EXPR=(av<bv?av:bv)"}
	_binary_ew_forward(op, op.variant.(ml.Min).b, "min.cu", "min_f32", opts[:], loc)
}
_min_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a_opts := [?]cstring{"-DOP_NAME=min_back_a_f32", "-DDA_EXPR=(av<=bv?1.0f:0.0f)"}
	b_opts := [?]cstring{"-DOP_NAME=min_back_b_f32", "-DDB_EXPR=(av<=bv?0.0f:1.0f)"}
	_binary_ew_backward(op, op.variant.(ml.Min).b, "min_back_a.cu", "min_back_a_f32", a_opts[:], "min_back_b.cu", "min_back_b_f32", b_opts[:], loc)
}

_mean_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	fmt.assertf(x.type == .F32, "mean requires F32 (got %v)", x.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size
	p := _compile_pipeline(MEAN_F32_SRC, "mean.cu", "mean_f32")
	xp := data(x).ptr; yp := data(y).ptr; c := i32(count); s := i32(size)
	args := [?]rawptr{&xp, &yp, &c, &s}
	_dispatch(p, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_mean_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	if gradient(x).ptr == 0 { return }
	fmt.assertf(x.type == .F32, "mean requires F32 (got %v)", x.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size; total := count * size
	p := _compile_pipeline(MEAN_BACK_F32_SRC, "mean_back_f32.cu", "mean_back_f32")
	dyp := gradient(y).ptr; dxp := gradient(x).ptr; c := i32(count); s := i32(size)
	args := [?]rawptr{&dyp, &dxp, &c, &s}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_sum_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	fmt.assertf(x.type == .F32, "sum requires F32 (got %v)", x.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size
	p := _compile_pipeline(SUM_F32_SRC, "sum.cu", "sum_f32")
	xp := data(x).ptr; yp := data(y).ptr; c := i32(count); s := i32(size)
	args := [?]rawptr{&xp, &yp, &c, &s}
	_dispatch(p, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_sum_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	if gradient(x).ptr == 0 { return }
	fmt.assertf(x.type == .F32, "sum requires F32 (got %v)", x.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size; total := count * size
	p := _compile_pipeline(SUM_BACK_F32_SRC, "sum_back_f32.cu", "sum_back_f32")
	dyp := gradient(y).ptr; dxp := gradient(x).ptr; c := i32(count); s := i32(size)
	args := [?]rawptr{&dyp, &dxp, &c, &s}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_max_reduce_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	fmt.assertf(x.type == .F32, "max_reduce requires F32 (got %v)", x.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size
	p := _compile_pipeline(MAX_REDUCE_F32_SRC, "max_reduce.cu", "max_reduce_f32")
	xp := data(x).ptr; yp := data(y).ptr; c := i32(count); s := i32(size)
	args := [?]rawptr{&xp, &yp, &c, &s}
	_dispatch(p, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_max_reduce_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	if gradient(x).ptr == 0 { return }
	fmt.assertf(x.type == .F32, "max_reduce requires F32 (got %v)", x.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size
	p := _compile_pipeline(MAX_REDUCE_BACK_F32_SRC, "max_reduce_back_f32.cu", "max_reduce_back_f32")
	xp := data(x).ptr; dyp := gradient(y).ptr; dxp := gradient(x).ptr; c := i32(count); s := i32(size)
	args := [?]rawptr{&xp, &dyp, &dxp, &c, &s}
	_dispatch(p, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_im2col_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Im2col)
	fmt.assertf(x.type == .F32, "im2col requires F32 (got %v)", x.type, loc=loc)

	n := i32(x.shape[0]); h := i32(x.shape[1]); w := i32(x.shape[2]); c := i32(x.shape[3])
	kh := i32(v.kernel_h); kw := i32(v.kernel_w)
	sh := i32(v.stride_h); sw := i32(v.stride_w)
	ph := i32(v.pad_h); pw := i32(v.pad_w)
	oh := i32(v.out_h); ow := i32(v.out_w)
	total := int(n) * int(oh) * int(ow) * int(kh) * int(kw) * int(c)

	xp := data(x).ptr; yp := data(y).ptr
	args := [?]rawptr{&xp, &yp, &n, &h, &w, &c, &kh, &kw, &sh, &sw, &ph, &pw, &oh, &ow}
	p := _compile_pipeline(IM2COL_F32_SRC, "im2col_f32.cu", "im2col_f32")
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_im2col_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Im2col)
	if gradient(x).ptr == 0 { return }

	n := i32(x.shape[0]); h := i32(x.shape[1]); w := i32(x.shape[2]); c := i32(x.shape[3])
	kh := i32(v.kernel_h); kw := i32(v.kernel_w)
	sh := i32(v.stride_h); sw := i32(v.stride_w)
	ph := i32(v.pad_h); pw := i32(v.pad_w)
	oh := i32(v.out_h); ow := i32(v.out_w)
	total := int(n) * int(oh) * int(ow) * int(kh) * int(kw) * int(c)

	dyp := gradient(y).ptr; dxp := gradient(x).ptr
	args := [?]rawptr{&dyp, &dxp, &n, &h, &w, &c, &kh, &kw, &sh, &sw, &ph, &pw, &oh, &ow}
	p := _compile_pipeline(IM2COL_BACK_F32_SRC, "im2col_back_f32.cu", "im2col_back_f32")
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_max_pool2d_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Max_Pool2d)
	fmt.assertf(x.type == .F32, "max_pool2d requires F32 (got %v)", x.type, loc=loc)

	n := i32(x.shape[0]); h := i32(x.shape[1]); w := i32(x.shape[2]); c := i32(x.shape[3])
	kh := i32(v.kernel_h); kw := i32(v.kernel_w)
	sh := i32(v.stride_h); sw := i32(v.stride_w)
	oh := i32(y.shape[1]); ow := i32(y.shape[2])
	total := int(n) * int(oh) * int(ow) * int(c)

	xp := data(x).ptr; yp := data(y).ptr
	args := [?]rawptr{&xp, &yp, &n, &h, &w, &c, &kh, &kw, &sh, &sw, &oh, &ow}
	p := _compile_pipeline(MAX_POOL2D_F32_SRC, "max_pool2d_f32.cu", "max_pool2d_f32")
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_max_pool2d_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Max_Pool2d)
	if gradient(x).ptr == 0 { return }

	n := i32(x.shape[0]); h := i32(x.shape[1]); w := i32(x.shape[2]); c := i32(x.shape[3])
	kh := i32(v.kernel_h); kw := i32(v.kernel_w)
	sh := i32(v.stride_h); sw := i32(v.stride_w)
	oh := i32(y.shape[1]); ow := i32(y.shape[2])
	total := int(n) * int(oh) * int(ow) * int(c)

	xp := data(x).ptr; dyp := gradient(y).ptr; dxp := gradient(x).ptr
	args := [?]rawptr{&xp, &dyp, &dxp, &n, &h, &w, &c, &kh, &kw, &sh, &sw, &oh, &ow}
	p := _compile_pipeline(MAX_POOL2D_BACK_F32_SRC, "max_pool2d_back_f32.cu", "max_pool2d_back_f32")
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_avg_pool2d_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Avg_Pool2d)
	fmt.assertf(x.type == .F32, "avg_pool2d requires F32 (got %v)", x.type, loc=loc)

	n := i32(x.shape[0]); h := i32(x.shape[1]); w := i32(x.shape[2]); c := i32(x.shape[3])
	kh := i32(v.kernel_h); kw := i32(v.kernel_w)
	sh := i32(v.stride_h); sw := i32(v.stride_w)
	oh := i32(y.shape[1]); ow := i32(y.shape[2])
	total := int(n) * int(oh) * int(ow) * int(c)

	xp := data(x).ptr; yp := data(y).ptr
	args := [?]rawptr{&xp, &yp, &n, &h, &w, &c, &kh, &kw, &sh, &sw, &oh, &ow}
	p := _compile_pipeline(AVG_POOL2D_F32_SRC, "avg_pool2d_f32.cu", "avg_pool2d_f32")
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_avg_pool2d_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Avg_Pool2d)
	if gradient(x).ptr == 0 { return }

	n := i32(x.shape[0]); h := i32(x.shape[1]); w := i32(x.shape[2]); c := i32(x.shape[3])
	kh := i32(v.kernel_h); kw := i32(v.kernel_w)
	sh := i32(v.stride_h); sw := i32(v.stride_w)
	oh := i32(y.shape[1]); ow := i32(y.shape[2])
	total := int(n) * int(oh) * int(ow) * int(c)

	dyp := gradient(y).ptr; dxp := gradient(x).ptr
	args := [?]rawptr{&dyp, &dxp, &n, &h, &w, &c, &kh, &kw, &sh, &sw, &oh, &ow}
	p := _compile_pipeline(AVG_POOL2D_BACK_F32_SRC, "avg_pool2d_back_f32.cu", "avg_pool2d_back_f32")
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_transpose_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	fmt.assertf(x.type == .F32, "transpose requires F32 (got %v)", x.type, loc=loc)
	rows := x.shape[0]; cols := x.shape[1]; total := rows * cols
	p := _compile_pipeline(TRANSPOSE_F32_SRC, "transpose_f32.cu", "transpose_f32")
	xp := data(x).ptr; yp := data(y).ptr; r := i32(rows); c := i32(cols)
	args := [?]rawptr{&xp, &yp, &r, &c}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_transpose_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	if gradient(x).ptr == 0 { return }
	rows := x.shape[0]; cols := x.shape[1]; total := rows * cols
	p := _compile_pipeline(TRANSPOSE_BACK_F32_SRC, "transpose_back_f32.cu", "transpose_back_f32")
	dyp := gradient(y).ptr; dxp := gradient(x).ptr; r := i32(rows); c := i32(cols)
	args := [?]rawptr{&dyp, &dxp, &r, &c}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_slice_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Slice)
	fmt.assertf(x.type == .F32, "slice requires F32 (got %v)", x.type, loc=loc)
	gctx := _gctx(loc)
	bytes := uint(ml.len(y) * 4)
	src_off := uint(v.start * 4)
	cuda.check(cuda.MemcpyDtoDAsync(data(y).ptr, data(x).ptr + cuda.DevicePtr(src_off), bytes, gctx.stream), loc=loc)
}
_slice_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Slice)
	if gradient(x).ptr == 0 { return }
	count := ml.len(y)
	dyp := gradient(y).ptr; dxp := gradient(x).ptr; cnt := i32(count); off := i32(v.start)
	args := [?]rawptr{&dyp, &dxp, &cnt, &off}
	p := _compile_pipeline(SLICE_LEADING_BACK_F32_SRC, "slice_leading_back_f32.cu", "slice_leading_back_f32")
	_dispatch(p, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_concat_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	output := op.output
	inputs := op.variant.(ml.Concat).inputs
	fmt.assertf(output.type == .F32, "concat requires F32 (got %v)", output.type, loc=loc)
	out_trailing := output.shape[output.rank - 1]
	p := _compile_pipeline(CONCAT_F32_SRC, "concat_f32.cu", "concat_f32")
	outp := data(output).ptr
	ot := i32(out_trailing)
	dst_col := 0
	for input in inputs {
		in_trailing := input.shape[input.rank - 1]
		leading := ml.len(input) / in_trailing
		inp := data(input).ptr
		ld := i32(leading); it := i32(in_trailing); dc := i32(dst_col)
		args := [?]rawptr{&inp, &outp, &ld, &it, &ot, &dc}
		_dispatch(p, _div_up(leading * in_trailing, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
		dst_col += in_trailing
	}
}
_concat_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	output := op.output
	inputs := op.variant.(ml.Concat).inputs
	out_trailing := output.shape[output.rank - 1]
	p := _compile_pipeline(CONCAT_BACK_F32_SRC, "concat_back_f32.cu", "concat_back_f32")
	dyp := gradient(output).ptr
	ot := i32(out_trailing)
	src_col := 0
	for input in inputs {
		in_trailing := input.shape[input.rank - 1]
		leading := ml.len(input) / in_trailing
		dxp := gradient(input).ptr
		if dxp != 0 {
			ld := i32(leading); it := i32(in_trailing); sc := i32(src_col)
			args := [?]rawptr{&dyp, &dxp, &ld, &it, &ot, &sc}
			_dispatch(p, _div_up(leading * in_trailing, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
		}
		src_col += in_trailing
	}
}

_layernorm_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Layernorm)
	fmt.assertf(x.type == .F32, "layernorm requires F32 (got %v)", x.type, loc=loc)
	fmt.assertf(v.weight.type == .F32, "layernorm requires F32 weight (got %v)", v.weight.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size
	p := _compile_pipeline(LAYERNORM_F32_SRC, "layernorm_f32.cu", "layernorm_f32")
	xp := data(x).ptr; wp := data(v.weight).ptr; yp := data(y).ptr
	mp := data(v.mean).ptr; rp := data(v.rstd).ptr
	c := i32(count); s := i32(size); eps := LAYERNORM_EPS
	args := [?]rawptr{&xp, &wp, &yp, &mp, &rp, &c, &s, &eps}
	_dispatch(p, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_layernorm_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Layernorm)
	fmt.assertf(x.type == .F32, "layernorm requires F32 (got %v)", x.type, loc=loc)
	size := x.shape[x.rank - 1]; count := ml.len(x) / size
	dxp := gradient(x).ptr; dwp := gradient(v.weight).ptr
	if dxp == 0 && dwp == 0 { return }
	p := _compile_pipeline(LAYERNORM_BACK_F32_SRC, "layernorm_back_f32.cu", "layernorm_back_f32")
	xp := data(x).ptr; wp := data(v.weight).ptr; mp := data(v.mean).ptr; rp := data(v.rstd).ptr
	dyp := gradient(y).ptr
	c := i32(count); s := i32(size)
	have_dx := i32(dxp != 0 ? 1 : 0); have_dw := i32(dwp != 0 ? 1 : 0)
	args := [?]rawptr{&xp, &wp, &mp, &rp, &dyp, &dxp, &dwp, &c, &s, &have_dx, &have_dw}
	_dispatch(p, u32(count), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_log_softmax_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	fmt.assertf(x.type == .F32, "log_softmax requires F32 (got %v)", x.type, loc=loc)
	cols := x.shape[x.rank - 1]; rows := ml.len(x) / cols
	p := _compile_pipeline(LOG_SOFTMAX_F32_SRC, "log_softmax.cu", "log_softmax_f32")
	xp := data(x).ptr; yp := data(y).ptr; rr := i32(rows); cc := i32(cols)
	args := [?]rawptr{&xp, &yp, &rr, &cc}
	_dispatch(p, _div_up(rows, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_log_softmax_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	if gradient(x).ptr == 0 { return }
	cols := x.shape[x.rank - 1]; rows := ml.len(x) / cols
	p := _compile_pipeline(LOG_SOFTMAX_BACK_F32_SRC, "log_softmax_back_f32.cu", "log_softmax_back_f32")
	yp := data(y).ptr; dyp := gradient(y).ptr; dxp := gradient(x).ptr; rr := i32(rows); cc := i32(cols)
	args := [?]rawptr{&yp, &dyp, &dxp, &rr, &cc}
	_dispatch(p, _div_up(rows, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_mean_squared_error_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	pred := op.input
	out  := op.output
	v    := op.variant.(ml.Mean_Squared_Error)
	fmt.assertf(pred.type == .F32, "mean_squared_error requires F32 (got %v)", pred.type, loc=loc)
	count := ml.len(out); sample_size := ml.len(pred) / count
	p := _compile_pipeline(MSE_F32_SRC, "mse_f32.cu", "mse_f32")
	pp := data(pred).ptr; tp := data(v.targets).ptr; outp := data(out).ptr
	c := i32(count); ss := i32(sample_size)
	args := [?]rawptr{&pp, &tp, &outp, &c, &ss}
	_dispatch(p, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_mean_squared_error_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	pred := op.input
	out  := op.output
	v    := op.variant.(ml.Mean_Squared_Error)
	if gradient(pred).ptr == 0 { return }
	count := ml.len(out); sample_size := ml.len(pred) / count; total := count * sample_size
	p := _compile_pipeline(MSE_BACK_F32_SRC, "mse_back_f32.cu", "mse_back_f32")
	pp := data(pred).ptr; tp := data(v.targets).ptr; dyp := gradient(out).ptr; dxp := gradient(pred).ptr
	c := i32(count); ss := i32(sample_size)
	args := [?]rawptr{&pp, &tp, &dyp, &dxp, &c, &ss}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_smooth_l1_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	pred := op.input
	out  := op.output
	v    := op.variant.(ml.Smooth_L1)
	fmt.assertf(pred.type == .F32, "smooth_l1 requires F32 (got %v)", pred.type, loc=loc)
	count := ml.len(out); sample_size := ml.len(pred) / count
	p := _compile_pipeline(SMOOTH_L1_F32_SRC, "smooth_l1_f32.cu", "smooth_l1_f32")
	pp := data(pred).ptr; tp := data(v.targets).ptr; outp := data(out).ptr
	c := i32(count); ss := i32(sample_size); beta := v.beta
	args := [?]rawptr{&pp, &tp, &outp, &c, &ss, &beta}
	_dispatch(p, _div_up(count, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_smooth_l1_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	pred := op.input
	out  := op.output
	v    := op.variant.(ml.Smooth_L1)
	if gradient(pred).ptr == 0 { return }
	count := ml.len(out); sample_size := ml.len(pred) / count; total := count * sample_size
	p := _compile_pipeline(SMOOTH_L1_BACK_F32_SRC, "smooth_l1_back_f32.cu", "smooth_l1_back_f32")
	pp := data(pred).ptr; tp := data(v.targets).ptr; dyp := gradient(out).ptr; dxp := gradient(pred).ptr
	c := i32(count); ss := i32(sample_size); beta := v.beta
	args := [?]rawptr{&pp, &tp, &dyp, &dxp, &c, &ss, &beta}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_batched_matmul_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a := op.input
	b := op.variant.(ml.Batched_Matmul).b
	c := op.output
	fmt.assertf(a.type == .F32, "batched_matmul requires F32 (got %v)", a.type, loc=loc)
	batch := a.shape[0]; m := a.shape[1]; k := a.shape[2]; n := b.shape[2]
	gctx := _gctx(loc)
	alpha := f32(1.0); beta := f32(0.0)
	ap := data(a).ptr; bp := data(b).ptr; cp := data(c).ptr
	cublas.check(cublas.GemmStridedBatchedEx(
		gctx.cublas_handle,
		.N, .N,
		i32(n), i32(m), i32(k),
		&alpha,
		rawptr(uintptr(bp)), .R_32F, i32(n), i64(k * n),
		rawptr(uintptr(ap)), .R_32F, i32(k), i64(m * k),
		&beta,
		rawptr(uintptr(cp)), .R_32F, i32(n), i64(m * n),
		i32(batch),
		._32F, .DEFAULT,
	), loc=loc)
}
_batched_matmul_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	a := op.input
	b := op.variant.(ml.Batched_Matmul).b
	c := op.output
	fmt.assertf(a.type == .F32, "batched_matmul requires F32 (got %v)", a.type, loc=loc)
	batch := a.shape[0]; m := a.shape[1]; k := a.shape[2]; n := b.shape[2]
	dap := gradient(a).ptr; dbp := gradient(b).ptr
	if dap == 0 && dbp == 0 { return }
	gctx := _gctx(loc)
	alpha := f32(1.0); beta := f32(1.0)
	ap := data(a).ptr; bp := data(b).ptr; dcp := gradient(c).ptr

	if dap != 0 {
		cublas.check(cublas.GemmStridedBatchedEx(
			gctx.cublas_handle,
			.T, .N,
			i32(k), i32(m), i32(n),
			&alpha,
			rawptr(uintptr(bp)),  .R_32F, i32(n), i64(k * n),
			rawptr(uintptr(dcp)), .R_32F, i32(n), i64(m * n),
			&beta,
			rawptr(uintptr(dap)), .R_32F, i32(k), i64(m * k),
			i32(batch),
			._32F, .DEFAULT,
		), loc=loc)
	}
	if dbp != 0 {
		cublas.check(cublas.GemmStridedBatchedEx(
			gctx.cublas_handle,
			.N, .T,
			i32(n), i32(k), i32(m),
			&alpha,
			rawptr(uintptr(dcp)), .R_32F, i32(n), i64(m * n),
			rawptr(uintptr(ap)),  .R_32F, i32(k), i64(m * k),
			&beta,
			rawptr(uintptr(dbp)), .R_32F, i32(n), i64(k * n),
			i32(batch),
			._32F, .DEFAULT,
		), loc=loc)
	}
}

_permute_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Permute)
	fmt.assertf(x.type == .F32, "permute requires F32 (got %v)", x.type, loc=loc)
	p := _compile_pipeline(PERMUTE_F32_SRC, "permute_f32.cu", "permute_f32")
	xp := data(x).ptr; yp := data(y).ptr
	s0 := i32(x.shape[0]); s1 := i32(x.shape[1]); s2 := i32(x.shape[2])
	a0 := i32(v.axes[0]); a1 := i32(v.axes[1]); a2 := i32(v.axes[2])
	args := [?]rawptr{&xp, &yp, &s0, &s1, &s2, &a0, &a1, &a2}
	_dispatch(p, _div_up(ml.len(y), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_permute_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	v := op.variant.(ml.Permute)
	if gradient(x).ptr == 0 { return }
	p := _compile_pipeline(PERMUTE_BACK_F32_SRC, "permute_back_f32.cu", "permute_back_f32")
	dyp := gradient(y).ptr; dxp := gradient(x).ptr
	s0 := i32(x.shape[0]); s1 := i32(x.shape[1]); s2 := i32(x.shape[2])
	a0 := i32(v.axes[0]); a1 := i32(v.axes[1]); a2 := i32(v.axes[2])
	args := [?]rawptr{&dyp, &dxp, &s0, &s1, &s2, &a0, &a1, &a2}
	_dispatch(p, _div_up(ml.len(y), 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_causal_mask_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	fmt.assertf(x.type == .F32, "causal_mask requires F32 (got %v)", x.type, loc=loc)
	T := x.shape[x.rank - 1]; block := T * T; n_blocks := ml.len(x) / block; total := ml.len(x)
	p := _compile_pipeline(CAUSAL_MASK_F32_SRC, "causal_mask_f32.cu", "causal_mask_f32")
	xp := data(x).ptr; yp := data(y).ptr; nb := i32(n_blocks); tt := i32(T)
	args := [?]rawptr{&xp, &yp, &nb, &tt}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}
_causal_mask_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	x := op.input
	y := op.output
	if gradient(x).ptr == 0 { return }
	T := x.shape[x.rank - 1]; block := T * T; n_blocks := ml.len(x) / block; total := ml.len(x)
	p := _compile_pipeline(CAUSAL_MASK_BACK_F32_SRC, "causal_mask_back_f32.cu", "causal_mask_back_f32")
	dyp := gradient(y).ptr; dxp := gradient(x).ptr; nb := i32(n_blocks); tt := i32(T)
	args := [?]rawptr{&dyp, &dxp, &nb, &tt}
	_dispatch(p, _div_up(total, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_lerp_assign_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	v := op.variant.(ml.Lerp_Assign)
	dst := op.output
	src := v.source
	fmt.assertf(dst.type == .F32, "lerp_assign requires F32 (got %v)", dst.type, loc=loc)
	n := ml.len(dst)
	p := _compile_pipeline(LERP_ASSIGN_F32_SRC, "lerp_assign_f32.cu", "lerp_assign_f32")
	dstp := data(dst).ptr; srcp := data(src).ptr; alpha := v.alpha; nn := i32(n)
	args := [?]rawptr{&dstp, &srcp, &alpha, &nn}
	_dispatch(p, _div_up(n, 256), 1, 1, 256, 1, 1, 0, args[:], loc)
}

_accumulate_mean_forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	src := op.input
	dst := op.output
	fmt.assertf(src.type == .F32, "accumulate_mean requires F32 (got %v)", src.type, loc=loc)
	n := ml.len(src)
	p := _compile_pipeline(ACCUMULATE_MEAN_F32_SRC, "accumulate_mean_f32.cu", "accumulate_mean_f32")
	srcp := data(src).ptr; dstp := data(dst).ptr; nn := i32(n)
	args := [?]rawptr{&srcp, &dstp, &nn}
	_dispatch(p, 1, 1, 1, 256, 1, 1, 0, args[:], loc)
}
