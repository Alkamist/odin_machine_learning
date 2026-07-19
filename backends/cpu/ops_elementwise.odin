package cpu

import "base:builtin"
import "base:intrinsics"

import "core:math"

import ml "../.."

_lerp_assign_forward :: proc(op: ml.Operation) {
	dst    := _data(op.output)
	source := _data(op.variant.(ml.Lerp_Assign).source)
	alpha  := op.variant.(ml.Lerp_Assign).alpha
	one_minus := 1 - alpha
	for i in 0 ..< builtin.len(dst) {
		dst[i] = one_minus * dst[i] + alpha * source[i]
	}
}

_accumulate_mean_forward :: proc(op: ml.Operation) {
	dst    := _data(op.output)
	source := _data(op.input)
	sum: f32
	for v in source {
		sum += v
	}
	dst[0] += sum / f32(builtin.len(source))
}

_cast_forward :: proc(op: ml.Operation) {
	src_bytes := transmute([]byte)op.input.buffers[.Data]
	dst_bytes := transmute([]byte)op.output.buffers[.Data]
	_cast_bytes(src_bytes, op.input.type, dst_bytes, op.output.type, op.input.count)
}

_cast_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) {
		return
	}
	src_grad := _gradient(op.output)
	dst_grad := _gradient(op.input)
	for i in 0 ..< op.input.count {
		dst_grad[i] += src_grad[i]
	}
}

_cast_bytes :: proc(src: []byte, src_type: ml.Data_Type, dst: []byte, dst_type: ml.Data_Type, count: int) {
	src_f32  := ([^]f32    )(raw_data(src))[:count] if src_type == .F32  else nil
	src_bf16 := ([^]ml.Bf16)(raw_data(src))[:count] if src_type == .Bf16 else nil

	dst_f32  := ([^]f32    )(raw_data(dst))[:count] if dst_type == .F32  else nil
	dst_bf16 := ([^]ml.Bf16)(raw_data(dst))[:count] if dst_type == .Bf16 else nil

	for i in 0 ..< count {
		v: f32
		#partial switch src_type {
		case .F32:  v = src_f32 [i]
		case .Bf16: v = ml.bf16_to_f32(src_bf16[i])
		}
		#partial switch dst_type {
		case .F32:  dst_f32 [i] = v
		case .Bf16: dst_bf16[i] = ml.bf16_from_f32(v)
		}
	}
}

_cast_bytes_accumulate :: proc(src: []byte, src_type: ml.Data_Type, dst: []byte, dst_type: ml.Data_Type, count: int) {
	src_f32  := ([^]f32    )(raw_data(src))[:count] if src_type == .F32  else nil
	src_bf16 := ([^]ml.Bf16)(raw_data(src))[:count] if src_type == .Bf16 else nil

	dst_f32  := ([^]f32    )(raw_data(dst))[:count] if dst_type == .F32  else nil
	dst_bf16 := ([^]ml.Bf16)(raw_data(dst))[:count] if dst_type == .Bf16 else nil

	for i in 0 ..< count {
		v: f32
		#partial switch src_type {
		case .F32:  v = src_f32 [i]
		case .Bf16: v = ml.bf16_to_f32(src_bf16[i])
		}
		#partial switch dst_type {
		case .F32:  dst_f32 [i] += v
		case .Bf16: dst_bf16[i]  = ml.bf16_from_f32(ml.bf16_to_f32(dst_bf16[i]) + v)
		}
	}
}

_broadcast_tiling :: #force_inline proc(a, b: ml.Tensor) -> (stride, width: int) {
	width  = ml.len(b)
	stride = ml.len(a) / width
	return
}

_typed_data :: #force_inline proc($T: typeid, t: ml.Tensor) -> [^]T {
	return ([^]T)(raw_data(transmute([]byte)t.buffers[.Data]))
}

_load :: #force_inline proc "contextless" (p: [^]$T, i: int) -> f32 {
	when T == ml.Bf16 {
		return ml.bf16_to_f32(p[i])
	} else {
		return p[i]
	}
}

_store :: #force_inline proc "contextless" (p: [^]$T, i: int, value: f32) {
	when T == ml.Bf16 {
		p[i] = ml.bf16_from_f32(value)
	} else {
		p[i] = value
	}
}

_add_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _add_forward_impl(f32,     op)
	case .Bf16: _add_forward_impl(ml.Bf16, op)
	}
}

_add_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Add).b
	stride, width := _broadcast_tiling(a, b)

	ap  := _typed_data(T, a)
	bp  := _typed_data(T, b)
	op_ := _typed_data(T, output)
	#no_bounds_check for i in 0 ..< stride {
		row := i * width
		for j in 0 ..< width {
			_store(op_, row + j, _load(ap, row + j) + _load(bp, j))
		}
	}
}

_add_backward :: proc(op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Add).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy       := _gradient(a), _gradient(b), _gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	#no_bounds_check for i in 0 ..< stride {
		row_da := da[i * width:]
		row_dy := dy[i * width:]
		for j in 0 ..< width {
			if have_da { row_da[j] += row_dy[j] }
			if have_db { db[j]     += row_dy[j] }
		}
	}
}

_sub_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _sub_forward_impl(f32,     op)
	case .Bf16: _sub_forward_impl(ml.Bf16, op)
	}
}

_sub_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Sub).b
	stride, width := _broadcast_tiling(a, b)

	ap  := _typed_data(T, a)
	bp  := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, _load(ap, o) - _load(bp, j))
		}
	}
}

_sub_backward :: proc(op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Sub).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy       := _gradient(a), _gradient(b), _gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if have_da { da[o] += dy[o] }
			if have_db { db[j] -= dy[o] }
		}
	}
}

_mul_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _mul_forward_impl(f32,     op)
	case .Bf16: _mul_forward_impl(ml.Bf16, op)
	}
}

_mul_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Mul).b
	stride, width := _broadcast_tiling(a, b)

	ap  := _typed_data(T, a)
	bp  := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, _load(ap, o) * _load(bp, j))
		}
	}
}

_mul_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _mul_backward_impl(f32,     op)
	case .Bf16: _mul_backward_impl(ml.Bf16, op)
	}
}

_mul_backward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Mul).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy       := _gradient(a), _gradient(b), _gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if have_da { da[o] += dy[o] * _load(bp, j) }
			if have_db { db[j] += dy[o] * _load(ap, o) }
		}
	}
}

_div_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _div_forward_impl(f32,     op)
	case .Bf16: _div_forward_impl(ml.Bf16, op)
	}
}

_div_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Div).b
	stride, width := _broadcast_tiling(a, b)

	ap  := _typed_data(T, a)
	bp  := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, _load(ap, o) / _load(bp, j))
		}
	}
}

_div_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _div_backward_impl(f32,     op)
	case .Bf16: _div_backward_impl(ml.Bf16, op)
	}
}

_div_backward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Div).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy       := _gradient(a), _gradient(b), _gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			a_v := _load(ap, o)
			b_v := _load(bp, j)
			if have_da { da[o] += dy[o] / b_v }
			if have_db { db[j] += dy[o] * (-a_v / (b_v * b_v)) }
		}
	}
}

_exp_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _exp_forward_impl(f32,      op)
	case .Bf16: _exp_forward_impl(ml.Bf16, op)
	}
}

_exp_forward_impl :: proc($T: typeid, op: ml.Operation) {
	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for i in 0 ..< ml.len(op.input) {
		_store(yp, i, math.exp(_load(xp, i)))
	}
}

_exp_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _exp_backward_impl(f32,      op)
	case .Bf16: _exp_backward_impl(ml.Bf16, op)
	}
}

_exp_backward_impl :: proc($T: typeid, op: ml.Operation) {
	dx, dy := _gradient(op.input), _gradient(op.output)
	yp     := _typed_data(T, op.output)
	for i in 0 ..< ml.len(op.input) {
		dx[i] += _load(yp, i) * dy[i]
	}
}

_sqrt_forward :: proc(op: ml.Operation, loc := #caller_location) {
	input  := op.input
	output := op.output

	assert(input.type == .F32, "Sqrt is F32-only", loc=loc)

	for i in 0 ..< ml.len(input) {
		_data(output)[i] = math.sqrt(_data(input)[i])
	}
}

_sqrt_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) {
		return
	}
	input, output := op.input, op.output

	dx, dy := _gradient(input), _gradient(output)
	y      := _data(output)
	for i in 0 ..< ml.len(input) {
		if y[i] > 0 {
			dx[i] += 0.5 / y[i] * dy[i]
		}
	}
}

_clamp_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< ml.len(input) {
		_data(output)[i] = math.clamp(_data(input)[i], min_val, max_val)
	}
}

_clamp_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) {
		return
	}
	input, output := op.input, op.output

	variant := op.variant.(ml.Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< ml.len(input) {
		if _data(input)[i] >= min_val && _data(input)[i] <= max_val {
			_gradient(input)[i] += _gradient(output)[i]
		}
	}
}

_min_forward :: proc(op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Min).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp, op_ := _data(a), _data(b), _data(output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			op_[o] = math.min(ap[o], bp[j])
		}
	}
}

_min_backward :: proc(op: ml.Operation) {
	a, output     := op.input, op.output
	b             := op.variant.(ml.Min).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp           := _data(a), _data(b)
	da, db, dy       := _gradient(a), _gradient(b), _gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if ap[o] <= bp[j] {
				if have_da { da[o] += dy[o] }
			} else {
				if have_db { db[j] += dy[o] }
			}
		}
	}
}

_max_forward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Max).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp, op_ := _data(a), _data(b), _data(output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			op_[o] = math.max(ap[o], bp[j])
		}
	}
}

_max_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Max).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp           := _data(a), _data(b)
	da, db, dy       := _gradient(a), _gradient(b), _gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if ap[o] >= bp[j] {
				if have_da { da[o] += dy[o] }
			} else {
				if have_db { db[j] += dy[o] }
			}
		}
	}
}


_unary_forward_dispatch :: proc(op: ml.Operation, fwd_f32: proc(x: f32) -> f32) {
	input, output := op.input, op.output
	#partial switch input.type {
	case .F32:
		x := _data(input)
		y := _data(output)
		#no_bounds_check for i in 0 ..< ml.len(input) {
			y[i] = fwd_f32(x[i])
		}
	case .Bf16:
		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< ml.len(input) {
			y_bf[i] = ml.bf16_from_f32(fwd_f32(ml.bf16_to_f32(x_bf[i])))
		}
	}
}

_unary_backward_dispatch :: proc(op: ml.Operation, local_grad_from_input: proc(x: f32) -> f32) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output
	#partial switch input.type {
	case .F32:
		x  := _data(input)
		dx := _gradient(input)
		dy := _gradient(output)
		#no_bounds_check for i in 0 ..< ml.len(input) {
			dx[i] += dy[i] * local_grad_from_input(x[i])
		}
	case .Bf16:
		x_bf := _data_bf16(input)
		dy   := _gradient(output)
		dx   := _gradient(input)
		for i in 0 ..< ml.len(input) {
			x_v := ml.bf16_to_f32(x_bf[i])
			dx[i] += dy[i] * local_grad_from_input(x_v)
		}
	}
}

_relu_forward :: proc(op: ml.Operation) {
	if op.input.type != .F32 {
		_unary_forward_dispatch(op, proc(x: f32) -> f32 { return x < 0 ? 0 : x })
		return
	}

	x := _data(op.input)
	y := _data(op.output)
	#no_bounds_check for i in 0 ..< ml.len(op.input) {
		y[i] = x[i] < 0 ? 0 : x[i]
	}
}

_relu_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	if op.input.type != .F32 {
		_unary_backward_dispatch(op, proc(x: f32) -> f32 { return x > 0 ? 1 : 0 })
		return
	}

	x  := _data(op.input)
	dx := _gradient(op.input)
	dy := _gradient(op.output)
	#no_bounds_check for i in 0 ..< ml.len(op.input) {
		dx[i] += x[i] > 0 ? dy[i] : 0
	}
}

_sigmoid_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 { return 1.0 / (1.0 + math.exp(-x)) })
}

_sigmoid_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		s := f32(1.0 / (1.0 + math.exp(-x)))
		return s * (1.0 - s)
	})
}

GELU_SCALING_FACTOR :: 0.7978845608028654 // math.sqrt(f32(2) / math.PI)

_gelu_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 {
		cube := f32(0.044715) * x * x * x
		return 0.5 * x * (1.0 + math.tanh(f32(GELU_SCALING_FACTOR) * (x + cube)))
	})
}

_gelu_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		cube     := f32(0.044715) * x * x * x
		tanh_arg := f32(GELU_SCALING_FACTOR) * (x + cube)
		tanh_out := math.tanh(tanh_arg)
		cosh_out := math.cosh(tanh_arg)
		sech_out := 1.0 / (cosh_out * cosh_out)
		return 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * f32(GELU_SCALING_FACTOR) * (1.0 + 3.0 * 0.044715 * x * x)
	})
}

_gelu_mul_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _gelu_mul_forward_impl(f32,      op)
	case .Bf16: _gelu_mul_forward_impl(ml.Bf16, op)
	}
}

_gelu_mul_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Gelu_Mul).b
	stride, width := _broadcast_tiling(a, b)

	gelu :: proc(x: f32) -> f32 {
		cube := f32(0.044715) * x * x * x
		return 0.5 * x * (1.0 + math.tanh(f32(GELU_SCALING_FACTOR) * (x + cube)))
	}

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, gelu(_load(ap, o)) * _load(bp, j))
		}
	}
}

_silu_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 {
		s := f32(1.0 / (1.0 + math.exp(-x)))
		return x * s
	})
}

_silu_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		s := f32(1.0 / (1.0 + math.exp(-x)))
		return s + x * s * (1.0 - s)
	})
}

_tanh_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 { return math.tanh(x) })
}

_tanh_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		t := math.tanh(x)
		return 1.0 - t * t
	})
}

