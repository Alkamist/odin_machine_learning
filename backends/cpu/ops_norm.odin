package cpu

import "base:builtin"
import "base:intrinsics"

import "core:math"

import ml "../.."

_rope_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rope_forward_impl(f32,      op)
	case .Bf16: _rope_forward_impl(ml.Bf16, op)
	}
}

_rope_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input             := op.input
	output            := op.output
	variant           := op.variant.(ml.Rope)
	head_count        := variant.head_count
	base              := variant.base
	pos_offset        := variant.position_offset
	rotate_pair_count := variant.rotate_pair_count
	cos_cache         := variant.cos_cache
	sin_cache         := variant.sin_cache
	token_count       := input.shape[0]
	head_size         := input.shape[input.rank - 1] / head_count
	half_head         := head_size / 2

	for pos in 0 ..< token_count {
		for i in 0 ..< rotate_pair_count {
			theta := f32(pos + pos_offset) / math.pow(base, f32(i * 2) / f32(head_size))
			cache_idx := pos * half_head + i
			_data(cos_cache)[cache_idx] = math.cos(theta)
			_data(sin_cache)[cache_idx] = math.sin(theta)
		}
	}

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< rotate_pair_count {
				cache_idx := t * half_head + i
				cos_val := _data(cos_cache)[cache_idx]
				sin_val := _data(sin_cache)[cache_idx]

				x := _load(xp, head_offset + i * 2)
				y := _load(xp, head_offset + i * 2 + 1)

				_store(yp, head_offset + i * 2,     x * cos_val - y * sin_val)
				_store(yp, head_offset + i * 2 + 1, x * sin_val + y * cos_val)
			}
			for i in rotate_pair_count ..< half_head {
				yp[head_offset + i * 2]     = xp[head_offset + i * 2]
				yp[head_offset + i * 2 + 1] = xp[head_offset + i * 2 + 1]
			}
		}
	}
}

_rope_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant           := op.variant.(ml.Rope)
	head_count        := variant.head_count
	rotate_pair_count := variant.rotate_pair_count
	cos_cache         := variant.cos_cache
	sin_cache         := variant.sin_cache
	token_count       := input.shape[0]
	head_size         := input.shape[input.rank - 1] / head_count
	half_head         := head_size / 2

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< rotate_pair_count {
				cache_idx := t * half_head + i
				cos_val := _data(cos_cache)[cache_idx]
				sin_val := _data(sin_cache)[cache_idx]

				grad_x := _gradient(output)[head_offset + i * 2]
				grad_y := _gradient(output)[head_offset + i * 2 + 1]

				_gradient(input)[head_offset + i * 2]     +=  grad_x * cos_val + grad_y * sin_val
				_gradient(input)[head_offset + i * 2 + 1] += -grad_x * sin_val + grad_y * cos_val
			}
			for i in rotate_pair_count ..< half_head {
				_gradient(input)[head_offset + i * 2]     += _gradient(output)[head_offset + i * 2]
				_gradient(input)[head_offset + i * 2 + 1] += _gradient(output)[head_offset + i * 2 + 1]
			}
		}
	}
}

LAYERNORM_EPSILON :: 1e-5

_layernorm_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _layernorm_forward_impl(f32,      op)
	case .Bf16: _layernorm_forward_impl(ml.Bf16, op)
	}
}

_layernorm_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := _data(variant.mean)
	rstd    := _data(variant.rstd)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	wp := _typed_data(T, weight)

	for c in 0 ..< count {
		offset := c * size

		m: f32
		for i in 0 ..< size {
			m += _load(xp, offset + i)
		}
		m /= f32(size)

		v: f32
		for i in 0 ..< size {
			x_shift := _load(xp, offset + i) - m
			v += x_shift * x_shift
		}
		v /= f32(size)

		s: f32 = 1.0 / math.sqrt(v + f32(LAYERNORM_EPSILON))
		for i in 0 ..< size {
			n := (s * (_load(xp, offset + i) - m))
			_store(yp, offset + i, n * _load(wp, i))
		}

		mean[c] = m
		rstd[c] = s
	}
}

_layernorm_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _layernorm_backward_impl(f32,      op)
	case .Bf16: _layernorm_backward_impl(ml.Bf16, op)
	}
}

_layernorm_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := _data(variant.mean)
	rstd    := _data(variant.rstd)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	have_dx, have_dw := ml.has_gradient(input), ml.has_gradient(weight)
	if !have_dx && !have_dw { return }

	xp := _typed_data(T, input)
	wp := _typed_data(T, weight)
	dx := _gradient(input)
	dw := _gradient(weight)
	dy := _gradient(output)

	for c in 0 ..< count {
		offset := c * size
		mean_c := mean[c]
		rstd_c := rstd[c]

		dnorm_mean:      f32
		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := (_load(xp, offset + i) - mean_c) * rstd_c
			dnorm := _load(wp, i) * dy[offset + i]
			dnorm_mean      += dnorm
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_mean      /= f32(size)
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			dy_v  := dy[offset + i]
			norm  := (_load(xp, offset + i) - mean_c) * rstd_c
			dnorm := _load(wp, i) * dy_v

			if have_dw {
				dw[i] += norm * dy_v
			}

			if have_dx {
				grad := dnorm - dnorm_mean - norm * dnorm_norm_mean
				grad *= rstd_c

				dx[offset + i] += grad
			}
		}
	}
}

_rmsnorm_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rmsnorm_forward_impl(f32,      op)
	case .Bf16: _rmsnorm_forward_impl(ml.Bf16, op)
	}
}

_rmsnorm_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Rmsnorm)
	weight  := variant.weight
	rstd    := _data(variant.rstd)
	eps     := variant.eps
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	wp := _typed_data(T, weight)

	for c in 0 ..< count {
		offset := c * size

		ms: f32
		for i in 0 ..< size {
			v := _load(xp, offset + i)
			ms += v * v
		}
		ms /= f32(size)

		s: f32 = 1.0 / math.sqrt(ms + eps)
		for i in 0 ..< size {
			_store(yp, offset + i, s * _load(xp, offset + i) * _load(wp, i))
		}

		rstd[c] = s
	}
}

_rmsnorm_rope_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rmsnorm_rope_forward_impl(f32,      op)
	case .Bf16: _rmsnorm_rope_forward_impl(ml.Bf16, op)
	}
}

_rmsnorm_rope_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input             := op.input
	output            := op.output
	variant           := op.variant.(ml.Rmsnorm_Rope)
	weight            := variant.weight
	eps               := variant.eps
	head_count        := variant.head_count
	rope_base         := variant.base
	pos_offset        := variant.position_offset
	rotate_pair_count := variant.rotate_pair_count
	token_count       := input.shape[0]
	head_size         := input.shape[1] / head_count
	half_head         := head_size / 2

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	wp := _typed_data(T, weight)

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			ms: f32
			for i in 0 ..< head_size {
				v := _load(xp, head_offset + i)
				ms += v * v
			}
			s: f32 = 1.0 / math.sqrt(ms / f32(head_size) + eps)

			for i in 0 ..< rotate_pair_count {
				theta   := f32(t + pos_offset) / math.pow(rope_base, f32(i * 2) / f32(head_size))
				cos_val := math.cos(theta)
				sin_val := math.sin(theta)
				n0 := s * _load(xp, head_offset + i * 2)     * _load(wp, i * 2)
				n1 := s * _load(xp, head_offset + i * 2 + 1) * _load(wp, i * 2 + 1)
				_store(yp, head_offset + i * 2,     n0 * cos_val - n1 * sin_val)
				_store(yp, head_offset + i * 2 + 1, n0 * sin_val + n1 * cos_val)
			}
			for i in rotate_pair_count ..< half_head {
				_store(yp, head_offset + i * 2,     s * _load(xp, head_offset + i * 2)     * _load(wp, i * 2))
				_store(yp, head_offset + i * 2 + 1, s * _load(xp, head_offset + i * 2 + 1) * _load(wp, i * 2 + 1))
			}
		}
	}
}

_add_rmsnorm_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _add_rmsnorm_forward_impl(f32,      op)
	case .Bf16: _add_rmsnorm_forward_impl(ml.Bf16, op)
	}
}

_add_rmsnorm_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a       := op.input
	normed  := op.output
	variant := op.variant.(ml.Add_Rmsnorm)
	b       := variant.b
	weight  := variant.weight
	resid   := variant.residual_out
	eps     := variant.eps
	size    := a.shape[a.rank - 1]
	count   := ml.len(a) / size

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	wp := _typed_data(T, weight)
	rp := _typed_data(T, resid)
	yp := _typed_data(T, normed)

	for c in 0 ..< count {
		offset := c * size

		ms: f32
		for i in 0 ..< size {
			_store(rp, offset + i, _load(ap, offset + i) + _load(bp, offset + i))
			vf := _load(rp, offset + i)
			ms += vf * vf
		}
		s: f32 = 1.0 / math.sqrt(ms / f32(size) + eps)
		for i in 0 ..< size {
			_store(yp, offset + i, s * _load(rp, offset + i) * _load(wp, i))
		}
	}
}

_rmsnorm_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rmsnorm_backward_impl(f32,      op)
	case .Bf16: _rmsnorm_backward_impl(ml.Bf16, op)
	}
}

_rmsnorm_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Rmsnorm)
	weight  := variant.weight
	rstd    := _data(variant.rstd)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	have_dx, have_dw := ml.has_gradient(input), ml.has_gradient(weight)
	if !have_dx && !have_dw { return }

	xp := _typed_data(T, input)
	wp := _typed_data(T, weight)
	dx := _gradient(input)
	dw := _gradient(weight)
	dy := _gradient(output)

	for c in 0 ..< count {
		offset := c * size
		rstd_c := rstd[c]

		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := _load(xp, offset + i) * rstd_c
			dnorm := _load(wp, i) * dy[offset + i]
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			dy_v  := dy[offset + i]
			norm  := _load(xp, offset + i) * rstd_c
			dnorm := _load(wp, i) * dy_v

			if have_dw {
				dw[i] += norm * dy_v
			}

			if have_dx {
				grad := (dnorm - norm * dnorm_norm_mean) * rstd_c
				dx[offset + i] += grad
			}
		}
	}
}

