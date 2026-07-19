package cpu

import "base:builtin"
import "base:intrinsics"

import ml "../.."

_im2col_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _im2col_forward_impl(f32,      op)
	case .Bf16: _im2col_forward_impl(ml.Bf16, op)
	}
}

_im2col_forward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Im2col)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)

	patch_size := v.kernel_h * v.kernel_w * c
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< v.out_h {
			for ox in 0 ..< v.out_w {
				row := ((n * v.out_h) + oy) * v.out_w + ox
				for ky in 0 ..< v.kernel_h {
					iy := oy * v.stride_h - v.pad_h + ky
					for kx in 0 ..< v.kernel_w {
						ix := ox * v.stride_w - v.pad_w + kx
						col_base := ((ky * v.kernel_w) + kx) * c
						if iy >= 0 && iy < h && ix >= 0 && ix < w {
							src_base := (((n * h) + iy) * w + ix) * c
							for ci in 0 ..< c {
								_store(yp, row * patch_size + col_base + ci, _load(xp, src_base + ci))
							}
						} else {
							for ci in 0 ..< c {
								_store(yp, row * patch_size + col_base + ci, 0)
							}
						}
					}
				}
			}
		}
	}
}

_im2col_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	v := op.variant.(ml.Im2col)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]

	dx, dy := _gradient(input), _gradient(op.output)

	patch_size := v.kernel_h * v.kernel_w * c
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< v.out_h {
			for ox in 0 ..< v.out_w {
				row := ((n * v.out_h) + oy) * v.out_w + ox
				for ky in 0 ..< v.kernel_h {
					iy := oy * v.stride_h - v.pad_h + ky
					if iy < 0 || iy >= h { continue }
					for kx in 0 ..< v.kernel_w {
						ix := ox * v.stride_w - v.pad_w + kx
						if ix < 0 || ix >= w { continue }
						col_base := ((ky * v.kernel_w) + kx) * c
						src_base := (((n * h) + iy) * w + ix) * c
						for ci in 0 ..< c {
							dx[src_base + ci] += dy[row * patch_size + col_base + ci]
						}
					}
				}
			}
		}
	}
}

_max_pool2d_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _max_pool2d_forward_impl(f32,      op)
	case .Bf16: _max_pool2d_forward_impl(ml.Bf16, op)
	}
}

_max_pool2d_forward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Max_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)

	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					best := _load(xp, (((n * h) + oy * v.stride_h) * w + ox * v.stride_w) * c + ci)
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							value := _load(xp, (((n * h) + iy) * w + ix) * c + ci)
							if value > best {
								best = value
							}
						}
					}
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					_store(yp, out_index, best)
				}
			}
		}
	}
}

_max_pool2d_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _max_pool2d_backward_impl(f32,      op)
	case .Bf16: _max_pool2d_backward_impl(ml.Bf16, op)
	}
}

_max_pool2d_backward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Max_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	xp := _typed_data(T, input)
	dx, dy := _gradient(input), _gradient(op.output)

	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					best_iy := oy * v.stride_h
					best_ix := ox * v.stride_w
					best_value := _load(xp, (((n * h) + best_iy) * w + best_ix) * c + ci)
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							value := _load(xp, (((n * h) + iy) * w + ix) * c + ci)
							if value > best_value {
								best_value = value
								best_iy    = iy
								best_ix    = ix
							}
						}
					}
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					dx[(((n * h) + best_iy) * w + best_ix) * c + ci] += dy[out_index]
				}
			}
		}
	}
}

_avg_pool2d_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _avg_pool2d_forward_impl(f32,      op)
	case .Bf16: _avg_pool2d_forward_impl(ml.Bf16, op)
	}
}

_avg_pool2d_forward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Avg_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)

	window := f32(v.kernel_h * v.kernel_w)
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					total: f32
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							total += _load(xp, (((n * h) + iy) * w + ix) * c + ci)
						}
					}
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					_store(yp, out_index, total / window)
				}
			}
		}
	}
}

_avg_pool2d_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _avg_pool2d_backward_impl(f32,      op)
	case .Bf16: _avg_pool2d_backward_impl(ml.Bf16, op)
	}
}

_avg_pool2d_backward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Avg_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	dx, dy := _gradient(input), _gradient(op.output)

	window := f32(v.kernel_h * v.kernel_w)
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					share := dy[out_index] / window
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							dx[(((n * h) + iy) * w + ix) * c + ci] += share
						}
					}
				}
			}
		}
	}
}

