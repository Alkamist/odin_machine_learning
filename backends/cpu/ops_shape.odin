package cpu

import "base:builtin"
import "base:intrinsics"

import ml "../.."

_transpose_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	rows    := input.shape[0]
	columns := input.shape[1]

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			_data(output)[j * rows + i] = _data(input)[i * columns + j]
		}
	}
}

_transpose_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output
	rows    := input.shape[0]
	columns := input.shape[1]

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			_gradient(input)[i * columns + j] += _gradient(output)[j * rows + i]
		}
	}
}

_select_forward :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	index_count := ml.len(op.variant.(ml.Select).indices)
	indices     := _typed_data(i32, op.variant.(ml.Select).indices)
	size        := ml.len(output) / index_count

	rows := input.shape[0]

	elem_size := ml.data_type_size(input.type)
	row_bytes := size * elem_size
	src_bytes := transmute([]byte)input.buffers [.Data]
	dst_bytes := transmute([]byte)output.buffers[.Data]

	for i in 0 ..< index_count {
		index := int(indices[i])
		assert(index >= 0 && index < rows, "select index out of bounds")
		src_off := index * row_bytes
		dst_off := i     * row_bytes
		builtin.copy(dst_bytes[dst_off:dst_off + row_bytes], src_bytes[src_off:src_off + row_bytes])
	}
}

_select_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	weight, output := op.input, op.output
	index_count    := ml.len(op.variant.(ml.Select).indices)
	indices        := _typed_data(i32, op.variant.(ml.Select).indices)
	size           := ml.len(output) / index_count

	dw, dy := _gradient(weight), _gradient(output)
	for i in 0 ..< index_count {
		for j in 0 ..< size {
			dw[int(indices[i]) * size + j] += dy[i * size + j]
		}
	}
}

_slice_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _slice_forward_impl(f32,      op)
	case .Bf16: _slice_forward_impl(ml.Bf16, op)
	}
}

_slice_forward_impl :: proc($T: typeid, op: ml.Operation) {
	start := op.variant.(ml.Slice).start
	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for i in 0 ..< ml.len(op.output) {
		yp[i] = xp[start + i]
	}
}

_slice_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice)
	start   := variant.start

	dx, dy := _gradient(input), _gradient(output)
	for i in 0 ..< ml.len(output) {
		dx[start + i] += dy[i]
	}
}

_slice_trailing_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _slice_trailing_forward_impl(f32,      op)
	case .Bf16: _slice_trailing_forward_impl(ml.Bf16, op)
	}
}

_slice_trailing_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	start := op.variant.(ml.Slice_Trailing).start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml._leading_count(input)

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			yp[out_off + i] = xp[in_off + i]
		}
	}
}

_slice_trailing_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice_Trailing)
	start   := variant.start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml._leading_count(input)

	dx, dy := _gradient(input), _gradient(output)
	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			dx[in_off + i] += dy[out_off + i]
		}
	}
}

_slice_leading_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _slice_leading_forward_impl(f32,      op)
	case .Bf16: _slice_leading_forward_impl(ml.Bf16, op)
	}
}

_slice_leading_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	start := op.variant.(ml.Slice_Leading).start

	leading  := input.shape[0]
	row_size := ml.len(input) / leading
	count    := output.shape[0] * row_size
	in_off   := start * row_size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	for i in 0 ..< count {
		yp[i] = xp[in_off + i]
	}
}

_slice_leading_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice_Leading)
	start   := variant.start

	leading  := input.shape[0]
	row_size := ml.len(input) / leading
	count    := output.shape[0] * row_size
	in_off   := start * row_size

	dx, dy := _gradient(input), _gradient(output)
	for i in 0 ..< count {
		dx[in_off + i] += dy[i]
	}
}

_concat_forward :: proc(op: ml.Operation) {
	#partial switch op.output.type {
	case .F32:  _concat_forward_impl(f32,      op)
	case .Bf16: _concat_forward_impl(ml.Bf16, op)
	}
}

_concat_forward_impl :: proc($T: typeid, op: ml.Operation) {
	output := op.output
	inputs := op.variant.(ml.Concat).inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

	yp := _typed_data(T, output)
	dst_col := 0
	for input in inputs {
		xp          := _typed_data(T, input)
		in_trailing := input.shape[input.rank - 1]
		for r in 0 ..< leading {
			out_off := r * out_trailing + dst_col
			in_off  := r * in_trailing
			for i in 0 ..< in_trailing {
				yp[out_off + i] = xp[in_off + i]
			}
		}
		dst_col += in_trailing
	}
}

_concat_backward :: proc(op: ml.Operation) {
	output := op.output

	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

	dy := _gradient(output)
	src_col := 0
	for input in inputs {
		in_trailing := input.shape[input.rank - 1]
		if ml.has_gradient(input) {
			dx := _gradient(input)
			for r in 0 ..< leading {
				out_off := r * out_trailing + src_col
				in_off  := r * in_trailing
				for i in 0 ..< in_trailing {
					dx[in_off + i] += dy[out_off + i]
				}
			}
		}
		src_col += in_trailing
	}
}


_permute_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _permute_forward_impl(f32,      op)
	case .Bf16: _permute_forward_impl(ml.Bf16, op)
	}
}

_permute_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input  := op.input
	output := op.output
	axes   := op.variant.(ml.Permute).axes

	in_shape   := [3]int{input.shape [0],           input.shape [1], input.shape [2]}
	out_shape  := [3]int{output.shape[0],           output.shape[1], output.shape[2]}
	in_strides := [3]int{in_shape[1] * in_shape[2], in_shape[2],     1              }

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	for i0 in 0 ..< out_shape[0] {
		for i1 in 0 ..< out_shape[1] {
			for i2 in 0 ..< out_shape[2] {
				src: [3]int
				src[axes[0]] = i0
				src[axes[1]] = i1
				src[axes[2]] = i2

				src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
				dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

				yp[dst_idx] = xp[src_idx]
			}
		}
	}
}

_permute_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input   := op.input
	output  := op.output
	axes    := op.variant.(ml.Permute).axes

	in_shape   := [3]int{input.shape [0],           input.shape [1], input.shape [2]}
	out_shape  := [3]int{output.shape[0],           output.shape[1], output.shape[2]}
	in_strides := [3]int{in_shape[1] * in_shape[2], in_shape[2],     1              }

	dx, dy := _gradient(input), _gradient(output)
	for i0 in 0 ..< out_shape[0] {
		for i1 in 0 ..< out_shape[1] {
			for i2 in 0 ..< out_shape[2] {
				src: [3]int
				src[axes[0]] = i0
				src[axes[1]] = i1
				src[axes[2]] = i2

				src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
				dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

				dx[src_idx] += dy[dst_idx]
			}
		}
	}
}

