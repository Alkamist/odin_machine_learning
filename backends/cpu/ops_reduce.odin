package cpu

import "base:builtin"
import "base:intrinsics"

import ml "../.."

_mean_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _mean_forward_impl(f32,      op)
	case .Bf16: _mean_forward_impl(ml.Bf16, op)
	}
}

_mean_forward_impl :: proc($T: typeid, op: ml.Operation) {
	count := ml.len(op.output)
	size  := ml.len(op.input) / count

	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		sum: f32
		for i in 0 ..< size {
			sum += _load(xp, sample * size + i)
		}
		_store(yp, sample, sum / f32(size))
	}
}

_mean_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	dx, dy := _gradient(input), _gradient(output)
	for sample in 0 ..< count {
		gradient_per_element := dy[sample] / f32(size)

		for i in 0 ..< size {
			dx[sample * size + i] += gradient_per_element
		}
	}
}

_sum_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _sum_forward_impl(f32,      op)
	case .Bf16: _sum_forward_impl(ml.Bf16, op)
	}
}

_sum_forward_impl :: proc($T: typeid, op: ml.Operation) {
	count := ml.len(op.output)
	size  := ml.len(op.input) / count

	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		total: f32
		for i in 0 ..< size {
			total += _load(xp, sample * size + i)
		}
		_store(yp, sample, total)
	}
}

_sum_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) {
		return
	}
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	dx, dy := _gradient(input), _gradient(output)
	for sample in 0 ..< count {
		for i in 0 ..< size {
			dx[sample * size + i] += dy[sample]
		}
	}
}

_max_reduce_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _max_reduce_forward_impl(f32,      op)
	case .Bf16: _max_reduce_forward_impl(ml.Bf16, op)
	}
}

_max_reduce_forward_impl :: proc($T: typeid, op: ml.Operation) {
	count := ml.len(op.output)
	size  := ml.len(op.input) / count

	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		best := _load(xp, sample * size)
		for i in 1 ..< size {
			value := _load(xp, sample * size + i)
			if value > best {
				best = value
			}
		}
		_store(yp, sample, best)
	}
}

_max_reduce_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _max_reduce_backward_impl(f32,      op)
	case .Bf16: _max_reduce_backward_impl(ml.Bf16, op)
	}
}

_max_reduce_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	xp     := _typed_data(T, input)
	dx, dy := _gradient(input), _gradient(output)
	for sample in 0 ..< count {
		best_index := 0
		best_value := _load(xp, sample * size)
		for i in 1 ..< size {
			value := _load(xp, sample * size + i)
			if value > best_value {
				best_value = value
				best_index = i
			}
		}
		dx[sample * size + best_index] += dy[sample]
	}
}

