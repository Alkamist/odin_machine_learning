package cpu

import "base:builtin"
import "base:runtime"
import "base:intrinsics"

import "core:math"

import ml "../.."

_softmax_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		for sample in 0 ..< count {
			max_value := math.NEG_INF_F32
			for i in 0 ..< size {
				index := sample * size + i
				max_value = math.max(max_value, _data(input)[index])
			}
			sum: f32
			for i in 0 ..< size {
				index := sample * size + i
				exp_val := math.exp(_data(input)[index] - max_value)
				_data(output)[index] = exp_val
				sum += exp_val
			}
			for i in 0 ..< size {
				index := sample * size + i
				_data(output)[index] /= sum
			}
		}
	case .Bf16:
		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for sample in 0 ..< count {
			runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

			base := sample * size
			max_value := math.NEG_INF_F32
			for i in 0 ..< size {
				v := ml.bf16_to_f32(x_bf[base + i])
				if v > max_value { max_value = v }
			}
			sum: f32
			scratch := make([]f32, size, context.temp_allocator)
			for i in 0 ..< size {
				e := math.exp(ml.bf16_to_f32(x_bf[base + i]) - max_value)
				scratch[i] = e
				sum += e
			}
			for i in 0 ..< size {
				y_bf[base + i] = ml.bf16_from_f32(scratch[i] / sum)
			}
		}
	}
}

_softmax_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	size  := op.input.shape[op.input.rank - 1]
	count := ml.len(op.input) / size

	#partial switch op.input.type {
	case .F32:
		_parallelize(count, count, op, proc(index: int, op: ml.Operation) {
			input, output := op.input, op.output
			size  := input.shape[input.rank - 1]
			base  := index * size

			out_data := _data(output)    [base:base + size]
			out_grad := _gradient(output)[base:base + size]
			in_grad  := _gradient(input) [base:base + size]

			dot: f32
			for i in 0 ..< size {
				dot += out_grad[i] * out_data[i]
			}

			for i in 0 ..< size {
				in_grad[i] += out_data[i] * (out_grad[i] - dot)
			}
		})
	case .Bf16:
		y_bf := _data_bf16(op.output)
		dy   := _gradient(op.output)
		dx   := _gradient(op.input)
		for sample in 0 ..< count {
			base := sample * size
			dot:  f32
			for i in 0 ..< size {
				dot += dy[base + i] * ml.bf16_to_f32(y_bf[base + i])
			}
			for i in 0 ..< size {
				y_v := ml.bf16_to_f32(y_bf[base + i])
				dx[base + i] += y_v * (dy[base + i] - dot)
			}
		}
	}
}

_log_softmax_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _log_softmax_forward_impl(f32,      op)
	case .Bf16: _log_softmax_forward_impl(ml.Bf16, op)
	}
}

_log_softmax_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input  := op.input
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		base := sample * size
		max_value := math.NEG_INF_F32
		for i in 0 ..< size {
			v := _load(xp, base + i)
			if v > max_value { max_value = v }
		}
		lse: f32
		for i in 0 ..< size {
			lse += math.exp(_load(xp, base + i) - max_value)
		}
		lse = math.ln(lse) + max_value
		for i in 0 ..< size {
			_store(yp, base + i, _load(xp, base + i) - lse)
		}
	}
}

_log_softmax_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _log_softmax_backward_impl(f32,      op)
	case .Bf16: _log_softmax_backward_impl(ml.Bf16, op)
	}
}

_log_softmax_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	yp := _typed_data(T, output)
	dy := _gradient(output)
	dx := _gradient(input)
	for sample in 0 ..< count {
		base := sample * size
		grad_sum: f32
		for i in 0 ..< size {
			grad_sum += dy[base + i]
		}
		for i in 0 ..< size {
			dx[base + i] += dy[base + i] - math.exp(_load(yp, base + i)) * grad_sum
		}
	}
}

_entropy_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _entropy_forward_impl(f32,      op)
	case .Bf16: _entropy_forward_impl(ml.Bf16, op)
	}
}

_entropy_forward_impl :: proc($T: typeid, op: ml.Operation) {
	probabilities := op.input
	size          := probabilities.shape[probabilities.rank - 1]
	count         := ml.len(probabilities) / size

	pp := _typed_data(T, probabilities)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		entropy_value: f32
		base := sample * size
		for i in 0 ..< size {
			p      := _load(pp, base + i)
			p_safe := math.max(p, f32(1e-8))
			entropy_value -= p * math.ln(p_safe)
		}
		_store(yp, sample, entropy_value)
	}
}

_entropy_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _entropy_backward_impl(f32,      op)
	case .Bf16: _entropy_backward_impl(ml.Bf16, op)
	}
}

_entropy_backward_impl :: proc($T: typeid, op: ml.Operation) {
	probabilities := op.input
	size  := probabilities.shape[probabilities.rank - 1]
	count := ml.len(probabilities) / size

	pp    := _typed_data(T, probabilities)
	dp    := _gradient(probabilities)
	d_out := _gradient(op.output)
	for sample in 0 ..< count {
		base   := sample * size
		dout_v := d_out[sample]
		for i in 0 ..< size {
			p      := _load(pp, base + i)
			p_safe := math.max(p, f32(1e-8))
			grad   := -(math.ln(p_safe) + 1.0)
			dp[base + i] += dout_v * grad
		}
	}
}

_mean_squared_error_forward :: proc(op: ml.Operation) {
	predictions := op.input
	output      := op.output
	targets     := op.variant.(ml.Mean_Squared_Error).targets
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		sum_squared_error: f32

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			diff  := _data(predictions)[index] - _data(targets)[index]
			sum_squared_error += diff * diff
		}

		_data(output)[sample] = sum_squared_error / f32(sample_size)
	}
}

_mean_squared_error_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	predictions, output := op.input, op.output
	targets := op.variant.(ml.Mean_Squared_Error).targets
	count   := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		scale := 2.0 / f32(sample_size)

		upstream_gradient := _gradient(output)[sample]

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			grad := scale * (_data(predictions)[index] - _data(targets)[index])
			_gradient(predictions)[index] += grad * upstream_gradient
		}
	}
}

_smooth_l1_forward :: proc(op: ml.Operation) {
	predictions := op.input
	output      := op.output
	variant     := op.variant.(ml.Smooth_L1)
	targets     := variant.targets
	beta        := variant.beta
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		sum: f32

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			diff  := _data(predictions)[index] - _data(targets)[index]
			if abs(diff) < beta {
				sum += 0.5 * diff * diff / beta
			} else {
				sum += abs(diff) - 0.5 * beta
			}
		}

		_data(output)[sample] = sum / f32(sample_size)
	}
}

_smooth_l1_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	predictions, output := op.input, op.output
	variant     := op.variant.(ml.Smooth_L1)
	targets     := variant.targets
	beta        := variant.beta
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		scale := 1.0 / f32(sample_size)

		upstream_gradient := _gradient(output)[sample]

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			diff  := _data(predictions)[index] - _data(targets)[index]
			grad  := math.clamp(diff / beta, -1, 1) * scale
			_gradient(predictions)[index] += grad * upstream_gradient
		}
	}
}

_cross_entropy_forward :: proc(op: ml.Operation) {
	input         := op.input
	output        := op.output
	variant       := op.variant.(ml.Cross_Entropy)
	probabilities := variant.probabilities
	sample_count  := ml.len(variant.targets)
	targets       := _typed_data(i32, variant.targets)
	class_size    := input.shape[input.rank - 1]

	for sample in 0 ..< sample_count {
		offset := sample * class_size
		target := int(targets[sample])
		assert(target >= 0 && target < class_size, "cross_entropy target out of bounds")

		max_value := math.NEG_INF_F32
		for i in 0 ..< class_size {
			index := offset + i
			max_value = math.max(max_value, _data(input)[index])
		}

		sum: f32
		for i in 0 ..< class_size {
			index := offset + i
			exp_val := math.exp(_data(input)[index] - max_value)
			_data(probabilities)[index] = exp_val
			sum += exp_val
		}

		for i in 0 ..< class_size {
			index := offset + i
			_data(probabilities)[index] /= sum
		}

		target_index := offset + target
		_data(output)[sample] = -_data(input)[target_index] + max_value + math.ln(sum)
	}
}

_cross_entropy_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant       := op.variant.(ml.Cross_Entropy)
	probabilities := variant.probabilities
	sample_count  := ml.len(variant.targets)
	targets       := _typed_data(i32, variant.targets)
	class_size    := input.shape[input.rank - 1]

	for sample in 0 ..< sample_count {
		offset := sample * class_size
		target := int(targets[sample])

		upstream_gradient := _gradient(output)[sample]

		for i in 0 ..< class_size {
			index := offset + i
			target_value: f32 = i == target ? 1 : 0

			grad := (_data(probabilities)[index] - target_value) * upstream_gradient

			_gradient(input)[index] += grad
		}
	}
}

