package cpu

import "base:builtin"
import "base:intrinsics"

import ml "../.."

_linear_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _linear_forward_f32 (op)
	case .Bf16: _linear_forward_bf16(op)
	}
}

_linear_forward_f32 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(op.input) / input_size

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	work := count * output_size * input_size

	if count >= output_size {
		_parallelize(count, count, jd, proc(c: int, jd: Job_Data) {
			op := jd.op
			input, output := op.input, op.output
			weight      := op.variant.(ml.Linear).weight
			output_size := weight.shape[0]
			input_size  := weight.shape[1]

			input_ptr  := ([^]f32)(raw_data(_data(input)))
			output_ptr := ([^]f32)(raw_data(_data(output)))
			weight_ptr := ([^]f32)(raw_data(_data(weight)))

			x := input_ptr[c * input_size:]
			y := output_ptr[c * output_size:]

			for o in 0 ..< output_size {
				y[o] = _simd_dot_f32(weight_ptr[o * input_size:], x, input_size)
			}
		}, work=work)
		return
	}

	_parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		input, output := op.input, op.output
		weight      := op.variant.(ml.Linear).weight
		output_size := weight.shape[0]
		input_size  := weight.shape[1]

		input_ptr  := ([^]f32)(raw_data(_data(input)))
		output_ptr := ([^]f32)(raw_data(_data(output)))
		weight_ptr := ([^]f32)(raw_data(_data(weight)))

		w_row := weight_ptr[o * input_size:]
		for c in 0 ..< jd.count {
			x := input_ptr[c * input_size:]
			output_ptr[c * output_size + o] = _simd_dot_f32(w_row, x, input_size)
		}
	}, work=work)
}

_linear_forward_bf16 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	count       := ml.len(op.input) / weight.shape[1]

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	_parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		input, output := op.input, op.output
		weight      := op.variant.(ml.Linear).weight
		output_size := weight.shape[0]
		input_size  := weight.shape[1]

		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		w_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Data]))

		w_row := w_bf[o * input_size:]
		for c in 0 ..< jd.count {
			x_row := x_bf[c * input_size:]
			y_bf[c * output_size + o] = ml.bf16_from_f32(_simd_dot_bf16_f32(w_row, x_row, input_size))
		}
	})
}

_linear_q4_k_forward :: proc(op: ml.Operation) {
	v := op.variant.(ml.Linear_Q4_K)
	output_size := v.weight.shape[0]
	input_size  := v.weight.shape[1]
	count       := ml.len(op.input) / input_size

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	_parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Linear_Q4_K)
		output_size := v.weight.shape[0]
		input_size  := v.weight.shape[1]
		num_blocks  := input_size / ml.K_QUANT_BLOCK_SIZE
		row_bytes   := num_blocks * ml.Q4_K_BLOCK_BYTES

		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Data]))
		w_pk := transmute([]byte)v.weight.buffers[.Data]

		w_row   := w_pk[o * row_bytes : (o + 1) * row_bytes]
		dequant := make([]f32, input_size)
		defer delete(dequant)
		ml.dequantize_q4_k(w_row, dequant)

		for c in 0 ..< jd.count {
			x_row := x_bf[c * input_size:]
			total: f32
			for k in 0 ..< input_size {
				total += dequant[k] * ml.bf16_to_f32(x_row[k])
			}
			y_bf[c * output_size + o] = ml.bf16_from_f32(total)
		}
	})
}

_linear_q6_k_forward :: proc(op: ml.Operation) {
	v := op.variant.(ml.Linear_Q6_K)
	output_size := v.weight.shape[0]
	input_size  := v.weight.shape[1]
	count       := ml.len(op.input) / input_size

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	_parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Linear_Q6_K)
		output_size := v.weight.shape[0]
		input_size  := v.weight.shape[1]
		num_blocks  := input_size / ml.K_QUANT_BLOCK_SIZE
		row_bytes   := num_blocks * ml.Q6_K_BLOCK_BYTES

		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Data]))
		w_pk := transmute([]byte)v.weight.buffers[.Data]

		w_row   := w_pk[o * row_bytes : (o + 1) * row_bytes]
		dequant := make([]f32, input_size)
		defer delete(dequant)
		ml.dequantize_q6_k(w_row, dequant)

		for c in 0 ..< jd.count {
			x_row := x_bf[c * input_size:]
			total: f32
			for k in 0 ..< input_size {
				total += dequant[k] * ml.bf16_to_f32(x_row[k])
			}
			y_bf[c * output_size + o] = ml.bf16_from_f32(total)
		}
	})
}

_linear_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _linear_backward_f32 (op)
	case .Bf16: _linear_backward_bf16(op)
	}
}

_linear_backward_bf16 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	count       := ml.len(op.input) / weight.shape[1]

	if ml.has_gradient(weight) {
		_parallelize(output_size, output_size, op, proc(o: int, op: ml.Operation) {
			weight      := op.variant.(ml.Linear).weight
			input_size  := weight.shape[1]
			output_size := weight.shape[0]
			count       := ml.len(op.input) / input_size

			x_bf := _data_bf16(op.input)
			dy   := _gradient(op.output)
			dw   := _gradient(weight)

			dw_row := dw[o * input_size:]
			for k in 0 ..< input_size {
				acc: f32
				for c in 0 ..< count {
					acc += ml.bf16_to_f32(x_bf[c * input_size + k]) * dy[c * output_size + o]
				}
				dw_row[k] += acc
			}
		})
	}

	if ml.has_gradient(op.input) {
		_parallelize(count, count, op, proc(c: int, op: ml.Operation) {
			weight      := op.variant.(ml.Linear).weight
			input_size  := weight.shape[1]
			output_size := weight.shape[0]

			w_bf := _data_bf16(weight)
			dy   := _gradient(op.output)
			dx   := _gradient(op.input)

			dx_row := dx[c * input_size:]
			dy_row := dy[c * output_size:]
			for k in 0 ..< input_size {
				acc: f32
				for o in 0 ..< output_size {
					acc += ml.bf16_to_f32(w_bf[o * input_size + k]) * dy_row[o]
				}
				dx_row[k] += acc
			}
		})
	}
}

_linear_backward_f32 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(op.input) / input_size

	work := count * output_size * input_size

	if ml.has_gradient(weight) {
		_parallelize(output_size, output_size, op, proc(o: int, op: ml.Operation) {
			input, output := op.input, op.output
			weight      := op.variant.(ml.Linear).weight
			output_size := weight.shape[0]
			input_size  := weight.shape[1]
			count       := ml.len(input) / input_size

			input_data_ptr  := ([^]f32)(raw_data(_data(input)))
			output_grad_ptr := ([^]f32)(raw_data(_gradient(output)))
			weight_grad_ptr := ([^]f32)(raw_data(_gradient(weight)))

			w_grad := weight_grad_ptr[o * input_size:]

			for b in 0 ..< count {
				dout := output_grad_ptr[b * output_size + o]
				if dout == 0 {
					continue
				}
				x := input_data_ptr[b * input_size:]
				_simd_axpy_f32(w_grad, x, dout, input_size)
			}
		}, work=work)
	}

	if ml.has_gradient(op.input) {
		_parallelize(count, count, op, proc(b: int, op: ml.Operation) {
			input, output := op.input, op.output
			weight      := op.variant.(ml.Linear).weight
			output_size := weight.shape[0]
			input_size  := weight.shape[1]

			input_grad_ptr  := ([^]f32)(raw_data(_gradient(input)))
			output_grad_ptr := ([^]f32)(raw_data(_gradient(output)))
			weight_data_ptr := ([^]f32)(raw_data(_data(weight)))

			dx := input_grad_ptr [b * input_size:]
			dy := output_grad_ptr[b * output_size:]

			for o in 0 ..< output_size {
				dout := dy[o]
				if dout == 0 {
					continue
				}
				w_data := weight_data_ptr[o * input_size:]
				_simd_axpy_f32(dx, w_data, dout, input_size)
			}
		}, work=work)
	}
}


_batched_matmul_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _batched_matmul_forward_f32 (op)
	case .Bf16: _batched_matmul_forward_bf16(op)
	}
}

_batched_matmul_forward_bf16 :: proc(op: ml.Operation) {
	a           := op.input
	batch_count := a.shape[0]
	m           := a.shape[1]

	_parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		bt      := op.variant.(ml.Batched_Matmul).b
		m       := a.shape[1]
		k_count := a.shape[2]
		n       := bt.shape[2]

		bi := idx / m
		i  := idx % m

		a_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)bt.buffers    [.Data]))
		c_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))

		a_row := a_bf[bi * m * k_count + i * k_count:]
		c_row := c_bf[bi * m * n + i * n:]

		for j in 0 ..< n {
			acc: f32
			for kk in 0 ..< k_count {
				acc += ml.bf16_to_f32(a_row[kk]) *
					   ml.bf16_to_f32(b_bf[bi * k_count * n + kk * n + j])
			}
			c_row[j] = ml.bf16_from_f32(acc)
		}
	})
}

_batched_matmul_forward_f32 :: proc(op: ml.Operation) {
	a := op.input
	batch_count := a.shape[0]
	m := a.shape[1]

	_parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		bt      := op.variant.(ml.Batched_Matmul).b

		m        := a.shape[1]
		kk_count := a.shape[2]
		n        := bt.shape[2]

		bi := idx / m
		i  := idx % m

		a_ptr := ([^]f32)(raw_data(_data(a)))
		b_ptr := ([^]f32)(raw_data(_data(bt)))
		c_ptr := ([^]f32)(raw_data(_data(output)))

		a_row := a_ptr[bi * m * kk_count + i * kk_count:]
		c_row := c_ptr[bi * m * n + i * n:]

		for j in 0 ..< n {
			c_row[j] = 0
		}
		for kk in 0 ..< kk_count {
			b_row := b_ptr[bi * kk_count * n + kk * n:]
			_simd_axpy_f32(c_row, b_row, a_row[kk], n)
		}
	})
}

_batched_matmul_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _batched_matmul_backward_f32 (op)
	case .Bf16: _batched_matmul_backward_bf16(op)
	}
}

_batched_matmul_backward_bf16 :: proc(op: ml.Operation) {
	a           := op.input
	bt          := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m           := a.shape[1]
	k           := a.shape[2]

	if ml.has_gradient(a) {
		_parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b
			m       := a.shape[1]
			k_count := a.shape[2]
			n       := bt.shape[2]

			bi := idx / m
			i  := idx % m

			b_bf := _data_bf16(bt)
			dc   := _gradient(output)
			da   := _gradient(a)

			dc_row := dc[bi * m * n + i * n:]
			da_row := da[bi * m * k_count + i * k_count:]

			for kk in 0 ..< k_count {
				acc: f32
				for j in 0 ..< n {
					acc += dc_row[j] * ml.bf16_to_f32(b_bf[bi * k_count * n + kk * n + j])
				}
				da_row[kk] += acc
			}
		})
	}

	if ml.has_gradient(bt) {
		_parallelize(batch_count * k, batch_count * k, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b
			m       := a.shape[1]
			k_count := a.shape[2]
			n       := bt.shape[2]

			bi := idx / k_count
			kk := idx % k_count

			a_bf := _data_bf16(a)
			dc   := _gradient(output)
			db   := _gradient(bt)

			db_row := db[bi * k_count * n + kk * n:]
			for j in 0 ..< n {
				acc: f32
				for ii in 0 ..< m {
					acc += ml.bf16_to_f32(a_bf[bi * m * k_count + ii * k_count + kk]) *
						   dc[bi * m * n + ii * n + j]
				}
				db_row[j] += acc
			}
		})
	}
}

_batched_matmul_backward_f32 :: proc(op: ml.Operation) {
	a := op.input
	bt := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m := a.shape[1]
	k := a.shape[2]

	if ml.has_gradient(a) {
		_parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b

			m        := a.shape[1]
			kk_count := a.shape[2]
			n        := bt.shape[2]

			bi := idx / m
			i  := idx % m

			a_grad_ptr := ([^]f32)(raw_data(_gradient(a)))
			b_data_ptr := ([^]f32)(raw_data(_data(bt)))
			c_grad_ptr := ([^]f32)(raw_data(_gradient(output)))

			dc_row := c_grad_ptr[bi * m * n + i * n:]
			da_row := a_grad_ptr[bi * m * kk_count + i * kk_count:]

			for kk in 0 ..< kk_count {
				b_row := b_data_ptr[bi * kk_count * n + kk * n:]
				da_row[kk] += _simd_dot_f32(dc_row, b_row, n)
			}
		})
	}

	if ml.has_gradient(bt) {
		_parallelize(batch_count * k, batch_count * k, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b

			m        := a.shape[1]
			kk_count := a.shape[2]
			n        := bt.shape[2]

			bi := idx / kk_count
			kk := idx % kk_count

			a_data_ptr := ([^]f32)(raw_data(_data(a)))
			b_grad_ptr := ([^]f32)(raw_data(_gradient(bt)))
			c_grad_ptr := ([^]f32)(raw_data(_gradient(output)))

			db_row := b_grad_ptr[bi * kk_count * n + kk * n:]

			for ii in 0 ..< m {
				a_ik   := a_data_ptr[bi * m * kk_count + ii * kk_count + kk]
				dc_row := c_grad_ptr[bi * m * n + ii * n:]
				_simd_axpy_f32(db_row, dc_row, a_ik, n)
			}
		})
	}
}

