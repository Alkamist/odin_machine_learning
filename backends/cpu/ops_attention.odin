package cpu

import "base:builtin"
import "base:runtime"
import "base:intrinsics"

import "core:math"

import ml "../.."

_causal_mask_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _causal_mask_forward_impl(f32,      op)
	case .Bf16: _causal_mask_forward_impl(ml.Bf16, op)
	}
}

_causal_mask_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input  := op.input
	output := op.output

	token_count := input.shape[input.rank - 1]
	block_size  := token_count * token_count
	n_blocks    := ml.len(input) / block_size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	neg_inf: T
	when T == ml.Bf16 {
		neg_inf = ml.bf16_from_f32(math.NEG_INF_F32)
	} else {
		neg_inf = math.NEG_INF_F32
	}

	for blk in 0 ..< n_blocks {
		offset := blk * block_size
		for t1 in 0 ..< token_count {
			for t2 in 0 ..< token_count {
				idx := offset + t1 * token_count + t2
				if t2 <= t1 {
					yp[idx] = xp[idx]
				} else {
					yp[idx] = neg_inf
				}
			}
		}
	}
}

_causal_mask_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := ml.len(input) / block_size

	dx, dy := _gradient(input), _gradient(output)
	for blk in 0 ..< n_blocks {
		offset := blk * block_size
		for t1 in 0 ..< T {
			for t2 in 0 ..= t1 {
				idx := offset + t1 * T + t2
				dx[idx] += dy[idx]
			}
		}
	}
}

_attention_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _attention_forward_f32 (op)
	case .Bf16: _attention_forward_bf16(op)
	}
}

_attention_forward_f32 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	_parallelize(v.n_q_heads, v.n_q_heads, op, proc(h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]f32)(raw_data(_data(op.input)))
		k_ptr   := ([^]f32)(raw_data(_data(v.key)))
		v_ptr   := ([^]f32)(raw_data(_data(v.value)))
		out_ptr := ([^]f32)(raw_data(_data(op.output)))
		sm_ptr  := ([^]f32)(raw_data(_data(v.softmax_outputs)))

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size
			q := q_ptr[q_offset:]

			sm_row_offset := h * token_count * token_count + t_q * token_count
			sm_row := sm_ptr[sm_row_offset:]

			t_k_max := token_count
			if causal { t_k_max = t_q + 1 }
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := t_k * kv_size + kv_h * head_size
				score := _simd_dot_f32(q, k_ptr[k_offset:], head_size) * inv_sqrt_d
				sm_row[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(sm_row[t_k] - max_score)
				sm_row[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp
			for t_k in t_k_min ..< t_k_max {
				sm_row[t_k] *= inv_sum
			}
			for t_k in 0 ..< t_k_min {
				sm_row[t_k] = 0
			}
			for t_k in t_k_max ..< token_count {
				sm_row[t_k] = 0
			}

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				out_ptr[out_offset + d] = 0
			}
			for t_k in t_k_min ..< t_k_max {
				v_offset := t_k * kv_size + kv_h * head_size
				_simd_axpy_f32(out_ptr[out_offset:], v_ptr[v_offset:], sm_row[t_k], head_size)
			}
		}
	})
}

_attention_forward_bf16 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	_parallelize(v.n_q_heads, v.n_q_heads, op, proc(h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers[.Data]))
		k_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.key.buffers   [.Data]))
		v_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.value.buffers [.Data]))
		out_ptr := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Data]))
		sm_ptr  := ([^]f32)(raw_data(_data(v.softmax_outputs)))

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size

			sm_row_offset := h * token_count * token_count + t_q * token_count
			sm_row := sm_ptr[sm_row_offset:]

			t_k_max := token_count
			if causal { t_k_max = t_q + 1 }
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := t_k * kv_size + kv_h * head_size
				score: f32
				for d in 0 ..< head_size {
					score += ml.bf16_to_f32(q_ptr[q_offset + d]) * ml.bf16_to_f32(k_ptr[k_offset + d])
				}
				score *= inv_sqrt_d
				sm_row[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(sm_row[t_k] - max_score)
				sm_row[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp
			for t_k in t_k_min ..< t_k_max {
				sm_row[t_k] *= inv_sum
			}
			for t_k in 0 ..< t_k_min {
				sm_row[t_k] = 0
			}
			for t_k in t_k_max ..< token_count {
				sm_row[t_k] = 0
			}

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				acc: f32
				for t_k in t_k_min ..< t_k_max {
					v_offset := t_k * kv_size + kv_h * head_size
					acc += sm_row[t_k] * ml.bf16_to_f32(v_ptr[v_offset + d])
				}
				out_ptr[out_offset + d] = ml.bf16_from_f32(acc)
			}
		}
	})
}

_attention_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _attention_backward_f32 (op)
	case .Bf16: _attention_backward_bf16(op)
	}
}

_attention_backward_f32 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	_parallelize(v.n_kv_heads, v.n_kv_heads, op, proc(kv_h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_data    := ([^]f32)(raw_data(_data(op.input)))
		q_grad    := ([^]f32)(raw_data(_gradient(op.input)))
		k_data    := ([^]f32)(raw_data(_data(v.key)))
		k_grad    := ([^]f32)(raw_data(_gradient(v.key)))
		v_data    := ([^]f32)(raw_data(_data(v.value)))
		v_grad    := ([^]f32)(raw_data(_gradient(v.value)))
		out_grad  := ([^]f32)(raw_data(_gradient(op.output)))
		sm_ptr    := ([^]f32)(raw_data(_data(v.softmax_outputs)))
		dp_ptr    := ([^]f32)(raw_data(_data(v.d_p_scratch)))

		have_dq := ml.has_gradient(op.input)
		have_dk := ml.has_gradient(v.key)
		have_dv := ml.has_gradient(v.value)

		for q_h_off in 0 ..< group_size {
			h := kv_h * group_size + q_h_off

			dp_base := h * token_count
			d_p     := dp_ptr[dp_base:]

			for t_q in 0 ..< token_count {
				t_k_max := token_count
				if causal { t_k_max = t_q + 1 }
				t_k_min := 0
				if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

				d_out_offset := t_q * q_size + h * head_size
				d_out := out_grad[d_out_offset:]

				sm_row_offset := h * token_count * token_count + t_q * token_count
				sm_row := sm_ptr[sm_row_offset:]

				for t_k in t_k_min ..< t_k_max {
					v_offset := t_k * kv_size + kv_h * head_size
					d_p[t_k] = _simd_dot_f32(d_out, v_data[v_offset:], head_size)
					if have_dv { _simd_axpy_f32(v_grad[v_offset:], d_out, sm_row[t_k], head_size) }
				}

				dot_dp_p: f32
				for t_k in t_k_min ..< t_k_max {
					dot_dp_p += d_p[t_k] * sm_row[t_k]
				}
				for t_k in t_k_min ..< t_k_max {
					d_p[t_k] = sm_row[t_k] * (d_p[t_k] - dot_dp_p) * inv_sqrt_d
				}

				q_offset := t_q * q_size + h * head_size
				d_q_vec  := q_grad[q_offset:]
				q_vec    := q_data[q_offset:]

				for t_k in t_k_min ..< t_k_max {
					k_offset := t_k * kv_size + kv_h * head_size
					if have_dq { _simd_axpy_f32(d_q_vec, k_data[k_offset:], d_p[t_k], head_size) }
					if have_dk { _simd_axpy_f32(k_grad[k_offset:], q_vec,   d_p[t_k], head_size) }
				}
			}
		}
	})
}

_attention_backward_bf16 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	_parallelize(v.n_kv_heads, v.n_kv_heads, op, proc(kv_h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_data   := _data_bf16(op.input)
		q_grad   := _gradient(op.input)
		k_data   := _data_bf16(v.key)
		k_grad   := _gradient(v.key)
		v_data   := _data_bf16(v.value)
		v_grad   := _gradient(v.value)
		out_grad := _gradient(op.output)
		sm_ptr   := ([^]f32)(raw_data(_data(v.softmax_outputs)))
		dp_ptr   := ([^]f32)(raw_data(_data(v.d_p_scratch)))

		have_dq := ml.has_gradient(op.input)
		have_dk := ml.has_gradient(v.key)
		have_dv := ml.has_gradient(v.value)

		for q_h_off in 0 ..< group_size {
			h := kv_h * group_size + q_h_off

			dp_base := h * token_count
			d_p     := dp_ptr[dp_base:]

			for t_q in 0 ..< token_count {
				t_k_max := token_count
				if causal { t_k_max = t_q + 1 }
				t_k_min := 0
				if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

				d_out_offset := t_q * q_size + h * head_size
				sm_row_offset := h * token_count * token_count + t_q * token_count
				sm_row := sm_ptr[sm_row_offset:]

				for t_k in t_k_min ..< t_k_max {
					v_offset := t_k * kv_size + kv_h * head_size
					dot: f32
					for d in 0 ..< head_size {
						dot += out_grad[d_out_offset + d] * ml.bf16_to_f32(v_data[v_offset + d])
					}
					d_p[t_k] = dot

					if have_dv {
						p_val := sm_row[t_k]
						for d in 0 ..< head_size {
							v_grad[v_offset + d] += out_grad[d_out_offset + d] * p_val
						}
					}
				}

				dot_dp_p: f32
				for t_k in t_k_min ..< t_k_max {
					dot_dp_p += d_p[t_k] * sm_row[t_k]
				}
				for t_k in t_k_min ..< t_k_max {
					d_p[t_k] = sm_row[t_k] * (d_p[t_k] - dot_dp_p) * inv_sqrt_d
				}

				q_offset := t_q * q_size + h * head_size

				for t_k in t_k_min ..< t_k_max {
					k_offset := t_k * kv_size + kv_h * head_size
					scale    := d_p[t_k]
					for d in 0 ..< head_size {
						q_d := ml.bf16_to_f32(q_data[q_offset + d])
						k_d := ml.bf16_to_f32(k_data[k_offset + d])
						if have_dq { q_grad[q_offset + d] += scale * k_d }
						if have_dk { k_grad[k_offset + d] += scale * q_d }
					}
				}
			}
		}
	})
}

_attention_cache_forward :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention_Cache)

	token_count := op.input.shape[0]
	kv_size     := v.key.shape[1]
	row_bytes   := kv_size * ml.data_type_size(v.key.type)
	cache_pos   := v.cache_position
	t_capacity  := v.k_cache.shape[0]

	k_new_bytes   := transmute([]byte)v.key.buffers    [.Data]
	v_new_bytes   := transmute([]byte)v.value.buffers  [.Data]
	k_cache_bytes := transmute([]byte)v.k_cache.buffers[.Data]
	v_cache_bytes := transmute([]byte)v.v_cache.buffers[.Data]

	first_phys := cache_pos % t_capacity
	first_count := token_count
	if first_phys + first_count > t_capacity { first_count = t_capacity - first_phys }
	first_bytes := first_count * row_bytes
	dst0 := first_phys * row_bytes
	if !v.k_cached {
		builtin.copy(k_cache_bytes[dst0:dst0 + first_bytes], k_new_bytes[:first_bytes])
	}
	if !v.v_cached {
		builtin.copy(v_cache_bytes[dst0:dst0 + first_bytes], v_new_bytes[:first_bytes])
	}

	if first_count < token_count {
		wrap_bytes := (token_count - first_count) * row_bytes
		if !v.k_cached {
			builtin.copy(k_cache_bytes[:wrap_bytes], k_new_bytes[first_bytes:first_bytes + wrap_bytes])
		}
		if !v.v_cached {
			builtin.copy(v_cache_bytes[:wrap_bytes], v_new_bytes[first_bytes:first_bytes + wrap_bytes])
		}
	}

	#partial switch op.input.type {
	case .F32:  _attention_cache_forward_f32 (op)
	case .Bf16: _attention_cache_forward_bf16(op)
	}
}

_attention_cache_forward_f32 :: proc(op: ml.Operation) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	v := op.variant.(ml.Attention_Cache)

	token_count := op.input.shape[0]
	k_total     := v.cache_position + token_count

	Job_Data :: struct {
		op:      ml.Operation,
		scratch: []f32,
		k_total: int,
	}
	jd := Job_Data{
		op      = op,
		scratch = make([]f32, v.n_q_heads * k_total, context.temp_allocator),
		k_total = k_total,
	}

	_parallelize(v.n_q_heads, v.n_q_heads, jd, proc(h: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Attention_Cache)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		cache_pos   := v.cache_position
		window      := v.window
		t_capacity  := v.k_cache.shape[0]
		k_total     := jd.k_total
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]f32)(raw_data(_data(op.input)))
		k_ptr   := ([^]f32)(raw_data(_data(v.k_cache)))
		v_ptr   := ([^]f32)(raw_data(_data(v.v_cache)))
		out_ptr := ([^]f32)(raw_data(_data(op.output)))

		scores := jd.scratch[h * k_total : (h + 1) * k_total]

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size
			q := q_ptr[q_offset:]

			t_k_max := cache_pos + t_q + 1
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
				score := _simd_dot_f32(q, k_ptr[k_offset:], head_size) * inv_sqrt_d
				scores[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(scores[t_k] - max_score)
				scores[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				out_ptr[out_offset + d] = 0
			}
			for t_k in t_k_min ..< t_k_max {
				v_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
				_simd_axpy_f32(out_ptr[out_offset:], v_ptr[v_offset:], scores[t_k] * inv_sum, head_size)
			}
		}
	})
}

_attention_cache_forward_bf16 :: proc(op: ml.Operation) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	v := op.variant.(ml.Attention_Cache)

	token_count := op.input.shape[0]
	k_total     := v.cache_position + token_count

	Job_Data :: struct {
		op:      ml.Operation,
		scratch: []f32,
		k_total: int,
	}
	jd := Job_Data{
		op      = op,
		scratch = make([]f32, v.n_q_heads * k_total, context.temp_allocator),
		k_total = k_total,
	}

	_parallelize(v.n_q_heads, v.n_q_heads, jd, proc(h: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Attention_Cache)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		cache_pos   := v.cache_position
		window      := v.window
		t_capacity  := v.k_cache.shape[0]
		k_total     := jd.k_total
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers  [.Data]))
		k_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.k_cache.buffers [.Data]))
		v_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.v_cache.buffers [.Data]))
		out_ptr := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers [.Data]))

		scores := jd.scratch[h * k_total : (h + 1) * k_total]

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size

			t_k_max := cache_pos + t_q + 1
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
				score := _simd_dot_bf16_f32(q_ptr[q_offset:], k_ptr[k_offset:], head_size) * inv_sqrt_d
				scores[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(scores[t_k] - max_score)
				scores[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				acc: f32
				for t_k in t_k_min ..< t_k_max {
					v_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
					acc += scores[t_k] * inv_sum * ml.bf16_to_f32(v_ptr[v_offset + d])
				}
				out_ptr[out_offset + d] = ml.bf16_from_f32(acc)
			}
		}
	})
}

_attention_cache_backward :: proc(op: ml.Operation, loc := #caller_location) {
	panic("Attention_Cache is _forward-only", loc)
}
