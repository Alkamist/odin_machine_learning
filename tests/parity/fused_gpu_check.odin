package ml_parity_tests

import "core:math"
import "core:testing"

import ml "../.."

FUSEDGPU_TOL       :: f64(2e-2)
FUSEDGPU_GEGLU_TOL :: f64(3e-2)

_fusedgpu_write_f16 :: proc(dst: []byte, v: f16) {
	bits := transmute(u16)v
	dst[0] = u8(bits & 0xff)
	dst[1] = u8(bits >> 8)
}

_fusedgpu_synth_q4k :: proc(dst: []byte, rows, blocks_per_row, salt: int) {
	for r in 0 ..< rows {
		for blk in 0 ..< blocks_per_row {
			base := (r * blocks_per_row + blk) * ml.Q4_K_BLOCK_BYTES
			d    := f16(0.008 + 0.002 * f32((r + blk + salt) % 5))
			dmin := f16(0.004 + 0.001 * f32((r + 2 * blk + salt) % 4))
			_fusedgpu_write_f16(dst[base + 0:], d)
			_fusedgpu_write_f16(dst[base + 2:], dmin)
			for i in 0 ..< 12 {
				dst[base + 4 + i] = u8((r * 7 + blk * 13 + i * 3 + salt) % 64)
			}
			for i in 0 ..< 128 {
				dst[base + 16 + i] = u8((r * 11 + blk * 17 + i * 5 + salt * 3) & 0xff)
			}
		}
	}
}

_fusedgpu_synth_q6k :: proc(dst: []byte, rows, blocks_per_row, salt: int) {
	for r in 0 ..< rows {
		for blk in 0 ..< blocks_per_row {
			base := (r * blocks_per_row + blk) * ml.Q6_K_BLOCK_BYTES
			for i in 0 ..< 128 {
				dst[base + i] = u8((r * 11 + blk * 7 + i * 5 + salt) & 0xff)
			}
			for i in 0 ..< 64 {
				dst[base + 128 + i] = u8((r * 13 + blk * 3 + i * 2 + salt) & 0xff)
			}
			for i in 0 ..< 16 {
				dst[base + 192 + i] = u8(i8((r * 3 + blk * 5 + i + salt) % 16 - 8))
			}
			_fusedgpu_write_f16(dst[base + 208:], f16(0.01 + 0.002 * f32((r + blk + salt) % 4)))
		}
	}
}

_fusedgpu_make_q4k :: proc(output_size, input_size, salt: int) -> ml.Tensor {
	blocks := input_size / ml.K_QUANT_BLOCK_SIZE
	bytes  := make([]byte, output_size * blocks * ml.Q4_K_BLOCK_BYTES)
	defer delete(bytes)
	_fusedgpu_synth_q4k(bytes, output_size, blocks, salt)
	w := ml.alloc(.Q4_K, {output_size, input_size}, persistent=false, buffers={.Data})
	ml.set_bytes(w, .Data, bytes)
	return w
}

_fusedgpu_make_q6k :: proc(output_size, input_size, salt: int) -> ml.Tensor {
	blocks := input_size / ml.K_QUANT_BLOCK_SIZE
	bytes  := make([]byte, output_size * blocks * ml.Q6_K_BLOCK_BYTES)
	defer delete(bytes)
	_fusedgpu_synth_q6k(bytes, output_size, blocks, salt)
	w := ml.alloc(.Q6_K, {output_size, input_size}, persistent=false, buffers={.Data})
	ml.set_bytes(w, .Data, bytes)
	return w
}

_fusedgpu_dequant_host :: proc(output_size, input_size, salt: int, q6: bool) -> []f32 {
	blocks := input_size / ml.K_QUANT_BLOCK_SIZE
	dst    := make([]f32, output_size * input_size)
	if q6 {
		bytes := make([]byte, output_size * blocks * ml.Q6_K_BLOCK_BYTES)
		defer delete(bytes)
		_fusedgpu_synth_q6k(bytes, output_size, blocks, salt)
		row_bytes := blocks * ml.Q6_K_BLOCK_BYTES
		for r in 0 ..< output_size {
			ml.dequantize_q6_k(bytes[r * row_bytes:][:row_bytes], dst[r * input_size:][:input_size])
		}
	} else {
		bytes := make([]byte, output_size * blocks * ml.Q4_K_BLOCK_BYTES)
		defer delete(bytes)
		_fusedgpu_synth_q4k(bytes, output_size, blocks, salt)
		row_bytes := blocks * ml.Q4_K_BLOCK_BYTES
		for r in 0 ..< output_size {
			ml.dequantize_q4_k(bytes[r * row_bytes:][:row_bytes], dst[r * input_size:][:input_size])
		}
	}
	return dst
}

_fusedgpu_reduce_and_backward :: proc(out: ml.Tensor, weights: []f32) {
	count := out.count
	out_f32 := out.type == .F32 ? out : ml.cast_to(out, .F32)
	sh := out_f32.shape
	w := ml.zeros(.F32, sh[:out_f32.rank])
	ml.set_data(w, weights[:count])
	weighted := ml.mul(out_f32, w)
	flat := ml.reshape(weighted, {1, count})
	ones := ml.scratch(.F32, {1, count})
	ml.fill_value(ones, 1.0 / f32(count))
	loss := ml.linear(flat, ones)
	ml.backward(loss)
}

_fusedgpu_check_attention_cache :: proc(t: ^testing.T, cuda_ctx: ^ml.Context) {
	ml.context_scope(cuda_ctx)

	n_q_heads := 2
	head_size := 4
	q_size    := n_q_heads * head_size
	kv_size   := q_size
	total     := 4
	prefill   := 2
	decode    := total - prefill

	q_full := make([]f32, total * q_size);  defer delete(q_full)
	k_full := make([]f32, total * kv_size); defer delete(k_full)
	v_full := make([]f32, total * kv_size); defer delete(v_full)
	for i in 0 ..< total * q_size {
		q_full[i] = 0.35 * math.sin(f32(i) * 0.29 + 0.1)
		k_full[i] = 0.35 * math.sin(f32(i) * 0.31 + 1.3)
		v_full[i] = 0.35 * math.sin(f32(i) * 0.27 + 2.5)
	}

	k_cache := ml.alloc(.Bf16, {total, kv_size}, persistent=true, buffers={.Data})
	v_cache := ml.alloc(.Bf16, {total, kv_size}, persistent=true, buffers={.Data})
	defer {
		ml.destroy(k_cache)
		ml.destroy(v_cache)
	}

	oracle := make([]f32, total * q_size); defer delete(oracle)
	ml.pass_begin(training=false)
	{
		q := _bf16p_make({total, q_size},  q_full)
		k := _bf16p_make({total, kv_size}, k_full)
		v := _bf16p_make({total, kv_size}, v_full)
		out := ml.attention(q, k, v, n_q_heads, n_kv_heads=n_q_heads, causal=true, window=0)
		_bf16p_read(out, oracle)
	}

	ml.pass_begin(training=false)
	{
		q := _bf16p_make({prefill, q_size},  q_full[:prefill * q_size])
		k := _bf16p_make({prefill, kv_size}, k_full[:prefill * kv_size])
		v := _bf16p_make({prefill, kv_size}, v_full[:prefill * kv_size])
		_ = ml.attention_with_cache(q, k, v, k_cache, v_cache, 0, n_q_heads, n_kv_heads=n_q_heads, window=0)
	}

	decoded := make([]f32, decode * q_size); defer delete(decoded)
	ml.pass_begin(training=false)
	{
		q := _bf16p_make({decode, q_size},  q_full[prefill * q_size:])
		k := _bf16p_make({decode, kv_size}, k_full[prefill * kv_size:])
		v := _bf16p_make({decode, kv_size}, v_full[prefill * kv_size:])
		out := ml.attention_with_cache(q, k, v, k_cache, v_cache, prefill, n_q_heads, n_kv_heads=n_q_heads, window=0)
		_bf16p_read(out, decoded)
	}

	_bf16p_compare(t, "attention_cache", "decode rows", oracle[prefill * q_size:], decoded, FUSEDGPU_TOL)
}

_fusedgpu_check_rmsnorm_rope_write_cache :: proc(t: ^testing.T, cuda_ctx: ^ml.Context) {
	ml.context_scope(cuda_ctx)

	head_count := 2
	head_size  := 4
	tokens     := 3
	trailing   := head_count * head_size
	n          := tokens * trailing

	x_src := make([]f32, n); defer delete(x_src)
	w_src := make([]f32, head_size); defer delete(w_src)
	for i in 0 ..< n {
		x_src[i] = 0.5 * math.sin(f32(i) * 0.33 + 0.2)
	}
	for i in 0 ..< head_size {
		w_src[i] = 0.8 + 0.2 * math.sin(f32(i) * 1.1)
	}

	oracle := make([]f32, n); defer delete(oracle)
	actual := make([]f32, n); defer delete(actual)

	saved := cuda_ctx.backend
	mod   := saved^
	mod.forward_ops -= {.Rmsnorm_Rope_Write_Cache, .Rmsnorm_Rope}
	cuda_ctx.backend = &mod
	ml.pass_begin(training=false)
	{
		x := _bf16p_make({tokens, trailing}, x_src)
		w := _bf16p_make({head_size}, w_src)
		out := ml.rmsnorm_rope(x, w, head_count, eps=1e-5, base=10000, position_offset=0, rope_fraction=1.0)
		_bf16p_read(out, oracle)
	}
	cuda_ctx.backend = saved

	ml.pass_begin(training=false)
	{
		x     := _bf16p_make({tokens, trailing}, x_src)
		w     := _bf16p_make({head_size}, w_src)
		cache := ml.alloc(.Bf16, {tokens, trailing}, persistent=false, buffers={.Data})
		_, _ = ml.rmsnorm_rope_write_cache(x, w, cache, tokens, head_count, eps=1e-5, base=10000, position_offset=0, rope_fraction=1.0)
		_bf16p_read(cache, actual)
	}

	_bf16p_compare(t, "rmsnorm_rope_write_cache", "cache", oracle, actual, FUSEDGPU_TOL)
}

_fusedgpu_check_gate_up_geglu :: proc(t: ^testing.T, cuda_ctx: ^ml.Context) {
	ml.context_scope(cuda_ctx)

	output_size := 8
	input_size  := ml.K_QUANT_BLOCK_SIZE

	x_src := make([]f32, input_size); defer delete(x_src)
	for i in 0 ..< input_size {
		x_src[i] = 0.4 * math.sin(f32(i) * 0.19 + 0.7)
	}

	oracle := make([]f32, output_size); defer delete(oracle)
	actual := make([]f32, output_size); defer delete(actual)

	saved := cuda_ctx.backend
	mod   := saved^
	mod.forward_ops -= {.Linear_Q4_K_Gate_Up_Geglu}
	cuda_ctx.backend = &mod
	ml.pass_begin(training=false)
	{
		x  := _bf16p_make({1, input_size}, x_src)
		wg := _fusedgpu_make_q4k(output_size, input_size, 1)
		wu := _fusedgpu_make_q4k(output_size, input_size, 9)
		out := ml.linear_q4_k_gate_up_geglu(x, wg, wu)
		_bf16p_read(out, oracle)
	}
	cuda_ctx.backend = saved

	ml.pass_begin(training=false)
	{
		x  := _bf16p_make({1, input_size}, x_src)
		wg := _fusedgpu_make_q4k(output_size, input_size, 1)
		wu := _fusedgpu_make_q4k(output_size, input_size, 9)
		out := ml.linear_q4_k_gate_up_geglu(x, wg, wu)
		_bf16p_read(out, actual)
	}

	_bf16p_compare(t, "linear_q4_k_gate_up_geglu", "output", oracle, actual, FUSEDGPU_GEGLU_TOL)
}

_fusedgpu_check_quant_backward_dx :: proc(t: ^testing.T, cuda_ctx: ^ml.Context, name: string, q6: bool) {
	ml.context_scope(cuda_ctx)

	output_size := 8
	input_size  := ml.K_QUANT_BLOCK_SIZE
	tokens      := 2
	salt        := 5

	x_src := make([]f32, tokens * input_size); defer delete(x_src)
	for i in 0 ..< tokens * input_size {
		x_src[i] = 0.4 * math.sin(f32(i) * 0.23 + 0.4)
	}
	weights := make([]f32, tokens * output_size); defer delete(weights)
	for i in 0 ..< tokens * output_size {
		weights[i] = math.cos(f32(i) * 0.61) + 0.5
	}

	dx_quant  := make([]f32, tokens * input_size); defer delete(dx_quant)
	dx_oracle := make([]f32, tokens * input_size); defer delete(dx_oracle)

	ml.pass_begin(training=true)
	{
		x   := _bf16p_make({tokens, input_size}, x_src)
		wq  := q6 ? _fusedgpu_make_q6k(output_size, input_size, salt) : _fusedgpu_make_q4k(output_size, input_size, salt)
		out := ml.linear(x, wq)
		_fusedgpu_reduce_and_backward(out, weights)
		_bf16p_read_grad(x, dx_quant)
	}

	deq := _fusedgpu_dequant_host(output_size, input_size, salt, q6); defer delete(deq)

	ml.pass_begin(training=true)
	{
		x   := _bf16p_make({tokens, input_size}, x_src)
		w   := _bf16p_make({output_size, input_size}, deq)
		out := ml.linear(x, w)
		_fusedgpu_reduce_and_backward(out, weights)
		_bf16p_read_grad(x, dx_oracle)
	}

	_bf16p_compare(t, name, "dx", dx_oracle, dx_quant, FUSEDGPU_TOL)
}
