package ml_tests

import "core:math"
import "core:mem"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

FUSED_CPU_CTX_SIZE :: 16 * 1024 * 1024

FUSED_BF16_TOL   :: f64(3e-2)
FUSED_QUANT_TOL  :: f64(2e-2)
FUSED_EXACT_TOL  :: f64(1e-5)
FUSED_REL_FLOOR  :: f64(1e-2)

_fused_fill :: proc(dst: []f32, phase: f32) {
	for i in 0 ..< len(dst) {
		dst[i] = 0.8 * math.sin(f32(i) * 0.37 + phase)
	}
}

_fused_make_bf16 :: proc(shape: []int, src: []f32) -> ml.Tensor {
	t   := ml.zeros(.Bf16, shape)
	buf := make([]ml.Bf16, len(src))
	defer delete(buf)
	for v, i in src {
		buf[i] = ml.bf16_from_f32(v)
	}
	ml.set_data_bytes(t, mem.slice_to_bytes(buf))
	return t
}

_fused_read_bf16 :: proc(t: ml.Tensor, dst: []f32) {
	buf := make([]ml.Bf16, t.count)
	defer delete(buf)
	ml.get_data_bytes(t, mem.slice_to_bytes(buf))
	for v, i in buf {
		dst[i] = ml.bf16_to_f32(v)
	}
}

_fused_round_bf16 :: proc(dst, src: []f32) {
	for v, i in src {
		dst[i] = ml.bf16_to_f32(ml.bf16_from_f32(v))
	}
}

_fused_compare :: proc(t: ^testing.T, name: string, oracle, actual: []f32, tol: f64) {
	for i in 0 ..< len(oracle) {
		a     := f64(oracle[i])
		b     := f64(actual[i])
		denom := max(max(abs(a), abs(b)), FUSED_REL_FLOOR)
		rel   := abs(a - b) / denom
		testing.expectf(t, rel <= tol,
			"%s elem %d oracle=%.6g actual=%.6g rel_err=%.4g (tol=%.3g)",
			name, i, a, b, rel, tol)
	}
}

@(test)
test_fused_gelu_mul :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	shape := []int{3, 8}
	n     := 24
	a_src := make([]f32, n); defer delete(a_src)
	b_src := make([]f32, n); defer delete(b_src)
	_fused_fill(a_src, 0.1)
	_fused_fill(b_src, 1.7)

	oracle := make([]f32, n); defer delete(oracle)
	actual := make([]f32, n); defer delete(actual)

	saved := ctx.backend
	mod   := saved^
	mod.forward_ops -= {.Gelu_Mul}
	ctx.backend = &mod
	ml.clear(training=false)
	{
		a   := _fused_make_bf16(shape, a_src)
		b   := _fused_make_bf16(shape, b_src)
		out := ml.gelu_mul(a, b)
		_fused_read_bf16(out, oracle)
	}
	ctx.backend = saved

	ml.clear(training=false)
	{
		a   := _fused_make_bf16(shape, a_src)
		b   := _fused_make_bf16(shape, b_src)
		out := ml.gelu_mul(a, b)
		_fused_read_bf16(out, actual)
	}

	_fused_compare(t, "gelu_mul", oracle, actual, FUSED_BF16_TOL)
}

@(test)
test_fused_add_rmsnorm :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	shape  := []int{3, 8}
	wshape := []int{8}
	n      := 24
	a_src := make([]f32, n); defer delete(a_src)
	b_src := make([]f32, n); defer delete(b_src)
	w_src := make([]f32, 8); defer delete(w_src)
	_fused_fill(a_src, 0.3)
	_fused_fill(b_src, 2.1)
	_fused_fill(w_src, 0.9)

	oracle_res := make([]f32, n); defer delete(oracle_res)
	oracle_nrm := make([]f32, n); defer delete(oracle_nrm)
	actual_res := make([]f32, n); defer delete(actual_res)
	actual_nrm := make([]f32, n); defer delete(actual_nrm)

	saved := ctx.backend
	mod   := saved^
	mod.forward_ops -= {.Add_Rmsnorm}
	ctx.backend = &mod
	ml.clear(training=false)
	{
		a := _fused_make_bf16(shape, a_src)
		b := _fused_make_bf16(shape, b_src)
		w := _fused_make_bf16(wshape, w_src)
		res, nrm := ml.add_rmsnorm(a, b, w)
		_fused_read_bf16(res, oracle_res)
		_fused_read_bf16(nrm, oracle_nrm)
	}
	ctx.backend = saved

	ml.clear(training=false)
	{
		a := _fused_make_bf16(shape, a_src)
		b := _fused_make_bf16(shape, b_src)
		w := _fused_make_bf16(wshape, w_src)
		res, nrm := ml.add_rmsnorm(a, b, w)
		_fused_read_bf16(res, actual_res)
		_fused_read_bf16(nrm, actual_nrm)
	}

	_fused_compare(t, "add_rmsnorm residual", oracle_res, actual_res, FUSED_BF16_TOL)
	_fused_compare(t, "add_rmsnorm normed",   oracle_nrm, actual_nrm, FUSED_BF16_TOL)
}

@(test)
test_fused_rmsnorm_rope :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	head_count := 2
	head_size  := 4
	tokens     := 3
	trailing   := head_count * head_size
	n          := tokens * trailing

	shape  := []int{tokens, trailing}
	wshape := []int{head_size}
	x_src := make([]f32, n); defer delete(x_src)
	w_src := make([]f32, head_size); defer delete(w_src)
	_fused_fill(x_src, 0.5)
	_fused_fill(w_src, 1.1)

	oracle := make([]f32, n); defer delete(oracle)
	actual := make([]f32, n); defer delete(actual)

	saved := ctx.backend
	mod   := saved^
	mod.forward_ops -= {.Rmsnorm_Rope}
	ctx.backend = &mod
	ml.clear(training=false)
	{
		x   := _fused_make_bf16(shape, x_src)
		w   := _fused_make_bf16(wshape, w_src)
		out := ml.rmsnorm_rope(x, w, head_count, 1e-5, 10000, 0, 1.0)
		_fused_read_bf16(out, oracle)
	}
	ctx.backend = saved

	ml.clear(training=false)
	{
		x   := _fused_make_bf16(shape, x_src)
		w   := _fused_make_bf16(wshape, w_src)
		out := ml.rmsnorm_rope(x, w, head_count, 1e-5, 10000, 0, 1.0)
		_fused_read_bf16(out, actual)
	}

	_fused_compare(t, "rmsnorm_rope", oracle, actual, FUSED_BF16_TOL)
}

@(test)
test_fused_cast_round_trip :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	shape := []int{4, 5}
	n     := 20
	src := make([]f32, n); defer delete(src)
	for i in 0 ..< n {
		src[i] = math.sin(f32(i) * 1.31) * 3.0 + 0.123
	}

	ml.clear(training=false)
	x    := ml.tensor(src, shape)
	xbf  := ml.cast_to(x, .Bf16)
	back := ml.cast_to(xbf, .F32)

	actual := make([]f32, n); defer delete(actual)
	ml.get_data(back, actual)

	oracle := make([]f32, n); defer delete(oracle)
	_fused_round_bf16(oracle, src)

	for i in 0 ..< n {
		testing.expectf(t, actual[i] == oracle[i],
			"cast round trip elem %d actual=%v oracle=%v", i, actual[i], oracle[i])
	}
}

@(test)
test_fused_lerp_assign :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	shape := []int{2, 3}
	n     := 6
	src_vals := make([]f32, n); defer delete(src_vals)
	host     := make([]f32, n); defer delete(host)
	_fused_fill(src_vals, 0.4)
	for i in 0 ..< n {
		host[i] = 0.2 * f32(i) - 0.5
	}

	ml.clear(training=false)
	dst := ml.zeros(.F32, shape)
	ml.set_data(dst, host)
	source := ml.zeros(.F32, shape)
	ml.set_data(source, src_vals)

	got := make([]f32, n); defer delete(got)
	alphas := []f32{0.25, 0.5, 0.8}
	for alpha in alphas {
		ml.lerp_assign(dst, source, alpha)
		for i in 0 ..< n {
			host[i] = (1 - alpha) * host[i] + alpha * src_vals[i]
		}
		ml.get_data(dst, got)
		_fused_compare(t, "lerp_assign", host, got, FUSED_EXACT_TOL)
	}
}

@(test)
test_fused_accumulate_mean :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	shape := []int{3, 4}
	n     := 12
	running: f32 = 0

	ml.clear(training=false)
	scalar_shape := []int{1}
	dst := ml.zeros(.F32, scalar_shape)
	zero := []f32{0}
	ml.set_data(dst, zero)

	src_vals := make([]f32, n); defer delete(src_vals)
	source   := ml.zeros(.F32, shape)

	got: [1]f32
	for iter in 0 ..< 3 {
		_fused_fill(src_vals, f32(iter) * 0.9 + 0.2)
		ml.set_data(source, src_vals)

		sum: f32 = 0
		for v in src_vals {
			sum += v
		}
		running += sum / f32(n)

		ml.accumulate_mean(dst, source)
		ml.get_data(dst, got[:])

		testing.expectf(t, abs(f64(got[0]) - f64(running)) <= 1e-5,
			"accumulate_mean iter %d got=%v want=%v", iter, got[0], running)
	}
}

_fused_write_f16 :: proc(dst: []byte, v: f16) {
	bits := transmute(u16)v
	dst[0] = u8(bits & 0xff)
	dst[1] = u8(bits >> 8)
}

_fused_synth_q4k :: proc(dst: []byte, rows, blocks_per_row: int) {
	for r in 0 ..< rows {
		for blk in 0 ..< blocks_per_row {
			base := (r * blocks_per_row + blk) * ml.Q4_K_BLOCK_BYTES
			d    := f16(0.008 + 0.002 * f32((r + blk) % 5))
			dmin := f16(0.004 + 0.001 * f32((r + 2 * blk) % 4))
			_fused_write_f16(dst[base + 0:], d)
			_fused_write_f16(dst[base + 2:], dmin)
			for i in 0 ..< 12 {
				dst[base + 4 + i] = u8((r * 7 + blk * 13 + i * 3) % 64)
			}
			for i in 0 ..< 128 {
				dst[base + 16 + i] = u8((r * 11 + blk * 17 + i * 5) & 0xff)
			}
		}
	}
}

_fused_synth_q6k :: proc(dst: []byte, rows, blocks_per_row: int) {
	for r in 0 ..< rows {
		for blk in 0 ..< blocks_per_row {
			base := (r * blocks_per_row + blk) * ml.Q6_K_BLOCK_BYTES
			for i in 0 ..< 128 {
				dst[base + i] = u8((r * 11 + blk * 7 + i * 5) & 0xff)
			}
			for i in 0 ..< 64 {
				dst[base + 128 + i] = u8((r * 13 + blk * 3 + i * 2) & 0xff)
			}
			for i in 0 ..< 16 {
				dst[base + 192 + i] = u8(i8((r * 3 + blk * 5 + i) % 16 - 8))
			}
			_fused_write_f16(dst[base + 208:], f16(0.01 + 0.002 * f32((r + blk) % 4)))
		}
	}
}

@(test)
test_fused_linear_q4_k :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	output_size := 8
	input_size  := ml.K_QUANT_BLOCK_SIZE
	tokens      := 3
	blocks      := input_size / ml.K_QUANT_BLOCK_SIZE

	ml.clear(training=false)

	weight_bytes := make([]byte, output_size * blocks * ml.Q4_K_BLOCK_BYTES)
	defer delete(weight_bytes)
	_fused_synth_q4k(weight_bytes, output_size, blocks)

	w_q := ml.zeros(.Q4_K, {output_size, input_size})
	ml.set_data_bytes(w_q, weight_bytes)

	w_f32_host := make([]f32, output_size * input_size); defer delete(w_f32_host)
	row_bytes  := blocks * ml.Q4_K_BLOCK_BYTES
	for r in 0 ..< output_size {
		ml.dequantize_q4_k(weight_bytes[r * row_bytes:][:row_bytes], w_f32_host[r * input_size:][:input_size])
	}
	w_f32 := ml.zeros(.F32, {output_size, input_size})
	ml.set_data(w_f32, w_f32_host)

	x_src := make([]f32, tokens * input_size); defer delete(x_src)
	_fused_fill(x_src, 0.6)
	x_round := make([]f32, tokens * input_size); defer delete(x_round)
	_fused_round_bf16(x_round, x_src)

	x_bf  := _fused_make_bf16({tokens, input_size}, x_src)
	x_f32 := ml.zeros(.F32, {tokens, input_size})
	ml.set_data(x_f32, x_round)

	fused_out  := ml.linear_q4_k(x_bf, w_q)
	oracle_out := ml.linear(x_f32, w_f32)

	fused  := make([]f32, tokens * output_size); defer delete(fused)
	oracle := make([]f32, tokens * output_size); defer delete(oracle)
	_fused_read_bf16(fused_out, fused)
	ml.get_data(oracle_out, oracle)

	_fused_compare(t, "linear_q4_k", oracle, fused, FUSED_QUANT_TOL)
}

@(test)
test_fused_linear_q6_k :: proc(t: ^testing.T) {
	ctx := cpu.context_create(FUSED_CPU_CTX_SIZE)
	ml.context_begin(ctx)
	defer {
		ml.context_end()
		cpu.context_destroy(ctx)
	}

	output_size := 8
	input_size  := ml.K_QUANT_BLOCK_SIZE
	tokens      := 3
	blocks      := input_size / ml.K_QUANT_BLOCK_SIZE

	ml.clear(training=false)

	weight_bytes := make([]byte, output_size * blocks * ml.Q6_K_BLOCK_BYTES)
	defer delete(weight_bytes)
	_fused_synth_q6k(weight_bytes, output_size, blocks)

	w_q := ml.zeros(.Q6_K, {output_size, input_size})
	ml.set_data_bytes(w_q, weight_bytes)

	w_f32_host := make([]f32, output_size * input_size); defer delete(w_f32_host)
	row_bytes  := blocks * ml.Q6_K_BLOCK_BYTES
	for r in 0 ..< output_size {
		ml.dequantize_q6_k(weight_bytes[r * row_bytes:][:row_bytes], w_f32_host[r * input_size:][:input_size])
	}
	w_f32 := ml.zeros(.F32, {output_size, input_size})
	ml.set_data(w_f32, w_f32_host)

	x_src := make([]f32, tokens * input_size); defer delete(x_src)
	_fused_fill(x_src, 0.35)
	x_round := make([]f32, tokens * input_size); defer delete(x_round)
	_fused_round_bf16(x_round, x_src)

	x_bf  := _fused_make_bf16({tokens, input_size}, x_src)
	x_f32 := ml.zeros(.F32, {tokens, input_size})
	ml.set_data(x_f32, x_round)

	fused_out  := ml.linear_q6_k(x_bf, w_q)
	oracle_out := ml.linear(x_f32, w_f32)

	fused  := make([]f32, tokens * output_size); defer delete(fused)
	oracle := make([]f32, tokens * output_size); defer delete(oracle)
	_fused_read_bf16(fused_out, fused)
	ml.get_data(oracle_out, oracle)

	_fused_compare(t, "linear_q6_k", oracle, fused, FUSED_QUANT_TOL)
}
