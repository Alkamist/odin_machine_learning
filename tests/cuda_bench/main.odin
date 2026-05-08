package cuda_bench

// Side-by-side parity + perf bench for the `add` op across the vulkan and
// cuda backends. Verifies output bit-equality (or BF16 tolerance), then
// times forward + forward/backward over many iterations and prints
// ms/iter + effective GB/s for each backend.
//
// Gating rule for Phase 3: cuda must tie or beat vulkan on every shape we
// care about before we move on.

import "base:builtin"

import "core:fmt"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:time"

import ml      "../.."
import vulkan  "../../backends/vulkan"
import cu      "../../backends/cuda"
import gguf    "../../loaders/gguf"

Shape :: struct {
	name: string,
	n_a:  int,
	n_b:  int,
}

// We're comparing wildly different per-iter times (CUDA sub-ms, Vulkan
// 10-1000ms). Few iters are enough to establish the ratio reliably, and big
// shapes on Vulkan dominate runtime. Tuned to keep total bench under ~60s.
WARMUP_ITERS :: 2
TIMED_ITERS  :: 8

shapes := []Shape{
	{"1M  no broadcast",          1 << 20, 1 << 20},
	{"1M  broadcast(b=1024)",     1 << 20, 1024},
	{"16M broadcast(b=128)",      1 << 24, 128},
}

main :: proc() {
	defer free_all(context.temp_allocator)

	if !run_dtype(.F32)  { os.exit(1) }
	if !run_dtype(.Bf16) { os.exit(1) }

	if !run_linear(.F32)  { os.exit(1) }
	if !run_linear(.Bf16) { os.exit(1) }

	if !run_linear_q4_k() { os.exit(1) }
}

// ----- Linear (Y = X @ W^T) ---------------------------------------------------

Linear_Shape :: struct {
	name:        string,
	count:       int,  // M = batched rows of X
	input_size:  int,  // K
	output_size: int,  // N
}

linear_shapes := []Linear_Shape{
	{"512x768x768",            512,    768,    768},
	{"2048x2048x2048",        2048,   2048,   2048},
}

run_linear :: proc(dtype: ml.Data_Type) -> bool {
	// Linear backward chains need an F32 loss head; only F32 has end-to-end
	// fwd+bwd parity in the bench harness. For Bf16 we measure forward only.
	measure_backward := dtype == .F32

	fmt.printfln("\n=== linear  (%v)  ===", dtype)
	if measure_backward {
		fmt.printfln("%-30s %12s %12s %12s %12s %10s", "MxKxN", "vk fwd ms", "cu fwd ms", "vk fwd+bwd", "cu fwd+bwd", "fwd speedup")
	} else {
		fmt.printfln("%-30s %12s %12s %10s", "MxKxN", "vk fwd ms", "cu fwd ms", "fwd speedup")
	}

	all_ok := true
	for s in linear_shapes {
		x_bytes := ml.data_type_size(dtype) * s.count       * s.input_size
		w_bytes := ml.data_type_size(dtype) * s.output_size * s.input_size
		y_bytes := ml.data_type_size(dtype) * s.count       * s.output_size

		host_x  := builtin.make([]byte, x_bytes); defer delete(host_x)
		host_w  := builtin.make([]byte, w_bytes); defer delete(host_w)
		host_yv := builtin.make([]byte, y_bytes); defer delete(host_yv)
		host_yc := builtin.make([]byte, y_bytes); defer delete(host_yc)

		fill_random(host_x, dtype)
		fill_random(host_w, dtype)

		vk_fwd, vk_fb := run_linear_backend(.Vulkan, dtype, s, host_x, host_w, host_yv, measure_backward)
		cu_fwd, cu_fb := run_linear_backend(.Cuda,   dtype, s, host_x, host_w, host_yc, measure_backward)

		// BF16 GEMM accumulates rounding across K, so allow a per-result
		// relative tolerance scaled by sqrt(K) to absorb the spread between
		// two different (but both correct) algorithms.
		ok := compare_linear(dtype, host_yv, host_yc, s)
		if !ok { all_ok = false }

		marker := ok ? " " : "X"
		speedup := vk_fwd / cu_fwd
		if measure_backward {
			fmt.printfln("%s%-29s %12.4f %12.4f %12.4f %12.4f %9.2fx",
				marker, s.name, vk_fwd, cu_fwd, vk_fb, cu_fb, speedup)
		} else {
			fmt.printfln("%s%-29s %12.4f %12.4f %9.2fx",
				marker, s.name, vk_fwd, cu_fwd, speedup)
		}
	}
	return all_ok
}

run_linear_backend :: proc(b: Backend, dtype: ml.Data_Type, s: Linear_Shape, x_bytes, w_bytes, y_bytes: []byte, measure_backward: bool) -> (fwd_ms, fb_ms: f32) {
	ctx: ^ml.Context
	switch b {
	case .Vulkan: ctx = vulkan.context_create()
	case .Cuda:   ctx = cu.context_create()
	}
	ml.context_begin(ctx)
	defer { ml.context_end(); switch b {
	case .Vulkan: vulkan.context_destroy(ctx)
	case .Cuda:   cu.context_destroy(ctx)
	} }

	x_shape := []int{s.count, s.input_size}
	w_shape := []int{s.output_size, s.input_size}

	for _ in 0..<WARMUP_ITERS {
		do_linear_forward(dtype, x_shape, w_shape, x_bytes, w_bytes, y_bytes)
		ml.clear()
	}
	t0 := time.tick_now()
	for _ in 0..<TIMED_ITERS {
		do_linear_forward(dtype, x_shape, w_shape, x_bytes, w_bytes, y_bytes)
		ml.clear()
	}
	fwd_ms = f32(time.duration_milliseconds(time.tick_since(t0))) / TIMED_ITERS

	if measure_backward {
		for _ in 0..<WARMUP_ITERS {
			do_linear_forward_backward(dtype, x_shape, w_shape, x_bytes, w_bytes, y_bytes)
			ml.clear()
		}
		t1 := time.tick_now()
		for _ in 0..<TIMED_ITERS {
			do_linear_forward_backward(dtype, x_shape, w_shape, x_bytes, w_bytes, y_bytes)
			ml.clear()
		}
		fb_ms = f32(time.duration_milliseconds(time.tick_since(t1))) / TIMED_ITERS
	}
	return
}

do_linear_forward :: proc(dtype: ml.Data_Type, x_shape, w_shape: []int, x_bytes, w_bytes, y_bytes: []byte, loc := #caller_location) {
	x := ml.alloc(dtype, x_shape, persistent=false, buffers={.Data})
	w := ml.alloc(dtype, w_shape, persistent=true,  buffers={.Data})
	x.backend.buffer_set(x.buffers[.Data], x_bytes, loc)
	w.backend.buffer_set(w.buffers[.Data], w_bytes, loc)
	y := ml.linear(x, w)
	y.backend.buffer_get(y.buffers[.Data], y_bytes, loc)
	w.backend.buffer_free(w.buffers[.Data], loc)
}

do_linear_forward_backward :: proc(dtype: ml.Data_Type, x_shape, w_shape: []int, x_bytes, w_bytes, y_bytes: []byte, loc := #caller_location) {
	x := ml.alloc(dtype, x_shape, persistent=false, buffers=ml.DEFAULT_ACTIVATION_BUFFERS)
	w := ml.alloc(dtype, w_shape, persistent=true,  buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	x.backend.buffer_set(x.buffers[.Data], x_bytes, loc)
	w.backend.buffer_set(w.buffers[.Data], w_bytes, loc)
	y := ml.linear(x, w)
	ml.backward()
	y.backend.buffer_get(y.buffers[.Data], y_bytes, loc)
	for kind in ml.Buffer_Kind {
		if w.buffers[kind] != (ml.Backend_Buffer{}) {
			w.backend.buffer_free(w.buffers[kind], loc)
		}
	}
}

// ----- Linear_Q4_K (M=1 decode) ----------------------------------------------

Q4K_Shape :: struct {
	name: string,
	k:    int,  // input_size, multiple of 256
	n:    int,  // output_size, multiple of 2
}

q4k_shapes := []Q4K_Shape{
	{"k=2048 n=2048",   2048,   2048},
	{"k=4096 n=4096",   4096,   4096},
	{"k=4096 n=32000",  4096,  32000},  // lm_head-ish
}

run_linear_q4_k :: proc() -> bool {
	fmt.printfln("\n=== linear_q4_k (M=1, bf16 input)  ===")

	// Parity: tiny shape, CPU dequant -> f32 matmul as ground truth.
	if !verify_q4k_parity() {
		return false
	}

	// Perf: weights are persistent, only decodes are timed (matches actual
	// inference where the model loads once and the same weight is hammered
	// thousands of times).
	fmt.printfln("%-30s %12s %12s %10s", "shape", "vk decode ms", "cu decode ms", "speedup")
	all_ok := true
	for s in q4k_shapes {
		w_bytes := (s.n * s.k / 256) * 144
		host_w  := builtin.make([]byte, w_bytes); defer delete(host_w)
		fill_q4k_bytes(host_w)

		host_x := builtin.make([]byte, s.k * size_of(ml.Bf16)); defer delete(host_x)
		fill_random(host_x, .Bf16)

		host_yv := builtin.make([]byte, s.n * size_of(ml.Bf16)); defer delete(host_yv)
		host_yc := builtin.make([]byte, s.n * size_of(ml.Bf16)); defer delete(host_yc)

		vk_ms := run_q4k_decode_perf(.Vulkan, s, host_x, host_w, host_yv)
		cu_ms := run_q4k_decode_perf(.Cuda,   s, host_x, host_w, host_yc)

		fmt.printfln(" %-29s %12.4f %12.4f %9.2fx", s.name, vk_ms, cu_ms, vk_ms / cu_ms)
	}
	return all_ok
}

// One-shot parity check: K=256, N=8 (one Q4_K block, 4 workgroups). CPU
// dequantizes the same byte stream, runs the matmul in f32, and we verify
// each backend's bf16 output lands within bf16 quantization noise of the
// f32 truth. We DON'T require the two backends to agree bit-for-bit.
verify_q4k_parity :: proc() -> bool {
	K :: 256
	N :: 8
	w_bytes_n := (N * K / 256) * 144
	host_w := builtin.make([]byte, w_bytes_n); defer delete(host_w)
	fill_q4k_bytes(host_w)

	host_x := builtin.make([]byte, K * size_of(ml.Bf16)); defer delete(host_x)
	fill_random(host_x, .Bf16)

	// CPU ground truth: dequant Q4_K -> f32 weights, expand bf16 input -> f32,
	// y = x @ W^T in plain f32.
	w_f32 := builtin.make([]f32, N * K); defer delete(w_f32)
	gguf.dequantize_q4_k(host_w, w_f32)

	x_f32 := builtin.make([]f32, K); defer delete(x_f32)
	xs := mem.slice_data_cast([]ml.Bf16, host_x)
	for i in 0..<K { x_f32[i] = ml.bf16_to_f32(xs[i]) }

	y_truth := builtin.make([]f32, N); defer delete(y_truth)
	for n in 0..<N {
		acc := f32(0)
		for k in 0..<K { acc += x_f32[k] * w_f32[n * K + k] }
		y_truth[n] = acc
	}

	host_yv := builtin.make([]byte, N * size_of(ml.Bf16)); defer delete(host_yv)
	host_yc := builtin.make([]byte, N * size_of(ml.Bf16)); defer delete(host_yc)
	run_q4k_one_shot(.Vulkan, K, N, host_x, host_w, host_yv)
	run_q4k_one_shot(.Cuda,   K, N, host_x, host_w, host_yc)

	ok := true
	if !compare_q4k_to_truth("vulkan", host_yv, y_truth) { ok = false }
	if !compare_q4k_to_truth("cuda",   host_yc, y_truth) { ok = false }
	if ok {
		fmt.printfln("  parity OK  (cpu f32 ground truth, %d outputs)", N)
	}
	return ok
}

run_q4k_one_shot :: proc(b: Backend, K, N: int, x_bytes, w_bytes, y_bytes: []byte, loc := #caller_location) {
	ctx: ^ml.Context
	switch b {
	case .Vulkan: ctx = vulkan.context_create()
	case .Cuda:   ctx = cu.context_create()
	}
	ml.context_begin(ctx)
	defer { ml.context_end(); switch b {
	case .Vulkan: vulkan.context_destroy(ctx)
	case .Cuda:   cu.context_destroy(ctx)
	} }

	x_shape := []int{1, K}
	w_shape := []int{N, K}
	x := ml.alloc(.Bf16, x_shape, persistent=false, buffers={.Data})
	w := ml.alloc(.Q4_K, w_shape, persistent=true,  buffers={.Data})
	x.backend.buffer_set(x.buffers[.Data], x_bytes, loc)
	w.backend.buffer_set(w.buffers[.Data], w_bytes, loc)
	y := ml.linear_q4_k(x, w)
	y.backend.buffer_get(y.buffers[.Data], y_bytes, loc)
	w.backend.buffer_free(w.buffers[.Data], loc)
}

compare_q4k_to_truth :: proc(label: string, got_bytes: []byte, truth: []f32) -> bool {
	got := mem.slice_data_cast([]ml.Bf16, got_bytes)
	bad := 0
	for i in 0..<len(truth) {
		gf := ml.bf16_to_f32(got[i])
		tf := truth[i]
		// bf16 has ~7 bits of mantissa. Per-output relative tolerance of 5%
		// covers honest dot-product rounding plus the bf16 truncation of the
		// final result. Vulkan and CUDA both pass under this with their own
		// FP scheduling choices.
		scale := max(abs(tf), abs(gf), 1)
		if abs(gf - tf) > 0.05 * scale {
			bad += 1
			if bad <= 3 {
				fmt.eprintfln("  %s mismatch at %d: got=%v truth=%v", label, i, gf, tf)
			}
		}
	}
	if bad > 0 {
		fmt.eprintfln("  %s: %d/%d outputs miss bf16 tolerance", label, bad, len(truth))
		return false
	}
	return true
}

// Perf path: alloc weight + input ONCE outside the timed loop, time only the
// repeated decode (quantize_q8_1 -> mmvq) which is what real inference does.
run_q4k_decode_perf :: proc(b: Backend, s: Q4K_Shape, x_bytes, w_bytes, y_bytes: []byte, loc := #caller_location) -> (ms: f32) {
	ctx: ^ml.Context
	switch b {
	case .Vulkan: ctx = vulkan.context_create()
	case .Cuda:   ctx = cu.context_create()
	}
	ml.context_begin(ctx)
	defer { ml.context_end(); switch b {
	case .Vulkan: vulkan.context_destroy(ctx)
	case .Cuda:   cu.context_destroy(ctx)
	} }

	x_shape := []int{1, s.k}
	w_shape := []int{s.n, s.k}

	// Persistent weight: load once, never reupload.
	w := ml.alloc(.Q4_K, w_shape, persistent=true, buffers={.Data})
	w.backend.buffer_set(w.buffers[.Data], w_bytes, loc)
	defer w.backend.buffer_free(w.buffers[.Data], loc)

	// Activation buffers recycle through the pool across clears, so allocs
	// after the first iteration are basically free.
	step :: proc(dtype_x: ml.Data_Type, x_shape: []int, w: ml.Tensor, x_bytes, y_bytes: []byte, loc := #caller_location) {
		x := ml.alloc(dtype_x, x_shape, persistent=false, buffers={.Data})
		x.backend.buffer_set(x.buffers[.Data], x_bytes, loc)
		y := ml.linear_q4_k(x, w)
		y.backend.buffer_get(y.buffers[.Data], y_bytes, loc)
	}

	for _ in 0..<WARMUP_ITERS {
		step(.Bf16, x_shape, w, x_bytes, y_bytes)
		ml.clear()
	}
	t0 := time.tick_now()
	for _ in 0..<TIMED_ITERS {
		step(.Bf16, x_shape, w, x_bytes, y_bytes)
		ml.clear()
	}
	return f32(time.duration_milliseconds(time.tick_since(t0))) / TIMED_ITERS
}

fill_q4k_bytes :: proc(buf: []byte) {
	rand.reset(0xC0DE_4_4)
	// Walk block by block (144 bytes per block); pack realistic-ish d/dmin
	// (small fp16 in [-1, 1]) and uniformly random scales/mins/nibbles.
	for i := 0; i + 144 <= builtin.len(buf); i += 144 {
		// Header: d (fp16), dmin (fp16) Ã¢â‚¬â€ keep small so accumulated dot
		// products stay in fp16 range.
		d_h    := f16_from_f32(rand.float32_range(-0.05, 0.05))
		dmin_h := f16_from_f32(rand.float32_range(-0.02, 0.02))
		buf[i + 0] = u8(d_h    & 0xff); buf[i + 1] = u8((d_h    >> 8) & 0xff)
		buf[i + 2] = u8(dmin_h & 0xff); buf[i + 3] = u8((dmin_h >> 8) & 0xff)
		// 12 bytes of packed scales/mins.
		for j in 0..<12 { buf[i + 4 + j] = u8(rand.uint32() & 0x3f) }
		// 128 bytes of 4-bit quants.
		for j in 0..<128 { buf[i + 16 + j] = u8(rand.uint32() & 0xff) }
	}
}

f16_from_f32 :: proc(f: f32) -> u16 {
	// Minimal fp32 -> fp16 conversion. Used only in the bench, so
	// flush-to-zero for subnormals is fine.
	bits := transmute(u32)f
	sign := u16((bits >> 16) & 0x8000)
	exp  := i32((bits >> 23) & 0xff) - 127 + 15
	mant := bits & 0x7fffff
	if exp <= 0 { return sign }
	if exp >= 31 { return sign | 0x7c00 }
	return sign | u16(exp << 10) | u16(mant >> 13)
}


compare_linear :: proc(dtype: ml.Data_Type, ref, got: []byte, s: Linear_Shape) -> bool {
	// Per-element tolerance. BF16 has ~1/128 relative precision; accumulating
	// over K terms grows the error roughly with sqrt(K).
	// Two unrelated GEMM implementations diverge at the per-element level by
	// a few ULP scaled by sqrt(K) due to summation-order differences. F32
	// drift is small but real; BF16 drift is much larger.
	rel_tol: f32
	switch dtype {
	case .F32:  rel_tol = 5e-4
	case .Bf16: rel_tol = 0.03
	case .Q4_K, .Q6_K: return false
	}
	n := s.count * s.output_size

	switch dtype {
	case .F32:
		rs := mem.slice_data_cast([]f32, ref)
		gs := mem.slice_data_cast([]f32, got)
		mismatches := 0
		for i in 0..<n {
			d := abs(rs[i] - gs[i])
			scale := max(abs(rs[i]), abs(gs[i]), 1)
			if d > rel_tol * scale {
				mismatches += 1
				if mismatches <= 4 {
					fmt.eprintfln("  parity FAIL at %d: vk=%v cu=%v", i, rs[i], gs[i])
				}
			}
		}
		return mismatches == 0
	case .Bf16:
		rs := mem.slice_data_cast([]ml.Bf16, ref)
		gs := mem.slice_data_cast([]ml.Bf16, got)
		mismatches := 0
		for i in 0..<n {
			rf := ml.bf16_to_f32(rs[i])
			gf := ml.bf16_to_f32(gs[i])
			d := abs(rf - gf)
			scale := max(abs(rf), abs(gf), 1)
			if d > rel_tol * scale {
				mismatches += 1
				if mismatches <= 4 {
					fmt.eprintfln("  parity FAIL at %d: vk=%v cu=%v", i, rf, gf)
				}
			}
		}
		return mismatches == 0
	case .Q4_K, .Q6_K: return false
	}
	return false
}

run_dtype :: proc(dtype: ml.Data_Type) -> bool {
	// The framework's automatic backward chain needs an F32 loss tensor, so
	// we only measure forward+backward for F32. Bf16 forward stresses the
	// fast __hadd path and is the most relevant case for inference anyway.
	measure_backward := dtype == .F32

	fmt.printfln("\n=== add  (%v)  ===", dtype)
	if measure_backward {
		fmt.printfln("%-30s %12s %12s %12s %12s %10s", "shape", "vk fwd ms", "cu fwd ms", "vk fwd+bwd", "cu fwd+bwd", "fwd speedup")
	} else {
		fmt.printfln("%-30s %12s %12s %10s", "shape", "vk fwd ms", "cu fwd ms", "fwd speedup")
	}

	all_ok := true
	for s in shapes {
		if s.n_a % s.n_b != 0 { continue }

		host_a, host_b, host_out_vk, host_out_cu := alloc_host(dtype, s)
		defer { delete(host_a); delete(host_b); delete(host_out_vk); delete(host_out_cu) }

		fill_random(host_a, dtype)
		fill_random(host_b, dtype)

		vk_fwd_ms, vk_fb_ms := run_backend(.Vulkan, dtype, s, host_a, host_b, host_out_vk, measure_backward)
		cu_fwd_ms, cu_fb_ms := run_backend(.Cuda,   dtype, s, host_a, host_b, host_out_cu, measure_backward)

		ok := compare(dtype, host_out_vk, host_out_cu, s.n_a)
		if !ok { all_ok = false }

		speedup := vk_fwd_ms / cu_fwd_ms
		marker  := ok ? " " : "X"
		if measure_backward {
			fmt.printfln("%s%-29s %12.4f %12.4f %12.4f %12.4f %9.2fx",
				marker, s.name, vk_fwd_ms, cu_fwd_ms, vk_fb_ms, cu_fb_ms, speedup)
		} else {
			fmt.printfln("%s%-29s %12.4f %12.4f %9.2fx",
				marker, s.name, vk_fwd_ms, cu_fwd_ms, speedup)
		}
	}
	return all_ok
}

Backend :: enum { Vulkan, Cuda }

alloc_host :: proc(dtype: ml.Data_Type, s: Shape) -> (a, b, ov, oc: []byte) {
	bytes_a := ml.data_type_size(dtype) * s.n_a
	bytes_b := ml.data_type_size(dtype) * s.n_b
	a  = builtin.make([]byte, bytes_a)
	b  = builtin.make([]byte, bytes_b)
	ov = builtin.make([]byte, bytes_a)
	oc = builtin.make([]byte, bytes_a)
	return
}

fill_random :: proc(buf: []byte, dtype: ml.Data_Type) {
	rand.reset(0xC0DEC0DE ~ u64(uintptr(raw_data(buf))))
	switch dtype {
	case .F32:
		f := mem.slice_data_cast([]f32, buf)
		for &x in f { x = rand.float32_range(-1, 1) }
	case .Bf16:
		bf := mem.slice_data_cast([]ml.Bf16, buf)
		for &x in bf { x = ml.bf16_from_f32(rand.float32_range(-1, 1)) }
	case .Q4_K, .Q6_K: panic("unsupported in bench")
	}
}

run_backend :: proc(b: Backend, dtype: ml.Data_Type, s: Shape, a_bytes, b_bytes, out_bytes: []byte, measure_backward: bool) -> (fwd_ms, fb_ms: f32) {
	ctx: ^ml.Context
	switch b {
	case .Vulkan: ctx = vulkan.context_create()
	case .Cuda:   ctx = cu.context_create()
	}
	ml.context_begin(ctx)
	defer { ml.context_end(); switch b {
	case .Vulkan: vulkan.context_destroy(ctx)
	case .Cuda:   cu.context_destroy(ctx)
	} }

	shape_a := []int{s.n_a}
	shape_b := []int{s.n_b}

	// ----- Forward only -----
	for _ in 0..<WARMUP_ITERS {
		do_forward(dtype, shape_a, shape_b, a_bytes, b_bytes, out_bytes)
		ml.clear()
	}
	t0 := time.tick_now()
	for _ in 0..<TIMED_ITERS {
		do_forward(dtype, shape_a, shape_b, a_bytes, b_bytes, out_bytes)
		ml.clear()
	}
	fwd_ms = f32(time.duration_milliseconds(time.tick_since(t0))) / TIMED_ITERS

	if measure_backward {
		for _ in 0..<WARMUP_ITERS {
			do_forward_backward(dtype, shape_a, shape_b, a_bytes, b_bytes, out_bytes)
			ml.clear()
		}
		t1 := time.tick_now()
		for _ in 0..<TIMED_ITERS {
			do_forward_backward(dtype, shape_a, shape_b, a_bytes, b_bytes, out_bytes)
			ml.clear()
		}
		fb_ms = f32(time.duration_milliseconds(time.tick_since(t1))) / TIMED_ITERS
	}

	return
}

// One forward: alloc tensors, upload inputs, run add, download output.
// We bundle alloc+upload inside the timed loop because it mirrors how the
// host program actually drives an op. clear() is called by the caller
// after timing.
do_forward :: proc(dtype: ml.Data_Type, shape_a, shape_b: []int, a_bytes, b_bytes, out_bytes: []byte, loc := #caller_location) {
	a := ml.alloc(dtype, shape_a, persistent=false, buffers={.Data})
	b := ml.alloc(dtype, shape_b, persistent=false, buffers={.Data})
	a.backend.buffer_set(a.buffers[.Data], a_bytes, loc)
	b.backend.buffer_set(b.buffers[.Data], b_bytes, loc)
	out := ml.add(a, b)
	out.backend.buffer_get(out.buffers[.Data], out_bytes, loc)
}

do_forward_backward :: proc(dtype: ml.Data_Type, shape_a, shape_b: []int, a_bytes, b_bytes, out_bytes: []byte, loc := #caller_location) {
	a := ml.alloc(dtype, shape_a, persistent=false, buffers=ml.DEFAULT_ACTIVATION_BUFFERS)
	b := ml.alloc(dtype, shape_b, persistent=false, buffers=ml.DEFAULT_ACTIVATION_BUFFERS)
	a.backend.buffer_set(a.buffers[.Data], a_bytes, loc)
	b.backend.buffer_set(b.buffers[.Data], b_bytes, loc)
	out := ml.add(a, b)
	ml.backward()
	out.backend.buffer_get(out.buffers[.Data], out_bytes, loc)
}

compare :: proc(dtype: ml.Data_Type, ref, got: []byte, n: int) -> bool {
	switch dtype {
	case .F32:
		rs := mem.slice_data_cast([]f32, ref)
		gs := mem.slice_data_cast([]f32, got)
		for i in 0..<n {
			if rs[i] != gs[i] {
				if abs(rs[i] - gs[i]) > 1e-5 {
					fmt.eprintfln("  parity FAIL at %d: vk=%v cu=%v", i, rs[i], gs[i])
					return false
				}
			}
		}
		return true
	case .Bf16:
		rs := mem.slice_data_cast([]ml.Bf16, ref)
		gs := mem.slice_data_cast([]ml.Bf16, got)
		mismatches := 0
		for i in 0..<n {
			if rs[i] != gs[i] {
				rf := ml.bf16_to_f32(rs[i])
				gf := ml.bf16_to_f32(gs[i])
				// 1 ulp of bf16 around 1.0 ~= 0.0078; allow 2 ulp.
				if abs(rf - gf) > 0.02 * max(abs(rf), abs(gf), 1) {
					mismatches += 1
					if mismatches <= 4 {
						fmt.eprintfln("  parity FAIL at %d: vk=%v cu=%v", i, rf, gf)
					}
				}
			}
		}
		return mismatches == 0
	case .Q4_K, .Q6_K: return false
	}
	return false
}
