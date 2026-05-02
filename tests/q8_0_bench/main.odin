package q8_0_bench

import "base:runtime"

import "core:fmt"
import "core:time"

import ml  "../.."
import cpu "../../backends/cpu"

WARMUP   :: 5
TIMED    :: 30
THREADS  :: 24
ARENA_GB :: 4

Shape :: struct {
	label:        string,
	out_size:     int,
	in_size:      int,
	tokens:       int,
}

main :: proc() {
	ctx := cpu.context_create(ARENA_GB * 1024 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	thread_counts := [?]int{4, 6, 8, 10, 12, 14, 16}
	for n_threads in thread_counts {
		cpu.set_thread_count(n_threads)
		fmt.printfln("--- threads = %v ---", n_threads)
		full_pass()
	}
	fmt.println()

	cpu.set_thread_count(THREADS)

	// Realistic Gemma E4B decode-time matmul shapes (batch=1).
	shapes := []Shape{
		{"q_proj         2560  ->  2048", 2048,   2560,  1},
		{"k_proj sliding 2560  ->   512",  512,   2560,  1},
		{"k_proj full    2560  ->  1024", 1024,   2560,  1},
		{"o_proj         2048  ->  2560", 2560,   2048,  1},
		{"mlp_gate       2560  -> 10240",10240,   2560,  1},
		{"mlp_down      10240  ->  2560", 2560,  10240,  1},
		{"ple_gate       2560  ->   256",  256,   2560,  1},
		{"ple_proj        256  ->  2560", 2560,    256,  1},
		{"lm_head        2560  ->262144",262144,  2560,  1},
	}

	fmt.println()
	fmt.println("=== bf16 baseline ===")
	for s in shapes {
		bench_bf16(s)
	}

	fmt.println()
	fmt.println("=== q8_0 ===")
	for s in shapes {
		bench_q8_0(s)
	}

	fmt.println()
	fmt.println("=== full Gemma E4B per-token (estimate) ===")
	full_pass()
}

bench_bf16 :: proc(s: Shape) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	x := ml.make(.Bf16, {s.tokens, s.in_size})
	w := ml.make(.Bf16, {s.out_size, s.in_size})
	ml.fill_normal(x, 0, 1)
	ml.fill_normal(w, 0, 1)
	defer ml.destroy(x)
	defer ml.destroy(w)

	for _ in 0 ..< WARMUP {
		ml.clear()
		_ = ml.linear(x, w)
	}

	t0 := time.tick_now()
	for _ in 0 ..< TIMED {
		ml.clear()
		_ = ml.linear(x, w)
	}
	ns := i64(time.tick_since(t0)) / TIMED
	report(s, ns, /*bytes_per_weight*/ 2)
}

bench_q8_0 :: proc(s: Shape) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	x  := ml.make(.Bf16, {s.tokens, s.in_size})
	wb := ml.make(.Bf16, {s.out_size, s.in_size})
	ml.fill_normal(x, 0, 1)
	ml.fill_normal(wb, 0, 1)
	defer ml.destroy(x)

	w_q, w_s := ml.quantize_q8_0(wb)
	ml.destroy(wb)
	defer ml.destroy(w_q)
	defer ml.destroy(w_s)

	for _ in 0 ..< WARMUP {
		ml.clear()
		_ = ml.linear_q8_0(x, w_q, w_s)
	}

	t0 := time.tick_now()
	for _ in 0 ..< TIMED {
		ml.clear()
		_ = ml.linear_q8_0(x, w_q, w_s)
	}
	ns := i64(time.tick_since(t0)) / TIMED
	// Q8_0 = 1 byte/wt + 4-byte scale per 32 weights = 1.125 bytes/weight.
	report_q8(s, ns)
}

report :: proc(s: Shape, ns: i64, bytes_per_weight: int) {
	weight_bytes := s.out_size * s.in_size * bytes_per_weight
	gbs := f64(weight_bytes) / f64(ns)
	fmt.printfln("  %v  %.3f ms   %.1f GB/s  (W=%.1f MB)",
		s.label,
		f64(ns) / 1e6,
		gbs,
		f64(weight_bytes) / (1024 * 1024))
}

report_q8 :: proc(s: Shape, ns: i64) {
	weight_bytes := s.out_size * s.in_size + s.out_size * (s.in_size / 32) * 4
	gbs := f64(weight_bytes) / f64(ns)
	fmt.printfln("  %v  %.3f ms   %.1f GB/s  (W=%.1f MB)",
		s.label,
		f64(ns) / 1e6,
		gbs,
		f64(weight_bytes) / (1024 * 1024))
}

// Approximate one Gemma E4B decode forward by issuing one matmul per
// projection summed across 42 layers. Uses average shapes — close enough
// to gauge tok/s impact of kernel changes.
full_pass :: proc() {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	// Per-layer projections (rough averages between sliding/full layers; not all
	// 42 layers do every projection due to KV-sharing, but for a kernel-cost
	// estimate this is close enough).
	hidden    :: 2560
	q_out     :: 2048
	kv_out    :: 768  // ~mid between 512 sliding and 1024 full
	mlp_inner :: 10240
	ple_dim   :: 256
	num_layers :: 42
	num_kv_layers :: 24 // shared layers reuse k/v from earlier

	make_q :: proc(out, in_dim: int) -> (ml.Tensor, ml.Tensor) {
		w := ml.make(.Bf16, {out, in_dim})
		ml.fill_normal(w, 0, 1)
		q, s := ml.quantize_q8_0(w)
		ml.destroy(w)
		return q, s
	}

	q_proj_w, q_proj_s := make_q(q_out, hidden)
	k_proj_w, k_proj_s := make_q(kv_out, hidden)
	v_proj_w, v_proj_s := make_q(kv_out, hidden)
	o_proj_w, o_proj_s := make_q(hidden, q_out)
	mlp_g_w,  mlp_g_s  := make_q(mlp_inner, hidden)
	mlp_u_w,  mlp_u_s  := make_q(mlp_inner, hidden)
	mlp_d_w,  mlp_d_s  := make_q(hidden, mlp_inner)
	ple_g_w,  ple_g_s  := make_q(ple_dim, hidden)
	ple_p_w,  ple_p_s  := make_q(hidden, ple_dim)
	lm_w,     lm_s     := make_q(262144, hidden)
	defer {
		ml.destroy(q_proj_w); ml.destroy(q_proj_s)
		ml.destroy(k_proj_w); ml.destroy(k_proj_s)
		ml.destroy(v_proj_w); ml.destroy(v_proj_s)
		ml.destroy(o_proj_w); ml.destroy(o_proj_s)
		ml.destroy(mlp_g_w);  ml.destroy(mlp_g_s)
		ml.destroy(mlp_u_w);  ml.destroy(mlp_u_s)
		ml.destroy(mlp_d_w);  ml.destroy(mlp_d_s)
		ml.destroy(ple_g_w);  ml.destroy(ple_g_s)
		ml.destroy(ple_p_w);  ml.destroy(ple_p_s)
		ml.destroy(lm_w);     ml.destroy(lm_s)
	}

	x_h     := ml.make(.Bf16, {1, hidden})
	x_qout  := ml.make(.Bf16, {1, q_out})
	x_inner := ml.make(.Bf16, {1, mlp_inner})
	x_ple   := ml.make(.Bf16, {1, ple_dim})
	ml.fill_normal(x_h, 0, 1)
	ml.fill_normal(x_qout, 0, 1)
	ml.fill_normal(x_inner, 0, 1)
	ml.fill_normal(x_ple, 0, 1)
	defer { ml.destroy(x_h); ml.destroy(x_qout); ml.destroy(x_inner); ml.destroy(x_ple) }

	one_token :: proc(args: ^struct {
		num_layers, num_kv_layers: int,
		x_h, x_qout, x_inner, x_ple:                                                           ml.Tensor,
		q_proj_w, q_proj_s, k_proj_w, k_proj_s, v_proj_w, v_proj_s, o_proj_w, o_proj_s:        ml.Tensor,
		mlp_g_w, mlp_g_s, mlp_u_w, mlp_u_s, mlp_d_w, mlp_d_s:                                  ml.Tensor,
		ple_g_w, ple_g_s, ple_p_w, ple_p_s, lm_w, lm_s:                                        ml.Tensor,
	}) {
		ml.clear()
		for _ in 0 ..< args.num_layers {
			_ = ml.linear_q8_0(args.x_h,    args.q_proj_w, args.q_proj_s)
			_ = ml.linear_q8_0(args.x_qout, args.o_proj_w, args.o_proj_s)
			_ = ml.linear_q8_0(args.x_h,    args.mlp_g_w,  args.mlp_g_s)
			_ = ml.linear_q8_0(args.x_h,    args.mlp_u_w,  args.mlp_u_s)
			_ = ml.linear_q8_0(args.x_inner,args.mlp_d_w,  args.mlp_d_s)
			_ = ml.linear_q8_0(args.x_h,    args.ple_g_w,  args.ple_g_s)
			_ = ml.linear_q8_0(args.x_ple,  args.ple_p_w,  args.ple_p_s)
		}
		for _ in 0 ..< args.num_kv_layers {
			_ = ml.linear_q8_0(args.x_h, args.k_proj_w, args.k_proj_s)
			_ = ml.linear_q8_0(args.x_h, args.v_proj_w, args.v_proj_s)
		}
		_ = ml.linear_q8_0(args.x_h, args.lm_w, args.lm_s)
	}

	args := struct {
		num_layers, num_kv_layers: int,
		x_h, x_qout, x_inner, x_ple:                                                           ml.Tensor,
		q_proj_w, q_proj_s, k_proj_w, k_proj_s, v_proj_w, v_proj_s, o_proj_w, o_proj_s:        ml.Tensor,
		mlp_g_w, mlp_g_s, mlp_u_w, mlp_u_s, mlp_d_w, mlp_d_s:                                  ml.Tensor,
		ple_g_w, ple_g_s, ple_p_w, ple_p_s, lm_w, lm_s:                                        ml.Tensor,
	}{
		num_layers = num_layers, num_kv_layers = num_kv_layers,
		x_h = x_h, x_qout = x_qout, x_inner = x_inner, x_ple = x_ple,
		q_proj_w = q_proj_w, q_proj_s = q_proj_s,
		k_proj_w = k_proj_w, k_proj_s = k_proj_s,
		v_proj_w = v_proj_w, v_proj_s = v_proj_s,
		o_proj_w = o_proj_w, o_proj_s = o_proj_s,
		mlp_g_w = mlp_g_w, mlp_g_s = mlp_g_s,
		mlp_u_w = mlp_u_w, mlp_u_s = mlp_u_s,
		mlp_d_w = mlp_d_w, mlp_d_s = mlp_d_s,
		ple_g_w = ple_g_w, ple_g_s = ple_g_s,
		ple_p_w = ple_p_w, ple_p_s = ple_p_s,
		lm_w = lm_w, lm_s = lm_s,
	}

	for _ in 0 ..< 3 { one_token(&args) }

	N :: 10
	t0 := time.tick_now()
	for _ in 0 ..< N { one_token(&args) }
	per_token_ns := i64(time.tick_since(t0)) / N
	tok_s := 1e9 / f64(per_token_ns)
	fmt.printfln("  per-token: %.1f ms   (%.2f tok/s)", f64(per_token_ns) / 1e6, tok_s)
}
