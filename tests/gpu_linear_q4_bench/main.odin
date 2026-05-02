package gpu_linear_q4_bench

import "base:builtin"

import "core:fmt"
import "core:mem"
import "core:time"

import ml  "../.."
import gpu "../../backends/gpu"

WARMUP :: 10
TIMED  :: 200

Shape :: struct {
	label: string,
	out:   int,
	in_:   int,
}

main :: proc() {
	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	// Decode-shape (M=1) projections used by Gemma 4 E4B.
	shapes := []Shape{
		{"q_proj         2560  ->  2048",   2048,   2560},
		{"k_proj sliding 2560  ->   512",    512,   2560},
		{"k_proj full    2560  ->  1024",   1024,   2560},
		{"o_proj         2048  ->  2560",   2560,   2048},
		{"mlp_gate       2560  -> 10240",  10240,   2560},
		{"mlp_down      10240  ->  2560",   2560,  10240},
		{"lm_head        2560  ->262144", 262144,   2560},
	}

	fmt.println("=== isolated linear_q4 dispatch (M=1, GPU) ===")
	fmt.println("(weight bytes / time = effective bandwidth; 3090 Ti peak ~ 1008 GB/s)")
	fmt.println()
	for s in shapes {
		bench(s)
	}
}

_bytes_of :: proc(s: []ml.Bf16) -> []byte { return mem.slice_to_bytes(s) }

bench :: proc(s: Shape) {
	x_bf := ml.alloc(.Bf16, {1, s.in_},                     persistent=true, buffers=ml.Buffer_Set{.Data})
	w_bf := ml.alloc(.Bf16, {s.out, s.in_},                 persistent=true, buffers=ml.Buffer_Set{.Data})
	defer ml.destroy(x_bf)

	ml.fill_normal(x_bf, 0, 1)
	ml.fill_normal(w_bf, 0, 1)

	w_q, w_s := ml.quantize_int4(w_bf)
	ml.destroy(w_bf)
	defer ml.destroy(w_q)
	defer ml.destroy(w_s)

	final_buf := builtin.make([]ml.Bf16, s.out)
	defer delete(final_buf)

	for _ in 0 ..< WARMUP {
		ml.clear()
		_ = ml.linear_q4(x_bf, w_q, w_s)
	}
	{
		y := ml.linear_q4(x_bf, w_q, w_s)
		ml.get_data_bytes(y, _bytes_of(final_buf))
	}

	// Without ml.clear() in the loop, activation buffers accumulate but we
	// isolate the per-dispatch CPU work from the per-clear pool reset.
	ml.clear()
	t_record_start := time.tick_now()
	for _ in 0 ..< TIMED {
		_ = ml.linear_q4(x_bf, w_q, w_s)
	}
	record_ns := i64(time.tick_since(t_record_start))
	ml.clear()

	t_sync_start := time.tick_now()
	{
		y := ml.linear_q4(x_bf, w_q, w_s)
		ml.get_data_bytes(y, _bytes_of(final_buf))
	}
	sync_ns := i64(time.tick_since(t_sync_start))

	per_op_record_ns := record_ns / TIMED
	weight_bytes := s.out * s.in_ / 2 + s.out * (s.in_ / 32) * 4
	gbs := f64(weight_bytes) * f64(TIMED) / f64(record_ns + sync_ns)
	pct_peak := gbs / 1008.0 * 100.0

	fmt.printfln("  %v   record %.3f ms/op   gpu-tail %.2f ms   eff %.1f GB/s (%.2f%%)",
		s.label, f64(per_op_record_ns) / 1e6, f64(sync_ns) / 1e6, gbs, pct_peak)
}
