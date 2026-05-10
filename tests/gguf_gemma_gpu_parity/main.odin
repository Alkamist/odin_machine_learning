package gguf_gemma_gpu_parity

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
import "core:time"

import ml    "../.."
import gpu   "../../backends/vulkan"
import gemma "../../networks/gemma"

// GPU parity for the GGUF Gemma 4 E4B model. Our Q4_K/Q6_K GPU shaders are
// M=1 only, so we drive `forward_cached` one token at a time â€” same shape
// the chat repl will hit during decode. Also reports decode tok/s once the
// cache is warm.

DATA_DIR    :: "gemma_data"
TOKENS_PATH :: DATA_DIR + "/prompt_tokens.bin"
LOGITS_PATH :: DATA_DIR + "/expected_logits.bin"

main :: proc() {
	if len(os.args) < 2 {
		fmt.eprintfln("usage: %v <gguf_path>", os.args[0])
		os.exit(2)
	}
	gguf_path := os.args[1]

	tokens, tokens_ok := load_tokens(TOKENS_PATH)
	if !tokens_ok {
		fmt.eprintln("could not load prompt_tokens.bin")
		os.exit(1)
	}
	defer delete(tokens)

	expected_shape, expected_logits, exp_ok := load_tensor(LOGITS_PATH)
	if !exp_ok {
		fmt.eprintln("could not load expected_logits.bin")
		os.exit(1)
	}
	defer delete(expected_shape)
	defer delete(expected_logits)

	fmt.printfln("Loaded %v prompt tokens.", builtin.len(tokens))
	fmt.printfln("Expected logits shape = %v.", expected_shape)

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)
	ml.clear({.No_Gradients})

	cfg := gemma.make_e4b_config()
	defer gemma.config_destroy(cfg)
	model := gemma.make(cfg, .Bf16)
	defer gemma.destroy(model)

	t_load := time.tick_now()
	if !gemma.load_gguf(&model, gguf_path) {
		fmt.eprintln("GGUF load failed.")
		os.exit(1)
	}
	fmt.printfln("Loaded GGUF in %.1f s.", time.duration_seconds(time.tick_since(t_load)))

	cache := gemma.cache_make(model, 256)
	defer gemma.cache_destroy(cache)

	vocab_size := expected_shape[1]
	all_logits := make([]f32, builtin.len(tokens) * vocab_size)
	defer delete(all_logits)

	// Warm-up + per-token timing. Enable GPU timestamp queries from token 1
	// onwards so the first-token pipeline-compilation cost doesn't pollute
	// the per-pipeline averages.
	t_total := time.tick_now()
	for token, pos in tokens {
		if pos == 1 {
			gpu.enable_timing()
			gpu.reset_timing()
		}
		t_step := time.tick_now()
		logits := gemma.forward_cached(model, &cache, []int{token})
		t_record := time.tick_since(t_step)
		// Read this position's logits â€” this is what triggers submit + wait.
		row := all_logits[pos * vocab_size : (pos + 1) * vocab_size]
		if logits.type == .F32 {
			ml.get_data(logits, row)
		} else {
			bf := make([]ml.Bf16, vocab_size)
			defer delete(bf)
			ml.get_data_bytes(logits, slice.bytes_from_ptr(raw_data(bf), vocab_size * 2))
			for v, i in bf {
				row[i] = ml.bf16_to_f32(v)
			}
		}
		t_total := time.tick_since(t_step)
		ml.clear()
		fmt.printfln("  token %v (id=%v): record=%.2f ms  sync+read=%.2f ms  total=%.2f ms",
			pos, token,
			time.duration_milliseconds(t_record),
			time.duration_milliseconds(t_total - t_record),
			time.duration_milliseconds(t_total))
	}
	total_s := time.duration_seconds(time.tick_since(t_total))
	fmt.printfln("Total: %.2f s for %v tokens (%.2f tok/s)", total_s, builtin.len(tokens), f64(builtin.len(tokens)) / total_s)

	// Steady-state decode benchmark. Drives forward_cached with the last
	// token from the prompt; cache keeps growing past prompt length. The
	// 5-token parity prompt is too short to measure small GPU changes
	// reliably, so we follow up with a longer warm + timed run.
	BENCH_WARMUP :: 32
	BENCH_RUN    :: 128
	last_tok := tokens[builtin.len(tokens) - 1]
	for _ in 0 ..< BENCH_WARMUP {
		_ = gemma.forward_cached(model, &cache, []int{last_tok})
		ml.clear()
	}
	gpu.reset_timing()
	t_bench := time.tick_now()
	for _ in 0 ..< BENCH_RUN {
		_ = gemma.forward_cached(model, &cache, []int{last_tok})
		ml.clear()
	}
	bench_s := time.duration_seconds(time.tick_since(t_bench))
	fmt.printfln("Bench: %v warmup + %v timed -> %.2f tok/s (%.2f ms/tok)",
		BENCH_WARMUP, BENCH_RUN, f64(BENCH_RUN) / bench_s, 1000.0 * bench_s / f64(BENCH_RUN))

	// Per-pipeline GPU timing aggregated over tokens 1..N (excludes warm-up).
	fmt.println()
	gpu.dump_timing()

	any_failed := false
	for position in 0 ..< builtin.len(tokens) {
		row_offset := position * vocab_size
		row_ours   := all_logits     [row_offset : row_offset + vocab_size]
		row_theirs := expected_logits[row_offset : row_offset + vocab_size]

		ours_top   := top_k(row_ours,   5)
		theirs_top := top_k(row_theirs, 5)

		overlap := 0
		for id in ours_top {
			for tid in theirs_top {
				if id == tid { overlap += 1; break }
			}
		}

		fmt.printfln("pos %v  ours top-5 = %v   theirs top-5 = %v   overlap=%v/5",
			position, ours_top, theirs_top, overlap)
		if overlap < 3 {
			any_failed = true
		}
	}

	max_abs_diff: f32
	mean_abs_diff: f64
	for i in 0 ..< builtin.len(all_logits) {
		d := math.abs(all_logits[i] - expected_logits[i])
		if d > max_abs_diff {
			max_abs_diff = d
		}
		mean_abs_diff += f64(d)
	}
	mean_abs_diff /= f64(builtin.len(all_logits))
	fmt.printfln("Logits diff vs HF bf16 reference: max abs = %.4f, mean abs = %.4f",
		max_abs_diff, f32(mean_abs_diff))

	if any_failed {
		fmt.eprintfln("FAIL: at least one position had < 3/5 top-5 overlap")
		os.exit(1)
	}
	fmt.println("PASS")
}

load_tokens :: proc(path: string) -> ([]int, bool) {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		return nil, false
	}
	defer delete(bytes)

	count := int((^u32le)(raw_data(bytes))^)
	if 4 + count * 4 > builtin.len(bytes) {
		return nil, false
	}

	out := make([]int, count)
	for i in 0 ..< count {
		out[i] = int((^i32)(&bytes[4 + i * 4])^)
	}
	return out, true
}

load_tensor :: proc(path: string) -> ([]int, []f32, bool) {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		return nil, nil, false
	}
	defer delete(bytes)

	if string(bytes[0:4]) != "TNSR" {
		return nil, nil, false
	}

	rank := int((^u32le)(&bytes[4])^)
	shape := make([]int, rank)
	for i in 0 ..< rank {
		shape[i] = int((^u32le)(&bytes[8 + i * 4])^)
	}

	count := 1
	for axis in shape {
		count *= axis
	}

	header := 8 + rank * 4
	floats := make([]f32, count)
	src    := slice.from_ptr((^f32)(&bytes[header]), count)
	copy(floats, src)
	return shape, floats, true
}

top_k :: proc(values: []f32, k: int) -> []int {
	indices := make([]int, builtin.len(values), context.temp_allocator)
	for i in 0 ..< builtin.len(values) {
		indices[i] = i
	}

	out := make([]int, k)
	for slot in 0 ..< k {
		best := slot
		for j in slot + 1 ..< builtin.len(indices) {
			if values[indices[j]] > values[indices[best]] {
				best = j
			}
		}
		indices[slot], indices[best] = indices[best], indices[slot]
		out[slot] = indices[slot]
	}
	return out
}
