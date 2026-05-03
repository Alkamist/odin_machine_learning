package gguf_gemma_parity

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import gemma "../../networks/gemma"

// Parity test: run a forward of the GGUF Q4_K_M Gemma 4 E4B model on the
// fixed prompt in `gemma_data/prompt_tokens.bin` and compare against the
// HF-reference logits in `gemma_data/expected_logits.bin` (generated from
// the bf16 safetensors weights via tools/gemma_dump.py). The Q4_K_M model
// won't match bit-exactly — quantization noise is real — but the top-5
// per-position IDs should overlap heavily and ranks should track.

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
		fmt.eprintln("could not load prompt_tokens.bin (run tools/gemma_dump.py first)")
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

	fmt.printfln("Loaded %v prompt tokens.",      builtin.len(tokens))
	fmt.printfln("Expected logits shape = %v.",   expected_shape)

	ctx := cpu.context_create(512 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
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

	fmt.printfln("Running forward on %v tokens (CPU; will take a while)...", builtin.len(tokens))
	t_fwd := time.tick_now()
	logits := gemma.forward(model, tokens)
	fmt.printfln("  forward done in %.1f s", time.duration_seconds(time.tick_since(t_fwd)))

	logits_buf := make([]f32, ml.len(logits))
	defer delete(logits_buf)
	if logits.type == .F32 {
		ml.get_data(logits, logits_buf)
	} else {
		bf := make([]ml.Bf16, ml.len(logits))
		defer delete(bf)
		ml.get_data_bytes(logits, slice.bytes_from_ptr(raw_data(bf), ml.len(logits) * 2))
		for v, i in bf do logits_buf[i] = ml.bf16_to_f32(v)
	}

	if !slice.equal(logits.shape[:logits.rank], expected_shape) {
		fmt.eprintfln("FAIL: logits shape %v != expected %v", logits.shape[:logits.rank], expected_shape)
		os.exit(1)
	}

	vocab_size := expected_shape[1]
	any_failed := false

	for position in 0 ..< logits.shape[0] {
		row_offset := position * vocab_size
		row_ours   := logits_buf    [row_offset : row_offset + vocab_size]
		row_theirs := expected_logits[row_offset : row_offset + vocab_size]

		ours_top   := top_k(row_ours,   5)
		theirs_top := top_k(row_theirs, 5)

		// How many of the top-5 IDs we share with the reference at this
		// position. >=3 is healthy under quantization noise.
		overlap := 0
		for id in ours_top {
			for tid in theirs_top do if id == tid { overlap += 1; break }
		}

		fmt.printfln("pos %v  ours top-5 = %v   theirs top-5 = %v   overlap=%v/5",
			position, ours_top, theirs_top, overlap)
		for id in ours_top {
			r_t := row_theirs[id]
			fmt.printfln("    id %v: ours=%.4f  theirs=%.4f", id, row_ours[id], r_t)
		}

		if overlap < 3 {
			any_failed = true
		}
	}

	// Mean / max abs diff over the whole logit tensor.
	max_abs_diff: f32
	mean_abs_diff: f64
	for i in 0 ..< builtin.len(logits_buf) {
		d := math.abs(logits_buf[i] - expected_logits[i])
		if d > max_abs_diff do max_abs_diff = d
		mean_abs_diff += f64(d)
	}
	mean_abs_diff /= f64(builtin.len(logits_buf))
	fmt.printfln("Logits diff vs HF bf16 reference: max abs = %.4f, mean abs = %.4f",
		max_abs_diff, f32(mean_abs_diff))

	if any_failed {
		fmt.eprintfln("FAIL: at least one position had < 3/5 top-5 overlap with HF reference")
		os.exit(1)
	}
	fmt.println("PASS: top-5 overlap healthy across all positions")
}

load_tokens :: proc(path: string) -> ([]int, bool) {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil do return nil, false
	defer delete(bytes)

	count := int((^u32le)(raw_data(bytes))^)
	if 4 + count * 4 > builtin.len(bytes) do return nil, false

	out := make([]int, count)
	for i in 0 ..< count {
		out[i] = int((^i32)(&bytes[4 + i * 4])^)
	}
	return out, true
}

load_tensor :: proc(path: string) -> ([]int, []f32, bool) {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil do return nil, nil, false
	defer delete(bytes)

	if string(bytes[0:4]) != "TNSR" do return nil, nil, false

	rank := int((^u32le)(&bytes[4])^)
	shape := make([]int, rank)
	for i in 0 ..< rank {
		shape[i] = int((^u32le)(&bytes[8 + i * 4])^)
	}

	count := 1
	for axis in shape do count *= axis

	header := 8 + rank * 4
	floats := make([]f32, count)
	src    := slice.from_ptr((^f32)(&bytes[header]), count)
	copy(floats, src)
	return shape, floats, true
}

top_k :: proc(values: []f32, k: int) -> []int {
	indices := make([]int, builtin.len(values), context.temp_allocator)
	for i in 0 ..< builtin.len(values) do indices[i] = i

	out := make([]int, k)
	for slot in 0 ..< k {
		best := slot
		for j in slot + 1 ..< builtin.len(indices) {
			if values[indices[j]] > values[indices[best]] do best = j
		}
		indices[slot], indices[best] = indices[best], indices[slot]
		out[slot] = indices[slot]
	}
	return out
}
