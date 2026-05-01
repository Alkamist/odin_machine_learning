package gemma_inference

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
import "core:time"

import ml    "../.."
import gpu   "../../backends/gpu"
import gemma "../../networks/gemma"

DATA_DIR        :: "gemma_data"
MODEL_PATH      :: DATA_DIR + "/model.safetensors"
TOKENS_PATH     :: DATA_DIR + "/prompt_tokens.bin"
LOGITS_PATH     :: DATA_DIR + "/expected_logits.bin"
HIDDEN_PATH     :: DATA_DIR + "/expected_final_hidden.bin"

LOGITS_TOLERANCE :: 5.0  // bf16 weights + bf16 GPU compute; loosen vs F32 CPU.

main :: proc() {
	defer fmt.println("Done.")

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	tokens := load_tokens(TOKENS_PATH) or_else _fatal("could not load tokens; run `python tools/gemma_dump.py` first")
	defer delete(tokens)

	expected_shape, expected_logits := load_tensor(LOGITS_PATH) or_else _fatal_t("could not load expected logits")
	defer delete(expected_shape)
	defer delete(expected_logits)

	fmt.printfln("Loaded %v prompt tokens.", builtin.len(tokens))
	fmt.printfln("Expected logits shape = %v.", expected_shape)

	fmt.println("Allocating Gemma 4 E4B (bf16, GPU) ...")
	cfg := gemma.make_e4b_config()
	defer gemma.config_destroy(cfg)
	model := gemma.make(cfg, .Bf16)
	defer gemma.destroy(model)

	fmt.println("Loading weights from", MODEL_PATH, "...")
	t_load := time.tick_now()
	if !gemma.load_safetensors(model, MODEL_PATH) {
		fmt.eprintln("Weight loading failed.")
		os.exit(1)
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	fmt.println("Running forward ...")
	t_forward := time.tick_now()
	logits, final_hidden := gemma.forward_with_hidden(model, tokens)
	logits_buf := make([]f32, ml.len(logits))
	defer delete(logits_buf)
	ml.get_data(logits, logits_buf)
	hidden_buf := make([]f32, ml.len(final_hidden))
	defer delete(hidden_buf)
	ml.get_data(final_hidden, hidden_buf)
	fmt.printfln("  forward done in %.2f s", f64(time.duration_seconds(time.tick_since(t_forward))))

	if expected_hidden_shape, expected_hidden, hidden_ok := load_tensor(HIDDEN_PATH); hidden_ok {
		defer delete(expected_hidden_shape)
		defer delete(expected_hidden)
		max_h_diff:  f32
		max_h_at:    int
		mean_h_diff: f32
		for i in 0 ..< builtin.len(hidden_buf) {
			diff := math.abs(hidden_buf[i] - expected_hidden[i])
			if diff > max_h_diff {
				max_h_diff = diff
				max_h_at   = i
			}
			mean_h_diff += diff
		}
		mean_h_diff /= f32(builtin.len(hidden_buf))
		fmt.printfln("Pre-lm_head hidden parity: shape=%v max abs diff=%.4e at %v (got %.5f vs expected %.5f), mean abs diff=%.4e",
			expected_hidden_shape, max_h_diff, max_h_at, hidden_buf[max_h_at], expected_hidden[max_h_at], mean_h_diff)
	} else {
		fmt.println("(no expected_final_hidden.bin — re-run gemma_dump.py to enable hidden-state diagnostic)")
	}

	if !slice.equal(logits.shape[:logits.rank], expected_shape) {
		fmt.eprintfln("logits shape %v != expected %v", logits.shape[:logits.rank], expected_shape)
		os.exit(1)
	}

	max_abs_diff: f32
	max_diff_at:  int
	for i in 0 ..< builtin.len(logits_buf) {
		diff := math.abs(logits_buf[i] - expected_logits[i])
		if diff > max_abs_diff {
			max_abs_diff = diff
			max_diff_at  = i
		}
	}

	vocab_size := expected_shape[1]
	fmt.printfln(
		"Logits parity: max abs diff = %.4e at index %v (got %.6f vs expected %.6f).",
		max_abs_diff, max_diff_at, logits_buf[max_diff_at], expected_logits[max_diff_at],
	)

	mean_abs_diff: f32
	for i in 0 ..< builtin.len(logits_buf) {
		mean_abs_diff += math.abs(logits_buf[i] - expected_logits[i])
	}
	mean_abs_diff /= f32(builtin.len(logits_buf))
	fmt.printfln("Logits parity: mean abs diff = %.4e", mean_abs_diff)

	for position in 0 ..< logits.shape[0] {
		row_offset := position * vocab_size
		row_ours   := logits_buf[row_offset:row_offset + vocab_size]
		row_theirs := expected_logits[row_offset:row_offset + vocab_size]
		ours_top   := top_k(row_ours,   5)
		theirs_top := top_k(row_theirs, 5)
		fmt.printfln("  pos %v  ours top-5 ids = %v, theirs = %v", position, ours_top, theirs_top)
		for id in ours_top {
			fmt.printfln("           id %v: ours = %.5f  theirs = %.5f  ratio = %.4f",
				id, row_ours[id], row_theirs[id],
				row_ours[id] / row_theirs[id] if math.abs(row_theirs[id]) > 1e-3 else 0)
		}
	}

	if max_abs_diff > LOGITS_TOLERANCE {
		fmt.eprintfln("FAIL: logits diverged beyond tolerance %.3f", f32(LOGITS_TOLERANCE))
		os.exit(1)
	}
	fmt.printfln("PASS: logits match HF reference within %.3f.", f32(LOGITS_TOLERANCE))
}

_fatal :: proc(message: string) -> []int {
	fmt.eprintln(message)
	os.exit(1)
}

_fatal_t :: proc(message: string) -> ([]int, []f32) {
	fmt.eprintln(message)
	os.exit(1)
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
