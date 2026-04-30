package smollm_inference

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import llama "../../networks/llama"

DATA_DIR     :: "smollm_data"
MODEL_PATH   :: DATA_DIR + "/model.safetensors"
TOKENS_PATH  :: DATA_DIR + "/prompt_tokens.bin"
LOGITS_PATH  :: DATA_DIR + "/expected_logits.bin"

LOGITS_TOLERANCE :: 0.5  // HF F32 vs our F32 routed through bf16 conversion + accumulator order.

main :: proc() {
	defer fmt.println("Done.")

	cpu.set_thread_count(8)
	ctx := cpu.context_create(1024 * 1024 * 64)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	tokens := load_tokens(TOKENS_PATH) or_else _fatal("could not load tokens; run `python tools/smollm_dump.py` first")
	defer delete(tokens)

	expected_shape, expected_logits := load_tensor(LOGITS_PATH) or_else _fatal_t("could not load expected logits")
	defer delete(expected_shape)
	defer delete(expected_logits)

	fmt.printfln("Loaded %v prompt tokens.", builtin.len(tokens))
	fmt.printfln("Expected logits shape = %v.", expected_shape)

	fmt.println("Allocating SmolLM2-135M ...")
	model := llama.make(llama.SMOLLM2_135M_CONFIG)
	defer llama.destroy(model)

	fmt.println("Loading weights from", MODEL_PATH, "...")
	t_load := time.tick_now()
	if !llama.load_safetensors(model, MODEL_PATH) {
		fmt.eprintln("Weight loading failed.")
		os.exit(1)
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	fmt.println("Running forward ...")
	t_forward := time.tick_now()
	logits     := llama.forward(model, tokens)
	logits_buf := make([]f32, ml.len(logits))
	defer delete(logits_buf)
	ml.get_data(logits, logits_buf)
	fmt.printfln("  forward done in %.2f s", f64(time.duration_seconds(time.tick_since(t_forward))))

	if !slice.equal(logits.shape[:logits.rank], expected_shape) {
		fmt.eprintfln(
			"logits shape %v != expected %v",
			logits.shape[:logits.rank], expected_shape,
		)
		os.exit(1)
	}

	max_abs_diff: f32
	max_diff_at: int
	for i in 0 ..< builtin.len(logits_buf) {
		diff := math.abs(logits_buf[i] - expected_logits[i])
		if diff > max_abs_diff {
			max_abs_diff = diff
			max_diff_at  = i
		}
	}

	token_count := tokens
	_ = token_count
	vocab_size := expected_shape[1]

	fmt.printfln(
		"Logits parity: max abs diff = %.4e at index %v (got %.6f vs expected %.6f).",
		max_abs_diff, max_diff_at, logits_buf[max_diff_at], expected_logits[max_diff_at],
	)

	for position in 0 ..< logits.shape[0] {
		row_offset := position * vocab_size
		ours_top   := top_k(logits_buf[row_offset:row_offset + vocab_size], 5)
		theirs_top := top_k(expected_logits[row_offset:row_offset + vocab_size], 5)
		fmt.printfln("  pos %v  ours top-5 ids = %v, theirs = %v", position, ours_top, theirs_top)
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

	// Partial selection: pick the top k via repeated max-find.
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
