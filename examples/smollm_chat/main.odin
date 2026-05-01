package smollm_chat

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import gpu   "../../backends/gpu"
import llama "../../networks/llama"
import gpt2  "../../tokenizers/gpt2"

DATA_DIR       :: "smollm_data"
MODEL_PATH     :: DATA_DIR + "/model.safetensors"
TOKENIZER_PATH :: DATA_DIR + "/tokenizer.json"

DEFAULT_PROMPT      :: "The capital of France is"
DEFAULT_MAX_TOKENS  :: 32
DEFAULT_TEMPERATURE :: 0.0
DEFAULT_TOP_K       :: 0

main :: proc() {
	prompt          := DEFAULT_PROMPT
	max_new_tokens  := DEFAULT_MAX_TOKENS
	temperature     := f32(DEFAULT_TEMPERATURE)
	top_k           := DEFAULT_TOP_K

	parse_args(&prompt, &max_new_tokens, &temperature, &top_k)

	cpu.set_thread_count(8)

	ctx := cpu.context_create(1024 * 1024 * 256)
	defer cpu.context_destroy(ctx)

	// ctx := gpu.context_create()
	// defer gpu.context_destroy(ctx)

	ml.context_scope(ctx)

	fmt.println("Loading tokenizer ...")
	tok, tok_ok := gpt2.load(TOKENIZER_PATH)
	if !tok_ok {
		fmt.eprintln("FAIL: could not load tokenizer.")
		os.exit(1)
	}
	defer gpt2.destroy(tok)

	fmt.println("Allocating SmolLM2-135M ...")
	model := llama.make(llama.SMOLLM2_135M_CONFIG)
	defer llama.destroy(model)

	fmt.println("Loading weights ...")
	t_load := time.tick_now()
	if !llama.load_safetensors(model, MODEL_PATH) {
		fmt.eprintln("FAIL: weight loading failed.")
		os.exit(1)
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	tokens := gpt2.encode(&tok, prompt)
	defer delete(tokens)
	if builtin.len(tokens) == 0 {
		fmt.eprintln("FAIL: prompt tokenized to zero tokens.")
		os.exit(1)
	}

	all_tokens: [dynamic]int
	defer delete(all_tokens)
	append(&all_tokens, ..tokens)

	fmt.printfln("Prompt = %q", prompt)
	fmt.printfln("  %v prompt tokens, generating up to %v new tokens (T=%.2f, top_k=%v)", builtin.len(tokens), max_new_tokens, f64(temperature), top_k)
	fmt.println("---")
	prompt_text := gpt2.decode(&tok, tokens)
	defer delete(prompt_text)
	fmt.print(prompt_text)

	logits_buffer: [dynamic]f32
	defer delete(logits_buffer)

	previous_decoded_length := builtin.len(prompt_text)
	t_generate := time.tick_now()
	for step in 0 ..< max_new_tokens {
		ml.clear()

		logits     := llama.forward(model, all_tokens[:])
		token_count := logits.shape[0]
		vocab_size  := logits.shape[1]
		total_floats := token_count * vocab_size

		resize(&logits_buffer, total_floats)
		ml.get_data(logits, logits_buffer[:])

		last_row := logits_buffer[(token_count - 1) * vocab_size : token_count * vocab_size]
		next_id  := sample_next(last_row, temperature, top_k)
		append(&all_tokens, next_id)

		current_decoded := gpt2.decode(&tok, all_tokens[:])
		defer delete(current_decoded)
		if builtin.len(current_decoded) > previous_decoded_length {
			fmt.print(current_decoded[previous_decoded_length:])
			os.flush(os.stdout)
		}
		previous_decoded_length = builtin.len(current_decoded)
		_ = step
	}
	fmt.println()
	fmt.println("---")
	fmt.printfln("generated %v tokens in %.2f s (%.1f tok/s)", max_new_tokens, f64(time.duration_seconds(time.tick_since(t_generate))), f64(max_new_tokens) / f64(time.duration_seconds(time.tick_since(t_generate))))
}

sample_next :: proc(logits: []f32, temperature: f32, top_k: int) -> int {
	if temperature <= 0 || top_k == 1 {
		best := 0
		for i in 1 ..< builtin.len(logits) do if logits[i] > logits[best] do best = i
		return best
	}

	candidate_count := top_k > 0 ? min(top_k, builtin.len(logits)) : builtin.len(logits)

	indices := make([]int, builtin.len(logits), context.temp_allocator)
	for i in 0 ..< builtin.len(logits) do indices[i] = i

	for slot in 0 ..< candidate_count {
		best := slot
		for i in slot + 1 ..< builtin.len(indices) {
			if logits[indices[i]] > logits[indices[best]] do best = i
		}
		indices[slot], indices[best] = indices[best], indices[slot]
	}

	max_logit := logits[indices[0]]
	probabilities := make([]f32, candidate_count, context.temp_allocator)
	sum: f32
	for slot in 0 ..< candidate_count {
		probabilities[slot] = math.exp_f32((logits[indices[slot]] - max_logit) / temperature)
		sum += probabilities[slot]
	}
	for slot in 0 ..< candidate_count do probabilities[slot] /= sum

	r := rand.float32()
	cumulative: f32
	for slot in 0 ..< candidate_count {
		cumulative += probabilities[slot]
		if r <= cumulative do return indices[slot]
	}
	return indices[candidate_count - 1]
}

parse_args :: proc(prompt: ^string, max_new_tokens: ^int, temperature: ^f32, top_k: ^int) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--prompt":
			if i + 1 >= builtin.len(args) do _usage_exit()
			prompt^ = args[i + 1]
			i += 2
		case "--max-tokens":
			if i + 1 >= builtin.len(args) do _usage_exit()
			max_new_tokens^ = _parse_int(args[i + 1])
			i += 2
		case "--temperature":
			if i + 1 >= builtin.len(args) do _usage_exit()
			temperature^ = f32(_parse_float(args[i + 1]))
			i += 2
		case "--top-k":
			if i + 1 >= builtin.len(args) do _usage_exit()
			top_k^ = _parse_int(args[i + 1])
			i += 2
		case "--help", "-h":
			_usage_exit()
		case:
			fmt.eprintfln("unknown argument: %v", arg)
			_usage_exit()
		}
	}
}

_usage_exit :: proc() {
	fmt.eprintln("usage: smollm_chat [--prompt TEXT] [--max-tokens N] [--temperature T] [--top-k K]")
	os.exit(1)
}

_parse_int :: proc(s: string) -> int {
	value: int
	negative := false
	cursor := 0
	if builtin.len(s) > 0 && s[0] == '-' {
		negative = true
		cursor = 1
	}
	for cursor < builtin.len(s) {
		c := s[cursor]
		if c < '0' || c > '9' {
			fmt.eprintfln("invalid integer: %q", s)
			os.exit(1)
		}
		value = value * 10 + int(c - '0')
		cursor += 1
	}
	return -value if negative else value
}

_parse_float :: proc(s: string) -> f64 {
	value:   f64 = 0
	scale:   f64 = 1
	in_frac      := false
	negative     := false
	cursor       := 0
	if builtin.len(s) > 0 && s[0] == '-' {
		negative = true
		cursor = 1
	}
	for cursor < builtin.len(s) {
		c := s[cursor]
		if c == '.' {
			in_frac = true
		} else if c >= '0' && c <= '9' {
			value = value * 10 + f64(c - '0')
			if in_frac do scale *= 10
		} else {
			fmt.eprintfln("invalid float: %q", s)
			os.exit(1)
		}
		cursor += 1
	}
	out := value / scale
	return -out if negative else out
}
