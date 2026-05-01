package gemma_chat

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml    "../.."
import gpu   "../../backends/gpu"
import gemma "../../networks/gemma"
import tok   "../../tokenizers/gemma"

DATA_DIR       :: "gemma_data"
MODEL_PATH     :: DATA_DIR + "/model.safetensors"
TOKENIZER_PATH :: DATA_DIR + "/tokenizer.json"

DEFAULT_PROMPT      :: "The capital of France is"
DEFAULT_MAX_TOKENS  :: 32
DEFAULT_TEMPERATURE :: 0.0
DEFAULT_TOP_K       :: 0
DEFAULT_TOP_P       :: f32(1.0) // 1.0 = disabled (keep all probability mass)
DEFAULT_T_MAX       :: 1024
DEFAULT_IGNORE_EOS  :: false
DEFAULT_NO_CACHE    :: false
DEFAULT_CHAT        :: false

EOS_TOKEN_ID :: 1 // Gemma 4 SentencePiece <eos>
END_OF_TURN_TEXT :: "<turn|>" // Gemma 4 IT closing turn marker (id 106)

main :: proc() {
	prompt          := DEFAULT_PROMPT
	max_new_tokens  := DEFAULT_MAX_TOKENS
	temperature     := f32(DEFAULT_TEMPERATURE)
	top_k           := DEFAULT_TOP_K
	top_p           := f32(DEFAULT_TOP_P)
	t_max           := DEFAULT_T_MAX
	ignore_eos      := DEFAULT_IGNORE_EOS
	no_cache        := DEFAULT_NO_CACHE
	use_chat        := DEFAULT_CHAT

	parse_args(&prompt, &max_new_tokens, &temperature, &top_k, &top_p, &t_max, &ignore_eos, &no_cache, &use_chat)

	if use_chat {
		// Gemma 4 IT chat template (`<|turn>` / `<turn|>` are the turn
		// markers, not the older `<start_of_turn>` / `<end_of_turn>` from
		// Gemma 2/3). Wrapping is required for IT weights to produce
		// coherent assistant responses.
		prompt = fmt.aprintf("<bos><|turn>user\n%v<turn|>\n<|turn>model\n", prompt)
	}

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	fmt.println("Loading tokenizer ...")
	tokenizer, tokenizer_ok := tok.load(TOKENIZER_PATH)
	if !tokenizer_ok {
		fmt.eprintln("FAIL: could not load tokenizer.")
		os.exit(1)
	}
	defer tok.destroy(tokenizer)

	fmt.println("Allocating Gemma 4 E4B (bf16, GPU) ...")
	cfg := gemma.make_e4b_config()
	defer gemma.config_destroy(cfg)
	model := gemma.make(cfg, .Bf16)
	defer gemma.destroy(model)

	fmt.println("Loading weights ...")
	t_load := time.tick_now()
	if !gemma.load_safetensors(model, MODEL_PATH) {
		fmt.eprintln("FAIL: weight loading failed.")
		os.exit(1)
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	prompt_tokens := tok.encode(&tokenizer, prompt)
	defer delete(prompt_tokens)
	if builtin.len(prompt_tokens) == 0 {
		fmt.eprintln("FAIL: prompt tokenized to zero tokens.")
		os.exit(1)
	}

	all_tokens: [dynamic]int
	defer delete(all_tokens)
	append(&all_tokens, ..prompt_tokens)

	fmt.printfln("Prompt = %q", prompt)
	fmt.printfln("  %v prompt tokens, generating up to %v new tokens (T=%.2f, top_k=%v, top_p=%.2f, t_max=%v)",
		builtin.len(prompt_tokens), max_new_tokens, f64(temperature), top_k, f64(top_p), t_max)
	fmt.println("---")
	prompt_text := tok.decode(&tokenizer, prompt_tokens)
	defer delete(prompt_text)
	fmt.print(prompt_text)
	os.flush(os.stdout)

	if builtin.len(prompt_tokens) + max_new_tokens > t_max {
		fmt.eprintfln("FAIL: prompt + max_new_tokens (%v) exceeds t_max (%v); rerun with --t-max", builtin.len(prompt_tokens) + max_new_tokens, t_max)
		os.exit(1)
	}

	cache := gemma.cache_make(model, t_max)
	defer gemma.cache_destroy(cache)

	end_of_turn_id := tokenizer.added_tokens[END_OF_TURN_TEXT] if END_OF_TURN_TEXT in tokenizer.added_tokens else -1

	vocab_size := cfg.vocab_size
	last_row := builtin.make([]f32, vocab_size)
	defer delete(last_row)
	prefill_buf := builtin.make([]f32, builtin.len(prompt_tokens) * vocab_size)
	defer delete(prefill_buf)

	previous_decoded_length := builtin.len(prompt_text)
	t_generate := time.tick_now()

	{
		ml.clear()
		logits: ml.Tensor
		if no_cache {
			logits = gemma.forward(model, all_tokens[:])
		} else {
			logits = gemma.forward_cached(model, &cache, prompt_tokens)
		}
		buf := make([]f32, ml.len(logits), context.temp_allocator)
		ml.get_data(logits, buf)
		copy(last_row, buf[(builtin.len(all_tokens) - 1) * vocab_size :])
	}

	generated := 0
	for step in 0 ..< max_new_tokens {
		next_id := sample_next(last_row, temperature, top_k, top_p)
		append(&all_tokens, next_id)
		generated += 1

		current_decoded := tok.decode(&tokenizer, all_tokens[:])
		defer delete(current_decoded)
		if builtin.len(current_decoded) > previous_decoded_length {
			fmt.print(current_decoded[previous_decoded_length:])
			os.flush(os.stdout)
		}
		previous_decoded_length = builtin.len(current_decoded)

		if !ignore_eos {
			if next_id == EOS_TOKEN_ID {
				fmt.println()
				fmt.println("(stopped on <eos>)")
				break
			}
			if next_id == end_of_turn_id {
				fmt.println()
				fmt.println("(stopped on <end_of_turn>)")
				break
			}
		}
		if step == max_new_tokens - 1 do break

		ml.clear()
		if no_cache {
			logits := gemma.forward(model, all_tokens[:])
			buf := make([]f32, ml.len(logits), context.temp_allocator)
			ml.get_data(logits, buf)
			copy(last_row, buf[(builtin.len(all_tokens) - 1) * vocab_size :])
		} else {
			single := [1]int{next_id}
			logits := gemma.forward_cached(model, &cache, single[:])
			ml.get_data(logits, last_row)
		}
	}
	fmt.println()
	fmt.println("---")
	elapsed := f64(time.duration_seconds(time.tick_since(t_generate)))
	fmt.printfln("generated %v tokens in %.2f s (%.1f tok/s)",
		generated, elapsed, f64(generated) / elapsed if elapsed > 0 else 0)
}

sample_next :: proc(logits: []f32, temperature: f32, top_k: int, top_p: f32) -> int {
	if temperature <= 0 || top_k == 1 {
		best := 0
		for i in 1 ..< builtin.len(logits) do if logits[i] > logits[best] do best = i
		return best
	}

	// Selection-sort the top-K (or full vocab if top_k <= 0) into the front
	// of `indices` in descending logit order, so prefixes of `indices` are
	// the leading candidates and we can apply top-p on top.
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

	// Top-p (nucleus) trim: keep the smallest prefix whose cumulative
	// probability ≥ top_p, drop the tail. Always keep at least the top-1.
	keep := candidate_count
	if top_p > 0 && top_p < 1 {
		cumulative: f32
		for slot in 0 ..< candidate_count {
			cumulative += probabilities[slot]
			if cumulative >= top_p {
				keep = slot + 1
				break
			}
		}
		// Renormalise the kept prefix.
		new_sum: f32
		for slot in 0 ..< keep do new_sum += probabilities[slot]
		if new_sum > 0 {
			for slot in 0 ..< keep do probabilities[slot] /= new_sum
		}
	}

	r := rand.float32()
	cumulative: f32
	for slot in 0 ..< keep {
		cumulative += probabilities[slot]
		if r <= cumulative do return indices[slot]
	}
	return indices[keep - 1]
}

parse_args :: proc(prompt: ^string, max_new_tokens: ^int, temperature: ^f32, top_k: ^int, top_p: ^f32, t_max: ^int, ignore_eos, no_cache, use_chat: ^bool) {
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
		case "--top-p":
			if i + 1 >= builtin.len(args) do _usage_exit()
			top_p^ = f32(_parse_float(args[i + 1]))
			i += 2
		case "--t-max":
			if i + 1 >= builtin.len(args) do _usage_exit()
			t_max^ = _parse_int(args[i + 1])
			i += 2
		case "--ignore-eos":
			ignore_eos^ = true
			i += 1
		case "--no-cache":
			no_cache^ = true
			i += 1
		case "--chat":
			use_chat^ = true
			i += 1
		case "--help", "-h":
			_usage_exit()
		case:
			fmt.eprintfln("unknown argument: %v", arg)
			_usage_exit()
		}
	}
}

_usage_exit :: proc() {
	fmt.eprintln("usage: gemma_chat [--prompt TEXT] [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--t-max N] [--ignore-eos] [--no-cache] [--chat]")
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
