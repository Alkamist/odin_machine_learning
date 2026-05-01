package gemma_chat_repl

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:strings"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import gpu   "../../backends/gpu"
import gemma "../../networks/gemma"
import tok   "../../tokenizers/gemma"

DATA_DIR       :: "gemma_data"
MODEL_PATH     :: DATA_DIR + "/model.safetensors"
TOKENIZER_PATH :: DATA_DIR + "/tokenizer.json"

DEFAULT_MAX_TOKENS  :: 512
DEFAULT_TEMPERATURE :: f32(0.8)
DEFAULT_TOP_K       :: 40
DEFAULT_TOP_P       :: f32(0.95)
DEFAULT_T_MAX       :: 4096
DEFAULT_CPU_ARENA   :: 2 * 1024 * 1024 * 1024

EOS_TOKEN_ID     :: 1
END_OF_TURN_TEXT :: "<turn|>"
BOS_TEXT         :: "<bos>"

main :: proc() {
	max_new_tokens := DEFAULT_MAX_TOKENS
	temperature    := DEFAULT_TEMPERATURE
	top_k          := DEFAULT_TOP_K
	top_p          := DEFAULT_TOP_P
	t_max          := DEFAULT_T_MAX
	use_cpu        := false
	cpu_arena      := DEFAULT_CPU_ARENA

	parse_args(&max_new_tokens, &temperature, &top_k, &top_p, &t_max, &use_cpu, &cpu_arena)

	cpu.set_thread_count(24)

	ctx := cpu.context_create(cpu_arena) if use_cpu else gpu.context_create()
	defer {
		if use_cpu {
			cpu.context_destroy(ctx)
		} else {
			gpu.context_destroy(ctx)
		}
	}

	ml.context_scope(ctx)

	fmt.println("Loading tokenizer ...")
	tokenizer, tokenizer_ok := tok.load(TOKENIZER_PATH)
	if !tokenizer_ok {
		fmt.eprintln("FAIL: could not load tokenizer.")
		os.exit(1)
	}
	defer tok.destroy(tokenizer)

	fmt.printfln("Allocating Gemma 4 E4B (bf16, %v) ...", "CPU" if use_cpu else "GPU")
	cfg := gemma.make_e4b_config()
	defer gemma.config_destroy(cfg)
	model := gemma.make(cfg, .Bf16)
	defer gemma.destroy(model)

	fmt.println("Loading weights ...")
	t_load := time.tick_now()
	// Loader stages every weight tensor through context.temp_allocator. The
	// guard rolls the arena back to its pre-load mark on scope exit so the
	// chat process doesn't keep ~10+ GB of stale scratch for its lifetime.
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		if !gemma.load_safetensors(model, MODEL_PATH) {
			fmt.eprintln("FAIL: weight loading failed.")
			os.exit(1)
		}
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	cache := gemma.cache_make(model, t_max)
	defer gemma.cache_destroy(cache)

	end_of_turn_id := tokenizer.added_tokens[END_OF_TURN_TEXT] if END_OF_TURN_TEXT in tokenizer.added_tokens else -1
	bos_id         := tokenizer.added_tokens[BOS_TEXT]         if BOS_TEXT         in tokenizer.added_tokens else -1

	vocab_size := cfg.vocab_size
	last_row := builtin.make([]f32, vocab_size)
	defer delete(last_row)

	all_tokens: [dynamic]int
	defer delete(all_tokens)
	if bos_id >= 0 do append(&all_tokens, bos_id)

	fmt.println()
	fmt.printfln("Gemma 4 chat (T=%.2f, top_k=%v, top_p=%.2f, t_max=%v, max_reply=%v).",
		f64(temperature), top_k, f64(top_p), t_max, max_new_tokens)
	fmt.println("Type your message and press Enter. Commands: :quit, :reset")
	fmt.println()

	input_buffer: [4096]byte
	for {
		fmt.print("> ")
		os.flush(os.stdout)

		line, line_ok := read_line(input_buffer[:])
		if !line_ok {
			fmt.println()
			break
		}
		line = strings.trim_space(line)
		if builtin.len(line) == 0 do continue

		switch line {
		case ":quit", ":exit", ":q":
			return
		case ":reset":
			gemma.cache_reset(&cache)
			clear(&all_tokens)
			if bos_id >= 0 do append(&all_tokens, bos_id)
			fmt.println("(conversation reset)")
			continue
		}

		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

		turn_text := fmt.tprintf("<|turn>user\n%v<turn|>\n<|turn>model\n", line)
		new_tokens := tok.encode(&tokenizer, turn_text)
		defer delete(new_tokens)
		if builtin.len(new_tokens) == 0 {
			fmt.eprintln("(empty tokenization, skipped)")
			continue
		}

		if cache.length + builtin.len(new_tokens) + 1 > t_max {
			fmt.eprintln("(context would overflow t_max; use :reset)")
			continue
		}

		append(&all_tokens, ..new_tokens)

		ml.clear()
		{
			logits := gemma.forward_cached(model, &cache, new_tokens)
			buf := make([]f32, ml.len(logits), context.temp_allocator)
			ml.get_data(logits, buf)
			copy(last_row, buf[(builtin.len(new_tokens) - 1) * vocab_size :])
		}

		reply_start := builtin.len(all_tokens)
		previous_decoded_length := 0
		generated := 0
		t_generate := time.tick_now()

		for step in 0 ..< max_new_tokens {
			if cache.length + 1 > t_max {
				fmt.println()
				fmt.println("(stopped: reached t_max)")
				break
			}

			next_id := sample_next(last_row, temperature, top_k, top_p)
			append(&all_tokens, next_id)
			generated += 1

			if next_id == EOS_TOKEN_ID || next_id == end_of_turn_id {
				break
			}

			reply_so_far := tok.decode(&tokenizer, all_tokens[reply_start:])
			defer delete(reply_so_far)
			if builtin.len(reply_so_far) > previous_decoded_length {
				fmt.print(reply_so_far[previous_decoded_length:])
				os.flush(os.stdout)
				previous_decoded_length = builtin.len(reply_so_far)
			}

			if step == max_new_tokens - 1 do break

			ml.clear()
			single := [1]int{next_id}
			logits := gemma.forward_cached(model, &cache, single[:])
			ml.get_data(logits, last_row)
		}

		fmt.println()
		elapsed := f64(time.duration_seconds(time.tick_since(t_generate)))
		fmt.printfln("  [%v tokens, %.2f s, %.1f tok/s]",
			generated, elapsed, f64(generated) / elapsed if elapsed > 0 else 0)
		fmt.println()
	}
}

read_line :: proc(buffer: []byte) -> (line: string, ok: bool) {
	cursor := 0
	one: [1]byte
	for cursor < builtin.len(buffer) {
		n, err := os.read(os.stdin, one[:])
		if err != nil || n == 0 {
			if cursor == 0 do return "", false
			break
		}
		c := one[0]
		if c == '\n' do break
		buffer[cursor] = c
		cursor += 1
	}
	if cursor > 0 && buffer[cursor - 1] == '\r' do cursor -= 1
	return string(buffer[:cursor]), true
}

sample_next :: proc(logits: []f32, temperature: f32, top_k: int, top_p: f32) -> int {
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

parse_args :: proc(max_new_tokens: ^int, temperature: ^f32, top_k: ^int, top_p: ^f32, t_max: ^int, use_cpu: ^bool, cpu_arena: ^int) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
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
		case "--cpu":
			use_cpu^ = true
			i += 1
		case "--cpu-arena":
			if i + 1 >= builtin.len(args) do _usage_exit()
			cpu_arena^ = _parse_int(args[i + 1])
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
	fmt.eprintln("usage: gemma_chat_repl [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--t-max N] [--cpu] [--cpu-arena BYTES]")
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
	value: f64 = 0
	scale: f64 = 1
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
