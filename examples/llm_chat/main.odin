package example_chat

import "base:builtin"

import "core:fmt"
import "core:log"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:strings"
import "core:time"

import ml "../../"
import    "../fetch"

Chat_Model :: struct {
	data:        rawptr,
	vocab_size:  int,

	eval:        proc(data: rawptr, tokens: []int, logits_out: []f32),
	encode_turn: proc(data: rawptr, user_text: string) -> []int,
	is_stop:     proc(data: rawptr, token: int) -> bool,
	decode:      proc(data: rawptr, tokens: []int) -> string,
	reset:       proc(data: rawptr),
	destroy:     proc(data: rawptr),
}

Options :: struct {
	arch:           string,
	model_path:     string,
	gguf_path:      string,
	max_new_tokens: int,
	temperature:    f32,
	top_k:          int,
	top_p:          f32,
	t_max:          int,
	threads:        int,
	cpu_arena:      int,
	system_prompt:  string,
}

DEFAULT_SYSTEM :: "You are a helpful AI assistant named SmolLM, trained by Hugging Face."

main :: proc() {
	// Without this the loaders' log.errorf calls are discarded, which turns a
	// missing or malformed weight file into a silent exit with no explanation.
	// .Info so that model load progress is visible; the chat transcript itself
	// is printed directly to stdout and does not go through the logger.
	context.logger = log.create_console_logger(.Info, {.Level, .Terminal_Color})
	defer log.destroy_console_logger(context.logger)

	options := Options{
		arch           = "llama",
		max_new_tokens = 512,
		temperature    = 0.8,
		top_k          = 40,
		top_p          = 0.95,
		t_max          = 4096,
		threads        = 8,
		cpu_arena      = 2 * 1024 * 1024 * 1024,
		system_prompt  = DEFAULT_SYSTEM,
	}
	parse_args(&options)

	ctx := backend_create(options)
	defer backend_destroy(ctx)

	ml.context_scope(ctx)

	model: Chat_Model
	ok: bool
	switch options.arch {
	case "llama":
		model, ok = llama_backend_make(options.model_path, options.t_max, options.system_prompt)
	case "gemma":
		model, ok = gemma_backend_make(options.gguf_path, options.t_max)
	case:
		log.errorf("unknown --arch %q (expected llama or gemma)", options.arch)
		os.exit(1)
	}
	if !ok {
		os.exit(1)
	}
	defer model.destroy(model.data)

	last_row := make([]f32, model.vocab_size)
	defer delete(last_row)

	fmt.println()
	fmt.printfln("%v chat (T=%.2f, top_k=%v, top_p=%.2f, t_max=%v, max_reply=%v).", options.arch, f64(options.temperature), options.top_k, f64(options.top_p), options.t_max, options.max_new_tokens)
	fmt.println("Type your message and press Enter. Commands: :quit, :reset")
	fmt.println()

	reply_tokens: [dynamic]int
	defer delete(reply_tokens)

	input_buffer: [4096]byte
	for {
		fmt.print("> ")
		os.flush(os.stdout)

		line, line_ok := fetch.read_line(input_buffer[:])
		if !line_ok {
			fmt.println()
			break
		}
		line = strings.trim_space(line)
		if builtin.len(line) == 0 {
			continue
		}

		switch line {
		case ":quit", ":exit", ":q":
			return
		case ":reset":
			model.reset(model.data)
			fmt.println("(conversation reset)")
			continue
		}

		free_all(context.temp_allocator)

		new_tokens := model.encode_turn(model.data, line)
		if builtin.len(new_tokens) == 0 {
			log.warn("empty tokenization, skipped")
			continue
		}

		t_prefill := time.tick_now()
		model.eval(model.data, new_tokens, last_row)
		prefill_elapsed := time.duration_seconds(time.tick_since(t_prefill))

		clear(&reply_tokens)
		previous_text_length := 0
		generated := 0
		t_generate := time.tick_now()

		for step in 0 ..< options.max_new_tokens {
			next_id := sample_next(last_row, options.temperature, options.top_k, options.top_p)
			generated += 1

			if model.is_stop(model.data, next_id) {
				break
			}

			append(&reply_tokens, next_id)
			reply_so_far := model.decode(model.data, reply_tokens[:])
			if builtin.len(reply_so_far) > previous_text_length {
				fmt.print(reply_so_far[previous_text_length:])
				os.flush(os.stdout)
				previous_text_length = builtin.len(reply_so_far)
			}

			if step == options.max_new_tokens - 1 {
				break
			}

			single := [1]int{next_id}
			model.eval(model.data, single[:], last_row)
		}

		fmt.println()
		decode_elapsed := time.duration_seconds(time.tick_since(t_generate))
		prefill_rate := f64(builtin.len(new_tokens)) / prefill_elapsed if prefill_elapsed > 0 else 0
		decode_rate  := f64(generated) / decode_elapsed if decode_elapsed > 0 else 0
		fmt.printfln("  [prefill %v tok / %.2f s = %.1f tok/s   decode %v tok / %.2f s = %.1f tok/s]",
			builtin.len(new_tokens), prefill_elapsed, prefill_rate, generated, decode_elapsed, decode_rate)
		fmt.println()
	}
}

_copy_last_row :: proc(logits: ml.Tensor, out: []f32) {
	rows  := logits.shape[0]
	vocab := logits.shape[1]
	if rows == 1 {
		ml.get_data(logits, out)
		return
	}
	buffer := make([]f32, ml.len(logits), context.temp_allocator)
	ml.get_data(logits, buffer)
	copy(out, buffer[(rows - 1) * vocab:])
}

sample_next :: proc(logits: []f32, temperature: f32, top_k: int, top_p: f32) -> int {
	n := builtin.len(logits)
	if temperature <= 0 || top_k == 1 {
		best := 0
		for i in 1 ..< n {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}

	candidate_count := top_k > 0 ? min(top_k, n) : n
	indices := make([]int, candidate_count, context.temp_allocator)

	for i in 0 ..< candidate_count {
		indices[i] = i
	}
	for i := candidate_count / 2 - 1; i >= 0; i -= 1 {
		_sift_down_min_logit(indices, logits, i, candidate_count)
	}
	for i in candidate_count ..< n {
		if logits[i] > logits[indices[0]] {
			indices[0] = i
			_sift_down_min_logit(indices, logits, 0, candidate_count)
		}
	}
	for end := candidate_count - 1; end > 0; end -= 1 {
		indices[0], indices[end] = indices[end], indices[0]
		_sift_down_min_logit(indices, logits, 0, end)
	}

	max_logit := logits[indices[0]]
	probabilities := make([]f32, candidate_count, context.temp_allocator)
	sum: f32
	for slot in 0 ..< candidate_count {
		probabilities[slot] = math.exp_f32((logits[indices[slot]] - max_logit) / temperature)
		sum += probabilities[slot]
	}
	for slot in 0 ..< candidate_count {
		probabilities[slot] /= sum
	}

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
		for slot in 0 ..< keep {
			new_sum += probabilities[slot]
		}
		if new_sum > 0 {
			for slot in 0 ..< keep {
				probabilities[slot] /= new_sum
			}
		}
	}

	r := rand.float32()
	cumulative: f32
	for slot in 0 ..< keep {
		cumulative += probabilities[slot]
		if r <= cumulative {
			return indices[slot]
		}
	}
	return indices[keep - 1]
}

_sift_down_min_logit :: proc(indices: []int, logits: []f32, start, n: int) {
	root := start
	for {
		child := 2 * root + 1
		if child >= n {
			return
		}
		if child + 1 < n && logits[indices[child + 1]] < logits[indices[child]] {
			child += 1
		}
		if logits[indices[root]] <= logits[indices[child]] {
			return
		}
		indices[root], indices[child] = indices[child], indices[root]
		root = child
	}
}

parse_args :: proc(options: ^Options) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		take_value :: proc(args: []string, i: int) -> string {
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			return args[i + 1]
		}
		switch arg {
		case "--arch":        options.arch           = take_value(args, i);                    i += 2
		case "--model":       options.model_path     = take_value(args, i);                    i += 2
		case "--gguf":        options.gguf_path      = take_value(args, i);                    i += 2
		case "--max-tokens":  options.max_new_tokens = _parse_int(take_value(args, i));        i += 2
		case "--temperature": options.temperature    = f32(_parse_float(take_value(args, i))); i += 2
		case "--top-k":       options.top_k          = _parse_int(take_value(args, i));        i += 2
		case "--top-p":       options.top_p          = f32(_parse_float(take_value(args, i))); i += 2
		case "--t-max":       options.t_max          = _parse_int(take_value(args, i));        i += 2
		case "--threads":     options.threads        = _parse_int(take_value(args, i));        i += 2
		case "--cpu-arena":   options.cpu_arena      = _parse_int(take_value(args, i));        i += 2
		case "--system":      options.system_prompt  = take_value(args, i);                    i += 2
		case "--help", "-h":  _usage_exit()
		case:
			log.errorf("unknown argument: %v", arg)
			_usage_exit()
		}
	}
}

_usage_exit :: proc() {
	fmt.eprintln("usage: example_chat [--arch llama|gemma] [--model PATH] [--gguf PATH] [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--t-max N] [--threads N] [--cpu-arena BYTES] [--system TEXT]")
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
			log.errorf("invalid integer: %q", s)
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
	in_frac  := false
	negative := false
	cursor   := 0
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
			if in_frac {
				scale *= 10
			}
		} else {
			log.errorf("invalid float: %q", s)
			os.exit(1)
		}
		cursor += 1
	}
	out := value / scale
	return -out if negative else out
}
