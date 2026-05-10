package smollm_chat_repl

// odin run examples/smollm_chat_repl -o:speed -- --model smollm_data/model_instruct.safetensors --cpu --threads 24 --max-tokens 200

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:strings"
import "core:sync"
import "core:sync/chan"
import "core:thread"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import gpu   "../../backends/cuda"
import llama "../../networks/llama"
import gpt2  "../../tokenizers/gpt2"

DATA_DIR             :: "smollm_data"
DEFAULT_MODEL_PATH   :: DATA_DIR + "/model.safetensors"
TOKENIZER_PATH       :: DATA_DIR + "/tokenizer.json"

DEFAULT_MAX_TOKENS  :: 512
DEFAULT_TEMPERATURE :: 0.8
DEFAULT_TOP_K       :: 40
DEFAULT_TOP_P       :: 0.95
DEFAULT_T_MAX       :: 4096
DEFAULT_THREADS     :: 8
DEFAULT_CPU_ARENA   :: 1 * 1024 * 1024 * 1024

DEFAULT_SYSTEM :: "You are a helpful AI assistant named SmolLM, trained by Hugging Face."

IM_START_TEXT  :: "<|im_start|>"
IM_END_TEXT    :: "<|im_end|>"
EOT_TEXT       :: "<|endoftext|>"

main :: proc() {
	max_new_tokens := DEFAULT_MAX_TOKENS
	temperature: f32 = DEFAULT_TEMPERATURE
	top_k          := DEFAULT_TOP_K
	top_p:       f32 = DEFAULT_TOP_P
	t_max          := DEFAULT_T_MAX
	use_cpu        := false
	cpu_arena      := DEFAULT_CPU_ARENA
	threads        := DEFAULT_THREADS
	system_prompt  := DEFAULT_SYSTEM
	model_path     := string(DEFAULT_MODEL_PATH)

	parse_args(&max_new_tokens, &temperature, &top_k, &top_p, &t_max, &use_cpu, &cpu_arena, &threads, &system_prompt, &model_path)

	cpu.set_thread_count(threads)

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
	tokenizer, tokenizer_ok := gpt2.load(TOKENIZER_PATH)
	if !tokenizer_ok {
		fmt.eprintln("FAIL: could not load tokenizer.")
		os.exit(1)
	}
	defer gpt2.destroy(tokenizer)

	im_start_id := tokenizer.added_tokens[IM_START_TEXT] if IM_START_TEXT in tokenizer.added_tokens else -1
	im_end_id   := tokenizer.added_tokens[IM_END_TEXT]   if IM_END_TEXT   in tokenizer.added_tokens else -1
	eot_id      := tokenizer.added_tokens[EOT_TEXT]      if EOT_TEXT      in tokenizer.added_tokens else -1
	if im_start_id < 0 || im_end_id < 0 {
		fmt.eprintln("FAIL: tokenizer missing <|im_start|> or <|im_end|>.")
		os.exit(1)
	}

	fmt.printfln("Allocating SmolLM2-135M (bf16, %v) ...", "CPU" if use_cpu else "GPU")
	model := llama.make(llama.SMOLLM2_135M_CONFIG, .Bf16)
	defer llama.destroy(model)

	fmt.printfln("Loading weights from %v ...", model_path)
	t_load := time.tick_now()
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		if is_smlw_file(model_path) {
			if !llama.load_smlw(model, model_path) {
				fmt.eprintln("FAIL: SMLW weight loading failed.")
				os.exit(1)
			}
		} else {
			if !llama.load_safetensors(model, model_path) {
				fmt.eprintln("FAIL: weight loading failed.")
				os.exit(1)
			}
		}
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	cache := llama.cache_make(model, t_max)
	defer llama.cache_destroy(cache)

	vocab_size := llama.SMOLLM2_135M_CONFIG.vocabulary_size
	last_row := builtin.make([]f32, vocab_size)
	defer delete(last_row)

	all_tokens: [dynamic]int
	defer delete(all_tokens)

	system_tokens := encode_chatml_turn(&tokenizer, "system", system_prompt, im_start_id, im_end_id, context.temp_allocator)
	append(&all_tokens, ..system_tokens)
	system_prefix_len := builtin.len(all_tokens)

	fmt.println()
	fmt.printfln("SmolLM2 chat (T=%.2f, top_k=%v, top_p=%.2f, t_max=%v, max_reply=%v).",
		f64(temperature), top_k, f64(top_p), t_max, max_new_tokens)
	fmt.println("Type your message and press Enter. Commands: :quit, :reset")
	fmt.println()

	printer: Printer
	_printer_start(&printer)
	defer _printer_stop(&printer)

	first_turn := true
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
		if builtin.len(line) == 0 {
			continue
		}

		switch line {
		case ":quit", ":exit", ":q":
			return
		case ":reset":
			llama.cache_reset(&cache)
			clear(&all_tokens)
			append(&all_tokens, ..system_tokens)
			first_turn = true
			fmt.println("(conversation reset)")
			continue
		}

		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

		user_tokens      := encode_chatml_turn  (&tokenizer, "user", line, im_start_id, im_end_id, context.temp_allocator)
		assistant_prefix := encode_chatml_prefix(&tokenizer, "assistant",  im_start_id,             context.temp_allocator)

		new_tokens: [dynamic]int
		new_tokens.allocator = context.temp_allocator
		if first_turn {
			append(&new_tokens, ..all_tokens[:system_prefix_len])
			first_turn = false
		}
		append(&new_tokens, ..user_tokens)
		append(&new_tokens, ..assistant_prefix)

		if cache.length + builtin.len(new_tokens) + 1 > t_max {
			fmt.eprintln("(context would overflow t_max; use :reset)")
			continue
		}

		append(&all_tokens, ..user_tokens)
		append(&all_tokens, ..assistant_prefix)

		t_prefill := time.tick_now()
		{
			ml.clear({.No_Gradients})
			logits := llama.forward_cached(model, &cache, new_tokens[:])
			buf := make([]f32, ml.len(logits), context.temp_allocator)
			ml.get_data(logits, buf)
			last_offset := (logits.shape[0] - 1) * vocab_size
			copy(last_row, buf[last_offset : last_offset + vocab_size])
		}
		prefill_elapsed := f64(time.duration_seconds(time.tick_since(t_prefill)))

		reply_start := builtin.len(all_tokens)
		previous_decoded_length := 0
		generated := 0
		t_generate := time.tick_now()

		for step in 0 ..< max_new_tokens {
			if cache.length + 1 > t_max {
				_printer_drain(&printer)
				fmt.println()
				fmt.println("(stopped: reached t_max)")
				break
			}

			next_id := sample_next(last_row, temperature, top_k, top_p)
			append(&all_tokens, next_id)
			generated += 1

			if next_id == im_end_id || next_id == eot_id {
				break
			}

			reply_so_far := gpt2.decode(&tokenizer, all_tokens[reply_start:])
			defer delete(reply_so_far)
			if builtin.len(reply_so_far) > previous_decoded_length {
				_printer_emit(&printer, reply_so_far[previous_decoded_length:])
				previous_decoded_length = builtin.len(reply_so_far)
			}

			if step == max_new_tokens - 1 {
				break
			}

			ml.clear({.No_Gradients})
			single := [1]int{next_id}
			logits := llama.forward_cached(model, &cache, single[:])
			ml.get_data(logits, last_row)
		}

		// Mirror the assistant's <|im_end|>\n into the running token list so the
		// next turn's prefill matches the formatting the model expects.
		if generated > 0 && all_tokens[builtin.len(all_tokens) - 1] != im_end_id {
			append(&all_tokens, im_end_id)
		}
		newline_ids := gpt2.encode(&tokenizer, "\n", context.temp_allocator)
		append(&all_tokens, ..newline_ids)

		_printer_drain(&printer)
		fmt.println()
		decode_elapsed := f64(time.duration_seconds(time.tick_since(t_generate)))
		prefill_rate := f64(builtin.len(new_tokens)) / prefill_elapsed if prefill_elapsed > 0 else 0
		decode_rate  := f64(generated) / decode_elapsed if decode_elapsed > 0 else 0
		fmt.printfln("  [prefill %v tok / %.2f s = %.1f tok/s   decode %v tok / %.2f s = %.1f tok/s]",
			builtin.len(new_tokens), prefill_elapsed, prefill_rate,
			generated, decode_elapsed, decode_rate)
		fmt.println()
	}
}

encode_chatml_turn :: proc(tok: ^gpt2.Tokenizer, role, content: string, im_start_id, im_end_id: int, allocator := context.allocator) -> []int {
	out: [dynamic]int
	out.allocator = allocator

	append(&out, im_start_id)

	header := fmt.tprintf("%v\n%v", role, content)
	header_ids := gpt2.encode(tok, header, context.temp_allocator)
	append(&out, ..header_ids)

	append(&out, im_end_id)

	newline_ids := gpt2.encode(tok, "\n", context.temp_allocator)
	append(&out, ..newline_ids)

	return out[:]
}

encode_chatml_prefix :: proc(tok: ^gpt2.Tokenizer, role: string, im_start_id: int, allocator := context.allocator) -> []int {
	out: [dynamic]int
	out.allocator = allocator

	append(&out, im_start_id)
	header := fmt.tprintf("%v\n", role)
	header_ids := gpt2.encode(tok, header, context.temp_allocator)
	append(&out, ..header_ids)

	return out[:]
}

is_smlw_file :: proc(path: string) -> bool {
	f, err := os.open(path, os.O_RDONLY)
	if err != nil {
		return false
	}
	defer os.close(f)
	header: [8]byte
	n, _ := os.read(f, header[:])
	return n == 8 && string(header[:]) == "SMLW0001"
}

read_line :: proc(buffer: []byte) -> (line: string, ok: bool) {
	cursor := 0
	one: [1]byte
	for cursor < builtin.len(buffer) {
		n, err := os.read(os.stdin, one[:])
		if err != nil || n == 0 {
			if cursor == 0 {
				return "", false
			}
			break
		}
		c := one[0]
		if c == '\n' {
			break
		}
		buffer[cursor] = c
		cursor += 1
	}
	if cursor > 0 && buffer[cursor - 1] == '\r' {
		cursor -= 1
	}
	return string(buffer[:cursor]), true
}

sample_next :: proc(logits: []f32, temperature: f32, top_k: int, top_p: f32) -> int {
	if temperature <= 0 || top_k == 1 {
		best := 0
		for i in 1 ..< builtin.len(logits) {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}

	candidate_count := top_k > 0 ? min(top_k, builtin.len(logits)) : builtin.len(logits)

	indices := make([]int, builtin.len(logits), context.temp_allocator)
	for i in 0 ..< builtin.len(logits) {
		indices[i] = i
	}

	for slot in 0 ..< candidate_count {
		best := slot
		for i in slot + 1 ..< builtin.len(indices) {
			if logits[indices[i]] > logits[indices[best]] {
				best = i
			}
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

parse_args :: proc(max_new_tokens: ^int, temperature: ^f32, top_k: ^int, top_p: ^f32, t_max: ^int, use_cpu: ^bool, cpu_arena: ^int, threads: ^int, system_prompt: ^string, model_path: ^string) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--max-tokens":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			max_new_tokens^ = _parse_int(args[i + 1])
			i += 2
		case "--temperature":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			temperature^ = f32(_parse_float(args[i + 1]))
			i += 2
		case "--top-k":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			top_k^ = _parse_int(args[i + 1])
			i += 2
		case "--top-p":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			top_p^ = f32(_parse_float(args[i + 1]))
			i += 2
		case "--t-max":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			t_max^ = _parse_int(args[i + 1])
			i += 2
		case "--cpu":
			use_cpu^ = true
			i += 1
		case "--cpu-arena":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			cpu_arena^ = _parse_int(args[i + 1])
			i += 2
		case "--threads":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			threads^ = _parse_int(args[i + 1])
			i += 2
		case "--system":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			system_prompt^ = args[i + 1]
			i += 2
		case "--model":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			model_path^ = args[i + 1]
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
	fmt.eprintln("usage: smollm_chat_repl [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--t-max N] [--cpu] [--cpu-arena BYTES] [--threads N] [--system TEXT] [--model PATH]")
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
			if in_frac {
				scale *= 10
			}
		} else {
			fmt.eprintfln("invalid float: %q", s)
			os.exit(1)
		}
		cursor += 1
	}
	out := value / scale
	return -out if negative else out
}

// Drains stdout writes onto a worker thread so the decode loop never blocks
// on os.flush. clear() at the start of each forward synchronises the GPU
// stream, so a per-token flush stall directly idles the GPU.
PRINTER_QUEUE_CAPACITY :: 256

Printer :: struct {
	ch:      chan.Chan(string),
	pending: sync.Wait_Group,
	thread:  ^thread.Thread,
}

_printer_proc :: proc(t: ^thread.Thread) {
	p := (^Printer)(t.data)
	for {
		msg, ok := chan.recv(p.ch)
		if !ok {
			return
		}
		fmt.print(msg)
		os.flush(os.stdout)
		delete(msg)
		sync.wait_group_done(&p.pending)
	}
}

_printer_start :: proc(p: ^Printer) {
	ch_err: runtime.Allocator_Error
	p.ch, ch_err = chan.create(chan.Chan(string), PRINTER_QUEUE_CAPACITY, context.allocator)
	if ch_err != .None {
		fmt.eprintln("FAIL: could not create printer channel.")
		os.exit(1)
	}
	p.thread = thread.create(_printer_proc)
	p.thread.data = p
	thread.start(p.thread)
}

_printer_stop :: proc(p: ^Printer) {
	chan.close(p.ch)
	thread.join(p.thread)
	thread.destroy(p.thread)
	chan.destroy(p.ch)
}

_printer_emit :: proc(p: ^Printer, s: string) {
	if builtin.len(s) == 0 {
		return
	}
	cloned := strings.clone(s)
	sync.wait_group_add(&p.pending, 1)
	chan.send(p.ch, cloned)
}

_printer_drain :: proc(p: ^Printer) {
	sync.wait_group_wait(&p.pending)
}
