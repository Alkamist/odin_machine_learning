package gemma_chat_repl

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
import gemma "../../networks/gemma"
import tok   "../../tokenizers/gemma"

DATA_DIR       :: "gemma_data"
MODEL_PATH     :: DATA_DIR + "/model.safetensors"
GGUF_PATH      :: DATA_DIR + "/model.gguf"
TOKENIZER_PATH :: DATA_DIR + "/tokenizer.json"

DEFAULT_MAX_TOKENS  :: 512
DEFAULT_TEMPERATURE :: 0.8
DEFAULT_TOP_K       :: 40
DEFAULT_TOP_P       :: 0.95
DEFAULT_T_MAX       :: 4096
DEFAULT_CPU_ARENA   :: 2 * 1024 * 1024 * 1024
DEFAULT_THREADS     :: 8

EOS_TOKEN_ID     :: 1
END_OF_TURN_TEXT :: "<turn|>"
BOS_TEXT         :: "<bos>"

main :: proc() {
	max_new_tokens := DEFAULT_MAX_TOKENS
	temperature: f32 = DEFAULT_TEMPERATURE
	top_k          := DEFAULT_TOP_K
	top_p:       f32 = DEFAULT_TOP_P
	t_max          := DEFAULT_T_MAX
	use_cpu        := false
	cpu_arena      := DEFAULT_CPU_ARENA
	threads        := DEFAULT_THREADS
	gguf_path      := ""
	timing         := false

	parse_args(&max_new_tokens, &temperature, &top_k, &top_p, &t_max, &use_cpu, &cpu_arena, &threads, &gguf_path, &timing)

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

	if timing && !use_cpu {
		gpu.enable_timing(true)
	} else if !use_cpu {
		gpu.enable_decode_graph(true)
	}

	fmt.println("Loading tokenizer ...")
	tokenizer, tokenizer_ok := tok.load(TOKENIZER_PATH)
	if !tokenizer_ok {
		fmt.eprintln("FAIL: could not load tokenizer.")
		os.exit(1)
	}
	defer tok.destroy(tokenizer)

	use_gguf := builtin.len(gguf_path) > 0 || os.exists(GGUF_PATH)
	weights_label := "Q4_K_M GGUF" if use_gguf else "bf16 safetensors"
	fmt.printfln("Allocating Gemma 4 E4B (%v, %v) ...", weights_label, "CPU" if use_cpu else "GPU")
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
		if use_gguf {
			path := gguf_path if builtin.len(gguf_path) > 0 else GGUF_PATH
			if !gemma.load_gguf(&model, path) {
				fmt.eprintln("FAIL: GGUF weight loading failed.")
				os.exit(1)
			}
		} else {
			if !gemma.load_safetensors(model, MODEL_PATH) {
				fmt.eprintln("FAIL: weight loading failed.")
				os.exit(1)
			}
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
	if bos_id >= 0 {
		append(&all_tokens, bos_id)
	}

	fmt.println()
	fmt.printfln("Gemma 4 chat (T=%.2f, top_k=%v, top_p=%.2f, t_max=%v, max_reply=%v).",
		f64(temperature), top_k, f64(top_p), t_max, max_new_tokens)
	fmt.println("Type your message and press Enter. Commands: :quit, :reset")
	fmt.println()

	printer: Printer
	_printer_start(&printer)
	defer _printer_stop(&printer)

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
			gemma.cache_reset(&cache)
			clear(&all_tokens)
			if bos_id >= 0 {
				append(&all_tokens, bos_id)
			}
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

		t_prefill := time.tick_now()
		{
			// Q4_K/Q6_K coopmat prefill requires M % 64 == 0, so feed prefill
			// in 64-token aligned chunks plus a single-token-per-step tail.
			// (bf16 prefill works for any M but uses the same chunking â€” the
			//  branching adds no real cost.)
			PREFILL_CHUNK :: 64
			pos := 0
			n := builtin.len(new_tokens)
			for pos < n {
				ml.clear({.No_Gradients})
				take := PREFILL_CHUNK
				if pos + take > n {
					take = n - pos
				}
				if take != PREFILL_CHUNK {
					take = 1
				}
				chunk := new_tokens[pos : pos + take]
				logits := gemma.forward_cached(model, &cache, chunk)
				if pos + take == n {
					buf := make([]f32, ml.len(logits), context.temp_allocator)
					ml.get_data(logits, buf)
					copy(last_row, buf[(take - 1) * vocab_size :])
				}
				pos += take
			}
		}
		prefill_elapsed := f64(time.duration_seconds(time.tick_since(t_prefill)))

		reply_start := builtin.len(all_tokens)
		previous_decoded_length := 0
		generated := 0
		if timing && !use_cpu {
			gpu.reset_timing()
		}
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

			if next_id == EOS_TOKEN_ID || next_id == end_of_turn_id {
				break
			}

			reply_so_far := tok.decode(&tokenizer, all_tokens[reply_start:])
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
			logits := gemma.forward_cached(model, &cache, single[:])
			ml.get_data(logits, last_row)
		}

		_printer_drain(&printer)
		fmt.println()
		elapsed := f64(time.duration_seconds(time.tick_since(t_generate)))
		prompt_rate := f64(builtin.len(new_tokens)) / prefill_elapsed if prefill_elapsed > 0 else 0
		decode_rate := f64(generated) / elapsed if elapsed > 0 else 0
		fmt.printfln("  [prefill %v tok / %.2f s = %.1f tok/s   decode %v tok / %.2f s = %.1f tok/s]",
			builtin.len(new_tokens), prefill_elapsed, prompt_rate,
			generated, elapsed, decode_rate)

		if timing && !use_cpu && generated > 0 {
			entries := gpu.timing_snapshot()
			defer delete(entries)
			gpu_total_ns: i64
			for e in entries {
				gpu_total_ns += e.total_ns
			}
			wall_ns := i64(time.duration_nanoseconds(time.tick_since(t_generate)))
			gpu_ms_per_tok   := f64(gpu_total_ns) / f64(generated) / 1e6
			wall_ms_per_tok  := f64(wall_ns)      / f64(generated) / 1e6
			gpu_pct          := 100.0 * f64(gpu_total_ns) / f64(wall_ns) if wall_ns > 0 else 0
			fmt.printfln("  [decode timing: gpu=%.2f ms/tok  wall=%.2f ms/tok  gpu/wall=%.1f%%]",
				gpu_ms_per_tok, wall_ms_per_tok, gpu_pct)
			shown := 0
			for e in entries {
				if shown >= 12 {
					break
				}
				avg_us := f64(e.total_ns) / f64(e.count) / 1e3
				share  := 100.0 * f64(e.total_ns) / f64(gpu_total_ns) if gpu_total_ns > 0 else 0
				fmt.printfln("    % 5.1f%%  % 7.1f us avg  x% -7d %s", share, avg_us, e.count, e.name)
				shown += 1
			}
		}

		fmt.println()
	}
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

parse_args :: proc(max_new_tokens: ^int, temperature: ^f32, top_k: ^int, top_p: ^f32, t_max: ^int, use_cpu: ^bool, cpu_arena: ^int, threads: ^int, gguf_path: ^string, timing: ^bool) {
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
		case "--gguf":
			if i + 1 >= builtin.len(args) {
				_usage_exit()
			}
			gguf_path^ = args[i + 1]
			i += 2
		case "--timing":
			timing^ = true
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
	fmt.eprintln("usage: gemma_chat_repl [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--t-max N] [--cpu] [--cpu-arena BYTES] [--threads N] [--gguf PATH] [--timing]")
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
// on os.flush. clear() at the start of each forward synchronises the CUDA
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