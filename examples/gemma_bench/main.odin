package gemma_bench

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:os"
import "core:slice"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import gpu   "../../backends/gpu"
import gemma "../../networks/gemma"
import tok   "../../tokenizers/gemma"

DATA_DIR       :: "gemma_data"
MODEL_PATH     :: DATA_DIR + "/model.safetensors"
TOKENIZER_PATH :: DATA_DIR + "/tokenizer.json"

EOS_TOKEN_ID     :: 1
END_OF_TURN_TEXT :: "<turn|>"
BOS_TEXT         :: "<bos>"

DEFAULT_PROMPT     :: "Explain in two paragraphs why pelicans are excellent fishers, and what makes their pouch anatomically unusual compared to other seabirds."
DEFAULT_GEN_TOKENS :: 128
DEFAULT_REPS       :: 3
DEFAULT_WARMUP     :: 1
DEFAULT_T_MAX      :: 4096
DEFAULT_THREADS    :: 8
DEFAULT_CPU_ARENA  :: 2 * 1024 * 1024 * 1024

Run_Result :: struct {
	prompt_tokens:    int,
	prefill_seconds:  f64,
	decoded_tokens:   int,
	decode_seconds:   f64,
}

main :: proc() {
	prompt_text  := DEFAULT_PROMPT
	prompt_file  := ""
	gen_tokens   := DEFAULT_GEN_TOKENS
	reps         := DEFAULT_REPS
	warmup       := DEFAULT_WARMUP
	t_max        := DEFAULT_T_MAX
	use_cpu      := false
	cpu_arena    := DEFAULT_CPU_ARENA
	threads      := DEFAULT_THREADS
	quant_mode   := gemma.Quant_Mode.None
	skip_weights := false

	parse_args(&prompt_text, &prompt_file, &gen_tokens, &reps, &warmup, &t_max, &use_cpu, &cpu_arena, &threads, &quant_mode, &skip_weights)

	if prompt_file != "" {
		bytes, err := os.read_entire_file_from_path(prompt_file, context.allocator)
		if err != nil {
			fmt.eprintfln("could not read --prompt-file %v: %v", prompt_file, err)
			os.exit(1)
		}
		prompt_text = string(bytes)
	}

	cpu.set_thread_count(threads)

	ctx := cpu.context_create(cpu_arena) if use_cpu else gpu.context_create()
	defer if use_cpu { cpu.context_destroy(ctx) } else { gpu.context_destroy(ctx) }
	ml.context_scope(ctx)
	ml.set_inference_only(true)

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

	if skip_weights {
		fmt.println("Skipping weight load (--skip-weights); model will produce garbage tokens but timing is valid.")
		if quant_mode != .None {
			gemma.quantize_for_inference_fake(&model, quant_mode)
		}
	} else {
		fmt.println("Loading weights ...")
		t_load := time.tick_now()
		{
			runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
			if !gemma.load_safetensors(model, MODEL_PATH) {
				fmt.eprintln("FAIL: weight loading failed.")
				os.exit(1)
			}
		}
		fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

		if quant_mode != .None {
			fmt.printfln("Quantizing linear weights to %v ...", quant_label(quant_mode))
			t_q := time.tick_now()
			runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
			gemma.quantize_for_inference(&model, quant_mode)
			fmt.printfln("  quantized in %.1f s", f64(time.duration_seconds(time.tick_since(t_q))))
		}
	}

	cache := gemma.cache_make(model, t_max)
	defer gemma.cache_destroy(cache)

	bos_id := tokenizer.added_tokens[BOS_TEXT] if BOS_TEXT in tokenizer.added_tokens else -1

	turn_text  := fmt.tprintf("<|turn>user\n%v<turn|>\n<|turn>model\n", prompt_text)
	user_ids   := tok.encode(&tokenizer, turn_text)
	defer delete(user_ids)

	prompt_ids: [dynamic]int
	defer delete(prompt_ids)
	if bos_id >= 0 do append(&prompt_ids, bos_id)
	append(&prompt_ids, ..user_ids)

	if builtin.len(prompt_ids) + gen_tokens >= t_max {
		fmt.eprintfln("prompt %v + gen %v exceeds t_max %v", builtin.len(prompt_ids), gen_tokens, t_max)
		os.exit(1)
	}

	vocab_size := cfg.vocab_size
	last_row := builtin.make([]f32, vocab_size)
	defer delete(last_row)

	fmt.printfln("\nbench: prompt=%v tok, generate=%v tok, warmup=%v, reps=%v, quant=%v\n",
		builtin.len(prompt_ids), gen_tokens, warmup, reps, quant_label(quant_mode))

	results := make([]Run_Result, reps, context.temp_allocator)

	total_runs := warmup + reps
	timing_run := !use_cpu && warmup + reps > 0
	if timing_run {
		gpu.enable_timing()
	}

	for run_idx in 0 ..< total_runs {
		is_warmup := run_idx < warmup
		label := "warmup" if is_warmup else "run"
		index := run_idx if is_warmup else run_idx - warmup
		fmt.printfln("--- %v %v ---", label, index + 1)

		// Time only the last timed rep so the dump reflects steady-state decode.
		last_rep := !is_warmup && index + 1 == reps
		if timing_run && last_rep {
			gpu.reset_timing()
		}

		gemma.cache_reset(&cache)
		ml.clear()

		t_prefill := time.tick_now()
		{
			logits := gemma.forward_cached(model, &cache, prompt_ids[:])
			buf := make([]f32, ml.len(logits), context.temp_allocator)
			ml.get_data(logits, buf)
			copy(last_row, buf[(builtin.len(prompt_ids) - 1) * vocab_size :])
		}
		prefill_seconds := f64(time.duration_seconds(time.tick_since(t_prefill)))

		generated := 0
		record_ns:  i64
		sync_ns:    i64
		gpu.reset_forward_stats()
		gpu.reset_alloc_stats()
		t_decode := time.tick_now()
		for _ in 0 ..< gen_tokens {
			runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

			next_id := argmax(last_row)
			generated += 1
			if next_id == EOS_TOKEN_ID do break

			ml.clear()
			single := [1]int{next_id}
			t_rec := time.tick_now()
			logits := gemma.forward_cached(model, &cache, single[:])
			record_ns += i64(time.tick_since(t_rec))
			t_s := time.tick_now()
			ml.get_data(logits, last_row)
			sync_ns += i64(time.tick_since(t_s))
		}
		decode_seconds := f64(time.duration_seconds(time.tick_since(t_decode)))
		if generated > 0 {
			ops, op_ns      := gpu.forward_stats()
			allocs, alloc_ns := gpu.alloc_stats()
			fmt.printfln("  (per token: record %.2f ms, sync %.2f ms)",
				f64(record_ns) / f64(generated) / 1e6,
				f64(sync_ns)   / f64(generated) / 1e6)
			fmt.printfln("  (per token: %v ops × %.1f us/op = %.2f ms in backend.forward)",
				ops / generated, f64(op_ns) / f64(ops) / 1e3, f64(op_ns) / f64(generated) / 1e6)
			fmt.printfln("  (per token: %v buffer_allocs × %.1f us = %.2f ms)",
				allocs / generated, f64(alloc_ns) / f64(allocs) / 1e3, f64(alloc_ns) / f64(generated) / 1e6)
			uploads, upload_ns := gpu.upload_stats()
			if uploads > 0 {
				fmt.printfln("  (per token: %v buffer_set uploads × %.2f ms = %.2f ms)",
					uploads / generated, f64(upload_ns) / f64(uploads) / 1e6, f64(upload_ns) / f64(generated) / 1e6)
			}
		}

		report(builtin.len(prompt_ids), prefill_seconds, generated, decode_seconds)

		if !is_warmup {
			results[index] = Run_Result{
				prompt_tokens   = builtin.len(prompt_ids),
				prefill_seconds = prefill_seconds,
				decoded_tokens  = generated,
				decode_seconds  = decode_seconds,
			}
		}
	}

	if timing_run {
		gpu.dump_timing()
	}

	if reps > 0 {
		prefill_rates := make([]f64, reps, context.temp_allocator)
		decode_rates  := make([]f64, reps, context.temp_allocator)
		prefill_durs  := make([]f64, reps, context.temp_allocator)
		decode_durs   := make([]f64, reps, context.temp_allocator)
		for r, i in results {
			prefill_rates[i] = f64(r.prompt_tokens)  / r.prefill_seconds if r.prefill_seconds > 0 else 0
			decode_rates [i] = f64(r.decoded_tokens) / r.decode_seconds  if r.decode_seconds  > 0 else 0
			prefill_durs[i]  = r.prefill_seconds
			decode_durs[i]   = r.decode_seconds
		}
		fmt.printfln("--- median over %v rep(s) ---", reps)
		fmt.printfln("prompt eval count:    %v token(s)", results[0].prompt_tokens)
		fmt.printfln("prompt eval duration: %.2f ms",     median(prefill_durs) * 1000)
		fmt.printfln("prompt eval rate:     %.2f tokens/s", median(prefill_rates))
		fmt.printfln("eval count:           %v token(s)", results[0].decoded_tokens)
		fmt.printfln("eval duration:        %.2f ms",     median(decode_durs) * 1000)
		fmt.printfln("eval rate:            %.2f tokens/s", median(decode_rates))
	}
}

report :: proc(prompt_tokens: int, prefill_s: f64, decoded: int, decode_s: f64) {
	prompt_rate := f64(prompt_tokens) / prefill_s if prefill_s > 0 else 0
	decode_rate := f64(decoded)       / decode_s  if decode_s  > 0 else 0
	fmt.printfln("prompt eval count:    %v token(s)", prompt_tokens)
	fmt.printfln("prompt eval duration: %.2f ms",      prefill_s * 1000)
	fmt.printfln("prompt eval rate:     %.2f tokens/s", prompt_rate)
	fmt.printfln("eval count:           %v token(s)", decoded)
	fmt.printfln("eval duration:        %.2f ms",      decode_s * 1000)
	fmt.printfln("eval rate:            %.2f tokens/s", decode_rate)
	fmt.println()
}

argmax :: proc(values: []f32) -> int {
	best := 0
	for i in 1 ..< builtin.len(values) {
		if values[i] > values[best] do best = i
	}
	return best
}

median :: proc(values: []f64) -> f64 {
	tmp := make([]f64, builtin.len(values), context.temp_allocator)
	copy(tmp, values)
	slice.sort(tmp)
	n := builtin.len(tmp)
	if n == 0 do return 0
	if n % 2 == 1 do return tmp[n / 2]
	return 0.5 * (tmp[n / 2 - 1] + tmp[n / 2])
}

quant_label :: proc(mode: gemma.Quant_Mode) -> string {
	switch mode {
	case .None: return "None"
	case .Int8: return "Int8"
	case .Int4: return "Int4"
	case .Q8_0: return "Q8_0"
	}
	return "?"
}

parse_args :: proc(
	prompt_text: ^string, prompt_file: ^string,
	gen_tokens: ^int, reps: ^int, warmup: ^int,
	t_max: ^int, use_cpu: ^bool, cpu_arena: ^int, threads: ^int,
	quant_mode: ^gemma.Quant_Mode, skip_weights: ^bool,
) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--prompt":
			if i + 1 >= builtin.len(args) do _usage_exit()
			prompt_text^ = args[i + 1]
			i += 2
		case "--prompt-file":
			if i + 1 >= builtin.len(args) do _usage_exit()
			prompt_file^ = args[i + 1]
			i += 2
		case "--gen-tokens":
			if i + 1 >= builtin.len(args) do _usage_exit()
			gen_tokens^ = _parse_int(args[i + 1])
			i += 2
		case "--reps":
			if i + 1 >= builtin.len(args) do _usage_exit()
			reps^ = _parse_int(args[i + 1])
			i += 2
		case "--warmup":
			if i + 1 >= builtin.len(args) do _usage_exit()
			warmup^ = _parse_int(args[i + 1])
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
		case "--threads":
			if i + 1 >= builtin.len(args) do _usage_exit()
			threads^ = _parse_int(args[i + 1])
			i += 2
		case "--quantize":
			if i + 1 >= builtin.len(args) do _usage_exit()
			switch args[i + 1] {
			case "q8", "int8":  quant_mode^ = .Int8
			case "q4", "int4":  quant_mode^ = .Int4
			case "q8_0":         quant_mode^ = .Q8_0
			case "none":         quant_mode^ = .None
			case:
				fmt.eprintfln("unknown --quantize value: %v", args[i + 1])
				_usage_exit()
			}
			i += 2
		case "--skip-weights":
			skip_weights^ = true
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
	fmt.eprintln("usage: gemma_bench [--prompt TEXT] [--prompt-file PATH] [--gen-tokens N] [--reps N] [--warmup N] [--t-max N] [--cpu] [--cpu-arena BYTES] [--threads N] [--quantize q8|q4|q8_0|none] [--skip-weights]")
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
