// QLoRA fine-tune of Gemma E4B on a small dataset.
//
// Loads the E4B GGUF (Q4_K base, frozen), attaches LoRA adapters on the
// attention projections, trains the adapters on a tokenized text corpus.
//
// Required: gemma_data/model.gguf and gemma_data/tokenizer.json from
// the gemma_chat_repl example (or supply via --gguf / --tokenizer).
//
// Defaults to training on examples/data/shakespeare.txt; pass
// --corpus PATH to use your own text file.
//
// odin run examples/gemma_qlora -o:speed -- --corpus my_dataset.txt
//
// LoRA adapter weights are saved at the end to gemma_lora_adapters.bin.

package main

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:os"
import "core:time"

import ml    "../../"
import gpu   "../../backends/cuda"
import gemma "../../networks/gemma"
import lora  "../../networks/lora"
import tok   "../../tokenizers/gemma"

DATA_DIR        :: "gemma_data"
DEFAULT_GGUF    :: DATA_DIR + "/model.gguf"
DEFAULT_TOKEN   :: DATA_DIR + "/tokenizer.json"
DEFAULT_CORPUS  :: "examples/data/shakespeare.txt"
ADAPTER_OUT     :: "gemma_lora_adapters.bin"

DEFAULT_RANK         :: 16
DEFAULT_ALPHA        :: f32(32)
DEFAULT_SEQ_LEN      :: 256
DEFAULT_STEPS        :: 500
DEFAULT_ACCUM        :: 4
DEFAULT_LR           :: f32(1e-4)
DEFAULT_LOG_EVERY    :: 10
DEFAULT_WEIGHT_DECAY :: f32(0.0)
DEFAULT_WARMUP       :: 30
DEFAULT_SEED         :: 0xC0FFEE
MAX_CORPUS_BYTES     :: 32 * 1024 // tokenizer is slow; cap for runtime encoding

main :: proc() {
	defer fmt.println("Finished")

	gguf_path      := DEFAULT_GGUF
	tokenizer_path := DEFAULT_TOKEN
	corpus_path    := DEFAULT_CORPUS
	tokens_path    := ""
	rank           := DEFAULT_RANK
	alpha          := DEFAULT_ALPHA
	seq_len        := DEFAULT_SEQ_LEN
	steps          := DEFAULT_STEPS
	accum          := DEFAULT_ACCUM
	lr             := DEFAULT_LR
	log_every      := DEFAULT_LOG_EVERY

	parse_args(&gguf_path, &tokenizer_path, &corpus_path, &tokens_path, &rank, &alpha, &seq_len, &steps, &accum, &lr, &log_every)

	rand.reset(DEFAULT_SEED)

	corpus_tokens: []int
	if tokens_path != "" {
		// Pre-tokenized binary path. Skips the slow runtime tokenization.
		// Format matches tools/tokenize: u32 LE count + count * i32 ids.
		fmt.printfln("Loading pre-tokenized corpus from %v ...", tokens_path)
		corpus_tokens = load_token_file(tokens_path)
		fmt.printfln("  %v tokens", builtin.len(corpus_tokens))
	} else {
		fmt.println("Loading tokenizer ...")
		tokenizer, tokenizer_ok := tok.load(tokenizer_path)
		if !tokenizer_ok {
			fmt.eprintfln("FAIL: could not load tokenizer at %v", tokenizer_path)
			os.exit(1)
		}
		defer tok.destroy(tokenizer)

		fmt.println("Reading corpus ...")
		corpus_bytes, corpus_err := os.read_entire_file_from_path(corpus_path, context.allocator)
		if corpus_err != nil {
			fmt.eprintfln("FAIL: could not read corpus at %v: %v", corpus_path, corpus_err)
			os.exit(1)
		}
		defer delete(corpus_bytes)

		// The gemma tokenizer's `encode` is unworkably slow on big corpora.
		// Cap the runtime path; for real fine-tunes, pre-tokenize via
		// `tools/tokenize` and pass the binary via --tokens.
		corpus_cap := MAX_CORPUS_BYTES
		if builtin.len(corpus_bytes) > corpus_cap {
			fmt.printfln("Corpus is %v bytes; capping at %v for runtime tokenization.", builtin.len(corpus_bytes), corpus_cap)
			fmt.println("(Pre-tokenize with tools/tokenize and pass --tokens PATH for the full corpus.)")
		} else {
			corpus_cap = builtin.len(corpus_bytes)
		}

		fmt.println("Tokenizing corpus (this can be slow) ...")
		t_tok := time.tick_now()
		corpus_text := string(corpus_bytes[:corpus_cap])
		corpus_tokens = tok.encode(&tokenizer, corpus_text)
		fmt.printfln("  corpus = %v bytes -> %v tokens (%.1f s)",
			corpus_cap, builtin.len(corpus_tokens),
			f64(time.duration_seconds(time.tick_since(t_tok))))
	}
	defer delete(corpus_tokens)

	if builtin.len(corpus_tokens) <= seq_len + 1 {
		fmt.eprintfln("FAIL: corpus too short for seq_len %v", seq_len)
		os.exit(1)
	}

	split          := (builtin.len(corpus_tokens) * 9) / 10
	train_tokens   := corpus_tokens[:split]
	val_tokens     := corpus_tokens[split:]

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	fmt.println("Allocating Gemma 4 E4B (bf16 + LoRA) ...")
	cfg := gemma.make_e4b_config()
	defer gemma.config_destroy(cfg)

	lora_cfg := gemma.LoRA_Config{
		rank    = rank,
		alpha   = alpha,
		targets = gemma.LORA_DEFAULT_TARGETS,
	}
	model := gemma.make(cfg, .Bf16, for_training = true, lora_cfg = lora_cfg)
	defer gemma.destroy(model)

	fmt.println("Loading GGUF (Q4_K base, frozen) ...")
	t_load := time.tick_now()
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		if !gemma.load_gguf(&model, gguf_path) {
			fmt.eprintln("FAIL: GGUF weight loading failed.")
			os.exit(1)
		}
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	gemma.randomize_lora(model)

	lora_params := gemma.lora_parameter_count(model)
	fmt.printfln("LoRA params: %v   (rank=%v, alpha=%.1f, targets=%v)", lora_params, rank, f64(alpha), lora_cfg.targets)
	fmt.printfln("Sequence length: %v   Steps: %v   Accum: %v   LR: %.2e", seq_len, steps, accum, f64(lr))
	fmt.println()

	inputs  := builtin.make([]int, seq_len)
	targets := builtin.make([]int, seq_len)
	defer delete(inputs)
	defer delete(targets)

	opt: ml.Optimizer
	t_start := time.tick_now()
	loss_running: f32
	loss_samples: int

	for step in 1 ..= steps {
		defer free_all(context.temp_allocator)

		sample_window(train_tokens, inputs, targets)

		ml.clear()

		logits     := gemma.forward(model, inputs)
		token_loss := ml.cross_entropy(logits, targets)

		ml.backward()

		loss_running += read_mean_loss(token_loss)
		loss_samples += 1

		cur_lr := learning_rate_at(step, steps, DEFAULT_WARMUP, lr, 0.1)
		if ml.optimize(&opt, period=accum, learning_rate=cur_lr, weight_decay=DEFAULT_WEIGHT_DECAY) {
			gemma.update_lora(opt, model)
		}

		if step % log_every == 0 {
			elapsed   := f64(time.duration_seconds(time.tick_since(t_start)))
			tokens    := step * seq_len
			tok_per_s := f64(tokens) / elapsed
			fmt.printfln(
				"step %5v  train_loss = %.4f  lr = %.2e  (%.0f tok/s)",
				step, loss_running / f32(loss_samples), cur_lr, tok_per_s,
			)
			loss_running = 0
			loss_samples = 0
		}
	}

	val_loss := evaluate(model, val_tokens, seq_len, 8)
	fmt.printfln("Final val_loss = %.4f", val_loss)

	fmt.println()
	fmt.printfln("Saving adapter weights to %v ...", ADAPTER_OUT)
	if !save_adapters(model, ADAPTER_OUT) {
		fmt.eprintln("FAIL: adapter save failed.")
		os.exit(1)
	}
	fmt.println("  saved.")
}

sample_window :: proc(corpus, inputs, targets: []int) {
	max_offset := builtin.len(corpus) - builtin.len(inputs) - 1
	offset     := rand.int_max(max_offset)
	for i in 0 ..< builtin.len(inputs) {
		inputs[i]  = corpus[offset + i]
		targets[i] = corpus[offset + i + 1]
	}
}

learning_rate_at :: proc(step, total_steps, warmup_steps: int, max_lr, min_lr_frac: f32) -> f32 {
	if step < warmup_steps {
		return max_lr * f32(step) / f32(warmup_steps)
	}
	progress := f32(step - warmup_steps) / f32(total_steps - warmup_steps)
	if progress > 1 {
		progress = 1
	}
	cosine := 0.5 * (1 + math.cos(math.PI * progress))
	return max_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)
}

read_mean_loss :: proc(loss_tensor: ml.Tensor) -> f32 {
	count := ml.len(loss_tensor)
	buf   := builtin.make([]f32, count, context.temp_allocator)
	ml.get_data(loss_tensor, buf)
	sum: f32
	for v in buf {
		sum += v
	}
	return sum / f32(count)
}

evaluate :: proc(model: gemma.Gemma, corpus: []int, seq_len: int, batches: int) -> f32 {
	inputs  := builtin.make([]int, seq_len)
	targets := builtin.make([]int, seq_len)
	defer delete(inputs)
	defer delete(targets)

	total: f32
	for _ in 0 ..< batches {
		defer free_all(context.temp_allocator)

		sample_window(corpus, inputs, targets)

		ml.clear({.No_Gradients})
		logits     := gemma.forward(model, inputs)
		token_loss := ml.cross_entropy(logits, targets)
		total += read_mean_loss(token_loss)
	}
	return total / f32(batches)
}

// Simple binary format for adapter weights:
//   magic   "LORA0001" (8 bytes)
//   layer_count (i32)
//   for each layer:
//     7 adapter slots (q, k, v, o, gate, up, down):
//       rank (i32, 0 = unused)
//       in_features (i32)
//       out_features (i32)
//       a_bytes (rank * in_features * 2)
//       b_bytes (out_features * rank * 2)
save_adapters :: proc(model: gemma.Gemma, path: string) -> bool {
	buf: [dynamic]byte
	defer delete(buf)

	append(&buf, ..transmute([]byte)string("LORA0001"))

	layer_count := i32(builtin.len(model.layers))
	append(&buf, ..mem.slice_to_bytes(([]i32{layer_count})[:]))

	for layer in model.layers {
		_save_adapter(&buf, layer.q_lora)
		_save_adapter(&buf, layer.k_lora)
		_save_adapter(&buf, layer.v_lora)
		_save_adapter(&buf, layer.o_lora)
		_save_adapter(&buf, layer.gate_lora)
		_save_adapter(&buf, layer.up_lora)
		_save_adapter(&buf, layer.down_lora)
	}

	return os.write_entire_file(path, buf[:]) == nil
}

_save_adapter :: proc(buf: ^[dynamic]byte, adapter: lora.Adapter) {
	header := [3]i32{i32(adapter.rank), i32(adapter.in_features), i32(adapter.out_features)}
	append(buf, ..mem.slice_to_bytes(header[:]))
	if adapter.rank == 0 {
		return
	}

	a_bytes := builtin.make([]byte, ml.len(adapter.a) * 2, context.temp_allocator)
	ml.get_data_bytes(adapter.a, a_bytes)
	append(buf, ..a_bytes)

	b_bytes := builtin.make([]byte, ml.len(adapter.b) * 2, context.temp_allocator)
	ml.get_data_bytes(adapter.b, b_bytes)
	append(buf, ..b_bytes)
}

parse_args :: proc(gguf, tokenizer, corpus, tokens: ^string, rank: ^int, alpha: ^f32, seq_len, steps, accum: ^int, lr: ^f32, log_every: ^int) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--gguf":      i += 1; gguf^      = args[i]; i += 1
		case "--tokenizer": i += 1; tokenizer^ = args[i]; i += 1
		case "--corpus":    i += 1; corpus^    = args[i]; i += 1
		case "--tokens":    i += 1; tokens^    = args[i]; i += 1
		case "--rank":      i += 1; rank^      = _parse_int(args[i]); i += 1
		case "--alpha":     i += 1; alpha^     = f32(_parse_float(args[i])); i += 1
		case "--seq-len":   i += 1; seq_len^   = _parse_int(args[i]); i += 1
		case "--steps":     i += 1; steps^     = _parse_int(args[i]); i += 1
		case "--accum":     i += 1; accum^     = _parse_int(args[i]); i += 1
		case "--lr":        i += 1; lr^        = f32(_parse_float(args[i])); i += 1
		case "--log-every": i += 1; log_every^ = _parse_int(args[i]); i += 1
		case "--help", "-h":
			fmt.println("usage: gemma_qlora [--gguf PATH] [--tokenizer PATH] [--corpus PATH] [--tokens PATH] [--rank N] [--alpha F] [--seq-len N] [--steps N] [--accum N] [--lr F] [--log-every N]")
			fmt.println("  --tokens PATH   pre-tokenized binary (u32 LE count + i32 ids); skips runtime tokenization")
			fmt.println("  --corpus PATH   raw text; tokenized at runtime (slow on >32 KB)")
			os.exit(0)
		case:
			fmt.eprintfln("unknown argument: %v", arg)
			os.exit(1)
		}
	}
}

load_token_file :: proc(path: string) -> []int {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("FAIL: could not read tokens file %v: %v", path, err)
		os.exit(1)
	}
	defer delete(bytes)

	if builtin.len(bytes) < 4 {
		fmt.eprintfln("FAIL: %v is too short to be a tokens file", path)
		os.exit(1)
	}

	count := int((^u32le)(raw_data(bytes))^)
	expected := 4 + count * 4
	if builtin.len(bytes) < expected {
		fmt.eprintfln("FAIL: %v claims %v tokens but file has %v bytes", path, count, builtin.len(bytes))
		os.exit(1)
	}

	out := builtin.make([]int, count)
	for i in 0 ..< count {
		out[i] = int((^i32)(&bytes[4 + i * 4])^)
	}
	return out
}

_parse_int :: proc(s: string) -> int {
	v: int
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
		v = v * 10 + int(c - '0')
		cursor += 1
	}
	return -v if negative else v
}

_parse_float :: proc(s: string) -> f64 {
	value: f64
	scale: f64 = 1
	in_frac    := false
	negative   := false
	in_exp     := false
	exp_neg    := false
	exp_val    := 0
	cursor     := 0
	if builtin.len(s) > 0 && s[0] == '-' {
		negative = true
		cursor = 1
	}
	for cursor < builtin.len(s) {
		c := s[cursor]
		if c == '.' {
			in_frac = true
		} else if c == 'e' || c == 'E' {
			in_exp = true
			cursor += 1
			if cursor < builtin.len(s) && s[cursor] == '-' {
				exp_neg = true
				cursor += 1
			} else if cursor < builtin.len(s) && s[cursor] == '+' {
				cursor += 1
			}
			continue
		} else if c >= '0' && c <= '9' {
			d := int(c - '0')
			if in_exp {
				exp_val = exp_val * 10 + d
			} else {
				value = value * 10 + f64(d)
				if in_frac {
					scale *= 10
				}
			}
		} else {
			fmt.eprintfln("invalid float: %q", s)
			os.exit(1)
		}
		cursor += 1
	}
	out := value / scale
	if in_exp {
		mul := math.pow(10.0, f64(exp_val))
		if exp_neg {
			out /= mul
		} else {
			out *= mul
		}
	}
	return -out if negative else out
}