// QLoRA fine-tune of Gemma 4 E4B on the ReaScript chat dataset.
//
// Loads the E4B GGUF (Q4_K base, frozen), attaches LoRA adapters, trains
// the adapters with assistant-only loss masking. Samples are pre-tokenized
// via tools/tokenize_chat into per-row records carrying the assistant span.
// The trainer greedily packs records into a fixed-length sequence per
// optimizer micro-step.
//
// Required:
//   gemma_data/model.gguf
//   gemma_data/tokenizer.json (only used to look up <bos> for sanity)
//   reascript/dataset/reascript_gemma.bin
//
//   odin run examples/reascript_qlora -o:speed -- --steps 800
//
// Adapter weights are saved at the end to reascript_qlora_adapters.bin.

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

DATA_PATH        :: "reascript/dataset/reascript_gemma.bin"
DEFAULT_GGUF     :: "gemma_data/model.gguf"
ADAPTER_OUT      :: "reascript_qlora_adapters.bin"
ADAPTER_OUT_BEST :: "reascript_qlora_adapters_best.bin"

DEFAULT_RANK         :: 16
DEFAULT_ALPHA        :: f32(32)
DEFAULT_SEQ_LEN      :: 1024
DEFAULT_STEPS        :: 800
DEFAULT_ACCUM        :: 8
DEFAULT_LR           :: f32(1e-4)
DEFAULT_LOG_EVERY    :: 10
DEFAULT_VAL_EVERY    :: 100
DEFAULT_VAL_BATCHES  :: 32
DEFAULT_WEIGHT_DECAY :: f32(0.0)
DEFAULT_WARMUP       :: 30
DEFAULT_SEED         :: 0xC0FFEE
DEFAULT_VAL_FRACTION :: 0.10

main :: proc() {
	defer fmt.println("Finished")

	gguf_path  := DEFAULT_GGUF
	data_path  := DATA_PATH
	rank       := DEFAULT_RANK
	alpha      := DEFAULT_ALPHA
	seq_len    := DEFAULT_SEQ_LEN
	steps      := DEFAULT_STEPS
	accum      := DEFAULT_ACCUM
	lr         := DEFAULT_LR
	log_every  := DEFAULT_LOG_EVERY
	val_every  := DEFAULT_VAL_EVERY

	parse_args(&gguf_path, &data_path, &rank, &alpha, &seq_len, &steps, &accum, &lr, &log_every, &val_every)

	rand.reset(DEFAULT_SEED)

	fmt.printfln("Loading dataset from %v ...", data_path)
	all_samples := load_chat_samples(data_path)
	defer destroy_samples(all_samples)
	fmt.printfln("  %v samples", builtin.len(all_samples))

	val_count := int(f32(builtin.len(all_samples)) * DEFAULT_VAL_FRACTION)
	if val_count < 1 {
		val_count = 1
	}
	train_samples := all_samples[:builtin.len(all_samples) - val_count]
	val_samples   := all_samples[builtin.len(all_samples) - val_count:]
	fmt.printfln("  train=%v   val=%v", builtin.len(train_samples), builtin.len(val_samples))

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

	pack_buf  := builtin.make([]int, seq_len + 1)
	defer delete(pack_buf)

	best_val := math.inf_f32(1)
	best_step := 0

	initial_val := evaluate(model, val_samples, pack_buf, seq_len, DEFAULT_VAL_BATCHES)
	fmt.printfln("Initial val_loss = %.4f", initial_val)
	maybe_save_best(model, initial_val, &best_val, &best_step, 0)
	fmt.println()

	opt: ml.Optimizer
	t_start := time.tick_now()
	loss_running: f32
	loss_samples: int
	loss_tokens_total: int

	for step in 1 ..= steps {
		defer free_all(context.temp_allocator)

		pack := build_pack(train_samples, pack_buf, seq_len, context.temp_allocator)
		if builtin.len(pack.loss_inputs) == 0 {
			// Vanishingly unlikely, but guard against an empty pack.
			continue
		}

		ml.clear()
		logits     := gemma.forward(model, pack.tokens)
		loss_logits := ml.select(logits, pack.loss_inputs)
		token_loss := ml.cross_entropy(loss_logits, pack.loss_targets)
		ml.backward()

		loss_running     += read_mean_loss(token_loss)
		loss_samples     += 1
		loss_tokens_total += builtin.len(pack.loss_inputs)

		cur_lr := learning_rate_at(step, steps, DEFAULT_WARMUP, lr, 0.1)
		if ml.optimize(&opt, period=accum, learning_rate=cur_lr, weight_decay=DEFAULT_WEIGHT_DECAY) {
			gemma.update_lora(opt, model)
		}

		if step % log_every == 0 {
			elapsed   := f64(time.duration_seconds(time.tick_since(t_start)))
			loss_tps  := f64(loss_tokens_total) / elapsed
			fmt.printfln(
				"step %5v  train_loss = %.4f  lr = %.2e  (%.0f loss-tok/s)",
				step, loss_running / f32(loss_samples), cur_lr, loss_tps,
			)
			loss_running = 0
			loss_samples = 0
		}

		if val_every > 0 && step % val_every == 0 {
			val_loss := evaluate(model, val_samples, pack_buf, seq_len, DEFAULT_VAL_BATCHES)
			fmt.printfln("           val_loss = %.4f", val_loss)
			maybe_save_best(model, val_loss, &best_val, &best_step, step)
		}
	}

	val_loss := evaluate(model, val_samples, pack_buf, seq_len, DEFAULT_VAL_BATCHES)
	fmt.printfln("Final val_loss = %.4f   (best %.4f at step %v)", val_loss, best_val, best_step)
	maybe_save_best(model, val_loss, &best_val, &best_step, steps)

	fmt.println()
	fmt.printfln("Saving final adapter weights to %v ...", ADAPTER_OUT)
	if !save_adapters(model, ADAPTER_OUT) {
		fmt.eprintln("FAIL: adapter save failed.")
		os.exit(1)
	}
	fmt.println("  saved.")
}

Chat_Sample :: struct {
	tokens:     []int,
	asst_start: int,
	asst_end:   int,
}

destroy_samples :: proc(samples: []Chat_Sample) {
	for s in samples {
		delete(s.tokens)
	}
	delete(samples)
}

load_chat_samples :: proc(path: string) -> []Chat_Sample {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("FAIL: could not read %v: %v", path, err)
		os.exit(1)
	}
	defer delete(bytes)

	if builtin.len(bytes) < 12 || string(bytes[:8]) != "RSCHAT01" {
		fmt.eprintfln("FAIL: %v is not an RSCHAT01 file", path)
		os.exit(1)
	}

	count := int((^i32)(&bytes[8])^)
	out := builtin.make([]Chat_Sample, count)

	cursor := 12
	for i in 0 ..< count {
		if cursor + 12 > builtin.len(bytes) {
			fmt.eprintfln("FAIL: %v truncated at sample %v header", path, i)
			os.exit(1)
		}
		header := ([^]i32)(&bytes[cursor])[:3]
		total_len  := int(header[0])
		asst_start := int(header[1])
		asst_end   := int(header[2])
		cursor += 12

		body_bytes := total_len * 4
		if cursor + body_bytes > builtin.len(bytes) {
			fmt.eprintfln("FAIL: %v truncated at sample %v body", path, i)
			os.exit(1)
		}

		tokens := builtin.make([]int, total_len)
		body := ([^]i32)(&bytes[cursor])[:total_len]
		for k in 0 ..< total_len {
			tokens[k] = int(body[k])
		}
		cursor += body_bytes

		out[i] = Chat_Sample{
			tokens     = tokens,
			asst_start = asst_start,
			asst_end   = asst_end,
		}
	}
	return out
}

Pack :: struct {
	tokens:        []int,  // input token sequence (length seq_len)
	loss_inputs:   []int,  // input positions whose target is an assistant token
	loss_targets:  []int,  // corresponding target token ids
}

// Greedily fills the buffer with whole samples (random order). Skips a
// sampled record if the assistant span doesn't fully fit; that span is
// what we want loss on, so a partial copy with the loss truncated would
// just waste compute.
build_pack :: proc(samples: []Chat_Sample, buffer: []int, seq_len: int, allocator := context.allocator) -> (pack: Pack) {
	loss_inputs:  [dynamic]int
	loss_targets: [dynamic]int
	loss_inputs.allocator  = allocator
	loss_targets.allocator = allocator

	pos := 0
	attempts := 0
	max_attempts := builtin.len(samples) * 3

	for pos < seq_len + 1 && attempts < max_attempts {
		attempts += 1
		idx := rand.int_max(builtin.len(samples))
		s := samples[idx]
		if pos + builtin.len(s.tokens) > seq_len + 1 {
			continue
		}

		copy(buffer[pos:], s.tokens)

		// Loss positions: target index k is in [pos+asst_start, pos+asst_end);
		// the corresponding input position is k - 1.
		span_start := pos + s.asst_start
		span_end   := pos + s.asst_end
		for k in span_start ..< span_end {
			input_pos := k - 1
			if input_pos < 0 || input_pos >= seq_len {
				continue
			}
			append(&loss_inputs,  input_pos)
			append(&loss_targets, buffer[k])
		}
		pos += builtin.len(s.tokens)
	}

	for i in pos ..< seq_len + 1 {
		buffer[i] = 0 // pad / EOS
	}

	pack.tokens       = buffer[:seq_len]
	pack.loss_inputs  = loss_inputs[:]
	pack.loss_targets = loss_targets[:]
	return
}

evaluate :: proc(model: gemma.Gemma, samples: []Chat_Sample, buffer: []int, seq_len: int, batches: int) -> f32 {
	total: f32
	count: int
	for _ in 0 ..< batches {
		defer free_all(context.temp_allocator)

		pack := build_pack(samples, buffer, seq_len, context.temp_allocator)
		if builtin.len(pack.loss_inputs) == 0 {
			continue
		}

		ml.clear({.No_Gradients})
		logits     := gemma.forward(model, pack.tokens)
		loss_logits := ml.select(logits, pack.loss_inputs)
		token_loss := ml.cross_entropy(loss_logits, pack.loss_targets)
		total += read_mean_loss(token_loss)
		count += 1
	}
	return total / f32(count) if count > 0 else 0
}

learning_rate_at :: proc(step, total_steps, warmup_steps: int, max_lr, min_lr_frac: f32) -> f32 {
	if step < warmup_steps {
		return max_lr * f32(step) / f32(warmup_steps)
	}
	denom := total_steps - warmup_steps
	if denom <= 0 {
		return max_lr
	}
	progress := f32(step - warmup_steps) / f32(denom)
	if progress > 1 {
		progress = 1
	}
	cosine := 0.5 * (1 + math.cos(math.PI * progress))
	return max_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)
}

maybe_save_best :: proc(model: gemma.Gemma, val_loss: f32, best_val: ^f32, best_step: ^int, step: int) {
	if val_loss >= best_val^ {
		return
	}
	best_val^  = val_loss
	best_step^ = step
	if !save_adapters(model, ADAPTER_OUT_BEST) {
		fmt.eprintfln("WARNING: failed to save best adapter checkpoint to %v", ADAPTER_OUT_BEST)
		return
	}
	fmt.printfln("           ** new best val_loss; saved %v", ADAPTER_OUT_BEST)
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

parse_args :: proc(gguf, data_path: ^string, rank: ^int, alpha: ^f32, seq_len, steps, accum: ^int, lr: ^f32, log_every, val_every: ^int) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--gguf":      i += 1; gguf^      = args[i]; i += 1
		case "--data":      i += 1; data_path^ = args[i]; i += 1
		case "--rank":      i += 1; rank^      = _parse_int(args[i]); i += 1
		case "--alpha":     i += 1; alpha^     = f32(_parse_float(args[i])); i += 1
		case "--seq-len":   i += 1; seq_len^   = _parse_int(args[i]); i += 1
		case "--steps":     i += 1; steps^     = _parse_int(args[i]); i += 1
		case "--accum":     i += 1; accum^     = _parse_int(args[i]); i += 1
		case "--lr":        i += 1; lr^        = f32(_parse_float(args[i])); i += 1
		case "--log-every": i += 1; log_every^ = _parse_int(args[i]); i += 1
		case "--val-every": i += 1; val_every^ = _parse_int(args[i]); i += 1
		case "--help", "-h":
			fmt.println("usage: reascript_qlora [--gguf PATH] [--data BIN] [--rank N] [--alpha F] [--seq-len N] [--steps N] [--accum N] [--lr F] [--log-every N] [--val-every N]")
			os.exit(0)
		case:
			fmt.eprintfln("unknown argument: %v", arg)
			os.exit(1)
		}
	}
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
