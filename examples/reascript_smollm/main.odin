// Full fine-tune of SmolLM2-135M-Instruct on the ReaScript chat dataset.
//
// Loads bf16 instruct weights, trains every parameter, applies assistant-only
// loss masking via ml.select. Samples are pre-tokenized into per-row records
// by tools/tokenize_chat and packed into a fixed-length sequence per step.
//
// Required:
//   smollm_data/model_instruct.safetensors
//   smollm_data/tokenizer.json
//   reascript/dataset/reascript_smollm.bin
//
//   odin run examples/reascript_smollm -o:speed -- --steps 1500
//
// Trained weights are saved at the end to reascript_smollm.safetensors.

package main

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml    "../../"
import gpu   "../../backends/cuda"
import llama "../../networks/llama"

DATA_PATH        :: "reascript/dataset/reascript_smollm.bin"
DEFAULT_MODEL    :: "smollm_data/model_instruct.safetensors"
WEIGHTS_OUT      :: "reascript_smollm_weights.bin"
WEIGHTS_OUT_BEST :: "reascript_smollm_weights_best.bin"

DEFAULT_SEQ_LEN      :: 2048
DEFAULT_STEPS        :: 1500
DEFAULT_ACCUM        :: 4
DEFAULT_LR           :: f32(1e-5)
DEFAULT_LOG_EVERY    :: 10
DEFAULT_VAL_EVERY    :: 100
DEFAULT_VAL_BATCHES  :: 32
DEFAULT_WEIGHT_DECAY :: f32(0.0)
DEFAULT_WARMUP       :: 50
DEFAULT_SEED         :: 0xC0FFEE
DEFAULT_VAL_FRACTION :: 0.10

main :: proc() {
	defer fmt.println("Finished")

	model_path := DEFAULT_MODEL
	data_path  := DATA_PATH
	seq_len    := DEFAULT_SEQ_LEN
	steps      := DEFAULT_STEPS
	accum      := DEFAULT_ACCUM
	lr         := DEFAULT_LR
	log_every  := DEFAULT_LOG_EVERY
	val_every  := DEFAULT_VAL_EVERY

	parse_args(&model_path, &data_path, &seq_len, &steps, &accum, &lr, &log_every, &val_every)

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

	fmt.println("Allocating SmolLM2-135M (F32 + Adam state) ...")
	model := llama.make(llama.SMOLLM2_135M_CONFIG)
	defer llama.destroy(model)

	fmt.printfln("Loading weights from %v ...", model_path)
	t_load := time.tick_now()
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		if !llama.load_safetensors(model, model_path) {
			fmt.eprintln("FAIL: weight loading failed.")
			os.exit(1)
		}
	}
	fmt.printfln("  loaded in %.1f s", f64(time.duration_seconds(time.tick_since(t_load))))

	fmt.printfln("Sequence length: %v   Steps: %v   Accum: %v   LR: %.2e", seq_len, steps, accum, f64(lr))
	fmt.println()

	pack_buf := builtin.make([]int, seq_len + 1)
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
			continue
		}

		ml.clear()
		logits      := llama.forward(model, pack.tokens)
		loss_logits := ml.select(logits, pack.loss_inputs)
		token_loss  := ml.cross_entropy(loss_logits, pack.loss_targets)
		ml.backward()

		loss_running     += read_mean_loss(token_loss)
		loss_samples     += 1
		loss_tokens_total += builtin.len(pack.loss_inputs)

		cur_lr := learning_rate_at(step, steps, DEFAULT_WARMUP, lr, 0.1)
		if ml.optimize(&opt, period=accum, learning_rate=cur_lr, weight_decay=DEFAULT_WEIGHT_DECAY) {
			llama.update(opt, model)
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
	fmt.printfln("Saving final weights to %v ...", WEIGHTS_OUT)
	if !save_weights(model, WEIGHTS_OUT) {
		fmt.eprintln("FAIL: weight save failed.")
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
	tokens:        []int,
	loss_inputs:   []int,
	loss_targets:  []int,
}

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
		buffer[i] = 0
	}

	pack.tokens       = buffer[:seq_len]
	pack.loss_inputs  = loss_inputs[:]
	pack.loss_targets = loss_targets[:]
	return
}

evaluate :: proc(model: llama.Llama, samples: []Chat_Sample, buffer: []int, seq_len: int, batches: int) -> f32 {
	total: f32
	count: int
	for _ in 0 ..< batches {
		defer free_all(context.temp_allocator)

		pack := build_pack(samples, buffer, seq_len, context.temp_allocator)
		if builtin.len(pack.loss_inputs) == 0 {
			continue
		}

		ml.clear({.No_Gradients})
		logits      := llama.forward(model, pack.tokens)
		loss_logits := ml.select(logits, pack.loss_inputs)
		token_loss  := ml.cross_entropy(loss_logits, pack.loss_targets)
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

maybe_save_best :: proc(model: llama.Llama, val_loss: f32, best_val: ^f32, best_step: ^int, step: int) {
	if val_loss >= best_val^ {
		return
	}
	best_val^  = val_loss
	best_step^ = step
	if !save_weights(model, WEIGHTS_OUT_BEST) {
		fmt.eprintfln("WARNING: failed to save best checkpoint to %v", WEIGHTS_OUT_BEST)
		return
	}
	fmt.printfln("           ** new best val_loss; saved %v", WEIGHTS_OUT_BEST)
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

// Simple per-tensor binary dump: name length + name + dtype + shape + bytes.
// Format (LE):
//   magic "SMLW0001" (8 bytes)
//   tensor_count (i32)
//   for each tensor:
//     name_len (i32) + name bytes
//     dtype (i32: 0=F32, 1=BF16)
//     rank (i32)
//     shape (rank * i32)
//     payload bytes
save_weights :: proc(model: llama.Llama, path: string) -> bool {
	buf: [dynamic]byte
	defer delete(buf)

	append(&buf, ..transmute([]byte)string("SMLW0001"))

	tensor_count_offset := builtin.len(buf)
	append(&buf, 0, 0, 0, 0)

	count: i32

	emit :: proc(buf: ^[dynamic]byte, count: ^i32, name: string, t: ml.Tensor) {
		_emit_named_tensor(buf, name, t)
		count^ += 1
	}

	emit(&buf, &count, "model.embed_tokens.weight", model.token_embeddings)
	for layer, i in model.layers {
		emit(&buf, &count, fmt.tprintf("model.layers.%v.input_layernorm.weight",          i), layer.input_norm_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.self_attn.q_proj.weight",         i), layer.q_proj_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.self_attn.k_proj.weight",         i), layer.k_proj_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.self_attn.v_proj.weight",         i), layer.v_proj_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.self_attn.o_proj.weight",         i), layer.o_proj_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.post_attention_layernorm.weight", i), layer.post_attn_norm_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.mlp.gate_proj.weight",            i), layer.gate_proj_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.mlp.up_proj.weight",              i), layer.up_proj_weight)
		emit(&buf, &count, fmt.tprintf("model.layers.%v.mlp.down_proj.weight",            i), layer.down_proj_weight)
	}
	emit(&buf, &count, "model.norm.weight", model.output_norm_weight)
	if !model.config.tied_embeddings {
		emit(&buf, &count, "lm_head.weight", model.lm_head_weight)
	}

	count_bytes := transmute([4]byte)count
	copy(buf[tensor_count_offset:tensor_count_offset + 4], count_bytes[:])

	return os.write_entire_file(path, buf[:]) == nil
}

_emit_named_tensor :: proc(buf: ^[dynamic]byte, name: string, t: ml.Tensor) {
	name_len := i32(builtin.len(name))
	name_len_bytes := transmute([4]byte)name_len
	append(buf, ..name_len_bytes[:])
	append(buf, ..transmute([]byte)name)

	dtype: i32
	switch t.type {
	case .F32:  dtype = 0
	case .Bf16: dtype = 1
	case .Q4_K, .Q6_K: panic("save_weights: quantized dtype not supported")
	}
	dtype_bytes := transmute([4]byte)dtype
	append(buf, ..dtype_bytes[:])

	rank := i32(t.rank)
	rank_bytes := transmute([4]byte)rank
	append(buf, ..rank_bytes[:])

	for d in 0 ..< t.rank {
		dim := i32(t.shape[d])
		dim_bytes := transmute([4]byte)dim
		append(buf, ..dim_bytes[:])
	}

	bytes_per_elem := 4 if t.type == .F32 else 2
	payload := builtin.make([]byte, ml.len(t) * bytes_per_elem, context.temp_allocator)
	ml.get_data_bytes(t, payload)
	append(buf, ..payload)
}

parse_args :: proc(model_path, data_path: ^string, seq_len, steps, accum: ^int, lr: ^f32, log_every, val_every: ^int) {
	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--model":     i += 1; model_path^ = args[i]; i += 1
		case "--data":      i += 1; data_path^  = args[i]; i += 1
		case "--seq-len":   i += 1; seq_len^    = _parse_int(args[i]); i += 1
		case "--steps":     i += 1; steps^      = _parse_int(args[i]); i += 1
		case "--accum":     i += 1; accum^      = _parse_int(args[i]); i += 1
		case "--lr":        i += 1; lr^         = f32(_parse_float(args[i])); i += 1
		case "--log-every": i += 1; log_every^  = _parse_int(args[i]); i += 1
		case "--val-every": i += 1; val_every^  = _parse_int(args[i]); i += 1
		case "--help", "-h":
			fmt.println("usage: reascript_smollm [--model PATH] [--data BIN] [--seq-len N] [--steps N] [--accum N] [--lr F] [--log-every N] [--val-every N]")
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
