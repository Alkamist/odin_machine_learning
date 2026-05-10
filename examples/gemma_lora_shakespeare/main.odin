// Two-phase validation of the LoRA mechanism on tiny Gemma:
//   Phase 1 — Full fine-tune the base from random init for PHASE1_STEPS.
//             Confirms the base reaches a non-trivial loss.
//   Phase 2 — Freeze the base, attach LoRA adapters, train only the
//             adapters for PHASE2_STEPS. Confirms loss continues to drop
//             with only adapter params trainable.
//
// If phase 2 reduces loss meaningfully below where phase 1 ended, the
// LoRA forward/backward path is wired correctly (frozen-weight skip in
// linear_backward, adapter parameter routing through Adam, etc.).
//
// odin run examples/gemma_lora_shakespeare -o:speed

package main

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml    "../../"
import gpu   "../../backends/cuda"
import gemma "../../networks/gemma"

DATA_PATH :: "examples/data/shakespeare.txt"

VOCAB_SIZE   :: 256
SEQ_LEN      :: 128
ACCUM_STEPS  :: 8
PHASE1_STEPS :: 1500
PHASE2_STEPS :: 1500
LOG_EVERY    :: 50

LEARNING_RATE :: 6e-4
LORA_LR       :: 6e-4
MIN_LR_FRAC   :: 0.1
WARMUP_STEPS  :: 100
WEIGHT_DECAY  :: 0.1
SEED          :: 0xC0FFEE

LORA_RANK  :: 16
LORA_ALPHA :: f32(32) // alpha = 2 * rank is a common default

tiny_config :: proc() -> gemma.Config {
	cfg := gemma.Config{
		num_hidden_layers           = 4,
		hidden_size                 = 256,
		intermediate_size           = 768,
		num_attention_heads         = 4,
		num_key_value_heads         = 2,
		head_dim_sliding            = 64,
		head_dim_full               = 64,
		vocab_size                  = VOCAB_SIZE,
		max_position_embeddings     = SEQ_LEN,
		sliding_window              = 64,
		hidden_size_per_layer_input = 64,
		num_kv_shared_layers        = 0,
		rope_base_sliding           = 10000,
		rope_base_full              = 10000,
		rope_fraction_full          = 1.0,
		rms_norm_eps                = 1e-6,
		final_logit_softcapping     = 0,
		tie_word_embeddings         = true,
	}

	cfg.layer_types = builtin.make([]gemma.Layer_Type, cfg.num_hidden_layers)
	for i in 0 ..< cfg.num_hidden_layers {
		cfg.layer_types[i] = .Full if (i + 1) % 4 == 0 else .Sliding
	}
	return cfg
}

main :: proc() {
	defer fmt.println("Finished")

	rand.reset(SEED)

	corpus := load_corpus(DATA_PATH)
	defer delete(corpus)
	fmt.printfln("Corpus: %v bytes.", builtin.len(corpus))

	split     := (builtin.len(corpus) * 9) / 10
	train_set := corpus[:split]
	val_set   := corpus[split:]

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	cfg := tiny_config()
	defer gemma.config_destroy(cfg)

	// ----- Phase 1: full fine-tune the base ------------------------------
	fmt.println("=== Phase 1: full fine-tune base ===")
	{
		model := gemma.make(cfg, .Bf16, for_training = true)
		defer gemma.destroy(model)
		gemma.randomize(model)

		opt: ml.Optimizer
		train_loop(
			model     = model,
			train_set = train_set,
			val_set   = val_set,
			opt       = &opt,
			steps     = PHASE1_STEPS,
			lr        = LEARNING_RATE,
			use_lora  = false,
		)

		// Save base weights to host so phase 2 starts from the same point.
		save_base_to_host(&saved_base, model)
	}

	// ----- Phase 2: freeze base, attach LoRA, train adapters --------------
	fmt.println()
	fmt.println("=== Phase 2: frozen base + LoRA adapters ===")
	{
		lora_cfg := gemma.LoRA_Config{
			rank    = LORA_RANK,
			alpha   = LORA_ALPHA,
			targets = gemma.LORA_DEFAULT_TARGETS,
		}
		model := gemma.make(cfg, .Bf16, for_training = true, lora_cfg = lora_cfg)
		defer gemma.destroy(model)

		// Restore the base from phase 1 so we're not starting from random.
		restore_base_from_host(saved_base, model)
		gemma.randomize_lora(model)

		fmt.printfln("LoRA params: %v (%.2f%% of base)",
			gemma.lora_parameter_count(model),
			100.0 * f64(gemma.lora_parameter_count(model)) / f64(count_base_parameters(cfg)))

		opt: ml.Optimizer
		train_loop(
			model     = model,
			train_set = train_set,
			val_set   = val_set,
			opt       = &opt,
			steps     = PHASE2_STEPS,
			lr        = LORA_LR,
			use_lora  = true,
		)
	}
}

// Storage for the phase-1 base weights, copied between phases.
Base_Weights :: struct {
	embed_tokens:     []byte,
	output_norm:      []f32,
	per_layer_proj:   []byte,
	per_layer_norm:   []f32,
	layers:           []Base_Layer,
}

Base_Layer :: struct {
	input_norm, post_attn_norm, pre_ff_norm, post_ff_norm: []f32,
	q, q_norm, o:                                          []byte,
	k, k_norm, v:                                          []byte,
	gate, up, down:                                        []byte,
	per_layer_input_gate, per_layer_proj:                  []byte,
	post_per_layer_norm, layer_scalar:                     []f32,
	has_kv:                                                bool,
}

saved_base: Base_Weights

save_base_to_host :: proc(dst: ^Base_Weights, model: gemma.Gemma) {
	dst.embed_tokens   = read_tensor_bytes(model.embed_tokens_weight)
	dst.output_norm    = read_tensor_f32  (model.output_norm_weight)
	dst.per_layer_proj = read_tensor_bytes(model.per_layer_model_projection_weight)
	dst.per_layer_norm = read_tensor_f32  (model.per_layer_projection_norm_weight)

	dst.layers = builtin.make([]Base_Layer, builtin.len(model.layers))
	for layer, idx in model.layers {
		bl := &dst.layers[idx]
		bl.input_norm     = read_tensor_f32(layer.input_norm_weight)
		bl.post_attn_norm = read_tensor_f32(layer.post_attention_norm_weight)
		bl.pre_ff_norm    = read_tensor_f32(layer.pre_feedforward_norm_weight)
		bl.post_ff_norm   = read_tensor_f32(layer.post_feedforward_norm_weight)

		bl.q      = read_tensor_bytes(layer.q_proj_weight)
		bl.q_norm = read_tensor_bytes(layer.q_norm_weight)
		bl.o      = read_tensor_bytes(layer.o_proj_weight)

		bl.has_kv = layer.k_proj_weight.backend != nil
		if bl.has_kv {
			bl.k      = read_tensor_bytes(layer.k_proj_weight)
			bl.k_norm = read_tensor_bytes(layer.k_norm_weight)
			bl.v      = read_tensor_bytes(layer.v_proj_weight)
		}

		bl.gate = read_tensor_bytes(layer.gate_proj_weight)
		bl.up   = read_tensor_bytes(layer.up_proj_weight)
		bl.down = read_tensor_bytes(layer.down_proj_weight)

		bl.per_layer_input_gate = read_tensor_bytes(layer.per_layer_input_gate_weight)
		bl.per_layer_proj       = read_tensor_bytes(layer.per_layer_projection_weight)
		bl.post_per_layer_norm  = read_tensor_f32  (layer.post_per_layer_input_norm_weight)
		bl.layer_scalar         = read_tensor_f32  (layer.layer_scalar)
	}

	// Per-layer-input bytes are host-side already; copy into a separate slot
	// so phase 2 can restore them too.
	saved_per_layer_bytes = builtin.make([]byte, builtin.len(model.embed_tokens_per_layer_bytes))
	copy(saved_per_layer_bytes, model.embed_tokens_per_layer_bytes)
}

saved_per_layer_bytes: []byte

restore_base_from_host :: proc(src: Base_Weights, model: gemma.Gemma) {
	write_tensor_bytes(model.embed_tokens_weight,                 src.embed_tokens)
	write_tensor_f32  (model.output_norm_weight,                  src.output_norm)
	write_tensor_bytes(model.per_layer_model_projection_weight,   src.per_layer_proj)
	write_tensor_f32  (model.per_layer_projection_norm_weight,    src.per_layer_norm)

	for layer, idx in model.layers {
		bl := src.layers[idx]
		write_tensor_f32  (layer.input_norm_weight,             bl.input_norm)
		write_tensor_f32  (layer.post_attention_norm_weight,    bl.post_attn_norm)
		write_tensor_f32  (layer.pre_feedforward_norm_weight,   bl.pre_ff_norm)
		write_tensor_f32  (layer.post_feedforward_norm_weight,  bl.post_ff_norm)

		write_tensor_bytes(layer.q_proj_weight, bl.q)
		write_tensor_bytes(layer.q_norm_weight, bl.q_norm)
		write_tensor_bytes(layer.o_proj_weight, bl.o)

		if bl.has_kv {
			write_tensor_bytes(layer.k_proj_weight, bl.k)
			write_tensor_bytes(layer.k_norm_weight, bl.k_norm)
			write_tensor_bytes(layer.v_proj_weight, bl.v)
		}

		write_tensor_bytes(layer.gate_proj_weight, bl.gate)
		write_tensor_bytes(layer.up_proj_weight,   bl.up)
		write_tensor_bytes(layer.down_proj_weight, bl.down)

		write_tensor_bytes(layer.per_layer_input_gate_weight,      bl.per_layer_input_gate)
		write_tensor_bytes(layer.per_layer_projection_weight,      bl.per_layer_proj)
		write_tensor_f32  (layer.post_per_layer_input_norm_weight, bl.post_per_layer_norm)
		write_tensor_f32  (layer.layer_scalar,                     bl.layer_scalar)
	}

	copy(model.embed_tokens_per_layer_bytes, saved_per_layer_bytes)
}

read_tensor_bytes :: proc(t: ml.Tensor) -> []byte {
	count := ml.len(t)
	bytes_per_elem := 2 if t.type == .Bf16 else 4
	out := builtin.make([]byte, count * bytes_per_elem)
	ml.get_data_bytes(t, out)
	return out
}

write_tensor_bytes :: proc(t: ml.Tensor, src: []byte) {
	ml.set_data_bytes(t, src)
}

read_tensor_f32 :: proc(t: ml.Tensor) -> []f32 {
	count := ml.len(t)
	out := builtin.make([]f32, count)
	if t.type == .F32 {
		ml.get_data(t, out)
	} else {
		// bf16 -> f32 via raw bytes + conversion
		byte_buf := builtin.make([]byte, count * 2, context.temp_allocator)
		ml.get_data_bytes(t, byte_buf)
		bf := ([^]ml.Bf16)(raw_data(byte_buf))[:count]
		for i in 0 ..< count {
			out[i] = ml.bf16_to_f32(bf[i])
		}
	}
	return out
}

write_tensor_f32 :: proc(t: ml.Tensor, src: []f32) {
	if t.type == .F32 {
		ml.set_data(t, src)
	} else {
		count := builtin.len(src)
		byte_buf := builtin.make([]byte, count * 2, context.temp_allocator)
		bf := ([^]ml.Bf16)(raw_data(byte_buf))[:count]
		for i in 0 ..< count {
			bf[i] = ml.bf16_from_f32(src[i])
		}
		ml.set_data_bytes(t, byte_buf)
	}
}

train_loop :: proc(model: gemma.Gemma, train_set, val_set: []int, opt: ^ml.Optimizer, steps: int, lr: f32, use_lora: bool) {
	inputs:  [SEQ_LEN]int
	targets: [SEQ_LEN]int

	t_start := time.tick_now()
	loss_running: f32
	loss_samples: int

	for step in 1 ..= steps {
		defer free_all(context.temp_allocator)

		sample_window(train_set, inputs[:], targets[:])

		ml.clear()

		logits     := gemma.forward(model, inputs[:])
		token_loss := ml.cross_entropy(logits, targets[:])

		ml.backward()

		loss_running += read_mean_loss(token_loss)
		loss_samples += 1

		cur_lr := learning_rate_at(step, steps, WARMUP_STEPS, lr, MIN_LR_FRAC)
		if ml.optimize(opt, period=ACCUM_STEPS, learning_rate=cur_lr, weight_decay=WEIGHT_DECAY) {
			if use_lora {
				gemma.update_lora(opt^, model)
			} else {
				gemma.update(opt^, model)
			}
		}

		if step % LOG_EVERY == 0 {
			elapsed   := f64(time.duration_seconds(time.tick_since(t_start)))
			tokens    := step * SEQ_LEN
			tok_per_s := f64(tokens) / elapsed
			fmt.printfln(
				"step %5v  train_loss = %.4f  lr = %.2e  (%.0f tok/s)",
				step, loss_running / f32(loss_samples), cur_lr, tok_per_s,
			)
			loss_running = 0
			loss_samples = 0
		}
	}

	val_loss := evaluate(model, val_set, 32)
	fmt.printfln("           final val_loss = %.4f", val_loss)
}

load_corpus :: proc(path: string) -> []int {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("Failed to read %v", path)
		os.exit(1)
	}
	defer delete(bytes)

	out := builtin.make([]int, builtin.len(bytes))
	for i in 0 ..< builtin.len(bytes) {
		out[i] = int(bytes[i])
	}
	return out
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

sample_window :: proc(corpus: []int, inputs, targets: []int) {
	max_offset := builtin.len(corpus) - builtin.len(inputs) - 1
	offset     := rand.int_max(max_offset)
	for i in 0 ..< builtin.len(inputs) {
		inputs[i]  = corpus[offset + i]
		targets[i] = corpus[offset + i + 1]
	}
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

evaluate :: proc(model: gemma.Gemma, corpus: []int, batches: int) -> f32 {
	inputs:  [SEQ_LEN]int
	targets: [SEQ_LEN]int

	total: f32
	for _ in 0 ..< batches {
		defer free_all(context.temp_allocator)

		sample_window(corpus, inputs[:], targets[:])

		ml.clear({.No_Gradients})
		logits     := gemma.forward(model, inputs[:])
		token_loss := ml.cross_entropy(logits, targets[:])
		total += read_mean_loss(token_loss)
	}
	return total / f32(batches)
}

count_base_parameters :: proc(cfg: gemma.Config) -> int {
	per_layer := 0
	for layer_type, layer_idx in cfg.layer_types {
		head_dim := cfg.head_dim_full if layer_type == .Full else cfg.head_dim_sliding
		q_size   := cfg.num_attention_heads * head_dim
		kv_size  := cfg.num_key_value_heads * head_dim

		per_layer += 4 * cfg.hidden_size
		per_layer += q_size * cfg.hidden_size + head_dim
		per_layer += cfg.hidden_size * q_size
		if layer_idx < cfg.num_hidden_layers - cfg.num_kv_shared_layers {
			per_layer += kv_size * cfg.hidden_size + head_dim
			per_layer += kv_size * cfg.hidden_size
		}
		per_layer += cfg.intermediate_size * cfg.hidden_size * 3
		per_layer += cfg.hidden_size_per_layer_input * cfg.hidden_size
		per_layer += cfg.hidden_size * cfg.hidden_size_per_layer_input
		per_layer += cfg.hidden_size
		per_layer += 1
	}

	embedding := cfg.vocab_size * cfg.hidden_size
	output    := cfg.hidden_size
	if !cfg.tie_word_embeddings {
		output += cfg.vocab_size * cfg.hidden_size
	}
	per_layer_proj := cfg.num_hidden_layers * cfg.hidden_size_per_layer_input * cfg.hidden_size + cfg.hidden_size_per_layer_input

	return embedding + per_layer + output + per_layer_proj
}
