// Train a small Gemma-style decoder from scratch on the Tiny Shakespeare
// corpus, byte-level. Mirrors examples/shakespeare/main.odin but exercises
// the Gemma-specific architecture (sliding/full layer alternation, qk-norm,
// per-layer inputs, final logit softcap). F32 path; bf16 mixed-precision
// is wired in at the optimizer/alloc layer but not exercised here.
//
// odin run examples/gemma_shakespeare -o:speed

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
TOTAL_STEPS  :: 5000
LOG_EVERY    :: 50
SAMPLE_EVERY :: 500
SAMPLE_LEN   :: 400
SAMPLE_TEMP  :: 0.8
SAMPLE_TOP_K :: 40

LEARNING_RATE :: 6e-4
MIN_LR_FRAC   :: 0.1
WARMUP_STEPS  :: 100
WEIGHT_DECAY  :: 0.1
SEED          :: 0xC0FFEE

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

	valid_bytes: [VOCAB_SIZE]bool
	for b in corpus {
		valid_bytes[b] = true
	}

	split     := (builtin.len(corpus) * 9) / 10
	train_set := corpus[:split]
	val_set   := corpus[split:]

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	cfg := tiny_config()
	defer gemma.config_destroy(cfg)

	model := gemma.make(cfg, .Bf16, for_training = true)
	defer gemma.destroy(model)

	gemma.randomize(model)

	param_count := count_parameters(cfg)
	fmt.printfln("Model: %v parameters.", param_count)

	opt: ml.Optimizer

	inputs:  [SEQ_LEN]int
	targets: [SEQ_LEN]int

	t_start := time.tick_now()
	loss_running: f32
	loss_samples: int

	for step in 1 ..= TOTAL_STEPS {
		defer free_all(context.temp_allocator)

		sample_window(train_set, inputs[:], targets[:])

		ml.clear()

		logits     := gemma.forward(model, inputs[:])
		token_loss := ml.cross_entropy(logits, targets[:])

		ml.backward()

		loss_running += read_mean_loss(token_loss)
		loss_samples += 1

		lr := learning_rate_at(step, TOTAL_STEPS, WARMUP_STEPS, LEARNING_RATE, MIN_LR_FRAC)
		if ml.optimize(&opt, period=ACCUM_STEPS, learning_rate=lr, weight_decay=WEIGHT_DECAY) {
			gemma.update(opt, model)
		}

		if step % LOG_EVERY == 0 {
			elapsed   := f64(time.duration_seconds(time.tick_since(t_start)))
			tokens    := step * SEQ_LEN
			tok_per_s := f64(tokens) / elapsed
			fmt.printfln(
				"step %5v  train_loss = %.4f  lr = %.2e  (%.0f tok/s)",
				step, loss_running / f32(loss_samples), lr, tok_per_s,
			)
			loss_running = 0
			loss_samples = 0
		}

		if step % SAMPLE_EVERY == 0 {
			val_loss := evaluate(model, val_set, 32)
			fmt.printfln("           val_loss   = %.4f", val_loss)
			fmt.println("---- sample ----")
			sample(model, "ROMEO:", SAMPLE_LEN, valid_bytes[:])
			fmt.println("\n----------------")
		}
	}
}

load_corpus :: proc(path: string) -> []int {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("Failed to read %v", path)
		os.exit(1)
	}
	defer delete(bytes)

	out := make([]int, builtin.len(bytes))
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
	buf   := make([]f32, count, context.temp_allocator)
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

sample :: proc(model: gemma.Gemma, prompt: string, gen_count: int, valid_bytes: []bool) {
	tokens := make([dynamic]int, 0, builtin.len(prompt) + gen_count)
	defer delete(tokens)
	for i in 0 ..< builtin.len(prompt) {
		append(&tokens, int(prompt[i]))
	}

	last_logits := make([]f32, VOCAB_SIZE)
	defer delete(last_logits)

	fmt.print(prompt)

	for _ in 0 ..< gen_count {
		defer free_all(context.temp_allocator)

		ml.clear({.No_Gradients})
		logits := gemma.forward(model, tokens[:])

		logits_buf := make([]f32, ml.len(logits), context.temp_allocator)
		ml.get_data(logits, logits_buf)
		last_offset := (logits.shape[0] - 1) * VOCAB_SIZE
		copy(last_logits, logits_buf[last_offset:last_offset + VOCAB_SIZE])

		mask_invalid(last_logits, valid_bytes)
		next_token := sample_token(last_logits, SAMPLE_TEMP, SAMPLE_TOP_K)
		fmt.print(rune(next_token))
		append(&tokens, next_token)
	}
}

mask_invalid :: proc(logits: []f32, valid_bytes: []bool) {
	for i in 0 ..< builtin.len(logits) {
		if !valid_bytes[i] {
			logits[i] = -1e30
		}
	}
}

sample_token :: proc(logits: []f32, temperature: f32, top_k: int) -> int {
	vocab := builtin.len(logits)

	indices := make([]int, vocab, context.temp_allocator)
	for i in 0 ..< vocab {
		indices[i] = i
	}

	keep := top_k
	if keep <= 0 || keep > vocab {
		keep = vocab
	}

	for slot in 0 ..< keep {
		best := slot
		for j in slot + 1 ..< vocab {
			if logits[indices[j]] > logits[indices[best]] {
				best = j
			}
		}
		indices[slot], indices[best] = indices[best], indices[slot]
	}

	max_logit := logits[indices[0]]

	probabilities := make([]f32, keep, context.temp_allocator)
	sum: f32
	for i in 0 ..< keep {
		probabilities[i] = math.exp((logits[indices[i]] - max_logit) / temperature)
		sum += probabilities[i]
	}
	for i in 0 ..< keep {
		probabilities[i] /= sum
	}

	r := rand.float32()
	cumulative: f32
	for i in 0 ..< keep {
		cumulative += probabilities[i]
		if r <= cumulative {
			return indices[i]
		}
	}
	return indices[keep - 1]
}

count_parameters :: proc(cfg: gemma.Config) -> int {
	per_layer := 0
	for layer_type, layer_idx in cfg.layer_types {
		head_dim := cfg.head_dim_full if layer_type == .Full else cfg.head_dim_sliding
		q_size   := cfg.num_attention_heads * head_dim
		kv_size  := cfg.num_key_value_heads * head_dim

		per_layer += 4 * cfg.hidden_size                       // 4 norms
		per_layer += q_size * cfg.hidden_size + head_dim       // q + q_norm
		per_layer += cfg.hidden_size * q_size                   // o
		if !is_kv_shared_layer(cfg, layer_idx) {
			per_layer += kv_size * cfg.hidden_size + head_dim  // k + k_norm
			per_layer += kv_size * cfg.hidden_size              // v
		}
		per_layer += cfg.intermediate_size * cfg.hidden_size * 3 // gate, up, down (down is same shape transposed)
		per_layer += cfg.hidden_size_per_layer_input * cfg.hidden_size
		per_layer += cfg.hidden_size * cfg.hidden_size_per_layer_input
		per_layer += cfg.hidden_size
		per_layer += 1                                           // layer_scalar
	}

	embedding := cfg.vocab_size * cfg.hidden_size
	output    := cfg.hidden_size
	if !cfg.tie_word_embeddings {
		output += cfg.vocab_size * cfg.hidden_size
	}
	per_layer_proj := cfg.num_hidden_layers * cfg.hidden_size_per_layer_input * cfg.hidden_size + cfg.hidden_size_per_layer_input

	return embedding + per_layer + output + per_layer_proj
}

is_kv_shared_layer :: proc(cfg: gemma.Config, layer_idx: int) -> bool {
	if cfg.num_kv_shared_layers == 0 {
		return false
	}
	return layer_idx >= cfg.num_hidden_layers - cfg.num_kv_shared_layers
}
