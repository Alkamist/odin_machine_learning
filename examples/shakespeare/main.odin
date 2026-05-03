// Train a small Llama-style decoder from scratch on the
// Tiny Shakespeare corpus, byte-level. Proof of concept for
// the training path through the GPU backend.

package main

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml    "../../"
import gpu   "../../backends/gpu"
// import cpu   "../../backends/cpu"
import llama "../../networks/llama"

DATA_PATH :: "../data/shakespeare.txt"

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
MIN_LR_FRAC   :: f32(0.1)
WARMUP_STEPS  :: 100
WEIGHT_DECAY  :: 0.1
SEED          :: u64(0xC0FFEE)

base_config :: proc(use_qk_norm: bool) -> llama.Config {
	return llama.Config{
		layer_count       = 6,
		n_q_heads         = 6,
		n_kv_heads        = 6,
		head_size         = 64,
		embedding_size    = 384,
		intermediate_size = 1024,
		vocabulary_size   = VOCAB_SIZE,
		rope_base         = 10000,
		tied_embeddings   = true,
		use_qk_norm       = use_qk_norm,
	}
}

main :: proc() {
	defer fmt.println("Finished")

	use_qk_norm := false
	if builtin.len(os.args) > 1 && os.args[1] == "qknorm" {
		use_qk_norm = true
	}
	rand.reset(SEED)
	config := base_config(use_qk_norm)
	fmt.printfln("Variant: use_qk_norm = %v", use_qk_norm)

	corpus := load_corpus(DATA_PATH)
	defer delete(corpus)
	fmt.printfln("Corpus: %v bytes.", builtin.len(corpus))

	// Bytes that never appear as targets get no signal to push their
	// logits down, so we mask them out at sampling time.
	valid_bytes: [VOCAB_SIZE]bool
	for b in corpus {
		valid_bytes[b] = true
	}

	split    := (builtin.len(corpus) * 9) / 10
	train_set := corpus[:split]
	val_set   := corpus[split:]

	// cpu.set_thread_count(24)

	// ctx := cpu.context_create(1024 * 1024 * 1024)
	// defer cpu.context_destroy(ctx)

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)

	ml.context_scope(ctx)

	model := llama.make(config)
	defer llama.destroy(model)

	param_count := count_parameters(config)
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

		logits     := llama.forward(model, inputs[:])
		token_loss := ml.cross_entropy(logits, targets[:])

		ml.backward()

		loss_running += read_mean_loss(token_loss)
		loss_samples += 1

		lr := learning_rate_at(step, TOTAL_STEPS, WARMUP_STEPS, LEARNING_RATE, MIN_LR_FRAC)
		if ml.optimize(&opt, period=ACCUM_STEPS, learning_rate=lr, weight_decay=WEIGHT_DECAY) {
			llama.update(opt, model)
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

evaluate :: proc(model: llama.Llama, corpus: []int, batches: int) -> f32 {
	inputs:  [SEQ_LEN]int
	targets: [SEQ_LEN]int

	total: f32
	for _ in 0 ..< batches {
		defer free_all(context.temp_allocator)

		sample_window(corpus, inputs[:], targets[:])

		ml.clear({.No_Gradients})
		logits     := llama.forward(model, inputs[:])
		token_loss := ml.cross_entropy(logits, targets[:])
		total += read_mean_loss(token_loss)
	}
	return total / f32(batches)
}

sample :: proc(model: llama.Llama, prompt: string, gen_count: int, valid_bytes: []bool) {
	t_max := builtin.len(prompt) + gen_count + 4
	cache := llama.cache_make(model, t_max)
	defer llama.cache_destroy(cache)

	prompt_tokens := make([]int, builtin.len(prompt), context.temp_allocator)
	for i in 0 ..< builtin.len(prompt) {
		prompt_tokens[i] = int(prompt[i])
	}

	ml.clear({.No_Gradients})
	logits := llama.forward_cached(model, &cache, prompt_tokens)

	last_logits := make([]f32, VOCAB_SIZE)
	defer delete(last_logits)
	logits_buf  := make([]f32, ml.len(logits), context.temp_allocator)
	ml.get_data(logits, logits_buf)
	last_offset := (logits.shape[0] - 1) * VOCAB_SIZE
	copy(last_logits, logits_buf[last_offset:last_offset + VOCAB_SIZE])

	fmt.print(prompt)

	mask_invalid(last_logits, valid_bytes)
	next_token := sample_token(last_logits, SAMPLE_TEMP, SAMPLE_TOP_K)
	fmt.print(rune(next_token))

	for _ in 0 ..< gen_count - 1 {
		defer free_all(context.temp_allocator)

		ml.clear({.No_Gradients})
		step_logits := llama.forward_cached(model, &cache, {next_token})
		ml.get_data(step_logits, last_logits)

		mask_invalid(last_logits, valid_bytes)
		next_token = sample_token(last_logits, SAMPLE_TEMP, SAMPLE_TOP_K)
		fmt.print(rune(next_token))
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

count_parameters :: proc(c: llama.Config) -> int {
	q_size  := c.n_q_heads  * c.head_size
	kv_size := c.n_kv_heads * c.head_size

	embedding_params := c.vocabulary_size * c.embedding_size

	per_layer :=
		c.embedding_size +
		q_size  * c.embedding_size +
		kv_size * c.embedding_size +
		kv_size * c.embedding_size +
		c.embedding_size * q_size +
		c.embedding_size +
		c.intermediate_size * c.embedding_size +
		c.intermediate_size * c.embedding_size +
		c.embedding_size * c.intermediate_size

	output_params := c.embedding_size
	if !c.tied_embeddings {
		output_params += c.vocabulary_size * c.embedding_size
	}

	return embedding_params + per_layer * c.layer_count + output_params
}