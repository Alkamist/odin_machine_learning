// Train a ~50M-param Llama+QK-norm decoder on TinyStories with the
// SmolLM2 GPT-2 BPE tokenizer (vocab=49152). Tokens are pre-encoded
// by `tools/tinystories_dump.py` into flat int32 binaries.
//
// odin run examples/tinystories -o:speed

package main

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml    "../../"
import gpu   "../../backends/cuda"
import llama "../../networks/llama"
import gpt2  "../../tokenizers/gpt2"

DATA_DIR       :: "examples/data"
TRAIN_TOKENS   :: DATA_DIR + "/tinystories_train.bin"
VALID_TOKENS   :: DATA_DIR + "/tinystories_valid.bin"
TOKENIZER_PATH :: "smollm_data/tokenizer.json"

VOCAB_SIZE   :: 49152
SEQ_LEN      :: 512
ACCUM_STEPS  :: 8
TOTAL_STEPS  :: 30000
LOG_EVERY    :: 200
SAMPLE_EVERY :: 3000
VAL_BATCHES  :: 16
SAMPLE_LEN   :: 200
SAMPLE_TEMP  :: 0.8
SAMPLE_TOP_K :: 40

LEARNING_RATE :: 6e-4
MIN_LR_FRAC   :: 0.1
WARMUP_STEPS  :: 200
WEIGHT_DECAY  :: 0.1
SEED          :: 0xC0FFEE

CONFIG :: llama.Config{
	layer_count       = 6,
	n_q_heads         = 8,
	n_kv_heads        = 2,
	head_size         = 64,
	embedding_size    = 512,
	intermediate_size = 2048,
	vocabulary_size   = VOCAB_SIZE,
	rope_base         = 10000,
	tied_embeddings   = true,
	use_qk_norm       = true,
}

SAMPLE_PROMPT :: "Once upon a time, there was a little girl"

main :: proc() {
	defer fmt.println("Finished")

	rand.reset(SEED)

	fmt.println("Loading tokens ...")
	train_set := load_tokens(TRAIN_TOKENS)
	defer delete(train_set)
	valid_set := load_tokens(VALID_TOKENS)
	defer delete(valid_set)
	fmt.printfln("  train = %v tokens, valid = %v tokens", builtin.len(train_set), builtin.len(valid_set))

	fmt.println("Loading tokenizer ...")
	tokenizer, tokenizer_ok := gpt2.load(TOKENIZER_PATH)
	if !tokenizer_ok {
		fmt.eprintln("FAIL: could not load tokenizer.")
		os.exit(1)
	}
	defer gpt2.destroy(tokenizer)

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)

	ml.context_scope(ctx)

	model := llama.make(CONFIG)
	defer llama.destroy(model)

	param_count := count_parameters(CONFIG)
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
			val_loss := evaluate(model, valid_set, VAL_BATCHES)
			fmt.printfln("           val_loss   = %.4f", val_loss)
			fmt.println("---- sample ----")
			sample(model, &tokenizer, SAMPLE_PROMPT, SAMPLE_LEN)
			fmt.println("\n----------------")
		}
	}
}

load_tokens :: proc(path: string) -> []int {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("FAIL: could not read %v", path)
		os.exit(1)
	}
	defer delete(bytes)

	count := int((^u32le)(raw_data(bytes))^)
	expected_bytes := 4 + count * 4
	if expected_bytes > builtin.len(bytes) {
		fmt.eprintfln("FAIL: %v has %v bytes but header claims %v tokens", path, builtin.len(bytes), count)
		os.exit(1)
	}

	out := make([]int, count)
	for i in 0 ..< count {
		out[i] = int((^i32)(&bytes[4 + i * 4])^)
	}
	return out
}

sample_window :: proc(corpus: []int, inputs, targets: []int) {
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

sample :: proc(model: llama.Llama, tokenizer: ^gpt2.Tokenizer, prompt: string, gen_count: int) {
	// Sampling uses non-cached `llama.forward` so it goes through the same
	// f32 attention path as training. Slower than KV-cached generation
	// (recomputes attention over the full prefix each step), but avoids
	// needing an f32 attention_with_cache; the sampling budget here is small.
	//
	// `all_tokens` lives in the default allocator — keeping it on the temp
	// allocator while the loop body's temp churn happens elsewhere risks
	// the array vanishing under us between iterations.
	prompt_tokens := gpt2.encode(tokenizer, prompt, context.temp_allocator)
	if builtin.len(prompt_tokens) == 0 {
		fmt.println("(empty prompt tokenization)")
		return
	}

	last_logits := make([]f32, VOCAB_SIZE)
	defer delete(last_logits)

	all_tokens := make([dynamic]int, 0, builtin.len(prompt_tokens) + gen_count)
	defer delete(all_tokens)
	append(&all_tokens, ..prompt_tokens)

	prompt_text := gpt2.decode(tokenizer, prompt_tokens, context.temp_allocator)
	fmt.print(prompt_text)
	previous_decoded_length := builtin.len(prompt_text)

	for _ in 0 ..< gen_count {
		ml.clear({.No_Gradients})
		logits := llama.forward(model, all_tokens[:])

		logits_buf := make([]f32, ml.len(logits), context.temp_allocator)
		ml.get_data(logits, logits_buf)
		last_offset := (logits.shape[0] - 1) * VOCAB_SIZE
		copy(last_logits, logits_buf[last_offset : last_offset + VOCAB_SIZE])

		next_token := sample_token(last_logits, SAMPLE_TEMP, SAMPLE_TOP_K)
		append(&all_tokens, next_token)

		decoded := gpt2.decode(tokenizer, all_tokens[:], context.temp_allocator)
		if builtin.len(decoded) > previous_decoded_length {
			fmt.print(decoded[previous_decoded_length:])
			os.flush(os.stdout)
			previous_decoded_length = builtin.len(decoded)
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

	if c.use_qk_norm {
		per_layer += c.head_size * 2
	}

	output_params := c.embedding_size
	if !c.tied_embeddings {
		output_params += c.vocabulary_size * c.embedding_size
	}

	return embedding_params + per_layer * c.layer_count + output_params
}