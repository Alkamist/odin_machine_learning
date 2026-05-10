// Train a small Llama+QK-norm decoder on a flat DAC token stream produced
// by `tools/dac_lm_dump.py`. Vocab is 18432 (9 codebooks * 2 channels of
// 1024, frame-interleaved with a per-slot offset). No text conditioning.
// Periodically writes a sample of generated tokens that can be decoded
// back to stereo audio with `tools/dac_lm_decode.py`.
//
//   odin run examples/dac_lm -o:speed -- train
//   odin run examples/dac_lm -o:speed -- generate out.bin [num_tokens]
//
// `train` saves the final weights to examples/data/dac_lm.ckpt.
// `generate` loads that checkpoint and writes a token .bin that
// dac_lm_decode.py can render to wav. num_tokens defaults to a few seconds;
// values larger than SEQ_LEN are produced via a sliding KV cache so RoPE
// positions stay in the trained range.

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

DATA_DIR        :: "examples/data"
TRAIN_TOKENS    :: DATA_DIR + "/dac_lm_train.bin"
VALID_TOKENS    :: DATA_DIR + "/dac_lm_valid.bin"
SAMPLE_DIR      :: DATA_DIR + "/dac_lm_samples"
CHECKPOINT_PATH :: DATA_DIR + "/dac_lm.ckpt"

// Per-slot vocab structure mirrors the dumper: 9 RVQ codebooks per channel,
// 2 channels frame-interleaved => 18 disjoint 1024-entry slots per frame.
NUM_CODEBOOKS    :: 9
NUM_CHANNELS     :: 2
NUM_SLOTS        :: NUM_CODEBOOKS * NUM_CHANNELS    // tokens per frame
CODEBOOK_VOCAB   :: 1024
VOCAB_SIZE       :: NUM_SLOTS * CODEBOOK_VOCAB      // 18432

DAC_FRAME_RATE    :: 86
TOKENS_PER_SECOND :: DAC_FRAME_RATE * NUM_SLOTS

// Rolling-window generation: when the KV cache is about to exceed the
// trained RoPE range, drop the oldest tokens and rebuild from the tail.
SEQ_LEN       :: 2048
GEN_KEEP_TAIL :: SEQ_LEN / 2

ACCUM_STEPS  :: 4
TOTAL_STEPS  :: 8000
LOG_EVERY    :: 100
SAMPLE_EVERY :: 1000
VAL_BATCHES  :: 8

SAMPLE_PRIME  :: NUM_SLOTS * 8                   // seed tokens (=8 frames)
SAMPLE_TOKENS :: SEQ_LEN - SAMPLE_PRIME          // length of the during-training samples
SAMPLE_TEMP   :: 0.1                             // overfit model -> stay close to memorized trajectories
SAMPLE_TOP_K  :: 1

GENERATE_DEFAULT_SECONDS :: 8

LEARNING_RATE :: 6e-4
MIN_LR_FRAC   :: 0.1
WARMUP_STEPS  :: 200
WEIGHT_DECAY  :: 0.1
SEED          :: 0xDAC10

CONFIG :: llama.Config{
	layer_count       = 4,
	n_q_heads         = 8,
	n_kv_heads        = 2,
	head_size         = 64,
	embedding_size    = 384,
	intermediate_size = 1536,
	vocabulary_size   = VOCAB_SIZE,
	rope_base         = 10000,
	tied_embeddings   = true,
	use_qk_norm       = true,
}

main :: proc() {
	defer fmt.println("Finished")

	args := os.args[1:]
	mode := "train"
	if builtin.len(args) > 0 {
		mode = args[0]
	}

	switch mode {
	case "train":    run_train()
	case "generate": run_generate(args[1:])
	case:
		fmt.eprintfln("unknown subcommand %q (expected train | generate)", mode)
		os.exit(1)
	}
}

run_train :: proc() {
	rand.reset(SEED)

	fmt.println("Loading DAC tokens ...")
	train_set := load_tokens(TRAIN_TOKENS)
	defer delete(train_set)
	valid_set := load_tokens(VALID_TOKENS)
	defer delete(valid_set)
	fmt.printfln("  train = %v tokens, valid = %v tokens", builtin.len(train_set), builtin.len(valid_set))

	if !os.exists(SAMPLE_DIR) {
		os.make_directory(SAMPLE_DIR)
	}

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
			sample_path := fmt.tprintf("%v/sample_step_%05v.bin", SAMPLE_DIR, step)
			emit_sample(model, train_set, sample_path)
			fmt.printfln("           wrote %v", sample_path)

			if save_checkpoint(CHECKPOINT_PATH, model) {
				fmt.printfln("           saved %v", CHECKPOINT_PATH)
			}
		}
	}

	if save_checkpoint(CHECKPOINT_PATH, model) {
		fmt.printfln("Saved final checkpoint to %v", CHECKPOINT_PATH)
	}
}

run_generate :: proc(args: []string) {
	if builtin.len(args) < 1 {
		fmt.eprintln("generate: usage: generate <out.bin> [num_tokens]")
		os.exit(1)
	}
	out_path := args[0]

	num_tokens := GENERATE_DEFAULT_SECONDS * TOKENS_PER_SECOND
	if builtin.len(args) >= 2 {
		parsed, ok := parse_positive_int(args[1])
		if !ok {
			fmt.eprintfln("generate: invalid num_tokens %q", args[1])
			os.exit(1)
		}
		num_tokens = parsed
	}
	num_tokens = (num_tokens / NUM_SLOTS) * NUM_SLOTS  // keep whole frames

	rand.reset(SEED)

	fmt.println("Loading DAC tokens ...")
	train_set := load_tokens(TRAIN_TOKENS)
	defer delete(train_set)
	fmt.printfln("  train = %v tokens", builtin.len(train_set))

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)

	ml.context_scope(ctx)

	model := llama.make(CONFIG)
	defer llama.destroy(model)

	if !load_checkpoint(CHECKPOINT_PATH, model) {
		fmt.eprintfln("generate: failed to load %v (run train first)", CHECKPOINT_PATH)
		os.exit(1)
	}
	fmt.printfln("Loaded %v", CHECKPOINT_PATH)

	prime_count := SAMPLE_PRIME
	if prime_count > builtin.len(train_set) - 1 {
		prime_count = builtin.len(train_set) - 1
	}
	prime_offset := rand.int_max(builtin.len(train_set) - prime_count - 1)
	prime_offset  = (prime_offset / NUM_SLOTS) * NUM_SLOTS

	prime_tokens := make([]int, prime_count)
	defer delete(prime_tokens)
	for i in 0 ..< prime_count {
		prime_tokens[i] = train_set[prime_offset + i]
	}

	tokens := generate_rolling(model, prime_tokens, num_tokens, SAMPLE_TEMP, SAMPLE_TOP_K)
	defer delete(tokens)

	write_token_bin_persistent(out_path, tokens)
	fmt.printfln("Wrote %v (%v tokens, %.2f s)", out_path, builtin.len(tokens), f64(builtin.len(tokens)) / f64(TOKENS_PER_SECOND))
}

// Generate `new_token_count` tokens following `prime`. When the KV cache
// would exceed the trained context, drop everything but the last
// GEN_KEEP_TAIL output tokens and re-encode them from cache_position 0.
generate_rolling :: proc(model: llama.Llama, prime: []int, new_token_count: int, temperature: f32, top_k: int) -> []int {
	cache := llama.cache_make(model, SEQ_LEN)
	defer llama.cache_destroy(cache)

	all_tokens := make([dynamic]int, 0, builtin.len(prime) + new_token_count)
	append(&all_tokens, ..prime)

	last_logits := make([]f32, VOCAB_SIZE)
	defer delete(last_logits)

	prime_into_cache(model, &cache, prime, last_logits)

	for step_index in 0 ..< new_token_count {
		slot_index := (builtin.len(prime) + step_index) % NUM_SLOTS
		mask_to_slot(last_logits, slot_index)
		next_token := sample_token(last_logits, temperature, top_k)
		append(&all_tokens, next_token)

		if cache.length + 1 >= SEQ_LEN {
			tail_start := builtin.len(all_tokens) - GEN_KEEP_TAIL
			if tail_start < 0 {
				tail_start = 0
			}
			llama.cache_reset(&cache)
			prime_into_cache(model, &cache, all_tokens[tail_start:], last_logits)
			continue
		}

		ml.clear({.No_Gradients})
		single := [1]int{next_token}
		step_logits := llama.forward_cached(model, &cache, single[:])
		ml.get_data(step_logits, last_logits)
	}

	return all_tokens[:]
}

prime_into_cache :: proc(model: llama.Llama, cache: ^llama.Cache, tokens: []int, last_logits: []f32) {
	ml.clear({.No_Gradients})
	logits := llama.forward_cached(model, cache, tokens)
	logits_buf := make([]f32, ml.len(logits), context.temp_allocator)
	ml.get_data(logits, logits_buf)
	last_offset := (logits.shape[0] - 1) * VOCAB_SIZE
	copy(last_logits, logits_buf[last_offset : last_offset + VOCAB_SIZE])
}

write_token_bin_persistent :: proc(path: string, tokens: []int) {
	count := u32le(builtin.len(tokens))
	byte_count := 4 + builtin.len(tokens) * 4
	buf := make([]u8, byte_count)
	defer delete(buf)

	(^u32le)(raw_data(buf))^ = count
	for i in 0 ..< builtin.len(tokens) {
		(^i32le)(&buf[4 + i * 4])^ = i32le(i32(tokens[i]))
	}

	if err := os.write_entire_file(path, buf); err != nil {
		fmt.eprintfln("FAIL: could not write %v: %v", path, err)
	}
}

parse_positive_int :: proc(s: string) -> (int, bool) {
	value := 0
	for r in s {
		if r < '0' || r > '9' {
			return 0, false
		}
		value = value * 10 + int(r - '0')
	}
	if value <= 0 {
		return 0, false
	}
	return value, true
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

emit_sample :: proc(model: llama.Llama, valid_set: []int, out_path: string) {
	prime_count := SAMPLE_PRIME
	if prime_count > builtin.len(valid_set) - 1 {
		prime_count = builtin.len(valid_set) - 1
	}
	prime_offset := rand.int_max(builtin.len(valid_set) - prime_count - 1)
	prime_offset = (prime_offset / NUM_SLOTS) * NUM_SLOTS  // align to slot-0 boundary

	prime_tokens := make([]int, prime_count, context.temp_allocator)
	for i in 0 ..< prime_count {
		prime_tokens[i] = valid_set[prime_offset + i]
	}

	t_max := prime_count + SAMPLE_TOKENS + 4
	cache := llama.cache_make(model, t_max)
	defer llama.cache_destroy(cache)

	last_logits := make([]f32, VOCAB_SIZE)
	defer delete(last_logits)

	ml.clear({.No_Gradients})
	logits := llama.forward_cached(model, &cache, prime_tokens)
	logits_buf := make([]f32, ml.len(logits), context.temp_allocator)
	ml.get_data(logits, logits_buf)
	last_offset := (logits.shape[0] - 1) * VOCAB_SIZE
	copy(last_logits, logits_buf[last_offset : last_offset + VOCAB_SIZE])

	all_tokens := make([dynamic]int, 0, prime_count + SAMPLE_TOKENS, context.temp_allocator)
	append(&all_tokens, ..prime_tokens)

	for step_index in 0 ..< SAMPLE_TOKENS {
		mask_to_slot(last_logits, (prime_count + step_index) % NUM_SLOTS)
		next_token := sample_token(last_logits, SAMPLE_TEMP, SAMPLE_TOP_K)
		append(&all_tokens, next_token)

		ml.clear({.No_Gradients})
		single := [1]int{next_token}
		step_logits := llama.forward_cached(model, &cache, single[:])
		ml.get_data(step_logits, last_logits)
	}

	write_token_bin(out_path, all_tokens[:])
}

write_token_bin :: proc(path: string, tokens: []int) {
	count := u32le(builtin.len(tokens))
	byte_count := 4 + builtin.len(tokens) * 4
	buf := make([]u8, byte_count, context.temp_allocator)

	(^u32le)(raw_data(buf))^ = count
	for i in 0 ..< builtin.len(tokens) {
		(^i32le)(&buf[4 + i * 4])^ = i32le(i32(tokens[i]))
	}

	if err := os.write_entire_file(path, buf); err != nil {
		fmt.eprintfln("FAIL: could not write %v: %v", path, err)
	}
}

// Force-zero probability outside the expected slot range for this position.
// Each slot's value range is disjoint by construction; this kills decoder
// glitches that would otherwise come from cross-slot leakage.
mask_to_slot :: proc(logits: []f32, slot_index: int) {
	low  := slot_index * CODEBOOK_VOCAB
	high := low + CODEBOOK_VOCAB
	for i in 0 ..< builtin.len(logits) {
		if i < low || i >= high {
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

	if c.use_qk_norm {
		per_layer += c.head_size * 2
	}

	output_params := c.embedding_size
	if !c.tied_embeddings {
		output_params += c.vocabulary_size * c.embedding_size
	}

	return embedding_params + per_layer * c.layer_count + output_params
}