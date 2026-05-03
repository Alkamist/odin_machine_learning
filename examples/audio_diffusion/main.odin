// Train a small DiT-style diffusion model on EnCodec-24k continuous latents,
// conditioned on the codec's coarse cb0 token per frame. The model learns
// p(latent | cb0): given a coarse-token sequence (from the autoregressive
// transformer at inference, or ground-truth at train time), produce the
// continuous 128-d latent per frame that EnCodec's decoder will turn into
// audio.
//
//   odin run examples/audio_diffusion -o:speed
//
// Periodically writes a generated latent stream as .bin files compatible
// with `tools/audio_diffusion_decode.py`, so you can render them to wav and
// listen.

package main

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml  "../../"
import gpu "../../backends/gpu"

DATA_DIR    :: "examples/data"
TRAIN_BIN   :: DATA_DIR + "/audio_diffusion_train.bin"
VALID_BIN   :: DATA_DIR + "/audio_diffusion_valid.bin"
SAMPLE_DIR  :: DATA_DIR + "/audio_diffusion_samples"

LATENT_DIM     :: 128
COARSE_VOCAB   :: 1024
SEQ_LEN        :: 256        // training/sampling window (frames)
MAX_SEQ_LEN    :: 512        // positional-embedding capacity
TIME_EMBED_DIM :: 32

HIDDEN_SIZE :: 384
N_HEADS     :: 6
HEAD_SIZE   :: HIDDEN_SIZE / N_HEADS
N_LAYERS    :: 4
FF_MULT     :: 4

TOTAL_STEPS  :: 8000
ACCUM_STEPS  :: 4
LOG_EVERY    :: 100
SAMPLE_EVERY :: 1000
SAMPLE_STEPS :: 50

LEARNING_RATE :: f32(2e-4)
WEIGHT_DECAY  :: f32(0)
EMA_DECAY     :: f32(0.999)
SEED          :: u64(0xAD1F0)

main :: proc() {
	defer fmt.println("Finished")

	rand.reset(SEED)

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	if !os.exists(SAMPLE_DIR) {
		os.make_directory(SAMPLE_DIR)
	}

	fmt.println("Loading audio diffusion data ...")
	train := dataset_load(TRAIN_BIN)
	defer dataset_destroy(train)
	valid := dataset_load(VALID_BIN)
	defer dataset_destroy(valid)
	fmt.printfln("  train: %v frames (%.1f s), valid: %v frames (%.1f s)",
		train.num_frames, f32(train.num_frames) / f32(train.frame_rate),
		valid.num_frames, f32(valid.num_frames) / f32(valid.frame_rate))
	fmt.printfln("  latent_mean = %.4f  latent_std = %.4f", train.latent_mean, train.latent_std)

	model := dit_make()
	defer dit_destroy(model)
	fmt.printfln("Model: %v parameters", count_parameters())

	ema := ema_make(dit_params(model), EMA_DECAY)
	defer ema_destroy(ema)

	loss_acc := ml.alloc(.F32, []int{1}, persistent=true, buffers={.Data})
	defer ml.destroy(loss_acc)
	ml.fill_value(loss_acc, 0)

	opt: ml.Optimizer

	x_t_buf      := make([]f32, SEQ_LEN * LATENT_DIM)
	velocity_buf := make([]f32, SEQ_LEN * LATENT_DIM)
	time_emb_buf := make([]f32, TIME_EMBED_DIM)
	cb0_buf      := make([]int, SEQ_LEN)
	defer delete(x_t_buf)
	defer delete(velocity_buf)
	defer delete(time_emb_buf)
	defer delete(cb0_buf)

	t_start := time.tick_now()

	for step in 1 ..= TOTAL_STEPS {
		defer free_all(context.temp_allocator)

		build_training_window(train, x_t_buf, velocity_buf, cb0_buf, time_emb_buf)

		ml.clear()

		x_t_tensor      := ml.reshape(ml.tensor(x_t_buf),      {SEQ_LEN, LATENT_DIM})
		target_tensor   := ml.reshape(ml.tensor(velocity_buf), {SEQ_LEN, LATENT_DIM})
		time_tensor     := ml.tensor(time_emb_buf)
		predicted_field := dit_forward(model, x_t_tensor, cb0_buf, time_tensor)
		loss            := ml.mean_squared_error(predicted_field, target_tensor)

		ml.backward()

		ml.accumulate_mean(loss_acc, loss)

		if ml.optimize(&opt, period=ACCUM_STEPS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY) {
			dit_update(opt, model)
			ema_update(&ema, dit_params(model))
		}

		if step % LOG_EVERY == 0 {
			loss_host: [1]f32
			ml.get_data(loss_acc, loss_host[:])
			elapsed := f64(time.duration_seconds(time.tick_since(t_start)))
			fmt.printfln(
				"step %5v  loss = %.5f  (%.0f frames/s)",
				step, loss_host[0] / f32(LOG_EVERY), f64(step * SEQ_LEN) / elapsed,
			)
			ml.fill_value(loss_acc, 0)
		}

		if step % SAMPLE_EVERY == 0 {
			sample_path := fmt.tprintf("%v/sample_step_%05v.bin", SAMPLE_DIR, step)
			ema_swap_in(&ema, dit_params(model))
			emit_sample(model, valid, sample_path)
			ema_swap_out(&ema, dit_params(model))
			fmt.printfln("           wrote %v", sample_path)
		}
	}
}

// --- Dataset --------------------------------------------------------------

Dataset :: struct {
	num_frames:     int,
	latent_dim:     int,
	codebook_vocab: int,
	sample_rate:    int,
	frame_rate:     int,
	latent_mean:    f32,
	latent_std:     f32,
	latents_norm:   []f32,    // [num_frames, latent_dim], normalized to ~unit variance
	cb0:            []int,
}

DATASET_HEADER_BYTES :: 64
DATASET_MAGIC        :: u32(0xC0DECDAB)

dataset_load :: proc(path: string) -> Dataset {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("FAIL: could not read %v", path)
		os.exit(1)
	}
	defer delete(bytes)

	read_u32 :: proc(bytes: []byte, offset: int) -> u32 {
		return u32((^u32le)(&bytes[offset])^)
	}
	read_f32 :: proc(bytes: []byte, offset: int) -> f32 {
		return f32((^f32le)(&bytes[offset])^)
	}

	magic := read_u32(bytes, 0)
	if magic != DATASET_MAGIC {
		fmt.eprintfln("FAIL: bad magic %#x in %v", magic, path)
		os.exit(1)
	}

	d: Dataset
	d.num_frames     = int(read_u32(bytes, 8))
	d.latent_dim     = int(read_u32(bytes, 12))
	d.codebook_vocab = int(read_u32(bytes, 16))
	d.sample_rate    = int(read_u32(bytes, 20))
	d.frame_rate     = int(read_u32(bytes, 24))
	d.latent_mean    = read_f32(bytes, 32)
	d.latent_std     = read_f32(bytes, 36)

	if d.latent_dim != LATENT_DIM {
		fmt.eprintfln("FAIL: %v latent_dim=%v, expected %v", path, d.latent_dim, LATENT_DIM)
		os.exit(1)
	}
	if d.codebook_vocab > COARSE_VOCAB {
		fmt.eprintfln("FAIL: %v codebook_vocab=%v, expected <= %v", path, d.codebook_vocab, COARSE_VOCAB)
		os.exit(1)
	}

	latent_byte_count := d.num_frames * d.latent_dim * 4
	cb0_byte_count    := d.num_frames * 4

	d.latents_norm = make([]f32, d.num_frames * d.latent_dim)
	d.cb0          = make([]int, d.num_frames)

	latent_offset := DATASET_HEADER_BYTES
	cb0_offset    := latent_offset + latent_byte_count
	if len(bytes) < cb0_offset + cb0_byte_count {
		fmt.eprintfln("FAIL: %v truncated (%v bytes)", path, len(bytes))
		os.exit(1)
	}

	inv_std := 1.0 / d.latent_std
	for i in 0 ..< len(d.latents_norm) {
		raw := read_f32(bytes, latent_offset + i * 4)
		d.latents_norm[i] = (raw - d.latent_mean) * inv_std
	}
	for i in 0 ..< d.num_frames {
		d.cb0[i] = int(i32((^i32le)(&bytes[cb0_offset + i * 4])^))
	}

	return d
}

dataset_destroy :: proc(d: Dataset) {
	delete(d.latents_norm)
	delete(d.cb0)
}

// --- Training-window construction ----------------------------------------

build_training_window :: proc(d: Dataset, x_t, velocity: []f32, cb0: []int, time_emb: []f32) {
	max_offset := d.num_frames - SEQ_LEN
	assert(max_offset > 0, "training set must be longer than SEQ_LEN")
	offset := rand.int_max(max_offset)

	t := rand.float32()

	for f in 0 ..< SEQ_LEN {
		cb0[f] = d.cb0[offset + f]
		for c in 0 ..< LATENT_DIM {
			i_window := f * LATENT_DIM + c
			i_corpus := (offset + f) * LATENT_DIM + c
			noise := f32(rand.float32_normal(0, 1))
			data  := d.latents_norm[i_corpus]
			x_t     [i_window] = (1.0 - t) * noise + t * data
			velocity[i_window] = data - noise
		}
	}

	write_time_embedding(t, time_emb)
}

write_time_embedding :: proc(t: f32, out: []f32) {
	half := len(out) / 2
	for i in 0 ..< half {
		exponent := f32(i) / f32(half - 1)
		freq     := 2 * math.PI * math.pow(f32(10.0), exponent)
		angle    := t * freq
		out[i * 2]     = math.sin(angle)
		out[i * 2 + 1] = math.cos(angle)
	}
}

// --- Model (DiT-style with AdaLN-Zero) ------------------------------------

DiT_Block :: struct {
	ada_attn_w: ml.Tensor,    // [3 * HIDDEN, HIDDEN], zero-init
	ada_attn_b: ml.Tensor,
	q_w, q_b:   ml.Tensor,
	k_w, k_b:   ml.Tensor,
	v_w, v_b:   ml.Tensor,
	o_w, o_b:   ml.Tensor,
	ada_mlp_w:  ml.Tensor,    // [3 * HIDDEN, HIDDEN], zero-init
	ada_mlp_b:  ml.Tensor,
	ff1_w, ff1_b: ml.Tensor,
	ff2_w, ff2_b: ml.Tensor,
}

DiT_Model :: struct {
	in_weight:   ml.Tensor,    // [HIDDEN, LATENT_DIM]
	in_bias:     ml.Tensor,
	cb0_embed:   ml.Tensor,    // [COARSE_VOCAB, HIDDEN]
	pos_embed:   ml.Tensor,    // [MAX_SEQ_LEN, HIDDEN]
	norm_w:      ml.Tensor,    // [HIDDEN], init to 1
	time_w1:     ml.Tensor,
	time_b1:     ml.Tensor,
	time_w2:     ml.Tensor,
	time_b2:     ml.Tensor,
	blocks:      [N_LAYERS]DiT_Block,
	final_ada_w: ml.Tensor,    // [2 * HIDDEN, HIDDEN], zero-init
	final_ada_b: ml.Tensor,
	out_weight:  ml.Tensor,    // zero-init per DiT
	out_bias:    ml.Tensor,
}

dit_make :: proc() -> (m: DiT_Model) {
	m.in_weight   = ml.make(.F32, {HIDDEN_SIZE, LATENT_DIM})
	m.in_bias     = ml.make(.F32, {HIDDEN_SIZE})
	m.cb0_embed   = ml.make(.F32, {COARSE_VOCAB, HIDDEN_SIZE})
	m.pos_embed   = ml.make(.F32, {MAX_SEQ_LEN, HIDDEN_SIZE})
	m.norm_w      = ml.make(.F32, {HIDDEN_SIZE})
	m.time_w1     = ml.make(.F32, {HIDDEN_SIZE, TIME_EMBED_DIM})
	m.time_b1     = ml.make(.F32, {HIDDEN_SIZE})
	m.time_w2     = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
	m.time_b2     = ml.make(.F32, {HIDDEN_SIZE})
	m.final_ada_w = ml.make(.F32, {2 * HIDDEN_SIZE, HIDDEN_SIZE})
	m.final_ada_b = ml.make(.F32, {2 * HIDDEN_SIZE})
	m.out_weight  = ml.make(.F32, {LATENT_DIM, HIDDEN_SIZE})
	m.out_bias    = ml.make(.F32, {LATENT_DIM})
	for i in 0 ..< N_LAYERS {
		m.blocks[i].ada_attn_w = ml.make(.F32, {3 * HIDDEN_SIZE, HIDDEN_SIZE})
		m.blocks[i].ada_attn_b = ml.make(.F32, {3 * HIDDEN_SIZE})
		m.blocks[i].q_w        = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
		m.blocks[i].q_b        = ml.make(.F32, {HIDDEN_SIZE})
		m.blocks[i].k_w        = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
		m.blocks[i].k_b        = ml.make(.F32, {HIDDEN_SIZE})
		m.blocks[i].v_w        = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
		m.blocks[i].v_b        = ml.make(.F32, {HIDDEN_SIZE})
		m.blocks[i].o_w        = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
		m.blocks[i].o_b        = ml.make(.F32, {HIDDEN_SIZE})
		m.blocks[i].ada_mlp_w  = ml.make(.F32, {3 * HIDDEN_SIZE, HIDDEN_SIZE})
		m.blocks[i].ada_mlp_b  = ml.make(.F32, {3 * HIDDEN_SIZE})
		m.blocks[i].ff1_w      = ml.make(.F32, {FF_MULT * HIDDEN_SIZE, HIDDEN_SIZE})
		m.blocks[i].ff1_b      = ml.make(.F32, {FF_MULT * HIDDEN_SIZE})
		m.blocks[i].ff2_w      = ml.make(.F32, {HIDDEN_SIZE, FF_MULT * HIDDEN_SIZE})
		m.blocks[i].ff2_b      = ml.make(.F32, {HIDDEN_SIZE})
	}

	ml.he_initialization(m.in_weight,  LATENT_DIM)
	ml.fill_value       (m.in_bias,    0)
	ml.fill_normal      (m.cb0_embed,  0, 0.02)
	ml.fill_normal      (m.pos_embed,  0, 0.02)
	ml.fill_value       (m.norm_w,     1)
	ml.he_initialization(m.time_w1,    TIME_EMBED_DIM)
	ml.fill_value       (m.time_b1,    0)
	ml.he_initialization(m.time_w2,    HIDDEN_SIZE)
	ml.fill_value       (m.time_b2,    0)
	ml.fill_value       (m.final_ada_w, 0)
	ml.fill_value       (m.final_ada_b, 0)
	ml.fill_value       (m.out_weight,  0)
	ml.fill_value       (m.out_bias,    0)
	for i in 0 ..< N_LAYERS {
		ml.fill_value       (m.blocks[i].ada_attn_w, 0)
		ml.fill_value       (m.blocks[i].ada_attn_b, 0)
		ml.he_initialization(m.blocks[i].q_w,        HIDDEN_SIZE)
		ml.fill_value       (m.blocks[i].q_b,        0)
		ml.he_initialization(m.blocks[i].k_w,        HIDDEN_SIZE)
		ml.fill_value       (m.blocks[i].k_b,        0)
		ml.he_initialization(m.blocks[i].v_w,        HIDDEN_SIZE)
		ml.fill_value       (m.blocks[i].v_b,        0)
		ml.he_initialization(m.blocks[i].o_w,        HIDDEN_SIZE)
		ml.fill_value       (m.blocks[i].o_b,        0)
		ml.fill_value       (m.blocks[i].ada_mlp_w,  0)
		ml.fill_value       (m.blocks[i].ada_mlp_b,  0)
		ml.he_initialization(m.blocks[i].ff1_w,      HIDDEN_SIZE)
		ml.fill_value       (m.blocks[i].ff1_b,      0)
		ml.he_initialization(m.blocks[i].ff2_w,      FF_MULT * HIDDEN_SIZE)
		ml.fill_value       (m.blocks[i].ff2_b,      0)
	}
	return
}

dit_destroy :: proc(m: DiT_Model) {
	ml.destroy(m.in_weight);   ml.destroy(m.in_bias)
	ml.destroy(m.cb0_embed)
	ml.destroy(m.pos_embed)
	ml.destroy(m.norm_w)
	ml.destroy(m.time_w1);     ml.destroy(m.time_b1)
	ml.destroy(m.time_w2);     ml.destroy(m.time_b2)
	ml.destroy(m.final_ada_w); ml.destroy(m.final_ada_b)
	ml.destroy(m.out_weight);  ml.destroy(m.out_bias)
	for i in 0 ..< N_LAYERS {
		ml.destroy(m.blocks[i].ada_attn_w); ml.destroy(m.blocks[i].ada_attn_b)
		ml.destroy(m.blocks[i].q_w);        ml.destroy(m.blocks[i].q_b)
		ml.destroy(m.blocks[i].k_w);        ml.destroy(m.blocks[i].k_b)
		ml.destroy(m.blocks[i].v_w);        ml.destroy(m.blocks[i].v_b)
		ml.destroy(m.blocks[i].o_w);        ml.destroy(m.blocks[i].o_b)
		ml.destroy(m.blocks[i].ada_mlp_w);  ml.destroy(m.blocks[i].ada_mlp_b)
		ml.destroy(m.blocks[i].ff1_w);      ml.destroy(m.blocks[i].ff1_b)
		ml.destroy(m.blocks[i].ff2_w);      ml.destroy(m.blocks[i].ff2_b)
	}
}

PER_BLOCK_TENSORS :: 16

dit_params :: proc(m: DiT_Model) -> []ml.Tensor {
	count := 13 + N_LAYERS * PER_BLOCK_TENSORS
	out   := make([]ml.Tensor, count, context.temp_allocator)
	out[0]  = m.in_weight
	out[1]  = m.in_bias
	out[2]  = m.cb0_embed
	out[3]  = m.pos_embed
	out[4]  = m.norm_w
	out[5]  = m.time_w1
	out[6]  = m.time_b1
	out[7]  = m.time_w2
	out[8]  = m.time_b2
	out[9]  = m.final_ada_w
	out[10] = m.final_ada_b
	out[11] = m.out_weight
	out[12] = m.out_bias
	for i in 0 ..< N_LAYERS {
		base := 13 + i * PER_BLOCK_TENSORS
		out[base + 0]  = m.blocks[i].ada_attn_w
		out[base + 1]  = m.blocks[i].ada_attn_b
		out[base + 2]  = m.blocks[i].q_w
		out[base + 3]  = m.blocks[i].q_b
		out[base + 4]  = m.blocks[i].k_w
		out[base + 5]  = m.blocks[i].k_b
		out[base + 6]  = m.blocks[i].v_w
		out[base + 7]  = m.blocks[i].v_b
		out[base + 8]  = m.blocks[i].o_w
		out[base + 9]  = m.blocks[i].o_b
		out[base + 10] = m.blocks[i].ada_mlp_w
		out[base + 11] = m.blocks[i].ada_mlp_b
		out[base + 12] = m.blocks[i].ff1_w
		out[base + 13] = m.blocks[i].ff1_b
		out[base + 14] = m.blocks[i].ff2_w
		out[base + 15] = m.blocks[i].ff2_b
	}
	return out
}

dit_update :: proc(opt: ml.Optimizer, m: DiT_Model) {
	for p in dit_params(m) {
		ml.update(opt, p)
	}
}

count_parameters :: proc() -> int {
	per_block :=
		3 * HIDDEN_SIZE * HIDDEN_SIZE + 3 * HIDDEN_SIZE +    // ada_attn
		HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE +             // q
		HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE +             // k
		HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE +             // v
		HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE +             // o
		3 * HIDDEN_SIZE * HIDDEN_SIZE + 3 * HIDDEN_SIZE +    // ada_mlp
		FF_MULT * HIDDEN_SIZE * HIDDEN_SIZE + FF_MULT * HIDDEN_SIZE +    // ff1
		HIDDEN_SIZE * FF_MULT * HIDDEN_SIZE + HIDDEN_SIZE                 // ff2

	misc :=
		HIDDEN_SIZE * LATENT_DIM + HIDDEN_SIZE +              // in
		COARSE_VOCAB * HIDDEN_SIZE +                          // cb0_embed
		MAX_SEQ_LEN * HIDDEN_SIZE +                           // pos_embed
		HIDDEN_SIZE +                                         // norm_w
		HIDDEN_SIZE * TIME_EMBED_DIM + HIDDEN_SIZE +          // time1
		HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE +             // time2
		2 * HIDDEN_SIZE * HIDDEN_SIZE + 2 * HIDDEN_SIZE +    // final_ada
		LATENT_DIM * HIDDEN_SIZE + LATENT_DIM                 // out

	return misc + N_LAYERS * per_block
}

// LN(x) * (1 + scale) + shift, with scale/shift broadcast across the leading dim.
modulate :: proc(normed, shift, scale: ml.Tensor) -> ml.Tensor {
	one      := ml.scalar(.F32, 1.0)
	scale_p1 := ml.add(scale, one)
	scaled   := ml.mul(normed, scale_p1)
	return ml.add(scaled, shift)
}

dit_block_forward :: proc(b: DiT_Block, x: ml.Tensor, t_hidden, norm_w: ml.Tensor) -> ml.Tensor {
	c_silu := ml.silu(t_hidden)

	// Attention sub-block.
	ada_attn := ml.linear(c_silu, b.ada_attn_w)
	ada_attn  = ml.add(ada_attn, b.ada_attn_b)
	shift1 := ml.slice_trailing(ada_attn, 0,                  HIDDEN_SIZE)
	scale1 := ml.slice_trailing(ada_attn, HIDDEN_SIZE,       2 * HIDDEN_SIZE)
	gate1  := ml.slice_trailing(ada_attn, 2 * HIDDEN_SIZE,   3 * HIDDEN_SIZE)

	normed_a := ml.layernorm(x, norm_w)
	mod_a    := modulate(normed_a, shift1, scale1)

	q := ml.add(ml.linear(mod_a, b.q_w), b.q_b)
	k := ml.add(ml.linear(mod_a, b.k_w), b.k_b)
	v := ml.add(ml.linear(mod_a, b.v_w), b.v_b)

	attn_out := ml.attention(q, k, v, n_q_heads=N_HEADS, causal=false)
	attn_out  = ml.add(ml.linear(attn_out, b.o_w), b.o_b)
	x_a := ml.add(x, ml.mul(attn_out, gate1))

	// MLP sub-block.
	ada_mlp := ml.linear(c_silu, b.ada_mlp_w)
	ada_mlp  = ml.add(ada_mlp, b.ada_mlp_b)
	shift2 := ml.slice_trailing(ada_mlp, 0,                  HIDDEN_SIZE)
	scale2 := ml.slice_trailing(ada_mlp, HIDDEN_SIZE,       2 * HIDDEN_SIZE)
	gate2  := ml.slice_trailing(ada_mlp, 2 * HIDDEN_SIZE,   3 * HIDDEN_SIZE)

	normed_m := ml.layernorm(x_a, norm_w)
	mod_m    := modulate(normed_m, shift2, scale2)

	ff := ml.linear(mod_m, b.ff1_w)
	ff  = ml.add(ff, b.ff1_b)
	ff  = ml.silu(ff)
	ff  = ml.linear(ff, b.ff2_w)
	ff  = ml.add(ff, b.ff2_b)

	return ml.add(x_a, ml.mul(ff, gate2))
}

dit_forward :: proc(m: DiT_Model, latent_t: ml.Tensor, cb0_codes: []int, time_emb: ml.Tensor) -> ml.Tensor {
	assert(builtin.len(cb0_codes) == SEQ_LEN, "dit_forward: cb0 length must equal SEQ_LEN")

	t_hidden := ml.linear(time_emb, m.time_w1)
	t_hidden  = ml.add(t_hidden, m.time_b1)
	t_hidden  = ml.silu(t_hidden)
	t_hidden  = ml.linear(t_hidden, m.time_w2)
	t_hidden  = ml.add(t_hidden, m.time_b2)

	pos_indices := make([]int, SEQ_LEN, context.temp_allocator)
	for i in 0 ..< SEQ_LEN do pos_indices[i] = i
	pos_emb := ml.select(m.pos_embed, pos_indices)
	cb0_emb := ml.select(m.cb0_embed, cb0_codes)

	x := ml.linear(latent_t, m.in_weight)
	x  = ml.add(x, m.in_bias)
	x  = ml.add(x, cb0_emb)
	x  = ml.add(x, pos_emb)

	for i in 0 ..< N_LAYERS {
		x = dit_block_forward(m.blocks[i], x, t_hidden, m.norm_w)
	}

	c_final   := ml.silu(t_hidden)
	ada_final := ml.linear(c_final, m.final_ada_w)
	ada_final  = ml.add(ada_final, m.final_ada_b)
	shift_f := ml.slice_trailing(ada_final, 0,           HIDDEN_SIZE)
	scale_f := ml.slice_trailing(ada_final, HIDDEN_SIZE, 2 * HIDDEN_SIZE)

	normed_final := ml.layernorm(x, m.norm_w)
	mod_final    := modulate(normed_final, shift_f, scale_f)

	out := ml.linear(mod_final, m.out_weight)
	out  = ml.add(out, m.out_bias)
	return out
}

// --- EMA ------------------------------------------------------------------

Ema :: struct {
	decay:  f32,
	shadow: []ml.Tensor,
	backup: [][]f32,
}

ema_make :: proc(params: []ml.Tensor, decay: f32) -> (ema: Ema) {
	ema.decay  = decay
	ema.shadow = make([]ml.Tensor, builtin.len(params))
	ema.backup = make([][]f32,     builtin.len(params))
	for p, i in params {
		shape := p.shape
		ema.shadow[i] = ml.alloc(.F32, shape[:p.rank], persistent=true, buffers={.Data})
		ml.lerp_assign(ema.shadow[i], p, 1.0)
		ema.backup[i] = make([]f32, ml.len(p))
	}
	return
}

ema_destroy :: proc(ema: Ema) {
	for s in ema.shadow do ml.destroy(s)
	for b in ema.backup do delete(b)
	delete(ema.shadow)
	delete(ema.backup)
}

ema_update :: proc(ema: ^Ema, params: []ml.Tensor) {
	alpha := 1 - ema.decay
	for p, i in params {
		ml.lerp_assign(ema.shadow[i], p, alpha)
	}
}

ema_swap_in :: proc(ema: ^Ema, params: []ml.Tensor) {
	for p, i in params {
		ml.get_data(p, ema.backup[i])
		host_shadow := make([]f32, ml.len(p), context.temp_allocator)
		ml.get_data(ema.shadow[i], host_shadow)
		ml.set_data(p, host_shadow)
	}
}

ema_swap_out :: proc(ema: ^Ema, params: []ml.Tensor) {
	for p, i in params {
		ml.set_data(p, ema.backup[i])
	}
}

// --- Sampling -------------------------------------------------------------

// Generate SEQ_LEN frames of latents conditioned on a random cb0 window
// from `valid`, then write a .bin file in the dataset format so
// audio_diffusion_decode.py can render it to wav.
emit_sample :: proc(m: DiT_Model, valid: Dataset, out_path: string) {
	max_offset := valid.num_frames - SEQ_LEN
	offset     := 0
	if max_offset > 0 {
		offset = rand.int_max(max_offset)
	}

	cb0_window := make([]int, SEQ_LEN, context.temp_allocator)
	for i in 0 ..< SEQ_LEN {
		cb0_window[i] = valid.cb0[offset + i]
	}

	x := make([]f32, SEQ_LEN * LATENT_DIM, context.temp_allocator)
	for i in 0 ..< builtin.len(x) {
		x[i] = f32(rand.float32_normal(0, 1))
	}

	time_emb := make([]f32, TIME_EMBED_DIM, context.temp_allocator)

	dt := f32(1.0) / f32(SAMPLE_STEPS)
	for step_index in 0 ..< SAMPLE_STEPS {
		t := f32(step_index) * dt
		write_time_embedding(t, time_emb)

		ml.clear({.No_Gradients})
		x_tensor    := ml.reshape(ml.tensor(x),        {SEQ_LEN, LATENT_DIM})
		time_tensor := ml.tensor(time_emb)
		velocity    := dit_forward(m, x_tensor, cb0_window, time_tensor)

		velocity_buf := make([]f32, ml.len(velocity), context.temp_allocator)
		ml.get_data(velocity, velocity_buf)
		for i in 0 ..< builtin.len(x) {
			x[i] += dt * velocity_buf[i]
		}
	}

	// De-normalize: x is in normalized space, write raw latents.
	raw_latents := make([]f32, SEQ_LEN * LATENT_DIM, context.temp_allocator)
	for i in 0 ..< builtin.len(raw_latents) {
		raw_latents[i] = x[i] * valid.latent_std + valid.latent_mean
	}

	write_dataset_bin(out_path, raw_latents, cb0_window, valid.latent_mean, valid.latent_std,
		valid.codebook_vocab, valid.sample_rate, valid.frame_rate)
}

write_dataset_bin :: proc(path: string, latents: []f32, cb0: []int, mean, std: f32, vocab, sr, fr: int) {
	num_frames := SEQ_LEN
	header_bytes := DATASET_HEADER_BYTES
	latent_bytes := num_frames * LATENT_DIM * 4
	cb0_bytes    := num_frames * 4
	total_bytes  := header_bytes + latent_bytes + cb0_bytes

	buf := make([]u8, total_bytes)
	defer delete(buf)

	(^u32le)(&buf[0])^   = u32le(DATASET_MAGIC)
	(^u32le)(&buf[4])^   = u32le(1)                  // version
	(^u32le)(&buf[8])^   = u32le(num_frames)
	(^u32le)(&buf[12])^  = u32le(LATENT_DIM)
	(^u32le)(&buf[16])^  = u32le(vocab)
	(^u32le)(&buf[20])^  = u32le(sr)
	(^u32le)(&buf[24])^  = u32le(fr)
	(^u32le)(&buf[28])^  = u32le(0)                  // reserved
	(^f32le)(&buf[32])^  = f32le(mean)
	(^f32le)(&buf[36])^  = f32le(std)

	latent_offset := header_bytes
	for i in 0 ..< builtin.len(latents) {
		(^f32le)(&buf[latent_offset + i * 4])^ = f32le(latents[i])
	}
	cb0_offset := latent_offset + latent_bytes
	for i in 0 ..< builtin.len(cb0) {
		(^i32le)(&buf[cb0_offset + i * 4])^ = i32le(i32(cb0[i]))
	}

	if err := os.write_entire_file(path, buf); err != nil {
		fmt.eprintfln("FAIL: could not write %v: %v", path, err)
	}
}