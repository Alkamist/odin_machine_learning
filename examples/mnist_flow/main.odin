// Train a small MLP as a conditional-OT flow-matching model on MNIST.
// The model learns a velocity field v_theta(x_t, t) such that integrating
// dx/dt = v(x, t) from t=0 (Gaussian noise) to t=1 (data) produces a digit.
//
//   odin run examples/mnist_flow -o:speed
//
// Periodically writes 8x8 sample grids to examples/data/mnist_flow_samples
// as plain PGM files (open with any image viewer / IrfanView / GIMP).
//
// Conditioning: the timestep is sin/cos-embedded, projected through a
// 2-layer time MLP to a hidden-dim vector, and added to every hidden
// activation. EMA of the parameters is used for sampling.

package main

import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:strconv"
import "core:encoding/csv"

import ml  "../../"
import gpu "../../backends/vulkan"

DATA_DIR    :: "examples/data"
TRAIN_CSV   :: DATA_DIR + "/mnist_train.csv"
SAMPLE_DIR  :: DATA_DIR + "/mnist_flow_samples"

IMAGE_SIZE     :: 784 // 28 * 28
TIME_EMBED_DIM :: 32
HIDDEN_SIZE    :: 512
HIDDEN_LAYERS  :: 4
FF_MULT        :: 4   // MLP expansion factor inside each AdaLN-Zero block

BATCH_SIZE   :: 256
TOTAL_STEPS  :: 40000
LOG_EVERY    :: 100
SAMPLE_EVERY :: 1000
SAMPLE_GRID  :: 8
SAMPLE_STEPS :: 50

LEARNING_RATE :: 2e-4
WEIGHT_DECAY  :: 0
EMA_DECAY     :: 0.999
SEED          :: 0xF10D1

main :: proc() {
	defer fmt.println("Finished")

	rand.reset(SEED)

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)

	ml.context_scope(ctx)

	if !os.exists(SAMPLE_DIR) {
		os.make_directory(SAMPLE_DIR)
	}

	fmt.println("Loading MNIST ...")
	train_set := mnist_load(TRAIN_CSV, 60000)
	defer mnist_destroy(train_set)
	fmt.printfln("  %v training images", train_set.samples)

	model := flow_model_make()
	defer flow_model_destroy(model)

	ema := ema_make(flow_model_params(model), EMA_DECAY)
	defer ema_destroy(ema)

	loss_acc := ml.alloc(.F32, []int{1}, persistent=true, buffers={.Data})
	defer ml.destroy(loss_acc)
	ml.fill_value(loss_acc, 0)

	opt: ml.Optimizer

	batch_x_t      := make([]f32, BATCH_SIZE * IMAGE_SIZE)
	batch_velocity := make([]f32, BATCH_SIZE * IMAGE_SIZE)
	batch_time_emb := make([]f32, BATCH_SIZE * TIME_EMBED_DIM)
	defer delete(batch_x_t)
	defer delete(batch_velocity)
	defer delete(batch_time_emb)

	for step in 1 ..= TOTAL_STEPS {
		defer free_all(context.temp_allocator)

		build_training_batch(train_set, batch_x_t, batch_velocity, batch_time_emb)

		ml.clear()

		image_tensor    := ml.reshape(ml.tensor(batch_x_t),      {BATCH_SIZE, IMAGE_SIZE})
		time_tensor     := ml.reshape(ml.tensor(batch_time_emb), {BATCH_SIZE, TIME_EMBED_DIM})
		target_tensor   := ml.reshape(ml.tensor(batch_velocity), {BATCH_SIZE, IMAGE_SIZE})
		predicted_field := flow_model_forward(model, image_tensor, time_tensor)
		loss            := ml.mean_squared_error(predicted_field, target_tensor)

		ml.backward()

		ml.accumulate_mean(loss_acc, loss)

		if ml.optimize(&opt, period=1, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY) {
			flow_model_update(opt, model)
			ema_update(&ema, flow_model_params(model))
		}

		if step % LOG_EVERY == 0 {
			loss_host: [1]f32
			ml.get_data(loss_acc, loss_host[:])
			fmt.printfln("step %5v  loss = %.5f", step, loss_host[0] / f32(LOG_EVERY))
			ml.fill_value(loss_acc, 0)
		}

		if step % SAMPLE_EVERY == 0 {
			sample_path := fmt.tprintf("%v/sample_step_%05v.pgm", SAMPLE_DIR, step)
			ema_swap_in(&ema, flow_model_params(model))
			write_sample_grid(model, SAMPLE_GRID, SAMPLE_STEPS, sample_path)
			ema_swap_out(&ema, flow_model_params(model))
			fmt.printfln("           wrote %v", sample_path)
		}
	}
}

// --- Model (DiT-style with AdaLN-Zero blocks) -----------------------------

// Each block produces (shift, scale, gate) from the conditioning vector via
// a zero-initialized linear, so at init scale=0, shift=0, gate=0 and the
// block is the identity. The output projection is also zero-initialized,
// so the network predicts zero velocity at init and learns to deviate.

Flow_Block :: struct {
	ada_w: ml.Tensor,    // [3 * HIDDEN_SIZE, HIDDEN_SIZE], zero-init
	ada_b: ml.Tensor,    // [3 * HIDDEN_SIZE], zero-init
	ff1_w: ml.Tensor,    // [FF_MULT * HIDDEN_SIZE, HIDDEN_SIZE]
	ff1_b: ml.Tensor,    // [FF_MULT * HIDDEN_SIZE]
	ff2_w: ml.Tensor,    // [HIDDEN_SIZE, FF_MULT * HIDDEN_SIZE]
	ff2_b: ml.Tensor,    // [HIDDEN_SIZE]
}

Flow_Model :: struct {
	in_weight:     ml.Tensor,
	in_bias:       ml.Tensor,
	norm_w:        ml.Tensor,    // shared LayerNorm gain, init to 1
	blocks:        [HIDDEN_LAYERS]Flow_Block,
	final_ada_w:   ml.Tensor,    // [2 * HIDDEN_SIZE, HIDDEN_SIZE], zero-init
	final_ada_b:   ml.Tensor,    // [2 * HIDDEN_SIZE], zero-init
	out_weight:    ml.Tensor,    // zero-init per DiT
	out_bias:      ml.Tensor,
	time_w1:       ml.Tensor,
	time_b1:       ml.Tensor,
	time_w2:       ml.Tensor,
	time_b2:       ml.Tensor,
}

flow_model_make :: proc() -> (model: Flow_Model) {
	model.in_weight   = ml.make(.F32, {HIDDEN_SIZE, IMAGE_SIZE})
	model.in_bias     = ml.make(.F32, {HIDDEN_SIZE})
	model.norm_w      = ml.make(.F32, {HIDDEN_SIZE})
	model.final_ada_w = ml.make(.F32, {2 * HIDDEN_SIZE, HIDDEN_SIZE})
	model.final_ada_b = ml.make(.F32, {2 * HIDDEN_SIZE})
	model.out_weight  = ml.make(.F32, {IMAGE_SIZE, HIDDEN_SIZE})
	model.out_bias    = ml.make(.F32, {IMAGE_SIZE})
	model.time_w1     = ml.make(.F32, {HIDDEN_SIZE, TIME_EMBED_DIM})
	model.time_b1     = ml.make(.F32, {HIDDEN_SIZE})
	model.time_w2     = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
	model.time_b2     = ml.make(.F32, {HIDDEN_SIZE})
	for i in 0 ..< HIDDEN_LAYERS {
		model.blocks[i].ada_w = ml.make(.F32, {3 * HIDDEN_SIZE,       HIDDEN_SIZE})
		model.blocks[i].ada_b = ml.make(.F32, {3 * HIDDEN_SIZE})
		model.blocks[i].ff1_w = ml.make(.F32, {FF_MULT * HIDDEN_SIZE, HIDDEN_SIZE})
		model.blocks[i].ff1_b = ml.make(.F32, {FF_MULT * HIDDEN_SIZE})
		model.blocks[i].ff2_w = ml.make(.F32, {HIDDEN_SIZE,           FF_MULT * HIDDEN_SIZE})
		model.blocks[i].ff2_b = ml.make(.F32, {HIDDEN_SIZE})
	}

	ml.he_initialization(model.in_weight, IMAGE_SIZE)
	ml.fill_value       (model.in_bias,   0)
	ml.fill_value       (model.norm_w,    1)
	ml.fill_value       (model.final_ada_w, 0)
	ml.fill_value       (model.final_ada_b, 0)
	ml.fill_value       (model.out_weight, 0)
	ml.fill_value       (model.out_bias,   0)
	ml.he_initialization(model.time_w1,   TIME_EMBED_DIM)
	ml.fill_value       (model.time_b1,   0)
	ml.he_initialization(model.time_w2,   HIDDEN_SIZE)
	ml.fill_value       (model.time_b2,   0)
	for i in 0 ..< HIDDEN_LAYERS {
		ml.fill_value       (model.blocks[i].ada_w, 0)
		ml.fill_value       (model.blocks[i].ada_b, 0)
		ml.he_initialization(model.blocks[i].ff1_w, HIDDEN_SIZE)
		ml.fill_value       (model.blocks[i].ff1_b, 0)
		ml.he_initialization(model.blocks[i].ff2_w, FF_MULT * HIDDEN_SIZE)
		ml.fill_value       (model.blocks[i].ff2_b, 0)
	}
	return
}

flow_model_destroy :: proc(model: Flow_Model) {
	ml.destroy(model.in_weight);   ml.destroy(model.in_bias)
	ml.destroy(model.norm_w)
	ml.destroy(model.final_ada_w); ml.destroy(model.final_ada_b)
	ml.destroy(model.out_weight);  ml.destroy(model.out_bias)
	ml.destroy(model.time_w1);     ml.destroy(model.time_b1)
	ml.destroy(model.time_w2);     ml.destroy(model.time_b2)
	for i in 0 ..< HIDDEN_LAYERS {
		ml.destroy(model.blocks[i].ada_w); ml.destroy(model.blocks[i].ada_b)
		ml.destroy(model.blocks[i].ff1_w); ml.destroy(model.blocks[i].ff1_b)
		ml.destroy(model.blocks[i].ff2_w); ml.destroy(model.blocks[i].ff2_b)
	}
}

flow_model_params :: proc(model: Flow_Model) -> []ml.Tensor {
	per_block :: 6
	count := 11 + HIDDEN_LAYERS * per_block
	out   := make([]ml.Tensor, count, context.temp_allocator)
	out[0]  = model.in_weight
	out[1]  = model.in_bias
	out[2]  = model.norm_w
	out[3]  = model.final_ada_w
	out[4]  = model.final_ada_b
	out[5]  = model.out_weight
	out[6]  = model.out_bias
	out[7]  = model.time_w1
	out[8]  = model.time_b1
	out[9]  = model.time_w2
	out[10] = model.time_b2
	for i in 0 ..< HIDDEN_LAYERS {
		base := 11 + i * per_block
		out[base + 0] = model.blocks[i].ada_w
		out[base + 1] = model.blocks[i].ada_b
		out[base + 2] = model.blocks[i].ff1_w
		out[base + 3] = model.blocks[i].ff1_b
		out[base + 4] = model.blocks[i].ff2_w
		out[base + 5] = model.blocks[i].ff2_b
	}
	return out
}

flow_model_update :: proc(opt: ml.Optimizer, model: Flow_Model) {
	for p in flow_model_params(model) {
		ml.update(opt, p)
	}
}

// modulate(x, shift, scale) = LN(x) * (1 + scale) + shift
modulate :: proc(normed, shift, scale: ml.Tensor) -> ml.Tensor {
	one      := ml.scalar(.F32, 1.0)
	scale_p1 := ml.add(scale, one)
	scaled   := ml.mul(normed, scale_p1)
	return ml.add(scaled, shift)
}

flow_model_forward :: proc(model: Flow_Model, image, time_emb: ml.Tensor) -> ml.Tensor {
	t_hidden := ml.linear(time_emb, model.time_w1)
	t_hidden  = ml.add(t_hidden, model.time_b1)
	t_hidden  = ml.silu(t_hidden)
	t_hidden  = ml.linear(t_hidden, model.time_w2)
	t_hidden  = ml.add(t_hidden, model.time_b2)

	h := ml.linear(image, model.in_weight)
	h  = ml.add(h, model.in_bias)

	for i in 0 ..< HIDDEN_LAYERS {
		c   := ml.silu(t_hidden)
		ada := ml.linear(c, model.blocks[i].ada_w)
		ada  = ml.add(ada, model.blocks[i].ada_b)

		shift := ml.slice_trailing(ada, 0,               HIDDEN_SIZE)
		scale := ml.slice_trailing(ada, HIDDEN_SIZE,     2 * HIDDEN_SIZE)
		gate  := ml.slice_trailing(ada, 2 * HIDDEN_SIZE, 3 * HIDDEN_SIZE)

		normed   := ml.layernorm(h, model.norm_w)
		modulated := modulate(normed, shift, scale)

		ff := ml.linear(modulated, model.blocks[i].ff1_w)
		ff  = ml.add(ff, model.blocks[i].ff1_b)
		ff  = ml.silu(ff)
		ff  = ml.linear(ff, model.blocks[i].ff2_w)
		ff  = ml.add(ff, model.blocks[i].ff2_b)

		gated := ml.mul(ff, gate)
		h      = ml.add(h, gated)
	}

	c_final   := ml.silu(t_hidden)
	ada_final := ml.linear(c_final, model.final_ada_w)
	ada_final  = ml.add(ada_final, model.final_ada_b)
	shift_f := ml.slice_trailing(ada_final, 0,               HIDDEN_SIZE)
	scale_f := ml.slice_trailing(ada_final, HIDDEN_SIZE, 2 * HIDDEN_SIZE)

	normed_final   := ml.layernorm(h, model.norm_w)
	modulated_final := modulate(normed_final, shift_f, scale_f)

	out := ml.linear(modulated_final, model.out_weight)
	out  = ml.add(out, model.out_bias)
	return out
}

// --- EMA ------------------------------------------------------------------

// Per-step blending happens on-device via ml.lerp_assign so training never
// has to round-trip the weights to host. The host-side `backup` buffer is
// only touched at sampling time, when we briefly install EMA weights into
// the live tensors.
Ema :: struct {
	decay:  f32,
	shadow: []ml.Tensor,
	backup: [][]f32,
}

ema_make :: proc(params: []ml.Tensor, decay: f32) -> (ema: Ema) {
	ema.decay  = decay
	ema.shadow = make([]ml.Tensor, len(params))
	ema.backup = make([][]f32,     len(params))
	for p, i in params {
		shape := p.shape
		ema.shadow[i] = ml.alloc(.F32, shape[:p.rank], persistent=true, buffers={.Data})
		ml.lerp_assign(ema.shadow[i], p, 1.0)   // shadow <- params
		ema.backup[i] = make([]f32, ml.len(p))
	}
	return
}

ema_destroy :: proc(ema: Ema) {
	for s in ema.shadow {
		ml.destroy(s)
	}
	for b in ema.backup {
		delete(b)
	}
	delete(ema.shadow)
	delete(ema.backup)
}

ema_update :: proc(ema: ^Ema, params: []ml.Tensor) {
	alpha := 1 - ema.decay
	for p, i in params {
		ml.lerp_assign(ema.shadow[i], p, alpha)
	}
}

// Stash live weights, install EMA into the model so sampling sees the
// averaged weights. Pair every call with ema_swap_out before training resumes.
ema_swap_in :: proc(ema: ^Ema, params: []ml.Tensor) {
	for p, i in params {
		ml.get_data(p,             ema.backup[i])
		host_shadow := make([]f32, ml.len(p), context.temp_allocator)
		ml.get_data(ema.shadow[i], host_shadow)
		ml.set_data(p,             host_shadow)
	}
}

ema_swap_out :: proc(ema: ^Ema, params: []ml.Tensor) {
	for p, i in params {
		ml.set_data(p, ema.backup[i])
	}
}

// --- Training-batch construction -----------------------------------------

build_training_batch :: proc(train: Mnist, x_t, velocity, time_emb: []f32) {
	for example_index in 0 ..< BATCH_SIZE {
		image_index := rand.int_max(train.samples)
		image       := train.inputs[image_index * IMAGE_SIZE:][:IMAGE_SIZE]

		t := rand.float32()

		image_offset := example_index * IMAGE_SIZE
		for pixel_index in 0 ..< IMAGE_SIZE {
			noise := f32(rand.float32_normal(0, 1))
			data  := image[pixel_index] * 2.0 - 1.0
			x_t     [image_offset + pixel_index] = (1.0 - t) * noise + t * data
			velocity[image_offset + pixel_index] = data - noise
		}

		write_time_embedding(t, time_emb[example_index * TIME_EMBED_DIM:][:TIME_EMBED_DIM])
	}
}

// Sinusoidal embedding of t in [0, 1]. Geometric frequencies up to ~2*pi*10
// rad: anything higher cycles many times across the unit interval and just
// looks like noise to the network.
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

// --- Sampling -------------------------------------------------------------

write_sample_grid :: proc(model: Flow_Model, grid, num_steps: int, out_path: string) {
	count := grid * grid

	x        := make([]f32, count * IMAGE_SIZE)
	time_emb := make([]f32, count * TIME_EMBED_DIM)
	defer delete(x)
	defer delete(time_emb)

	for i in 0 ..< len(x) {
		x[i] = f32(rand.float32_normal(0, 1))
	}

	dt := f32(1.0) / f32(num_steps)
	for step_index in 0 ..< num_steps {
		t := f32(step_index) * dt
		for example_index in 0 ..< count {
			write_time_embedding(t, time_emb[example_index * TIME_EMBED_DIM:][:TIME_EMBED_DIM])
		}

		ml.clear({.No_Gradients})
		image_tensor := ml.reshape(ml.tensor(x),        {count, IMAGE_SIZE})
		time_tensor  := ml.reshape(ml.tensor(time_emb), {count, TIME_EMBED_DIM})
		velocity     := flow_model_forward(model, image_tensor, time_tensor)

		velocity_buf := make([]f32, ml.len(velocity), context.temp_allocator)
		ml.get_data(velocity, velocity_buf)
		for i in 0 ..< len(x) {
			x[i] += dt * velocity_buf[i]
		}
	}

	write_pgm_grid(out_path, x, grid)
}

write_pgm_grid :: proc(path: string, samples: []f32, grid: int) {
	tile     := 28
	side     := grid * tile
	pixels   := make([]u8, side * side)
	defer delete(pixels)

	for sample_index in 0 ..< grid * grid {
		gx := sample_index % grid
		gy := sample_index / grid
		src := samples[sample_index * IMAGE_SIZE:][:IMAGE_SIZE]
		for row in 0 ..< tile {
			for col in 0 ..< tile {
				value := (src[row * tile + col] + 1.0) * 0.5
				if value < 0 {
					value = 0
				}
				if value > 1 {
					value = 1
				}
				dst_row := gy * tile + row
				dst_col := gx * tile + col
				pixels[dst_row * side + dst_col] = u8(value * 255.0)
			}
		}
	}

	header  := fmt.tprintf("P5\n%v %v\n255\n", side, side)
	payload := make([]u8, len(header) + len(pixels))
	defer delete(payload)
	copy(payload[:len(header)], transmute([]u8)header)
	copy(payload[len(header):], pixels)

	if err := os.write_entire_file(path, payload); err != nil {
		fmt.eprintfln("FAIL: could not write %v: %v", path, err)
	}
}

// --- Misc -----------------------------------------------------------------

Mnist :: struct {
	samples: int,
	inputs:  []f32,
}

mnist_load :: proc(path: string, samples: int, allocator := context.allocator) -> (mnist: Mnist) {
	bytes, err := os.read_entire_file(path, context.temp_allocator)
	if err != nil {
		fmt.eprintfln("FAIL: could not read %v", path)
		os.exit(1)
	}

	csv_reader: csv.Reader
	csv.reader_init_with_string(&csv_reader, cast(string)bytes, context.temp_allocator)
	defer csv.reader_destroy(&csv_reader)

	_, _ = csv.read(&csv_reader, context.temp_allocator)

	mnist.inputs = make([]f32, samples * IMAGE_SIZE, allocator)

	for i in 0 ..< samples {
		row, read_err := csv.read(&csv_reader, context.temp_allocator)
		if read_err != nil {
			break
		}
		for j in 0 ..< IMAGE_SIZE {
			value, _ := strconv.parse_i64(row[j + 1])
			mnist.inputs[i * IMAGE_SIZE + j] = f32(value) / 255.0
		}
	}

	mnist.samples = samples
	return
}

mnist_destroy :: proc(mnist: Mnist) {
	delete(mnist.inputs)
}
