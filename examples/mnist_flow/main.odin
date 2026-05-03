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
import gpu "../../backends/gpu"

DATA_DIR    :: "examples/data"
TRAIN_CSV   :: DATA_DIR + "/mnist_train.csv"
SAMPLE_DIR  :: DATA_DIR + "/mnist_flow_samples"

IMAGE_SIZE     :: 784 // 28 * 28
TIME_EMBED_DIM :: 32
HIDDEN_SIZE    :: 1024
HIDDEN_LAYERS  :: 3

BATCH_SIZE   :: 256
TOTAL_STEPS  :: 40000
LOG_EVERY    :: 100
SAMPLE_EVERY :: 1000
SAMPLE_GRID  :: 8
SAMPLE_STEPS :: 50

LEARNING_RATE :: 2e-4
WEIGHT_DECAY  :: 0
EMA_DECAY     :: f32(0.999)
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

// --- Model ----------------------------------------------------------------

Flow_Model :: struct {
	in_weight:   ml.Tensor,
	in_bias:     ml.Tensor,
	hidden_w:    [HIDDEN_LAYERS]ml.Tensor,
	hidden_b:    [HIDDEN_LAYERS]ml.Tensor,
	out_weight:  ml.Tensor,
	out_bias:    ml.Tensor,
	time_w1:     ml.Tensor,
	time_b1:     ml.Tensor,
	time_w2:     ml.Tensor,
	time_b2:     ml.Tensor,
}

flow_model_make :: proc() -> (model: Flow_Model) {
	model.in_weight  = ml.make(.F32, {HIDDEN_SIZE, IMAGE_SIZE})
	model.in_bias    = ml.make(.F32, {HIDDEN_SIZE})
	model.out_weight = ml.make(.F32, {IMAGE_SIZE, HIDDEN_SIZE})
	model.out_bias   = ml.make(.F32, {IMAGE_SIZE})
	model.time_w1    = ml.make(.F32, {HIDDEN_SIZE, TIME_EMBED_DIM})
	model.time_b1    = ml.make(.F32, {HIDDEN_SIZE})
	model.time_w2    = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
	model.time_b2    = ml.make(.F32, {HIDDEN_SIZE})
	for i in 0 ..< HIDDEN_LAYERS {
		model.hidden_w[i] = ml.make(.F32, {HIDDEN_SIZE, HIDDEN_SIZE})
		model.hidden_b[i] = ml.make(.F32, {HIDDEN_SIZE})
	}

	ml.he_initialization(model.in_weight,  IMAGE_SIZE)
	ml.he_initialization(model.out_weight, HIDDEN_SIZE)
	ml.he_initialization(model.time_w1,    TIME_EMBED_DIM)
	ml.he_initialization(model.time_w2,    HIDDEN_SIZE)
	ml.fill_value(model.in_bias,   0)
	ml.fill_value(model.out_bias,  0)
	ml.fill_value(model.time_b1,   0)
	ml.fill_value(model.time_b2,   0)
	for i in 0 ..< HIDDEN_LAYERS {
		ml.he_initialization(model.hidden_w[i], HIDDEN_SIZE)
		ml.fill_value(model.hidden_b[i], 0)
	}
	return
}

flow_model_destroy :: proc(model: Flow_Model) {
	ml.destroy(model.in_weight);  ml.destroy(model.in_bias)
	ml.destroy(model.out_weight); ml.destroy(model.out_bias)
	ml.destroy(model.time_w1);    ml.destroy(model.time_b1)
	ml.destroy(model.time_w2);    ml.destroy(model.time_b2)
	for i in 0 ..< HIDDEN_LAYERS {
		ml.destroy(model.hidden_w[i])
		ml.destroy(model.hidden_b[i])
	}
}

flow_model_params :: proc(model: Flow_Model) -> []ml.Tensor {
	out := make([]ml.Tensor, 8 + HIDDEN_LAYERS * 2, context.temp_allocator)
	out[0] = model.in_weight
	out[1] = model.in_bias
	out[2] = model.out_weight
	out[3] = model.out_bias
	out[4] = model.time_w1
	out[5] = model.time_b1
	out[6] = model.time_w2
	out[7] = model.time_b2
	for i in 0 ..< HIDDEN_LAYERS {
		out[8 + i * 2]     = model.hidden_w[i]
		out[8 + i * 2 + 1] = model.hidden_b[i]
	}
	return out
}

flow_model_update :: proc(opt: ml.Optimizer, model: Flow_Model) {
	for p in flow_model_params(model) {
		ml.update(opt, p)
	}
}

flow_model_forward :: proc(model: Flow_Model, image, time_emb: ml.Tensor) -> ml.Tensor {
	t_hidden := ml.linear(time_emb, model.time_w1)
	t_hidden  = ml.add(t_hidden, model.time_b1)
	t_hidden  = ml.silu(t_hidden)
	t_hidden  = ml.linear(t_hidden, model.time_w2)
	t_hidden  = ml.add(t_hidden, model.time_b2)
	t_hidden  = ml.silu(t_hidden)

	h := ml.linear(image, model.in_weight)
	h  = ml.add(h, model.in_bias)
	h  = ml.add(h, t_hidden)
	h  = ml.silu(h)

	for i in 0 ..< HIDDEN_LAYERS {
		residual := h
		h = ml.linear(h, model.hidden_w[i])
		h = ml.add(h, model.hidden_b[i])
		h = ml.add(h, t_hidden)
		h = ml.silu(h)
		h = ml.add(h, residual)
	}

	h = ml.linear(h, model.out_weight)
	h = ml.add(h, model.out_bias)
	return h
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
				if value < 0 do value = 0
				if value > 1 do value = 1
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
