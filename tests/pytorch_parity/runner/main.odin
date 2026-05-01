package pytorch_parity_runner

import "base:builtin"

import "core:fmt"
import "core:os"
import "core:mem"
import "core:strconv"
import "core:path/filepath"

import ml  "../../.."
import cpu "../../../backends/cpu"
import gpu "../../../backends/gpu"
import mlp   "../../../networks/mlp"
import llama "../../../networks/llama"

MAGIC :: "TNSR"

main :: proc() {
	if builtin.len(os.args) < 4 || builtin.len(os.args) > 5 {
		fmt.eprintln("usage: runner <backend:cpu|gpu> <test_name> <artifacts_dir> [thread_count]")
		os.exit(1)
	}
	backend_name  := os.args[1]
	test_name     := os.args[2]
	artifacts_dir := os.args[3]
	thread_count  := 1
	if builtin.len(os.args) == 5 {
		parsed, ok := strconv.parse_int(os.args[4])
		if ok {
			thread_count = parsed
		}
	}

	ctx: ^ml.Context
	switch backend_name {
	case "cpu":
		ctx = cpu.context_create(64 * 1024 * 1024)
	case "gpu":
		ctx = gpu.context_create()
	case:
		fmt.eprintfln("unknown backend: %v (expected cpu or gpu)", backend_name)
		os.exit(1)
	}
	defer {
		switch backend_name {
		case "cpu": cpu.context_destroy(ctx)
		case "gpu": gpu.context_destroy(ctx)
		}
	}
	ml.context_scope(ctx)

	if backend_name == "cpu" {
		cpu.set_thread_count(thread_count)
	}

	switch test_name {
	case "add_equal":      run_binary_op(artifacts_dir, .Add)
	case "add_broadcast":  run_binary_op(artifacts_dir, .Add)
	case "sub_broadcast":  run_binary_op(artifacts_dir, .Sub)
	case "mul_broadcast":  run_binary_op(artifacts_dir, .Mul)
	case "div_broadcast":  run_binary_op(artifacts_dir, .Div)
	case "linear_1d":      run_linear(artifacts_dir)
	case "linear_2d":      run_linear(artifacts_dir)
	case "linear_big":     run_linear(artifacts_dir)
	case "mean":           run_unary(artifacts_dir, .Mean)
	case "softmax":        run_unary(artifacts_dir, .Softmax)
	case "log_softmax":    run_unary(artifacts_dir, .Log_Softmax)
	case "layernorm":      run_layernorm(artifacts_dir)
	case "rmsnorm":        run_rmsnorm(artifacts_dir)
	case "rmsnorm_big":    run_rmsnorm(artifacts_dir)
	case "cross_entropy":  run_cross_entropy(artifacts_dir)
	case "batched_matmul": run_batched_matmul(artifacts_dir)
	case "permute":        run_permute(artifacts_dir)
	case "attention_causal":  run_attention(artifacts_dir)
	case "attention_acausal": run_attention(artifacts_dir)
	case "attention_xfmr":    run_attention(artifacts_dir)
	case "attention_gqa":     run_attention(artifacts_dir)
	case "attention_gqa_big": run_attention(artifacts_dir)
	case "attention_window":     run_attention(artifacts_dir)
	case "attention_window_big": run_attention(artifacts_dir)
	case "tied_embeddings":   run_tied_embeddings(artifacts_dir)
	case "mlp_train":         run_mlp_train(artifacts_dir)
	case "mlp_train_period12":run_mlp_train(artifacts_dir)
	case "select":            run_select(artifacts_dir)
	case "slice_trailing":    run_slice_trailing(artifacts_dir)
	case "concat3":           run_concat3(artifacts_dir)
	case "gelu":              run_unary(artifacts_dir, .Gelu)
	case "relu":              run_unary(artifacts_dir, .Relu)
	case "silu":              run_unary(artifacts_dir, .Silu)
	case "tanh":              run_unary(artifacts_dir, .Tanh)
	case "sigmoid":           run_unary(artifacts_dir, .Sigmoid)
	case "rope":              run_rope(artifacts_dir)
	case "rope_xfmr":         run_rope(artifacts_dir)
	case "transformer_train_bf16": run_transformer_train_bf16(artifacts_dir)
	case "llama_train":            run_llama_train(artifacts_dir)
	case "kv_cache_decode":        run_kv_cache_decode(artifacts_dir)
	case:
		fmt.eprintfln("unknown test: %v", test_name)
		os.exit(1)
	}
}

Binary_Op :: enum { Add, Sub, Mul, Div }

run_binary_op :: proc(dir: string, op: Binary_Op) {
	a_shape, a_data := load_tensor(_path(dir, "input_a.bin"))
	b_shape, b_data := load_tensor(_path(dir, "input_b.bin"))

	a := ml.alloc(.F32, a_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(a)
	b := ml.alloc(.F32, b_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(b)

	ml.set_data(a, a_data)
	ml.set_data(b, b_data)

	out: ml.Tensor
	switch op {
	case .Add: out = ml.add(a, b)
	case .Sub: out = ml.sub(a, b)
	case .Mul: out = ml.mul(a, b)
	case .Div: out = ml.div(a, b)
	}
	ml.backward()

	out_data := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	a_grad := builtin.make([]f32, ml.len(a), context.temp_allocator)
	b_grad := builtin.make([]f32, ml.len(b), context.temp_allocator)
	ml.get_gradient(a, a_grad)
	ml.get_gradient(b, b_grad)

	save_tensor(_path(dir, "odin_out.bin"),    a_shape, out_data)
	save_tensor(_path(dir, "odin_grad_a.bin"), a_shape, a_grad)
	save_tensor(_path(dir, "odin_grad_b.bin"), b_shape, b_grad)
}

run_linear :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	w_shape, w_data := load_tensor(_path(dir, "input_w.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	w := ml.alloc(.F32, w_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(w)

	ml.set_data(x, x_data)
	ml.set_data(w, w_data)

	out := ml.linear(x, w)
	ml.backward()

	out_shape := builtin.make([]int, out.rank, context.temp_allocator)
	for i in 0 ..< out.rank {
		out_shape[i] = out.shape[i]
	}

	out_data := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	w_grad := builtin.make([]f32, ml.len(w), context.temp_allocator)
	ml.get_gradient(x, x_grad)
	ml.get_gradient(w, w_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
	save_tensor(_path(dir, "odin_grad_w.bin"), w_shape,   w_grad)
}

Unary_Op :: enum { Mean, Softmax, Log_Softmax, Gelu, Relu, Silu, Tanh, Sigmoid }

run_unary :: proc(dir: string, op: Unary_Op) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	out: ml.Tensor
	switch op {
	case .Mean:        out = ml.mean       (x)
	case .Softmax:     out = ml.softmax    (x)
	case .Log_Softmax: out = ml.log_softmax(x)
	case .Gelu:        out = ml.gelu       (x)
	case .Relu:        out = ml.relu       (x)
	case .Silu:        out = ml.silu       (x)
	case .Tanh:        out = ml.tanh       (x)
	case .Sigmoid:     out = ml.sigmoid    (x)
	}
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	ml.get_gradient(x, x_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
}

run_layernorm :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	w_shape, w_data := load_tensor(_path(dir, "input_w.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	w := ml.alloc(.F32, w_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(w)

	ml.set_data(x, x_data)
	ml.set_data(w, w_data)

	out := ml.layernorm(x, w)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	w_grad := builtin.make([]f32, ml.len(w), context.temp_allocator)
	ml.get_gradient(x, x_grad)
	ml.get_gradient(w, w_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
	save_tensor(_path(dir, "odin_grad_w.bin"), w_shape,   w_grad)
}

run_rmsnorm :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	w_shape, w_data := load_tensor(_path(dir, "input_w.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	w := ml.alloc(.F32, w_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(w)

	ml.set_data(x, x_data)
	ml.set_data(w, w_data)

	out := ml.rmsnorm(x, w)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	w_grad := builtin.make([]f32, ml.len(w), context.temp_allocator)
	ml.get_gradient(x, x_grad)
	ml.get_gradient(w, w_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
	save_tensor(_path(dir, "odin_grad_w.bin"), w_shape,   w_grad)
}

run_cross_entropy :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	targets         := load_int_array(_path(dir, "targets.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	out := ml.cross_entropy(x, targets)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	ml.get_gradient(x, x_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
}

run_batched_matmul :: proc(dir: string) {
	a_shape, a_data := load_tensor(_path(dir, "input_a.bin"))
	b_shape, b_data := load_tensor(_path(dir, "input_b.bin"))

	a := ml.alloc(.F32, a_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(a)
	b := ml.alloc(.F32, b_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(b)

	ml.set_data(a, a_data)
	ml.set_data(b, b_data)

	out := ml.batched_matmul(a, b)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	a_grad := builtin.make([]f32, ml.len(a), context.temp_allocator)
	b_grad := builtin.make([]f32, ml.len(b), context.temp_allocator)
	ml.get_gradient(a, a_grad)
	ml.get_gradient(b, b_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_a.bin"), a_shape,   a_grad)
	save_tensor(_path(dir, "odin_grad_b.bin"), b_shape,   b_grad)
}

run_permute :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	axes_array      := load_int_array(_path(dir, "axes.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	axes := [3]int{axes_array[0], axes_array[1], axes_array[2]}
	out  := ml.permute(x, axes)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	ml.get_gradient(x, x_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
}

run_attention :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	config          := load_int_array(_path(dir, "config.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	n_q_heads  := config[0]
	causal     := config[1] != 0
	n_kv_heads := n_q_heads
	if builtin.len(config) >= 3 { n_kv_heads = config[2] }
	window := 0
	if builtin.len(config) >= 4 { window = config[3] }

	q_size  := x_shape[1] * n_q_heads  / (n_q_heads + 2 * n_kv_heads)
	kv_size := x_shape[1] * n_kv_heads / (n_q_heads + 2 * n_kv_heads)
	q := ml.slice_trailing(x, 0,                q_size)
	k := ml.slice_trailing(x, q_size,           q_size + kv_size)
	v := ml.slice_trailing(x, q_size + kv_size, q_size + 2 * kv_size)

	out := ml.attention(q, k, v, n_q_heads, n_kv_heads, causal=causal, window=window)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	ml.get_gradient(x, x_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
}

run_mlp_train :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	y_shape, y_data := load_tensor(_path(dir, "input_y.bin"))
	config          := load_int_array(_path(dir, "config.bin"))

	step_count  := config[0]
	period      := config[1]
	layer_count := builtin.len(config) - 2 - 1
	sizes := builtin.make([]int, layer_count + 1, context.temp_allocator)
	for i in 0 ..< layer_count + 1 {
		sizes[i] = config[2 + i]
	}

	model := mlp.make(..sizes)
	defer mlp.destroy(model)

	for layer_index in 0 ..< layer_count {
		w_path := fmt.tprintf("%v/init_w_%v.bin", dir, layer_index)
		b_path := fmt.tprintf("%v/init_b_%v.bin", dir, layer_index)
		_, w_data := load_tensor(w_path)
		_, b_data := load_tensor(b_path)
		ml.set_data(model.layers[layer_index].weight, w_data)
		ml.set_data(model.layers[layer_index].bias,   b_data)
	}

	x_target := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x_target)
	y_target := ml.alloc(.F32, y_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(y_target)
	ml.set_data(x_target, x_data)
	ml.set_data(y_target, y_data)

	losses := builtin.make([]f32, step_count, context.temp_allocator)

	opt: ml.Optimizer
	for step in 0 ..< step_count {
		ml.clear()
		predictions := mlp.forward(model, x_target)
		per_sample  := ml.mean_squared_error(predictions, y_target)
		loss        := ml.mean(per_sample)
		ml.backward()

		if ml.optimize(&opt, period=period, learning_rate=0.01) {
			mlp.update(opt, model)
		}

		loss_buf: [1]f32
		ml.get_data(loss, loss_buf[:])
		losses[step] = loss_buf[0]
	}

	losses_shape := []int{step_count}
	save_tensor(_path(dir, "odin_losses.bin"), losses_shape, losses)
}

run_select :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	indices         := load_int_array(_path(dir, "indices.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	out := ml.select(x, indices)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	ml.get_gradient(x, x_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
}

run_tied_embeddings :: proc(dir: string) {
	w_shape, w_data := load_tensor(_path(dir, "input_w.bin"))
	indices         := load_int_array(_path(dir, "indices.bin"))
	targets         := load_int_array(_path(dir, "targets.bin"))

	w := ml.alloc(.F32, w_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(w)
	ml.set_data(w, w_data)

	embeds    := ml.select(w, indices)
	logits    := ml.linear(embeds, w)
	per_token := ml.cross_entropy(logits, targets)
	loss      := ml.mean(per_token)
	ml.backward()

	loss_buf: [1]f32
	ml.get_data(loss, loss_buf[:])

	w_grad := builtin.make([]f32, ml.len(w), context.temp_allocator)
	ml.get_gradient(w, w_grad)

	save_tensor(_path(dir, "odin_loss.bin"),   {1},     loss_buf[:])
	save_tensor(_path(dir, "odin_grad_w.bin"), w_shape, w_grad)
}

run_slice_trailing :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	config          := load_int_array(_path(dir, "config.bin"))
	start := config[0]
	end   := config[1]

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	out := ml.slice_trailing(x, start, end)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	ml.get_gradient(x, x_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
}

run_concat3 :: proc(dir: string) {
	a_shape, a_data := load_tensor(_path(dir, "input_a.bin"))
	b_shape, b_data := load_tensor(_path(dir, "input_b.bin"))
	c_shape, c_data := load_tensor(_path(dir, "input_c.bin"))

	a := ml.alloc(.F32, a_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(a)
	b := ml.alloc(.F32, b_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(b)
	c := ml.alloc(.F32, c_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(c)
	ml.set_data(a, a_data)
	ml.set_data(b, b_data)
	ml.set_data(c, c_data)

	out := ml.concat(a, b, c)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	a_grad := builtin.make([]f32, ml.len(a), context.temp_allocator)
	b_grad := builtin.make([]f32, ml.len(b), context.temp_allocator)
	c_grad := builtin.make([]f32, ml.len(c), context.temp_allocator)
	ml.get_gradient(a, a_grad)
	ml.get_gradient(b, b_grad)
	ml.get_gradient(c, c_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_a.bin"), a_shape,   a_grad)
	save_tensor(_path(dir, "odin_grad_b.bin"), b_shape,   b_grad)
	save_tensor(_path(dir, "odin_grad_c.bin"), c_shape,   c_grad)
}

run_rope :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	config          := load_int_array(_path(dir, "config.bin"))

	x := ml.alloc(.F32, x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	head_count := config[0]
	out := ml.rope(x, head_count)
	ml.backward()

	out_shape := _shape_slice(out)
	out_data  := builtin.make([]f32, ml.len(out), context.temp_allocator)
	ml.get_data(out, out_data)

	x_grad := builtin.make([]f32, ml.len(x), context.temp_allocator)
	ml.get_gradient(x, x_grad)

	save_tensor(_path(dir, "odin_out.bin"),    out_shape, out_data)
	save_tensor(_path(dir, "odin_grad_x.bin"), x_shape,   x_grad)
}

// Transformer parity test running the "FP32 master, bf16 compute" recipe.
// Mirrors networks/transformer/transformer.odin's forward but casts every
// master parameter to bf16 inside the graph each step. Logits are cast back
// to F32 for cross_entropy so the loss tensor stays F32 (required by
// ml.backward).
run_transformer_train_bf16 :: proc(dir: string) {
	tokens  := load_int_array(_path(dir, "tokens.bin"))
	targets := load_int_array(_path(dir, "targets.bin"))
	config  := load_int_array(_path(dir, "config.bin"))

	step_count       := config[0]
	period           := config[1]
	layer_count      := config[2]
	head_count       := config[3]
	embedding_size   := config[4]
	vocabulary_size  := config[5]
	token_count      := config[6]
	learning_rate    := f32(config[7]) / 1_000_000
	_ = token_count

	hidden_size := 4 * embedding_size

	Layer :: struct {
		norm0_weight, qkv_weight, proj_weight, norm1_weight, mlp_up_weight, mlp_down_weight: ml.Tensor,
	}

	make_param :: proc(shape: []int) -> ml.Tensor {
		return ml.alloc(.F32, shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	}

	token_embeddings := make_param({vocabulary_size, embedding_size})
	defer ml.destroy(token_embeddings)

	layers := builtin.make([]Layer, layer_count, context.temp_allocator)
	for &layer in layers {
		layer.norm0_weight    = make_param({embedding_size})
		layer.qkv_weight      = make_param({3 * embedding_size, embedding_size})
		layer.proj_weight     = make_param({embedding_size,     embedding_size})
		layer.norm1_weight    = make_param({embedding_size})
		layer.mlp_up_weight   = make_param({hidden_size,    embedding_size})
		layer.mlp_down_weight = make_param({embedding_size, hidden_size})
	}
	defer for layer in layers {
		ml.destroy(layer.norm0_weight)
		ml.destroy(layer.qkv_weight)
		ml.destroy(layer.proj_weight)
		ml.destroy(layer.norm1_weight)
		ml.destroy(layer.mlp_up_weight)
		ml.destroy(layer.mlp_down_weight)
	}

	norm_weight   := make_param({embedding_size})
	output_weight := make_param({vocabulary_size, embedding_size})
	defer ml.destroy(norm_weight)
	defer ml.destroy(output_weight)

	load_into :: proc(t: ml.Tensor, path: string) {
		_, vals := load_tensor(path)
		ml.set_data(t, vals)
	}

	load_into(token_embeddings, _path(dir, "init_token_embeddings.bin"))
	for layer, i in layers {
		load_into(layer.norm0_weight,    fmt.tprintf("%v/init_layer%v_norm0_weight.bin",    dir, i))
		load_into(layer.qkv_weight,      fmt.tprintf("%v/init_layer%v_qkv_weight.bin",      dir, i))
		load_into(layer.proj_weight,     fmt.tprintf("%v/init_layer%v_proj_weight.bin",     dir, i))
		load_into(layer.norm1_weight,    fmt.tprintf("%v/init_layer%v_norm1_weight.bin",    dir, i))
		load_into(layer.mlp_up_weight,   fmt.tprintf("%v/init_layer%v_mlp_up_weight.bin",   dir, i))
		load_into(layer.mlp_down_weight, fmt.tprintf("%v/init_layer%v_mlp_down_weight.bin", dir, i))
	}
	load_into(norm_weight,   _path(dir, "init_norm_weight.bin"))
	load_into(output_weight, _path(dir, "init_output_weight.bin"))

	losses := builtin.make([]f32, step_count, context.temp_allocator)
	opt: ml.Optimizer

	for step in 0 ..< step_count {
		ml.clear()

		residual := ml.cast_to(ml.select(token_embeddings, tokens), .Bf16)

		for layer in layers {
			n0_w  := ml.cast_to(layer.norm0_weight,    .Bf16)
			qkv_w := ml.cast_to(layer.qkv_weight,      .Bf16)
			pr_w  := ml.cast_to(layer.proj_weight,     .Bf16)
			n1_w  := ml.cast_to(layer.norm1_weight,    .Bf16)
			up_w  := ml.cast_to(layer.mlp_up_weight,   .Bf16)
			dn_w  := ml.cast_to(layer.mlp_down_weight, .Bf16)

			normed := ml.layernorm(residual, n0_w)
			qkv    := ml.linear(normed, qkv_w)

			q := ml.slice_trailing(qkv, 0,                  embedding_size)
			k := ml.slice_trailing(qkv, embedding_size,     2 * embedding_size)
			v := ml.slice_trailing(qkv, 2 * embedding_size, 3 * embedding_size)
			q  = ml.rope(q, head_count)
			k  = ml.rope(k, head_count)

			attn_out := ml.attention(q, k, v, head_count)
			attn_out  = ml.linear(attn_out, pr_w)
			residual  = ml.add(residual, attn_out)

			normed_mlp := ml.layernorm(residual, n1_w)
			mlp_out    := ml.linear(normed_mlp, up_w)
			mlp_out     = ml.gelu(mlp_out)
			mlp_out     = ml.linear(mlp_out, dn_w)
			residual    = ml.add(residual, mlp_out)
		}

		nm_w  := ml.cast_to(norm_weight,   .Bf16)
		out_w := ml.cast_to(output_weight, .Bf16)
		out_bf16   := ml.layernorm(residual, nm_w)
		logits_bf16 := ml.linear(out_bf16, out_w)
		logits      := ml.cast_to(logits_bf16, .F32)

		per_token := ml.cross_entropy(logits, targets)
		loss      := ml.mean(per_token)
		ml.backward()

		if ml.optimize(&opt, period=period, learning_rate=learning_rate) {
			ml.update(opt, token_embeddings)
			for layer in layers {
				ml.update(opt, layer.norm0_weight)
				ml.update(opt, layer.qkv_weight)
				ml.update(opt, layer.proj_weight)
				ml.update(opt, layer.norm1_weight)
				ml.update(opt, layer.mlp_up_weight)
				ml.update(opt, layer.mlp_down_weight)
			}
			ml.update(opt, norm_weight)
			ml.update(opt, output_weight)
		}

		loss_buf: [1]f32
		ml.get_data(loss, loss_buf[:])
		losses[step] = loss_buf[0]
	}

	losses_shape := []int{step_count}
	save_tensor(_path(dir, "odin_losses.bin"), losses_shape, losses)
}

_shape_slice :: proc(t: ml.Tensor) -> []int {
	out := builtin.make([]int, t.rank, context.temp_allocator)
	for i in 0 ..< t.rank {
		out[i] = t.shape[i]
	}
	return out
}

load_int_array :: proc(path: string) -> []int {
	bytes, err := os.read_entire_file(path, context.temp_allocator)
	if err != nil {
		fmt.eprintfln("failed to read %v", path)
		os.exit(1)
	}
	if builtin.len(bytes) < 4 {
		fmt.eprintfln("int array too small: %v", path)
		os.exit(1)
	}
	count := int((^u32)(&bytes[0])^)
	if builtin.len(bytes) != 4 + count * 4 {
		fmt.eprintfln("int array size mismatch in %v", path)
		os.exit(1)
	}
	out := builtin.make([]int, count, context.temp_allocator)
	for i in 0 ..< count {
		out[i] = int((^i32)(&bytes[4 + i * 4])^)
	}
	return out
}

// F32 Llama-shape training-step parity test, exercising
// `networks/llama/llama.odin` end-to-end against a faithful PyTorch reference.
run_llama_train :: proc(dir: string) {
	tokens  := load_int_array(_path(dir, "tokens.bin"))
	targets := load_int_array(_path(dir, "targets.bin"))
	config_ints := load_int_array(_path(dir, "config.bin"))

	step_count        := config_ints[0]
	period            := config_ints[1]
	layer_count       := config_ints[2]
	n_q_heads         := config_ints[3]
	n_kv_heads        := config_ints[4]
	head_size         := config_ints[5]
	embedding_size    := config_ints[6]
	intermediate_size := config_ints[7]
	vocabulary_size   := config_ints[8]
	token_count       := config_ints[9]
	learning_rate     := f32(config_ints[10]) / 1_000_000
	rope_base         := f32(config_ints[11])
	_ = token_count

	cfg := llama.Config{
		layer_count       = layer_count,
		n_q_heads         = n_q_heads,
		n_kv_heads        = n_kv_heads,
		head_size         = head_size,
		embedding_size    = embedding_size,
		intermediate_size = intermediate_size,
		vocabulary_size   = vocabulary_size,
		rope_base         = rope_base,
		tied_embeddings   = true,
	}
	model := llama.make(cfg)
	defer llama.destroy(model)

	load_into :: proc(t: ml.Tensor, path: string) {
		_, vals := load_tensor(path)
		ml.set_data(t, vals)
	}

	load_into(model.token_embeddings, _path(dir, "init_token_embeddings.bin"))
	for layer, i in model.layers {
		load_into(layer.input_norm_weight,     fmt.tprintf("%v/init_layer%v_input_norm_weight.bin",     dir, i))
		load_into(layer.q_proj_weight,         fmt.tprintf("%v/init_layer%v_q_proj_weight.bin",         dir, i))
		load_into(layer.k_proj_weight,         fmt.tprintf("%v/init_layer%v_k_proj_weight.bin",         dir, i))
		load_into(layer.v_proj_weight,         fmt.tprintf("%v/init_layer%v_v_proj_weight.bin",         dir, i))
		load_into(layer.o_proj_weight,         fmt.tprintf("%v/init_layer%v_o_proj_weight.bin",         dir, i))
		load_into(layer.post_attn_norm_weight, fmt.tprintf("%v/init_layer%v_post_attn_norm_weight.bin", dir, i))
		load_into(layer.gate_proj_weight,      fmt.tprintf("%v/init_layer%v_gate_proj_weight.bin",      dir, i))
		load_into(layer.up_proj_weight,        fmt.tprintf("%v/init_layer%v_up_proj_weight.bin",        dir, i))
		load_into(layer.down_proj_weight,      fmt.tprintf("%v/init_layer%v_down_proj_weight.bin",      dir, i))
	}
	load_into(model.output_norm_weight, _path(dir, "init_output_norm_weight.bin"))

	losses := builtin.make([]f32, step_count, context.temp_allocator)
	opt: ml.Optimizer

	for step in 0 ..< step_count {
		ml.clear()

		logits    := llama.forward(model, tokens)
		per_token := ml.cross_entropy(logits, targets)
		loss      := ml.mean(per_token)
		ml.backward()

		if ml.optimize(&opt, period=period, learning_rate=learning_rate) {
			llama.update(opt, model)
		}

		loss_buf: [1]f32
		ml.get_data(loss, loss_buf[:])
		losses[step] = loss_buf[0]
	}

	save_tensor(_path(dir, "odin_losses.bin"), {step_count}, losses)
}

run_kv_cache_decode :: proc(dir: string) {
	tokens      := load_int_array(_path(dir, "tokens.bin"))
	config_ints := load_int_array(_path(dir, "config.bin"))

	layer_count       := config_ints[0]
	n_q_heads         := config_ints[1]
	n_kv_heads        := config_ints[2]
	head_size         := config_ints[3]
	embedding_size    := config_ints[4]
	intermediate_size := config_ints[5]
	vocabulary_size   := config_ints[6]
	prompt_count      := config_ints[7]
	decode_count      := config_ints[8]
	rope_base         := f32(config_ints[9])

	total := prompt_count + decode_count

	cfg := llama.Config{
		layer_count       = layer_count,
		n_q_heads         = n_q_heads,
		n_kv_heads        = n_kv_heads,
		head_size         = head_size,
		embedding_size    = embedding_size,
		intermediate_size = intermediate_size,
		vocabulary_size   = vocabulary_size,
		rope_base         = rope_base,
		tied_embeddings   = true,
	}
	model := llama.make(cfg)
	defer llama.destroy(model)

	load_into :: proc(t: ml.Tensor, path: string) {
		_, vals := load_tensor(path)
		ml.set_data(t, vals)
	}

	load_into(model.token_embeddings, _path(dir, "init_token_embeddings.bin"))
	for layer, i in model.layers {
		load_into(layer.input_norm_weight,     fmt.tprintf("%v/init_layer%v_input_norm_weight.bin",     dir, i))
		load_into(layer.q_proj_weight,         fmt.tprintf("%v/init_layer%v_q_proj_weight.bin",         dir, i))
		load_into(layer.k_proj_weight,         fmt.tprintf("%v/init_layer%v_k_proj_weight.bin",         dir, i))
		load_into(layer.v_proj_weight,         fmt.tprintf("%v/init_layer%v_v_proj_weight.bin",         dir, i))
		load_into(layer.o_proj_weight,         fmt.tprintf("%v/init_layer%v_o_proj_weight.bin",         dir, i))
		load_into(layer.post_attn_norm_weight, fmt.tprintf("%v/init_layer%v_post_attn_norm_weight.bin", dir, i))
		load_into(layer.gate_proj_weight,      fmt.tprintf("%v/init_layer%v_gate_proj_weight.bin",      dir, i))
		load_into(layer.up_proj_weight,        fmt.tprintf("%v/init_layer%v_up_proj_weight.bin",        dir, i))
		load_into(layer.down_proj_weight,      fmt.tprintf("%v/init_layer%v_down_proj_weight.bin",      dir, i))
	}
	load_into(model.output_norm_weight, _path(dir, "init_output_norm_weight.bin"))

	cache := llama.cache_make(model, total)
	defer llama.cache_destroy(cache)

	all_logits := builtin.make([]f32, total * vocabulary_size, context.temp_allocator)

	{
		ml.clear()
		logits := llama.forward_cached(model, &cache, tokens[:prompt_count])
		row    := builtin.make([]f32, prompt_count * vocabulary_size, context.temp_allocator)
		ml.get_data(logits, row)
		copy(all_logits[:prompt_count * vocabulary_size], row)
	}

	step_buf := builtin.make([]f32, vocabulary_size, context.temp_allocator)
	for i in 0 ..< decode_count {
		ml.clear()
		logits := llama.forward_cached(model, &cache, tokens[prompt_count + i : prompt_count + i + 1])
		ml.get_data(logits, step_buf)
		dst := all_logits[(prompt_count + i) * vocabulary_size : (prompt_count + i + 1) * vocabulary_size]
		copy(dst, step_buf)
	}

	save_tensor(_path(dir, "odin_logits.bin"), {total, vocabulary_size}, all_logits)
}

_path :: proc(dir, name: string) -> string {
	joined, _ := filepath.join({dir, name}, context.temp_allocator)
	return joined
}

load_tensor :: proc(path: string) -> (shape: []int, data: []f32) {
	bytes, err := os.read_entire_file(path, context.temp_allocator)
	if err != nil {
		fmt.eprintfln("failed to read %v", path)
		os.exit(1)
	}
	if builtin.len(bytes) < 8 || string(bytes[:4]) != MAGIC {
		fmt.eprintfln("bad magic in %v", path)
		os.exit(1)
	}
	rank := int((^u32)(&bytes[4])^)
	header_end := 8 + rank * 4
	shape = builtin.make([]int, rank, context.temp_allocator)
	for i in 0 ..< rank {
		shape[i] = int((^u32)(&bytes[8 + i * 4])^)
	}
	count := 1
	for d in shape {
		count *= d
	}
	expected_size := header_end + count * 4
	if builtin.len(bytes) != expected_size {
		fmt.eprintfln("size mismatch in %v: expected %v got %v", path, expected_size, builtin.len(bytes))
		os.exit(1)
	}
	data = builtin.make([]f32, count, context.temp_allocator)
	mem.copy(raw_data(data), &bytes[header_end], count * 4)
	return
}

save_tensor :: proc(path: string, shape: []int, data: []f32) {
	rank := builtin.len(shape)
	header_size := 8 + rank * 4
	total := header_size + builtin.len(data) * 4
	buf := builtin.make([]byte, total, context.temp_allocator)

	copy(buf[:4], MAGIC)
	(^u32)(&buf[4])^ = u32(rank)
	for d, i in shape {
		(^u32)(&buf[8 + i * 4])^ = u32(d)
	}
	mem.copy(&buf[header_size], raw_data(data), builtin.len(data) * 4)

	if err := os.write_entire_file(path, buf); err != nil {
		fmt.eprintfln("failed to write %v", path)
		os.exit(1)
	}
}
