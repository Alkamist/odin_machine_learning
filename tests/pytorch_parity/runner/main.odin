package pytorch_parity_runner

import "base:builtin"

import "core:fmt"
import "core:os"
import "core:mem"
import "core:strconv"
import "core:path/filepath"

import ml  "../../.."
import cpu "../../../backend_cpu"
import gpu "../../../backend_gpu"
import mlp "../../../mlp"

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
	defer ml.context_destroy(ctx)
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
	case "cross_entropy":  run_cross_entropy(artifacts_dir)
	case "batched_matmul": run_batched_matmul(artifacts_dir)
	case "permute":        run_permute(artifacts_dir)
	case "attention_causal":  run_attention(artifacts_dir)
	case "attention_acausal": run_attention(artifacts_dir)
	case "attention_xfmr":    run_attention(artifacts_dir)
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
	case:
		fmt.eprintfln("unknown test: %v", test_name)
		os.exit(1)
	}
}

Binary_Op :: enum { Add, Sub, Mul, Div }

run_binary_op :: proc(dir: string, op: Binary_Op) {
	a_shape, a_data := load_tensor(_path(dir, "input_a.bin"))
	b_shape, b_data := load_tensor(_path(dir, "input_b.bin"))

	a := ml.alloc(a_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(a)
	b := ml.alloc(b_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	w := ml.alloc(w_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	w := ml.alloc(w_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

run_cross_entropy :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	targets         := load_int_array(_path(dir, "targets.bin"))

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	a := ml.alloc(a_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(a)
	b := ml.alloc(b_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x)
	ml.set_data(x, x_data)

	head_count := config[0]
	causal     := config[1] != 0

	out := ml.attention(x, head_count, causal=causal)
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

	x_target := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(x_target)
	y_target := ml.alloc(y_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

run_slice_trailing :: proc(dir: string) {
	x_shape, x_data := load_tensor(_path(dir, "input_x.bin"))
	config          := load_int_array(_path(dir, "config.bin"))
	start := config[0]
	end   := config[1]

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	a := ml.alloc(a_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(a)
	b := ml.alloc(b_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	defer ml.destroy(b)
	c := ml.alloc(c_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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

	x := ml.alloc(x_shape, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
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
