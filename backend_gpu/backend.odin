package machine_learning_backend_gpu

import "base:builtin"
import "base:runtime"

import "core:fmt"

import vk "vendor:vulkan"

import ml ".."

_backend := ml.Backend_VTable{
	init            = init,
	destroy         = destroy,
	clear           = clear,
	forward         = forward,
	backward        = backward,
	update          = update,
	buffer_alloc    = buffer_alloc,
	buffer_free     = buffer_free,
	buffer_get      = buffer_get,
	buffer_set      = buffer_set,
	buffer_copy     = buffer_copy,
}

// Lazy-init Vulkan and return the vtable. Wire into ml via:
//   ctx := ml.context_create(N, gpu.backend())
@(require_results)
backend :: proc() -> ^ml.Backend_VTable {
	device_init()
	return &_backend
}

init :: proc(ctx: ^ml.Context, size: int, loc: runtime.Source_Code_Location) {
	device_init()

	gctx, err := builtin.new(Gpu_Context, allocator=context.allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate Gpu_Context: %v", err, loc=loc)

	_create_command_pool(gctx, loc)
	_create_descriptor_pool(gctx, loc)

	ctx.backend_data = gctx
}

destroy :: proc(ctx: ^ml.Context, loc: runtime.Source_Code_Location) {
	gctx := cast(^Gpu_Context)ctx.backend_data
	if gctx == nil { return }

	// Any pending GPU work is flushed by clear() and by the synchronous
	// buffer_get / buffer_set / buffer_copy paths, so the batch must not
	// be active here. If it is, the user dropped state on the floor.
	fmt.assertf(!gctx.batch.active, "gpu.destroy called with an active batch; missed a flush?", loc=loc)

	for buffer in gctx.activations {
		_destroy_gpu_buffer(buffer)
	}
	builtin.delete(gctx.activations)

	for _, list in gctx.pool {
		for buffer in list {
			_destroy_gpu_buffer(buffer)
		}
		builtin.delete(list)
	}
	builtin.delete(gctx.pool)

	builtin.delete(gctx.sizes)

	if gctx.staging.buffer != 0 {
		if gctx.staging.mapped != nil {
			vk.UnmapMemory(_gpu.device, gctx.staging.memory)
		}
		vk.DestroyBuffer(_gpu.device, gctx.staging.buffer, nil)
		vk.FreeMemory(_gpu.device, gctx.staging.memory, nil)
	}
	builtin.delete(gctx.pending_downloads)

	builtin.delete(gctx.batch.descriptor_sets)
	builtin.delete(gctx.batch.pending_buffers)
	builtin.delete(gctx.batch.pending_memories)

	if gctx.descriptor_pool != 0 {
		vk.DestroyDescriptorPool(_gpu.device, gctx.descriptor_pool, nil)
	}
	if gctx.command_pool != 0 {
		vk.DestroyCommandPool(_gpu.device, gctx.command_pool, nil)
	}
	builtin.free(gctx, allocator=context.allocator, loc=loc)
	ctx.backend_data = nil
}

clear :: proc(loc: runtime.Source_Code_Location) {
	gctx := _gctx(loc)
	if gctx.batch.active {
		end_batch(loc)
	}

	for buffer in gctx.activations {
		count := gctx.sizes[buffer.buffer]
		list, ok := &gctx.pool[count]
		if !ok {
			gctx.pool[count] = builtin.make([dynamic]Gpu_Buffer)
			list = &gctx.pool[count]
		}
		append(list, buffer)
	}
	builtin.clear(&gctx.activations)
}

buffer_alloc :: proc(len: int, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	gctx := _gctx(loc)
	size := vk.DeviceSize(len * size_of(f32))
	usage := vk.BufferUsageFlags{.STORAGE_BUFFER, .TRANSFER_SRC, .TRANSFER_DST}

	gpu_buffer: Gpu_Buffer
	if !persist {
		if list, ok := &gctx.pool[len]; ok && builtin.len(list^) > 0 {
			gpu_buffer = pop(list)
		}
	}
	if gpu_buffer.buffer == 0 {
		gpu_buffer.buffer, gpu_buffer.memory = _create_buffer(size, usage, {.DEVICE_LOCAL}, loc)
		gctx.sizes[gpu_buffer.buffer] = len
	}

	_record_fill_zero(gpu_buffer.buffer, size, loc)

	if !persist {
		append(&gctx.activations, gpu_buffer)
	}

	return transmute(ml.Backend_Buffer)gpu_buffer
}

buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	gpu_buffer := transmute(Gpu_Buffer)buffer
	if gpu_buffer.buffer == 0 { return }

	gctx := _gctx(loc)
	delete_key(&gctx.sizes, gpu_buffer.buffer)
	_destroy_gpu_buffer(gpu_buffer)
}

buffer_get :: proc(buffer: ml.Backend_Buffer, data: []f32, loc: runtime.Source_Code_Location) {
	gpu_buffer := transmute(Gpu_Buffer)buffer
	if gpu_buffer.buffer == 0 || builtin.len(data) == 0 { return }
	_download(gpu_buffer.buffer, data, loc)
}

buffer_set :: proc(buffer: ml.Backend_Buffer, data: []f32, loc: runtime.Source_Code_Location) {
	gpu_buffer := transmute(Gpu_Buffer)buffer
	if gpu_buffer.buffer == 0 || builtin.len(data) == 0 { return }
	_upload(gpu_buffer.buffer, data, loc)
}

buffer_copy :: proc(dst, src: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	dst_buffer := transmute(Gpu_Buffer)dst
	src_buffer := transmute(Gpu_Buffer)src
	if dst_buffer.buffer == 0 || src_buffer.buffer == 0 { return }

	gctx := _gctx(loc)
	count, ok := gctx.sizes[src_buffer.buffer]
	fmt.assertf(ok, "buffer_copy: source buffer is not registered with this context", loc=loc)
	size := vk.DeviceSize(count * size_of(f32))
	_copy(dst_buffer.buffer, src_buffer.buffer, size, loc)
}

_destroy_gpu_buffer :: proc(buffer: Gpu_Buffer) {
	if buffer.buffer != 0 { vk.DestroyBuffer(_gpu.device, buffer.buffer, nil) }
	if buffer.memory != 0 { vk.FreeMemory   (_gpu.device, buffer.memory, nil) }
}

@(require_results)
data :: #force_inline proc(t: ml.Tensor) -> Gpu_Buffer {
	return transmute(Gpu_Buffer)t.buffers[.Data]
}

@(require_results)
gradient :: #force_inline proc(t: ml.Tensor) -> Gpu_Buffer {
	return transmute(Gpu_Buffer)t.buffers[.Gradient]
}

@(require_results)
adam_m :: #force_inline proc(t: ml.Tensor) -> Gpu_Buffer {
	return transmute(Gpu_Buffer)t.buffers[.Adam_M]
}

@(require_results)
adam_v :: #force_inline proc(t: ml.Tensor) -> Gpu_Buffer {
	return transmute(Gpu_Buffer)t.buffers[.Adam_V]
}

update :: proc(opt: ml.Optimizer, t: ^ml.Tensor, loc: runtime.Source_Code_Location) {
	d := data(t^)
	g := gradient(t^)
	m := adam_m(t^)
	v := adam_v(t^)

	fmt.assertf(d.buffer != 0, "update: tensor Data buffer missing",     loc=loc)
	fmt.assertf(g.buffer != 0, "update: tensor Gradient buffer missing", loc=loc)
	fmt.assertf(m.buffer != 0, "update: tensor Adam_M buffer missing",   loc=loc)
	fmt.assertf(v.buffer != 0, "update: tensor Adam_V buffer missing",   loc=loc)

	if _adam_step_pipeline == nil {
		_adam_step_pipeline = _make_pipeline(ADAM_STEP_SPIRV, 4, size_of(Adam_Params))
	}
	n := ml.len(t^)
	params := Adam_Params{
		n     = u32(n),
		lr    = opt.learning_rate,
		beta1 = opt.beta1,
		beta2 = opt.beta2,
		eps   = opt.epsilon,
		wd    = opt.weight_decay,
		bc1   = opt.bias_correction1,
		bc2   = opt.bias_correction2,
	}
	bufs := [4]vk.Buffer{d.buffer, g.buffer, m.buffer, v.buffer}
	_dispatch(_adam_step_pipeline, bufs[:], &params, _div_up(n, 256))
}

forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	switch _ in op.variant {
	case ml.Add:                add_forward                (op)
	case ml.Sub:                sub_forward                (op)
	case ml.Mul:                mul_forward                (op)
	case ml.Div:                div_forward                (op)
	case ml.Exp:                exp_forward                (op)
	case ml.Clamp:              clamp_forward              (op)
	case ml.Min:                min_forward                (op)
	case ml.Max:                max_forward                (op)
	case ml.Mean:               mean_forward               (op)
	case ml.Transpose:          transpose_forward          (op)
	case ml.Select:             select_forward             (op)
	case ml.Slice:              slice_forward              (op)
	case ml.Slice_Trailing:     slice_trailing_forward     (op)
	case ml.Concat:             concat_forward             (op)
	case ml.Linear:             linear_forward             (op)
	case ml.Rope:               rope_forward               (op)
	case ml.Layernorm:          layernorm_forward          (op)
	case ml.Softmax:            softmax_forward            (op)
	case ml.Entropy:            entropy_forward            (op)
	case ml.Log_Softmax:        log_softmax_forward        (op)
	case ml.Mean_Squared_Error: mean_squared_error_forward (op)
	case ml.Cross_Entropy:      cross_entropy_forward      (op)
	case ml.Relu:               relu_forward               (op)
	case ml.Sigmoid:            sigmoid_forward            (op)
	case ml.Gelu:               gelu_forward               (op)
	case ml.Silu:               silu_forward               (op)
	case ml.Tanh:               tanh_forward               (op)
	case ml.Batched_Matmul:     batched_matmul_forward     (op)
	case ml.Permute:            permute_forward            (op)
	case ml.Causal_Mask:        causal_mask_forward        (op)
	case ml.Attention:          attention_forward          (op)
	}
}

backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	switch _ in op.variant {
	case ml.Add:                add_backward               (op)
	case ml.Sub:                sub_backward               (op)
	case ml.Mul:                mul_backward               (op)
	case ml.Div:                div_backward               (op)
	case ml.Exp:                exp_backward               (op)
	case ml.Clamp:              clamp_backward             (op)
	case ml.Min:                min_backward               (op)
	case ml.Max:                max_backward               (op)
	case ml.Mean:               mean_backward              (op)
	case ml.Transpose:          transpose_backward         (op)
	case ml.Select:             select_backward            (op)
	case ml.Slice:              slice_backward             (op)
	case ml.Slice_Trailing:     slice_trailing_backward    (op)
	case ml.Concat:             concat_backward            (op)
	case ml.Linear:             linear_backward            (op)
	case ml.Rope:               rope_backward              (op)
	case ml.Layernorm:          layernorm_backward         (op)
	case ml.Softmax:            softmax_backward           (op)
	case ml.Entropy:            entropy_backward           (op)
	case ml.Log_Softmax:        log_softmax_backward       (op)
	case ml.Mean_Squared_Error: mean_squared_error_backward(op)
	case ml.Cross_Entropy:      cross_entropy_backward     (op)
	case ml.Relu:               relu_backward              (op)
	case ml.Sigmoid:            sigmoid_backward           (op)
	case ml.Gelu:               gelu_backward              (op)
	case ml.Silu:               silu_backward              (op)
	case ml.Tanh:               tanh_backward              (op)
	case ml.Batched_Matmul:     batched_matmul_backward    (op)
	case ml.Permute:            permute_backward           (op)
	case ml.Causal_Mask:        causal_mask_backward       (op)
	case ml.Attention:          attention_backward         (op)
	}
}

add_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b

	if _add_pipeline == nil {
		_add_pipeline = _make_pipeline(ADD_SPIRV, 3, size_of(Add_Params))
	}
	params := Add_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
	_dispatch(_add_pipeline, bufs[:], &params, _div_up(ml.len(a), ADD_LOCAL_SIZE))
}

add_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b
	stride := ml.len(a) / ml.len(b)

	if _add_back_a_pipeline == nil {
		_add_back_a_pipeline = _make_pipeline(ADD_BACK_A_SPIRV, 2, size_of(Add_Back_A_Params))
	}
	a_params := Add_Back_A_Params{n = u32(ml.len(a))}
	a_bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(a).buffer}
	_dispatch(_add_back_a_pipeline, a_bufs[:], &a_params, _div_up(ml.len(a), 256))

	if _add_back_b_pipeline == nil {
		_add_back_b_pipeline = _make_pipeline(ADD_BACK_B_SPIRV, 2, size_of(Add_Back_B_Params))
	}
	b_params := Add_Back_B_Params{n_b = u32(ml.len(b)), stride = u32(stride)}
	b_bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(b).buffer}
	_dispatch(_add_back_b_pipeline, b_bufs[:], &b_params, _div_up(ml.len(b), 256))
}

sub_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Sub).b

	if _sub_pipeline == nil {
		_sub_pipeline = _make_pipeline(SUB_SPIRV, 3, size_of(Sub_Params))
	}
	params := Sub_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
	_dispatch(_sub_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

sub_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Sub).b
	stride := ml.len(a) / ml.len(b)

	if _sub_back_a_pipeline == nil {
		_sub_back_a_pipeline = _make_pipeline(SUB_BACK_A_SPIRV, 2, size_of(Sub_Back_A_Params))
	}
	a_params := Sub_Back_A_Params{n = u32(ml.len(a))}
	a_bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(a).buffer}
	_dispatch(_sub_back_a_pipeline, a_bufs[:], &a_params, _div_up(ml.len(a), 256))

	if _sub_back_b_pipeline == nil {
		_sub_back_b_pipeline = _make_pipeline(SUB_BACK_B_SPIRV, 2, size_of(Sub_Back_B_Params))
	}
	b_params := Sub_Back_B_Params{n_b = u32(ml.len(b)), stride = u32(stride)}
	b_bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(b).buffer}
	_dispatch(_sub_back_b_pipeline, b_bufs[:], &b_params, _div_up(ml.len(b), 256))
}

mul_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Mul).b

	if _mul_pipeline == nil {
		_mul_pipeline = _make_pipeline(MUL_SPIRV, 3, size_of(Mul_Params))
	}
	params := Mul_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
	_dispatch(_mul_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

mul_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Mul).b
	stride := ml.len(a) / ml.len(b)

	if _mul_back_a_pipeline == nil {
		_mul_back_a_pipeline = _make_pipeline(MUL_BACK_A_SPIRV, 3, size_of(Mul_Back_A_Params))
	}
	if _mul_back_b_pipeline == nil {
		_mul_back_b_pipeline = _make_pipeline(MUL_BACK_B_SPIRV, 3, size_of(Mul_Back_B_Params))
	}

	a_params := Mul_Back_A_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
	a_bufs   := [3]vk.Buffer{data(b).buffer, gradient(output).buffer, gradient(a).buffer}
	_dispatch(_mul_back_a_pipeline, a_bufs[:], &a_params, _div_up(ml.len(a), 256))

	b_params := Mul_Back_B_Params{n_b = u32(ml.len(b)), stride = u32(stride)}
	b_bufs   := [3]vk.Buffer{data(a).buffer, gradient(output).buffer, gradient(b).buffer}
	_dispatch(_mul_back_b_pipeline, b_bufs[:], &b_params, _div_up(ml.len(b), 256))
}

div_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Div).b

	if _div_pipeline == nil {
		_div_pipeline = _make_pipeline(DIV_SPIRV, 3, size_of(Div_Params))
	}
	params := Div_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
	_dispatch(_div_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

div_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Div).b
	stride := ml.len(a) / ml.len(b)

	if _div_back_a_pipeline == nil {
		_div_back_a_pipeline = _make_pipeline(DIV_BACK_A_SPIRV, 3, size_of(Div_Back_A_Params))
	}
	a_params := Div_Back_A_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
	a_bufs   := [3]vk.Buffer{data(b).buffer, gradient(output).buffer, gradient(a).buffer}
	_dispatch(_div_back_a_pipeline, a_bufs[:], &a_params, _div_up(ml.len(a), 256))

	if _div_back_b_pipeline == nil {
		_div_back_b_pipeline = _make_pipeline(DIV_BACK_B_SPIRV, 4, size_of(Div_Back_B_Params))
	}
	b_params := Div_Back_B_Params{n_b = u32(ml.len(b)), stride = u32(stride)}
	b_bufs   := [4]vk.Buffer{data(a).buffer, data(b).buffer, gradient(output).buffer, gradient(b).buffer}
	_dispatch(_div_back_b_pipeline, b_bufs[:], &b_params, _div_up(ml.len(b), 256))
}

exp_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _exp_pipeline == nil {
		_exp_pipeline = _make_pipeline(EXP_SPIRV, 2, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_exp_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

exp_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _exp_back_pipeline == nil {
		_exp_back_pipeline = _make_pipeline(EXP_BACK_SPIRV, 3, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [3]vk.Buffer{data(y).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_exp_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

clamp_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Clamp)
	if _clamp_pipeline == nil {
		_clamp_pipeline = _make_pipeline(CLAMP_SPIRV, 2, size_of(Clamp_Params))
	}
	params := Clamp_Params{n = u32(ml.len(x)), min_val = v.min_val, max_val = v.max_val}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_clamp_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

clamp_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Clamp)
	if _clamp_back_pipeline == nil {
		_clamp_back_pipeline = _make_pipeline(CLAMP_BACK_SPIRV, 3, size_of(Clamp_Params))
	}
	params := Clamp_Params{n = u32(ml.len(x)), min_val = v.min_val, max_val = v.max_val}
	bufs   := [3]vk.Buffer{data(x).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_clamp_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

min_forward :: proc(op: ml.Operation) {
	a := op.input; y := op.output; b := op.variant.(ml.Min).b
	if _min_pipeline == nil {
		_min_pipeline = _make_pipeline(MIN_SPIRV, 3, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(y).buffer}
	_dispatch(_min_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

min_backward :: proc(op: ml.Operation) {
	a := op.input; y := op.output; b := op.variant.(ml.Min).b
	if _min_back_pipeline == nil {
		_min_back_pipeline = _make_pipeline(MIN_BACK_SPIRV, 5, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [5]vk.Buffer{data(a).buffer, data(b).buffer, gradient(y).buffer, gradient(a).buffer, gradient(b).buffer}
	_dispatch(_min_back_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

max_forward :: proc(op: ml.Operation) {
	a := op.input; y := op.output; b := op.variant.(ml.Max).b
	if _max_pipeline == nil {
		_max_pipeline = _make_pipeline(MAX_SPIRV, 3, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(y).buffer}
	_dispatch(_max_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

max_backward :: proc(op: ml.Operation) {
	a := op.input; y := op.output; b := op.variant.(ml.Max).b
	if _max_back_pipeline == nil {
		_max_back_pipeline = _make_pipeline(MAX_BACK_SPIRV, 5, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [5]vk.Buffer{data(a).buffer, data(b).buffer, gradient(y).buffer, gradient(a).buffer, gradient(b).buffer}
	_dispatch(_max_back_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

mean_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	count := ml.len(y)
	size  := ml.len(x) / count
	if _mean_pipeline == nil {
		_mean_pipeline = _make_pipeline(MEAN_SPIRV, 2, size_of(Mean_Params))
	}
	params := Mean_Params{count = u32(count), size = u32(size)}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_mean_pipeline, bufs[:], &params, u32(count))
}

mean_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	count := ml.len(y)
	size  := ml.len(x) / count
	if _mean_back_pipeline == nil {
		_mean_back_pipeline = _make_pipeline(MEAN_BACK_SPIRV, 2, size_of(Mean_Params))
	}
	params := Mean_Params{count = u32(count), size = u32(size)}
	bufs   := [2]vk.Buffer{gradient(y).buffer, gradient(x).buffer}
	_dispatch(_mean_back_pipeline, bufs[:], &params, _div_up(count * size, 256))
}

transpose_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	rows := x.shape[0]
	cols := x.shape[1]
	if _transpose_pipeline == nil {
		_transpose_pipeline = _make_pipeline(TRANSPOSE_SPIRV, 2, size_of(Transpose_Params))
	}
	params := Transpose_Params{rows = u32(rows), cols = u32(cols)}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_transpose_pipeline, bufs[:], &params, _div_up(cols, 16), _div_up(rows, 16))
}

transpose_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	rows := x.shape[0]
	cols := x.shape[1]
	if _transpose_back_pipeline == nil {
		_transpose_back_pipeline = _make_pipeline(TRANSPOSE_BACK_SPIRV, 2, size_of(Transpose_Params))
	}
	params := Transpose_Params{rows = u32(rows), cols = u32(cols)}
	bufs   := [2]vk.Buffer{gradient(y).buffer, gradient(x).buffer}
	_dispatch(_transpose_back_pipeline, bufs[:], &params, _div_up(cols, 16), _div_up(rows, 16))
}

select_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(y) / builtin.len(indices)

	idx_buf, idx_mem := _upload_indices(indices)

	if _select_pipeline == nil {
		_select_pipeline = _make_pipeline(SELECT_SPIRV, 3, size_of(Select_Params))
	}
	params := Select_Params{n_indices = u32(builtin.len(indices)), size = u32(size)}
	bufs   := [3]vk.Buffer{data(x).buffer, idx_buf, data(y).buffer}
	_dispatch(_select_pipeline, bufs[:], &params, _div_up(size, 256), u32(builtin.len(indices)), 1)

	_queue_destroy_buffer(idx_buf, idx_mem)
}

select_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(y) / builtin.len(indices)
	vocab   := x.shape[0]

	idx_buf, idx_mem := _upload_indices(indices)

	if _select_back_pipeline == nil {
		_select_back_pipeline = _make_pipeline(SELECT_BACK_SPIRV, 3, size_of(Select_Back_Params))
	}
	params := Select_Back_Params{vocab = u32(vocab), n_indices = u32(builtin.len(indices)), size = u32(size)}
	bufs   := [3]vk.Buffer{idx_buf, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_select_back_pipeline, bufs[:], &params, _div_up(vocab, 16), _div_up(size, 16), 1)

	_queue_destroy_buffer(idx_buf, idx_mem)
}

slice_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output; v := op.variant.(ml.Slice)
	if _slice_pipeline == nil {
		_slice_pipeline = _make_pipeline(SLICE_SPIRV, 2, size_of(Slice_Params))
	}
	params := Slice_Params{n = u32(ml.len(y)), start = u32(v.start)}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_slice_pipeline, bufs[:], &params, _div_up(ml.len(y), 256))
}

slice_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output; v := op.variant.(ml.Slice)
	if _slice_back_pipeline == nil {
		_slice_back_pipeline = _make_pipeline(SLICE_BACK_SPIRV, 2, size_of(Slice_Params))
	}
	params := Slice_Params{n = u32(ml.len(y)), start = u32(v.start)}
	bufs   := [2]vk.Buffer{gradient(y).buffer, gradient(x).buffer}
	_dispatch(_slice_back_pipeline, bufs[:], &params, _div_up(ml.len(y), 256))
}

slice_trailing_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output; v := op.variant.(ml.Slice_Trailing)

	trailing     := x.shape[x.rank - 1]
	new_trailing := y.shape[y.rank - 1]
	leading      := ml.len(x) / trailing

	if _slice_trailing_pipeline == nil {
		_slice_trailing_pipeline = _make_pipeline(SLICE_TRAILING_SPIRV, 2, size_of(Slice_Trailing_Params))
	}
	params := Slice_Trailing_Params{
		leading      = u32(leading),
		trailing     = u32(trailing),
		new_trailing = u32(new_trailing),
		start        = u32(v.start),
	}
	bufs := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_slice_trailing_pipeline, bufs[:], &params, _div_up(leading * new_trailing, 256))
}

slice_trailing_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output; v := op.variant.(ml.Slice_Trailing)

	trailing     := x.shape[x.rank - 1]
	new_trailing := y.shape[y.rank - 1]
	leading      := ml.len(x) / trailing

	if _slice_trailing_back_pipeline == nil {
		_slice_trailing_back_pipeline = _make_pipeline(SLICE_TRAILING_BACK_SPIRV, 2, size_of(Slice_Trailing_Back_Params))
	}
	params := Slice_Trailing_Back_Params{
		leading      = u32(leading),
		trailing     = u32(trailing),
		new_trailing = u32(new_trailing),
		start        = u32(v.start),
	}
	bufs := [2]vk.Buffer{gradient(x).buffer, gradient(y).buffer}
	_dispatch(_slice_trailing_back_pipeline, bufs[:], &params, _div_up(leading * new_trailing, 256))
}

concat_forward :: proc(op: ml.Operation) {
	output  := op.output
	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	fmt.assertf(builtin.len(inputs) == 3, "GPU concat only supports 3 inputs (got %v)", builtin.len(inputs))
	a, b, c := inputs[0], inputs[1], inputs[2]

	t_a     := a.shape[a.rank - 1]
	t_b     := b.shape[b.rank - 1]
	t_c     := c.shape[c.rank - 1]
	leading := ml.len(a) / t_a

	if _concat3_pipeline == nil {
		_concat3_pipeline = _make_pipeline(CONCAT3_SPIRV, 4, size_of(Concat3_Params))
	}
	params := Concat3_Params{leading = u32(leading), t_a = u32(t_a), t_b = u32(t_b), t_c = u32(t_c)}
	bufs   := [4]vk.Buffer{data(a).buffer, data(b).buffer, data(c).buffer, data(output).buffer}
	total  := leading * (t_a + t_b + t_c)
	_dispatch(_concat3_pipeline, bufs[:], &params, _div_up(total, 256))
}

concat_backward :: proc(op: ml.Operation) {
	output  := op.output
	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	fmt.assertf(builtin.len(inputs) == 3, "GPU concat only supports 3 inputs (got %v)", builtin.len(inputs))
	a, b, c := inputs[0], inputs[1], inputs[2]

	t_a     := a.shape[a.rank - 1]
	t_b     := b.shape[b.rank - 1]
	t_c     := c.shape[c.rank - 1]
	leading := ml.len(a) / t_a

	if _concat3_back_pipeline == nil {
		_concat3_back_pipeline = _make_pipeline(CONCAT3_BACK_SPIRV, 4, size_of(Concat3_Back_Params))
	}
	params := Concat3_Back_Params{leading = u32(leading), t_a = u32(t_a), t_b = u32(t_b), t_c = u32(t_c)}
	bufs   := [4]vk.Buffer{gradient(a).buffer, gradient(b).buffer, gradient(c).buffer, gradient(output).buffer}
	total  := leading * (t_a + t_b + t_c)
	_dispatch(_concat3_back_pipeline, bufs[:], &params, _div_up(total, 256))
}

linear_forward :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	if _linear_pipeline == nil {
		_linear_pipeline = _make_pipeline(LINEAR_SPIRV, 3, size_of(Linear_Params))
	}
	params := Linear_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}
	bufs := [3]vk.Buffer{data(input).buffer, data(weight).buffer, data(output).buffer}
	_dispatch(
		_linear_pipeline, bufs[:], &params,
		_div_up(count,       LINEAR_LOCAL_X),
		_div_up(output_size, LINEAR_LOCAL_Y),
		1,
	)
}

linear_backward :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	if _linear_back_input_pipeline == nil {
		_linear_back_input_pipeline = _make_pipeline(LINEAR_BACK_INPUT_SPIRV, 3, size_of(Linear_Back_Params))
	}
	if _linear_back_weight_pipeline == nil {
		_linear_back_weight_pipeline = _make_pipeline(LINEAR_BACK_WEIGHT_SPIRV, 3, size_of(Linear_Back_Params))
	}
	params := Linear_Back_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}

	dx_bufs := [3]vk.Buffer{gradient(output).buffer, data(weight).buffer, gradient(input).buffer}
	_dispatch(
		_linear_back_input_pipeline, dx_bufs[:], &params,
		_div_up(count,      16),
		_div_up(input_size, 16),
		1,
	)

	dw_bufs := [3]vk.Buffer{data(input).buffer, gradient(output).buffer, gradient(weight).buffer}
	_dispatch(
		_linear_back_weight_pipeline, dw_bufs[:], &params,
		_div_up(output_size, 16),
		_div_up(input_size,  16),
		1,
	)
}

rope_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Rope)
	token_count := x.shape[0]
	head_size   := x.shape[x.rank - 1] / v.head_count

	if _rope_pipeline == nil {
		_rope_pipeline = _make_pipeline(ROPE_SPIRV, 2, size_of(Rope_Params))
	}
	params := Rope_Params{
		token_count = u32(token_count),
		head_count  = u32(v.head_count),
		head_size   = u32(head_size),
		base        = v.base,
	}
	bufs        := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	total_pairs := token_count * v.head_count * (head_size / 2)
	_dispatch(_rope_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
}

rope_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Rope)
	token_count := x.shape[0]
	head_size   := x.shape[x.rank - 1] / v.head_count

	if _rope_back_pipeline == nil {
		_rope_back_pipeline = _make_pipeline(ROPE_BACK_SPIRV, 2, size_of(Rope_Back_Params))
	}
	params := Rope_Back_Params{
		token_count = u32(token_count),
		head_count  = u32(v.head_count),
		head_size   = u32(head_size),
		base        = v.base,
	}
	bufs        := [2]vk.Buffer{gradient(x).buffer, gradient(y).buffer}
	total_pairs := token_count * v.head_count * (head_size / 2)
	_dispatch(_rope_back_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
}

layernorm_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Layernorm)
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size

	if _layernorm_stats_pipeline == nil {
		_layernorm_stats_pipeline = _make_pipeline(LAYERNORM_STATS_SPIRV, 3, size_of(Layernorm_Stats_Params))
	}
	if _layernorm_pipeline == nil {
		_layernorm_pipeline = _make_pipeline(LAYERNORM_SPIRV, 3, size_of(Layernorm_Params))
	}

	stats_params := Layernorm_Stats_Params{count = u32(count), size = u32(size)}
	stats_bufs   := [3]vk.Buffer{data(x).buffer, data(v.mean).buffer, data(v.rstd).buffer}
	_dispatch(_layernorm_stats_pipeline, stats_bufs[:], &stats_params, u32(count))

	fwd_params := Layernorm_Params{count = u32(count), size = u32(size)}
	fwd_bufs   := [3]vk.Buffer{data(x).buffer, data(v.weight).buffer, data(y).buffer}
	_dispatch(_layernorm_pipeline, fwd_bufs[:], &fwd_params, u32(count))
}

layernorm_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Layernorm)
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size

	if _layernorm_back_input_pipeline == nil {
		_layernorm_back_input_pipeline = _make_pipeline(LAYERNORM_BACK_INPUT_SPIRV, 6, size_of(Layernorm_Back_Params))
	}
	if _layernorm_back_weight_pipeline == nil {
		_layernorm_back_weight_pipeline = _make_pipeline(LAYERNORM_BACK_WEIGHT_SPIRV, 5, size_of(Layernorm_Back_Params))
	}

	params := Layernorm_Back_Params{count = u32(count), size = u32(size)}

	in_bufs := [6]vk.Buffer{
		data(x).buffer, data(v.weight).buffer, gradient(y).buffer,
		data(v.mean).buffer, data(v.rstd).buffer, gradient(x).buffer,
	}
	_dispatch(_layernorm_back_input_pipeline, in_bufs[:], &params, u32(count))

	w_bufs := [5]vk.Buffer{
		data(x).buffer, gradient(y).buffer,
		data(v.mean).buffer, data(v.rstd).buffer, gradient(v.weight).buffer,
	}
	_dispatch(_layernorm_back_weight_pipeline, w_bufs[:], &params, _div_up(size, 256))
}

softmax_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size
	if _softmax_pipeline == nil {
		_softmax_pipeline = _make_pipeline(SOFTMAX_SPIRV, 2, size_of(Softmax_Params))
	}
	params := Softmax_Params{count = u32(count), size = u32(size)}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_softmax_pipeline, bufs[:], &params, u32(count))
}

softmax_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size
	if _softmax_back_pipeline == nil {
		_softmax_back_pipeline = _make_pipeline(SOFTMAX_BACK_SPIRV, 3, size_of(Softmax_Back_Params))
	}
	params := Softmax_Back_Params{count = u32(count), size = u32(size)}
	bufs   := [3]vk.Buffer{data(y).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_softmax_back_pipeline, bufs[:], &params, u32(count))
}

entropy_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size
	if _entropy_pipeline == nil {
		_entropy_pipeline = _make_pipeline(ENTROPY_SPIRV, 2, size_of(Entropy_Params))
	}
	params := Entropy_Params{count = u32(count), size = u32(size)}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_entropy_pipeline, bufs[:], &params, u32(count))
}

entropy_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size
	if _entropy_back_pipeline == nil {
		_entropy_back_pipeline = _make_pipeline(ENTROPY_BACK_SPIRV, 3, size_of(Entropy_Params))
	}
	params := Entropy_Params{count = u32(count), size = u32(size)}
	bufs   := [3]vk.Buffer{data(x).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_entropy_back_pipeline, bufs[:], &params, _div_up(count * size, 256))
}

log_softmax_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size
	if _log_softmax_pipeline == nil {
		_log_softmax_pipeline = _make_pipeline(LOG_SOFTMAX_SPIRV, 2, size_of(Log_Softmax_Params))
	}
	params := Log_Softmax_Params{count = u32(count), size = u32(size)}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_log_softmax_pipeline, bufs[:], &params, u32(count))
}

log_softmax_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	size  := x.shape[x.rank - 1]
	count := ml.len(x) / size
	if _log_softmax_back_pipeline == nil {
		_log_softmax_back_pipeline = _make_pipeline(LOG_SOFTMAX_BACK_SPIRV, 3, size_of(Log_Softmax_Params))
	}
	params := Log_Softmax_Params{count = u32(count), size = u32(size)}
	bufs   := [3]vk.Buffer{data(y).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_log_softmax_back_pipeline, bufs[:], &params, u32(count))
}

mean_squared_error_forward :: proc(op: ml.Operation) {
	predictions := op.input; y := op.output
	targets := op.variant.(ml.Mean_Squared_Error).targets
	count := ml.len(y)
	size  := ml.len(predictions) / count
	if _mean_squared_error_pipeline == nil {
		_mean_squared_error_pipeline = _make_pipeline(MEAN_SQUARED_ERROR_SPIRV, 3, size_of(Mean_Squared_Error_Params))
	}
	params := Mean_Squared_Error_Params{count = u32(count), size = u32(size)}
	bufs   := [3]vk.Buffer{data(predictions).buffer, data(targets).buffer, data(y).buffer}
	_dispatch(_mean_squared_error_pipeline, bufs[:], &params, u32(count))
}

mean_squared_error_backward :: proc(op: ml.Operation) {
	predictions := op.input; y := op.output
	targets := op.variant.(ml.Mean_Squared_Error).targets
	count := ml.len(y)
	size  := ml.len(predictions) / count
	if _mean_squared_error_back_pipeline == nil {
		_mean_squared_error_back_pipeline = _make_pipeline(MEAN_SQUARED_ERROR_BACK_SPIRV, 4, size_of(Mean_Squared_Error_Params))
	}
	params := Mean_Squared_Error_Params{count = u32(count), size = u32(size)}
	bufs   := [4]vk.Buffer{data(predictions).buffer, data(targets).buffer, gradient(y).buffer, gradient(predictions).buffer}
	_dispatch(_mean_squared_error_back_pipeline, bufs[:], &params, _div_up(ml.len(predictions), 256))
}

cross_entropy_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Cross_Entropy)
	class_size := x.shape[x.rank - 1]

	tgt_buf, tgt_mem := _upload_indices(v.targets)

	if _cross_entropy_pipeline == nil {
		_cross_entropy_pipeline = _make_pipeline(CROSS_ENTROPY_SPIRV, 4, size_of(Cross_Entropy_Params))
	}
	params := Cross_Entropy_Params{count = u32(builtin.len(v.targets)), class_size = u32(class_size)}
	bufs   := [4]vk.Buffer{data(x).buffer, tgt_buf, data(v.probabilities).buffer, data(y).buffer}
	_dispatch(_cross_entropy_pipeline, bufs[:], &params, u32(builtin.len(v.targets)))

	_queue_destroy_buffer(tgt_buf, tgt_mem)
}

cross_entropy_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Cross_Entropy)
	class_size := x.shape[x.rank - 1]

	tgt_buf, tgt_mem := _upload_indices(v.targets)

	if _cross_entropy_back_pipeline == nil {
		_cross_entropy_back_pipeline = _make_pipeline(CROSS_ENTROPY_BACK_SPIRV, 4, size_of(Cross_Entropy_Params))
	}
	params := Cross_Entropy_Params{count = u32(builtin.len(v.targets)), class_size = u32(class_size)}
	bufs   := [4]vk.Buffer{data(v.probabilities).buffer, tgt_buf, gradient(y).buffer, gradient(x).buffer}
	total  := builtin.len(v.targets) * class_size
	_dispatch(_cross_entropy_back_pipeline, bufs[:], &params, _div_up(total, 256))

	_queue_destroy_buffer(tgt_buf, tgt_mem)
}

relu_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _relu_pipeline == nil {
		_relu_pipeline = _make_pipeline(RELU_SPIRV, 2, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_relu_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

relu_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _relu_back_pipeline == nil {
		_relu_back_pipeline = _make_pipeline(RELU_BACK_SPIRV, 3, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [3]vk.Buffer{data(x).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_relu_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

sigmoid_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _sigmoid_pipeline == nil {
		_sigmoid_pipeline = _make_pipeline(SIGMOID_SPIRV, 2, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_sigmoid_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

sigmoid_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _sigmoid_back_pipeline == nil {
		_sigmoid_back_pipeline = _make_pipeline(SIGMOID_BACK_SPIRV, 3, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [3]vk.Buffer{data(y).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_sigmoid_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

gelu_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _gelu_pipeline == nil {
		_gelu_pipeline = _make_pipeline(GELU_SPIRV, 2, size_of(Gelu_Params))
	}
	params := Gelu_Params{n = u32(ml.len(x))}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_gelu_pipeline, bufs[:], &params, _div_up(ml.len(x), GELU_LOCAL_SIZE))
}

gelu_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _gelu_back_pipeline == nil {
		_gelu_back_pipeline = _make_pipeline(GELU_BACK_SPIRV, 3, size_of(Gelu_Back_Params))
	}
	params := Gelu_Back_Params{n = u32(ml.len(x))}
	bufs   := [3]vk.Buffer{data(x).buffer, gradient(x).buffer, gradient(y).buffer}
	_dispatch(_gelu_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

silu_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _silu_pipeline == nil {
		_silu_pipeline = _make_pipeline(SILU_SPIRV, 2, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_silu_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

silu_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _silu_back_pipeline == nil {
		_silu_back_pipeline = _make_pipeline(SILU_BACK_SPIRV, 3, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [3]vk.Buffer{data(x).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_silu_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

tanh_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _tanh_pipeline == nil {
		_tanh_pipeline = _make_pipeline(TANH_SPIRV, 2, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_tanh_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

tanh_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	if _tanh_back_pipeline == nil {
		_tanh_back_pipeline = _make_pipeline(TANH_BACK_SPIRV, 3, size_of(Activation_Params))
	}
	params := Activation_Params{n = u32(ml.len(x))}
	bufs   := [3]vk.Buffer{data(y).buffer, gradient(y).buffer, gradient(x).buffer}
	_dispatch(_tanh_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

batched_matmul_forward :: proc(op: ml.Operation) {
	a := op.input; output := op.output
	b := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m := a.shape[1]
	k := a.shape[2]
	n := b.shape[2]

	if _batched_matmul_pipeline == nil {
		_batched_matmul_pipeline = _make_pipeline(BATCHED_MATMUL_SPIRV, 3, size_of(Batched_Matmul_Params))
	}
	params := Batched_Matmul_Params{
		batch_count = u32(batch_count),
		m           = u32(m),
		k           = u32(k),
		n           = u32(n),
	}
	bufs := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
	_dispatch(
		_batched_matmul_pipeline, bufs[:], &params,
		_div_up(m, BATCHED_MATMUL_LOCAL_X),
		_div_up(n, BATCHED_MATMUL_LOCAL_Y),
		u32(batch_count),
	)
}

batched_matmul_backward :: proc(op: ml.Operation) {
	a := op.input; output := op.output
	b := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m := a.shape[1]
	k := a.shape[2]
	n := b.shape[2]

	if _batched_matmul_back_input_pipeline == nil {
		_batched_matmul_back_input_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_INPUT_SPIRV, 3, size_of(Batched_Matmul_Params))
	}
	if _batched_matmul_back_weight_pipeline == nil {
		_batched_matmul_back_weight_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_WEIGHT_SPIRV, 3, size_of(Batched_Matmul_Params))
	}
	params := Batched_Matmul_Params{
		batch_count = u32(batch_count),
		m           = u32(m),
		k           = u32(k),
		n           = u32(n),
	}

	da_bufs := [3]vk.Buffer{gradient(output).buffer, data(b).buffer, gradient(a).buffer}
	_dispatch(
		_batched_matmul_back_input_pipeline, da_bufs[:], &params,
		_div_up(m, BATCHED_MATMUL_LOCAL_X),
		_div_up(k, BATCHED_MATMUL_LOCAL_Y),
		u32(batch_count),
	)

	db_bufs := [3]vk.Buffer{data(a).buffer, gradient(output).buffer, gradient(b).buffer}
	_dispatch(
		_batched_matmul_back_weight_pipeline, db_bufs[:], &params,
		_div_up(k, BATCHED_MATMUL_LOCAL_X),
		_div_up(n, BATCHED_MATMUL_LOCAL_Y),
		u32(batch_count),
	)
}

permute_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	axes := op.variant.(ml.Permute).axes

	if _permute_pipeline == nil {
		_permute_pipeline = _make_pipeline(PERMUTE_SPIRV, 2, size_of(Permute_Params))
	}
	params := Permute_Params{
		out_d0 = u32(y.shape[0]),
		out_d1 = u32(y.shape[1]),
		out_d2 = u32(y.shape[2]),
		in_d1  = u32(x.shape[1]),
		in_d2  = u32(x.shape[2]),
		axes_0 = u32(axes[0]),
		axes_1 = u32(axes[1]),
		axes_2 = u32(axes[2]),
	}
	bufs := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_permute_pipeline, bufs[:], &params, _div_up(ml.len(y), 256))
}

permute_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	axes := op.variant.(ml.Permute).axes

	if _permute_back_pipeline == nil {
		_permute_back_pipeline = _make_pipeline(PERMUTE_BACK_SPIRV, 2, size_of(Permute_Params))
	}
	params := Permute_Params{
		out_d0 = u32(y.shape[0]),
		out_d1 = u32(y.shape[1]),
		out_d2 = u32(y.shape[2]),
		in_d1  = u32(x.shape[1]),
		in_d2  = u32(x.shape[2]),
		axes_0 = u32(axes[0]),
		axes_1 = u32(axes[1]),
		axes_2 = u32(axes[2]),
	}
	bufs := [2]vk.Buffer{gradient(y).buffer, gradient(x).buffer}
	_dispatch(_permute_back_pipeline, bufs[:], &params, _div_up(ml.len(y), 256))
}

causal_mask_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	T := x.shape[x.rank - 1]
	if _causal_mask_pipeline == nil {
		_causal_mask_pipeline = _make_pipeline(CAUSAL_MASK_SPIRV, 2, size_of(Causal_Mask_Params))
	}
	params := Causal_Mask_Params{total = u32(ml.len(x)), T = u32(T)}
	bufs   := [2]vk.Buffer{data(x).buffer, data(y).buffer}
	_dispatch(_causal_mask_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

causal_mask_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	T := x.shape[x.rank - 1]
	if _causal_mask_back_pipeline == nil {
		_causal_mask_back_pipeline = _make_pipeline(CAUSAL_MASK_BACK_SPIRV, 2, size_of(Causal_Mask_Params))
	}
	params := Causal_Mask_Params{total = u32(ml.len(x)), T = u32(T)}
	bufs   := [2]vk.Buffer{gradient(x).buffer, gradient(y).buffer}
	_dispatch(_causal_mask_back_pipeline, bufs[:], &params, _div_up(ml.len(x), 256))
}

attention_forward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Attention)
	token_count := x.shape[0]
	embed_size  := y.shape[1]
	head_size   := embed_size / v.head_count
	fmt.assertf(head_size <= 256, "GPU attention currently caps head_size at 256 (got %v)", head_size)

	if _attention_pipeline == nil {
		_attention_pipeline = _make_pipeline(ATTENTION_SPIRV, 3, size_of(Attention_Params))
	}
	params := Attention_Params{
		head_count  = u32(v.head_count),
		head_size   = u32(head_size),
		token_count = u32(token_count),
		embed_size  = u32(embed_size),
		causal      = v.causal ? 1 : 0,
	}
	bufs := [3]vk.Buffer{data(x).buffer, data(y).buffer, data(v.lse).buffer}
	_dispatch(_attention_pipeline, bufs[:], &params, u32(v.head_count), u32(token_count))
}

attention_backward :: proc(op: ml.Operation) {
	x := op.input; y := op.output
	v := op.variant.(ml.Attention)
	token_count := x.shape[0]
	embed_size  := y.shape[1]
	head_size   := embed_size / v.head_count

	if _attention_back_d_pipeline == nil {
		_attention_back_d_pipeline = _make_pipeline(ATTENTION_BACK_D_SPIRV, 3, size_of(Attention_Back_D_Params))
	}
	if _attention_back_kv_pipeline == nil {
		_attention_back_kv_pipeline = _make_pipeline(ATTENTION_BACK_KV_SPIRV, 5, size_of(Attention_Params))
	}
	if _attention_back_q_pipeline == nil {
		_attention_back_q_pipeline = _make_pipeline(ATTENTION_BACK_Q_SPIRV, 5, size_of(Attention_Params))
	}

	d_params := Attention_Back_D_Params{
		head_count  = u32(v.head_count),
		head_size   = u32(head_size),
		token_count = u32(token_count),
		embed_size  = u32(embed_size),
	}
	d_bufs := [3]vk.Buffer{data(y).buffer, gradient(y).buffer, data(v.d_acc).buffer}
	_dispatch(_attention_back_d_pipeline, d_bufs[:], &d_params, u32(v.head_count), u32(token_count))

	bk_params := Attention_Params{
		head_count  = u32(v.head_count),
		head_size   = u32(head_size),
		token_count = u32(token_count),
		embed_size  = u32(embed_size),
		causal      = v.causal ? 1 : 0,
	}
	kv_bufs := [5]vk.Buffer{
		data(x).buffer, gradient(y).buffer, data(v.lse).buffer,
		data(v.d_acc).buffer, gradient(x).buffer,
	}
	_dispatch(_attention_back_kv_pipeline, kv_bufs[:], &bk_params, u32(v.head_count), u32(token_count))

	q_bufs := [5]vk.Buffer{
		data(x).buffer, gradient(y).buffer, data(v.lse).buffer,
		data(v.d_acc).buffer, gradient(x).buffer,
	}
	_dispatch(_attention_back_q_pipeline, q_bufs[:], &bk_params, u32(v.head_count), u32(token_count))
}

_upload_indices :: proc(indices: []int, loc := #caller_location) -> (buf: vk.Buffer, m: vk.DeviceMemory) {
	n := builtin.len(indices)
	idx_size := vk.DeviceSize(n * size_of(u32))
	buf, m = _create_buffer(idx_size, {.STORAGE_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT}, loc)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, m, 0, idx_size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(indices) failed: %v", res, loc=loc)
	arr := ([^]u32)(mapped)
	for v, i in indices {
		arr[i] = u32(v)
	}
	vk.UnmapMemory(_gpu.device, m)
	return
}

upload_tensor :: proc(t: ml.Tensor, src: []f32, loc := #caller_location) {
	t.vtable.buffer_set(t.buffers[.Data], src, loc)
}

download_tensor :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	t.vtable.buffer_get(t.buffers[.Data], dst, loc)
}

download_tensor_gradient :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	t.vtable.buffer_get(t.buffers[.Gradient], dst, loc)
}
