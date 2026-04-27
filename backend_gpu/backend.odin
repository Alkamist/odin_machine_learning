// GPU implementation of the ml.Backend interface.
//
// `gpu_alloc` allocates a pair of DEVICE_LOCAL Vulkan buffers (data + grad)
// for each Tensor and stashes a Gpu_Storage handle in `Tensor.storage`.
// `gpu_clear_storage` releases them in bulk on `ml.clear()`, mirroring the
// CPU backend's arena reset.
//
// The forward / backward dispatch procs are stubs at the moment — they
// panic on every op variant. Op kernels migrate from `gpu_transformer/`
// (and the existing GpuTensor-based code in `ops.odin`) into this dispatch
// switch one at a time.
package gpu

import "core:fmt"
import "core:mem"
import vk "vendor:vulkan"

import ml ".."

// Storage held by an `ml.Tensor` whose `backend` is the GPU. ml leaves
// `Tensor.storage` as a rawptr; the GPU backend interprets it as a pointer
// to one of these. All buffers are DEVICE_LOCAL.
//
// Activations populate just `buffer` + `grad_buffer`; parameters populate
// the Adam state too. `_destroy_gpu_storage` only frees populated entries
// so the same struct works for both.
Gpu_Storage :: struct {
	buffer:      vk.Buffer,
	memory:      vk.DeviceMemory,
	grad_buffer: vk.Buffer,
	grad_memory: vk.DeviceMemory,
	count:       int,                // f32 element count

	// Optional per-parameter Adam state. Zero / nil for activations.
	adam_m_buffer: vk.Buffer,
	adam_m_memory: vk.DeviceMemory,
	adam_v_buffer: vk.Buffer,
	adam_v_memory: vk.DeviceMemory,
}

// Singleton backend instance. Wire it into a Context via:
//
//   ctx := ml.context_create(N, gpu.backend())
_gpu_backend := ml.Backend{
	alloc                   = gpu_alloc,
	free                    = gpu_free,
	clear_storage           = gpu_clear_storage,
	set_data                = gpu_set_data,
	get_data                = gpu_get_data,
	parameter_update        = gpu_parameter_update,
	parameter_copy          = gpu_parameter_copy,
	context_alloc           = gpu_context_alloc,
	context_free            = gpu_context_free,
	context_begin           = gpu_context_begin,
	context_end             = gpu_context_end,
	flush                   = gpu_flush,
	fill_gradient_with_ones = gpu_fill_gradient_with_ones,
	forward                 = gpu_forward,
	backward                = gpu_backward,
}

// Get the GPU backend singleton. Lazy-initializes Vulkan on first call —
// callers don't need to `gpu.init()` explicitly. After getting the
// backend, just hand it to `ml.context_create(size, gpu.backend())`;
// the GPU `Gpu_Context` is allocated and bound automatically.
@(require_results)
backend :: proc() -> ^ml.Backend {
	init()
	return &_gpu_backend
}

// Backend.context_alloc: create a fresh Gpu_Context. Stored on the
// owning `ml.Context` as `backend_data`.
gpu_context_alloc :: proc() -> rawptr {
	gctx := context_create()
	return rawptr(gctx)
}

gpu_context_free :: proc(data: rawptr) {
	if data == nil { return }
	context_destroy(cast(^Gpu_Context)data)
}

gpu_context_begin :: proc(data: rawptr) {
	context_begin(cast(^Gpu_Context)data)
}

gpu_context_end :: proc() {
	context_end()
}

// Flush any in-flight batch (submit + queueWaitIdle). Called by `ml.clear`
// before activations are recycled and by `get_data` before reading host
// memory, so callers don't have to manage `begin_batch` / `end_batch`
// manually.
gpu_flush :: proc() {
	gctx := _current_gpu_ctx
	if gctx != nil && gctx.batch.active {
		end_batch()
	}
}

// f32(1.0) bit pattern — vkCmdFillBuffer writes a u32 stamp across the
// buffer, so we hand it the IEEE 754 representation of 1.0.
F32_ONE_BITS :: u32(0x3F800000)

// Allocate `Gpu_Storage` for `t` with `n`-element data + gradient buffers
// and `extra_buffers` additional same-shape buffers (2 → adam_m + adam_v
// for parameters). When `persistent=false`, the allocation is also
// registered on the active gctx's activation list, so `ml.clear()` /
// `clear_storage` recycles it into the per-context pool. When
// `persistent=true`, the storage survives clear and the caller frees via
// `free`.
gpu_alloc :: proc(t: ^ml.Tensor, n: int, persistent: bool, extra_buffers: int) {
	gctx := _current_gpu_ctx
	fmt.assertf(gctx != nil, "no active gpu Context — call gpu.context_begin / context_scope before ml ops on a GPU context")
	fmt.assertf(t.type == .F32, "gpu_alloc: only F32 is supported (got %v)", t.type)
	fmt.assertf(extra_buffers == 0 || extra_buffers == 2, "gpu_alloc: extra_buffers must be 0 or 2 (got %v)", extra_buffers)

	size  := vk.DeviceSize(n * size_of(f32))
	usage := vk.BufferUsageFlags{.STORAGE_BUFFER, .TRANSFER_SRC, .TRANSFER_DST}

	storage: ^Gpu_Storage
	if !persistent {
		// Activation: try to recycle from the pool first. The pool only
		// holds 2-buffer storages (extra_buffers always 0 for activations).
		if list, ok := &gctx.pool[n]; ok && len(list^) > 0 {
			storage = pop(list)
		}
	}
	if storage == nil {
		storage = new(Gpu_Storage)
		storage.count = n
		storage.buffer,      storage.memory      = _create_buffer(size, usage, {.DEVICE_LOCAL})
		storage.grad_buffer, storage.grad_memory = _create_buffer(size, usage, {.DEVICE_LOCAL})
		if extra_buffers >= 2 {
			storage.adam_m_buffer, storage.adam_m_memory = _create_buffer(size, usage, {.DEVICE_LOCAL})
			storage.adam_v_buffer, storage.adam_v_memory = _create_buffer(size, usage, {.DEVICE_LOCAL})
		}
	}

	// Match CPU's `zeros` semantics: every buffer starts at 0. In a batch
	// the fills record into the open CB (free — no extra submit); outside
	// a batch they fall back to one-shot.
	_record_fill_zero(storage.buffer,      size)
	_record_fill_zero(storage.grad_buffer, size)
	if storage.adam_m_buffer != 0 do _record_fill_zero(storage.adam_m_buffer, size)
	if storage.adam_v_buffer != 0 do _record_fill_zero(storage.adam_v_buffer, size)

	if !persistent {
		append(&gctx.allocations, storage)
	}
	t.data = storage
}

gpu_free :: proc(t: ^ml.Tensor) {
	if t.data == nil { return }
	storage := cast(^Gpu_Storage)t.data
	_destroy_gpu_storage(storage)
	t.data = nil
}

gpu_set_data :: proc(t: ^ml.Tensor, src: []f32) {
	upload_tensor(t^, src)
}

gpu_get_data :: proc(t: ^ml.Tensor, dst: []f32) {
	download_tensor(t^, dst)
}

// Adam(W) step + zero gradient on GPU. Push constants pack the optimizer
// state; the shader does one thread per element.
gpu_parameter_update :: proc(opt: ml.Optimizer, p: ^ml.Tensor) {
	storage := cast(^Gpu_Storage)p.data
	fmt.assertf(storage != nil && storage.adam_m_buffer != 0,
		"gpu_parameter_update: parameter has no Adam storage — was it allocated by ml.make under a GPU context?")

	if _adam_step_pipeline == nil {
		_adam_step_pipeline = _make_pipeline(ADAM_STEP_SPIRV, 4, size_of(Adam_Params))
	}
	params := Adam_Params{
		n     = u32(storage.count),
		lr    = opt.learning_rate,
		beta1 = opt.beta1,
		beta2 = opt.beta2,
		eps   = opt.epsilon,
		wd    = opt.weight_decay,
		bc1   = opt.bias_correction1,
		bc2   = opt.bias_correction2,
	}
	bufs := [4]vk.Buffer{
		storage.buffer, storage.grad_buffer,
		storage.adam_m_buffer, storage.adam_v_buffer,
	}
	_dispatch(_adam_step_pipeline, bufs[:], &params, _div_up(storage.count, 256))
}

// Full parameter copy: data + gradient + adam_m + adam_v. Executes as
// four `CmdCopyBuffer`s recorded into the active batch (or one-shot
// each if no batch is active).
gpu_parameter_copy :: proc(dst, src: ^ml.Tensor) {
	dst_s := cast(^Gpu_Storage)dst.data
	src_s := cast(^Gpu_Storage)src.data
	fmt.assertf(dst_s.count == src_s.count, "gpu_parameter_copy size mismatch: dst=%v src=%v", dst_s.count, src_s.count)
	size := vk.DeviceSize(dst_s.count * size_of(f32))

	gctx := _current_gpu_ctx
	if gctx != nil && gctx.batch.active {
		region := vk.BufferCopy{ srcOffset = 0, dstOffset = 0, size = size }
		vk.CmdCopyBuffer(gctx.batch.cmd, src_s.buffer,        dst_s.buffer,        1, &region)
		vk.CmdCopyBuffer(gctx.batch.cmd, src_s.grad_buffer,   dst_s.grad_buffer,   1, &region)
		vk.CmdCopyBuffer(gctx.batch.cmd, src_s.adam_m_buffer, dst_s.adam_m_buffer, 1, &region)
		vk.CmdCopyBuffer(gctx.batch.cmd, src_s.adam_v_buffer, dst_s.adam_v_buffer, 1, &region)
		return
	}
	_one_shot_copy(src_s.buffer,        dst_s.buffer,        size)
	_one_shot_copy(src_s.grad_buffer,   dst_s.grad_buffer,   size)
	_one_shot_copy(src_s.adam_m_buffer, dst_s.adam_m_buffer, size)
	_one_shot_copy(src_s.adam_v_buffer, dst_s.adam_v_buffer, size)
}

gpu_fill_gradient_with_ones :: proc(t: ^ml.Tensor) {
	storage := _storage(t^)
	size    := vk.DeviceSize(storage.count * size_of(f32))
	_record_fill(storage.grad_buffer, size, F32_ONE_BITS)
}

gpu_clear_storage :: proc() {
	gctx := _current_gpu_ctx
	if gctx == nil {
		return
	}

	// Push every live activation back into the pool keyed by element count
	// so the next ml.zeros for the same shape reuses it. Buffers stay alive
	// across `ml.clear` cycles; the only extra work next round is the
	// CmdFillBuffer that re-zeros them, which records into the next batch
	// command buffer.
	for storage in gctx.allocations {
		list, ok := &gctx.pool[storage.count]
		if !ok {
			gctx.pool[storage.count] = make([dynamic]^Gpu_Storage)
			list = &gctx.pool[storage.count]
		}
		append(list, storage)
	}
	clear(&gctx.allocations)
}

gpu_forward :: proc(op: ml.Operation) {
	switch _ in op.variant {
	case ml.Add:            gpu_add_forward            (op)
	case ml.Mul:            gpu_mul_forward            (op)
	case ml.Linear:         gpu_linear_forward         (op)
	case ml.Gelu:           gpu_gelu_forward           (op)
	case ml.Select:         gpu_select_forward         (op)
	case ml.Rope:           gpu_rope_forward           (op)
	case ml.Slice_Trailing: gpu_slice_trailing_forward (op)
	case ml.Concat:         gpu_concat_forward         (op)
	case ml.Softmax:        gpu_softmax_forward        (op)
	case ml.Permute:        gpu_permute_forward        (op)
	case ml.Causal_Mask:    gpu_causal_mask_forward    (op)
	case ml.Batched_Matmul: gpu_batched_matmul_forward (op)
	case ml.Layernorm:      gpu_layernorm_forward      (op)
	case ml.Cross_Entropy:  gpu_cross_entropy_forward  (op)
	case ml.Mean:           gpu_mean_forward           (op)
	case ml.Relu:           gpu_relu_forward           (op)
	case ml.Sigmoid:        gpu_sigmoid_forward        (op)
	case ml.Silu:           gpu_silu_forward           (op)
	case ml.Tanh:           gpu_tanh_forward           (op)
	case ml.Exp:            gpu_exp_forward            (op)
	case ml.Clamp:          gpu_clamp_forward          (op)
	case ml.Min:            gpu_min_forward            (op)
	case ml.Max:            gpu_max_forward            (op)
	case ml.Sub:            gpu_sub_forward            (op)
	case ml.Div:            gpu_div_forward            (op)
	case ml.Transpose:      gpu_transpose_forward      (op)
	case ml.Slice:          gpu_slice_forward          (op)
	case ml.Log_Softmax:    gpu_log_softmax_forward    (op)
	case ml.Entropy:        gpu_entropy_forward        (op)
	case ml.Mean_Squared_Error: gpu_mean_squared_error_forward (op)
	}
}

gpu_backward :: proc(op: ml.Operation) {
	switch _ in op.variant {
	case ml.Add:            gpu_add_backward            (op)
	case ml.Mul:            gpu_mul_backward            (op)
	case ml.Linear:         gpu_linear_backward         (op)
	case ml.Gelu:           gpu_gelu_backward           (op)
	case ml.Select:         gpu_select_backward         (op)
	case ml.Rope:           gpu_rope_backward           (op)
	case ml.Slice_Trailing: gpu_slice_trailing_backward (op)
	case ml.Concat:         gpu_concat_backward         (op)
	case ml.Softmax:        gpu_softmax_backward        (op)
	case ml.Permute:        gpu_permute_backward        (op)
	case ml.Causal_Mask:    gpu_causal_mask_backward    (op)
	case ml.Batched_Matmul: gpu_batched_matmul_backward (op)
	case ml.Layernorm:      gpu_layernorm_backward      (op)
	case ml.Cross_Entropy:  gpu_cross_entropy_backward  (op)
	case ml.Mean:           gpu_mean_backward           (op)
	case ml.Relu:           gpu_relu_backward           (op)
	case ml.Sigmoid:        gpu_sigmoid_backward        (op)
	case ml.Silu:           gpu_silu_backward           (op)
	case ml.Tanh:           gpu_tanh_backward           (op)
	case ml.Exp:            gpu_exp_backward            (op)
	case ml.Clamp:          gpu_clamp_backward          (op)
	case ml.Min:            gpu_min_backward            (op)
	case ml.Max:            gpu_max_backward            (op)
	case ml.Sub:            gpu_sub_backward            (op)
	case ml.Div:            gpu_div_backward            (op)
	case ml.Transpose:      gpu_transpose_backward      (op)
	case ml.Slice:          gpu_slice_backward          (op)
	case ml.Log_Softmax:    gpu_log_softmax_backward    (op)
	case ml.Entropy:        gpu_entropy_backward        (op)
	case ml.Mean_Squared_Error: gpu_mean_squared_error_backward (op)
	}
}

// add: out = a + b. Same-shape only on the GPU backend for now —
// broadcasting (variant.stride > 1) needs a different shader.
gpu_add_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Add)
	b       := variant.b

	fmt.assertf(variant.stride == 1, "gpu add: broadcasting (stride=%v) not yet supported on GPU backend", variant.stride)

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	out_storage := _storage(output)

	if _add_pipeline == nil {
		_add_pipeline = _make_pipeline(ADD_SPIRV, 3, size_of(Add_Params))
	}
	params := Add_Params{ n = u32(a_storage.count) }
	bufs   := [3]vk.Buffer{ a_storage.buffer, b_storage.buffer, out_storage.buffer }
	_dispatch(_add_pipeline, bufs[:], &params, _div_up(a_storage.count, ADD_LOCAL_SIZE))
}

// add backward: a.gradient += output.gradient; b.gradient += output.gradient.
gpu_add_backward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Add)
	b       := variant.b

	fmt.assertf(variant.stride == 1, "gpu add backward: broadcasting (stride=%v) not yet supported on GPU backend", variant.stride)

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	out_storage := _storage(output)

	if _add_back_pipeline == nil {
		_add_back_pipeline = _make_pipeline(ADD_BACK_SPIRV, 3, size_of(Add_Back_Params))
	}
	params := Add_Back_Params{ n = u32(out_storage.count) }
	bufs   := [3]vk.Buffer{ a_storage.grad_buffer, b_storage.grad_buffer, out_storage.grad_buffer }
	_dispatch(_add_back_pipeline, bufs[:], &params, _div_up(out_storage.count, 256))
}

// mul: out = a * b with broadcast. b's length must divide a's length;
// stride = len(a) / len(b) repeats b across the leading axis.
gpu_mul_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Mul)
	b       := variant.b

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	out_storage := _storage(output)

	if _mul_pipeline == nil {
		_mul_pipeline = _make_pipeline(MUL_SPIRV, 3, size_of(Mul_Params))
	}
	params := Mul_Params{ n = u32(a_storage.count), n_b = u32(b_storage.count) }
	bufs   := [3]vk.Buffer{ a_storage.buffer, b_storage.buffer, out_storage.buffer }
	_dispatch(_mul_pipeline, bufs[:], &params, _div_up(a_storage.count, 256))
}

// mul backward: a.grad[o] += dy[o] * b[o%n_b];  b.grad[j] += sum_i dy[i*n_b+j]*a[i*n_b+j].
// Two kernels: per-output for a (no race), per-b-element with stride
// reduction for b (no race).
gpu_mul_backward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Mul)
	b       := variant.b

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	out_storage := _storage(output)

	if _mul_back_a_pipeline == nil {
		_mul_back_a_pipeline = _make_pipeline(MUL_BACK_A_SPIRV, 3, size_of(Mul_Back_A_Params))
	}
	if _mul_back_b_pipeline == nil {
		_mul_back_b_pipeline = _make_pipeline(MUL_BACK_B_SPIRV, 3, size_of(Mul_Back_B_Params))
	}

	a_params := Mul_Back_A_Params{ n = u32(a_storage.count), n_b = u32(b_storage.count) }
	a_bufs   := [3]vk.Buffer{ b_storage.buffer, out_storage.grad_buffer, a_storage.grad_buffer }
	_dispatch(_mul_back_a_pipeline, a_bufs[:], &a_params, _div_up(a_storage.count, 256))

	b_params := Mul_Back_B_Params{ n_b = u32(b_storage.count), stride = u32(variant.stride) }
	b_bufs   := [3]vk.Buffer{ a_storage.buffer, out_storage.grad_buffer, b_storage.grad_buffer }
	_dispatch(_mul_back_b_pipeline, b_bufs[:], &b_params, _div_up(b_storage.count, 256))
}

// linear: y = x · W^T. Both forward and backward route through the
// existing tiled-GEMM shaders.
gpu_linear_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Linear)
	weight  := variant.weight

	x_storage := _storage(input)
	w_storage := _storage(weight)
	y_storage := _storage(output)

	if _linear_pipeline == nil {
		_linear_pipeline = _make_pipeline(LINEAR_SPIRV, 3, size_of(Linear_Params))
	}
	params := Linear_Params{
		count       = u32(variant.count),
		input_size  = u32(variant.input_size),
		output_size = u32(variant.output_size),
	}
	bufs := [3]vk.Buffer{ x_storage.buffer, w_storage.buffer, y_storage.buffer }
	_dispatch(
		_linear_pipeline,
		bufs[:],
		&params,
		_div_up(variant.count,       LINEAR_LOCAL_X),
		_div_up(variant.output_size, LINEAR_LOCAL_Y),
		1,
	)
}

gpu_linear_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Linear)
	weight  := variant.weight

	x_storage  := _storage(input)
	w_storage  := _storage(weight)
	dy_storage := _storage(output)

	if _linear_back_input_pipeline == nil {
		_linear_back_input_pipeline = _make_pipeline(LINEAR_BACK_INPUT_SPIRV, 3, size_of(Linear_Back_Params))
	}
	if _linear_back_weight_pipeline == nil {
		_linear_back_weight_pipeline = _make_pipeline(LINEAR_BACK_WEIGHT_SPIRV, 3, size_of(Linear_Back_Params))
	}
	params := Linear_Back_Params{
		count       = u32(variant.count),
		input_size  = u32(variant.input_size),
		output_size = u32(variant.output_size),
	}

	// dx[c, k] += sum_o W[o, k] * dy[c, o]
	dx_bufs := [3]vk.Buffer{ dy_storage.grad_buffer, w_storage.buffer, x_storage.grad_buffer }
	_dispatch(
		_linear_back_input_pipeline,
		dx_bufs[:],
		&params,
		_div_up(variant.count,      16),
		_div_up(variant.input_size, 16),
		1,
	)

	// dW[o, k] += sum_c x[c, k] * dy[c, o]
	dw_bufs := [3]vk.Buffer{ x_storage.buffer, dy_storage.grad_buffer, w_storage.grad_buffer }
	_dispatch(
		_linear_back_weight_pipeline,
		dw_bufs[:],
		&params,
		_div_up(variant.output_size, 16),
		_div_up(variant.input_size,  16),
		1,
	)
}

// gelu: elementwise tanh-approximation; backward reads the original input
// (saved on op.input.data buffer) and the upstream gradient.
gpu_gelu_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	x_storage := _storage(input)
	y_storage := _storage(output)

	if _gelu_pipeline == nil {
		_gelu_pipeline = _make_pipeline(GELU_SPIRV, 2, size_of(Gelu_Params))
	}
	params := Gelu_Params{ n = u32(x_storage.count) }
	bufs   := [2]vk.Buffer{ x_storage.buffer, y_storage.buffer }
	_dispatch(_gelu_pipeline, bufs[:], &params, _div_up(x_storage.count, GELU_LOCAL_SIZE))
}

gpu_gelu_backward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	x_storage := _storage(input)
	y_storage := _storage(output)

	if _gelu_back_pipeline == nil {
		_gelu_back_pipeline = _make_pipeline(GELU_BACK_SPIRV, 3, size_of(Gelu_Back_Params))
	}
	params := Gelu_Back_Params{ n = u32(x_storage.count) }
	bufs   := [3]vk.Buffer{ x_storage.buffer, x_storage.grad_buffer, y_storage.grad_buffer }
	_dispatch(_gelu_back_pipeline, bufs[:], &params, _div_up(x_storage.count, 256))
}

// select: out[i, j] = table[indices[i], j]. The indices array lives on the
// op's variant (a CPU-side []int); we upload it into a transient
// host-visible buffer per call.
gpu_select_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Select)
	indices := variant.indices
	size    := variant.size

	in_storage  := _storage(input)
	out_storage := _storage(output)

	idx_buf, idx_mem := _upload_indices(indices)

	if _select_pipeline == nil {
		_select_pipeline = _make_pipeline(SELECT_SPIRV, 3, size_of(Select_Params))
	}
	params := Select_Params{ n_indices = u32(len(indices)), size = u32(size) }
	bufs   := [3]vk.Buffer{ in_storage.buffer, idx_buf, out_storage.buffer }
	_dispatch(_select_pipeline, bufs[:], &params, _div_up(size, 256), u32(len(indices)), 1)

	_queue_destroy_buffer(idx_buf, idx_mem)
}

gpu_select_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Select)
	indices := variant.indices
	size    := variant.size
	vocab   := input.shape[0]

	in_storage  := _storage(input)
	out_storage := _storage(output)

	idx_buf, idx_mem := _upload_indices(indices)

	if _select_back_pipeline == nil {
		_select_back_pipeline = _make_pipeline(SELECT_BACK_SPIRV, 3, size_of(Select_Back_Params))
	}
	params := Select_Back_Params{
		vocab     = u32(vocab),
		n_indices = u32(len(indices)),
		size      = u32(size),
	}
	bufs := [3]vk.Buffer{ idx_buf, out_storage.grad_buffer, in_storage.grad_buffer }
	_dispatch(_select_back_pipeline, bufs[:], &params, _div_up(vocab, 16), _div_up(size, 16), 1)

	_queue_destroy_buffer(idx_buf, idx_mem)
}

// rope: rotary position embedding. The cos/sin caches in the op variant
// stay on whatever backend allocated them (this backend leaves them
// untouched — the GPU shader recomputes cos/sin inline).
gpu_rope_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Rope)

	in_storage  := _storage(input)
	out_storage := _storage(output)

	if _rope_pipeline == nil {
		_rope_pipeline = _make_pipeline(ROPE_SPIRV, 2, size_of(Rope_Params))
	}
	params := Rope_Params{
		token_count = u32(variant.token_count),
		head_count  = u32(variant.head_count),
		head_size   = u32(variant.head_size),
		base        = variant.base,
	}
	bufs        := [2]vk.Buffer{ in_storage.buffer, out_storage.buffer }
	total_pairs := variant.token_count * variant.head_count * (variant.head_size / 2)
	_dispatch(_rope_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
}

gpu_rope_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Rope)

	in_storage  := _storage(input)
	out_storage := _storage(output)

	if _rope_back_pipeline == nil {
		_rope_back_pipeline = _make_pipeline(ROPE_BACK_SPIRV, 2, size_of(Rope_Back_Params))
	}
	params := Rope_Back_Params{
		token_count = u32(variant.token_count),
		head_count  = u32(variant.head_count),
		head_size   = u32(variant.head_size),
		base        = variant.base,
	}
	bufs        := [2]vk.Buffer{ in_storage.grad_buffer, out_storage.grad_buffer }
	total_pairs := variant.token_count * variant.head_count * (variant.head_size / 2)
	_dispatch(_rope_back_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
}

// slice_trailing: out[r, i] = input[r, start + i] for i in [0, end - start).
gpu_slice_trailing_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Slice_Trailing)
	start   := variant.start

	in_storage  := _storage(input)
	out_storage := _storage(output)

	trailing     := input.shape [input.rank  - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := in_storage.count / trailing

	if _slice_trailing_pipeline == nil {
		_slice_trailing_pipeline = _make_pipeline(SLICE_TRAILING_SPIRV, 2, size_of(Slice_Trailing_Params))
	}
	params := Slice_Trailing_Params{
		leading      = u32(leading),
		trailing     = u32(trailing),
		new_trailing = u32(new_trailing),
		start        = u32(start),
	}
	bufs := [2]vk.Buffer{ in_storage.buffer, out_storage.buffer }
	_dispatch(_slice_trailing_pipeline, bufs[:], &params, _div_up(leading * new_trailing, 256))
}

gpu_slice_trailing_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Slice_Trailing)
	start   := variant.start

	in_storage  := _storage(input)
	out_storage := _storage(output)

	trailing     := input.shape [input.rank  - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := in_storage.count / trailing

	if _slice_trailing_back_pipeline == nil {
		_slice_trailing_back_pipeline = _make_pipeline(SLICE_TRAILING_BACK_SPIRV, 2, size_of(Slice_Trailing_Back_Params))
	}
	params := Slice_Trailing_Back_Params{
		leading      = u32(leading),
		trailing     = u32(trailing),
		new_trailing = u32(new_trailing),
		start        = u32(start),
	}
	bufs := [2]vk.Buffer{ in_storage.grad_buffer, out_storage.grad_buffer }
	_dispatch(_slice_trailing_back_pipeline, bufs[:], &params, _div_up(leading * new_trailing, 256))
}

// concat: trailing-dim concat. GPU only supports exactly 3 inputs (the
// transformer's QKV-build use case) — the existing concat3 shader. Other
// arities will need their own kernels or a generic dispatch.
gpu_concat_forward :: proc(op: ml.Operation) {
	output  := op.output
	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	fmt.assertf(len(inputs) == 3, "GPU concat only supports 3 inputs (got %v)", len(inputs))
	a, b, c := inputs[0], inputs[1], inputs[2]

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	c_storage   := _storage(c)
	out_storage := _storage(output)

	t_a     := a.shape[a.rank - 1]
	t_b     := b.shape[b.rank - 1]
	t_c     := c.shape[c.rank - 1]
	leading := a_storage.count / t_a

	if _concat3_pipeline == nil {
		_concat3_pipeline = _make_pipeline(CONCAT3_SPIRV, 4, size_of(Concat3_Params))
	}
	params := Concat3_Params{
		leading = u32(leading),
		t_a     = u32(t_a),
		t_b     = u32(t_b),
		t_c     = u32(t_c),
	}
	bufs  := [4]vk.Buffer{ a_storage.buffer, b_storage.buffer, c_storage.buffer, out_storage.buffer }
	total := leading * (t_a + t_b + t_c)
	_dispatch(_concat3_pipeline, bufs[:], &params, _div_up(total, 256))
}

gpu_concat_backward :: proc(op: ml.Operation) {
	output  := op.output
	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	fmt.assertf(len(inputs) == 3, "GPU concat only supports 3 inputs (got %v)", len(inputs))
	a, b, c := inputs[0], inputs[1], inputs[2]

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	c_storage   := _storage(c)
	out_storage := _storage(output)

	t_a     := a.shape[a.rank - 1]
	t_b     := b.shape[b.rank - 1]
	t_c     := c.shape[c.rank - 1]
	leading := a_storage.count / t_a

	if _concat3_back_pipeline == nil {
		_concat3_back_pipeline = _make_pipeline(CONCAT3_BACK_SPIRV, 4, size_of(Concat3_Back_Params))
	}
	params := Concat3_Back_Params{
		leading = u32(leading),
		t_a     = u32(t_a),
		t_b     = u32(t_b),
		t_c     = u32(t_c),
	}
	bufs  := [4]vk.Buffer{ a_storage.grad_buffer, b_storage.grad_buffer, c_storage.grad_buffer, out_storage.grad_buffer }
	total := leading * (t_a + t_b + t_c)
	_dispatch(_concat3_back_pipeline, bufs[:], &params, _div_up(total, 256))
}

// softmax: per-row softmax along the trailing dim. Forward existing kernel
// is one-workgroup-per-row; backward is the same shape, plus a reduction.
gpu_softmax_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	x_storage := _storage(input)
	y_storage := _storage(output)

	size  := input.shape[input.rank - 1]
	count := x_storage.count / size

	if _softmax_pipeline == nil {
		_softmax_pipeline = _make_pipeline(SOFTMAX_SPIRV, 2, size_of(Softmax_Params))
	}
	params := Softmax_Params{ count = u32(count), size = u32(size) }
	bufs   := [2]vk.Buffer{ x_storage.buffer, y_storage.buffer }
	_dispatch(_softmax_pipeline, bufs[:], &params, u32(count))
}

gpu_softmax_backward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	x_storage := _storage(input)
	y_storage := _storage(output)

	size  := input.shape[input.rank - 1]
	count := x_storage.count / size

	if _softmax_back_pipeline == nil {
		_softmax_back_pipeline = _make_pipeline(SOFTMAX_BACK_SPIRV, 3, size_of(Softmax_Back_Params))
	}
	params := Softmax_Back_Params{ count = u32(count), size = u32(size) }
	bufs   := [3]vk.Buffer{ y_storage.buffer, y_storage.grad_buffer, x_storage.grad_buffer }
	_dispatch(_softmax_back_pipeline, bufs[:], &params, u32(count))
}

// permute: 3-D axis reorder. One thread per output element, looks up its
// source via the axis mapping passed via push constants.
gpu_permute_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	axes    := op.variant.(ml.Permute).axes

	in_storage  := _storage(input)
	out_storage := _storage(output)

	if _permute_pipeline == nil {
		_permute_pipeline = _make_pipeline(PERMUTE_SPIRV, 2, size_of(Permute_Params))
	}
	params := Permute_Params{
		out_d0 = u32(output.shape[0]),
		out_d1 = u32(output.shape[1]),
		out_d2 = u32(output.shape[2]),
		in_d1  = u32(input.shape[1]),
		in_d2  = u32(input.shape[2]),
		axes_0 = u32(axes[0]),
		axes_1 = u32(axes[1]),
		axes_2 = u32(axes[2]),
	}
	bufs := [2]vk.Buffer{ in_storage.buffer, out_storage.buffer }
	_dispatch(_permute_pipeline, bufs[:], &params, _div_up(out_storage.count, 256))
}

gpu_permute_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	axes    := op.variant.(ml.Permute).axes

	in_storage  := _storage(input)
	out_storage := _storage(output)

	if _permute_back_pipeline == nil {
		_permute_back_pipeline = _make_pipeline(PERMUTE_BACK_SPIRV, 2, size_of(Permute_Params))
	}
	params := Permute_Params{
		out_d0 = u32(output.shape[0]),
		out_d1 = u32(output.shape[1]),
		out_d2 = u32(output.shape[2]),
		in_d1  = u32(input.shape[1]),
		in_d2  = u32(input.shape[2]),
		axes_0 = u32(axes[0]),
		axes_1 = u32(axes[1]),
		axes_2 = u32(axes[2]),
	}
	bufs := [2]vk.Buffer{ out_storage.grad_buffer, in_storage.grad_buffer }
	_dispatch(_permute_back_pipeline, bufs[:], &params, _div_up(out_storage.count, 256))
}

// causal_mask: replace upper-triangle of trailing [T, T] with -inf; backward
// is identity for unmasked positions, blocked for masked.
gpu_causal_mask_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	x_storage := _storage(input)
	y_storage := _storage(output)

	T := input.shape[input.rank - 1]

	if _causal_mask_pipeline == nil {
		_causal_mask_pipeline = _make_pipeline(CAUSAL_MASK_SPIRV, 2, size_of(Causal_Mask_Params))
	}
	params := Causal_Mask_Params{ total = u32(x_storage.count), T = u32(T) }
	bufs   := [2]vk.Buffer{ x_storage.buffer, y_storage.buffer }
	_dispatch(_causal_mask_pipeline, bufs[:], &params, _div_up(x_storage.count, 256))
}

gpu_causal_mask_backward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	x_storage := _storage(input)
	y_storage := _storage(output)

	T := input.shape[input.rank - 1]

	if _causal_mask_back_pipeline == nil {
		_causal_mask_back_pipeline = _make_pipeline(CAUSAL_MASK_BACK_SPIRV, 2, size_of(Causal_Mask_Params))
	}
	params := Causal_Mask_Params{ total = u32(x_storage.count), T = u32(T) }
	bufs   := [2]vk.Buffer{ x_storage.grad_buffer, y_storage.grad_buffer }
	_dispatch(_causal_mask_back_pipeline, bufs[:], &params, _div_up(x_storage.count, 256))
}

// batched_matmul: C[bi, i, j] = sum_k A[bi, i, k] * B[bi, k, j]. Rank-3
// inputs / output. Dispatch is per-output-element with a k reduction in
// the inner loop; backward splits into a back-input and back-weight kernel
// like `linear`.
gpu_batched_matmul_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Batched_Matmul)
	b       := variant.b

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	out_storage := _storage(output)

	if _batched_matmul_pipeline == nil {
		_batched_matmul_pipeline = _make_pipeline(BATCHED_MATMUL_SPIRV, 3, size_of(Batched_Matmul_Params))
	}
	params := Batched_Matmul_Params{
		batch_count = u32(variant.batch_count),
		m           = u32(variant.m),
		k           = u32(variant.k),
		n           = u32(variant.n),
	}
	bufs := [3]vk.Buffer{ a_storage.buffer, b_storage.buffer, out_storage.buffer }
	_dispatch(
		_batched_matmul_pipeline,
		bufs[:],
		&params,
		_div_up(variant.m, BATCHED_MATMUL_LOCAL_X),
		_div_up(variant.n, BATCHED_MATMUL_LOCAL_Y),
		u32(variant.batch_count),
	)
}

gpu_batched_matmul_backward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Batched_Matmul)
	b       := variant.b

	a_storage   := _storage(a)
	b_storage   := _storage(b)
	out_storage := _storage(output)

	if _batched_matmul_back_input_pipeline == nil {
		_batched_matmul_back_input_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_INPUT_SPIRV, 3, size_of(Batched_Matmul_Params))
	}
	if _batched_matmul_back_weight_pipeline == nil {
		_batched_matmul_back_weight_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_WEIGHT_SPIRV, 3, size_of(Batched_Matmul_Params))
	}
	params := Batched_Matmul_Params{
		batch_count = u32(variant.batch_count),
		m           = u32(variant.m),
		k           = u32(variant.k),
		n           = u32(variant.n),
	}

	// dA[bi, i, k] += sum_j dC[bi, i, j] * B[bi, k, j]
	da_bufs := [3]vk.Buffer{ out_storage.grad_buffer, b_storage.buffer, a_storage.grad_buffer }
	_dispatch(
		_batched_matmul_back_input_pipeline,
		da_bufs[:],
		&params,
		_div_up(variant.m, BATCHED_MATMUL_LOCAL_X),
		_div_up(variant.k, BATCHED_MATMUL_LOCAL_Y),
		u32(variant.batch_count),
	)

	// dB[bi, k, j] += sum_i A[bi, i, k] * dC[bi, i, j]
	db_bufs := [3]vk.Buffer{ a_storage.buffer, out_storage.grad_buffer, b_storage.grad_buffer }
	_dispatch(
		_batched_matmul_back_weight_pipeline,
		db_bufs[:],
		&params,
		_div_up(variant.k, BATCHED_MATMUL_LOCAL_X),
		_div_up(variant.n, BATCHED_MATMUL_LOCAL_Y),
		u32(variant.batch_count),
	)
}

// layernorm: per-row normalize along the trailing dim and apply per-feature
// gain. Forward dispatches `layernorm_stats` (writes mean/rstd) then
// `layernorm` (writes y); backward reads those stats. CPU bakes mean/rstd
// into the op variant's tensors; on GPU those tensors are GPU-resident
// buffers populated by `layernorm_stats`.
gpu_layernorm_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd

	x_storage    := _storage(input)
	w_storage    := _storage(weight)
	y_storage    := _storage(output)
	mean_storage := _storage(mean)
	rstd_storage := _storage(rstd)

	if _layernorm_stats_pipeline == nil {
		_layernorm_stats_pipeline = _make_pipeline(LAYERNORM_STATS_SPIRV, 3, size_of(Layernorm_Stats_Params))
	}
	if _layernorm_pipeline == nil {
		_layernorm_pipeline = _make_pipeline(LAYERNORM_SPIRV, 3, size_of(Layernorm_Params))
	}

	stats_params := Layernorm_Stats_Params{ count = u32(variant.count), size = u32(variant.size) }
	stats_bufs   := [3]vk.Buffer{ x_storage.buffer, mean_storage.buffer, rstd_storage.buffer }
	_dispatch(_layernorm_stats_pipeline, stats_bufs[:], &stats_params, u32(variant.count))

	fwd_params := Layernorm_Params{ count = u32(variant.count), size = u32(variant.size) }
	fwd_bufs   := [3]vk.Buffer{ x_storage.buffer, w_storage.buffer, y_storage.buffer }
	_dispatch(_layernorm_pipeline, fwd_bufs[:], &fwd_params, u32(variant.count))
}

gpu_layernorm_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd

	x_storage    := _storage(input)
	w_storage    := _storage(weight)
	y_storage    := _storage(output)
	mean_storage := _storage(mean)
	rstd_storage := _storage(rstd)

	if _layernorm_back_input_pipeline == nil {
		_layernorm_back_input_pipeline = _make_pipeline(LAYERNORM_BACK_INPUT_SPIRV, 6, size_of(Layernorm_Back_Params))
	}
	if _layernorm_back_weight_pipeline == nil {
		_layernorm_back_weight_pipeline = _make_pipeline(LAYERNORM_BACK_WEIGHT_SPIRV, 5, size_of(Layernorm_Back_Params))
	}

	back_params := Layernorm_Back_Params{ count = u32(variant.count), size = u32(variant.size) }

	in_bufs := [6]vk.Buffer{
		x_storage.buffer, w_storage.buffer, y_storage.grad_buffer,
		mean_storage.buffer, rstd_storage.buffer, x_storage.grad_buffer,
	}
	_dispatch(_layernorm_back_input_pipeline, in_bufs[:], &back_params, u32(variant.count))

	w_bufs := [5]vk.Buffer{
		x_storage.buffer, y_storage.grad_buffer,
		mean_storage.buffer, rstd_storage.buffer, w_storage.grad_buffer,
	}
	_dispatch(_layernorm_back_weight_pipeline, w_bufs[:], &back_params, _div_up(variant.size, 256))
}

// cross_entropy: softmax + per-sample NLL. Forward writes probabilities
// (held on the op variant) and per-sample loss; backward accumulates
// dx[idx] = (prob - one_hot) * dy[sample]. Targets are uploaded as u32
// into a transient HOST_VISIBLE buffer per call (same pattern as
// `select`'s indices).
gpu_cross_entropy_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Cross_Entropy)
	probs   := variant.probabilities
	targets := variant.targets

	x_storage   := _storage(input)
	p_storage   := _storage(probs)
	out_storage := _storage(output)

	tgt_buf, tgt_mem := _upload_indices(targets)

	if _cross_entropy_pipeline == nil {
		_cross_entropy_pipeline = _make_pipeline(CROSS_ENTROPY_SPIRV, 4, size_of(Cross_Entropy_Params))
	}
	params := Cross_Entropy_Params{
		count      = u32(len(targets)),
		class_size = u32(variant.class_size),
	}
	bufs := [4]vk.Buffer{ x_storage.buffer, tgt_buf, p_storage.buffer, out_storage.buffer }
	_dispatch(_cross_entropy_pipeline, bufs[:], &params, u32(len(targets)))

	_queue_destroy_buffer(tgt_buf, tgt_mem)
}

gpu_cross_entropy_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Cross_Entropy)
	probs   := variant.probabilities
	targets := variant.targets

	x_storage   := _storage(input)
	p_storage   := _storage(probs)
	out_storage := _storage(output)

	tgt_buf, tgt_mem := _upload_indices(targets)

	if _cross_entropy_back_pipeline == nil {
		_cross_entropy_back_pipeline = _make_pipeline(CROSS_ENTROPY_BACK_SPIRV, 4, size_of(Cross_Entropy_Params))
	}
	params := Cross_Entropy_Params{
		count      = u32(len(targets)),
		class_size = u32(variant.class_size),
	}
	bufs  := [4]vk.Buffer{ p_storage.buffer, tgt_buf, out_storage.grad_buffer, x_storage.grad_buffer }
	total := len(targets) * variant.class_size
	_dispatch(_cross_entropy_back_pipeline, bufs[:], &params, _div_up(total, 256))

	_queue_destroy_buffer(tgt_buf, tgt_mem)
}

// mean: y[row] = (1/size) * sum_i x[row*size + i]. Workgroup per row +
// shared-memory reduction for forward; thread-per-input element for
// backward (gradient is uniform across the input row).
gpu_mean_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Mean)

	x_storage := _storage(input)
	y_storage := _storage(output)

	if _mean_pipeline == nil {
		_mean_pipeline = _make_pipeline(MEAN_SPIRV, 2, size_of(Mean_Params))
	}
	params := Mean_Params{ count = u32(variant.count), size = u32(variant.size) }
	bufs   := [2]vk.Buffer{ x_storage.buffer, y_storage.buffer }
	_dispatch(_mean_pipeline, bufs[:], &params, u32(variant.count))
}

gpu_mean_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Mean)

	x_storage := _storage(input)
	y_storage := _storage(output)

	if _mean_back_pipeline == nil {
		_mean_back_pipeline = _make_pipeline(MEAN_BACK_SPIRV, 2, size_of(Mean_Params))
	}
	params := Mean_Params{ count = u32(variant.count), size = u32(variant.size) }
	bufs   := [2]vk.Buffer{ y_storage.grad_buffer, x_storage.grad_buffer }
	total  := variant.count * variant.size
	_dispatch(_mean_back_pipeline, bufs[:], &params, _div_up(total, 256))
}

// --- Elementwise activations ---

gpu_relu_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	if _relu_pipeline == nil { _relu_pipeline = _make_pipeline(RELU_SPIRV, 2, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_relu_pipeline, bufs[:], &params, _div_up(x.count, 256))
}
gpu_relu_backward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	if _relu_back_pipeline == nil { _relu_back_pipeline = _make_pipeline(RELU_BACK_SPIRV, 3, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [3]vk.Buffer{ x.buffer, y.grad_buffer, x.grad_buffer }
	_dispatch(_relu_back_pipeline, bufs[:], &params, _div_up(x.count, 256))
}

gpu_sigmoid_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	if _sigmoid_pipeline == nil { _sigmoid_pipeline = _make_pipeline(SIGMOID_SPIRV, 2, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_sigmoid_pipeline, bufs[:], &params, _div_up(x.count, 256))
}
gpu_sigmoid_backward :: proc(op: ml.Operation) {
	// reads y (output), dy, writes dx — matches CPU which uses sigmoid_value = output.data[i]
	x := _storage(op.input); y := _storage(op.output)
	if _sigmoid_back_pipeline == nil { _sigmoid_back_pipeline = _make_pipeline(SIGMOID_BACK_SPIRV, 3, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [3]vk.Buffer{ y.buffer, y.grad_buffer, x.grad_buffer }
	_dispatch(_sigmoid_back_pipeline, bufs[:], &params, _div_up(x.count, 256))
}

gpu_silu_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	if _silu_pipeline == nil { _silu_pipeline = _make_pipeline(SILU_SPIRV, 2, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_silu_pipeline, bufs[:], &params, _div_up(x.count, 256))
}
gpu_silu_backward :: proc(op: ml.Operation) {
	// reads x (input), dy, writes dx — silu' depends on sigmoid(x) which the shader recomputes
	x := _storage(op.input); y := _storage(op.output)
	if _silu_back_pipeline == nil { _silu_back_pipeline = _make_pipeline(SILU_BACK_SPIRV, 3, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [3]vk.Buffer{ x.buffer, y.grad_buffer, x.grad_buffer }
	_dispatch(_silu_back_pipeline, bufs[:], &params, _div_up(x.count, 256))
}

gpu_tanh_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	if _tanh_pipeline == nil { _tanh_pipeline = _make_pipeline(TANH_SPIRV, 2, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_tanh_pipeline, bufs[:], &params, _div_up(x.count, 256))
}
gpu_tanh_backward :: proc(op: ml.Operation) {
	// reads y (output), dy — tanh' = 1 - y^2
	x := _storage(op.input); y := _storage(op.output)
	if _tanh_back_pipeline == nil { _tanh_back_pipeline = _make_pipeline(TANH_BACK_SPIRV, 3, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [3]vk.Buffer{ y.buffer, y.grad_buffer, x.grad_buffer }
	_dispatch(_tanh_back_pipeline, bufs[:], &params, _div_up(x.count, 256))
}

gpu_exp_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	if _exp_pipeline == nil { _exp_pipeline = _make_pipeline(EXP_SPIRV, 2, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_exp_pipeline, bufs[:], &params, _div_up(x.count, 256))
}
gpu_exp_backward :: proc(op: ml.Operation) {
	// reads y (output) — exp' = exp(x) = y
	x := _storage(op.input); y := _storage(op.output)
	if _exp_back_pipeline == nil { _exp_back_pipeline = _make_pipeline(EXP_BACK_SPIRV, 3, size_of(Activation_Params)) }
	params := Activation_Params{ n = u32(x.count) }
	bufs   := [3]vk.Buffer{ y.buffer, y.grad_buffer, x.grad_buffer }
	_dispatch(_exp_back_pipeline, bufs[:], &params, _div_up(x.count, 256))
}

gpu_clamp_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Clamp)
	if _clamp_pipeline == nil { _clamp_pipeline = _make_pipeline(CLAMP_SPIRV, 2, size_of(Clamp_Params)) }
	params := Clamp_Params{ n = u32(x.count), min_val = v.min_val, max_val = v.max_val }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_clamp_pipeline, bufs[:], &params, _div_up(x.count, 256))
}
gpu_clamp_backward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Clamp)
	if _clamp_back_pipeline == nil { _clamp_back_pipeline = _make_pipeline(CLAMP_BACK_SPIRV, 3, size_of(Clamp_Params)) }
	params := Clamp_Params{ n = u32(x.count), min_val = v.min_val, max_val = v.max_val }
	bufs   := [3]vk.Buffer{ x.buffer, y.grad_buffer, x.grad_buffer }
	_dispatch(_clamp_back_pipeline, bufs[:], &params, _div_up(x.count, 256))
}

// --- Same-shape elementwise binary ---

gpu_min_forward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	b := _storage(op.variant.(ml.Min).b)
	if _min_pipeline == nil { _min_pipeline = _make_pipeline(MIN_SPIRV, 3, size_of(MinMax_Params)) }
	params := MinMax_Params{ n = u32(a.count) }
	bufs   := [3]vk.Buffer{ a.buffer, b.buffer, y.buffer }
	_dispatch(_min_pipeline, bufs[:], &params, _div_up(a.count, 256))
}
gpu_min_backward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	b := _storage(op.variant.(ml.Min).b)
	if _min_back_pipeline == nil { _min_back_pipeline = _make_pipeline(MIN_BACK_SPIRV, 5, size_of(MinMax_Params)) }
	params := MinMax_Params{ n = u32(a.count) }
	bufs   := [5]vk.Buffer{ a.buffer, b.buffer, y.grad_buffer, a.grad_buffer, b.grad_buffer }
	_dispatch(_min_back_pipeline, bufs[:], &params, _div_up(a.count, 256))
}

gpu_max_forward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	b := _storage(op.variant.(ml.Max).b)
	if _max_pipeline == nil { _max_pipeline = _make_pipeline(MAX_SPIRV, 3, size_of(MinMax_Params)) }
	params := MinMax_Params{ n = u32(a.count) }
	bufs   := [3]vk.Buffer{ a.buffer, b.buffer, y.buffer }
	_dispatch(_max_pipeline, bufs[:], &params, _div_up(a.count, 256))
}
gpu_max_backward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	b := _storage(op.variant.(ml.Max).b)
	if _max_back_pipeline == nil { _max_back_pipeline = _make_pipeline(MAX_BACK_SPIRV, 5, size_of(MinMax_Params)) }
	params := MinMax_Params{ n = u32(a.count) }
	bufs   := [5]vk.Buffer{ a.buffer, b.buffer, y.grad_buffer, a.grad_buffer, b.grad_buffer }
	_dispatch(_max_back_pipeline, bufs[:], &params, _div_up(a.count, 256))
}

// --- Broadcast elementwise binary ---

gpu_sub_forward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	b := _storage(op.variant.(ml.Sub).b)
	if _sub_pipeline == nil { _sub_pipeline = _make_pipeline(SUB_SPIRV, 3, size_of(Sub_Params)) }
	params := Sub_Params{ n = u32(a.count), n_b = u32(b.count) }
	bufs   := [3]vk.Buffer{ a.buffer, b.buffer, y.buffer }
	_dispatch(_sub_pipeline, bufs[:], &params, _div_up(a.count, 256))
}
gpu_sub_backward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Sub); b := _storage(v.b)

	if _sub_back_a_pipeline == nil { _sub_back_a_pipeline = _make_pipeline(SUB_BACK_A_SPIRV, 2, size_of(Sub_Back_A_Params)) }
	a_params := Sub_Back_A_Params{ n = u32(a.count) }
	a_bufs   := [2]vk.Buffer{ y.grad_buffer, a.grad_buffer }
	_dispatch(_sub_back_a_pipeline, a_bufs[:], &a_params, _div_up(a.count, 256))

	if _sub_back_b_pipeline == nil { _sub_back_b_pipeline = _make_pipeline(SUB_BACK_B_SPIRV, 2, size_of(Sub_Back_B_Params)) }
	b_params := Sub_Back_B_Params{ n_b = u32(b.count), stride = u32(v.stride) }
	b_bufs   := [2]vk.Buffer{ y.grad_buffer, b.grad_buffer }
	_dispatch(_sub_back_b_pipeline, b_bufs[:], &b_params, _div_up(b.count, 256))
}

gpu_div_forward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	b := _storage(op.variant.(ml.Div).b)
	if _div_pipeline == nil { _div_pipeline = _make_pipeline(DIV_SPIRV, 3, size_of(Div_Params)) }
	params := Div_Params{ n = u32(a.count), n_b = u32(b.count) }
	bufs   := [3]vk.Buffer{ a.buffer, b.buffer, y.buffer }
	_dispatch(_div_pipeline, bufs[:], &params, _div_up(a.count, 256))
}
gpu_div_backward :: proc(op: ml.Operation) {
	a := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Div); b := _storage(v.b)

	if _div_back_a_pipeline == nil { _div_back_a_pipeline = _make_pipeline(DIV_BACK_A_SPIRV, 3, size_of(Div_Back_A_Params)) }
	a_params := Div_Back_A_Params{ n = u32(a.count), n_b = u32(b.count) }
	a_bufs   := [3]vk.Buffer{ b.buffer, y.grad_buffer, a.grad_buffer }
	_dispatch(_div_back_a_pipeline, a_bufs[:], &a_params, _div_up(a.count, 256))

	if _div_back_b_pipeline == nil { _div_back_b_pipeline = _make_pipeline(DIV_BACK_B_SPIRV, 4, size_of(Div_Back_B_Params)) }
	b_params := Div_Back_B_Params{ n_b = u32(b.count), stride = u32(v.stride) }
	b_bufs   := [4]vk.Buffer{ a.buffer, b.buffer, y.grad_buffer, b.grad_buffer }
	_dispatch(_div_back_b_pipeline, b_bufs[:], &b_params, _div_up(b.count, 256))
}

// --- Shape ops ---

gpu_transpose_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Transpose)
	rows := v.rows
	cols := x.count / rows
	if _transpose_pipeline == nil { _transpose_pipeline = _make_pipeline(TRANSPOSE_SPIRV, 2, size_of(Transpose_Params)) }
	params := Transpose_Params{ rows = u32(rows), cols = u32(cols) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_transpose_pipeline, bufs[:], &params, _div_up(cols, 16), _div_up(rows, 16))
}
gpu_transpose_backward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Transpose)
	rows := v.rows
	cols := x.count / rows
	if _transpose_back_pipeline == nil { _transpose_back_pipeline = _make_pipeline(TRANSPOSE_BACK_SPIRV, 2, size_of(Transpose_Params)) }
	params := Transpose_Params{ rows = u32(rows), cols = u32(cols) }
	bufs   := [2]vk.Buffer{ y.grad_buffer, x.grad_buffer }
	_dispatch(_transpose_back_pipeline, bufs[:], &params, _div_up(cols, 16), _div_up(rows, 16))
}

gpu_slice_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Slice)
	if _slice_pipeline == nil { _slice_pipeline = _make_pipeline(SLICE_SPIRV, 2, size_of(Slice_Params)) }
	params := Slice_Params{ n = u32(y.count), start = u32(v.start) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_slice_pipeline, bufs[:], &params, _div_up(y.count, 256))
}
gpu_slice_backward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Slice)
	if _slice_back_pipeline == nil { _slice_back_pipeline = _make_pipeline(SLICE_BACK_SPIRV, 2, size_of(Slice_Params)) }
	params := Slice_Params{ n = u32(y.count), start = u32(v.start) }
	bufs   := [2]vk.Buffer{ y.grad_buffer, x.grad_buffer }
	_dispatch(_slice_back_pipeline, bufs[:], &params, _div_up(y.count, 256))
}

// --- Row-reductions ---

gpu_log_softmax_forward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Log_Softmax)
	if _log_softmax_pipeline == nil { _log_softmax_pipeline = _make_pipeline(LOG_SOFTMAX_SPIRV, 2, size_of(Log_Softmax_Params)) }
	params := Log_Softmax_Params{ count = u32(v.count), size = u32(v.size) }
	bufs   := [2]vk.Buffer{ x.buffer, y.buffer }
	_dispatch(_log_softmax_pipeline, bufs[:], &params, u32(v.count))
}
gpu_log_softmax_backward :: proc(op: ml.Operation) {
	x := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Log_Softmax)
	if _log_softmax_back_pipeline == nil { _log_softmax_back_pipeline = _make_pipeline(LOG_SOFTMAX_BACK_SPIRV, 3, size_of(Log_Softmax_Params)) }
	params := Log_Softmax_Params{ count = u32(v.count), size = u32(v.size) }
	bufs   := [3]vk.Buffer{ y.buffer, y.grad_buffer, x.grad_buffer }
	_dispatch(_log_softmax_back_pipeline, bufs[:], &params, u32(v.count))
}

gpu_entropy_forward :: proc(op: ml.Operation) {
	p_s := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Entropy)
	if _entropy_pipeline == nil { _entropy_pipeline = _make_pipeline(ENTROPY_SPIRV, 2, size_of(Entropy_Params)) }
	params := Entropy_Params{ count = u32(v.count), size = u32(v.size) }
	bufs   := [2]vk.Buffer{ p_s.buffer, y.buffer }
	_dispatch(_entropy_pipeline, bufs[:], &params, u32(v.count))
}
gpu_entropy_backward :: proc(op: ml.Operation) {
	p_s := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Entropy)
	if _entropy_back_pipeline == nil { _entropy_back_pipeline = _make_pipeline(ENTROPY_BACK_SPIRV, 3, size_of(Entropy_Params)) }
	params := Entropy_Params{ count = u32(v.count), size = u32(v.size) }
	bufs   := [3]vk.Buffer{ p_s.buffer, y.grad_buffer, p_s.grad_buffer }
	total  := v.count * v.size
	_dispatch(_entropy_back_pipeline, bufs[:], &params, _div_up(total, 256))
}

gpu_mean_squared_error_forward :: proc(op: ml.Operation) {
	p_s := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Mean_Squared_Error); t := _storage(v.targets)
	size := p_s.count / v.count
	if _mean_squared_error_pipeline == nil { _mean_squared_error_pipeline = _make_pipeline(MEAN_SQUARED_ERROR_SPIRV, 3, size_of(Mean_Squared_Error_Params)) }
	params := Mean_Squared_Error_Params{ count = u32(v.count), size = u32(size) }
	bufs   := [3]vk.Buffer{ p_s.buffer, t.buffer, y.buffer }
	_dispatch(_mean_squared_error_pipeline, bufs[:], &params, u32(v.count))
}
gpu_mean_squared_error_backward :: proc(op: ml.Operation) {
	p_s := _storage(op.input); y := _storage(op.output)
	v := op.variant.(ml.Mean_Squared_Error); t := _storage(v.targets)
	size := p_s.count / v.count
	if _mean_squared_error_back_pipeline == nil { _mean_squared_error_back_pipeline = _make_pipeline(MEAN_SQUARED_ERROR_BACK_SPIRV, 4, size_of(Mean_Squared_Error_Params)) }
	params := Mean_Squared_Error_Params{ count = u32(v.count), size = u32(size) }
	bufs   := [4]vk.Buffer{ p_s.buffer, t.buffer, y.grad_buffer, p_s.grad_buffer }
	total  := p_s.count
	_dispatch(_mean_squared_error_back_pipeline, bufs[:], &params, _div_up(total, 256))
}

// Allocate a HOST_VISIBLE indices buffer, write `indices` into it as u32,
// and return (buffer, memory). Caller is responsible for queuing destruction
// after the dispatch via `_queue_destroy_buffer`.
_upload_indices :: proc(indices: []int, loc := #caller_location) -> (buf: vk.Buffer, mem_handle: vk.DeviceMemory) {
	n := len(indices)
	idx_size := vk.DeviceSize(n * size_of(u32))
	buf, mem_handle = _create_buffer(idx_size, {.STORAGE_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT}, loc)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, mem_handle, 0, idx_size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(indices) failed: %v", res, loc=loc)
	arr := ([^]u32)(mapped)
	for v, i in indices { arr[i] = u32(v) }
	vk.UnmapMemory(_gpu.device, mem_handle)
	return
}

// Copy `src` (CPU) into the GPU-resident data buffer of `t`. Uses the
// per-context cached staging buffer — no per-call allocation. If a batch
// is active, the host→staging memcpy happens immediately and the
// staging→device copy is recorded into the batch CB; this preserves
// command-order ordering relative to other commands in the batch (a
// later-recorded zero-fill on the same buffer would otherwise clobber
// the upload). Outside a batch, falls back to one-shot.
upload_tensor :: proc(t: ml.Tensor, src: []f32, loc := #caller_location) {
	fmt.assertf(t.vtable == &_gpu_backend, "upload_tensor: tensor is not on the GPU backend", loc=loc)
	storage := _storage(t)
	fmt.assertf(len(src) == storage.count, "upload_tensor size mismatch: src=%v storage.count=%v", len(src), storage.count, loc=loc)

	size := vk.DeviceSize(storage.count * size_of(f32))
	gctx := _current_gpu_ctx

	if gctx.batch.active {
		// Stage host→staging at a per-batch offset (so multiple uploads
		// in one batch don't clobber each other), then record the
		// staging→device copy into the batch CB.
		offset := gctx.batch.staging_offset
		needed := offset + size
		_ensure_staging(needed, loc)
		gctx.batch.staging_offset = needed

		dst := rawptr(uintptr(gctx.staging.mapped) + uintptr(offset))
		mem.copy(dst, raw_data(src), int(size))

		region := vk.BufferCopy{ srcOffset = offset, dstOffset = 0, size = size }
		vk.CmdCopyBuffer(gctx.batch.cmd, gctx.staging.buffer, storage.buffer, 1, &region)
		return
	}

	_ensure_staging(size, loc)
	mem.copy(gctx.staging.mapped, raw_data(src), int(size))
	_one_shot_copy(gctx.staging.buffer, storage.buffer, size)
}

// Copy the GPU-resident data buffer of `t` into `dst` (CPU). When called
// inside an active batch, the device→staging copy is recorded into the
// batch CB and the staging→host memcpy is deferred until `end_batch`'s
// queueWaitIdle returns — so a forward + download cycle pays one submit
// total, not two. Outside a batch, falls back to a one-shot copy.
download_tensor :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	_download_buffer(t, _storage(t).buffer, dst, loc)
}

// Read the gradient buffer of `t` into `dst`. Symmetric with `download_tensor`
// but reads from `grad_buffer` instead of `buffer` — useful for verifying
// backward kernels.
download_tensor_gradient :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	_download_buffer(t, _storage(t).grad_buffer, dst, loc)
}

_download_buffer :: proc(t: ml.Tensor, src: vk.Buffer, dst: []f32, loc := #caller_location) {
	fmt.assertf(t.vtable == &_gpu_backend, "download_tensor: tensor is not on the GPU backend", loc=loc)
	storage := _storage(t)
	fmt.assertf(len(dst) == storage.count, "download size mismatch: dst=%v storage.count=%v", len(dst), storage.count, loc=loc)

	size := vk.DeviceSize(storage.count * size_of(f32))
	gctx := _current_gpu_ctx

	if gctx.batch.active {
		// Fold the device→staging copy into the active batch and end the
		// batch — the caller wants synchronous data on host, which means
		// the GPU has to flush. Bump a per-batch offset so multiple
		// downloads in one batch don't clobber each other's staging
		// region (e.g. someone calls download twice before flushing).
		// Add a SHADER_WRITE → TRANSFER_READ barrier so the copy sees
		// prior dispatch writes.
		offset := gctx.batch.staging_offset
		needed := offset + size
		_ensure_staging(needed, loc)
		gctx.batch.staging_offset = needed

		barrier := vk.MemoryBarrier{
			sType         = .MEMORY_BARRIER,
			srcAccessMask = {.SHADER_WRITE},
			dstAccessMask = {.TRANSFER_READ},
		}
		vk.CmdPipelineBarrier(
			gctx.batch.cmd,
			{.COMPUTE_SHADER}, {.TRANSFER},
			{}, 1, &barrier, 0, nil, 0, nil,
		)
		region := vk.BufferCopy{ srcOffset = 0, dstOffset = offset, size = size }
		vk.CmdCopyBuffer(gctx.batch.cmd, src, gctx.staging.buffer, 1, &region)

		append(&gctx.pending_downloads, Pending_Download{ dst = dst, offset = offset, size = size })
		end_batch()
		return
	}

	_ensure_staging(size, loc)
	_one_shot_copy(src, gctx.staging.buffer, size)
	mem.copy(raw_data(dst), gctx.staging.mapped, int(size))
}

_storage :: #force_inline proc(t: ml.Tensor) -> ^Gpu_Storage {
	return cast(^Gpu_Storage)t.data
}

_destroy_gpu_storage :: proc(storage: ^Gpu_Storage) {
	if storage.buffer        != 0 do vk.DestroyBuffer(_gpu.device, storage.buffer,        nil)
	if storage.memory        != 0 do vk.FreeMemory   (_gpu.device, storage.memory,        nil)
	if storage.grad_buffer   != 0 do vk.DestroyBuffer(_gpu.device, storage.grad_buffer,   nil)
	if storage.grad_memory   != 0 do vk.FreeMemory   (_gpu.device, storage.grad_memory,   nil)
	if storage.adam_m_buffer != 0 do vk.DestroyBuffer(_gpu.device, storage.adam_m_buffer, nil)
	if storage.adam_m_memory != 0 do vk.FreeMemory   (_gpu.device, storage.adam_m_memory, nil)
	if storage.adam_v_buffer != 0 do vk.DestroyBuffer(_gpu.device, storage.adam_v_buffer, nil)
	if storage.adam_v_memory != 0 do vk.FreeMemory   (_gpu.device, storage.adam_v_memory, nil)
	free(storage)
}
