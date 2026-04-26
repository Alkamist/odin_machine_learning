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
// to one of these. Both buffers are DEVICE_LOCAL.
Gpu_Storage :: struct {
	buffer:      vk.Buffer,
	memory:      vk.DeviceMemory,
	grad_buffer: vk.Buffer,
	grad_memory: vk.DeviceMemory,
	count:       int,                // f32 element count
}

// Singleton backend instance. Wire it into a Context via:
//
//   ctx := ml.context_create(N, gpu.backend())
_gpu_backend := ml.Backend{
	name                    = "gpu",
	alloc                   = gpu_alloc,
	clear_storage           = gpu_clear_storage,
	fill_gradient_with_ones = gpu_fill_gradient_with_ones,
	forward                 = gpu_forward,
	backward                = gpu_backward,
}

@(require_results)
backend :: #force_inline proc() -> ^ml.Backend {
	return &_gpu_backend
}

gpu_alloc :: proc(t: ^ml.Tensor, n: int) {
	gctx := _current_gpu_ctx
	fmt.assertf(gctx != nil, "no active gpu Context — call gpu.context_begin / context_scope before ml ops on a GPU context")

	storage := new(Gpu_Storage)
	storage.count = n

	size := vk.DeviceSize(n * size_of(f32))
	storage.buffer,      storage.memory      = _create_buffer(size, {.STORAGE_BUFFER, .TRANSFER_SRC, .TRANSFER_DST}, {.DEVICE_LOCAL})
	storage.grad_buffer, storage.grad_memory = _create_buffer(size, {.STORAGE_BUFFER, .TRANSFER_SRC, .TRANSFER_DST}, {.DEVICE_LOCAL})

	// Match CPU's `zeros` semantics: both data and gradient start at 0.
	// One-shot covers both buffers so we only pay one queue submit.
	cmd := _begin_one_shot()
	vk.CmdFillBuffer(cmd, storage.buffer,      0, size, 0)
	vk.CmdFillBuffer(cmd, storage.grad_buffer, 0, size, 0)
	_end_one_shot(cmd)

	append(&gctx.allocations, storage)
	t.storage = storage
}

// f32(1.0) bit pattern — vkCmdFillBuffer writes a u32 stamp across the
// buffer, so we hand it the IEEE 754 representation of 1.0.
F32_ONE_BITS :: u32(0x3F800000)

gpu_fill_gradient_with_ones :: proc(t: ^ml.Tensor) {
	storage := _storage(t^)
	size    := vk.DeviceSize(storage.count * size_of(f32))

	cmd := _begin_one_shot()
	vk.CmdFillBuffer(cmd, storage.grad_buffer, 0, size, F32_ONE_BITS)
	_end_one_shot(cmd)
}

gpu_clear_storage :: proc() {
	gctx := _current_gpu_ctx
	if gctx == nil {
		return
	}

	for storage in gctx.allocations {
		_destroy_gpu_storage(storage)
	}
	clear(&gctx.allocations)
}

gpu_forward :: proc(op: ml.Operation) {
	switch _ in op.variant {
	case ml.Add:            gpu_add_forward            (op)
	case ml.Linear:         gpu_linear_forward         (op)
	case ml.Gelu:           gpu_gelu_forward           (op)
	case ml.Select:         gpu_select_forward         (op)
	case ml.Rope:           gpu_rope_forward           (op)
	case ml.Slice_Trailing: gpu_slice_trailing_forward (op)
	case ml.Concat:         gpu_concat_forward         (op)
	case ml.Softmax:        gpu_softmax_forward        (op)
	case ml.Permute:        gpu_permute_forward        (op)
	case ml.Causal_Mask:    gpu_causal_mask_forward    (op)
	case ml.Sub, ml.Mul, ml.Div, ml.Exp, ml.Clamp, ml.Min, ml.Max, ml.Mean,
	     ml.Transpose, ml.Slice,
	     ml.Layernorm, ml.Entropy, ml.Log_Softmax,
	     ml.Mean_Squared_Error, ml.Cross_Entropy, ml.Relu, ml.Sigmoid,
	     ml.Silu, ml.Tanh, ml.Batched_Matmul:
		fmt.panicf("gpu_forward: op variant %v not yet implemented on GPU backend", op.variant)
	}
}

gpu_backward :: proc(op: ml.Operation) {
	switch _ in op.variant {
	case ml.Add:            gpu_add_backward            (op)
	case ml.Linear:         gpu_linear_backward         (op)
	case ml.Gelu:           gpu_gelu_backward           (op)
	case ml.Select:         gpu_select_backward         (op)
	case ml.Rope:           gpu_rope_backward           (op)
	case ml.Slice_Trailing: gpu_slice_trailing_backward (op)
	case ml.Concat:         gpu_concat_backward         (op)
	case ml.Softmax:        gpu_softmax_backward        (op)
	case ml.Permute:        gpu_permute_backward        (op)
	case ml.Causal_Mask:    gpu_causal_mask_backward    (op)
	case ml.Sub, ml.Mul, ml.Div, ml.Exp, ml.Clamp, ml.Min, ml.Max, ml.Mean,
	     ml.Transpose, ml.Slice,
	     ml.Layernorm, ml.Entropy, ml.Log_Softmax,
	     ml.Mean_Squared_Error, ml.Cross_Entropy, ml.Relu, ml.Sigmoid,
	     ml.Silu, ml.Tanh, ml.Batched_Matmul:
		fmt.panicf("gpu_backward: op variant %v not yet implemented on GPU backend", op.variant)
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

// Copy `src` (CPU) into the GPU-resident data buffer of `t`. Useful for
// seeding inputs / parameters in test code.
upload_tensor :: proc(t: ml.Tensor, src: []f32, loc := #caller_location) {
	fmt.assertf(t.backend == &_gpu_backend, "upload_tensor: tensor is not on the GPU backend", loc=loc)
	storage := _storage(t)
	fmt.assertf(len(src) == storage.count, "upload_tensor size mismatch: src=%v storage.count=%v", len(src), storage.count, loc=loc)

	size := vk.DeviceSize(storage.count * size_of(f32))
	stage_buf, stage_mem := _create_buffer(size, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT})
	defer {
		vk.DestroyBuffer(_gpu.device, stage_buf, nil)
		vk.FreeMemory(_gpu.device, stage_mem, nil)
	}

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, stage_mem, 0, size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(upload_tensor) failed: %v", res)
	mem.copy(mapped, raw_data(src), int(size))
	vk.UnmapMemory(_gpu.device, stage_mem)

	_one_shot_copy(stage_buf, storage.buffer, size)
}

// Copy the GPU-resident data buffer of `t` into `dst` (CPU).
download_tensor :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	fmt.assertf(t.backend == &_gpu_backend, "download_tensor: tensor is not on the GPU backend", loc=loc)
	storage := _storage(t)
	fmt.assertf(len(dst) == storage.count, "download_tensor size mismatch: dst=%v storage.count=%v", len(dst), storage.count, loc=loc)

	size := vk.DeviceSize(storage.count * size_of(f32))
	stage_buf, stage_mem := _create_buffer(size, {.TRANSFER_DST}, {.HOST_VISIBLE, .HOST_COHERENT})
	defer {
		vk.DestroyBuffer(_gpu.device, stage_buf, nil)
		vk.FreeMemory(_gpu.device, stage_mem, nil)
	}

	_one_shot_copy(storage.buffer, stage_buf, size)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, stage_mem, 0, size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(download_tensor) failed: %v", res)
	mem.copy(raw_data(dst), mapped, int(size))
	vk.UnmapMemory(_gpu.device, stage_mem)
}

// Read the gradient buffer of `t` into `dst`. Symmetric with `download_tensor`
// but reads from `grad_buffer` instead of `buffer` — useful for verifying
// backward kernels.
download_tensor_gradient :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	fmt.assertf(t.backend == &_gpu_backend, "download_tensor_gradient: tensor is not on the GPU backend", loc=loc)
	storage := _storage(t)
	fmt.assertf(len(dst) == storage.count, "download_tensor_gradient size mismatch: dst=%v storage.count=%v", len(dst), storage.count, loc=loc)

	size := vk.DeviceSize(storage.count * size_of(f32))
	stage_buf, stage_mem := _create_buffer(size, {.TRANSFER_DST}, {.HOST_VISIBLE, .HOST_COHERENT})
	defer {
		vk.DestroyBuffer(_gpu.device, stage_buf, nil)
		vk.FreeMemory(_gpu.device, stage_mem, nil)
	}

	_one_shot_copy(storage.grad_buffer, stage_buf, size)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, stage_mem, 0, size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(download_tensor_gradient) failed: %v", res)
	mem.copy(raw_data(dst), mapped, int(size))
	vk.UnmapMemory(_gpu.device, stage_mem)
}

_storage :: #force_inline proc(t: ml.Tensor) -> ^Gpu_Storage {
	return cast(^Gpu_Storage)t.storage
}

_destroy_gpu_storage :: proc(storage: ^Gpu_Storage) {
	if storage.buffer      != 0 do vk.DestroyBuffer(_gpu.device, storage.buffer, nil)
	if storage.memory      != 0 do vk.FreeMemory   (_gpu.device, storage.memory, nil)
	if storage.grad_buffer != 0 do vk.DestroyBuffer(_gpu.device, storage.grad_buffer, nil)
	if storage.grad_memory != 0 do vk.FreeMemory   (_gpu.device, storage.grad_memory, nil)
	free(storage)
}
