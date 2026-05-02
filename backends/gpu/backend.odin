package machine_learning_backend_gpu

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:mem"
import "core:sync"
import "core:time"

import vk "vendor:vulkan"

import ml "../../"

@(require_results)
context_create :: proc(allocator := context.allocator, loc := #caller_location) -> ^ml.Context {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	_device_init_locked()

	gctx, err := builtin.new(Context, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate Context: %v", err, loc=loc)

	_create_command_pool(gctx, loc)
	_create_descriptor_pool(gctx, loc)

	ml._context_init(gctx, {
		clear        = clear,
		forward      = forward,
		backward     = backward,
		update       = update,
		buffer_alloc = buffer_alloc,
		buffer_free  = buffer_free,
		buffer_get   = buffer_get,
		buffer_set   = buffer_set,
		buffer_copy  = buffer_copy,
	}, allocator, loc)

	return gctx
}

context_destroy :: proc(ctx: ^ml.Context, allocator := context.allocator, loc := #caller_location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gctx := cast(^Context)ctx

	// Any pending GPU work is flushed by clear() and by the synchronous
	// buffer_get / buffer_set / buffer_copy paths, so the batch must not
	// be active here. If it is, the user dropped state on the floor.
	fmt.assertf(!gctx.batch.active, "context_destroy called with an active batch; missed a flush?", loc=loc)

	for slot in gctx.activation_pool {
		vk.DestroyBuffer(_gpu.device, slot.buf, nil)
	}
	builtin.delete(gctx.activation_pool)

	for arena in gctx.activation_arenas {
		vk.FreeMemory(_gpu.device, arena.memory, nil)
	}
	builtin.delete(gctx.activation_arenas)

	for block in gctx.persistent_pool {
		vk.FreeMemory(_gpu.device, block.memory, nil)
	}
	builtin.delete(gctx.persistent_pool)

	builtin.delete(gctx.sizes)

	if gctx.staging.buffer != 0 {
		if gctx.staging.mapped != nil {
			vk.UnmapMemory(_gpu.device, gctx.staging.memory)
		}
		vk.DestroyBuffer(_gpu.device, gctx.staging.buffer, nil)
		vk.FreeMemory(_gpu.device, gctx.staging.memory, nil)
	}
	builtin.delete(gctx.pending_downloads)

	builtin.delete(gctx.batch.pending_buffers)
	builtin.delete(gctx.batch.pending_memories)

	if gctx.query_pool != 0 {
		vk.DestroyQueryPool(_gpu.device, gctx.query_pool, nil)
	}
	delete(gctx.timing_totals)
	builtin.delete(gctx.pending_queries)

	if gctx.descriptor_pool != 0 {
		vk.DestroyDescriptorPool(_gpu.device, gctx.descriptor_pool, nil)
	}
	if gctx.command_pool != 0 {
		vk.DestroyCommandPool(_gpu.device, gctx.command_pool, nil)
	}

	ml._context_destroy(ctx, loc)
	builtin.free(gctx, allocator=allocator, loc=loc)
}

clear :: proc(loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gctx := _gctx(loc)
	if gctx.batch.active {
		end_batch(loc)
	}

	gctx.activation_cursor = 0
	for &arena in gctx.activation_arenas {
		arena.used = 0
	}
}

_alloc_count: int
_alloc_ns:    i64
_upload_count: int
_upload_ns:    i64

reset_alloc_stats :: proc() {
	_alloc_count  = 0
	_alloc_ns     = 0
	_upload_count = 0
	_upload_ns    = 0
}

alloc_stats :: proc() -> (count: int, ns: i64) {
	return _alloc_count, _alloc_ns
}

upload_stats :: proc() -> (count: int, ns: i64) {
	return _upload_count, _upload_ns
}

buffer_alloc :: proc(byte_count: int, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	t_start := time.tick_now()
	defer {
		_alloc_count += 1
		_alloc_ns    += i64(time.tick_since(t_start))
	}
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gctx := _gctx(loc)
	size := vk.DeviceSize(byte_count)
	usage := vk.BufferUsageFlags{.STORAGE_BUFFER, .TRANSFER_SRC, .TRANSFER_DST}

	gpu_buffer: Gpu_Buffer
	needs_zero := true
	if persist {
		gpu_buffer.buffer, gpu_buffer.memory = _create_pooled_persistent_buffer(size, usage, {.DEVICE_LOCAL}, loc)
		gctx.sizes[gpu_buffer.buffer] = byte_count
	} else {
		fresh: bool
		gpu_buffer.buffer, gpu_buffer.memory, fresh = _create_pooled_activation_buffer(size, usage, {.DEVICE_LOCAL}, loc)
		// Reused activation slots already have their `.sizes` entry from the
		// original allocation; only fresh slots need a zero-fill (and they
		// rely on it because the kernel may not write every byte).
		if fresh {
			gctx.sizes[gpu_buffer.buffer] = byte_count
		} else {
			needs_zero = false
		}
	}

	if needs_zero {
		_record_fill_zero(gpu_buffer.buffer, size, loc)
	}

	return transmute(ml.Backend_Buffer)gpu_buffer
}

buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gpu_buffer := transmute(Gpu_Buffer)buffer
	if gpu_buffer.buffer == 0 { return }

	gctx := _gctx(loc)
	delete_key(&gctx.sizes, gpu_buffer.buffer)
	_destroy_gpu_buffer(gpu_buffer)
}

buffer_get :: proc(buffer: ml.Backend_Buffer, dst: []byte, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gpu_buffer := transmute(Gpu_Buffer)buffer
	if gpu_buffer.buffer == 0 || builtin.len(dst) == 0 { return }
	_download(gpu_buffer.buffer, dst, loc)
}

buffer_set :: proc(buffer: ml.Backend_Buffer, src: []byte, loc: runtime.Source_Code_Location) {
	t_start := time.tick_now()
	defer {
		_upload_count += 1
		_upload_ns    += i64(time.tick_since(t_start))
	}
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gpu_buffer := transmute(Gpu_Buffer)buffer
	if gpu_buffer.buffer == 0 || builtin.len(src) == 0 { return }
	_upload(gpu_buffer.buffer, src, loc)
}

buffer_copy :: proc(dst, src: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	dst_buffer := transmute(Gpu_Buffer)dst
	src_buffer := transmute(Gpu_Buffer)src
	if dst_buffer.buffer == 0 || src_buffer.buffer == 0 { return }

	gctx := _gctx(loc)
	byte_count, ok := gctx.sizes[src_buffer.buffer]
	fmt.assertf(ok, "buffer_copy: source buffer is not registered with this context", loc=loc)
	size := vk.DeviceSize(byte_count)
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

update :: proc(opt: ml.Optimizer, t: ml.Tensor, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	d := data(t)
	g := gradient(t)
	m := adam_m(t)
	v := adam_v(t)

	fmt.assertf(d.buffer != 0, "update: tensor Data buffer missing",     loc=loc)
	fmt.assertf(g.buffer != 0, "update: tensor Gradient buffer missing", loc=loc)
	fmt.assertf(m.buffer != 0, "update: tensor Adam_M buffer missing",   loc=loc)
	fmt.assertf(v.buffer != 0, "update: tensor Adam_V buffer missing",   loc=loc)

	if _adam_step_pipeline == nil {
		_adam_step_pipeline = _make_pipeline(ADAM_STEP_SPIRV, 4, size_of(Adam_Params))
	}
	n := ml.len(t)
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

_forward_op_count: int
_forward_op_ns:    i64

reset_forward_stats :: proc() {
	_forward_op_count = 0
	_forward_op_ns    = 0
}

forward_stats :: proc() -> (count: int, ns: i64) {
	return _forward_op_count, _forward_op_ns
}

forward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	t_start := time.tick_now()
	defer {
		_forward_op_count += 1
		_forward_op_ns    += i64(time.tick_since(t_start))
	}

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
	case ml.Linear_Q4_K:        linear_q4_k_forward        (op)
	case ml.Linear_Q6_K:        linear_q6_k_forward        (op)
	case ml.Rope:               rope_forward               (op)
	case ml.Layernorm:          layernorm_forward          (op)
	case ml.Rmsnorm:            rmsnorm_forward            (op)
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
	case ml.Attention_Cache:    attention_cache_forward    (op)
	case ml.Cast:               cast_forward               (op)
	}
}

backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

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
	case ml.Linear_Q4_K:        fmt.panicf("GPU linear_q4_k_backward: linear_q4_k is forward-only (inference path)")
	case ml.Linear_Q6_K:        fmt.panicf("GPU linear_q6_k_backward: linear_q6_k is forward-only (inference path)")
	case ml.Rope:               rope_backward              (op)
	case ml.Layernorm:          layernorm_backward         (op)
	case ml.Rmsnorm:            rmsnorm_backward           (op)
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
	case ml.Attention_Cache:    attention_cache_backward   (op)
	case ml.Cast:               cast_backward              (op)
	}
}

cast_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	if input.type == output.type {
		_copy(data(output).buffer, data(input).buffer, vk.DeviceSize(ml.len(input) * ml.data_type_size(input.type)))
		return
	}

	n          := u32(ml.len(input))
	pair_count := (n + 1) / 2
	params     := Cast_Params{n = n, pair_count = pair_count}
	bufs       := [2]vk.Buffer{data(input).buffer, data(output).buffer}

	switch {
	case input.type == .F32 && output.type == .Bf16:
		if _cast_f32_to_bf16_pipeline == nil {
			_cast_f32_to_bf16_pipeline = _make_pipeline(CAST_F32_TO_BF16_SPIRV, 2, size_of(Cast_Params))
		}
		_dispatch(_cast_f32_to_bf16_pipeline, bufs[:], &params, _div_up(int(pair_count), 256))
	case input.type == .Bf16 && output.type == .F32:
		if _cast_bf16_to_f32_pipeline == nil {
			_cast_bf16_to_f32_pipeline = _make_pipeline(CAST_BF16_TO_F32_SPIRV, 2, size_of(Cast_Params))
		}
		_dispatch(_cast_bf16_to_f32_pipeline, bufs[:], &params, _div_up(int(pair_count), 256))
	case:
		fmt.panicf("GPU cast: unsupported pair (%v -> %v)", input.type, output.type)
	}
}

cast_backward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	if input.type == output.type {
		// Same-type cast forward is a memcpy; backward accumulates dy into dx.
		// No element-wise op currently does plain accumulate, but a fused
		// path is overkill for a degenerate case — fall through to a small
		// add. For now: assume same-type cast is rare and just panic if it
		// shows up in a backward pass so we notice.
		fmt.panicf("GPU cast_backward: same-type cast not implemented")
	}

	n          := u32(ml.len(input))
	pair_count := (n + 1) / 2
	params     := Cast_Params{n = n, pair_count = pair_count}

	switch {
	case input.type == .F32 && output.type == .Bf16:
		if _cast_bf16_to_f32_back_pipeline == nil {
			_cast_bf16_to_f32_back_pipeline = _make_pipeline(CAST_BF16_TO_F32_BACK_SPIRV, 2, size_of(Cast_Params))
		}
		bufs := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
		_dispatch(_cast_bf16_to_f32_back_pipeline, bufs[:], &params, _div_up(int(pair_count), 256))
	case input.type == .Bf16 && output.type == .F32:
		if _cast_f32_to_bf16_back_pipeline == nil {
			_cast_f32_to_bf16_back_pipeline = _make_pipeline(CAST_F32_TO_BF16_BACK_SPIRV, 2, size_of(Cast_Params))
		}
		bufs := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
		_dispatch(_cast_f32_to_bf16_back_pipeline, bufs[:], &params, _div_up(int(pair_count), 256))
	case:
		fmt.panicf("GPU cast_backward: unsupported pair (%v -> %v)", input.type, output.type)
	}
}

add_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b

	#partial switch a.type {
	case .F32:
		if _add_pipeline == nil {
			_add_pipeline = _make_pipeline(ADD_SPIRV, 3, size_of(Add_Params))
		}
		params := Add_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
		bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_add_pipeline, bufs[:], &params, _div_up(ml.len(a), ADD_LOCAL_SIZE))
	case .Bf16:
		if _add_bf16_pipeline == nil {
			_add_bf16_pipeline = _make_pipeline(ADD_BF16_SPIRV, 3, size_of(Add_Bf16_Params))
		}
		pair_count := (ml.len(a) + 1) / 2
		params := Add_Bf16_Params{
			n          = u32(ml.len(a)),
			n_b        = u32(ml.len(b)),
			pair_count = u32(pair_count),
		}
		bufs := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_add_bf16_pipeline, bufs[:], &params, _div_up(pair_count, ADD_LOCAL_SIZE))
	}
}

add_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b
	stride := ml.len(a) / ml.len(b)

	#partial switch a.type {
	case .F32:
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
	case .Bf16:
		a_pair_count := (ml.len(a) + 1) / 2
		b_pair_count := (ml.len(b) + 1) / 2

		if _add_back_a_bf16_pipeline == nil {
			_add_back_a_bf16_pipeline = _make_pipeline(ADD_BACK_A_BF16_SPIRV, 2, size_of(Add_Back_A_Bf16_Params))
		}
		a_params := Add_Back_A_Bf16_Params{
			n          = u32(ml.len(a)),
			pair_count = u32(a_pair_count),
		}
		a_bufs := [2]vk.Buffer{gradient(output).buffer, gradient(a).buffer}
		_dispatch(_add_back_a_bf16_pipeline, a_bufs[:], &a_params, _div_up(a_pair_count, 256))

		if _add_back_b_bf16_pipeline == nil {
			_add_back_b_bf16_pipeline = _make_pipeline(ADD_BACK_B_BF16_SPIRV, 2, size_of(Add_Back_B_Bf16_Params))
		}
		b_params := Add_Back_B_Bf16_Params{
			n_b        = u32(ml.len(b)),
			stride     = u32(stride),
			pair_count = u32(b_pair_count),
		}
		b_bufs := [2]vk.Buffer{gradient(output).buffer, gradient(b).buffer}
		_dispatch(_add_back_b_bf16_pipeline, b_bufs[:], &b_params, _div_up(b_pair_count, 256))
	}
}

sub_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Sub).b

	#partial switch a.type {
	case .F32:
		if _sub_pipeline == nil {
			_sub_pipeline = _make_pipeline(SUB_SPIRV, 3, size_of(Sub_Params))
		}
		params := Sub_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
		bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_sub_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
	case .Bf16:
		if _sub_bf16_pipeline == nil {
			_sub_bf16_pipeline = _make_pipeline(SUB_BF16_SPIRV, 3, size_of(Sub_Bf16_Params))
		}
		pair_count := (ml.len(a) + 1) / 2
		params := Sub_Bf16_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b)), pair_count = u32(pair_count)}
		bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_sub_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

sub_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Sub).b
	stride := ml.len(a) / ml.len(b)

	#partial switch a.type {
	case .F32:
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
	case .Bf16:
		a_pair_count := (ml.len(a) + 1) / 2
		b_pair_count := (ml.len(b) + 1) / 2

		if _sub_back_a_bf16_pipeline == nil {
			_sub_back_a_bf16_pipeline = _make_pipeline(SUB_BACK_A_BF16_SPIRV, 2, size_of(Sub_Back_A_Bf16_Params))
		}
		a_params := Sub_Back_A_Bf16_Params{n = u32(ml.len(a)), pair_count = u32(a_pair_count)}
		a_bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(a).buffer}
		_dispatch(_sub_back_a_bf16_pipeline, a_bufs[:], &a_params, _div_up(a_pair_count, 256))

		if _sub_back_b_bf16_pipeline == nil {
			_sub_back_b_bf16_pipeline = _make_pipeline(SUB_BACK_B_BF16_SPIRV, 2, size_of(Sub_Back_B_Bf16_Params))
		}
		b_params := Sub_Back_B_Bf16_Params{n_b = u32(ml.len(b)), stride = u32(stride), pair_count = u32(b_pair_count)}
		b_bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(b).buffer}
		_dispatch(_sub_back_b_bf16_pipeline, b_bufs[:], &b_params, _div_up(b_pair_count, 256))
	}
}

mul_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Mul).b

	#partial switch a.type {
	case .F32:
		if _mul_pipeline == nil {
			_mul_pipeline = _make_pipeline(MUL_SPIRV, 3, size_of(Mul_Params))
		}
		params := Mul_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
		bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_mul_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
	case .Bf16:
		if _mul_bf16_pipeline == nil {
			_mul_bf16_pipeline = _make_pipeline(MUL_BF16_SPIRV, 3, size_of(Mul_Bf16_Params))
		}
		pair_count := (ml.len(a) + 1) / 2
		params := Mul_Bf16_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b)), pair_count = u32(pair_count)}
		bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_mul_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

mul_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Mul).b
	stride := ml.len(a) / ml.len(b)

	#partial switch a.type {
	case .F32:
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
	case .Bf16:
		a_pair_count := (ml.len(a) + 1) / 2
		b_pair_count := (ml.len(b) + 1) / 2

		if _mul_back_a_bf16_pipeline == nil {
			_mul_back_a_bf16_pipeline = _make_pipeline(MUL_BACK_A_BF16_SPIRV, 3, size_of(Mul_Back_A_Bf16_Params))
		}
		a_params := Mul_Back_A_Bf16_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b)), pair_count = u32(a_pair_count)}
		a_bufs   := [3]vk.Buffer{data(b).buffer, gradient(output).buffer, gradient(a).buffer}
		_dispatch(_mul_back_a_bf16_pipeline, a_bufs[:], &a_params, _div_up(a_pair_count, 256))

		if _mul_back_b_bf16_pipeline == nil {
			_mul_back_b_bf16_pipeline = _make_pipeline(MUL_BACK_B_BF16_SPIRV, 3, size_of(Mul_Back_B_Bf16_Params))
		}
		b_params := Mul_Back_B_Bf16_Params{n_b = u32(ml.len(b)), stride = u32(stride), pair_count = u32(b_pair_count)}
		b_bufs   := [3]vk.Buffer{data(a).buffer, gradient(output).buffer, gradient(b).buffer}
		_dispatch(_mul_back_b_bf16_pipeline, b_bufs[:], &b_params, _div_up(b_pair_count, 256))
	}
}

div_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Div).b

	#partial switch a.type {
	case .F32:
		if _div_pipeline == nil {
			_div_pipeline = _make_pipeline(DIV_SPIRV, 3, size_of(Div_Params))
		}
		params := Div_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b))}
		bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_div_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
	case .Bf16:
		if _div_bf16_pipeline == nil {
			_div_bf16_pipeline = _make_pipeline(DIV_BF16_SPIRV, 3, size_of(Div_Bf16_Params))
		}
		pair_count := (ml.len(a) + 1) / 2
		params := Div_Bf16_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b)), pair_count = u32(pair_count)}
		bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
		_dispatch(_div_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

div_backward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Div).b
	stride := ml.len(a) / ml.len(b)

	#partial switch a.type {
	case .F32:
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
	case .Bf16:
		a_pair_count := (ml.len(a) + 1) / 2
		b_pair_count := (ml.len(b) + 1) / 2

		if _div_back_a_bf16_pipeline == nil {
			_div_back_a_bf16_pipeline = _make_pipeline(DIV_BACK_A_BF16_SPIRV, 3, size_of(Div_Back_A_Bf16_Params))
		}
		a_params := Div_Back_A_Bf16_Params{n = u32(ml.len(a)), n_b = u32(ml.len(b)), pair_count = u32(a_pair_count)}
		a_bufs   := [3]vk.Buffer{data(b).buffer, gradient(output).buffer, gradient(a).buffer}
		_dispatch(_div_back_a_bf16_pipeline, a_bufs[:], &a_params, _div_up(a_pair_count, 256))

		if _div_back_b_bf16_pipeline == nil {
			_div_back_b_bf16_pipeline = _make_pipeline(DIV_BACK_B_BF16_SPIRV, 4, size_of(Div_Back_B_Bf16_Params))
		}
		b_params := Div_Back_B_Bf16_Params{n_b = u32(ml.len(b)), stride = u32(stride), pair_count = u32(b_pair_count)}
		b_bufs   := [4]vk.Buffer{data(a).buffer, data(b).buffer, gradient(output).buffer, gradient(b).buffer}
		_dispatch(_div_back_b_bf16_pipeline, b_bufs[:], &b_params, _div_up(b_pair_count, 256))
	}
}

exp_forward  :: proc(op: ml.Operation) { _unary_forward_gpu (op.input, op.output, EXP_SPIRV, &_exp_pipeline, EXP_BF16_SPIRV, &_exp_bf16_pipeline) }
exp_backward :: proc(op: ml.Operation) { _unary_backward_gpu(op.input, op.output, true, EXP_BACK_SPIRV, &_exp_back_pipeline, EXP_BACK_BF16_SPIRV, &_exp_back_bf16_pipeline) }

clamp_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Clamp)

	if _clamp_pipeline == nil {
		_clamp_pipeline = _make_pipeline(CLAMP_SPIRV, 2, size_of(Clamp_Params))
	}
	params := Clamp_Params{n = u32(ml.len(input)), min_val = variant.min_val, max_val = variant.max_val}
	bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
	_dispatch(_clamp_pipeline, bufs[:], &params, _div_up(ml.len(input), 256))
}

clamp_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Clamp)

	if _clamp_back_pipeline == nil {
		_clamp_back_pipeline = _make_pipeline(CLAMP_BACK_SPIRV, 3, size_of(Clamp_Params))
	}
	params := Clamp_Params{n = u32(ml.len(input)), min_val = variant.min_val, max_val = variant.max_val}
	bufs   := [3]vk.Buffer{data(input).buffer, gradient(output).buffer, gradient(input).buffer}
	_dispatch(_clamp_back_pipeline, bufs[:], &params, _div_up(ml.len(input), 256))
}

min_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Min).b

	if _min_pipeline == nil {
		_min_pipeline = _make_pipeline(MIN_SPIRV, 3, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
	_dispatch(_min_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

min_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b         := op.variant.(ml.Min).b

	if _min_back_pipeline == nil {
		_min_back_pipeline = _make_pipeline(MIN_BACK_SPIRV, 5, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [5]vk.Buffer{data(a).buffer, data(b).buffer, gradient(output).buffer, gradient(a).buffer, gradient(b).buffer}
	_dispatch(_min_back_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

max_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Max).b

	if _max_pipeline == nil {
		_max_pipeline = _make_pipeline(MAX_SPIRV, 3, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}
	_dispatch(_max_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

max_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b         := op.variant.(ml.Max).b

	if _max_back_pipeline == nil {
		_max_back_pipeline = _make_pipeline(MAX_BACK_SPIRV, 5, size_of(MinMax_Params))
	}
	params := MinMax_Params{n = u32(ml.len(a))}
	bufs   := [5]vk.Buffer{data(a).buffer, data(b).buffer, gradient(output).buffer, gradient(a).buffer, gradient(b).buffer}
	_dispatch(_max_back_pipeline, bufs[:], &params, _div_up(ml.len(a), 256))
}

mean_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	count  := ml.len(output)
	size   := ml.len(input) / count

	#partial switch input.type {
	case .F32:
		if _mean_pipeline == nil {
			_mean_pipeline = _make_pipeline(MEAN_SPIRV, 2, size_of(Mean_Params))
		}
		params := Mean_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_mean_pipeline, bufs[:], &params, u32(count))
	case .Bf16:
		if _mean_bf16_pipeline == nil {
			_mean_bf16_pipeline = _make_pipeline(MEAN_BF16_SPIRV, 2, size_of(Mean_Params))
		}
		params := Mean_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_mean_bf16_pipeline, bufs[:], &params, u32(count))
	}
}

mean_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	#partial switch input.type {
	case .F32:
		if _mean_back_pipeline == nil {
			_mean_back_pipeline = _make_pipeline(MEAN_BACK_SPIRV, 2, size_of(Mean_Params))
		}
		params := Mean_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
		_dispatch(_mean_back_pipeline, bufs[:], &params, _div_up(count * size, 256))
	case .Bf16:
		if _mean_back_bf16_pipeline == nil {
			_mean_back_bf16_pipeline = _make_pipeline(MEAN_BACK_BF16_SPIRV, 2, size_of(Mean_Back_Bf16_Params))
		}
		pair_count := (count * size + 1) / 2
		params := Mean_Back_Bf16_Params{count = u32(count), size = u32(size), pair_count = u32(pair_count)}
		bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
		_dispatch(_mean_back_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

transpose_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	rows    := input.shape[0]
	columns := input.shape[1]

	if _transpose_pipeline == nil {
		_transpose_pipeline = _make_pipeline(TRANSPOSE_SPIRV, 2, size_of(Transpose_Params))
	}
	params := Transpose_Params{rows = u32(rows), cols = u32(columns)}
	bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
	_dispatch(_transpose_pipeline, bufs[:], &params, _div_up(columns, 16), _div_up(rows, 16))
}

transpose_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	rows    := input.shape[0]
	columns := input.shape[1]

	if _transpose_back_pipeline == nil {
		_transpose_back_pipeline = _make_pipeline(TRANSPOSE_BACK_SPIRV, 2, size_of(Transpose_Params))
	}
	params := Transpose_Params{rows = u32(rows), cols = u32(columns)}
	bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
	_dispatch(_transpose_back_pipeline, bufs[:], &params, _div_up(columns, 16), _div_up(rows, 16))
}

select_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(output) / builtin.len(indices)

	indices_buf, indices_mem := _upload_indices(indices)

	#partial switch input.type {
	case .F32:
		if _select_pipeline == nil {
			_select_pipeline = _make_pipeline(SELECT_SPIRV, 3, size_of(Select_Params))
		}
		params := Select_Params{n_indices = u32(builtin.len(indices)), size = u32(size)}
		bufs   := [3]vk.Buffer{data(input).buffer, indices_buf, data(output).buffer}
		_dispatch(_select_pipeline, bufs[:], &params, _div_up(size, 256), u32(builtin.len(indices)), 1)
	case .Bf16:
		fmt.assertf(size % 2 == 0, "GPU bf16 select requires even row size (got %v)", size)
		if _select_bf16_pipeline == nil {
			_select_bf16_pipeline = _make_pipeline(SELECT_BF16_SPIRV, 3, size_of(Select_Params))
		}
		pair_count := size / 2
		params := Select_Params{n_indices = u32(builtin.len(indices)), size = u32(size)}
		bufs   := [3]vk.Buffer{data(input).buffer, indices_buf, data(output).buffer}
		_dispatch(_select_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256), u32(builtin.len(indices)), 1)
	}

	_queue_destroy_buffer(indices_buf, indices_mem)
}

select_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(output) / builtin.len(indices)
	vocab   := input.shape[0]

	indices_buf, indices_mem := _upload_indices(indices)

	if _select_back_pipeline == nil {
		_select_back_pipeline = _make_pipeline(SELECT_BACK_SPIRV, 3, size_of(Select_Back_Params))
	}
	params := Select_Back_Params{vocab = u32(vocab), n_indices = u32(builtin.len(indices)), size = u32(size)}
	bufs   := [3]vk.Buffer{indices_buf, gradient(output).buffer, gradient(input).buffer}
	_dispatch(_select_back_pipeline, bufs[:], &params, _div_up(vocab, 16), _div_up(size, 16), 1)

	_queue_destroy_buffer(indices_buf, indices_mem)
}

slice_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Slice)

	#partial switch input.type {
	case .F32:
		if _slice_pipeline == nil {
			_slice_pipeline = _make_pipeline(SLICE_SPIRV, 2, size_of(Slice_Params))
		}
		params := Slice_Params{n = u32(ml.len(output)), start = u32(variant.start)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_slice_pipeline, bufs[:], &params, _div_up(ml.len(output), 256))
	case .Bf16:
		if _slice_bf16_pipeline == nil {
			_slice_bf16_pipeline = _make_pipeline(SLICE_BF16_SPIRV, 2, size_of(Slice_Bf16_Params))
		}
		pair_count := (ml.len(output) + 1) / 2
		params := Slice_Bf16_Params{n = u32(ml.len(output)), start = u32(variant.start), pair_count = u32(pair_count)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_slice_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

slice_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	variant := op.variant.(ml.Slice)

	#partial switch input.type {
	case .F32:
		if _slice_back_pipeline == nil {
			_slice_back_pipeline = _make_pipeline(SLICE_BACK_SPIRV, 2, size_of(Slice_Params))
		}
		params := Slice_Params{n = u32(ml.len(output)), start = u32(variant.start)}
		bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
		_dispatch(_slice_back_pipeline, bufs[:], &params, _div_up(ml.len(output), 256))
	case .Bf16:
		if _slice_back_bf16_pipeline == nil {
			_slice_back_bf16_pipeline = _make_pipeline(SLICE_BACK_BF16_SPIRV, 2, size_of(Slice_Back_Bf16_Params))
		}
		dx_pair_count := (ml.len(input) + 1) / 2
		params := Slice_Back_Bf16_Params{n = u32(ml.len(output)), start = u32(variant.start), dx_pair_count = u32(dx_pair_count)}
		bufs   := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
		_dispatch(_slice_back_bf16_pipeline, bufs[:], &params, _div_up(dx_pair_count, 256))
	}
}

slice_trailing_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Slice_Trailing)

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml.len(input) / trailing

	#partial switch input.type {
	case .F32:
		if _slice_trailing_pipeline == nil {
			_slice_trailing_pipeline = _make_pipeline(SLICE_TRAILING_SPIRV, 2, size_of(Slice_Trailing_Params))
		}
		params := Slice_Trailing_Params{
			leading      = u32(leading),
			trailing     = u32(trailing),
			new_trailing = u32(new_trailing),
			start        = u32(variant.start),
		}
		bufs := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_slice_trailing_pipeline, bufs[:], &params, _div_up(leading * new_trailing, 256))
	case .Bf16:
		if _slice_trailing_bf16_pipeline == nil {
			_slice_trailing_bf16_pipeline = _make_pipeline(SLICE_TRAILING_BF16_SPIRV, 2, size_of(Slice_Trailing_Bf16_Params))
		}
		pair_count := (leading * new_trailing + 1) / 2
		params := Slice_Trailing_Bf16_Params{
			leading      = u32(leading),
			trailing     = u32(trailing),
			new_trailing = u32(new_trailing),
			start        = u32(variant.start),
			pair_count   = u32(pair_count),
		}
		bufs := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_slice_trailing_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

slice_trailing_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	variant := op.variant.(ml.Slice_Trailing)

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml.len(input) / trailing

	#partial switch input.type {
	case .F32:
		if _slice_trailing_back_pipeline == nil {
			_slice_trailing_back_pipeline = _make_pipeline(SLICE_TRAILING_BACK_SPIRV, 2, size_of(Slice_Trailing_Back_Params))
		}
		params := Slice_Trailing_Back_Params{
			leading      = u32(leading),
			trailing     = u32(trailing),
			new_trailing = u32(new_trailing),
			start        = u32(variant.start),
		}
		bufs := [2]vk.Buffer{gradient(input).buffer, gradient(output).buffer}
		_dispatch(_slice_trailing_back_pipeline, bufs[:], &params, _div_up(leading * new_trailing, 256))
	case .Bf16:
		if _slice_trailing_back_bf16_pipeline == nil {
			_slice_trailing_back_bf16_pipeline = _make_pipeline(SLICE_TRAILING_BACK_BF16_SPIRV, 2, size_of(Slice_Trailing_Back_Bf16_Params))
		}
		dx_pair_count := (leading * trailing + 1) / 2
		params := Slice_Trailing_Back_Bf16_Params{
			leading       = u32(leading),
			trailing      = u32(trailing),
			new_trailing  = u32(new_trailing),
			start         = u32(variant.start),
			dx_pair_count = u32(dx_pair_count),
		}
		bufs := [2]vk.Buffer{gradient(input).buffer, gradient(output).buffer}
		_dispatch(_slice_trailing_back_bf16_pipeline, bufs[:], &params, _div_up(dx_pair_count, 256))
	}
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
	total   := leading * (t_a + t_b + t_c)

	#partial switch output.type {
	case .F32:
		if _concat3_pipeline == nil {
			_concat3_pipeline = _make_pipeline(CONCAT3_SPIRV, 4, size_of(Concat3_Params))
		}
		params := Concat3_Params{leading = u32(leading), t_a = u32(t_a), t_b = u32(t_b), t_c = u32(t_c)}
		bufs   := [4]vk.Buffer{data(a).buffer, data(b).buffer, data(c).buffer, data(output).buffer}
		_dispatch(_concat3_pipeline, bufs[:], &params, _div_up(total, 256))
	case .Bf16:
		if _concat3_bf16_pipeline == nil {
			_concat3_bf16_pipeline = _make_pipeline(CONCAT3_BF16_SPIRV, 4, size_of(Concat3_Bf16_Params))
		}
		pair_count := (total + 1) / 2
		params := Concat3_Bf16_Params{
			leading = u32(leading), t_a = u32(t_a), t_b = u32(t_b), t_c = u32(t_c),
			pair_count = u32(pair_count),
		}
		bufs := [4]vk.Buffer{data(a).buffer, data(b).buffer, data(c).buffer, data(output).buffer}
		_dispatch(_concat3_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
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

	#partial switch output.type {
	case .F32:
		if _concat3_back_pipeline == nil {
			_concat3_back_pipeline = _make_pipeline(CONCAT3_BACK_SPIRV, 4, size_of(Concat3_Back_Params))
		}
		params := Concat3_Back_Params{leading = u32(leading), t_a = u32(t_a), t_b = u32(t_b), t_c = u32(t_c)}
		bufs   := [4]vk.Buffer{gradient(a).buffer, gradient(b).buffer, gradient(c).buffer, gradient(output).buffer}
		total  := leading * (t_a + t_b + t_c)
		_dispatch(_concat3_back_pipeline, bufs[:], &params, _div_up(total, 256))
	case .Bf16:
		if _concat3_back_bf16_pipeline == nil {
			_concat3_back_bf16_pipeline = _make_pipeline(CONCAT3_BACK_BF16_SPIRV, 4, size_of(Concat3_Back_Bf16_Params))
		}
		pair_a := (leading * t_a + 1) / 2
		pair_b := (leading * t_b + 1) / 2
		pair_c := (leading * t_c + 1) / 2
		total_pairs := pair_a + pair_b + pair_c
		params := Concat3_Back_Bf16_Params{
			leading = u32(leading), t_a = u32(t_a), t_b = u32(t_b), t_c = u32(t_c),
			pair_a = u32(pair_a), pair_b = u32(pair_b), pair_c = u32(pair_c),
		}
		bufs := [4]vk.Buffer{gradient(a).buffer, gradient(b).buffer, gradient(c).buffer, gradient(output).buffer}
		_dispatch(_concat3_back_bf16_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
	}
}

linear_forward :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(input) / input_size

	params := Linear_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}
	bufs := [3]vk.Buffer{data(input).buffer, data(weight).buffer, data(output).buffer}

	#partial switch input.type {
	case .F32:
		if _linear_pipeline == nil {
			_linear_pipeline = _make_pipeline(LINEAR_SPIRV, 3, size_of(Linear_Params))
		}
		_dispatch(
			_linear_pipeline, bufs[:], &params,
			_div_up(count,       LINEAR_LOCAL_X),
			_div_up(output_size, LINEAR_LOCAL_Y),
			1,
		)
	case .Bf16:
		// Decode (M=1) shape: use the GEMV-shape shader. The tiled bf16 path
		// (TILE_M=32) wastes 31/32 of M-dim compute, which is the dominant
		// cost on Gemma 4 GGUF — lm_head reads 1.34 GB of bf16 weight per
		// token, plus q_proj/k_proj. One workgroup per pair of output rows
		// so adjacent bf16-packed writes don't collide.
		if count == 1 && input_size % 2 == 0 && output_size % LINEAR_BF16_GEMV_ROWS_PER == 0 {
			if _linear_bf16_gemv_pipeline == nil {
				_linear_bf16_gemv_pipeline = _make_pipeline(LINEAR_BF16_GEMV_SPIRV, 3, size_of(Linear_Params))
			}
			_dispatch(
				_linear_bf16_gemv_pipeline, bufs[:], &params,
				_div_up(output_size, LINEAR_BF16_GEMV_ROWS_PER),
				1,
				1,
			)
			return
		}

		coopmat_eligible := _gpu.coopmat_bf16 &&
			count       % LINEAR_BF16_COOPMAT_BM == 0 &&
			output_size % LINEAR_BF16_COOPMAT_BN == 0 &&
			input_size  % LINEAR_BF16_COOPMAT_BK == 0

		if coopmat_eligible {
			if _linear_bf16_coopmat_pipeline == nil {
				_linear_bf16_coopmat_pipeline = _make_pipeline(LINEAR_BF16_COOPMAT_SPIRV, 3, size_of(Linear_Params))
			}
			_dispatch(
				_linear_bf16_coopmat_pipeline, bufs[:], &params,
				u32(count       / LINEAR_BF16_COOPMAT_BM),
				u32(output_size / LINEAR_BF16_COOPMAT_BN),
				1,
			)
		} else {
			fmt.assertf(input_size  % 2 == 0, "GPU bf16 linear requires even input_size, got %v",  input_size)
			fmt.assertf(output_size % 2 == 0, "GPU bf16 linear requires even output_size, got %v", output_size)

			if _linear_bf16_pipeline == nil {
				_linear_bf16_pipeline = _make_pipeline(LINEAR_BF16_SPIRV, 3, size_of(Linear_Params))
			}
			_dispatch(
				_linear_bf16_pipeline, bufs[:], &params,
				_div_up(count,       LINEAR_BF16_LOCAL_X),
				_div_up(output_size, LINEAR_BF16_LOCAL_Y),
				1,
			)
		}
	}
}

// GPU forward for the GGUF Q6_K linear op. M=1 (decode) only.
linear_q6_k_forward :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	v           := op.variant.(ml.Linear_Q6_K)
	output_size := v.weight.shape[0]
	input_size  := v.weight.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "GPU linear_q6_k requires input_size %% 256 == 0, got %v", input_size)
	fmt.assertf(output_size % LINEAR_Q6_K_GEMV_ROWS_PER == 0, "GPU linear_q6_k requires output_size %% %v == 0, got %v", LINEAR_Q6_K_GEMV_ROWS_PER, output_size)
	fmt.assertf(count == 1, "GPU linear_q6_k_forward currently supports M=1 (decode); got M=%v", count)

	params := Linear_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}
	bufs := [3]vk.Buffer{
		data(input)   .buffer,
		data(v.weight).buffer,
		data(output)  .buffer,
	}

	if _linear_q6_k_gemv_pipeline == nil {
		_linear_q6_k_gemv_pipeline = _make_pipeline(LINEAR_Q6_K_GEMV_SPIRV, 3, size_of(Linear_Params))
	}
	_dispatch(
		_linear_q6_k_gemv_pipeline, bufs[:], &params,
		_div_up(output_size, LINEAR_Q6_K_GEMV_ROWS_PER),
		1,
		1,
	)
}

// GPU forward for the GGUF Q4_K linear op. M=1 (decode) only — dispatches
// the GEMV-shape shader. M>1 is not yet implemented.
linear_q4_k_forward :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	v           := op.variant.(ml.Linear_Q4_K)
	output_size := v.weight.shape[0]
	input_size  := v.weight.shape[1]
	count       := ml.len(input) / input_size

	fmt.assertf(input_size  % ml.K_QUANT_BLOCK_SIZE == 0, "GPU linear_q4_k requires input_size %% 256 == 0, got %v", input_size)
	fmt.assertf(output_size % LINEAR_Q4_K_GEMV_ROWS_PER == 0, "GPU linear_q4_k requires output_size %% %v == 0, got %v", LINEAR_Q4_K_GEMV_ROWS_PER, output_size)
	fmt.assertf(count == 1, "GPU linear_q4_k_forward currently supports M=1 (decode); got M=%v", count)

	params := Linear_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}
	bufs := [3]vk.Buffer{
		data(input)   .buffer,
		data(v.weight).buffer,
		data(output)  .buffer,
	}

	if _linear_q4_k_gemv_pipeline == nil {
		_linear_q4_k_gemv_pipeline = _make_pipeline(LINEAR_Q4_K_GEMV_SPIRV, 3, size_of(Linear_Params))
	}
	_dispatch(
		_linear_q4_k_gemv_pipeline, bufs[:], &params,
		_div_up(output_size, LINEAR_Q4_K_GEMV_ROWS_PER),
		1,
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

	params := Linear_Back_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}
	dx_bufs := [3]vk.Buffer{gradient(output).buffer, data(weight).buffer, gradient(input).buffer}
	dw_bufs := [3]vk.Buffer{data(input).buffer, gradient(output).buffer, gradient(weight).buffer}

	#partial switch input.type {
	case .F32:
		if _linear_back_input_pipeline == nil {
			_linear_back_input_pipeline = _make_pipeline(LINEAR_BACK_INPUT_SPIRV, 3, size_of(Linear_Back_Params))
		}
		if _linear_back_weight_pipeline == nil {
			_linear_back_weight_pipeline = _make_pipeline(LINEAR_BACK_WEIGHT_SPIRV, 3, size_of(Linear_Back_Params))
		}
		_dispatch(
			_linear_back_input_pipeline, dx_bufs[:], &params,
			_div_up(count,      16),
			_div_up(input_size, 16),
			1,
		)
		_dispatch(
			_linear_back_weight_pipeline, dw_bufs[:], &params,
			_div_up(output_size, 16),
			_div_up(input_size,  16),
			1,
		)
	case .Bf16:
		fmt.assertf(input_size % 2 == 0, "GPU bf16 linear_backward requires even input_size, got %v", input_size)

		// Coopmat backward eligibility: dx wants count%BM, input_size%BN, output_size%BK;
		// dw wants output_size%BM, input_size%BN, count%BK. Since BM == BN we need
		// count, input_size, output_size all multiples of BM and (count, output_size)
		// also multiples of BK. With BK <= BM that reduces to multiples of BM.
		coopmat_eligible := _gpu.coopmat_bf16 &&
			count       % LINEAR_BF16_COOPMAT_BM == 0 &&
			output_size % LINEAR_BF16_COOPMAT_BM == 0 &&
			input_size  % LINEAR_BF16_COOPMAT_BN == 0

		if coopmat_eligible {
			if _linear_back_input_bf16_coopmat_pipeline == nil {
				_linear_back_input_bf16_coopmat_pipeline = _make_pipeline(LINEAR_BACK_INPUT_BF16_COOPMAT_SPIRV, 3, size_of(Linear_Back_Params))
			}
			if _linear_back_weight_bf16_coopmat_pipeline == nil {
				_linear_back_weight_bf16_coopmat_pipeline = _make_pipeline(LINEAR_BACK_WEIGHT_BF16_COOPMAT_SPIRV, 3, size_of(Linear_Back_Params))
			}
			_dispatch(
				_linear_back_input_bf16_coopmat_pipeline, dx_bufs[:], &params,
				u32(count      / LINEAR_BF16_COOPMAT_BM),
				u32(input_size / LINEAR_BF16_COOPMAT_BN),
				1,
			)
			_dispatch(
				_linear_back_weight_bf16_coopmat_pipeline, dw_bufs[:], &params,
				u32(output_size / LINEAR_BF16_COOPMAT_BM),
				u32(input_size  / LINEAR_BF16_COOPMAT_BN),
				1,
			)
		} else {
			if _linear_back_input_bf16_pipeline == nil {
				_linear_back_input_bf16_pipeline = _make_pipeline(LINEAR_BACK_INPUT_BF16_SPIRV, 3, size_of(Linear_Back_Params))
			}
			if _linear_back_weight_bf16_pipeline == nil {
				_linear_back_weight_bf16_pipeline = _make_pipeline(LINEAR_BACK_WEIGHT_BF16_SPIRV, 3, size_of(Linear_Back_Params))
			}
			k_pair_count := input_size / 2
			_dispatch(
				_linear_back_input_bf16_pipeline, dx_bufs[:], &params,
				_div_up(count,        16),
				_div_up(k_pair_count, 16),
				1,
			)
			_dispatch(
				_linear_back_weight_bf16_pipeline, dw_bufs[:], &params,
				_div_up(output_size,  16),
				_div_up(k_pair_count, 16),
				1,
			)
		}
	}
}

rope_forward :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	variant     := op.variant.(ml.Rope)
	token_count := input.shape[0]
	head_size   := input.shape[input.rank - 1] / variant.head_count

	params := Rope_Params{
		token_count       = u32(token_count),
		head_count        = u32(variant.head_count),
		head_size         = u32(head_size),
		base              = variant.base,
		position_offset   = u32(variant.position_offset),
		rotate_pair_count = u32(variant.rotate_pair_count),
	}
	bufs        := [2]vk.Buffer{data(input).buffer, data(output).buffer}
	total_pairs := token_count * variant.head_count * (head_size / 2)

	#partial switch input.type {
	case .F32:
		if _rope_pipeline == nil {
			_rope_pipeline = _make_pipeline(ROPE_SPIRV, 2, size_of(Rope_Params))
		}
		_dispatch(_rope_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
	case .Bf16:
		fmt.assertf(head_size % 2 == 0, "GPU bf16 rope requires even head_size (got %v)", head_size)
		if _rope_bf16_pipeline == nil {
			_rope_bf16_pipeline = _make_pipeline(ROPE_BF16_SPIRV, 2, size_of(Rope_Params))
		}
		_dispatch(_rope_bf16_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
	}
}

rope_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	variant     := op.variant.(ml.Rope)
	token_count := input.shape[0]
	head_size   := input.shape[input.rank - 1] / variant.head_count

	params := Rope_Back_Params{
		token_count       = u32(token_count),
		head_count        = u32(variant.head_count),
		head_size         = u32(head_size),
		base              = variant.base,
		position_offset   = u32(variant.position_offset),
		rotate_pair_count = u32(variant.rotate_pair_count),
	}
	bufs        := [2]vk.Buffer{gradient(input).buffer, gradient(output).buffer}
	total_pairs := token_count * variant.head_count * (head_size / 2)

	#partial switch input.type {
	case .F32:
		if _rope_back_pipeline == nil {
			_rope_back_pipeline = _make_pipeline(ROPE_BACK_SPIRV, 2, size_of(Rope_Back_Params))
		}
		_dispatch(_rope_back_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
	case .Bf16:
		fmt.assertf(head_size % 2 == 0, "GPU bf16 rope_backward requires even head_size (got %v)", head_size)
		if _rope_back_bf16_pipeline == nil {
			_rope_back_bf16_pipeline = _make_pipeline(ROPE_BACK_BF16_SPIRV, 2, size_of(Rope_Back_Params))
		}
		_dispatch(_rope_back_bf16_pipeline, bufs[:], &params, _div_up(total_pairs, 256))
	}
}

layernorm_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	is_bf16 := input.type == .Bf16
	if is_bf16 {
		fmt.assertf(size % 2 == 0, "GPU bf16 layernorm requires even size (got %v)", size)
	}

	stats_pipe: ^Pipeline
	fwd_pipe:   ^Pipeline
	if is_bf16 {
		if _layernorm_stats_bf16_pipeline == nil {
			_layernorm_stats_bf16_pipeline = _make_pipeline(LAYERNORM_STATS_BF16_SPIRV, 3, size_of(Layernorm_Stats_Params))
		}
		if _layernorm_bf16_pipeline == nil {
			_layernorm_bf16_pipeline = _make_pipeline(LAYERNORM_BF16_SPIRV, 3, size_of(Layernorm_Params))
		}
		stats_pipe = _layernorm_stats_bf16_pipeline
		fwd_pipe   = _layernorm_bf16_pipeline
	} else {
		if _layernorm_stats_pipeline == nil {
			_layernorm_stats_pipeline = _make_pipeline(LAYERNORM_STATS_SPIRV, 3, size_of(Layernorm_Stats_Params))
		}
		if _layernorm_pipeline == nil {
			_layernorm_pipeline = _make_pipeline(LAYERNORM_SPIRV, 3, size_of(Layernorm_Params))
		}
		stats_pipe = _layernorm_stats_pipeline
		fwd_pipe   = _layernorm_pipeline
	}

	stats_params := Layernorm_Stats_Params{count = u32(count), size = u32(size)}
	stats_bufs   := [3]vk.Buffer{data(input).buffer, data(variant.mean).buffer, data(variant.rstd).buffer}
	_dispatch(stats_pipe, stats_bufs[:], &stats_params, u32(count))

	fwd_params := Layernorm_Params{count = u32(count), size = u32(size)}
	fwd_bufs   := [3]vk.Buffer{data(input).buffer, data(variant.weight).buffer, data(output).buffer}
	_dispatch(fwd_pipe, fwd_bufs[:], &fwd_params, u32(count))
}

layernorm_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	variant := op.variant.(ml.Layernorm)
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	is_bf16 := input.type == .Bf16
	if is_bf16 {
		fmt.assertf(size % 2 == 0, "GPU bf16 layernorm_backward requires even size (got %v)", size)

		if _layernorm_back_input_bf16_pipeline == nil {
			_layernorm_back_input_bf16_pipeline = _make_pipeline(LAYERNORM_BACK_INPUT_BF16_SPIRV, 6, size_of(Layernorm_Back_Params))
		}
		if _layernorm_back_weight_bf16_pipeline == nil {
			_layernorm_back_weight_bf16_pipeline = _make_pipeline(LAYERNORM_BACK_WEIGHT_BF16_SPIRV, 5, size_of(Layernorm_Back_Weight_Bf16_Params))
		}

		params := Layernorm_Back_Params{count = u32(count), size = u32(size)}
		input_bufs := [6]vk.Buffer{
			data(input).buffer, data(variant.weight).buffer, gradient(output).buffer,
			data(variant.mean).buffer, data(variant.rstd).buffer, gradient(input).buffer,
		}
		_dispatch(_layernorm_back_input_bf16_pipeline, input_bufs[:], &params, u32(count))

		pair_count := size / 2
		w_params := Layernorm_Back_Weight_Bf16_Params{count = u32(count), size = u32(size), pair_count = u32(pair_count)}
		weight_bufs := [5]vk.Buffer{
			data(input).buffer, gradient(output).buffer,
			data(variant.mean).buffer, data(variant.rstd).buffer, gradient(variant.weight).buffer,
		}
		_dispatch(_layernorm_back_weight_bf16_pipeline, weight_bufs[:], &w_params, _div_up(pair_count, 256))
	} else {
		if _layernorm_back_input_pipeline == nil {
			_layernorm_back_input_pipeline = _make_pipeline(LAYERNORM_BACK_INPUT_SPIRV, 6, size_of(Layernorm_Back_Params))
		}
		if _layernorm_back_weight_pipeline == nil {
			_layernorm_back_weight_pipeline = _make_pipeline(LAYERNORM_BACK_WEIGHT_SPIRV, 5, size_of(Layernorm_Back_Params))
		}

		params := Layernorm_Back_Params{count = u32(count), size = u32(size)}

		input_bufs := [6]vk.Buffer{
			data(input).buffer, data(variant.weight).buffer, gradient(output).buffer,
			data(variant.mean).buffer, data(variant.rstd).buffer, gradient(input).buffer,
		}
		_dispatch(_layernorm_back_input_pipeline, input_bufs[:], &params, u32(count))

		weight_bufs := [5]vk.Buffer{
			data(input).buffer, gradient(output).buffer,
			data(variant.mean).buffer, data(variant.rstd).buffer, gradient(variant.weight).buffer,
		}
		_dispatch(_layernorm_back_weight_pipeline, weight_bufs[:], &params, _div_up(size, 256))
	}
}

rmsnorm_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Rmsnorm)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	is_bf16 := input.type == .Bf16
	if is_bf16 {
		fmt.assertf(size % 2 == 0, "GPU bf16 rmsnorm requires even size (got %v)", size)
	}

	stats_pipe: ^Pipeline
	fwd_pipe:   ^Pipeline
	if is_bf16 {
		if _rmsnorm_stats_bf16_pipeline == nil {
			_rmsnorm_stats_bf16_pipeline = _make_pipeline(RMSNORM_STATS_BF16_SPIRV, 2, size_of(Rmsnorm_Stats_Params))
		}
		if _rmsnorm_bf16_pipeline == nil {
			_rmsnorm_bf16_pipeline = _make_pipeline(RMSNORM_BF16_SPIRV, 3, size_of(Rmsnorm_Params))
		}
		stats_pipe = _rmsnorm_stats_bf16_pipeline
		fwd_pipe   = _rmsnorm_bf16_pipeline
	} else {
		if _rmsnorm_stats_pipeline == nil {
			_rmsnorm_stats_pipeline = _make_pipeline(RMSNORM_STATS_SPIRV, 2, size_of(Rmsnorm_Stats_Params))
		}
		if _rmsnorm_pipeline == nil {
			_rmsnorm_pipeline = _make_pipeline(RMSNORM_SPIRV, 3, size_of(Rmsnorm_Params))
		}
		stats_pipe = _rmsnorm_stats_pipeline
		fwd_pipe   = _rmsnorm_pipeline
	}

	stats_params := Rmsnorm_Stats_Params{count = u32(count), size = u32(size), eps = variant.eps}
	stats_bufs   := [2]vk.Buffer{data(input).buffer, data(variant.rstd).buffer}
	_dispatch(stats_pipe, stats_bufs[:], &stats_params, u32(count))

	fwd_params := Rmsnorm_Params{count = u32(count), size = u32(size), eps = variant.eps}
	fwd_bufs   := [3]vk.Buffer{data(input).buffer, data(variant.weight).buffer, data(output).buffer}
	_dispatch(fwd_pipe, fwd_bufs[:], &fwd_params, u32(count))
}

rmsnorm_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	variant := op.variant.(ml.Rmsnorm)
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	is_bf16 := input.type == .Bf16
	if is_bf16 {
		fmt.assertf(size % 2 == 0, "GPU bf16 rmsnorm_backward requires even size (got %v)", size)

		if _rmsnorm_back_input_bf16_pipeline == nil {
			_rmsnorm_back_input_bf16_pipeline = _make_pipeline(RMSNORM_BACK_INPUT_BF16_SPIRV, 5, size_of(Rmsnorm_Back_Params))
		}
		if _rmsnorm_back_weight_bf16_pipeline == nil {
			_rmsnorm_back_weight_bf16_pipeline = _make_pipeline(RMSNORM_BACK_WEIGHT_BF16_SPIRV, 4, size_of(Rmsnorm_Back_Weight_Bf16_Params))
		}

		params := Rmsnorm_Back_Params{count = u32(count), size = u32(size)}
		input_bufs := [5]vk.Buffer{
			data(input).buffer, data(variant.weight).buffer, gradient(output).buffer,
			data(variant.rstd).buffer, gradient(input).buffer,
		}
		_dispatch(_rmsnorm_back_input_bf16_pipeline, input_bufs[:], &params, u32(count))

		pair_count := size / 2
		w_params := Rmsnorm_Back_Weight_Bf16_Params{count = u32(count), size = u32(size), pair_count = u32(pair_count)}
		weight_bufs := [4]vk.Buffer{
			data(input).buffer, gradient(output).buffer,
			data(variant.rstd).buffer, gradient(variant.weight).buffer,
		}
		_dispatch(_rmsnorm_back_weight_bf16_pipeline, weight_bufs[:], &w_params, _div_up(pair_count, 256))
	} else {
		if _rmsnorm_back_input_pipeline == nil {
			_rmsnorm_back_input_pipeline = _make_pipeline(RMSNORM_BACK_INPUT_SPIRV, 5, size_of(Rmsnorm_Back_Params))
		}
		if _rmsnorm_back_weight_pipeline == nil {
			_rmsnorm_back_weight_pipeline = _make_pipeline(RMSNORM_BACK_WEIGHT_SPIRV, 4, size_of(Rmsnorm_Back_Params))
		}

		params := Rmsnorm_Back_Params{count = u32(count), size = u32(size)}

		input_bufs := [5]vk.Buffer{
			data(input).buffer, data(variant.weight).buffer, gradient(output).buffer,
			data(variant.rstd).buffer, gradient(input).buffer,
		}
		_dispatch(_rmsnorm_back_input_pipeline, input_bufs[:], &params, u32(count))

		weight_bufs := [4]vk.Buffer{
			data(input).buffer, gradient(output).buffer,
			data(variant.rstd).buffer, gradient(variant.weight).buffer,
		}
		_dispatch(_rmsnorm_back_weight_pipeline, weight_bufs[:], &params, _div_up(size, 256))
	}
}

softmax_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		if _softmax_pipeline == nil {
			_softmax_pipeline = _make_pipeline(SOFTMAX_SPIRV, 2, size_of(Softmax_Params))
		}
		params := Softmax_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_softmax_pipeline, bufs[:], &params, u32(count))
	case .Bf16:
		fmt.assertf(size % 2 == 0, "GPU bf16 softmax requires even size (got %v)", size)
		if _softmax_bf16_pipeline == nil {
			_softmax_bf16_pipeline = _make_pipeline(SOFTMAX_BF16_SPIRV, 2, size_of(Softmax_Params))
		}
		params := Softmax_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_softmax_bf16_pipeline, bufs[:], &params, u32(count))
	}
}

softmax_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		if _softmax_back_pipeline == nil {
			_softmax_back_pipeline = _make_pipeline(SOFTMAX_BACK_SPIRV, 3, size_of(Softmax_Back_Params))
		}
		params := Softmax_Back_Params{count = u32(count), size = u32(size)}
		bufs   := [3]vk.Buffer{data(output).buffer, gradient(output).buffer, gradient(input).buffer}
		_dispatch(_softmax_back_pipeline, bufs[:], &params, u32(count))
	case .Bf16:
		fmt.assertf(size % 2 == 0, "GPU bf16 softmax_backward requires even size (got %v)", size)
		if _softmax_back_bf16_pipeline == nil {
			_softmax_back_bf16_pipeline = _make_pipeline(SOFTMAX_BACK_BF16_SPIRV, 3, size_of(Softmax_Back_Params))
		}
		params := Softmax_Back_Params{count = u32(count), size = u32(size)}
		bufs   := [3]vk.Buffer{data(output).buffer, gradient(output).buffer, gradient(input).buffer}
		_dispatch(_softmax_back_bf16_pipeline, bufs[:], &params, u32(count))
	}
}

entropy_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		if _entropy_pipeline == nil {
			_entropy_pipeline = _make_pipeline(ENTROPY_SPIRV, 2, size_of(Entropy_Params))
		}
		params := Entropy_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_entropy_pipeline, bufs[:], &params, u32(count))
	case .Bf16:
		fmt.assertf(size % 2 == 0, "GPU bf16 entropy requires even size (got %v)", size)
		if _entropy_bf16_pipeline == nil {
			_entropy_bf16_pipeline = _make_pipeline(ENTROPY_BF16_SPIRV, 2, size_of(Entropy_Bf16_Params))
		}
		out_pair_count := (count + 1) / 2
		params := Entropy_Bf16_Params{count = u32(count), size = u32(size), out_pair_count = u32(out_pair_count)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_entropy_bf16_pipeline, bufs[:], &params, u32(out_pair_count))
		return
	}
}

entropy_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		if _entropy_back_pipeline == nil {
			_entropy_back_pipeline = _make_pipeline(ENTROPY_BACK_SPIRV, 3, size_of(Entropy_Params))
		}
		params := Entropy_Params{count = u32(count), size = u32(size)}
		bufs   := [3]vk.Buffer{data(input).buffer, gradient(output).buffer, gradient(input).buffer}
		_dispatch(_entropy_back_pipeline, bufs[:], &params, _div_up(count * size, 256))
	case .Bf16:
		fmt.assertf(size % 2 == 0, "GPU bf16 entropy_backward requires even size (got %v)", size)
		if _entropy_back_bf16_pipeline == nil {
			_entropy_back_bf16_pipeline = _make_pipeline(ENTROPY_BACK_BF16_SPIRV, 3, size_of(Entropy_Back_Bf16_Params))
		}
		pair_count := (count * size) / 2
		params := Entropy_Back_Bf16_Params{count = u32(count), size = u32(size), pair_count = u32(pair_count)}
		bufs   := [3]vk.Buffer{data(input).buffer, gradient(output).buffer, gradient(input).buffer}
		_dispatch(_entropy_back_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

log_softmax_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		if _log_softmax_pipeline == nil {
			_log_softmax_pipeline = _make_pipeline(LOG_SOFTMAX_SPIRV, 2, size_of(Log_Softmax_Params))
		}
		params := Log_Softmax_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_log_softmax_pipeline, bufs[:], &params, u32(count))
	case .Bf16:
		fmt.assertf(size % 2 == 0, "GPU bf16 log_softmax requires even size (got %v)", size)
		if _log_softmax_bf16_pipeline == nil {
			_log_softmax_bf16_pipeline = _make_pipeline(LOG_SOFTMAX_BF16_SPIRV, 2, size_of(Log_Softmax_Params))
		}
		params := Log_Softmax_Params{count = u32(count), size = u32(size)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_log_softmax_bf16_pipeline, bufs[:], &params, u32(count))
	}
}

log_softmax_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		if _log_softmax_back_pipeline == nil {
			_log_softmax_back_pipeline = _make_pipeline(LOG_SOFTMAX_BACK_SPIRV, 3, size_of(Log_Softmax_Params))
		}
		params := Log_Softmax_Params{count = u32(count), size = u32(size)}
		bufs   := [3]vk.Buffer{data(output).buffer, gradient(output).buffer, gradient(input).buffer}
		_dispatch(_log_softmax_back_pipeline, bufs[:], &params, u32(count))
	case .Bf16:
		fmt.assertf(size % 2 == 0, "GPU bf16 log_softmax_backward requires even size (got %v)", size)
		if _log_softmax_back_bf16_pipeline == nil {
			_log_softmax_back_bf16_pipeline = _make_pipeline(LOG_SOFTMAX_BACK_BF16_SPIRV, 3, size_of(Log_Softmax_Params))
		}
		params := Log_Softmax_Params{count = u32(count), size = u32(size)}
		bufs   := [3]vk.Buffer{data(output).buffer, gradient(output).buffer, gradient(input).buffer}
		_dispatch(_log_softmax_back_bf16_pipeline, bufs[:], &params, u32(count))
	}
}

mean_squared_error_forward :: proc(op: ml.Operation) {
	predictions := op.input
	output      := op.output
	targets     := op.variant.(ml.Mean_Squared_Error).targets
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	if _mean_squared_error_pipeline == nil {
		_mean_squared_error_pipeline = _make_pipeline(MEAN_SQUARED_ERROR_SPIRV, 3, size_of(Mean_Squared_Error_Params))
	}
	params := Mean_Squared_Error_Params{count = u32(count), size = u32(sample_size)}
	bufs   := [3]vk.Buffer{data(predictions).buffer, data(targets).buffer, data(output).buffer}
	_dispatch(_mean_squared_error_pipeline, bufs[:], &params, u32(count))
}

mean_squared_error_backward :: proc(op: ml.Operation) {
	predictions, output := op.input, op.output
	targets     := op.variant.(ml.Mean_Squared_Error).targets
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	if _mean_squared_error_back_pipeline == nil {
		_mean_squared_error_back_pipeline = _make_pipeline(MEAN_SQUARED_ERROR_BACK_SPIRV, 4, size_of(Mean_Squared_Error_Params))
	}
	params := Mean_Squared_Error_Params{count = u32(count), size = u32(sample_size)}
	bufs   := [4]vk.Buffer{data(predictions).buffer, data(targets).buffer, gradient(output).buffer, gradient(predictions).buffer}
	_dispatch(_mean_squared_error_back_pipeline, bufs[:], &params, _div_up(ml.len(predictions), 256))
}

cross_entropy_forward :: proc(op: ml.Operation) {
	input      := op.input
	output     := op.output
	variant    := op.variant.(ml.Cross_Entropy)
	class_size := input.shape[input.rank - 1]

	targets_buf, targets_mem := _upload_indices(variant.targets)

	if _cross_entropy_pipeline == nil {
		_cross_entropy_pipeline = _make_pipeline(CROSS_ENTROPY_SPIRV, 4, size_of(Cross_Entropy_Params))
	}
	params := Cross_Entropy_Params{count = u32(builtin.len(variant.targets)), class_size = u32(class_size)}
	bufs   := [4]vk.Buffer{data(input).buffer, targets_buf, data(variant.probabilities).buffer, data(output).buffer}
	_dispatch(_cross_entropy_pipeline, bufs[:], &params, u32(builtin.len(variant.targets)))

	_queue_destroy_buffer(targets_buf, targets_mem)
}

cross_entropy_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	variant    := op.variant.(ml.Cross_Entropy)
	class_size := input.shape[input.rank - 1]

	targets_buf, targets_mem := _upload_indices(variant.targets)

	if _cross_entropy_back_pipeline == nil {
		_cross_entropy_back_pipeline = _make_pipeline(CROSS_ENTROPY_BACK_SPIRV, 4, size_of(Cross_Entropy_Params))
	}
	params := Cross_Entropy_Params{count = u32(builtin.len(variant.targets)), class_size = u32(class_size)}
	bufs   := [4]vk.Buffer{data(variant.probabilities).buffer, targets_buf, gradient(output).buffer, gradient(input).buffer}
	total  := builtin.len(variant.targets) * class_size
	_dispatch(_cross_entropy_back_pipeline, bufs[:], &params, _div_up(total, 256))

	_queue_destroy_buffer(targets_buf, targets_mem)
}

_unary_forward_gpu :: proc(input, output: ml.Tensor,
                           f32_spirv: []u8,  f32_pipe:  ^^Pipeline,
                           bf16_spirv: []u8, bf16_pipe: ^^Pipeline) {
	n := ml.len(input)
	#partial switch input.type {
	case .F32:
		if f32_pipe^ == nil {
			f32_pipe^ = _make_pipeline(f32_spirv, 2, size_of(Activation_Params))
		}
		params := Activation_Params{n = u32(n)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(f32_pipe^, bufs[:], &params, _div_up(n, 256))
	case .Bf16:
		if bf16_pipe^ == nil {
			bf16_pipe^ = _make_pipeline(bf16_spirv, 2, size_of(Activation_Bf16_Params))
		}
		pair_count := (n + 1) / 2
		params := Activation_Bf16_Params{n = u32(n), pair_count = u32(pair_count)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(bf16_pipe^, bufs[:], &params, _div_up(pair_count, 256))
	}
}

_unary_backward_gpu :: proc(input, output: ml.Tensor, ref_is_output: bool,
                            f32_spirv: []u8,  f32_pipe:  ^^Pipeline,
                            bf16_spirv: []u8, bf16_pipe: ^^Pipeline) {
	n := ml.len(input)
	ref_buf := ref_is_output ? data(output).buffer : data(input).buffer
	#partial switch input.type {
	case .F32:
		if f32_pipe^ == nil {
			f32_pipe^ = _make_pipeline(f32_spirv, 3, size_of(Activation_Params))
		}
		params := Activation_Params{n = u32(n)}
		bufs   := [3]vk.Buffer{ref_buf, gradient(output).buffer, gradient(input).buffer}
		_dispatch(f32_pipe^, bufs[:], &params, _div_up(n, 256))
	case .Bf16:
		if bf16_pipe^ == nil {
			bf16_pipe^ = _make_pipeline(bf16_spirv, 3, size_of(Activation_Bf16_Params))
		}
		pair_count := (n + 1) / 2
		params := Activation_Bf16_Params{n = u32(n), pair_count = u32(pair_count)}
		bufs   := [3]vk.Buffer{ref_buf, gradient(output).buffer, gradient(input).buffer}
		_dispatch(bf16_pipe^, bufs[:], &params, _div_up(pair_count, 256))
	}
}

relu_forward    :: proc(op: ml.Operation) { _unary_forward_gpu (op.input, op.output, RELU_SPIRV,    &_relu_pipeline,    RELU_BF16_SPIRV,    &_relu_bf16_pipeline) }
relu_backward   :: proc(op: ml.Operation) { _unary_backward_gpu(op.input, op.output, false, RELU_BACK_SPIRV,    &_relu_back_pipeline,    RELU_BACK_BF16_SPIRV,    &_relu_back_bf16_pipeline) }
sigmoid_forward :: proc(op: ml.Operation) { _unary_forward_gpu (op.input, op.output, SIGMOID_SPIRV, &_sigmoid_pipeline, SIGMOID_BF16_SPIRV, &_sigmoid_bf16_pipeline) }
sigmoid_backward:: proc(op: ml.Operation) { _unary_backward_gpu(op.input, op.output, true,  SIGMOID_BACK_SPIRV, &_sigmoid_back_pipeline, SIGMOID_BACK_BF16_SPIRV, &_sigmoid_back_bf16_pipeline) }
silu_forward    :: proc(op: ml.Operation) { _unary_forward_gpu (op.input, op.output, SILU_SPIRV,    &_silu_pipeline,    SILU_BF16_SPIRV,    &_silu_bf16_pipeline) }
silu_backward   :: proc(op: ml.Operation) { _unary_backward_gpu(op.input, op.output, false, SILU_BACK_SPIRV,    &_silu_back_pipeline,    SILU_BACK_BF16_SPIRV,    &_silu_back_bf16_pipeline) }
tanh_forward    :: proc(op: ml.Operation) { _unary_forward_gpu (op.input, op.output, TANH_SPIRV,    &_tanh_pipeline,    TANH_BF16_SPIRV,    &_tanh_bf16_pipeline) }
tanh_backward   :: proc(op: ml.Operation) { _unary_backward_gpu(op.input, op.output, true,  TANH_BACK_SPIRV,    &_tanh_back_pipeline,    TANH_BACK_BF16_SPIRV,    &_tanh_back_bf16_pipeline) }

gelu_forward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	n := ml.len(input)
	#partial switch input.type {
	case .F32:
		if _gelu_pipeline == nil {
			_gelu_pipeline = _make_pipeline(GELU_SPIRV, 2, size_of(Gelu_Params))
		}
		params := Gelu_Params{n = u32(n)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_gelu_pipeline, bufs[:], &params, _div_up(n, GELU_LOCAL_SIZE))
	case .Bf16:
		if _gelu_bf16_pipeline == nil {
			_gelu_bf16_pipeline = _make_pipeline(GELU_BF16_SPIRV, 2, size_of(Gelu_Bf16_Params))
		}
		pair_count := (n + 1) / 2
		params := Gelu_Bf16_Params{n = u32(n), pair_count = u32(pair_count)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_gelu_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

gelu_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	n := ml.len(input)
	#partial switch input.type {
	case .F32:
		if _gelu_back_pipeline == nil {
			_gelu_back_pipeline = _make_pipeline(GELU_BACK_SPIRV, 3, size_of(Gelu_Back_Params))
		}
		params := Gelu_Back_Params{n = u32(n)}
		bufs   := [3]vk.Buffer{data(input).buffer, gradient(input).buffer, gradient(output).buffer}
		_dispatch(_gelu_back_pipeline, bufs[:], &params, _div_up(n, 256))
	case .Bf16:
		if _gelu_back_bf16_pipeline == nil {
			_gelu_back_bf16_pipeline = _make_pipeline(GELU_BACK_BF16_SPIRV, 3, size_of(Gelu_Bf16_Params))
		}
		pair_count := (n + 1) / 2
		params := Gelu_Bf16_Params{n = u32(n), pair_count = u32(pair_count)}
		bufs   := [3]vk.Buffer{data(input).buffer, gradient(input).buffer, gradient(output).buffer}
		_dispatch(_gelu_back_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

batched_matmul_forward :: proc(op: ml.Operation) {
	a           := op.input
	output      := op.output
	b           := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m           := a.shape[1]
	k           := a.shape[2]
	n           := b.shape[2]

	params := Batched_Matmul_Params{
		batch_count = u32(batch_count),
		m           = u32(m),
		k           = u32(k),
		n           = u32(n),
	}
	bufs := [3]vk.Buffer{data(a).buffer, data(b).buffer, data(output).buffer}

	#partial switch a.type {
	case .F32:
		if _batched_matmul_pipeline == nil {
			_batched_matmul_pipeline = _make_pipeline(BATCHED_MATMUL_SPIRV, 3, size_of(Batched_Matmul_Params))
		}
		_dispatch(
			_batched_matmul_pipeline, bufs[:], &params,
			_div_up(m, BATCHED_MATMUL_LOCAL_X),
			_div_up(n, BATCHED_MATMUL_LOCAL_Y),
			u32(batch_count),
		)
	case .Bf16:
		fmt.assertf(n % 2 == 0, "GPU bf16 batched_matmul requires even n, got %v", n)

		coopmat_eligible := _gpu.coopmat_bf16 &&
			m % LINEAR_BF16_COOPMAT_BM == 0 &&
			n % LINEAR_BF16_COOPMAT_BN == 0 &&
			k % LINEAR_BF16_COOPMAT_BK == 0

		if coopmat_eligible {
			if _batched_matmul_bf16_coopmat_pipeline == nil {
				_batched_matmul_bf16_coopmat_pipeline = _make_pipeline(BATCHED_MATMUL_BF16_COOPMAT_SPIRV, 3, size_of(Batched_Matmul_Params))
			}
			_dispatch(
				_batched_matmul_bf16_coopmat_pipeline, bufs[:], &params,
				u32(m / LINEAR_BF16_COOPMAT_BM),
				u32(n / LINEAR_BF16_COOPMAT_BN),
				u32(batch_count),
			)
		} else {
			if _batched_matmul_bf16_pipeline == nil {
				_batched_matmul_bf16_pipeline = _make_pipeline(BATCHED_MATMUL_BF16_SPIRV, 3, size_of(Batched_Matmul_Params))
			}
			_dispatch(
				_batched_matmul_bf16_pipeline, bufs[:], &params,
				_div_up(m,     BATCHED_MATMUL_LOCAL_X),
				_div_up(n / 2, BATCHED_MATMUL_LOCAL_Y),
				u32(batch_count),
			)
		}
	}
}

batched_matmul_backward :: proc(op: ml.Operation) {
	a, output   := op.input, op.output
	b           := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m           := a.shape[1]
	k           := a.shape[2]
	n           := b.shape[2]

	params := Batched_Matmul_Params{
		batch_count = u32(batch_count),
		m           = u32(m),
		k           = u32(k),
		n           = u32(n),
	}
	da_bufs := [3]vk.Buffer{gradient(output).buffer, data(b).buffer, gradient(a).buffer}
	db_bufs := [3]vk.Buffer{data(a).buffer, gradient(output).buffer, gradient(b).buffer}

	#partial switch a.type {
	case .F32:
		if _batched_matmul_back_input_pipeline == nil {
			_batched_matmul_back_input_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_INPUT_SPIRV, 3, size_of(Batched_Matmul_Params))
		}
		if _batched_matmul_back_weight_pipeline == nil {
			_batched_matmul_back_weight_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_WEIGHT_SPIRV, 3, size_of(Batched_Matmul_Params))
		}
		_dispatch(
			_batched_matmul_back_input_pipeline, da_bufs[:], &params,
			_div_up(m, BATCHED_MATMUL_LOCAL_X),
			_div_up(k, BATCHED_MATMUL_LOCAL_Y),
			u32(batch_count),
		)
		_dispatch(
			_batched_matmul_back_weight_pipeline, db_bufs[:], &params,
			_div_up(k, BATCHED_MATMUL_LOCAL_X),
			_div_up(n, BATCHED_MATMUL_LOCAL_Y),
			u32(batch_count),
		)
	case .Bf16:
		fmt.assertf(k % 2 == 0, "GPU bf16 batched_matmul_backward requires even k, got %v", k)
		fmt.assertf(n % 2 == 0, "GPU bf16 batched_matmul_backward requires even n, got %v", n)

		// dA: M_out=m, N_out=k, K_inner=n. dB: M_out=k, N_out=n, K_inner=m.
		// All three (m, k, n) must be multiples of BM (= BN = 64), and the
		// K_inner dim of each kernel must be a multiple of BK = 16. With BK
		// dividing BM, the BM checks subsume the BK ones.
		coopmat_eligible := _gpu.coopmat_bf16 &&
			m % LINEAR_BF16_COOPMAT_BM == 0 &&
			k % LINEAR_BF16_COOPMAT_BM == 0 &&
			n % LINEAR_BF16_COOPMAT_BM == 0

		if coopmat_eligible {
			if _batched_matmul_back_input_bf16_coopmat_pipeline == nil {
				_batched_matmul_back_input_bf16_coopmat_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_INPUT_BF16_COOPMAT_SPIRV, 3, size_of(Batched_Matmul_Params))
			}
			if _batched_matmul_back_weight_bf16_coopmat_pipeline == nil {
				_batched_matmul_back_weight_bf16_coopmat_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_WEIGHT_BF16_COOPMAT_SPIRV, 3, size_of(Batched_Matmul_Params))
			}
			_dispatch(
				_batched_matmul_back_input_bf16_coopmat_pipeline, da_bufs[:], &params,
				u32(m / LINEAR_BF16_COOPMAT_BM),
				u32(k / LINEAR_BF16_COOPMAT_BN),
				u32(batch_count),
			)
			_dispatch(
				_batched_matmul_back_weight_bf16_coopmat_pipeline, db_bufs[:], &params,
				u32(k / LINEAR_BF16_COOPMAT_BM),
				u32(n / LINEAR_BF16_COOPMAT_BN),
				u32(batch_count),
			)
		} else {
			if _batched_matmul_back_input_bf16_pipeline == nil {
				_batched_matmul_back_input_bf16_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_INPUT_BF16_SPIRV, 3, size_of(Batched_Matmul_Params))
			}
			if _batched_matmul_back_weight_bf16_pipeline == nil {
				_batched_matmul_back_weight_bf16_pipeline = _make_pipeline(BATCHED_MATMUL_BACK_WEIGHT_BF16_SPIRV, 3, size_of(Batched_Matmul_Params))
			}
			_dispatch(
				_batched_matmul_back_input_bf16_pipeline, da_bufs[:], &params,
				_div_up(m,     BATCHED_MATMUL_LOCAL_X),
				_div_up(k / 2, BATCHED_MATMUL_LOCAL_Y),
				u32(batch_count),
			)
			_dispatch(
				_batched_matmul_back_weight_bf16_pipeline, db_bufs[:], &params,
				_div_up(k,     BATCHED_MATMUL_LOCAL_X),
				_div_up(n / 2, BATCHED_MATMUL_LOCAL_Y),
				u32(batch_count),
			)
		}
	}
}

permute_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	axes   := op.variant.(ml.Permute).axes

	#partial switch input.type {
	case .F32:
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
		bufs := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_permute_pipeline, bufs[:], &params, _div_up(ml.len(output), 256))
	case .Bf16:
		if _permute_bf16_pipeline == nil {
			_permute_bf16_pipeline = _make_pipeline(PERMUTE_BF16_SPIRV, 2, size_of(Permute_Bf16_Params))
		}
		pair_count := (ml.len(output) + 1) / 2
		params := Permute_Bf16_Params{
			out_d0 = u32(output.shape[0]),
			out_d1 = u32(output.shape[1]),
			out_d2 = u32(output.shape[2]),
			in_d1  = u32(input.shape[1]),
			in_d2  = u32(input.shape[2]),
			axes_0 = u32(axes[0]),
			axes_1 = u32(axes[1]),
			axes_2 = u32(axes[2]),
			pair_count = u32(pair_count),
		}
		bufs := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_permute_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

permute_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	axes := op.variant.(ml.Permute).axes

	#partial switch input.type {
	case .F32:
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
		bufs := [2]vk.Buffer{gradient(output).buffer, gradient(input).buffer}
		_dispatch(_permute_back_pipeline, bufs[:], &params, _div_up(ml.len(output), 256))
	case .Bf16:
		if _permute_back_bf16_pipeline == nil {
			_permute_back_bf16_pipeline = _make_pipeline(PERMUTE_BACK_BF16_SPIRV, 2, size_of(Permute_Back_Bf16_Params))
		}
		pair_count := (ml.len(input) + 1) / 2
		params := Permute_Back_Bf16_Params{
			out_d0 = u32(output.shape[0]),
			out_d1 = u32(output.shape[1]),
			out_d2 = u32(output.shape[2]),
			in_d1  = u32(input.shape[1]),
			in_d2  = u32(input.shape[2]),
			axes_0 = u32(axes[0]),
			axes_1 = u32(axes[1]),
			axes_2 = u32(axes[2]),
			in_d0      = u32(input.shape[0]),
			pair_count = u32(pair_count),
		}
		bufs := [2]vk.Buffer{gradient(input).buffer, gradient(output).buffer}
		_dispatch(_permute_back_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

causal_mask_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	T      := input.shape[input.rank - 1]

	#partial switch input.type {
	case .F32:
		if _causal_mask_pipeline == nil {
			_causal_mask_pipeline = _make_pipeline(CAUSAL_MASK_SPIRV, 2, size_of(Causal_Mask_Params))
		}
		params := Causal_Mask_Params{total = u32(ml.len(input)), T = u32(T)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_causal_mask_pipeline, bufs[:], &params, _div_up(ml.len(input), 256))
	case .Bf16:
		if _causal_mask_bf16_pipeline == nil {
			_causal_mask_bf16_pipeline = _make_pipeline(CAUSAL_MASK_BF16_SPIRV, 2, size_of(Causal_Mask_Bf16_Params))
		}
		pair_count := (ml.len(input) + 1) / 2
		params := Causal_Mask_Bf16_Params{total = u32(ml.len(input)), T = u32(T), pair_count = u32(pair_count)}
		bufs   := [2]vk.Buffer{data(input).buffer, data(output).buffer}
		_dispatch(_causal_mask_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

causal_mask_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	T := input.shape[input.rank - 1]

	#partial switch input.type {
	case .F32:
		if _causal_mask_back_pipeline == nil {
			_causal_mask_back_pipeline = _make_pipeline(CAUSAL_MASK_BACK_SPIRV, 2, size_of(Causal_Mask_Params))
		}
		params := Causal_Mask_Params{total = u32(ml.len(input)), T = u32(T)}
		bufs   := [2]vk.Buffer{gradient(input).buffer, gradient(output).buffer}
		_dispatch(_causal_mask_back_pipeline, bufs[:], &params, _div_up(ml.len(input), 256))
	case .Bf16:
		if _causal_mask_back_bf16_pipeline == nil {
			_causal_mask_back_bf16_pipeline = _make_pipeline(CAUSAL_MASK_BACK_BF16_SPIRV, 2, size_of(Causal_Mask_Bf16_Params))
		}
		pair_count := (ml.len(input) + 1) / 2
		params := Causal_Mask_Bf16_Params{total = u32(ml.len(input)), T = u32(T), pair_count = u32(pair_count)}
		bufs   := [2]vk.Buffer{gradient(input).buffer, gradient(output).buffer}
		_dispatch(_causal_mask_back_bf16_pipeline, bufs[:], &params, _div_up(pair_count, 256))
	}
}

attention_forward :: proc(op: ml.Operation) {
	query   := op.input
	output  := op.output
	variant := op.variant.(ml.Attention)
	key     := variant.key
	value   := variant.value
	token_count := query.shape[0]
	q_size      := query.shape[1]
	kv_size     := key.shape[1]
	head_size   := q_size / variant.n_q_heads
	fmt.assertf(head_size <= 512, "GPU attention currently caps head_size at 512 (got %v)", head_size)
	fmt.assertf(query.type == .F32 || query.type == .Bf16, "GPU attention only supports F32 or Bf16 (got %v)", query.type)

	is_bf16 := query.type == .Bf16
	if is_bf16 {
		fmt.assertf(head_size % 2 == 0, "GPU bf16 attention requires even head_size (got %v)", head_size)
	}
	fmt.assertf(variant.window == 0 || variant.causal, "GPU attention window > 0 requires causal=true")

	params := Attention_Params{
		n_q_heads   = u32(variant.n_q_heads),
		n_kv_heads  = u32(variant.n_kv_heads),
		head_size   = u32(head_size),
		token_count = u32(token_count),
		q_size      = u32(q_size),
		kv_size     = u32(kv_size),
		causal      = variant.causal ? 1 : 0,
		window      = u32(variant.window),
	}
	bufs := [5]vk.Buffer{
		data(query).buffer, data(key).buffer, data(value).buffer,
		data(output).buffer, data(variant.lse).buffer,
	}

	use_coopmat := false && is_bf16 && _gpu.coopmat_bf16 && head_size % 16 == 0 && head_size <= 64 && variant.n_q_heads == variant.n_kv_heads

	if use_coopmat {
		if _attention_bf16_coopmat_pipeline == nil {
			_attention_bf16_coopmat_pipeline = _make_pipeline(ATTENTION_BF16_COOPMAT_SPIRV, 5, size_of(Attention_Params))
		}
		_dispatch(
			_attention_bf16_coopmat_pipeline, bufs[:], &params,
			u32(variant.n_q_heads),
			u32(_div_up(token_count, ATTENTION_BF16_COOPMAT_BR)),
		)
	} else if is_bf16 {
		if _attention_bf16_pipeline == nil {
			_attention_bf16_pipeline = _make_pipeline(ATTENTION_BF16_SPIRV, 5, size_of(Attention_Params))
		}
		_dispatch(_attention_bf16_pipeline, bufs[:], &params, u32(variant.n_q_heads), u32(token_count))
	} else {
		if _attention_pipeline == nil {
			_attention_pipeline = _make_pipeline(ATTENTION_SPIRV, 5, size_of(Attention_Params))
		}
		_dispatch(_attention_pipeline, bufs[:], &params, u32(variant.n_q_heads), u32(token_count))
	}
}

attention_backward :: proc(op: ml.Operation) {
	query   := op.input
	output  := op.output
	variant := op.variant.(ml.Attention)
	key     := variant.key
	value   := variant.value
	token_count := query.shape[0]
	q_size      := query.shape[1]
	kv_size     := key.shape[1]
	head_size   := q_size / variant.n_q_heads

	is_bf16 := query.type == .Bf16
	if is_bf16 {
		fmt.assertf(head_size % 2 == 0, "GPU bf16 attention requires even head_size (got %v)", head_size)
	}
	fmt.assertf(variant.window == 0 || variant.causal, "GPU attention_backward window > 0 requires causal=true")

	back_d_pipeline:  ^Pipeline
	back_kv_pipeline: ^Pipeline
	back_q_pipeline:  ^Pipeline
	if is_bf16 {
		if _attention_back_d_bf16_pipeline == nil {
			_attention_back_d_bf16_pipeline = _make_pipeline(ATTENTION_BACK_D_BF16_SPIRV, 3, size_of(Attention_Back_D_Params))
		}
		if _attention_back_kv_bf16_pipeline == nil {
			_attention_back_kv_bf16_pipeline = _make_pipeline(ATTENTION_BACK_KV_BF16_SPIRV, 8, size_of(Attention_Params))
		}
		if _attention_back_q_bf16_pipeline == nil {
			_attention_back_q_bf16_pipeline = _make_pipeline(ATTENTION_BACK_Q_BF16_SPIRV, 7, size_of(Attention_Params))
		}
		back_d_pipeline  = _attention_back_d_bf16_pipeline
		back_kv_pipeline = _attention_back_kv_bf16_pipeline
		back_q_pipeline  = _attention_back_q_bf16_pipeline
	} else {
		if _attention_back_d_pipeline == nil {
			_attention_back_d_pipeline = _make_pipeline(ATTENTION_BACK_D_SPIRV, 3, size_of(Attention_Back_D_Params))
		}
		if _attention_back_kv_pipeline == nil {
			_attention_back_kv_pipeline = _make_pipeline(ATTENTION_BACK_KV_SPIRV, 8, size_of(Attention_Params))
		}
		if _attention_back_q_pipeline == nil {
			_attention_back_q_pipeline = _make_pipeline(ATTENTION_BACK_Q_SPIRV, 7, size_of(Attention_Params))
		}
		back_d_pipeline  = _attention_back_d_pipeline
		back_kv_pipeline = _attention_back_kv_pipeline
		back_q_pipeline  = _attention_back_q_pipeline
	}

	d_params := Attention_Back_D_Params{
		n_q_heads   = u32(variant.n_q_heads),
		head_size   = u32(head_size),
		token_count = u32(token_count),
		q_size      = u32(q_size),
	}
	d_bufs := [3]vk.Buffer{data(output).buffer, gradient(output).buffer, data(variant.d_acc).buffer}
	_dispatch(back_d_pipeline, d_bufs[:], &d_params, u32(variant.n_q_heads), u32(token_count))

	back_params := Attention_Params{
		n_q_heads   = u32(variant.n_q_heads),
		n_kv_heads  = u32(variant.n_kv_heads),
		head_size   = u32(head_size),
		token_count = u32(token_count),
		q_size      = u32(q_size),
		kv_size     = u32(kv_size),
		causal      = variant.causal ? 1 : 0,
		window      = u32(variant.window),
	}
	kv_bufs := [8]vk.Buffer{
		data(query).buffer, data(key).buffer, data(value).buffer,
		gradient(output).buffer, data(variant.lse).buffer, data(variant.d_acc).buffer,
		gradient(key).buffer, gradient(value).buffer,
	}
	_dispatch(back_kv_pipeline, kv_bufs[:], &back_params, u32(variant.n_kv_heads), u32(token_count))

	q_bufs := [7]vk.Buffer{
		data(query).buffer, data(key).buffer, data(value).buffer,
		gradient(output).buffer, data(variant.lse).buffer, data(variant.d_acc).buffer,
		gradient(query).buffer,
	}
	_dispatch(back_q_pipeline, q_bufs[:], &back_params, u32(variant.n_q_heads), u32(token_count))
}

_upload_indices :: proc(indices: []int, loc := #caller_location) -> (buffer: vk.Buffer, memory: vk.DeviceMemory) {
	count := builtin.len(indices)
	size  := vk.DeviceSize(count * size_of(u32))
	buffer, memory = _create_buffer(size, {.STORAGE_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT}, loc)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, memory, 0, size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(indices) failed: %v", res, loc=loc)
	indices_u32 := ([^]u32)(mapped)
	for index, i in indices {
		indices_u32[i] = u32(index)
	}
	vk.UnmapMemory(_gpu.device, memory)
	return
}

attention_cache_forward :: proc(op: ml.Operation) {
	query   := op.input
	output  := op.output
	variant := op.variant.(ml.Attention_Cache)
	key     := variant.key
	value   := variant.value
	k_cache := variant.k_cache
	v_cache := variant.v_cache

	token_count := query.shape[0]
	q_size      := query.shape[1]
	kv_size     := key.shape[1]
	head_size   := q_size / variant.n_q_heads
	fmt.assertf(head_size <= 512, "GPU attention_with_cache caps head_size at 512 (got %v)", head_size)
	fmt.assertf(query.type == .F32 || query.type == .Bf16, "GPU attention_with_cache only supports F32 or Bf16 (got %v)", query.type)

	is_bf16 := query.type == .Bf16
	if is_bf16 {
		fmt.assertf(head_size % 2 == 0, "GPU bf16 attention_with_cache requires even head_size (got %v)", head_size)
	}

	gctx := _gctx()
	if !gctx.batch.active {
		begin_batch()
	}

	row_bytes  := vk.DeviceSize(kv_size * ml.data_type_size(key.type))
	t_capacity := k_cache.shape[0]
	first_phys := variant.cache_position % t_capacity
	first_count := token_count
	if first_phys + first_count > t_capacity { first_count = t_capacity - first_phys }
	first_size := vk.DeviceSize(first_count) * row_bytes
	first_dst  := vk.DeviceSize(first_phys) * row_bytes

	k_first := vk.BufferCopy{srcOffset = 0, dstOffset = first_dst, size = first_size}
	v_first := vk.BufferCopy{srcOffset = 0, dstOffset = first_dst, size = first_size}
	vk.CmdCopyBuffer(gctx.batch.cmd, data(key).buffer,   data(k_cache).buffer, 1, &k_first)
	vk.CmdCopyBuffer(gctx.batch.cmd, data(value).buffer, data(v_cache).buffer, 1, &v_first)

	if first_count < token_count {
		wrap_src  := first_size
		wrap_size := vk.DeviceSize(token_count - first_count) * row_bytes
		k_wrap := vk.BufferCopy{srcOffset = wrap_src, dstOffset = 0, size = wrap_size}
		v_wrap := vk.BufferCopy{srcOffset = wrap_src, dstOffset = 0, size = wrap_size}
		vk.CmdCopyBuffer(gctx.batch.cmd, data(key).buffer,   data(k_cache).buffer, 1, &k_wrap)
		vk.CmdCopyBuffer(gctx.batch.cmd, data(value).buffer, data(v_cache).buffer, 1, &v_wrap)
	}

	params := Attention_Cache_Params{
		n_q_heads      = u32(variant.n_q_heads),
		n_kv_heads     = u32(variant.n_kv_heads),
		head_size      = u32(head_size),
		q_token_count  = u32(token_count),
		cache_position = u32(variant.cache_position),
		q_size         = u32(q_size),
		kv_size        = u32(kv_size),
		window         = u32(variant.window),
		t_capacity     = u32(t_capacity),
	}
	bufs := [4]vk.Buffer{
		data(query).buffer, data(k_cache).buffer, data(v_cache).buffer, data(output).buffer,
	}

	if is_bf16 {
		if _attention_cache_bf16_pipeline == nil {
			_attention_cache_bf16_pipeline = _make_pipeline(ATTENTION_CACHE_BF16_SPIRV, 4, size_of(Attention_Cache_Params))
		}
		_dispatch(_attention_cache_bf16_pipeline, bufs[:], &params, u32(variant.n_q_heads), u32(token_count))
	} else {
		if _attention_cache_pipeline == nil {
			_attention_cache_pipeline = _make_pipeline(ATTENTION_CACHE_SPIRV, 4, size_of(Attention_Cache_Params))
		}
		_dispatch(_attention_cache_pipeline, bufs[:], &params, u32(variant.n_q_heads), u32(token_count))
	}
}

attention_cache_backward :: proc(op: ml.Operation) {
	fmt.panicf("attention_with_cache is forward-only (inference path); backward is not implemented")
}

enable_timing :: proc(capacity: u32 = 4096, loc := #caller_location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)
	gctx := _gctx(loc)
	if gctx.timing_enabled do return
	info := vk.QueryPoolCreateInfo{
		sType      = .QUERY_POOL_CREATE_INFO,
		queryType  = .TIMESTAMP,
		queryCount = capacity,
	}
	res := vk.CreateQueryPool(_gpu.device, &info, nil, &gctx.query_pool)
	fmt.assertf(res == .SUCCESS, "vkCreateQueryPool failed: %v", res, loc=loc)
	gctx.query_capacity  = capacity
	gctx.query_used      = 0
	gctx.timing_enabled  = true
}

reset_timing :: proc(loc := #caller_location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)
	gctx := _gctx(loc)
	builtin.clear(&gctx.timing_totals)
}

Timing_Entry :: struct {
	pipeline: ^Pipeline,
	total_ns: i64,
	count:    int,
}

dump_timing :: proc(loc := #caller_location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)
	gctx := _gctx(loc)
	if builtin.len(gctx.timing_totals) == 0 {
		fmt.println("(no timing data)")
		return
	}
	entries := builtin.make([dynamic]Timing_Entry, 0, builtin.len(gctx.timing_totals), context.temp_allocator)
	for p, stat in gctx.timing_totals {
		append(&entries, Timing_Entry{pipeline = p, total_ns = stat.total_ns, count = stat.count})
	}
	// Insertion sort by total_ns desc — small N (≤ ~50 unique pipelines).
	for i in 1 ..< builtin.len(entries) {
		j := i
		for j > 0 && entries[j].total_ns > entries[j - 1].total_ns {
			entries[j], entries[j - 1] = entries[j - 1], entries[j]
			j -= 1
		}
	}
	total_ns: i64
	for e in entries do total_ns += e.total_ns
	fmt.printfln("--- GPU timing (total %.2f ms across %v unique pipelines) ---",
		f64(total_ns) / 1e6, builtin.len(entries))
	fmt.println("  rank   total_ms     %    count   us/op   pipeline_id")
	for e, i in entries {
		pct := 100.0 * f64(e.total_ns) / f64(total_ns)
		us_per := f64(e.total_ns) / f64(e.count) / 1e3
		fmt.printfln("  %4v   %8.2f  %5.1f  %6v  %6.1f   %p",
			i + 1, f64(e.total_ns) / 1e6, pct, e.count, us_per, e.pipeline)
	}
}

upload_tensor :: proc(t: ml.Tensor, src: []f32, loc := #caller_location) {
	assert(t.type == .F32, "upload_tensor with []f32 requires an F32 tensor", loc=loc)
	t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(src), loc)
}

download_tensor :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	assert(t.type == .F32, "download_tensor with []f32 requires an F32 tensor", loc=loc)
	t.backend.buffer_get(t.buffers[.Data], mem.slice_to_bytes(dst), loc)
}

download_tensor_gradient :: proc(t: ml.Tensor, dst: []f32, loc := #caller_location) {
	assert(t.type == .F32, "download_tensor_gradient with []f32 requires an F32 tensor", loc=loc)
	t.backend.buffer_get(t.buffers[.Gradient], mem.slice_to_bytes(dst), loc)
}
