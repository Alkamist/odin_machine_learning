package cpu

import "base:builtin"
import "base:runtime"
import "base:intrinsics"

import "core:fmt"
import "core:mem"
import "core:math"
import "core:simd"
import "core:sync"
import "core:thread"

import ml "../.."

when thread.IS_SUPPORTED {
	_thread_pool_context: runtime.Context

	Worker :: struct {
		thread:    ^thread.Thread,
		id:        int,
		start_sem: sync.Sema,
	}

	Dispatch :: struct {
		chunk_proc: proc(start, end: int, raw: rawptr),
		data:       rawptr,
		job_count:  int,
		task_count: int,
	}

	_thread_count: int = 1

	_workers:    []^Worker
	_shutdown:   bool
	_dispatch:   Dispatch
	_done_wg:    sync.Wait_Group
	_pool_mutex: sync.Mutex

	_worker_proc :: proc(t: ^thread.Thread) {
		w := cast(^Worker)t.data
		_enable_flush_to_zero()

		for {
			sync.sema_wait(&w.start_sem)
			if _shutdown {
				return
			}

			d := _dispatch
			if w.id < d.task_count {
				chunk := (d.job_count + d.task_count - 1) / d.task_count
				start := w.id * chunk
				end   := start + chunk
				if end > d.job_count {
					end = d.job_count
				}

				if start < end {
					d.chunk_proc(start, end, d.data)
				}
			}

			sync.wait_group_done(&_done_wg)
		}
	}

	_startup_thread_pool :: proc(thread_count: int) {
		_thread_pool_context = context

		_shutdown = false
		n := thread_count - 1
		_workers = builtin.make([]^Worker, n)
		for i in 0 ..< n {
			w            := builtin.new(Worker)
			w.id          = i + 1
			w.thread      = thread.create(_worker_proc)
			w.thread.data = w

			thread.start(w.thread)

			_workers[i] = w
		}
	}

	_cleanup_thread_pool :: proc() {
		_shutdown = true

		for w in _workers {
			sync.sema_post(&w.start_sem)
		}

		for w in _workers {
			thread.join(w.thread)
			thread.destroy(w.thread)
			builtin.free(w)
		}

		builtin.delete(_workers)

		_workers = nil
	}

	thread_count :: #force_inline proc() -> int {
		return _thread_count
	}

	set_thread_count :: proc(count: int, loc := #caller_location) {
		assert(count > 0, "Thread count must be at least 1", loc=loc)

		sync.mutex_lock(&_pool_mutex)
		defer sync.mutex_unlock(&_pool_mutex)

		if count == _thread_count {
			return
		}

		if _thread_count > 1 {
			_cleanup_thread_pool()
		}

		if count == 1 {
			_thread_count = 1
			return
		}

		_startup_thread_pool(count)
		_thread_count = count
	}

	@(fini)
	thread_pool_fini :: proc "contextless" () {
		if _thread_count <= 1 {
			return
		}

		context = _thread_pool_context
		_cleanup_thread_pool()
	}

	PARALLELIZE_MIN_WORK :: 24 * 1024

	parallelize :: proc(job_count, task_count: int, data: $Data, job: proc(index: int, data: Data), work := max(int)) {
		Thunk_Data :: struct {
			data: Data,
			job:  proc(index: int, data: Data),
		}

		thunk :: proc(start, end: int, raw: rawptr) {
			td := cast(^Thunk_Data)raw
			for i in start ..< end {
				td.job(i, td.data)
			}
		}

		if job_count <= 1 {
			job(0, data)
			return
		}

		if work < PARALLELIZE_MIN_WORK {
			for i in 0 ..< job_count {
				job(i, data)
			}
			return
		}

		if _thread_count <= 1 || task_count <= 1 {
			for i in 0 ..< job_count {
				job(i, data)
			}
			return
		}

		sync.mutex_lock(&_pool_mutex)
		defer sync.mutex_unlock(&_pool_mutex)

		n := task_count
		if n > _thread_count { n = _thread_count }
		if n > job_count     { n = job_count     }

		if n <= 1 {
			for i in 0 ..< job_count {
				job(i, data)
			}
			return
		}

		td := Thunk_Data{data=data, job=job}

		_dispatch = Dispatch{
			chunk_proc = thunk,
			data       = &td,
			job_count  = job_count,
			task_count = n,
		}

		sync.wait_group_add(&_done_wg, n - 1)
		for i in 0 ..< n - 1 {
			sync.sema_post(&_workers[i].start_sem)
		}

		chunk := (job_count + n - 1) / n
		end   := chunk
		if end > job_count {
			end = job_count
		}
		thunk(0, end, &td)

		sync.wait_group_wait(&_done_wg)
	}
} else {
	thread_count :: #force_inline proc() -> int {
		return 1
	}

	set_thread_count :: proc(count: int, loc := #caller_location) {
	}

	parallelize :: proc(job_count, task_count: int, data: $Data, job: proc(index: int, data: Data), work := max(int)) {
		for i in 0 ..< job_count {
			job(i, data)
		}
	}
}

when intrinsics.has_target_feature("avx") {
	SIMD_LANES :: 8
	F32x8      :: #simd[SIMD_LANES]f32

	_simd_dot_f32 :: #force_inline proc "contextless" (a, b: [^]f32, n: int) -> f32 {
		acc: F32x8
		i := 0
		for ; i + SIMD_LANES <= n; i += SIMD_LANES {
			av := intrinsics.unaligned_load((^F32x8)(&a[i]))
			bv := intrinsics.unaligned_load((^F32x8)(&b[i]))
			acc = simd.fma(av, bv, acc)
		}
		sum := simd.reduce_add_bisect(acc)
		for ; i < n; i += 1 {
			sum += a[i] * b[i]
		}
		return sum
	}

	_simd_axpy_f32 :: #force_inline proc "contextless" (y, x: [^]f32, a: f32, n: int) {
		av := F32x8(a)
		i  := 0
		for ; i + SIMD_LANES <= n; i += SIMD_LANES {
			xv := intrinsics.unaligned_load((^F32x8)(&x[i]))
			yv := intrinsics.unaligned_load((^F32x8)(&y[i]))
			intrinsics.unaligned_store((^F32x8)(&y[i]), simd.fma(xv, av, yv))
		}
		for ; i < n; i += 1 {
			y[i] += a * x[i]
		}
	}

	_bf16x8_to_f32x8 :: #force_inline proc "contextless" (b: #simd[8]u16) -> F32x8 {
		zeros: #simd[8]u16
		wide := intrinsics.simd_shuffle(zeros, b,
			0, 8,  1,  9, 2, 10, 3, 11,
			4, 12, 5, 13, 6, 14, 7, 15,
		)
		return transmute(F32x8)wide
	}

	_simd_dot_bf16_f32 :: #force_inline proc "contextless" (a, b: [^]ml.Bf16, n: int) -> f32 {
		acc: F32x8
		i := 0
		for ; i + SIMD_LANES <= n; i += SIMD_LANES {
			au := intrinsics.unaligned_load((^#simd[8]u16)(&a[i]))
			bu := intrinsics.unaligned_load((^#simd[8]u16)(&b[i]))
			acc = simd.fma(_bf16x8_to_f32x8(au), _bf16x8_to_f32x8(bu), acc)
		}
		sum := simd.reduce_add_bisect(acc)
		for ; i < n; i += 1 {
			sum += ml.bf16_to_f32(a[i]) * ml.bf16_to_f32(b[i])
		}
		return sum
	}

} else {
	_simd_dot_f32 :: #force_inline proc "contextless" (a, b: [^]f32, n: int) -> f32 {
		s0, s1, s2, s3: f32
		i := 0
		for ; i + 4 <= n; i += 4 {
			s0 += a[i + 0] * b[i + 0]
			s1 += a[i + 1] * b[i + 1]
			s2 += a[i + 2] * b[i + 2]
			s3 += a[i + 3] * b[i + 3]
		}
		sum := (s0 + s1) + (s2 + s3)
		for ; i < n; i += 1 {
			sum += a[i] * b[i]
		}
		return sum
	}

	_simd_axpy_f32 :: #force_inline proc "contextless" (y, x: [^]f32, a: f32, n: int) {
		for i in 0 ..< n {
			y[i] += a * x[i]
		}
	}

	_simd_dot_bf16_f32 :: #force_inline proc "contextless" (a, b: [^]ml.Bf16, n: int) -> f32 {
		s0, s1, s2, s3: f32
		i := 0
		for ; i + 4 <= n; i += 4 {
			s0 += ml.bf16_to_f32(a[i + 0]) * ml.bf16_to_f32(b[i + 0])
			s1 += ml.bf16_to_f32(a[i + 1]) * ml.bf16_to_f32(b[i + 1])
			s2 += ml.bf16_to_f32(a[i + 2]) * ml.bf16_to_f32(b[i + 2])
			s3 += ml.bf16_to_f32(a[i + 3]) * ml.bf16_to_f32(b[i + 3])
		}
		sum := (s0 + s1) + (s2 + s3)
		for ; i < n; i += 1 {
			sum += ml.bf16_to_f32(a[i]) * ml.bf16_to_f32(b[i])
		}
		return sum
	}

}

Context :: struct {
	using _: ml.Context,

	arena:      mem.Arena,
	persistent: map[rawptr]bool,
}

POISON_TRANSIENT :: #config(ML_CPU_POISON, false)

_backend := ml.Backend{
	clear        = clear,
	forward      = forward,
	backward     = backward,
	update       = update,
	buffer_alloc = buffer_alloc,
	buffer_free  = buffer_free,
	buffer_get   = buffer_get,
	buffer_set   = buffer_set,
	buffer_copy  = buffer_copy,

	buffer_sq_sum_accumulate = buffer_sq_sum_accumulate,
	buffer_scale             = buffer_scale,

	forward_ops  = ml.OPERATION_SET_ALL - {.Linear_Q4_K_Gate_Up_Geglu, .Rmsnorm_Rope_Write_Cache},
	backward_ops = ml.OPERATION_SET_ALL - {
		.Linear_Q4_K, .Linear_Q4_K_Gate_Up_Geglu, .Linear_Q6_K,
		.Rmsnorm_Rope, .Rmsnorm_Rope_Write_Cache, .Add_Rmsnorm,
		.Gelu_Mul, .Lerp_Assign, .Accumulate_Mean,
	},
}

@(require_results)
context_create :: proc(size: int, allocator := context.allocator, loc := #caller_location) -> ^ml.Context {
	_enable_flush_to_zero()

	ctx, ctx_err := builtin.new(Context, allocator=allocator, loc=loc)
	assert(ctx_err == nil, "Failed to allocate Context", loc=loc)

	arena_buf, arena_buf_err := builtin.make([]byte, size, allocator=context.allocator, loc=loc)
	assert(arena_buf_err == nil, "Failed to allocate CPU backend arena data", loc=loc)
	mem.arena_init(&ctx.arena, arena_buf)

	ctx.persistent = builtin.make(map[rawptr]bool, allocator=allocator)

	ml._context_init(ctx, &_backend, allocator, loc)

	return ctx
}

context_destroy :: proc(ctx: ^ml.Context, allocator := context.allocator, loc := #caller_location) {
	ctx := cast(^Context)ctx
	ml._context_destroy(ctx, loc)
	accumulator_bytes := transmute([]byte)ctx.grad_norm_accumulator
	if raw_data(accumulator_bytes) != nil {
		builtin.delete(accumulator_bytes, loc=loc)
	}
	builtin.delete(ctx.arena.data, loc=loc)
	builtin.delete(ctx.persistent)
	builtin.free(ctx, allocator=allocator, loc=loc)
}

clear :: proc(loc: runtime.Source_Code_Location) {
	ctx := cast(^Context)ml.current_context(loc=loc)
	mem.arena_free_all(&ctx.arena)
}

_buffer_get :: #force_inline proc(t: ml.Tensor, kind: ml.Buffer_Kind) -> []f32 {
	bytes := transmute([]byte)t.buffers[kind]
	return ([^]f32)(raw_data(bytes))[:t.count]
}

@(require_results)
data :: #force_inline proc(t: ml.Tensor) -> []f32 {
	bytes := transmute([]byte)t.buffers[.Data]
	return ([^]f32)(raw_data(bytes))[:t.count]
}

@(require_results)
gradient :: #force_inline proc(t: ml.Tensor) -> []f32 {
	bytes := transmute([]byte)t.buffers[.Gradient]
	return ([^]f32)(raw_data(bytes))[:t.count]
}

@(require_results)
data_bf16 :: #force_inline proc(t: ml.Tensor) -> []ml.Bf16 {
	bytes := transmute([]byte)t.buffers[.Data]
	return ([^]ml.Bf16)(raw_data(bytes))[:t.count]
}

@(require_results)
_moment :: #force_inline proc(buffer: ml.Backend_Buffer, count: int) -> []f32 {
	bytes := transmute([]byte)buffer
	return ([^]f32)(raw_data(bytes))[:count]
}

buffer_alloc :: proc(byte_count: int, kind: ml.Buffer_Kind, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	ctx       := cast(^Context)ml.current_context(loc=loc)
	allocator := persist ? context.allocator : mem.arena_allocator(&ctx.arena)

	bytes, err := builtin.make([]byte, byte_count, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate CPU buffer: %v", err, loc=loc)

	if persist {
		ctx.persistent[rawptr(raw_data(bytes))] = true
	} else {
		when POISON_TRANSIENT {
			if kind == .Data {
				for &word in mem.slice_data_cast([]u32, bytes) {
					word = 0x7fc0_0000 // quiet NaN
				}
			}
		}
	}

	return transmute([ml.BACKEND_BUFFER_MAX_SIZE]byte)bytes
}

buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	bytes := transmute([]byte)buffer
	if raw_data(bytes) == nil {
		return
	}
	ctx := cast(^Context)ml.current_context(loc=loc)
	if rawptr(raw_data(bytes)) not_in ctx.persistent {
		return
	}
	builtin.delete_key(&ctx.persistent, rawptr(raw_data(bytes)))
	builtin.delete(bytes, loc=loc)
}

buffer_get :: proc(buffer: ml.Backend_Buffer, dst: []byte, loc: runtime.Source_Code_Location) {
	builtin.copy(dst, transmute([]byte)buffer)
}

buffer_set :: proc(buffer: ml.Backend_Buffer, src: []byte, loc: runtime.Source_Code_Location) {
	builtin.copy(transmute([]byte)buffer, src)
}

buffer_copy :: proc(dst, src: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	builtin.copy(transmute([]byte)dst, transmute([]byte)src)
}

buffer_sq_sum_accumulate :: proc(buffer: ml.Backend_Buffer, count: int, accumulator: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	g   := ([^]f32)(raw_data(transmute([]byte)buffer))[:count]
	acc := (^f64)(raw_data(transmute([]byte)accumulator))

	total := f64(0)
	for i in 0 ..< count {
		total += f64(g[i]) * f64(g[i])
	}
	acc^ += total
}

buffer_scale :: proc(buffer: ml.Backend_Buffer, count: int, scale: f32, loc: runtime.Source_Code_Location) {
	g := ([^]f32)(raw_data(transmute([]byte)buffer))[:count]
	for i in 0 ..< count {
		g[i] *= scale
	}
}

update :: proc(opt: ml.Optimizer, t: ml.Tensor, m_buf, v_buf: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	g := gradient(t)
	m := _moment(m_buf, t.count)
	v := _moment(v_buf, t.count)

	assert(g != nil, "Tensor Gradient is nil", loc=loc)
	assert(m != nil, "Optimizer moment m is nil", loc=loc)
	assert(v != nil, "Optimizer moment v is nil", loc=loc)

	d_bf: [^]ml.Bf16
	d_f32: []f32
	#partial switch t.type {
	case .F32:  d_f32 = data(t)
	case .Bf16: d_bf  = ([^]ml.Bf16)(raw_data(transmute([]byte)t.buffers[.Data]))
	case:       panic("only F32 and Bf16 parameters are trainable", loc)
	}

	for i in 0 ..< t.count {
		grad := g[i]

		m[i] = opt.beta1 * m[i] + (1 - opt.beta1) * grad
		v[i] = opt.beta2 * v[i] + (1 - opt.beta2) * grad * grad

		m_hat := m[i] / opt.bias_correction1
		v_hat := v[i] / opt.bias_correction2

		weight := t.type == .Bf16 ? ml.bf16_to_f32(d_bf[i]) : d_f32[i]
		weight = weight * (1 - opt.learning_rate * opt.weight_decay) - opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)
		if t.type == .Bf16 {
			d_bf[i] = ml.bf16_from_f32(weight)
		} else {
			d_f32[i] = weight
		}

		g[i] = 0
	}
}

_alloc_scratch :: proc(op: ^ml.Operation, loc: runtime.Source_Code_Location) {
	#partial switch &v in op.variant {
	case ml.Attention:
		token_count := op.input.shape[0]
		v.softmax_outputs = ml.scratch(.F32, {v.n_q_heads, token_count, token_count}, loc=loc)
		if ml.is_training(loc=loc) {
			v.d_p_scratch = ml.scratch(.F32, {v.n_q_heads, token_count}, loc=loc)
		}
	case ml.Rope:
		token_count := op.input.shape[0]
		input_size  := op.input.shape[op.input.rank - 1]
		half_head   := (input_size / v.head_count) / 2
		v.cos_cache = ml.scratch(.F32, {token_count * half_head}, loc=loc)
		v.sin_cache = ml.scratch(.F32, {token_count * half_head}, loc=loc)
	case ml.Layernorm:
		count := ml.len(op.input) / op.input.shape[op.input.rank - 1]
		v.mean = ml.scratch(.F32, {count}, loc=loc)
		v.rstd = ml.scratch(.F32, {count}, loc=loc)
	case ml.Rmsnorm:
		count := ml.len(op.input) / op.input.shape[op.input.rank - 1]
		v.rstd = ml.scratch(.F32, {count}, loc=loc)
	case ml.Cross_Entropy:
		shape := op.input.shape
		v.probabilities = ml.scratch(op.input.type, shape[:op.input.rank], loc=loc)
	}
}

forward :: proc(op: ^ml.Operation, loc: runtime.Source_Code_Location) {
	_alloc_scratch(op, loc)
	op := op^
	switch _ in op.variant {
	case ml.Add:                add_forward                (op)
	case ml.Sub:                sub_forward                (op)
	case ml.Mul:                mul_forward                (op)
	case ml.Div:                div_forward                (op)
	case ml.Exp:                exp_forward                (op)
	case ml.Sqrt:               sqrt_forward               (op, loc)
	case ml.Clamp:              clamp_forward              (op)
	case ml.Min:                min_forward                (op)
	case ml.Max:                max_forward                (op)
	case ml.Mean:               mean_forward               (op)
	case ml.Sum:                sum_forward                (op)
	case ml.Max_Reduce:         max_reduce_forward         (op)
	case ml.Im2col:             im2col_forward             (op)
	case ml.Max_Pool2d:         max_pool2d_forward         (op)
	case ml.Avg_Pool2d:         avg_pool2d_forward         (op)
	case ml.Transpose:          transpose_forward          (op)
	case ml.Select:             select_forward             (op)
	case ml.Slice:              slice_forward              (op)
	case ml.Slice_Trailing:     slice_trailing_forward     (op)
	case ml.Slice_Leading:      slice_leading_forward      (op)
	case ml.Concat:             concat_forward             (op)
	case ml.Linear:             linear_forward             (op)
	case ml.Linear_Q4_K:        linear_q4_k_forward        (op)
	case ml.Linear_Q4_K_Gate_Up_Geglu: panic("Linear_Q4_K_Gate_Up_Geglu is unreachable (the op decomposes when the capability is absent)", loc)
	case ml.Linear_Q6_K:        linear_q6_k_forward        (op)
	case ml.Rope:               rope_forward               (op)
	case ml.Layernorm:          layernorm_forward          (op)
	case ml.Rmsnorm:            rmsnorm_forward            (op)
	case ml.Rmsnorm_Rope:       rmsnorm_rope_forward       (op)
	case ml.Rmsnorm_Rope_Write_Cache: panic("backend does not advertise the Rmsnorm_Rope_Write_Cache capability", loc)
	case ml.Add_Rmsnorm:        add_rmsnorm_forward        (op)
	case ml.Softmax:            softmax_forward            (op)
	case ml.Entropy:            entropy_forward            (op)
	case ml.Log_Softmax:        log_softmax_forward        (op)
	case ml.Mean_Squared_Error: mean_squared_error_forward (op)
	case ml.Smooth_L1:          smooth_l1_forward          (op)
	case ml.Cross_Entropy:      cross_entropy_forward      (op)
	case ml.Relu:               relu_forward               (op)
	case ml.Sigmoid:            sigmoid_forward            (op)
	case ml.Gelu:               gelu_forward               (op)
	case ml.Gelu_Mul:           gelu_mul_forward           (op)
	case ml.Silu:               silu_forward               (op)
	case ml.Tanh:               tanh_forward               (op)
	case ml.Batched_Matmul:     batched_matmul_forward     (op)
	case ml.Permute:            permute_forward            (op)
	case ml.Causal_Mask:        causal_mask_forward        (op)
	case ml.Attention:          attention_forward          (op)
	case ml.Attention_Cache:    attention_cache_forward    (op)
	case ml.Cast:               cast_forward               (op)
	case ml.Lerp_Assign:        lerp_assign_forward        (op)
	case ml.Accumulate_Mean:    accumulate_mean_forward    (op)
	}
}

backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	switch _ in op.variant {
	case ml.Add:                add_backward               (op)
	case ml.Sub:                sub_backward               (op)
	case ml.Mul:                mul_backward               (op)
	case ml.Div:                div_backward               (op)
	case ml.Exp:                exp_backward               (op)
	case ml.Sqrt:               sqrt_backward              (op)
	case ml.Clamp:              clamp_backward             (op)
	case ml.Min:                min_backward               (op)
	case ml.Max:                max_backward               (op)
	case ml.Mean:               mean_backward              (op)
	case ml.Sum:                sum_backward               (op)
	case ml.Max_Reduce:         max_reduce_backward        (op)
	case ml.Im2col:             im2col_backward            (op)
	case ml.Max_Pool2d:         max_pool2d_backward        (op)
	case ml.Avg_Pool2d:         avg_pool2d_backward        (op)
	case ml.Transpose:          transpose_backward         (op)
	case ml.Select:             select_backward            (op)
	case ml.Slice:              slice_backward             (op)
	case ml.Slice_Trailing:     slice_trailing_backward    (op)
	case ml.Slice_Leading:      slice_leading_backward     (op)
	case ml.Concat:             concat_backward            (op)
	case ml.Linear:             linear_backward            (op)
	case ml.Linear_Q4_K:        panic("Linear_Q4_K is forward-only", loc)
	case ml.Linear_Q4_K_Gate_Up_Geglu: panic("Linear_Q4_K_Gate_Up_Geglu is forward-only", loc)
	case ml.Linear_Q6_K:        panic("Linear_Q6_K is forward-only", loc)
	case ml.Rope:               rope_backward              (op)
	case ml.Layernorm:          layernorm_backward         (op)
	case ml.Rmsnorm:            rmsnorm_backward           (op)
	case ml.Rmsnorm_Rope:       panic("Rmsnorm_Rope is forward-only", loc)
	case ml.Rmsnorm_Rope_Write_Cache: panic("Rmsnorm_Rope_Write_Cache is forward-only", loc)
	case ml.Add_Rmsnorm:        panic("Add_Rmsnorm is forward-only", loc)
	case ml.Softmax:            softmax_backward           (op)
	case ml.Entropy:            entropy_backward           (op)
	case ml.Log_Softmax:        log_softmax_backward       (op)
	case ml.Mean_Squared_Error: mean_squared_error_backward(op)
	case ml.Smooth_L1:          smooth_l1_backward         (op)
	case ml.Cross_Entropy:      cross_entropy_backward     (op)
	case ml.Relu:               relu_backward              (op)
	case ml.Sigmoid:            sigmoid_backward           (op)
	case ml.Gelu:               gelu_backward              (op)
	case ml.Gelu_Mul:           panic("Gelu_Mul is forward-only", loc)
	case ml.Silu:               silu_backward              (op)
	case ml.Tanh:               tanh_backward              (op)
	case ml.Batched_Matmul:     batched_matmul_backward    (op)
	case ml.Permute:            permute_backward           (op)
	case ml.Causal_Mask:        causal_mask_backward       (op)
	case ml.Attention:          attention_backward         (op)
	case ml.Attention_Cache:    attention_cache_backward   (op, loc)
	case ml.Cast:               cast_backward              (op)
	case ml.Lerp_Assign:        panic("Lerp_Assign is forward-only", loc)
	case ml.Accumulate_Mean:    panic("Accumulate_Mean is forward-only", loc)
	}
}

lerp_assign_forward :: proc(op: ml.Operation) {
	dst    := data(op.output)
	source := data(op.variant.(ml.Lerp_Assign).source)
	alpha  := op.variant.(ml.Lerp_Assign).alpha
	one_minus := 1 - alpha
	for i in 0 ..< builtin.len(dst) {
		dst[i] = one_minus * dst[i] + alpha * source[i]
	}
}

accumulate_mean_forward :: proc(op: ml.Operation) {
	dst    := data(op.output)
	source := data(op.input)
	sum: f32
	for v in source {
		sum += v
	}
	dst[0] += sum / f32(builtin.len(source))
}

cast_forward :: proc(op: ml.Operation) {
	src_bytes := transmute([]byte)op.input.buffers[.Data]
	dst_bytes := transmute([]byte)op.output.buffers[.Data]
	_cast_bytes(src_bytes, op.input.type, dst_bytes, op.output.type, op.input.count)
}

cast_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	src_grad := gradient(op.output)
	dst_grad := gradient(op.input)
	for i in 0 ..< op.input.count {
		dst_grad[i] += src_grad[i]
	}
}

_cast_bytes :: proc(src: []byte, src_type: ml.Data_Type, dst: []byte, dst_type: ml.Data_Type, count: int) {
	src_f32  := ([^]f32    )(raw_data(src))[:count] if src_type == .F32  else nil
	src_bf16 := ([^]ml.Bf16)(raw_data(src))[:count] if src_type == .Bf16 else nil

	dst_f32  := ([^]f32    )(raw_data(dst))[:count] if dst_type == .F32  else nil
	dst_bf16 := ([^]ml.Bf16)(raw_data(dst))[:count] if dst_type == .Bf16 else nil

	for i in 0 ..< count {
		v: f32
		#partial switch src_type {
		case .F32:  v = src_f32 [i]
		case .Bf16: v = ml.bf16_to_f32(src_bf16[i])
		}
		#partial switch dst_type {
		case .F32:  dst_f32 [i] = v
		case .Bf16: dst_bf16[i] = ml.bf16_from_f32(v)
		}
	}
}

_cast_bytes_accumulate :: proc(src: []byte, src_type: ml.Data_Type, dst: []byte, dst_type: ml.Data_Type, count: int) {
	src_f32  := ([^]f32    )(raw_data(src))[:count] if src_type == .F32  else nil
	src_bf16 := ([^]ml.Bf16)(raw_data(src))[:count] if src_type == .Bf16 else nil

	dst_f32  := ([^]f32    )(raw_data(dst))[:count] if dst_type == .F32  else nil
	dst_bf16 := ([^]ml.Bf16)(raw_data(dst))[:count] if dst_type == .Bf16 else nil

	for i in 0 ..< count {
		v: f32
		#partial switch src_type {
		case .F32:  v = src_f32 [i]
		case .Bf16: v = ml.bf16_to_f32(src_bf16[i])
		}
		#partial switch dst_type {
		case .F32:  dst_f32 [i] += v
		case .Bf16: dst_bf16[i]  = ml.bf16_from_f32(ml.bf16_to_f32(dst_bf16[i]) + v)
		}
	}
}

_broadcast_tiling :: #force_inline proc(a, b: ml.Tensor) -> (stride, width: int) {
	width  = ml.len(b)
	stride = ml.len(a) / width
	return
}

_typed_data :: #force_inline proc($T: typeid, t: ml.Tensor) -> [^]T {
	return ([^]T)(raw_data(transmute([]byte)t.buffers[.Data]))
}

_load :: #force_inline proc "contextless" (p: [^]$T, i: int) -> f32 {
	when T == ml.Bf16 {
		return ml.bf16_to_f32(p[i])
	} else {
		return p[i]
	}
}

_store :: #force_inline proc "contextless" (p: [^]$T, i: int, value: f32) {
	when T == ml.Bf16 {
		p[i] = ml.bf16_from_f32(value)
	} else {
		p[i] = value
	}
}

add_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _add_forward_impl(f32,      op)
	case .Bf16: _add_forward_impl(ml.Bf16, op)
	}
}

_add_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Add).b
	stride, width := _broadcast_tiling(a, b)

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	op_ := _typed_data(T, output)
	#no_bounds_check for i in 0 ..< stride {
		row := i * width
		for j in 0 ..< width {
			_store(op_, row + j, _load(ap, row + j) + _load(bp, j))
		}
	}
}

add_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b      := op.variant.(ml.Add).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy := gradient(a), gradient(b), gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	#no_bounds_check for i in 0 ..< stride {
		row_da := da[i * width:]
		row_dy := dy[i * width:]
		for j in 0 ..< width {
			if have_da { row_da[j] += row_dy[j] }
			if have_db { db[j]     += row_dy[j] }
		}
	}
}

sub_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _sub_forward_impl(f32,      op)
	case .Bf16: _sub_forward_impl(ml.Bf16, op)
	}
}

_sub_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Sub).b
	stride, width := _broadcast_tiling(a, b)

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, _load(ap, o) - _load(bp, j))
		}
	}
}

sub_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b      := op.variant.(ml.Sub).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy := gradient(a), gradient(b), gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if have_da { da[o] += dy[o] }
			if have_db { db[j] -= dy[o] }
		}
	}
}

mul_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _mul_forward_impl(f32,      op)
	case .Bf16: _mul_forward_impl(ml.Bf16, op)
	}
}

_mul_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Mul).b
	stride, width := _broadcast_tiling(a, b)

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, _load(ap, o) * _load(bp, j))
		}
	}
}

mul_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _mul_backward_impl(f32,      op)
	case .Bf16: _mul_backward_impl(ml.Bf16, op)
	}
}

_mul_backward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Mul).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy := gradient(a), gradient(b), gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if have_da { da[o] += dy[o] * _load(bp, j) }
			if have_db { db[j] += dy[o] * _load(ap, o) }
		}
	}
}

div_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _div_forward_impl(f32,      op)
	case .Bf16: _div_forward_impl(ml.Bf16, op)
	}
}

_div_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Div).b
	stride, width := _broadcast_tiling(a, b)

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, _load(ap, o) / _load(bp, j))
		}
	}
}

div_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _div_backward_impl(f32,      op)
	case .Bf16: _div_backward_impl(ml.Bf16, op)
	}
}

_div_backward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Div).b
	stride, width := _broadcast_tiling(a, b)

	da, db, dy := gradient(a), gradient(b), gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			a_v := _load(ap, o)
			b_v := _load(bp, j)
			if have_da { da[o] += dy[o] / b_v }
			if have_db { db[j] += dy[o] * (-a_v / (b_v * b_v)) }
		}
	}
}

exp_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _exp_forward_impl(f32,      op)
	case .Bf16: _exp_forward_impl(ml.Bf16, op)
	}
}

_exp_forward_impl :: proc($T: typeid, op: ml.Operation) {
	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for i in 0 ..< ml.len(op.input) {
		_store(yp, i, math.exp(_load(xp, i)))
	}
}

exp_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _exp_backward_impl(f32,      op)
	case .Bf16: _exp_backward_impl(ml.Bf16, op)
	}
}

_exp_backward_impl :: proc($T: typeid, op: ml.Operation) {
	dx, dy := gradient(op.input), gradient(op.output)
	yp := _typed_data(T, op.output)
	for i in 0 ..< ml.len(op.input) {
		dx[i] += _load(yp, i) * dy[i]
	}
}

sqrt_forward :: proc(op: ml.Operation, loc := #caller_location) {
	input  := op.input
	output := op.output

	assert(input.type == .F32, "Sqrt is F32-only", loc=loc)

	for i in 0 ..< ml.len(input) {
		data(output)[i] = math.sqrt(data(input)[i])
	}
}

sqrt_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	dx, dy := gradient(input), gradient(output)
	y := data(output)
	for i in 0 ..< ml.len(input) {
		if y[i] > 0 {
			dx[i] += 0.5 / y[i] * dy[i]
		}
	}
}

clamp_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< ml.len(input) {
		data(output)[i] = math.clamp(data(input)[i], min_val, max_val)
	}
}

clamp_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant := op.variant.(ml.Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< ml.len(input) {
		if data(input)[i] >= min_val && data(input)[i] <= max_val {
			gradient(input)[i] += gradient(output)[i]
		}
	}
}

min_forward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Min).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp, op_ := data(a), data(b), data(output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			op_[o] = math.min(ap[o], bp[j])
		}
	}
}

min_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Min).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp := data(a), data(b)
	da, db, dy := gradient(a), gradient(b), gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if ap[o] <= bp[j] {
				if have_da { da[o] += dy[o] }
			} else {
				if have_db { db[j] += dy[o] }
			}
		}
	}
}

max_forward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Max).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp, op_ := data(a), data(b), data(output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			op_[o] = math.max(ap[o], bp[j])
		}
	}
}

max_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Max).b
	stride, width := _broadcast_tiling(a, b)

	ap, bp := data(a), data(b)
	da, db, dy := gradient(a), gradient(b), gradient(output)
	have_da, have_db := ml.has_gradient(a), ml.has_gradient(b)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			if ap[o] >= bp[j] {
				if have_da { da[o] += dy[o] }
			} else {
				if have_db { db[j] += dy[o] }
			}
		}
	}
}

mean_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _mean_forward_impl(f32,      op)
	case .Bf16: _mean_forward_impl(ml.Bf16, op)
	}
}

_mean_forward_impl :: proc($T: typeid, op: ml.Operation) {
	count := ml.len(op.output)
	size  := ml.len(op.input) / count

	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		sum: f32
		for i in 0 ..< size {
			sum += _load(xp, sample * size + i)
		}
		_store(yp, sample, sum / f32(size))
	}
}

mean_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	dx, dy := gradient(input), gradient(output)
	for sample in 0 ..< count {
		gradient_per_element := dy[sample] / f32(size)

		for i in 0 ..< size {
			dx[sample * size + i] += gradient_per_element
		}
	}
}

sum_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _sum_forward_impl(f32,      op)
	case .Bf16: _sum_forward_impl(ml.Bf16, op)
	}
}

_sum_forward_impl :: proc($T: typeid, op: ml.Operation) {
	count := ml.len(op.output)
	size  := ml.len(op.input) / count

	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		total: f32
		for i in 0 ..< size {
			total += _load(xp, sample * size + i)
		}
		_store(yp, sample, total)
	}
}

sum_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	dx, dy := gradient(input), gradient(output)
	for sample in 0 ..< count {
		for i in 0 ..< size {
			dx[sample * size + i] += dy[sample]
		}
	}
}

max_reduce_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _max_reduce_forward_impl(f32,      op)
	case .Bf16: _max_reduce_forward_impl(ml.Bf16, op)
	}
}

_max_reduce_forward_impl :: proc($T: typeid, op: ml.Operation) {
	count := ml.len(op.output)
	size  := ml.len(op.input) / count

	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		best := _load(xp, sample * size)
		for i in 1 ..< size {
			value := _load(xp, sample * size + i)
			if value > best {
				best = value
			}
		}
		_store(yp, sample, best)
	}
}

max_reduce_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _max_reduce_backward_impl(f32,      op)
	case .Bf16: _max_reduce_backward_impl(ml.Bf16, op)
	}
}

_max_reduce_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	xp := _typed_data(T, input)
	dx, dy := gradient(input), gradient(output)
	for sample in 0 ..< count {
		best_index := 0
		best_value := _load(xp, sample * size)
		for i in 1 ..< size {
			value := _load(xp, sample * size + i)
			if value > best_value {
				best_value = value
				best_index = i
			}
		}
		dx[sample * size + best_index] += dy[sample]
	}
}

im2col_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _im2col_forward_impl(f32,      op)
	case .Bf16: _im2col_forward_impl(ml.Bf16, op)
	}
}

_im2col_forward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Im2col)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)

	patch_size := v.kernel_h * v.kernel_w * c
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< v.out_h {
			for ox in 0 ..< v.out_w {
				row := ((n * v.out_h) + oy) * v.out_w + ox
				for ky in 0 ..< v.kernel_h {
					iy := oy * v.stride_h - v.pad_h + ky
					for kx in 0 ..< v.kernel_w {
						ix := ox * v.stride_w - v.pad_w + kx
						col_base := ((ky * v.kernel_w) + kx) * c
						if iy >= 0 && iy < h && ix >= 0 && ix < w {
							src_base := (((n * h) + iy) * w + ix) * c
							for ci in 0 ..< c {
								_store(yp, row * patch_size + col_base + ci, _load(xp, src_base + ci))
							}
						} else {
							for ci in 0 ..< c {
								_store(yp, row * patch_size + col_base + ci, 0)
							}
						}
					}
				}
			}
		}
	}
}

im2col_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	v := op.variant.(ml.Im2col)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]

	dx, dy := gradient(input), gradient(op.output)

	patch_size := v.kernel_h * v.kernel_w * c
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< v.out_h {
			for ox in 0 ..< v.out_w {
				row := ((n * v.out_h) + oy) * v.out_w + ox
				for ky in 0 ..< v.kernel_h {
					iy := oy * v.stride_h - v.pad_h + ky
					if iy < 0 || iy >= h { continue }
					for kx in 0 ..< v.kernel_w {
						ix := ox * v.stride_w - v.pad_w + kx
						if ix < 0 || ix >= w { continue }
						col_base := ((ky * v.kernel_w) + kx) * c
						src_base := (((n * h) + iy) * w + ix) * c
						for ci in 0 ..< c {
							dx[src_base + ci] += dy[row * patch_size + col_base + ci]
						}
					}
				}
			}
		}
	}
}

max_pool2d_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _max_pool2d_forward_impl(f32,      op)
	case .Bf16: _max_pool2d_forward_impl(ml.Bf16, op)
	}
}

_max_pool2d_forward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Max_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)

	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					best := _load(xp, (((n * h) + oy * v.stride_h) * w + ox * v.stride_w) * c + ci)
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							value := _load(xp, (((n * h) + iy) * w + ix) * c + ci)
							if value > best {
								best = value
							}
						}
					}
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					_store(yp, out_index, best)
				}
			}
		}
	}
}

max_pool2d_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _max_pool2d_backward_impl(f32,      op)
	case .Bf16: _max_pool2d_backward_impl(ml.Bf16, op)
	}
}

_max_pool2d_backward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Max_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	xp := _typed_data(T, input)
	dx, dy := gradient(input), gradient(op.output)

	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					best_iy := oy * v.stride_h
					best_ix := ox * v.stride_w
					best_value := _load(xp, (((n * h) + best_iy) * w + best_ix) * c + ci)
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							value := _load(xp, (((n * h) + iy) * w + ix) * c + ci)
							if value > best_value {
								best_value = value
								best_iy    = iy
								best_ix    = ix
							}
						}
					}
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					dx[(((n * h) + best_iy) * w + best_ix) * c + ci] += dy[out_index]
				}
			}
		}
	}
}

avg_pool2d_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _avg_pool2d_forward_impl(f32,      op)
	case .Bf16: _avg_pool2d_forward_impl(ml.Bf16, op)
	}
}

_avg_pool2d_forward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Avg_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)

	window := f32(v.kernel_h * v.kernel_w)
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					total: f32
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							total += _load(xp, (((n * h) + iy) * w + ix) * c + ci)
						}
					}
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					_store(yp, out_index, total / window)
				}
			}
		}
	}
}

avg_pool2d_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _avg_pool2d_backward_impl(f32,      op)
	case .Bf16: _avg_pool2d_backward_impl(ml.Bf16, op)
	}
}

_avg_pool2d_backward_impl :: proc($T: typeid, op: ml.Operation) {
	v := op.variant.(ml.Avg_Pool2d)
	input := op.input
	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]
	out_h := op.output.shape[1]
	out_w := op.output.shape[2]

	dx, dy := gradient(input), gradient(op.output)

	window := f32(v.kernel_h * v.kernel_w)
	for n in 0 ..< input.shape[0] {
		for oy in 0 ..< out_h {
			for ox in 0 ..< out_w {
				for ci in 0 ..< c {
					out_index := (((n * out_h) + oy) * out_w + ox) * c + ci
					share := dy[out_index] / window
					for ky in 0 ..< v.kernel_h {
						iy := oy * v.stride_h + ky
						for kx in 0 ..< v.kernel_w {
							ix := ox * v.stride_w + kx
							dx[(((n * h) + iy) * w + ix) * c + ci] += share
						}
					}
				}
			}
		}
	}
}

transpose_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	rows    := input.shape[0]
	columns := input.shape[1]

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			data(output)[j * rows + i] = data(input)[i * columns + j]
		}
	}
}

transpose_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output
	rows    := input.shape[0]
	columns := input.shape[1]

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			gradient(input)[i * columns + j] += gradient(output)[j * rows + i]
		}
	}
}

select_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(output) / builtin.len(indices)

	elem_size := ml.data_type_size(input.type)
	row_bytes := size * elem_size
	src_bytes := transmute([]byte)input.buffers [.Data]
	dst_bytes := transmute([]byte)output.buffers[.Data]

	for index, i in indices {
		src_off := index * row_bytes
		dst_off := i     * row_bytes
		builtin.copy(dst_bytes[dst_off:dst_off + row_bytes], src_bytes[src_off:src_off + row_bytes])
	}
}

select_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	weight, output := op.input, op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(output) / builtin.len(indices)

	dw, dy := gradient(weight), gradient(output)
	for i in 0 ..< builtin.len(indices) {
		for j in 0 ..< size {
			dw[indices[i] * size + j] += dy[i * size + j]
		}
	}
}

slice_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _slice_forward_impl(f32,      op)
	case .Bf16: _slice_forward_impl(ml.Bf16, op)
	}
}

_slice_forward_impl :: proc($T: typeid, op: ml.Operation) {
	start := op.variant.(ml.Slice).start
	xp := _typed_data(T, op.input)
	yp := _typed_data(T, op.output)
	for i in 0 ..< ml.len(op.output) {
		yp[i] = xp[start + i]
	}
}

slice_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice)
	start   := variant.start

	dx, dy := gradient(input), gradient(output)
	for i in 0 ..< ml.len(output) {
		dx[start + i] += dy[i]
	}
}

slice_trailing_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _slice_trailing_forward_impl(f32,      op)
	case .Bf16: _slice_trailing_forward_impl(ml.Bf16, op)
	}
}

_slice_trailing_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	start := op.variant.(ml.Slice_Trailing).start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml._leading_count(input)

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			yp[out_off + i] = xp[in_off + i]
		}
	}
}

slice_trailing_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice_Trailing)
	start   := variant.start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml._leading_count(input)

	dx, dy := gradient(input), gradient(output)
	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			dx[in_off + i] += dy[out_off + i]
		}
	}
}

slice_leading_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _slice_leading_forward_impl(f32,      op)
	case .Bf16: _slice_leading_forward_impl(ml.Bf16, op)
	}
}

_slice_leading_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	start := op.variant.(ml.Slice_Leading).start

	leading  := input.shape[0]
	row_size := ml.len(input) / leading
	count    := output.shape[0] * row_size
	in_off   := start * row_size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	for i in 0 ..< count {
		yp[i] = xp[in_off + i]
	}
}

slice_leading_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice_Leading)
	start   := variant.start

	leading  := input.shape[0]
	row_size := ml.len(input) / leading
	count    := output.shape[0] * row_size
	in_off   := start * row_size

	dx, dy := gradient(input), gradient(output)
	for i in 0 ..< count {
		dx[in_off + i] += dy[i]
	}
}

concat_forward :: proc(op: ml.Operation) {
	#partial switch op.output.type {
	case .F32:  _concat_forward_impl(f32,      op)
	case .Bf16: _concat_forward_impl(ml.Bf16, op)
	}
}

_concat_forward_impl :: proc($T: typeid, op: ml.Operation) {
	output := op.output
	inputs := op.variant.(ml.Concat).inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

	yp := _typed_data(T, output)
	dst_col := 0
	for input in inputs {
		xp          := _typed_data(T, input)
		in_trailing := input.shape[input.rank - 1]
		for r in 0 ..< leading {
			out_off := r * out_trailing + dst_col
			in_off  := r * in_trailing
			for i in 0 ..< in_trailing {
				yp[out_off + i] = xp[in_off + i]
			}
		}
		dst_col += in_trailing
	}
}

concat_backward :: proc(op: ml.Operation) {
	output := op.output

	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

	dy := gradient(output)
	src_col := 0
	for input in inputs {
		in_trailing := input.shape[input.rank - 1]
		if ml.has_gradient(input) {
			dx := gradient(input)
			for r in 0 ..< leading {
				out_off := r * out_trailing + src_col
				in_off  := r * in_trailing
				for i in 0 ..< in_trailing {
					dx[in_off + i] += dy[out_off + i]
				}
			}
		}
		src_col += in_trailing
	}
}

linear_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  linear_forward_f32 (op)
	case .Bf16: linear_forward_bf16(op)
	}
}

linear_forward_f32 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(op.input) / input_size

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	work := count * output_size * input_size

	if count >= output_size {
		parallelize(count, count, jd, proc(c: int, jd: Job_Data) {
			op := jd.op
			input, output := op.input, op.output
			weight      := op.variant.(ml.Linear).weight
			output_size := weight.shape[0]
			input_size  := weight.shape[1]

			input_ptr  := ([^]f32)(raw_data(data(input)))
			output_ptr := ([^]f32)(raw_data(data(output)))
			weight_ptr := ([^]f32)(raw_data(data(weight)))

			x := input_ptr[c * input_size:]
			y := output_ptr[c * output_size:]

			for o in 0 ..< output_size {
				y[o] = _simd_dot_f32(weight_ptr[o * input_size:], x, input_size)
			}
		}, work=work)
		return
	}

	parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		input, output := op.input, op.output
		weight      := op.variant.(ml.Linear).weight
		output_size := weight.shape[0]
		input_size  := weight.shape[1]

		input_ptr  := ([^]f32)(raw_data(data(input)))
		output_ptr := ([^]f32)(raw_data(data(output)))
		weight_ptr := ([^]f32)(raw_data(data(weight)))

		w_row := weight_ptr[o * input_size:]
		for c in 0 ..< jd.count {
			x := input_ptr[c * input_size:]
			output_ptr[c * output_size + o] = _simd_dot_f32(w_row, x, input_size)
		}
	}, work=work)
}

linear_forward_bf16 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	count       := ml.len(op.input) / weight.shape[1]

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		input, output := op.input, op.output
		weight      := op.variant.(ml.Linear).weight
		output_size := weight.shape[0]
		input_size  := weight.shape[1]

		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		w_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Data]))

		w_row := w_bf[o * input_size:]
		for c in 0 ..< jd.count {
			x_row := x_bf[c * input_size:]
			y_bf[c * output_size + o] = ml.bf16_from_f32(_simd_dot_bf16_f32(w_row, x_row, input_size))
		}
	})
}

linear_q4_k_forward :: proc(op: ml.Operation) {
	v := op.variant.(ml.Linear_Q4_K)
	output_size := v.weight.shape[0]
	input_size  := v.weight.shape[1]
	count       := ml.len(op.input) / input_size

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Linear_Q4_K)
		output_size := v.weight.shape[0]
		input_size  := v.weight.shape[1]
		num_blocks  := input_size / ml.K_QUANT_BLOCK_SIZE
		row_bytes   := num_blocks * ml.Q4_K_BLOCK_BYTES

		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Data]))
		w_pk := transmute([]byte)v.weight.buffers[.Data]

		w_row   := w_pk[o * row_bytes : (o + 1) * row_bytes]
		dequant := make([]f32, input_size)
		defer delete(dequant)
		ml.dequantize_q4_k(w_row, dequant)

		for c in 0 ..< jd.count {
			x_row := x_bf[c * input_size:]
			total: f32
			for k in 0 ..< input_size {
				total += dequant[k] * ml.bf16_to_f32(x_row[k])
			}
			y_bf[c * output_size + o] = ml.bf16_from_f32(total)
		}
	})
}

linear_q6_k_forward :: proc(op: ml.Operation) {
	v := op.variant.(ml.Linear_Q6_K)
	output_size := v.weight.shape[0]
	input_size  := v.weight.shape[1]
	count       := ml.len(op.input) / input_size

	Job_Data :: struct {
		op:    ml.Operation,
		count: int,
	}
	jd := Job_Data{op = op, count = count}

	parallelize(output_size, output_size, jd, proc(o: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Linear_Q6_K)
		output_size := v.weight.shape[0]
		input_size  := v.weight.shape[1]
		num_blocks  := input_size / ml.K_QUANT_BLOCK_SIZE
		row_bytes   := num_blocks * ml.Q6_K_BLOCK_BYTES

		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Data]))
		w_pk := transmute([]byte)v.weight.buffers[.Data]

		w_row   := w_pk[o * row_bytes : (o + 1) * row_bytes]
		dequant := make([]f32, input_size)
		defer delete(dequant)
		ml.dequantize_q6_k(w_row, dequant)

		for c in 0 ..< jd.count {
			x_row := x_bf[c * input_size:]
			total: f32
			for k in 0 ..< input_size {
				total += dequant[k] * ml.bf16_to_f32(x_row[k])
			}
			y_bf[c * output_size + o] = ml.bf16_from_f32(total)
		}
	})
}

linear_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  linear_backward_f32 (op)
	case .Bf16: linear_backward_bf16(op)
	}
}

linear_backward_bf16 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	count       := ml.len(op.input) / weight.shape[1]

	if ml.has_gradient(weight) {
		parallelize(output_size, output_size, op, proc(o: int, op: ml.Operation) {
			weight      := op.variant.(ml.Linear).weight
			input_size  := weight.shape[1]
			output_size := weight.shape[0]
			count       := ml.len(op.input) / input_size

			x_bf := data_bf16(op.input)
			dy   := gradient(op.output)
			dw   := gradient(weight)

			dw_row := dw[o * input_size:]
			for k in 0 ..< input_size {
				acc: f32
				for c in 0 ..< count {
					acc += ml.bf16_to_f32(x_bf[c * input_size + k]) * dy[c * output_size + o]
				}
				dw_row[k] += acc
			}
		})
	}

	if ml.has_gradient(op.input) {
		parallelize(count, count, op, proc(c: int, op: ml.Operation) {
			weight      := op.variant.(ml.Linear).weight
			input_size  := weight.shape[1]
			output_size := weight.shape[0]

			w_bf := data_bf16(weight)
			dy   := gradient(op.output)
			dx   := gradient(op.input)

			dx_row := dx[c * input_size:]
			dy_row := dy[c * output_size:]
			for k in 0 ..< input_size {
				acc: f32
				for o in 0 ..< output_size {
					acc += ml.bf16_to_f32(w_bf[o * input_size + k]) * dy_row[o]
				}
				dx_row[k] += acc
			}
		})
	}
}

linear_backward_f32 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(op.input) / input_size

	work := count * output_size * input_size

	if ml.has_gradient(weight) {
		parallelize(output_size, output_size, op, proc(o: int, op: ml.Operation) {
			input, output := op.input, op.output
			weight      := op.variant.(ml.Linear).weight
			output_size := weight.shape[0]
			input_size  := weight.shape[1]
			count       := ml.len(input) / input_size

			input_data_ptr  := ([^]f32)(raw_data(data(input)))
			output_grad_ptr := ([^]f32)(raw_data(gradient(output)))
			weight_grad_ptr := ([^]f32)(raw_data(gradient(weight)))

			w_grad := weight_grad_ptr[o * input_size:]

			for b in 0 ..< count {
				dout := output_grad_ptr[b * output_size + o]
				if dout == 0 {
					continue
				}
				x := input_data_ptr[b * input_size:]
				_simd_axpy_f32(w_grad, x, dout, input_size)
			}
		}, work=work)
	}

	if ml.has_gradient(op.input) {
		parallelize(count, count, op, proc(b: int, op: ml.Operation) {
			input, output := op.input, op.output
			weight      := op.variant.(ml.Linear).weight
			output_size := weight.shape[0]
			input_size  := weight.shape[1]

			input_grad_ptr  := ([^]f32)(raw_data(gradient(input)))
			output_grad_ptr := ([^]f32)(raw_data(gradient(output)))
			weight_data_ptr := ([^]f32)(raw_data(data(weight)))

			dx := input_grad_ptr [b * input_size:]
			dy := output_grad_ptr[b * output_size:]

			for o in 0 ..< output_size {
				dout := dy[o]
				if dout == 0 {
					continue
				}
				w_data := weight_data_ptr[o * input_size:]
				_simd_axpy_f32(dx, w_data, dout, input_size)
			}
		}, work=work)
	}
}

rope_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rope_forward_impl(f32,      op)
	case .Bf16: _rope_forward_impl(ml.Bf16, op)
	}
}

_rope_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input             := op.input
	output            := op.output
	variant           := op.variant.(ml.Rope)
	head_count        := variant.head_count
	base              := variant.base
	pos_offset        := variant.position_offset
	rotate_pair_count := variant.rotate_pair_count
	cos_cache         := variant.cos_cache
	sin_cache         := variant.sin_cache
	token_count       := input.shape[0]
	head_size         := input.shape[input.rank - 1] / head_count
	half_head         := head_size / 2

	for pos in 0 ..< token_count {
		for i in 0 ..< rotate_pair_count {
			theta := f32(pos + pos_offset) / math.pow(base, f32(i * 2) / f32(head_size))
			cache_idx := pos * half_head + i
			data(cos_cache)[cache_idx] = math.cos(theta)
			data(sin_cache)[cache_idx] = math.sin(theta)
		}
	}

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< rotate_pair_count {
				cache_idx := t * half_head + i
				cos_val := data(cos_cache)[cache_idx]
				sin_val := data(sin_cache)[cache_idx]

				x := _load(xp, head_offset + i * 2)
				y := _load(xp, head_offset + i * 2 + 1)

				_store(yp, head_offset + i * 2,     x * cos_val - y * sin_val)
				_store(yp, head_offset + i * 2 + 1, x * sin_val + y * cos_val)
			}
			for i in rotate_pair_count ..< half_head {
				yp[head_offset + i * 2]     = xp[head_offset + i * 2]
				yp[head_offset + i * 2 + 1] = xp[head_offset + i * 2 + 1]
			}
		}
	}
}

rope_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant           := op.variant.(ml.Rope)
	head_count        := variant.head_count
	rotate_pair_count := variant.rotate_pair_count
	cos_cache         := variant.cos_cache
	sin_cache         := variant.sin_cache
	token_count       := input.shape[0]
	head_size         := input.shape[input.rank - 1] / head_count
	half_head         := head_size / 2

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< rotate_pair_count {
				cache_idx := t * half_head + i
				cos_val := data(cos_cache)[cache_idx]
				sin_val := data(sin_cache)[cache_idx]

				grad_x := gradient(output)[head_offset + i * 2]
				grad_y := gradient(output)[head_offset + i * 2 + 1]

				gradient(input)[head_offset + i * 2]     +=  grad_x * cos_val + grad_y * sin_val
				gradient(input)[head_offset + i * 2 + 1] += -grad_x * sin_val + grad_y * cos_val
			}
			for i in rotate_pair_count ..< half_head {
				gradient(input)[head_offset + i * 2]     += gradient(output)[head_offset + i * 2]
				gradient(input)[head_offset + i * 2 + 1] += gradient(output)[head_offset + i * 2 + 1]
			}
		}
	}
}

LAYERNORM_EPSILON :: 1e-5

layernorm_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _layernorm_forward_impl(f32,      op)
	case .Bf16: _layernorm_forward_impl(ml.Bf16, op)
	}
}

_layernorm_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := data(variant.mean)
	rstd    := data(variant.rstd)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	wp := _typed_data(T, weight)

	for c in 0 ..< count {
		offset := c * size

		m: f32
		for i in 0 ..< size {
			m += _load(xp, offset + i)
		}
		m /= f32(size)

		v: f32
		for i in 0 ..< size {
			x_shift := _load(xp, offset + i) - m
			v += x_shift * x_shift
		}
		v /= f32(size)

		s: f32 = 1.0 / math.sqrt(v + f32(LAYERNORM_EPSILON))
		for i in 0 ..< size {
			n := (s * (_load(xp, offset + i) - m))
			_store(yp, offset + i, n * _load(wp, i))
		}

		mean[c] = m
		rstd[c] = s
	}
}

layernorm_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _layernorm_backward_impl(f32,      op)
	case .Bf16: _layernorm_backward_impl(ml.Bf16, op)
	}
}

_layernorm_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := data(variant.mean)
	rstd    := data(variant.rstd)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	have_dx, have_dw := ml.has_gradient(input), ml.has_gradient(weight)
	if !have_dx && !have_dw { return }

	xp := _typed_data(T, input)
	wp := _typed_data(T, weight)
	dx := gradient(input)
	dw := gradient(weight)
	dy := gradient(output)

	for c in 0 ..< count {
		offset := c * size
		mean_c := mean[c]
		rstd_c := rstd[c]

		dnorm_mean:      f32
		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := (_load(xp, offset + i) - mean_c) * rstd_c
			dnorm := _load(wp, i) * dy[offset + i]
			dnorm_mean      += dnorm
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_mean      /= f32(size)
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			dy_v  := dy[offset + i]
			norm  := (_load(xp, offset + i) - mean_c) * rstd_c
			dnorm := _load(wp, i) * dy_v

			if have_dw {
				dw[i] += norm * dy_v
			}

			if have_dx {
				grad := dnorm - dnorm_mean - norm * dnorm_norm_mean
				grad *= rstd_c

				dx[offset + i] += grad
			}
		}
	}
}

rmsnorm_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rmsnorm_forward_impl(f32,      op)
	case .Bf16: _rmsnorm_forward_impl(ml.Bf16, op)
	}
}

_rmsnorm_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Rmsnorm)
	weight  := variant.weight
	rstd    := data(variant.rstd)
	eps     := variant.eps
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	wp := _typed_data(T, weight)

	for c in 0 ..< count {
		offset := c * size

		ms: f32
		for i in 0 ..< size {
			v := _load(xp, offset + i)
			ms += v * v
		}
		ms /= f32(size)

		s: f32 = 1.0 / math.sqrt(ms + eps)
		for i in 0 ..< size {
			_store(yp, offset + i, s * _load(xp, offset + i) * _load(wp, i))
		}

		rstd[c] = s
	}
}

rmsnorm_rope_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rmsnorm_rope_forward_impl(f32,      op)
	case .Bf16: _rmsnorm_rope_forward_impl(ml.Bf16, op)
	}
}

_rmsnorm_rope_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input             := op.input
	output            := op.output
	variant           := op.variant.(ml.Rmsnorm_Rope)
	weight            := variant.weight
	eps               := variant.eps
	head_count        := variant.head_count
	rope_base         := variant.base
	pos_offset        := variant.position_offset
	rotate_pair_count := variant.rotate_pair_count
	token_count       := input.shape[0]
	head_size         := input.shape[1] / head_count
	half_head         := head_size / 2

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	wp := _typed_data(T, weight)

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			ms: f32
			for i in 0 ..< head_size {
				v := _load(xp, head_offset + i)
				ms += v * v
			}
			s: f32 = 1.0 / math.sqrt(ms / f32(head_size) + eps)

			for i in 0 ..< rotate_pair_count {
				theta   := f32(t + pos_offset) / math.pow(rope_base, f32(i * 2) / f32(head_size))
				cos_val := math.cos(theta)
				sin_val := math.sin(theta)
				n0 := s * _load(xp, head_offset + i * 2)     * _load(wp, i * 2)
				n1 := s * _load(xp, head_offset + i * 2 + 1) * _load(wp, i * 2 + 1)
				_store(yp, head_offset + i * 2,     n0 * cos_val - n1 * sin_val)
				_store(yp, head_offset + i * 2 + 1, n0 * sin_val + n1 * cos_val)
			}
			for i in rotate_pair_count ..< half_head {
				_store(yp, head_offset + i * 2,     s * _load(xp, head_offset + i * 2)     * _load(wp, i * 2))
				_store(yp, head_offset + i * 2 + 1, s * _load(xp, head_offset + i * 2 + 1) * _load(wp, i * 2 + 1))
			}
		}
	}
}

add_rmsnorm_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _add_rmsnorm_forward_impl(f32,      op)
	case .Bf16: _add_rmsnorm_forward_impl(ml.Bf16, op)
	}
}

_add_rmsnorm_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a       := op.input
	normed  := op.output
	variant := op.variant.(ml.Add_Rmsnorm)
	b       := variant.b
	weight  := variant.weight
	resid   := variant.residual_out
	eps     := variant.eps
	size    := a.shape[a.rank - 1]
	count   := ml.len(a) / size

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	wp := _typed_data(T, weight)
	rp := _typed_data(T, resid)
	yp := _typed_data(T, normed)

	for c in 0 ..< count {
		offset := c * size

		ms: f32
		for i in 0 ..< size {
			_store(rp, offset + i, _load(ap, offset + i) + _load(bp, offset + i))
			vf := _load(rp, offset + i)
			ms += vf * vf
		}
		s: f32 = 1.0 / math.sqrt(ms / f32(size) + eps)
		for i in 0 ..< size {
			_store(yp, offset + i, s * _load(rp, offset + i) * _load(wp, i))
		}
	}
}

rmsnorm_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _rmsnorm_backward_impl(f32,      op)
	case .Bf16: _rmsnorm_backward_impl(ml.Bf16, op)
	}
}

_rmsnorm_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Rmsnorm)
	weight  := variant.weight
	rstd    := data(variant.rstd)
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	have_dx, have_dw := ml.has_gradient(input), ml.has_gradient(weight)
	if !have_dx && !have_dw { return }

	xp := _typed_data(T, input)
	wp := _typed_data(T, weight)
	dx := gradient(input)
	dw := gradient(weight)
	dy := gradient(output)

	for c in 0 ..< count {
		offset := c * size
		rstd_c := rstd[c]

		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := _load(xp, offset + i) * rstd_c
			dnorm := _load(wp, i) * dy[offset + i]
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			dy_v  := dy[offset + i]
			norm  := _load(xp, offset + i) * rstd_c
			dnorm := _load(wp, i) * dy_v

			if have_dw {
				dw[i] += norm * dy_v
			}

			if have_dx {
				grad := (dnorm - norm * dnorm_norm_mean) * rstd_c
				dx[offset + i] += grad
			}
		}
	}
}

softmax_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	#partial switch input.type {
	case .F32:
		for sample in 0 ..< count {
			max_value := math.NEG_INF_F32
			for i in 0 ..< size {
				index := sample * size + i
				max_value = math.max(max_value, data(input)[index])
			}
			sum: f32
			for i in 0 ..< size {
				index := sample * size + i
				exp_val := math.exp(data(input)[index] - max_value)
				data(output)[index] = exp_val
				sum += exp_val
			}
			for i in 0 ..< size {
				index := sample * size + i
				data(output)[index] /= sum
			}
		}
	case .Bf16:
		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for sample in 0 ..< count {
			runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

			base := sample * size
			max_value := math.NEG_INF_F32
			for i in 0 ..< size {
				v := ml.bf16_to_f32(x_bf[base + i])
				if v > max_value { max_value = v }
			}
			sum: f32
			scratch := make([]f32, size, context.temp_allocator)
			for i in 0 ..< size {
				e := math.exp(ml.bf16_to_f32(x_bf[base + i]) - max_value)
				scratch[i] = e
				sum += e
			}
			for i in 0 ..< size {
				y_bf[base + i] = ml.bf16_from_f32(scratch[i] / sum)
			}
		}
	}
}

softmax_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	size  := op.input.shape[op.input.rank - 1]
	count := ml.len(op.input) / size

	#partial switch op.input.type {
	case .F32:
		parallelize(count, count, op, proc(index: int, op: ml.Operation) {
			input, output := op.input, op.output
			size  := input.shape[input.rank - 1]
			base  := index * size

			out_data := data(output)    [base:base + size]
			out_grad := gradient(output)[base:base + size]
			in_grad  := gradient(input) [base:base + size]

			dot: f32
			for i in 0 ..< size {
				dot += out_grad[i] * out_data[i]
			}

			for i in 0 ..< size {
				in_grad[i] += out_data[i] * (out_grad[i] - dot)
			}
		})
	case .Bf16:
		y_bf := data_bf16(op.output)
		dy   := gradient(op.output)
		dx   := gradient(op.input)
		for sample in 0 ..< count {
			base := sample * size
			dot:  f32
			for i in 0 ..< size {
				dot += dy[base + i] * ml.bf16_to_f32(y_bf[base + i])
			}
			for i in 0 ..< size {
				y_v := ml.bf16_to_f32(y_bf[base + i])
				dx[base + i] += y_v * (dy[base + i] - dot)
			}
		}
	}
}

log_softmax_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _log_softmax_forward_impl(f32,      op)
	case .Bf16: _log_softmax_forward_impl(ml.Bf16, op)
	}
}

_log_softmax_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input  := op.input
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	xp := _typed_data(T, input)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		base := sample * size
		max_value := math.NEG_INF_F32
		for i in 0 ..< size {
			v := _load(xp, base + i)
			if v > max_value { max_value = v }
		}
		lse: f32
		for i in 0 ..< size {
			lse += math.exp(_load(xp, base + i) - max_value)
		}
		lse = math.ln(lse) + max_value
		for i in 0 ..< size {
			_store(yp, base + i, _load(xp, base + i) - lse)
		}
	}
}

log_softmax_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _log_softmax_backward_impl(f32,      op)
	case .Bf16: _log_softmax_backward_impl(ml.Bf16, op)
	}
}

_log_softmax_backward_impl :: proc($T: typeid, op: ml.Operation) {
	input, output := op.input, op.output
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	yp := _typed_data(T, output)
	dy := gradient(output)
	dx := gradient(input)
	for sample in 0 ..< count {
		base := sample * size
		grad_sum: f32
		for i in 0 ..< size {
			grad_sum += dy[base + i]
		}
		for i in 0 ..< size {
			dx[base + i] += dy[base + i] - math.exp(_load(yp, base + i)) * grad_sum
		}
	}
}

entropy_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _entropy_forward_impl(f32,      op)
	case .Bf16: _entropy_forward_impl(ml.Bf16, op)
	}
}

_entropy_forward_impl :: proc($T: typeid, op: ml.Operation) {
	probabilities := op.input
	size          := probabilities.shape[probabilities.rank - 1]
	count         := ml.len(probabilities) / size

	pp := _typed_data(T, probabilities)
	yp := _typed_data(T, op.output)
	for sample in 0 ..< count {
		entropy_value: f32
		base := sample * size
		for i in 0 ..< size {
			p      := _load(pp, base + i)
			p_safe := math.max(p, f32(1e-8))
			entropy_value -= p * math.ln(p_safe)
		}
		_store(yp, sample, entropy_value)
	}
}

entropy_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	#partial switch op.input.type {
	case .F32:  _entropy_backward_impl(f32,      op)
	case .Bf16: _entropy_backward_impl(ml.Bf16, op)
	}
}

_entropy_backward_impl :: proc($T: typeid, op: ml.Operation) {
	probabilities := op.input
	size  := probabilities.shape[probabilities.rank - 1]
	count := ml.len(probabilities) / size

	pp    := _typed_data(T, probabilities)
	dp    := gradient(probabilities)
	d_out := gradient(op.output)
	for sample in 0 ..< count {
		base   := sample * size
		dout_v := d_out[sample]
		for i in 0 ..< size {
			p      := _load(pp, base + i)
			p_safe := math.max(p, f32(1e-8))
			grad   := -(math.ln(p_safe) + 1.0)
			dp[base + i] += dout_v * grad
		}
	}
}

mean_squared_error_forward :: proc(op: ml.Operation) {
	predictions := op.input
	output      := op.output
	targets     := op.variant.(ml.Mean_Squared_Error).targets
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		sum_squared_error: f32

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			diff  := data(predictions)[index] - data(targets)[index]
			sum_squared_error += diff * diff
		}

		data(output)[sample] = sum_squared_error / f32(sample_size)
	}
}

mean_squared_error_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	predictions, output := op.input, op.output
	targets := op.variant.(ml.Mean_Squared_Error).targets
	count   := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		scale := 2.0 / f32(sample_size)

		upstream_gradient := gradient(output)[sample]

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			grad := scale * (data(predictions)[index] - data(targets)[index])
			gradient(predictions)[index] += grad * upstream_gradient
		}
	}
}

smooth_l1_forward :: proc(op: ml.Operation) {
	predictions := op.input
	output      := op.output
	variant     := op.variant.(ml.Smooth_L1)
	targets     := variant.targets
	beta        := variant.beta
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		sum: f32

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			diff  := data(predictions)[index] - data(targets)[index]
			if abs(diff) < beta {
				sum += 0.5 * diff * diff / beta
			} else {
				sum += abs(diff) - 0.5 * beta
			}
		}

		data(output)[sample] = sum / f32(sample_size)
	}
}

smooth_l1_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	predictions, output := op.input, op.output
	variant     := op.variant.(ml.Smooth_L1)
	targets     := variant.targets
	beta        := variant.beta
	count       := ml.len(output)
	sample_size := ml.len(predictions) / count

	for sample in 0 ..< count {
		scale := 1.0 / f32(sample_size)

		upstream_gradient := gradient(output)[sample]

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			diff  := data(predictions)[index] - data(targets)[index]
			grad  := math.clamp(diff / beta, -1, 1) * scale
			gradient(predictions)[index] += grad * upstream_gradient
		}
	}
}

cross_entropy_forward :: proc(op: ml.Operation) {
	input         := op.input
	output        := op.output
	variant       := op.variant.(ml.Cross_Entropy)
	probabilities := variant.probabilities
	targets       := variant.targets
	class_size    := input.shape[input.rank - 1]

	for sample in 0 ..< builtin.len(targets) {
		offset := sample * class_size
		target := targets[sample]

		max_value := math.NEG_INF_F32
		for i in 0 ..< class_size {
			index := offset + i
			max_value = math.max(max_value, data(input)[index])
		}

		sum: f32
		for i in 0 ..< class_size {
			index := offset + i
			exp_val := math.exp(data(input)[index] - max_value)
			data(probabilities)[index] = exp_val
			sum += exp_val
		}

		for i in 0 ..< class_size {
			index := offset + i
			data(probabilities)[index] /= sum
		}

		target_index := offset + target
		data(output)[sample] = -data(input)[target_index] + max_value + math.ln(sum)
	}
}

cross_entropy_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output

	variant       := op.variant.(ml.Cross_Entropy)
	probabilities := variant.probabilities
	targets       := variant.targets
	class_size    := input.shape[input.rank - 1]

	for sample in 0 ..< builtin.len(targets) {
		offset := sample * class_size
		target := targets[sample]

		upstream_gradient := gradient(output)[sample]

		for i in 0 ..< class_size {
			index := offset + i
			target_value: f32 = i == target ? 1 : 0

			grad := (data(probabilities)[index] - target_value) * upstream_gradient

			gradient(input)[index] += grad
		}
	}
}

_unary_forward_dispatch :: proc(op: ml.Operation, fwd_f32: proc(x: f32) -> f32) {
	input, output := op.input, op.output
	#partial switch input.type {
	case .F32:
		x := data(input)
		y := data(output)
		#no_bounds_check for i in 0 ..< ml.len(input) {
			y[i] = fwd_f32(x[i])
		}
	case .Bf16:
		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< ml.len(input) {
			y_bf[i] = ml.bf16_from_f32(fwd_f32(ml.bf16_to_f32(x_bf[i])))
		}
	}
}

_unary_backward_dispatch :: proc(op: ml.Operation, local_grad_from_input: proc(x: f32) -> f32) {
	if !ml.has_gradient(op.input) { return }
	input, output := op.input, op.output
	#partial switch input.type {
	case .F32:
		x  := data(input)
		dx := gradient(input)
		dy := gradient(output)
		#no_bounds_check for i in 0 ..< ml.len(input) {
			dx[i] += dy[i] * local_grad_from_input(x[i])
		}
	case .Bf16:
		x_bf := data_bf16(input)
		dy   := gradient(output)
		dx   := gradient(input)
		for i in 0 ..< ml.len(input) {
			x_v := ml.bf16_to_f32(x_bf[i])
			dx[i] += dy[i] * local_grad_from_input(x_v)
		}
	}
}

relu_forward :: proc(op: ml.Operation) {
	if op.input.type != .F32 {
		_unary_forward_dispatch(op, proc(x: f32) -> f32 { return x < 0 ? 0 : x })
		return
	}

	x := data(op.input)
	y := data(op.output)
	#no_bounds_check for i in 0 ..< ml.len(op.input) {
		y[i] = x[i] < 0 ? 0 : x[i]
	}
}

relu_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	if op.input.type != .F32 {
		_unary_backward_dispatch(op, proc(x: f32) -> f32 { return x > 0 ? 1 : 0 })
		return
	}

	x  := data(op.input)
	dx := gradient(op.input)
	dy := gradient(op.output)
	#no_bounds_check for i in 0 ..< ml.len(op.input) {
		dx[i] += x[i] > 0 ? dy[i] : 0
	}
}

sigmoid_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 { return 1.0 / (1.0 + math.exp(-x)) })
}

sigmoid_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		s := f32(1.0 / (1.0 + math.exp(-x)))
		return s * (1.0 - s)
	})
}

GELU_SCALING_FACTOR :: 0.7978845608028654 // math.sqrt(f32(2) / math.PI)

gelu_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 {
		cube := f32(0.044715) * x * x * x
		return 0.5 * x * (1.0 + math.tanh(f32(GELU_SCALING_FACTOR) * (x + cube)))
	})
}

gelu_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		cube     := f32(0.044715) * x * x * x
		tanh_arg := f32(GELU_SCALING_FACTOR) * (x + cube)
		tanh_out := math.tanh(tanh_arg)
		cosh_out := math.cosh(tanh_arg)
		sech_out := 1.0 / (cosh_out * cosh_out)
		return 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * f32(GELU_SCALING_FACTOR) * (1.0 + 3.0 * 0.044715 * x * x)
	})
}

gelu_mul_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _gelu_mul_forward_impl(f32,      op)
	case .Bf16: _gelu_mul_forward_impl(ml.Bf16, op)
	}
}

_gelu_mul_forward_impl :: proc($T: typeid, op: ml.Operation) {
	a, output := op.input, op.output
	b := op.variant.(ml.Gelu_Mul).b
	stride, width := _broadcast_tiling(a, b)

	gelu :: proc(x: f32) -> f32 {
		cube := f32(0.044715) * x * x * x
		return 0.5 * x * (1.0 + math.tanh(f32(GELU_SCALING_FACTOR) * (x + cube)))
	}

	ap := _typed_data(T, a)
	bp := _typed_data(T, b)
	op_ := _typed_data(T, output)
	for i in 0 ..< stride {
		for j in 0 ..< width {
			o := i * width + j
			_store(op_, o, gelu(_load(ap, o)) * _load(bp, j))
		}
	}
}

silu_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 {
		s := f32(1.0 / (1.0 + math.exp(-x)))
		return x * s
	})
}

silu_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		s := f32(1.0 / (1.0 + math.exp(-x)))
		return s + x * s * (1.0 - s)
	})
}

tanh_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 { return math.tanh(x) })
}

tanh_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 {
		t := math.tanh(x)
		return 1.0 - t * t
	})
}

batched_matmul_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  batched_matmul_forward_f32 (op)
	case .Bf16: batched_matmul_forward_bf16(op)
	}
}

batched_matmul_forward_bf16 :: proc(op: ml.Operation) {
	a           := op.input
	batch_count := a.shape[0]
	m           := a.shape[1]

	parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		bt      := op.variant.(ml.Batched_Matmul).b
		m       := a.shape[1]
		k_count := a.shape[2]
		n       := bt.shape[2]

		bi := idx / m
		i  := idx % m

		a_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)bt.buffers    [.Data]))
		c_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))

		a_row := a_bf[bi * m * k_count + i * k_count:]
		c_row := c_bf[bi * m * n + i * n:]

		for j in 0 ..< n {
			acc: f32
			for kk in 0 ..< k_count {
				acc += ml.bf16_to_f32(a_row[kk]) *
				       ml.bf16_to_f32(b_bf[bi * k_count * n + kk * n + j])
			}
			c_row[j] = ml.bf16_from_f32(acc)
		}
	})
}

batched_matmul_forward_f32 :: proc(op: ml.Operation) {
	a := op.input
	batch_count := a.shape[0]
	m := a.shape[1]

	parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		bt      := op.variant.(ml.Batched_Matmul).b

		m        := a.shape[1]
		kk_count := a.shape[2]
		n        := bt.shape[2]

		bi := idx / m
		i  := idx % m

		a_ptr := ([^]f32)(raw_data(data(a)))
		b_ptr := ([^]f32)(raw_data(data(bt)))
		c_ptr := ([^]f32)(raw_data(data(output)))

		a_row := a_ptr[bi * m * kk_count + i * kk_count:]
		c_row := c_ptr[bi * m * n + i * n:]

		for j in 0 ..< n {
			c_row[j] = 0
		}
		for kk in 0 ..< kk_count {
			b_row := b_ptr[bi * kk_count * n + kk * n:]
			_simd_axpy_f32(c_row, b_row, a_row[kk], n)
		}
	})
}

batched_matmul_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  batched_matmul_backward_f32 (op)
	case .Bf16: batched_matmul_backward_bf16(op)
	}
}

batched_matmul_backward_bf16 :: proc(op: ml.Operation) {
	a           := op.input
	bt          := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m           := a.shape[1]
	k           := a.shape[2]

	if ml.has_gradient(a) {
		parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b
			m       := a.shape[1]
			k_count := a.shape[2]
			n       := bt.shape[2]

			bi := idx / m
			i  := idx % m

			b_bf := data_bf16(bt)
			dc   := gradient(output)
			da   := gradient(a)

			dc_row := dc[bi * m * n + i * n:]
			da_row := da[bi * m * k_count + i * k_count:]

			for kk in 0 ..< k_count {
				acc: f32
				for j in 0 ..< n {
					acc += dc_row[j] * ml.bf16_to_f32(b_bf[bi * k_count * n + kk * n + j])
				}
				da_row[kk] += acc
			}
		})
	}

	if ml.has_gradient(bt) {
		parallelize(batch_count * k, batch_count * k, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b
			m       := a.shape[1]
			k_count := a.shape[2]
			n       := bt.shape[2]

			bi := idx / k_count
			kk := idx % k_count

			a_bf := data_bf16(a)
			dc   := gradient(output)
			db   := gradient(bt)

			db_row := db[bi * k_count * n + kk * n:]
			for j in 0 ..< n {
				acc: f32
				for ii in 0 ..< m {
					acc += ml.bf16_to_f32(a_bf[bi * m * k_count + ii * k_count + kk]) *
					       dc[bi * m * n + ii * n + j]
				}
				db_row[j] += acc
			}
		})
	}
}

batched_matmul_backward_f32 :: proc(op: ml.Operation) {
	a := op.input
	bt := op.variant.(ml.Batched_Matmul).b
	batch_count := a.shape[0]
	m := a.shape[1]
	k := a.shape[2]

	if ml.has_gradient(a) {
		parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b

			m        := a.shape[1]
			kk_count := a.shape[2]
			n        := bt.shape[2]

			bi := idx / m
			i  := idx % m

			a_grad_ptr := ([^]f32)(raw_data(gradient(a)))
			b_data_ptr := ([^]f32)(raw_data(data(bt)))
			c_grad_ptr := ([^]f32)(raw_data(gradient(output)))

			dc_row := c_grad_ptr[bi * m * n + i * n:]
			da_row := a_grad_ptr[bi * m * kk_count + i * kk_count:]

			for kk in 0 ..< kk_count {
				b_row := b_data_ptr[bi * kk_count * n + kk * n:]
				da_row[kk] += _simd_dot_f32(dc_row, b_row, n)
			}
		})
	}

	if ml.has_gradient(bt) {
		parallelize(batch_count * k, batch_count * k, op, proc(idx: int, op: ml.Operation) {
			a       := op.input
			output  := op.output
			bt      := op.variant.(ml.Batched_Matmul).b

			m        := a.shape[1]
			kk_count := a.shape[2]
			n        := bt.shape[2]

			bi := idx / kk_count
			kk := idx % kk_count

			a_data_ptr := ([^]f32)(raw_data(data(a)))
			b_grad_ptr := ([^]f32)(raw_data(gradient(bt)))
			c_grad_ptr := ([^]f32)(raw_data(gradient(output)))

			db_row := b_grad_ptr[bi * kk_count * n + kk * n:]

			for ii in 0 ..< m {
				a_ik   := a_data_ptr[bi * m * kk_count + ii * kk_count + kk]
				dc_row := c_grad_ptr[bi * m * n + ii * n:]
				_simd_axpy_f32(db_row, dc_row, a_ik, n)
			}
		})
	}
}

permute_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _permute_forward_impl(f32,      op)
	case .Bf16: _permute_forward_impl(ml.Bf16, op)
	}
}

_permute_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input  := op.input
	output := op.output
	axes   := op.variant.(ml.Permute).axes

	in_shape   := [3]int{input.shape [0],           input.shape [1], input.shape [2]}
	out_shape  := [3]int{output.shape[0],           output.shape[1], output.shape[2]}
	in_strides := [3]int{in_shape[1] * in_shape[2], in_shape[2],     1              }

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	for i0 in 0 ..< out_shape[0] {
		for i1 in 0 ..< out_shape[1] {
			for i2 in 0 ..< out_shape[2] {
				src: [3]int
				src[axes[0]] = i0
				src[axes[1]] = i1
				src[axes[2]] = i2

				src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
				dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

				yp[dst_idx] = xp[src_idx]
			}
		}
	}
}

permute_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input   := op.input
	output  := op.output
	axes    := op.variant.(ml.Permute).axes

	in_shape   := [3]int{input.shape [0],           input.shape [1], input.shape [2]}
	out_shape  := [3]int{output.shape[0],           output.shape[1], output.shape[2]}
	in_strides := [3]int{in_shape[1] * in_shape[2], in_shape[2],     1              }

	dx, dy := gradient(input), gradient(output)
	for i0 in 0 ..< out_shape[0] {
		for i1 in 0 ..< out_shape[1] {
			for i2 in 0 ..< out_shape[2] {
				src: [3]int
				src[axes[0]] = i0
				src[axes[1]] = i1
				src[axes[2]] = i2

				src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
				dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

				dx[src_idx] += dy[dst_idx]
			}
		}
	}
}

causal_mask_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  _causal_mask_forward_impl(f32,      op)
	case .Bf16: _causal_mask_forward_impl(ml.Bf16, op)
	}
}

_causal_mask_forward_impl :: proc($T: typeid, op: ml.Operation) {
	input  := op.input
	output := op.output

	token_count := input.shape[input.rank - 1]
	block_size  := token_count * token_count
	n_blocks    := ml.len(input) / block_size

	xp := _typed_data(T, input)
	yp := _typed_data(T, output)
	neg_inf: T
	when T == ml.Bf16 {
		neg_inf = ml.bf16_from_f32(math.NEG_INF_F32)
	} else {
		neg_inf = math.NEG_INF_F32
	}

	for blk in 0 ..< n_blocks {
		offset := blk * block_size
		for t1 in 0 ..< token_count {
			for t2 in 0 ..< token_count {
				idx := offset + t1 * token_count + t2
				if t2 <= t1 {
					yp[idx] = xp[idx]
				} else {
					yp[idx] = neg_inf
				}
			}
		}
	}
}

causal_mask_backward :: proc(op: ml.Operation) {
	if !ml.has_gradient(op.input) { return }
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := ml.len(input) / block_size

	dx, dy := gradient(input), gradient(output)
	for blk in 0 ..< n_blocks {
		offset := blk * block_size
		for t1 in 0 ..< T {
			for t2 in 0 ..= t1 {
				idx := offset + t1 * T + t2
				dx[idx] += dy[idx]
			}
		}
	}
}

attention_forward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  attention_forward_f32 (op)
	case .Bf16: attention_forward_bf16(op)
	}
}

attention_forward_f32 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	parallelize(v.n_q_heads, v.n_q_heads, op, proc(h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]f32)(raw_data(data(op.input)))
		k_ptr   := ([^]f32)(raw_data(data(v.key)))
		v_ptr   := ([^]f32)(raw_data(data(v.value)))
		out_ptr := ([^]f32)(raw_data(data(op.output)))
		sm_ptr  := ([^]f32)(raw_data(data(v.softmax_outputs)))

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size
			q := q_ptr[q_offset:]

			sm_row_offset := h * token_count * token_count + t_q * token_count
			sm_row := sm_ptr[sm_row_offset:]

			t_k_max := token_count
			if causal { t_k_max = t_q + 1 }
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := t_k * kv_size + kv_h * head_size
				score := _simd_dot_f32(q, k_ptr[k_offset:], head_size) * inv_sqrt_d
				sm_row[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(sm_row[t_k] - max_score)
				sm_row[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp
			for t_k in t_k_min ..< t_k_max {
				sm_row[t_k] *= inv_sum
			}
			for t_k in 0 ..< t_k_min {
				sm_row[t_k] = 0
			}
			for t_k in t_k_max ..< token_count {
				sm_row[t_k] = 0
			}

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				out_ptr[out_offset + d] = 0
			}
			for t_k in t_k_min ..< t_k_max {
				v_offset := t_k * kv_size + kv_h * head_size
				_simd_axpy_f32(out_ptr[out_offset:], v_ptr[v_offset:], sm_row[t_k], head_size)
			}
		}
	})
}

attention_forward_bf16 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	parallelize(v.n_q_heads, v.n_q_heads, op, proc(h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers[.Data]))
		k_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.key.buffers   [.Data]))
		v_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.value.buffers [.Data]))
		out_ptr := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Data]))
		sm_ptr  := ([^]f32)(raw_data(data(v.softmax_outputs)))

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size

			sm_row_offset := h * token_count * token_count + t_q * token_count
			sm_row := sm_ptr[sm_row_offset:]

			t_k_max := token_count
			if causal { t_k_max = t_q + 1 }
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := t_k * kv_size + kv_h * head_size
				score: f32
				for d in 0 ..< head_size {
					score += ml.bf16_to_f32(q_ptr[q_offset + d]) * ml.bf16_to_f32(k_ptr[k_offset + d])
				}
				score *= inv_sqrt_d
				sm_row[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(sm_row[t_k] - max_score)
				sm_row[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp
			for t_k in t_k_min ..< t_k_max {
				sm_row[t_k] *= inv_sum
			}
			for t_k in 0 ..< t_k_min {
				sm_row[t_k] = 0
			}
			for t_k in t_k_max ..< token_count {
				sm_row[t_k] = 0
			}

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				acc: f32
				for t_k in t_k_min ..< t_k_max {
					v_offset := t_k * kv_size + kv_h * head_size
					acc += sm_row[t_k] * ml.bf16_to_f32(v_ptr[v_offset + d])
				}
				out_ptr[out_offset + d] = ml.bf16_from_f32(acc)
			}
		}
	})
}

attention_backward :: proc(op: ml.Operation) {
	#partial switch op.input.type {
	case .F32:  attention_backward_f32 (op)
	case .Bf16: attention_backward_bf16(op)
	}
}

attention_backward_f32 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	parallelize(v.n_kv_heads, v.n_kv_heads, op, proc(kv_h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_data    := ([^]f32)(raw_data(data(op.input)))
		q_grad    := ([^]f32)(raw_data(gradient(op.input)))
		k_data    := ([^]f32)(raw_data(data(v.key)))
		k_grad    := ([^]f32)(raw_data(gradient(v.key)))
		v_data    := ([^]f32)(raw_data(data(v.value)))
		v_grad    := ([^]f32)(raw_data(gradient(v.value)))
		out_grad  := ([^]f32)(raw_data(gradient(op.output)))
		sm_ptr    := ([^]f32)(raw_data(data(v.softmax_outputs)))
		dp_ptr    := ([^]f32)(raw_data(data(v.d_p_scratch)))

		have_dq := ml.has_gradient(op.input)
		have_dk := ml.has_gradient(v.key)
		have_dv := ml.has_gradient(v.value)

		for q_h_off in 0 ..< group_size {
			h := kv_h * group_size + q_h_off

			dp_base := h * token_count
			d_p     := dp_ptr[dp_base:]

			for t_q in 0 ..< token_count {
				t_k_max := token_count
				if causal { t_k_max = t_q + 1 }
				t_k_min := 0
				if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

				d_out_offset := t_q * q_size + h * head_size
				d_out := out_grad[d_out_offset:]

				sm_row_offset := h * token_count * token_count + t_q * token_count
				sm_row := sm_ptr[sm_row_offset:]

				for t_k in t_k_min ..< t_k_max {
					v_offset := t_k * kv_size + kv_h * head_size
					d_p[t_k] = _simd_dot_f32(d_out, v_data[v_offset:], head_size)
					if have_dv { _simd_axpy_f32(v_grad[v_offset:], d_out, sm_row[t_k], head_size) }
				}

				dot_dp_p: f32
				for t_k in t_k_min ..< t_k_max {
					dot_dp_p += d_p[t_k] * sm_row[t_k]
				}
				for t_k in t_k_min ..< t_k_max {
					d_p[t_k] = sm_row[t_k] * (d_p[t_k] - dot_dp_p) * inv_sqrt_d
				}

				q_offset := t_q * q_size + h * head_size
				d_q_vec  := q_grad[q_offset:]
				q_vec    := q_data[q_offset:]

				for t_k in t_k_min ..< t_k_max {
					k_offset := t_k * kv_size + kv_h * head_size
					if have_dq { _simd_axpy_f32(d_q_vec, k_data[k_offset:], d_p[t_k], head_size) }
					if have_dk { _simd_axpy_f32(k_grad[k_offset:], q_vec,   d_p[t_k], head_size) }
				}
			}
		}
	})
}

attention_backward_bf16 :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention)

	parallelize(v.n_kv_heads, v.n_kv_heads, op, proc(kv_h: int, op: ml.Operation) {
		v := op.variant.(ml.Attention)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		causal      := v.causal
		window      := v.window
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_data   := data_bf16(op.input)
		q_grad   := gradient(op.input)
		k_data   := data_bf16(v.key)
		k_grad   := gradient(v.key)
		v_data   := data_bf16(v.value)
		v_grad   := gradient(v.value)
		out_grad := gradient(op.output)
		sm_ptr   := ([^]f32)(raw_data(data(v.softmax_outputs)))
		dp_ptr   := ([^]f32)(raw_data(data(v.d_p_scratch)))

		have_dq := ml.has_gradient(op.input)
		have_dk := ml.has_gradient(v.key)
		have_dv := ml.has_gradient(v.value)

		for q_h_off in 0 ..< group_size {
			h := kv_h * group_size + q_h_off

			dp_base := h * token_count
			d_p     := dp_ptr[dp_base:]

			for t_q in 0 ..< token_count {
				t_k_max := token_count
				if causal { t_k_max = t_q + 1 }
				t_k_min := 0
				if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

				d_out_offset := t_q * q_size + h * head_size
				sm_row_offset := h * token_count * token_count + t_q * token_count
				sm_row := sm_ptr[sm_row_offset:]

				for t_k in t_k_min ..< t_k_max {
					v_offset := t_k * kv_size + kv_h * head_size
					dot: f32
					for d in 0 ..< head_size {
						dot += out_grad[d_out_offset + d] * ml.bf16_to_f32(v_data[v_offset + d])
					}
					d_p[t_k] = dot

					if have_dv {
						p_val := sm_row[t_k]
						for d in 0 ..< head_size {
							v_grad[v_offset + d] += out_grad[d_out_offset + d] * p_val
						}
					}
				}

				dot_dp_p: f32
				for t_k in t_k_min ..< t_k_max {
					dot_dp_p += d_p[t_k] * sm_row[t_k]
				}
				for t_k in t_k_min ..< t_k_max {
					d_p[t_k] = sm_row[t_k] * (d_p[t_k] - dot_dp_p) * inv_sqrt_d
				}

				q_offset := t_q * q_size + h * head_size

				for t_k in t_k_min ..< t_k_max {
					k_offset := t_k * kv_size + kv_h * head_size
					scale    := d_p[t_k]
					for d in 0 ..< head_size {
						q_d := ml.bf16_to_f32(q_data[q_offset + d])
						k_d := ml.bf16_to_f32(k_data[k_offset + d])
						if have_dq { q_grad[q_offset + d] += scale * k_d }
						if have_dk { k_grad[k_offset + d] += scale * q_d }
					}
				}
			}
		}
	})
}

attention_cache_forward :: proc(op: ml.Operation) {
	v := op.variant.(ml.Attention_Cache)

	token_count := op.input.shape[0]
	kv_size     := v.key.shape[1]
	row_bytes   := kv_size * ml.data_type_size(v.key.type)
	cache_pos   := v.cache_position
	t_capacity  := v.k_cache.shape[0]

	k_new_bytes   := transmute([]byte)v.key.buffers    [.Data]
	v_new_bytes   := transmute([]byte)v.value.buffers  [.Data]
	k_cache_bytes := transmute([]byte)v.k_cache.buffers[.Data]
	v_cache_bytes := transmute([]byte)v.v_cache.buffers[.Data]

	first_phys := cache_pos % t_capacity
	first_count := token_count
	if first_phys + first_count > t_capacity { first_count = t_capacity - first_phys }
	first_bytes := first_count * row_bytes
	dst0 := first_phys * row_bytes
	builtin.copy(k_cache_bytes[dst0:dst0 + first_bytes], k_new_bytes[:first_bytes])
	builtin.copy(v_cache_bytes[dst0:dst0 + first_bytes], v_new_bytes[:first_bytes])

	if first_count < token_count {
		wrap_bytes := (token_count - first_count) * row_bytes
		builtin.copy(k_cache_bytes[:wrap_bytes], k_new_bytes[first_bytes:first_bytes + wrap_bytes])
		builtin.copy(v_cache_bytes[:wrap_bytes], v_new_bytes[first_bytes:first_bytes + wrap_bytes])
	}

	#partial switch op.input.type {
	case .F32:  attention_cache_forward_f32 (op)
	case .Bf16: attention_cache_forward_bf16(op)
	}
}

attention_cache_forward_f32 :: proc(op: ml.Operation) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	v := op.variant.(ml.Attention_Cache)

	token_count := op.input.shape[0]
	k_total     := v.cache_position + token_count

	Job_Data :: struct {
		op:      ml.Operation,
		scratch: []f32,
		k_total: int,
	}
	jd := Job_Data{
		op      = op,
		scratch = make([]f32, v.n_q_heads * k_total, context.temp_allocator),
		k_total = k_total,
	}

	parallelize(v.n_q_heads, v.n_q_heads, jd, proc(h: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Attention_Cache)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		cache_pos   := v.cache_position
		window      := v.window
		t_capacity  := v.k_cache.shape[0]
		k_total     := jd.k_total
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]f32)(raw_data(data(op.input)))
		k_ptr   := ([^]f32)(raw_data(data(v.k_cache)))
		v_ptr   := ([^]f32)(raw_data(data(v.v_cache)))
		out_ptr := ([^]f32)(raw_data(data(op.output)))

		scores := jd.scratch[h * k_total : (h + 1) * k_total]

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size
			q := q_ptr[q_offset:]

			t_k_max := cache_pos + t_q + 1
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
				score := _simd_dot_f32(q, k_ptr[k_offset:], head_size) * inv_sqrt_d
				scores[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(scores[t_k] - max_score)
				scores[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				out_ptr[out_offset + d] = 0
			}
			for t_k in t_k_min ..< t_k_max {
				v_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
				_simd_axpy_f32(out_ptr[out_offset:], v_ptr[v_offset:], scores[t_k] * inv_sum, head_size)
			}
		}
	})
}

attention_cache_forward_bf16 :: proc(op: ml.Operation) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	v := op.variant.(ml.Attention_Cache)

	token_count := op.input.shape[0]
	k_total     := v.cache_position + token_count

	Job_Data :: struct {
		op:      ml.Operation,
		scratch: []f32,
		k_total: int,
	}
	jd := Job_Data{
		op      = op,
		scratch = make([]f32, v.n_q_heads * k_total, context.temp_allocator),
		k_total = k_total,
	}

	parallelize(v.n_q_heads, v.n_q_heads, jd, proc(h: int, jd: Job_Data) {
		op := jd.op
		v := op.variant.(ml.Attention_Cache)

		token_count := op.input.shape[0]
		q_size      := op.input.shape[1]
		kv_size     := v.key.shape[1]
		head_size   := q_size / v.n_q_heads
		group_size  := v.n_q_heads / v.n_kv_heads
		kv_h        := h / group_size
		cache_pos   := v.cache_position
		window      := v.window
		t_capacity  := v.k_cache.shape[0]
		k_total     := jd.k_total
		inv_sqrt_d  := 1.0 / math.sqrt(f32(head_size))

		q_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers  [.Data]))
		k_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.k_cache.buffers [.Data]))
		v_ptr   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.v_cache.buffers [.Data]))
		out_ptr := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers [.Data]))

		scores := jd.scratch[h * k_total : (h + 1) * k_total]

		for t_q in 0 ..< token_count {
			q_offset := t_q * q_size + h * head_size

			t_k_max := cache_pos + t_q + 1
			t_k_min := 0
			if window > 0 && t_k_max > window { t_k_min = t_k_max - window }

			max_score := math.NEG_INF_F32
			for t_k in t_k_min ..< t_k_max {
				k_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
				score := _simd_dot_bf16_f32(q_ptr[q_offset:], k_ptr[k_offset:], head_size) * inv_sqrt_d
				scores[t_k] = score
				if score > max_score { max_score = score }
			}

			sum_exp: f32
			for t_k in t_k_min ..< t_k_max {
				e := math.exp(scores[t_k] - max_score)
				scores[t_k] = e
				sum_exp += e
			}
			inv_sum := 1.0 / sum_exp

			out_offset := t_q * q_size + h * head_size
			for d in 0 ..< head_size {
				acc: f32
				for t_k in t_k_min ..< t_k_max {
					v_offset := (t_k % t_capacity) * kv_size + kv_h * head_size
					acc += scores[t_k] * inv_sum * ml.bf16_to_f32(v_ptr[v_offset + d])
				}
				out_ptr[out_offset + d] = ml.bf16_from_f32(acc)
			}
		}
	})
}

attention_cache_backward :: proc(op: ml.Operation, loc := #caller_location) {
	panic("Attention_Cache is forward-only", loc)
}
