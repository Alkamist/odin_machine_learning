package cpu

import "base:builtin"
import "base:runtime"
import "base:intrinsics"

import "core:fmt"
import "core:mem"
import "core:math"
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

	@(thread_local) _in_parallelize: bool

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
					_in_parallelize = true
					d.chunk_proc(start, end, d.data)
					_in_parallelize = false
				}
			}

			sync.wait_group_done(&_done_wg)
		}
	}

	_startup_thread_pool :: proc(thread_count: int) {
		context.allocator = runtime.default_allocator()
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
		context.allocator = runtime.default_allocator()
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

	set_thread_count :: proc(count: int, loc := #caller_location) {
		assert(count > 0, "thread count must be at least 1", loc=loc)

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
	_thread_pool_fini :: proc "contextless" () {
		if _thread_count <= 1 {
			return
		}

		context = _thread_pool_context
		_cleanup_thread_pool()
	}

	PARALLELIZE_MIN_WORK :: 24 * 1024

	_parallelize :: proc(job_count, task_count: int, data: $Data, job: proc(index: int, data: Data), work := max(int)) {
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

		assert(!_in_parallelize, "parallelize called from inside a parallelize job — this deadlocks the worker pool")

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
		_in_parallelize = true
		thunk(0, end, &td)
		_in_parallelize = false

		sync.wait_group_wait(&_done_wg)
	}
} else {
	set_thread_count :: proc(count: int, loc := #caller_location) {
	}

	_parallelize :: proc(job_count, task_count: int, data: $Data, job: proc(index: int, data: Data), work := max(int)) {
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
	clear        = _clear,
	forward      = _forward,
	backward     = _backward,
	update       = _update,
	buffer_alloc = _buffer_alloc,
	buffer_free  = _buffer_free,
	buffer_get   = _buffer_get,
	buffer_set   = _buffer_set,
	buffer_copy  = _buffer_copy,

	buffer_sq_sum_accumulate = _buffer_sq_sum_accumulate,
	buffer_scale             = _buffer_scale,

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
	assert(ctx_err == nil, "failed to allocate Context", loc=loc)

	arena_buf, arena_buf_err := builtin.make([]byte, size, allocator=context.allocator, loc=loc)
	assert(arena_buf_err == nil, "failed to allocate CPU backend arena data", loc=loc)
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

_clear :: proc(loc: runtime.Source_Code_Location) {
	ctx := cast(^Context)ml.current_context(loc=loc)
	mem.arena_free_all(&ctx.arena)
}

@(require_results)
_data :: #force_inline proc(t: ml.Tensor) -> []f32 {
	bytes := transmute([]byte)t.buffers[.Data]
	return ([^]f32)(raw_data(bytes))[:t.count]
}

@(require_results)
_gradient :: #force_inline proc(t: ml.Tensor) -> []f32 {
	bytes := transmute([]byte)t.buffers[.Gradient]
	return ([^]f32)(raw_data(bytes))[:t.count]
}

@(require_results)
_data_bf16 :: #force_inline proc(t: ml.Tensor) -> []ml.Bf16 {
	bytes := transmute([]byte)t.buffers[.Data]
	return ([^]ml.Bf16)(raw_data(bytes))[:t.count]
}

@(require_results)
_moment :: #force_inline proc(buffer: ml.Backend_Buffer, count: int) -> []f32 {
	bytes := transmute([]byte)buffer
	return ([^]f32)(raw_data(bytes))[:count]
}

_buffer_alloc :: proc(byte_count: int, kind: ml.Buffer_Kind, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	ctx       := cast(^Context)ml.current_context(loc=loc)
	allocator := persist ? context.allocator : mem.arena_allocator(&ctx.arena)

	bytes, err := builtin.make([]byte, byte_count, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "failed to allocate CPU buffer: %v", err, loc=loc)

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

_buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
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

_buffer_get :: proc(buffer: ml.Backend_Buffer, dst: []byte, loc: runtime.Source_Code_Location) {
	builtin.copy(dst, transmute([]byte)buffer)
}

_buffer_set :: proc(buffer: ml.Backend_Buffer, src: []byte, loc: runtime.Source_Code_Location) {
	builtin.copy(transmute([]byte)buffer, src)
}

_buffer_copy :: proc(dst, src: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	builtin.copy(transmute([]byte)dst, transmute([]byte)src)
}

_buffer_sq_sum_accumulate :: proc(buffer: ml.Backend_Buffer, count: int, accumulator: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	g   := ([^]f32)(raw_data(transmute([]byte)buffer))[:count]
	acc := (^f64)(raw_data(transmute([]byte)accumulator))

	total := f64(0)
	for i in 0 ..< count {
		total += f64(g[i]) * f64(g[i])
	}
	acc^ += total
}

_buffer_scale :: proc(buffer: ml.Backend_Buffer, count: int, scale: f32, loc: runtime.Source_Code_Location) {
	g := ([^]f32)(raw_data(transmute([]byte)buffer))[:count]
	for i in 0 ..< count {
		g[i] *= scale
	}
}

_update :: proc(opt: ml.Optimizer, t: ml.Tensor, m_buf, v_buf: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	g := _gradient(t)
	m := _moment(m_buf, t.count)
	v := _moment(v_buf, t.count)

	assert(g != nil, "tensor gradient is nil", loc=loc)
	assert(m != nil, "optimizer moment m is nil", loc=loc)
	assert(v != nil, "optimizer moment v is nil", loc=loc)

	d_bf: [^]ml.Bf16
	d_f32: []f32
	#partial switch t.type {
	case .F32:  d_f32 = _data(t)
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

_forward :: proc(op: ^ml.Operation, loc: runtime.Source_Code_Location) {
	_alloc_scratch(op, loc)
	op := op^
	switch _ in op.variant {
	case ml.Add:                       _add_forward(op)
	case ml.Sub:                       _sub_forward(op)
	case ml.Mul:                       _mul_forward(op)
	case ml.Div:                       _div_forward(op)
	case ml.Exp:                       _exp_forward(op)
	case ml.Sqrt:                      _sqrt_forward(op, loc)
	case ml.Clamp:                     _clamp_forward(op)
	case ml.Min:                       _min_forward(op)
	case ml.Max:                       _max_forward(op)
	case ml.Mean:                      _mean_forward(op)
	case ml.Sum:                       _sum_forward(op)
	case ml.Max_Reduce:                _max_reduce_forward(op)
	case ml.Im2col:                    _im2col_forward(op)
	case ml.Max_Pool2d:                _max_pool2d_forward(op)
	case ml.Avg_Pool2d:                _avg_pool2d_forward(op)
	case ml.Transpose:                 _transpose_forward(op)
	case ml.Select:                    _select_forward(op)
	case ml.Slice:                     _slice_forward(op)
	case ml.Slice_Trailing:            _slice_trailing_forward(op)
	case ml.Slice_Leading:             _slice_leading_forward(op)
	case ml.Concat:                    _concat_forward(op)
	case ml.Linear:                    _linear_forward(op)
	case ml.Linear_Q4_K:               _linear_q4_k_forward(op)
	case ml.Linear_Q4_K_Gate_Up_Geglu: panic("Linear_Q4_K_Gate_Up_Geglu is unreachable (the op decomposes when the capability is absent)", loc)
	case ml.Linear_Q6_K:               _linear_q6_k_forward(op)
	case ml.Rope:                      _rope_forward(op)
	case ml.Layernorm:                 _layernorm_forward(op)
	case ml.Rmsnorm:                   _rmsnorm_forward(op)
	case ml.Rmsnorm_Rope:              _rmsnorm_rope_forward(op)
	case ml.Rmsnorm_Rope_Write_Cache:  panic("backend does not advertise the Rmsnorm_Rope_Write_Cache capability", loc)
	case ml.Add_Rmsnorm:               _add_rmsnorm_forward(op)
	case ml.Softmax:                   _softmax_forward(op)
	case ml.Entropy:                   _entropy_forward(op)
	case ml.Log_Softmax:               _log_softmax_forward(op)
	case ml.Mean_Squared_Error:        _mean_squared_error_forward(op)
	case ml.Smooth_L1:                 _smooth_l1_forward(op)
	case ml.Cross_Entropy:             _cross_entropy_forward(op)
	case ml.Relu:                      _relu_forward(op)
	case ml.Sigmoid:                   _sigmoid_forward(op)
	case ml.Gelu:                      _gelu_forward(op)
	case ml.Gelu_Mul:                  _gelu_mul_forward(op)
	case ml.Silu:                      _silu_forward(op)
	case ml.Tanh:                      _tanh_forward(op)
	case ml.Batched_Matmul:            _batched_matmul_forward(op)
	case ml.Permute:                   _permute_forward(op)
	case ml.Causal_Mask:               _causal_mask_forward(op)
	case ml.Attention:                 _attention_forward(op)
	case ml.Attention_Cache:           _attention_cache_forward(op)
	case ml.Cast:                      _cast_forward(op)
	case ml.Lerp_Assign:               _lerp_assign_forward(op)
	case ml.Accumulate_Mean:           _accumulate_mean_forward(op)
	}
}

_backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) {
	switch _ in op.variant {
	case ml.Add:                       _add_backward(op)
	case ml.Sub:                       _sub_backward(op)
	case ml.Mul:                       _mul_backward(op)
	case ml.Div:                       _div_backward(op)
	case ml.Exp:                       _exp_backward(op)
	case ml.Sqrt:                      _sqrt_backward(op)
	case ml.Clamp:                     _clamp_backward(op)
	case ml.Min:                       _min_backward(op)
	case ml.Max:                       _max_backward(op)
	case ml.Mean:                      _mean_backward(op)
	case ml.Sum:                       _sum_backward(op)
	case ml.Max_Reduce:                _max_reduce_backward(op)
	case ml.Im2col:                    _im2col_backward(op)
	case ml.Max_Pool2d:                _max_pool2d_backward(op)
	case ml.Avg_Pool2d:                _avg_pool2d_backward(op)
	case ml.Transpose:                 _transpose_backward(op)
	case ml.Select:                    _select_backward(op)
	case ml.Slice:                     _slice_backward(op)
	case ml.Slice_Trailing:            _slice_trailing_backward(op)
	case ml.Slice_Leading:             _slice_leading_backward(op)
	case ml.Concat:                    _concat_backward(op)
	case ml.Linear:                    _linear_backward(op)
	case ml.Linear_Q4_K:               panic("Linear_Q4_K is _forward-only", loc)
	case ml.Linear_Q4_K_Gate_Up_Geglu: panic("Linear_Q4_K_Gate_Up_Geglu is _forward-only", loc)
	case ml.Linear_Q6_K:               panic("Linear_Q6_K is _forward-only", loc)
	case ml.Rope:                      _rope_backward(op)
	case ml.Layernorm:                 _layernorm_backward(op)
	case ml.Rmsnorm:                   _rmsnorm_backward(op)
	case ml.Rmsnorm_Rope:              panic("Rmsnorm_Rope is _forward-only", loc)
	case ml.Rmsnorm_Rope_Write_Cache:  panic("Rmsnorm_Rope_Write_Cache is _forward-only", loc)
	case ml.Add_Rmsnorm:               panic("Add_Rmsnorm is _forward-only", loc)
	case ml.Softmax:                   _softmax_backward(op)
	case ml.Entropy:                   _entropy_backward(op)
	case ml.Log_Softmax:               _log_softmax_backward(op)
	case ml.Mean_Squared_Error:        _mean_squared_error_backward(op)
	case ml.Smooth_L1:                 _smooth_l1_backward(op)
	case ml.Cross_Entropy:             _cross_entropy_backward(op)
	case ml.Relu:                      _relu_backward(op)
	case ml.Sigmoid:                   _sigmoid_backward(op)
	case ml.Gelu:                      _gelu_backward(op)
	case ml.Gelu_Mul:                  panic("Gelu_Mul is _forward-only", loc)
	case ml.Silu:                      _silu_backward(op)
	case ml.Tanh:                      _tanh_backward(op)
	case ml.Batched_Matmul:            _batched_matmul_backward(op)
	case ml.Permute:                   _permute_backward(op)
	case ml.Causal_Mask:               _causal_mask_backward(op)
	case ml.Attention:                 _attention_backward(op)
	case ml.Attention_Cache:           _attention_cache_backward(op, loc)
	case ml.Cast:                      _cast_backward(op)
	case ml.Lerp_Assign:               panic("Lerp_Assign is _forward-only", loc)
	case ml.Accumulate_Mean:           panic("Accumulate_Mean is _forward-only", loc)
	}
}

