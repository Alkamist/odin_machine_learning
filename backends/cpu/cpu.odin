package machine_learning_backend_cpu

import "base:builtin"
import "base:runtime"
import "base:intrinsics"

import "core:fmt"
import "core:mem"
import "core:math"
import "core:simd"
import "core:sync"
import "core:thread"

import ml "../../"

@(thread_local)
_global_odin_context: runtime.Context

when thread.IS_SUPPORTED {
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

		for {
			sync.sema_wait(&w.start_sem)
			if _shutdown do return

			d := _dispatch
			if w.id < d.task_count {
				chunk := (d.job_count + d.task_count - 1) / d.task_count
				start := w.id * chunk
				end   := start + chunk
				if end > d.job_count do end = d.job_count

				if start < end {
					d.chunk_proc(start, end, d.data)
				}
			}

			sync.wait_group_done(&_done_wg)
		}
	}

	_startup_thread_pool :: proc(thread_count: int) {
		_global_odin_context = context

		_shutdown = false
		n := thread_count - 1
		_workers = builtin.make([]^Worker, n)
		for i in 0 ..< n {
			w                    := builtin.new(Worker)
			w.id                  = i + 1
			w.thread              = thread.create(_worker_proc)
			w.thread.data         = w
			w.thread.init_context = _global_odin_context

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

		context = _global_odin_context

		_cleanup_thread_pool()
	}

	parallelize :: proc(job_count, task_count: int, data: $Data, job: proc(index: int, data: Data)) {
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

	parallelize :: proc(job_count, task_count: int, data: $Data, job: proc(index: int, data: Data)) {
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
}

Context :: struct {
	using _: ml.Context,
	arena:   mem.Arena,
}

@(require_results)
context_create :: proc(size: int, allocator := context.allocator, loc := #caller_location) -> ^ml.Context {
	ctx, ctx_err := builtin.new(Context, allocator=allocator, loc=loc)
	assert(ctx_err == nil, "Failed to allocate Context", loc=loc)

	arena_buf, arena_buf_err := builtin.make([]byte, size, allocator=context.allocator, loc=loc)
	assert(arena_buf_err == nil, "Failed to allocate CPU backend arena data", loc=loc)
	mem.arena_init(&ctx.arena, arena_buf)

	ml._context_init(ctx, {
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

	return ctx
}

context_destroy :: proc(ctx: ^ml.Context, allocator := context.allocator, loc := #caller_location) {
	ctx := cast(^Context)ctx
	ml._context_destroy(ctx, loc)
	builtin.delete(ctx.arena.data, loc=loc)
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
adam_m :: #force_inline proc(t: ml.Tensor) -> []f32 {
	bytes := transmute([]byte)t.buffers[.Adam_M]
	return ([^]f32)(raw_data(bytes))[:t.count]
}

@(require_results)
adam_v :: #force_inline proc(t: ml.Tensor) -> []f32 {
	bytes := transmute([]byte)t.buffers[.Adam_V]
	return ([^]f32)(raw_data(bytes))[:t.count]
}

buffer_alloc :: proc(byte_count: int, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	ctx       := cast(^Context)ml.current_context(loc=loc)
	allocator := persist ? context.allocator : mem.arena_allocator(&ctx.arena)

	bytes, err := builtin.make([]byte, byte_count, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate CPU buffer: %v", err, loc=loc)

	return transmute([ml.BACKEND_BUFFER_MAX_SIZE]byte)bytes
}

buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	builtin.delete(transmute([]byte)buffer, loc=loc)
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

update :: proc(opt: ml.Optimizer, t: ^ml.Tensor, loc: runtime.Source_Code_Location) {
	d := data(t^)
	g := gradient(t^)
	m := adam_m(t^)
	v := adam_v(t^)

	assert(d != nil, "Tensor Data is nil", loc=loc)
	assert(g != nil, "Tensor Gradient is nil", loc=loc)
	assert(m != nil, "Tensor Adam_M is nil", loc=loc)
	assert(v != nil, "Tensor Adam_V is nil", loc=loc)

	for i in 0 ..< builtin.len(d) {
		grad := g[i]

		m[i] = opt.beta1 * m[i] + (1 - opt.beta1) * grad
		v[i] = opt.beta2 * v[i] + (1 - opt.beta2) * grad * grad

		m_hat := m[i] / opt.bias_correction1
		v_hat := v[i] / opt.bias_correction2

		d[i] = d[i] * (1 - opt.learning_rate * opt.weight_decay) - opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)
		g[i] = 0
	}
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
	src_bytes := transmute([]byte)op.input.buffers[.Data]
	dst_bytes := transmute([]byte)op.output.buffers[.Data]
	_cast_bytes(src_bytes, op.input.type, dst_bytes, op.output.type, op.input.count)
}

cast_backward :: proc(op: ml.Operation) {
	dst_grad := transmute([]byte)op.output.buffers[.Gradient]
	src_grad := transmute([]byte)op.input.buffers[.Gradient]
	_cast_bytes_accumulate(dst_grad, op.output.type, src_grad, op.input.type, op.input.count)
}

_cast_bytes :: proc(src: []byte, src_type: ml.Data_Type, dst: []byte, dst_type: ml.Data_Type, count: int) {
	src_f32  := ([^]f32    )(raw_data(src))[:count] if src_type == .F32  else nil
	src_bf16 := ([^]ml.Bf16)(raw_data(src))[:count] if src_type == .Bf16 else nil
	src_f16  := ([^]f16    )(raw_data(src))[:count] if src_type == .F16  else nil

	dst_f32  := ([^]f32    )(raw_data(dst))[:count] if dst_type == .F32  else nil
	dst_bf16 := ([^]ml.Bf16)(raw_data(dst))[:count] if dst_type == .Bf16 else nil
	dst_f16  := ([^]f16    )(raw_data(dst))[:count] if dst_type == .F16  else nil

	for i in 0 ..< count {
		v: f32
		switch src_type {
		case .F32:  v = src_f32 [i]
		case .F16:  v = f32(src_f16[i])
		case .Bf16: v = ml.bf16_to_f32(src_bf16[i])
		}
		switch dst_type {
		case .F32:  dst_f32 [i] = v
		case .F16:  dst_f16 [i] = f16(v)
		case .Bf16: dst_bf16[i] = ml.bf16_from_f32(v)
		}
	}
}

_cast_bytes_accumulate :: proc(src: []byte, src_type: ml.Data_Type, dst: []byte, dst_type: ml.Data_Type, count: int) {
	src_f32  := ([^]f32    )(raw_data(src))[:count] if src_type == .F32  else nil
	src_bf16 := ([^]ml.Bf16)(raw_data(src))[:count] if src_type == .Bf16 else nil
	src_f16  := ([^]f16    )(raw_data(src))[:count] if src_type == .F16  else nil

	dst_f32  := ([^]f32    )(raw_data(dst))[:count] if dst_type == .F32  else nil
	dst_bf16 := ([^]ml.Bf16)(raw_data(dst))[:count] if dst_type == .Bf16 else nil
	dst_f16  := ([^]f16    )(raw_data(dst))[:count] if dst_type == .F16  else nil

	for i in 0 ..< count {
		v: f32
		switch src_type {
		case .F32:  v = src_f32 [i]
		case .F16:  v = f32(src_f16[i])
		case .Bf16: v = ml.bf16_to_f32(src_bf16[i])
		}
		switch dst_type {
		case .F32:  dst_f32 [i] += v
		case .F16:  dst_f16 [i] += f16(v)
		case .Bf16: dst_bf16[i]  = ml.bf16_from_f32(ml.bf16_to_f32(dst_bf16[i]) + v)
		}
	}
}

add_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Add).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				data(output)[o] = data(a)[o] + data(b)[j]
			}
		}
	case .Bf16:
		a_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Data]))
		o_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				o_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(a_bf[o]) + ml.bf16_to_f32(b_bf[j]))
			}
		}
	case .F16:
		a_h := ([^]f16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_h := ([^]f16)(raw_data(transmute([]byte)b.buffers     [.Data]))
		o_h := ([^]f16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				o_h[o] = f16(f32(a_h[o]) + f32(b_h[j]))
			}
		}
	}
}

add_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b      := op.variant.(ml.Add).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				gradient(a)[o] += gradient(output)[o]
				gradient(b)[j] += gradient(output)[o]
			}
		}
	case .Bf16:
		da_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Gradient]))
		db_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				dy := ml.bf16_to_f32(dy_bf[o])
				da_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(da_bf[o]) + dy)
				db_bf[j] = ml.bf16_from_f32(ml.bf16_to_f32(db_bf[j]) + dy)
			}
		}
	case .F16:
		da_h := ([^]f16)(raw_data(transmute([]byte)a.buffers     [.Gradient]))
		db_h := ([^]f16)(raw_data(transmute([]byte)b.buffers     [.Gradient]))
		dy_h := ([^]f16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				dy := f32(dy_h[o])
				da_h[o] = f16(f32(da_h[o]) + dy)
				db_h[j] = f16(f32(db_h[j]) + dy)
			}
		}
	}
}

sub_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Sub).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				data(output)[o] = data(a)[o] - data(b)[j]
			}
		}
	case .Bf16:
		a_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Data]))
		o_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				o_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(a_bf[o]) - ml.bf16_to_f32(b_bf[j]))
			}
		}
	case .F16: fmt.panicf("CPU sub_forward: F16 not yet supported")
	}
}

sub_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b      := op.variant.(ml.Sub).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				gradient(a)[o] += gradient(output)[o]
				gradient(b)[j] -= gradient(output)[o]
			}
		}
	case .Bf16:
		da_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Gradient]))
		db_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				dy := ml.bf16_to_f32(dy_bf[o])
				da_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(da_bf[o]) + dy)
				db_bf[j] = ml.bf16_from_f32(ml.bf16_to_f32(db_bf[j]) - dy)
			}
		}
	case .F16: fmt.panicf("CPU sub_backward: F16 not yet supported")
	}
}

mul_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Mul).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				data(output)[o] = data(a)[o] * data(b)[j]
			}
		}
	case .Bf16:
		a_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Data]))
		o_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				o_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(a_bf[o]) * ml.bf16_to_f32(b_bf[j]))
			}
		}
	case .F16: fmt.panicf("CPU mul_forward: F16 not yet supported")
	}
}

mul_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b      := op.variant.(ml.Mul).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				gradient(a)[o] += gradient(output)[o] * data(b)[j]
				gradient(b)[j] += gradient(output)[o] * data(a)[o]
			}
		}
	case .Bf16:
		a_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Data]))
		da_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Gradient]))
		db_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				dy   := ml.bf16_to_f32(dy_bf[o])
				a_v  := ml.bf16_to_f32(a_bf [o])
				b_v  := ml.bf16_to_f32(b_bf [j])
				da_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(da_bf[o]) + dy * b_v)
				db_bf[j] = ml.bf16_from_f32(ml.bf16_to_f32(db_bf[j]) + dy * a_v)
			}
		}
	case .F16: fmt.panicf("CPU mul_backward: F16 not yet supported")
	}
}

div_forward :: proc(op: ml.Operation) {
	a      := op.input
	output := op.output
	b      := op.variant.(ml.Div).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				data(output)[o] = data(a)[o] / data(b)[j]
			}
		}
	case .Bf16:
		a_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Data]))
		o_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				o_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(a_bf[o]) / ml.bf16_to_f32(b_bf[j]))
			}
		}
	case .F16: fmt.panicf("CPU div_forward: F16 not yet supported")
	}
}

div_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output
	b      := op.variant.(ml.Div).b
	stride := ml.len(a) / ml.len(b)

	switch a.type {
	case .F32:
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				gradient(a)[o] += gradient(output)[o] / data(b)[j]
				gradient(b)[j] += gradient(output)[o] * (-data(a)[o] / (data(b)[j] * data(b)[j]))
			}
		}
	case .Bf16:
		a_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		b_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Data]))
		da_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Gradient]))
		db_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)b.buffers     [.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i in 0 ..< stride {
			for j in 0 ..< ml.len(b) {
				o := i * ml.len(b) + j
				dy   := ml.bf16_to_f32(dy_bf[o])
				a_v  := ml.bf16_to_f32(a_bf [o])
				b_v  := ml.bf16_to_f32(b_bf [j])
				da_bf[o] = ml.bf16_from_f32(ml.bf16_to_f32(da_bf[o]) + dy / b_v)
				db_bf[j] = ml.bf16_from_f32(ml.bf16_to_f32(db_bf[j]) + dy * (-a_v / (b_v * b_v)))
			}
		}
	case .F16: fmt.panicf("CPU div_backward: F16 not yet supported")
	}
}

exp_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	switch input.type {
	case .F32:
		for i in 0 ..< ml.len(input) {
			data(output)[i] = math.exp(data(input)[i])
		}
	case .Bf16:
		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< ml.len(input) {
			y_bf[i] = ml.bf16_from_f32(math.exp(ml.bf16_to_f32(x_bf[i])))
		}
	case .F16: fmt.panicf("CPU exp_forward: F16 not yet supported")
	}
}

exp_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	switch input.type {
	case .F32:
		for i in 0 ..< ml.len(input) {
			gradient(input)[i] += data(output)[i] * gradient(output)[i]
		}
	case .Bf16:
		y_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
		for i in 0 ..< ml.len(input) {
			dx_bf[i] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[i]) +
				ml.bf16_to_f32(y_bf[i]) * ml.bf16_to_f32(dy_bf[i]))
		}
	case .F16: fmt.panicf("CPU exp_backward: F16 not yet supported")
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
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Min)
	b       := variant.b

	for i in 0 ..< ml.len(a) {
		data(output)[i] = math.min(data(a)[i], data(b)[i])
	}
}

min_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output

	variant := op.variant.(ml.Min)
	b       := variant.b

	for i in 0 ..< ml.len(a) {
		if data(a)[i] <= data(b)[i] {
			gradient(a)[i] += gradient(output)[i]
		} else {
			gradient(b)[i] += gradient(output)[i]
		}
	}
}

max_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Max)
	b       := variant.b

	for i in 0 ..< ml.len(a) {
		data(output)[i] = math.max(data(a)[i], data(b)[i])
	}
}

max_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output

	variant := op.variant.(ml.Max)
	b       := variant.b

	for i in 0 ..< ml.len(a) {
		if data(a)[i] >= data(b)[i] {
			gradient(a)[i] += gradient(output)[i]
		} else {
			gradient(b)[i] += gradient(output)[i]
		}
	}
}

mean_forward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  mean_forward_f32 (op)
	case .Bf16: mean_forward_bf16(op)
	case .F16:  fmt.panicf("CPU mean_forward: F16 not yet supported")
	}
}

mean_forward_f32 :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	count  := ml.len(output)
	size   := ml.len(input) / count

	for sample in 0 ..< count {
		sum: f32
		for i in 0 ..< size {
			index := sample * size + i
			sum += data(input)[index]
		}
		data(output)[sample] = sum / f32(size)
	}
}

mean_forward_bf16 :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	count  := ml.len(output)
	size   := ml.len(input) / count

	in_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
	out_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))

	for sample in 0 ..< count {
		sum: f32
		for i in 0 ..< size {
			sum += ml.bf16_to_f32(in_bf[sample * size + i])
		}
		out_bf[sample] = ml.bf16_from_f32(sum / f32(size))
	}
}

mean_backward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  mean_backward_f32 (op)
	case .Bf16: mean_backward_bf16(op)
	case .F16:  fmt.panicf("CPU mean_backward: F16 not yet supported")
	}
}

mean_backward_f32 :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	for sample in 0 ..< count {
		gradient_per_element := gradient(output)[sample] / f32(size)

		for i in 0 ..< size {
			input_index := sample * size + i
			gradient(input)[input_index] += gradient_per_element
		}
	}
}

mean_backward_bf16 :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	count := ml.len(output)
	size  := ml.len(input) / count

	dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
	dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))

	for sample in 0 ..< count {
		gradient_per_element := ml.bf16_to_f32(dy_bf[sample]) / f32(size)

		for i in 0 ..< size {
			idx := sample * size + i
			dx_bf[idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[idx]) + gradient_per_element)
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

	// Pure byte-copy of `size`-element rows; works at any dtype.
	elem_size  := ml.data_type_size(input.type)
	row_bytes  := size * elem_size
	src_bytes  := transmute([]byte)input.buffers [.Data]
	dst_bytes  := transmute([]byte)output.buffers[.Data]
	for index, i in indices {
		src_off := index * row_bytes
		dst_off := i     * row_bytes
		builtin.copy(dst_bytes[dst_off:dst_off + row_bytes], src_bytes[src_off:src_off + row_bytes])
	}
}

select_backward :: proc(op: ml.Operation) {
	weight, output := op.input, op.output
	indices := op.variant.(ml.Select).indices
	size    := ml.len(output) / builtin.len(indices)

	switch weight.type {
	case .F32:
		for i in 0 ..< builtin.len(indices) {
			for j in 0 ..< size {
				gradient(weight)[indices[i] * size + j] += gradient(output)[i * size + j]
			}
		}
	case .Bf16:
		dw := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Gradient]))
		dy := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i in 0 ..< builtin.len(indices) {
			for j in 0 ..< size {
				dst_idx := indices[i] * size + j
				src_idx := i * size + j
				dw[dst_idx] = ml.bf16_from_f32(ml.bf16_to_f32(dw[dst_idx]) + ml.bf16_to_f32(dy[src_idx]))
			}
		}
	case .F16:
		fmt.panicf("CPU select_backward: F16 not yet supported")
	}
}

slice_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Slice)
	start   := variant.start
	end     := variant.end

	switch input.type {
	case .F32:
		builtin.copy(data(output), data(input)[start:end])
	case .Bf16:
		in_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		out_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< ml.len(output) {
			out_bf[i] = in_bf[start + i]
		}
	case .F16:  fmt.panicf("CPU slice_forward: F16 not yet supported")
	}
}

slice_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice)
	start   := variant.start

	switch input.type {
	case .F32:
		for i in 0 ..< ml.len(output) {
			gradient(input)[start + i] += gradient(output)[i]
		}
	case .Bf16:
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i in 0 ..< ml.len(output) {
			idx := start + i
			dx_bf[idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[idx]) + ml.bf16_to_f32(dy_bf[i]))
		}
	case .F16:  fmt.panicf("CPU slice_backward: F16 not yet supported")
	}
}

slice_trailing_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Slice_Trailing)
	start   := variant.start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml._leading_count(input)

	switch input.type {
	case .F32:
		for r in 0 ..< leading {
			in_off  := r * trailing + start
			out_off := r * new_trailing
			for i in 0 ..< new_trailing {
				data(output)[out_off + i] = data(input)[in_off + i]
			}
		}
	case .Bf16:
		in_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input .buffers[.Data]))
		out_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for r in 0 ..< leading {
			in_off  := r * trailing + start
			out_off := r * new_trailing
			for i in 0 ..< new_trailing {
				out_bf[out_off + i] = in_bf[in_off + i]
			}
		}
	case .F16: fmt.panicf("CPU slice_trailing_forward: F16 not yet supported")
	}
}

slice_trailing_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice_Trailing)
	start   := variant.start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml._leading_count(input)

	switch input.type {
	case .F32:
		for r in 0 ..< leading {
			in_off  := r * trailing + start
			out_off := r * new_trailing
			for i in 0 ..< new_trailing {
				gradient(input)[in_off + i] += gradient(output)[out_off + i]
			}
		}
	case .Bf16:
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input .buffers[.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for r in 0 ..< leading {
			in_off  := r * trailing + start
			out_off := r * new_trailing
			for i in 0 ..< new_trailing {
				idx := in_off + i
				dx_bf[idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[idx]) + ml.bf16_to_f32(dy_bf[out_off + i]))
			}
		}
	case .F16: fmt.panicf("CPU slice_trailing_backward: F16 not yet supported")
	}
}

concat_forward :: proc(op: ml.Operation) {
	output  := op.output
	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

	switch output.type {
	case .F32:
		dst_col := 0
		for input in inputs {
			in_trailing := input.shape[input.rank - 1]
			for r in 0 ..< leading {
				out_off := r * out_trailing + dst_col
				in_off  := r * in_trailing
				for i in 0 ..< in_trailing {
					data(output)[out_off + i] = data(input)[in_off + i]
				}
			}
			dst_col += in_trailing
		}
	case .Bf16:
		out_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		dst_col := 0
		for input in inputs {
			in_bf       := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers[.Data]))
			in_trailing := input.shape[input.rank - 1]
			for r in 0 ..< leading {
				out_off := r * out_trailing + dst_col
				in_off  := r * in_trailing
				for i in 0 ..< in_trailing {
					out_bf[out_off + i] = in_bf[in_off + i]
				}
			}
			dst_col += in_trailing
		}
	case .F16:  fmt.panicf("CPU concat_forward: F16 not yet supported")
	}
}

concat_backward :: proc(op: ml.Operation) {
	output := op.output

	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

	switch output.type {
	case .F32:
		src_col := 0
		for input in inputs {
			in_trailing := input.shape[input.rank - 1]
			for r in 0 ..< leading {
				out_off := r * out_trailing + src_col
				in_off  := r * in_trailing
				for i in 0 ..< in_trailing {
					gradient(input)[in_off + i] += gradient(output)[out_off + i]
				}
			}
			src_col += in_trailing
		}
	case .Bf16:
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		src_col := 0
		for input in inputs {
			dx_bf       := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers[.Gradient]))
			in_trailing := input.shape[input.rank - 1]
			for r in 0 ..< leading {
				out_off := r * out_trailing + src_col
				in_off  := r * in_trailing
				for i in 0 ..< in_trailing {
					idx := in_off + i
					dx_bf[idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[idx]) + ml.bf16_to_f32(dy_bf[out_off + i]))
				}
			}
			src_col += in_trailing
		}
	case .F16:  fmt.panicf("CPU concat_backward: F16 not yet supported")
	}
}

linear_forward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  linear_forward_f32 (op)
	case .Bf16: linear_forward_bf16(op)
	case .F16:  fmt.panicf("CPU linear_forward: F16 not yet supported")
	}
}

linear_forward_f32 :: proc(op: ml.Operation) {
	weight := op.variant.(ml.Linear).weight
	count  := ml.len(op.input) / weight.shape[1]

	parallelize(count, count, op, proc(index: int, op: ml.Operation) {
		input, output := op.input, op.output
		weight      := op.variant.(ml.Linear).weight
		output_size := weight.shape[0]
		input_size  := weight.shape[1]

		input_ptr  := ([^]f32)(raw_data(data(input)))
		output_ptr := ([^]f32)(raw_data(data(output)))
		weight_ptr := ([^]f32)(raw_data(data(weight)))

		x := input_ptr [index * input_size:]
		y := output_ptr[index * output_size:]

		for o in 0 ..< output_size {
			y[o] = _simd_dot_f32(weight_ptr[o * input_size:], x, input_size)
		}
	})
}

linear_forward_bf16 :: proc(op: ml.Operation) {
	weight := op.variant.(ml.Linear).weight
	count  := ml.len(op.input) / weight.shape[1]

	parallelize(count, count, op, proc(index: int, op: ml.Operation) {
		input, output := op.input, op.output
		weight      := op.variant.(ml.Linear).weight
		output_size := weight.shape[0]
		input_size  := weight.shape[1]

		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		w_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Data]))

		x_row := x_bf[index * input_size:]
		y_row := y_bf[index * output_size:]

		for o in 0 ..< output_size {
			w_row := w_bf[o * input_size:]
			acc:  f32
			for k in 0 ..< input_size {
				acc += ml.bf16_to_f32(w_row[k]) * ml.bf16_to_f32(x_row[k])
			}
			y_row[o] = ml.bf16_from_f32(acc)
		}
	})
}

linear_backward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  linear_backward_f32 (op)
	case .Bf16: linear_backward_bf16(op)
	case .F16:  fmt.panicf("CPU linear_backward: F16 not yet supported")
	}
}

linear_backward_bf16 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	count       := ml.len(op.input) / weight.shape[1]

	parallelize(output_size, output_size, op, proc(o: int, op: ml.Operation) {
		weight      := op.variant.(ml.Linear).weight
		input_size  := weight.shape[1]
		output_size := weight.shape[0]
		count       := ml.len(op.input) / input_size

		x_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Data]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Gradient]))
		dw_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers   [.Gradient]))

		dw_row := dw_bf[o * input_size:]
		for k in 0 ..< input_size {
			acc: f32
			for c in 0 ..< count {
				acc += ml.bf16_to_f32(x_bf[c * input_size + k]) *
				       ml.bf16_to_f32(dy_bf[c * output_size + o])
			}
			dw_row[k] = ml.bf16_from_f32(ml.bf16_to_f32(dw_row[k]) + acc)
		}
	})

	parallelize(count, count, op, proc(c: int, op: ml.Operation) {
		weight      := op.variant.(ml.Linear).weight
		input_size  := weight.shape[1]
		output_size := weight.shape[0]

		w_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers   [.Data]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Gradient]))
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Gradient]))

		dx_row := dx_bf[c * input_size:]
		dy_row := dy_bf[c * output_size:]
		for k in 0 ..< input_size {
			acc: f32
			for o in 0 ..< output_size {
				acc += ml.bf16_to_f32(w_bf[o * input_size + k]) *
				       ml.bf16_to_f32(dy_row[o])
			}
			dx_row[k] = ml.bf16_from_f32(ml.bf16_to_f32(dx_row[k]) + acc)
		}
	})
}

linear_backward_f32 :: proc(op: ml.Operation) {
	weight      := op.variant.(ml.Linear).weight
	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	count       := ml.len(op.input) / input_size

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
			if dout == 0 do continue
			x := input_data_ptr[b * input_size:]
			_simd_axpy_f32(w_grad, x, dout, input_size)
		}
	})

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
			if dout == 0 do continue
			w_data := weight_data_ptr[o * input_size:]
			_simd_axpy_f32(dx, w_data, dout, input_size)
		}
	})
}

rope_forward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  rope_forward_f32 (op)
	case .Bf16: rope_forward_bf16(op)
	case .F16:  fmt.panicf("CPU rope_forward: F16 not yet supported")
	}
}

rope_forward_f32 :: proc(op: ml.Operation) {
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

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< rotate_pair_count {
				cache_idx := t * half_head + i
				cos_val := data(cos_cache)[cache_idx]
				sin_val := data(sin_cache)[cache_idx]

				x := data(input)[head_offset + i * 2]
				y := data(input)[head_offset + i * 2 + 1]

				data(output)[head_offset + i * 2]     = x * cos_val - y * sin_val
				data(output)[head_offset + i * 2 + 1] = x * sin_val + y * cos_val
			}
			for i in rotate_pair_count ..< half_head {
				data(output)[head_offset + i * 2]     = data(input)[head_offset + i * 2]
				data(output)[head_offset + i * 2 + 1] = data(input)[head_offset + i * 2 + 1]
			}
		}
	}
}

rope_forward_bf16 :: proc(op: ml.Operation) {
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

	in_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
	out_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< rotate_pair_count {
				cache_idx := t * half_head + i
				cos_val := data(cos_cache)[cache_idx]
				sin_val := data(sin_cache)[cache_idx]

				x := ml.bf16_to_f32(in_bf[head_offset + i * 2])
				y := ml.bf16_to_f32(in_bf[head_offset + i * 2 + 1])

				out_bf[head_offset + i * 2]     = ml.bf16_from_f32(x * cos_val - y * sin_val)
				out_bf[head_offset + i * 2 + 1] = ml.bf16_from_f32(x * sin_val + y * cos_val)
			}
			for i in rotate_pair_count ..< half_head {
				out_bf[head_offset + i * 2]     = in_bf[head_offset + i * 2]
				out_bf[head_offset + i * 2 + 1] = in_bf[head_offset + i * 2 + 1]
			}
		}
	}
}

rope_backward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  rope_backward_f32 (op)
	case .Bf16: rope_backward_bf16(op)
	case .F16:  fmt.panicf("CPU rope_backward: F16 not yet supported")
	}
}

rope_backward_f32 :: proc(op: ml.Operation) {
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

rope_backward_bf16 :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant           := op.variant.(ml.Rope)
	head_count        := variant.head_count
	rotate_pair_count := variant.rotate_pair_count
	cos_cache         := variant.cos_cache
	sin_cache         := variant.sin_cache
	token_count       := input.shape[0]
	head_size         := input.shape[input.rank - 1] / head_count
	half_head         := head_size / 2

	dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
	dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< rotate_pair_count {
				cache_idx := t * half_head + i
				cos_val := data(cos_cache)[cache_idx]
				sin_val := data(sin_cache)[cache_idx]

				grad_x := ml.bf16_to_f32(dy_bf[head_offset + i * 2])
				grad_y := ml.bf16_to_f32(dy_bf[head_offset + i * 2 + 1])

				lo_idx := head_offset + i * 2
				hi_idx := lo_idx + 1
				dx_bf[lo_idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[lo_idx]) +  grad_x * cos_val + grad_y * sin_val)
				dx_bf[hi_idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[hi_idx]) + -grad_x * sin_val + grad_y * cos_val)
			}
			for i in rotate_pair_count ..< half_head {
				lo_idx := head_offset + i * 2
				hi_idx := lo_idx + 1
				dx_bf[lo_idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[lo_idx]) + ml.bf16_to_f32(dy_bf[lo_idx]))
				dx_bf[hi_idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[hi_idx]) + ml.bf16_to_f32(dy_bf[hi_idx]))
			}
		}
	}
}

LAYERNORM_EPSILON :: 1e-5

layernorm_forward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  layernorm_forward_f32 (op)
	case .Bf16: layernorm_forward_bf16(op)
	case .F16:  fmt.panicf("CPU layernorm_forward: F16 not yet supported")
	}
}

layernorm_forward_f32 :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	for c in 0 ..< count {
		offset := c * size

		m: f32
		for i in 0 ..< size {
			m += data(input)[offset + i]
		}
		m /= f32(size)

		v: f32
		for i in 0 ..< size {
			x_shift := data(input)[offset + i] - m
			v += x_shift * x_shift
		}
		v /= f32(size)

		s: f32 = 1.0 / math.sqrt(v + f32(LAYERNORM_EPSILON))
		for i in 0 ..< size {
			n := (s * (data(input)[offset + i] - m))
			o := n * data(weight)[i]
			data(output)[offset + i] = o
		}

		data(mean)[c] = m
		data(rstd)[c] = s
	}
}

layernorm_forward_bf16 :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
	y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
	w_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Data]))

	for c in 0 ..< count {
		offset := c * size

		m: f32
		for i in 0 ..< size {
			m += ml.bf16_to_f32(x_bf[offset + i])
		}
		m /= f32(size)

		v: f32
		for i in 0 ..< size {
			x_shift := ml.bf16_to_f32(x_bf[offset + i]) - m
			v += x_shift * x_shift
		}
		v /= f32(size)

		s: f32 = 1.0 / math.sqrt(v + f32(LAYERNORM_EPSILON))
		for i in 0 ..< size {
			n := s * (ml.bf16_to_f32(x_bf[offset + i]) - m)
			y_bf[offset + i] = ml.bf16_from_f32(n * ml.bf16_to_f32(w_bf[i]))
		}

		data(mean)[c] = m
		data(rstd)[c] = s
	}
}

layernorm_backward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  layernorm_backward_f32 (op)
	case .Bf16: layernorm_backward_bf16(op)
	case .F16:  fmt.panicf("CPU layernorm_backward: F16 not yet supported")
	}
}

layernorm_backward_f32 :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	for c in 0 ..< count {
		offset := c * size

		dnorm_mean:      f32
		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := (data(input)[offset + i] - data(mean)[c]) * data(rstd)[c]
			dnorm := data(weight)[i] * gradient(output)[offset + i]
			dnorm_mean      += dnorm
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_mean      /= f32(size)
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			norm  := (data(input)[offset + i] - data(mean)[c]) * data(rstd)[c]
			dnorm := data(weight)[i] * gradient(output)[offset + i]

			gradient(weight)[i] += norm * gradient(output)[offset + i]

			grad: f32
			grad += dnorm
			grad -= dnorm_mean
			grad -= norm * dnorm_norm_mean
			grad *= data(rstd)[c]

			gradient(input)[offset + i] += grad
		}
	}
}

layernorm_backward_bf16 :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd
	size    := input.shape[input.rank - 1]
	count   := ml.len(input) / size

	x_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
	w_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Data]))
	dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
	dw_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Gradient]))
	dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))

	for c in 0 ..< count {
		offset := c * size
		mean_c := data(mean)[c]
		rstd_c := data(rstd)[c]

		dnorm_mean:      f32
		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := (ml.bf16_to_f32(x_bf[offset + i]) - mean_c) * rstd_c
			dnorm := ml.bf16_to_f32(w_bf[i]) * ml.bf16_to_f32(dy_bf[offset + i])
			dnorm_mean      += dnorm
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_mean      /= f32(size)
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			x_v   := ml.bf16_to_f32(x_bf[offset + i])
			dy_v  := ml.bf16_to_f32(dy_bf[offset + i])
			w_v   := ml.bf16_to_f32(w_bf[i])
			norm  := (x_v - mean_c) * rstd_c
			dnorm := w_v * dy_v

			dw_bf[i] = ml.bf16_from_f32(ml.bf16_to_f32(dw_bf[i]) + norm * dy_v)

			grad := dnorm - dnorm_mean - norm * dnorm_norm_mean
			grad *= rstd_c

			dx_bf[offset + i] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[offset + i]) + grad)
		}
	}
}

rmsnorm_forward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  rmsnorm_forward_f32 (op)
	case .Bf16: rmsnorm_forward_bf16(op)
	case .F16:  fmt.panicf("CPU rmsnorm_forward: F16 not yet supported")
	}
}

rmsnorm_forward_f32 :: proc(op: ml.Operation) {
	input        := op.input
	output       := op.output
	variant      := op.variant.(ml.Rmsnorm)
	weight       := variant.weight
	rstd         := variant.rstd
	eps          := variant.eps
	weight_bias  := f32(1.0) if variant.unit_offset else f32(0.0)
	size         := input.shape[input.rank - 1]
	count        := ml.len(input) / size

	for c in 0 ..< count {
		offset := c * size

		ms: f32
		for i in 0 ..< size {
			v := data(input)[offset + i]
			ms += v * v
		}
		ms /= f32(size)

		s: f32 = 1.0 / math.sqrt(ms + eps)
		for i in 0 ..< size {
			data(output)[offset + i] = s * data(input)[offset + i] * (data(weight)[i] + weight_bias)
		}

		data(rstd)[c] = s
	}
}

rmsnorm_forward_bf16 :: proc(op: ml.Operation) {
	input       := op.input
	output      := op.output
	variant     := op.variant.(ml.Rmsnorm)
	weight      := variant.weight
	rstd        := variant.rstd
	eps         := variant.eps
	weight_bias := f32(1.0) if variant.unit_offset else f32(0.0)
	size        := input.shape[input.rank - 1]
	count       := ml.len(input) / size

	x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
	y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
	w_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Data]))

	for c in 0 ..< count {
		offset := c * size

		ms: f32
		for i in 0 ..< size {
			v := ml.bf16_to_f32(x_bf[offset + i])
			ms += v * v
		}
		ms /= f32(size)

		s: f32 = 1.0 / math.sqrt(ms + eps)
		for i in 0 ..< size {
			y := s * ml.bf16_to_f32(x_bf[offset + i]) * (ml.bf16_to_f32(w_bf[i]) + weight_bias)
			y_bf[offset + i] = ml.bf16_from_f32(y)
		}

		data(rstd)[c] = s
	}
}

rmsnorm_backward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  rmsnorm_backward_f32 (op)
	case .Bf16: rmsnorm_backward_bf16(op)
	case .F16:  fmt.panicf("CPU rmsnorm_backward: F16 not yet supported")
	}
}

rmsnorm_backward_f32 :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant     := op.variant.(ml.Rmsnorm)
	weight      := variant.weight
	rstd        := variant.rstd
	weight_bias := f32(1.0) if variant.unit_offset else f32(0.0)
	size        := input.shape[input.rank - 1]
	count       := ml.len(input) / size

	for c in 0 ..< count {
		offset := c * size
		rstd_c := data(rstd)[c]

		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := data(input)[offset + i] * rstd_c
			dnorm := (data(weight)[i] + weight_bias) * gradient(output)[offset + i]
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			x_v   := data(input)[offset + i]
			dy_v  := gradient(output)[offset + i]
			w_v   := data(weight)[i] + weight_bias
			norm  := x_v * rstd_c
			dnorm := w_v * dy_v

			gradient(weight)[i] += norm * dy_v

			grad := (dnorm - norm * dnorm_norm_mean) * rstd_c
			gradient(input)[offset + i] += grad
		}
	}
}

rmsnorm_backward_bf16 :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant     := op.variant.(ml.Rmsnorm)
	weight      := variant.weight
	rstd        := variant.rstd
	weight_bias := f32(1.0) if variant.unit_offset else f32(0.0)
	size        := input.shape[input.rank - 1]
	count       := ml.len(input) / size

	x_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
	w_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Data]))
	dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
	dw_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)weight.buffers[.Gradient]))
	dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))

	for c in 0 ..< count {
		offset := c * size
		rstd_c := data(rstd)[c]

		dnorm_norm_mean: f32
		for i in 0 ..< size {
			norm  := ml.bf16_to_f32(x_bf[offset + i]) * rstd_c
			dnorm := (ml.bf16_to_f32(w_bf[i]) + weight_bias) * ml.bf16_to_f32(dy_bf[offset + i])
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			x_v   := ml.bf16_to_f32(x_bf[offset + i])
			dy_v  := ml.bf16_to_f32(dy_bf[offset + i])
			w_v   := ml.bf16_to_f32(w_bf[i]) + weight_bias
			norm  := x_v * rstd_c
			dnorm := w_v * dy_v

			dw_bf[i] = ml.bf16_from_f32(ml.bf16_to_f32(dw_bf[i]) + norm * dy_v)

			grad := (dnorm - norm * dnorm_norm_mean) * rstd_c
			dx_bf[offset + i] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[offset + i]) + grad)
		}
	}
}

softmax_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	switch input.type {
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
		free_all(context.temp_allocator)
	case .F16: fmt.panicf("CPU softmax_forward: F16 not yet supported")
	}
}

softmax_backward :: proc(op: ml.Operation) {
	size  := op.input.shape[op.input.rank - 1]
	count := ml.len(op.input) / size

	switch op.input.type {
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
		y_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Data]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Gradient]))
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input .buffers[.Gradient]))
		for sample in 0 ..< count {
			base := sample * size
			dot:  f32
			for i in 0 ..< size {
				dot += ml.bf16_to_f32(dy_bf[base + i]) * ml.bf16_to_f32(y_bf[base + i])
			}
			for i in 0 ..< size {
				y_v  := ml.bf16_to_f32(y_bf [base + i])
				dy_v := ml.bf16_to_f32(dy_bf[base + i])
				dx_bf[base + i] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[base + i]) + y_v * (dy_v - dot))
			}
		}
	case .F16: fmt.panicf("CPU softmax_backward: F16 not yet supported")
	}
}

log_softmax_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	size   := input.shape[input.rank - 1]
	count  := ml.len(input) / size

	switch input.type {
	case .F32:
		for sample in 0 ..< count {
			max_value := math.NEG_INF_F32
			for i in 0 ..< size {
				index := sample * size + i
				max_value = math.max(max_value, data(input)[index])
			}
			log_sum_exp: f32
			for i in 0 ..< size {
				index := sample * size + i
				log_sum_exp += math.exp(data(input)[index] - max_value)
			}
			log_sum_exp = math.ln(log_sum_exp) + max_value
			for i in 0 ..< size {
				index := sample * size + i
				data(output)[index] = data(input)[index] - log_sum_exp
			}
		}
	case .Bf16:
		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for sample in 0 ..< count {
			base := sample * size
			max_value := math.NEG_INF_F32
			for i in 0 ..< size {
				v := ml.bf16_to_f32(x_bf[base + i])
				if v > max_value { max_value = v }
			}
			lse: f32
			for i in 0 ..< size {
				lse += math.exp(ml.bf16_to_f32(x_bf[base + i]) - max_value)
			}
			lse = math.ln(lse) + max_value
			for i in 0 ..< size {
				y_bf[base + i] = ml.bf16_from_f32(ml.bf16_to_f32(x_bf[base + i]) - lse)
			}
		}
	case .F16: fmt.panicf("CPU log_softmax_forward: F16 not yet supported")
	}
}

log_softmax_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output
	size  := input.shape[input.rank - 1]
	count := ml.len(input) / size

	switch input.type {
	case .F32:
		for sample in 0 ..< count {
			gradient_sum: f32
			for i in 0 ..< size {
				output_index := sample * size + i
				gradient_sum += gradient(output)[output_index]
			}
			for i in 0 ..< size {
				index := sample * size + i
				gradient(input)[index] += gradient(output)[index] - math.exp(data(output)[index]) * gradient_sum
			}
		}
	case .Bf16:
		y_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input .buffers[.Gradient]))
		for sample in 0 ..< count {
			base := sample * size
			grad_sum: f32
			for i in 0 ..< size {
				grad_sum += ml.bf16_to_f32(dy_bf[base + i])
			}
			for i in 0 ..< size {
				dy_v := ml.bf16_to_f32(dy_bf[base + i])
				y_v  := ml.bf16_to_f32(y_bf [base + i])
				dx_bf[base + i] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[base + i]) + dy_v - math.exp(y_v) * grad_sum)
			}
		}
	case .F16: fmt.panicf("CPU log_softmax_backward: F16 not yet supported")
	}
}

entropy_forward :: proc(op: ml.Operation) {
	probabilities := op.input
	output        := op.output
	size          := probabilities.shape[probabilities.rank - 1]
	count         := ml.len(probabilities) / size

	switch probabilities.type {
	case .F32:
		for sample in 0 ..< count {
			entropy_value: f32
			for i in 0 ..< size {
				index := sample * size + i
				p      := data(probabilities)[index]
				p_safe := math.max(p, 1e-8)
				entropy_value -= p * math.ln(p_safe)
			}
			data(output)[sample] = entropy_value
		}
	case .Bf16:
		p_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)probabilities.buffers[.Data]))
		o_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output       .buffers[.Data]))
		for sample in 0 ..< count {
			entropy_value: f32
			base := sample * size
			for i in 0 ..< size {
				p      := ml.bf16_to_f32(p_bf[base + i])
				p_safe := math.max(p, f32(1e-8))
				entropy_value -= p * math.ln(p_safe)
			}
			o_bf[sample] = ml.bf16_from_f32(entropy_value)
		}
	case .F16: fmt.panicf("CPU entropy_forward: F16 not yet supported")
	}
}

entropy_backward :: proc(op: ml.Operation) {
	probabilities, output := op.input, op.output
	size  := probabilities.shape[probabilities.rank - 1]
	count := ml.len(probabilities) / size

	switch probabilities.type {
	case .F32:
		for sample in 0 ..< count {
			for i in 0 ..< size {
				index := sample * size + i
				p      := data(probabilities)[index]
				p_safe := math.max(p, 1e-8)
				grad := -(math.ln(p_safe) + 1.0)
				gradient(probabilities)[index] += gradient(output)[sample] * grad
			}
		}
	case .Bf16:
		p_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)probabilities.buffers[.Data]))
		dp_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)probabilities.buffers[.Gradient]))
		do_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output       .buffers[.Gradient]))
		for sample in 0 ..< count {
			base    := sample * size
			dout_v  := ml.bf16_to_f32(do_bf[sample])
			for i in 0 ..< size {
				p      := ml.bf16_to_f32(p_bf[base + i])
				p_safe := math.max(p, f32(1e-8))
				grad   := -(math.ln(p_safe) + 1.0)
				dp_bf[base + i] = ml.bf16_from_f32(ml.bf16_to_f32(dp_bf[base + i]) + dout_v * grad)
			}
		}
	case .F16: fmt.panicf("CPU entropy_backward: F16 not yet supported")
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

		// Find the maximum value for numerical stability.
		max_value := math.NEG_INF_F32
		for i in 0 ..< class_size {
			index := offset + i
			max_value = math.max(max_value, data(input)[index])
		}

		// Compute exponentials and sum for softmax denominator.
		sum: f32
		for i in 0 ..< class_size {
			index := offset + i
			exp_val := math.exp(data(input)[index] - max_value)
			data(probabilities)[index] = exp_val
			sum += exp_val
		}

		// Normalize to get actual probabilities.
		for i in 0 ..< class_size {
			index := offset + i
			data(probabilities)[index] /= sum
		}

		// Compute negative log likelihood.
		target_index := offset + target
		data(output)[sample] = -data(input)[target_index] + max_value + math.ln(sum)
	}
}

cross_entropy_backward :: proc(op: ml.Operation) {
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
	switch input.type {
	case .F32:
		for i in 0 ..< ml.len(input) {
			data(output)[i] = fwd_f32(data(input)[i])
		}
	case .Bf16:
		x_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		y_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i in 0 ..< ml.len(input) {
			y_bf[i] = ml.bf16_from_f32(fwd_f32(ml.bf16_to_f32(x_bf[i])))
		}
	case .F16: fmt.panicf("CPU unary forward: F16 not yet supported")
	}
}

_unary_backward_dispatch :: proc(op: ml.Operation, local_grad_from_input: proc(x: f32) -> f32) {
	input, output := op.input, op.output
	switch input.type {
	case .F32:
		for i in 0 ..< ml.len(input) {
			gradient(input)[i] += gradient(output)[i] * local_grad_from_input(data(input)[i])
		}
	case .Bf16:
		x_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
		for i in 0 ..< ml.len(input) {
			x_v := ml.bf16_to_f32(x_bf[i])
			dx_bf[i] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[i]) +
				ml.bf16_to_f32(dy_bf[i]) * local_grad_from_input(x_v))
		}
	case .F16: fmt.panicf("CPU unary backward: F16 not yet supported")
	}
}

relu_forward :: proc(op: ml.Operation) {
	_unary_forward_dispatch(op, proc(x: f32) -> f32 { return x < 0 ? 0 : x })
}

relu_backward :: proc(op: ml.Operation) {
	_unary_backward_dispatch(op, proc(x: f32) -> f32 { return x > 0 ? 1 : 0 })
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
	switch op.input.type {
	case .F32:  batched_matmul_forward_f32 (op)
	case .Bf16: batched_matmul_forward_bf16(op)
	case .F16:  fmt.panicf("CPU batched_matmul_forward: F16 not yet supported")
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

		for kk in 0 ..< kk_count {
			b_row := b_ptr[bi * kk_count * n + kk * n:]
			_simd_axpy_f32(c_row, b_row, a_row[kk], n)
		}
	})
}

batched_matmul_backward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  batched_matmul_backward_f32 (op)
	case .Bf16: batched_matmul_backward_bf16(op)
	case .F16:  fmt.panicf("CPU batched_matmul_backward: F16 not yet supported")
	}
}

batched_matmul_backward_bf16 :: proc(op: ml.Operation) {
	a           := op.input
	batch_count := a.shape[0]
	m           := a.shape[1]
	k           := a.shape[2]

	parallelize(batch_count * m, batch_count * m, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		bt      := op.variant.(ml.Batched_Matmul).b
		m       := a.shape[1]
		k_count := a.shape[2]
		n       := bt.shape[2]

		bi := idx / m
		i  := idx % m

		b_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)bt.buffers    [.Data]))
		dc_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		da_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Gradient]))

		dc_row := dc_bf[bi * m * n + i * n:]
		da_row := da_bf[bi * m * k_count + i * k_count:]

		for kk in 0 ..< k_count {
			acc: f32
			for j in 0 ..< n {
				acc += ml.bf16_to_f32(dc_row[j]) *
				       ml.bf16_to_f32(b_bf[bi * k_count * n + kk * n + j])
			}
			da_row[kk] = ml.bf16_from_f32(ml.bf16_to_f32(da_row[kk]) + acc)
		}
	})

	parallelize(batch_count * k, batch_count * k, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		bt      := op.variant.(ml.Batched_Matmul).b
		m       := a.shape[1]
		k_count := a.shape[2]
		n       := bt.shape[2]

		bi := idx / k_count
		kk := idx % k_count

		a_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)a.buffers     [.Data]))
		dc_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		db_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)bt.buffers    [.Gradient]))

		db_row := db_bf[bi * k_count * n + kk * n:]
		for j in 0 ..< n {
			acc: f32
			for ii in 0 ..< m {
				acc += ml.bf16_to_f32(a_bf[bi * m * k_count + ii * k_count + kk]) *
				       ml.bf16_to_f32(dc_bf[bi * m * n + ii * n + j])
			}
			db_row[j] = ml.bf16_from_f32(ml.bf16_to_f32(db_row[j]) + acc)
		}
	})
}

batched_matmul_backward_f32 :: proc(op: ml.Operation) {
	a := op.input
	batch_count := a.shape[0]
	m := a.shape[1]
	k := a.shape[2]

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

permute_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output
	axes   := op.variant.(ml.Permute).axes

	in_shape   := [3]int{input.shape [0],           input.shape [1], input.shape [2]}
	out_shape  := [3]int{output.shape[0],           output.shape[1], output.shape[2]}
	in_strides := [3]int{in_shape[1] * in_shape[2], in_shape[2],     1              }

	switch input.type {
	case .F32:
		for i0 in 0 ..< out_shape[0] {
			for i1 in 0 ..< out_shape[1] {
				for i2 in 0 ..< out_shape[2] {
					src: [3]int
					src[axes[0]] = i0
					src[axes[1]] = i1
					src[axes[2]] = i2

					src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
					dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

					data(output)[dst_idx] = data(input)[src_idx]
				}
			}
		}
	case .Bf16:
		in_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		out_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for i0 in 0 ..< out_shape[0] {
			for i1 in 0 ..< out_shape[1] {
				for i2 in 0 ..< out_shape[2] {
					src: [3]int
					src[axes[0]] = i0
					src[axes[1]] = i1
					src[axes[2]] = i2

					src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
					dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

					out_bf[dst_idx] = in_bf[src_idx]
				}
			}
		}
	case .F16:  fmt.panicf("CPU permute_forward: F16 not yet supported")
	}
}

permute_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	axes    := op.variant.(ml.Permute).axes

	in_shape   := [3]int{input.shape [0],           input.shape [1], input.shape [2]}
	out_shape  := [3]int{output.shape[0],           output.shape[1], output.shape[2]}
	in_strides := [3]int{in_shape[1] * in_shape[2], in_shape[2],     1              }

	switch input.type {
	case .F32:
		for i0 in 0 ..< out_shape[0] {
			for i1 in 0 ..< out_shape[1] {
				for i2 in 0 ..< out_shape[2] {
					src: [3]int
					src[axes[0]] = i0
					src[axes[1]] = i1
					src[axes[2]] = i2

					src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
					dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

					gradient(input)[src_idx] += gradient(output)[dst_idx]
				}
			}
		}
	case .Bf16:
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for i0 in 0 ..< out_shape[0] {
			for i1 in 0 ..< out_shape[1] {
				for i2 in 0 ..< out_shape[2] {
					src: [3]int
					src[axes[0]] = i0
					src[axes[1]] = i1
					src[axes[2]] = i2

					src_idx := src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2]
					dst_idx := (i0 * out_shape[1] + i1) * out_shape[2] + i2

					dx_bf[src_idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[src_idx]) + ml.bf16_to_f32(dy_bf[dst_idx]))
				}
			}
		}
	case .F16:  fmt.panicf("CPU permute_backward: F16 not yet supported")
	}
}

causal_mask_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := ml.len(input) / block_size

	switch input.type {
	case .F32:
		for blk in 0 ..< n_blocks {
			offset := blk * block_size
			for t1 in 0 ..< T {
				for t2 in 0 ..< T {
					idx := offset + t1 * T + t2
					if t2 <= t1 {
						data(output)[idx] = data(input)[idx]
					} else {
						data(output)[idx] = math.NEG_INF_F32
					}
				}
			}
		}
	case .Bf16:
		neg_inf_bf := ml.bf16_from_f32(math.NEG_INF_F32)
		in_bf  := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Data]))
		out_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Data]))
		for blk in 0 ..< n_blocks {
			offset := blk * block_size
			for t1 in 0 ..< T {
				for t2 in 0 ..< T {
					idx := offset + t1 * T + t2
					if t2 <= t1 {
						out_bf[idx] = in_bf[idx]
					} else {
						out_bf[idx] = neg_inf_bf
					}
				}
			}
		}
	case .F16:  fmt.panicf("CPU causal_mask_forward: F16 not yet supported")
	}
}

causal_mask_backward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := ml.len(input) / block_size

	switch input.type {
	case .F32:
		for blk in 0 ..< n_blocks {
			offset := blk * block_size
			for t1 in 0 ..< T {
				for t2 in 0 ..= t1 {
					idx := offset + t1 * T + t2
					gradient(input)[idx] += gradient(output)[idx]
				}
			}
		}
	case .Bf16:
		dx_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)input.buffers [.Gradient]))
		dy_bf := ([^]ml.Bf16)(raw_data(transmute([]byte)output.buffers[.Gradient]))
		for blk in 0 ..< n_blocks {
			offset := blk * block_size
			for t1 in 0 ..< T {
				for t2 in 0 ..= t1 {
					idx := offset + t1 * T + t2
					dx_bf[idx] = ml.bf16_from_f32(ml.bf16_to_f32(dx_bf[idx]) + ml.bf16_to_f32(dy_bf[idx]))
				}
			}
		}
	case .F16:  fmt.panicf("CPU causal_mask_backward: F16 not yet supported")
	}
}

attention_forward :: proc(op: ml.Operation) {
	switch op.input.type {
	case .F32:  attention_forward_f32 (op)
	case .Bf16: attention_forward_bf16(op)
	case .F16:  fmt.panicf("CPU attention_forward: F16 not yet supported")
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
	switch op.input.type {
	case .F32:  attention_backward_f32 (op)
	case .Bf16: attention_backward_bf16(op)
	case .F16:  fmt.panicf("CPU attention_backward: F16 not yet supported")
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
					_simd_axpy_f32(v_grad[v_offset:], d_out, sm_row[t_k], head_size)
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
					_simd_axpy_f32(d_q_vec, k_data[k_offset:], d_p[t_k], head_size)
					_simd_axpy_f32(k_grad[k_offset:], q_vec,   d_p[t_k], head_size)
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

		q_data   := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Data]))
		q_grad   := ([^]ml.Bf16)(raw_data(transmute([]byte)op.input.buffers [.Gradient]))
		k_data   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.key.buffers    [.Data]))
		k_grad   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.key.buffers    [.Gradient]))
		v_data   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.value.buffers  [.Data]))
		v_grad   := ([^]ml.Bf16)(raw_data(transmute([]byte)v.value.buffers  [.Gradient]))
		out_grad := ([^]ml.Bf16)(raw_data(transmute([]byte)op.output.buffers[.Gradient]))
		sm_ptr   := ([^]f32)(raw_data(data(v.softmax_outputs)))
		dp_ptr   := ([^]f32)(raw_data(data(v.d_p_scratch)))

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
						d_out_d := ml.bf16_to_f32(out_grad[d_out_offset + d])
						dot += d_out_d * ml.bf16_to_f32(v_data[v_offset + d])
					}
					d_p[t_k] = dot

					p_val := sm_row[t_k]
					for d in 0 ..< head_size {
						existing := ml.bf16_to_f32(v_grad[v_offset + d])
						contrib  := ml.bf16_to_f32(out_grad[d_out_offset + d]) * p_val
						v_grad[v_offset + d] = ml.bf16_from_f32(existing + contrib)
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
						q_d  := ml.bf16_to_f32(q_data[q_offset + d])
						k_d  := ml.bf16_to_f32(k_data[k_offset + d])
						dq_d := ml.bf16_to_f32(q_grad[q_offset + d])
						dk_d := ml.bf16_to_f32(k_grad[k_offset + d])
						q_grad[q_offset + d] = ml.bf16_from_f32(dq_d + scale * k_d)
						k_grad[k_offset + d] = ml.bf16_from_f32(dk_d + scale * q_d)
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

	k_new_bytes := transmute([]byte)v.key.buffers     [.Data]
	v_new_bytes := transmute([]byte)v.value.buffers   [.Data]
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

	switch op.input.type {
	case .F32:  attention_cache_forward_f32 (op)
	case .Bf16: attention_cache_forward_bf16(op)
	case .F16:  fmt.panicf("CPU attention_cache_forward: F16 not yet supported")
	}
}

attention_cache_forward_f32 :: proc(op: ml.Operation) {
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
				score: f32
				for d in 0 ..< head_size {
					score += ml.bf16_to_f32(q_ptr[q_offset + d]) * ml.bf16_to_f32(k_ptr[k_offset + d])
				}
				score *= inv_sqrt_d
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

attention_cache_backward :: proc(op: ml.Operation) {
	fmt.panicf("attention_with_cache is forward-only (inference path); backward is not implemented")
}