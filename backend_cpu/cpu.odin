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

import ml "../"

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

		sync.mutex_lock(&_pool_mutex)
		defer sync.mutex_unlock(&_pool_mutex)

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

backend := ml.Backend_VTable{
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

Backend_Data :: struct {
	arena: mem.Arena,
}

init :: proc(ctx: ^ml.Context, size: int, loc: runtime.Source_Code_Location) {
	backend_data, backend_data_err := builtin.new(Backend_Data, allocator=context.allocator, loc=loc)
	assert(backend_data_err == nil, "Failed to allocate CPU Backend_Data", loc=loc)

	arena_buf, arena_buf_err := builtin.make([]byte, size, allocator=context.allocator, loc=loc)
	assert(arena_buf_err == nil, "Failed to allocate CPU backend arena data", loc=loc)

	mem.arena_init(&backend_data.arena, arena_buf)
	ctx.backend_data = backend_data
}

destroy :: proc(ctx: ^ml.Context, loc: runtime.Source_Code_Location) {
	backend_data := cast(^Backend_Data)ctx.backend_data
	if backend_data == nil { return }

	builtin.delete(backend_data.arena.data, allocator=context.allocator, loc=loc)
	builtin.free(backend_data, allocator=context.allocator, loc=loc)
	ctx.backend_data = nil
}

clear :: proc(loc: runtime.Source_Code_Location) {
	backend_data := cast(^Backend_Data)ml.current_context(loc=loc).backend_data
	mem.arena_free_all(&backend_data.arena)
}

_buffer_get :: #force_inline proc(t: ml.Tensor, kind: ml.Buffer_Kind) -> []f32 {
	return transmute([]f32)t.buffers[kind]
}

@(require_results)
data :: #force_inline proc(t: ml.Tensor) -> []f32 {
	return transmute([]f32)t.buffers[.Data]
}

@(require_results)
gradient :: #force_inline proc(t: ml.Tensor) -> []f32 {
	return transmute([]f32)t.buffers[.Gradient]
}

@(require_results)
adam_m :: #force_inline proc(t: ml.Tensor) -> []f32 {
	return transmute([]f32)t.buffers[.Adam_M]
}

@(require_results)
adam_v :: #force_inline proc(t: ml.Tensor) -> []f32 {
	return transmute([]f32)t.buffers[.Adam_V]
}

buffer_alloc :: proc(len: int, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	backend_data := cast(^Backend_Data)ml.current_context(loc=loc).backend_data
	allocator := persist ? context.allocator : mem.arena_allocator(&backend_data.arena)

	data, err := builtin.make([]f32, len, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate CPU buffer: %v", err, loc=loc)

	return transmute([ml.BACKEND_BUFFER_MAX_SIZE]byte)data
}

buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	builtin.delete(transmute([]f32)buffer, loc=loc)
}

buffer_get :: proc(buffer: ml.Backend_Buffer, data: []f32, loc: runtime.Source_Code_Location) {
	builtin.copy(data, transmute([]f32)buffer)
}

buffer_set :: proc(buffer: ml.Backend_Buffer, data: []f32, loc: runtime.Source_Code_Location) {
	builtin.copy(transmute([]f32)buffer, data)
}

buffer_copy :: proc(dst, src: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	builtin.copy(transmute([]f32)dst, transmute([]f32)src)
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
	}
}

add_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Add)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			data(output)[o] = data(a)[o] + data(b)[j]
		}
	}
}

add_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output

	variant := op.variant.(ml.Add)
	b       := variant.b

	stride := ml.len(a) / ml.len(b)
	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			gradient(a)[o] += gradient(output)[o]
			gradient(b)[j] += gradient(output)[o]
		}
	}
}

sub_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Sub)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			data(output)[o] = data(a)[o] - data(b)[j]
		}
	}
}

sub_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output

	variant := op.variant.(ml.Sub)
	b       := variant.b

	stride := ml.len(a) / ml.len(b)
	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			gradient(a)[o] += gradient(output)[o]
			gradient(b)[j] -= gradient(output)[o]
		}
	}
}

mul_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Mul)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			data(output)[o] = data(a)[o] * data(b)[j]
		}
	}
}

mul_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output

	variant := op.variant.(ml.Mul)
	b       := variant.b

	stride := ml.len(a) / ml.len(b)
	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			gradient(a)[o] += gradient(output)[o] * data(b)[j]
			gradient(b)[j] += gradient(output)[o] * data(a)[o]
		}
	}
}

div_forward :: proc(op: ml.Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(ml.Div)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			data(output)[o] = data(a)[o] / data(b)[j]
		}
	}
}

div_backward :: proc(op: ml.Operation) {
	a, output := op.input, op.output

	variant := op.variant.(ml.Div)
	b       := variant.b

	stride := ml.len(a) / ml.len(b)
	for i in 0 ..< stride {
		for j in 0 ..< ml.len(b) {
			o := i * ml.len(b) + j
			gradient(a)[o] += gradient(output)[o] / data(b)[j]
			gradient(b)[j] += gradient(output)[o] * (-data(a)[o] / (data(b)[j] * data(b)[j]))
		}
	}
}

exp_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< ml.len(input) {
		data(output)[i] = math.exp(data(input)[i])
	}
}

exp_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	for i in 0 ..< ml.len(input) {
		gradient(input)[i] += data(output)[i] * gradient(output)[i]
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
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Mean)
	size    := variant.size
	count   := variant.count

	for sample in 0 ..< count {
		sum: f32
		for i in 0 ..< size {
			index := sample * size + i
			sum += data(input)[index]
		}
		data(output)[sample] = sum / f32(size)
	}
}

mean_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Mean)
	size    := variant.size
	count   := variant.count

	for sample in 0 ..< count {
		gradient_per_element := gradient(output)[sample] / f32(size)

		for i in 0 ..< size {
			input_index := sample * size + i
			gradient(input)[input_index] += gradient_per_element
		}
	}
}

transpose_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Transpose)
	rows    := variant.rows
	columns := ml.len(input) / rows

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			data(output)[j * rows + i] = data(input)[i * columns + j]
		}
	}
}

transpose_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Transpose)
	rows    := variant.rows

	columns := ml.len(input) / rows

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			gradient(input)[i * columns + j] += gradient(output)[j * rows + i]
		}
	}
}

select_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Select)
	indices := variant.indices
	size    := variant.size

	for i in 0 ..< builtin.len(indices) {
		for j in 0 ..< size {
			data(output)[i * size + j] = data(input)[indices[i] * size + j]
		}
	}
}

select_backward :: proc(op: ml.Operation) {
	weight, output := op.input, op.output

	variant := op.variant.(ml.Select)
	indices := variant.indices
	size    := variant.size

	for i in 0 ..< builtin.len(indices) {
		for j in 0 ..< size {
			gradient(weight)[indices[i] * size + j] += gradient(output)[i * size + j]
		}
	}
}

slice_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Slice)
	start   := variant.start
	end     := variant.end

	builtin.copy(data(output), data(input)[start:end])
}

slice_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice)
	start   := variant.start

	for i in 0 ..< ml.len(output) {
		gradient(input)[start + i] += gradient(output)[i]
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

	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			data(output)[out_off + i] = data(input)[in_off + i]
		}
	}
}

slice_trailing_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Slice_Trailing)
	start   := variant.start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := ml._leading_count(input)

	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			gradient(input)[in_off + i] += gradient(output)[out_off + i]
		}
	}
}

concat_forward :: proc(op: ml.Operation) {
	output  := op.output
	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

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
}

concat_backward :: proc(op: ml.Operation) {
	output := op.output

	variant := op.variant.(ml.Concat)
	inputs  := variant.inputs

	leading      := ml._leading_count(inputs[0])
	out_trailing := output.shape[output.rank - 1]

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
}

linear_forward :: proc(op: ml.Operation) {
	count := op.variant.(ml.Linear).count

	parallelize(count, count, op, proc(index: int, op: ml.Operation) {
		input, output := op.input, op.output

		variant     := op.variant.(ml.Linear)
		weight      := variant.weight
		input_size  := variant.input_size
		output_size := variant.output_size

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

linear_backward :: proc(op: ml.Operation) {
	variant     := op.variant.(ml.Linear)
	count       := variant.count
	output_size := variant.output_size

	parallelize(output_size, output_size, op, proc(o: int, op: ml.Operation) {
		input, output := op.input, op.output
		variant     := op.variant.(ml.Linear)
		weight      := variant.weight
		input_size  := variant.input_size
		output_size := variant.output_size
		count       := variant.count

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
		variant     := op.variant.(ml.Linear)
		weight      := variant.weight
		input_size  := variant.input_size
		output_size := variant.output_size

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
	input       := op.input
	output      := op.output
	variant     := op.variant.(ml.Rope)
	token_count := variant.token_count
	head_count  := variant.head_count
	head_size   := variant.head_size
	base        := variant.base
	cos_cache   := variant.cos_cache
	sin_cache   := variant.sin_cache

	for pos in 0 ..< token_count {
		for i in 0 ..< head_size / 2 {
			theta := f32(pos) / math.pow(base, f32(i * 2) / f32(head_size))
			cache_idx := pos * (head_size / 2) + i
			data(cos_cache)[cache_idx] = math.cos(theta)
			data(sin_cache)[cache_idx] = math.sin(theta)
		}
	}

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< head_size / 2 {
				cache_idx := t * (head_size / 2) + i
				cos_val := data(cos_cache)[cache_idx]
				sin_val := data(sin_cache)[cache_idx]

				x := data(input)[head_offset + i * 2]
				y := data(input)[head_offset + i * 2 + 1]

				data(output)[head_offset + i * 2]     = x * cos_val - y * sin_val
				data(output)[head_offset + i * 2 + 1] = x * sin_val + y * cos_val
			}
		}
	}
}

rope_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant     := op.variant.(ml.Rope)
	token_count := variant.token_count
	head_count  := variant.head_count
	head_size   := variant.head_size
	cos_cache   := variant.cos_cache
	sin_cache   := variant.sin_cache

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< head_size / 2 {
				cache_idx := t * (head_size / 2) + i
				cos_val := data(cos_cache)[cache_idx]
				sin_val := data(sin_cache)[cache_idx]

				grad_x := gradient(output)[head_offset + i * 2]
				grad_y := gradient(output)[head_offset + i * 2 + 1]

				gradient(input)[head_offset + i * 2]     +=  grad_x * cos_val + grad_y * sin_val
				gradient(input)[head_offset + i * 2 + 1] += -grad_x * sin_val + grad_y * cos_val
			}
		}
	}
}

layernorm_forward :: proc(op: ml.Operation) {
	EPSILON :: 1e-5

	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd
	count   := variant.count
	size    := variant.size

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

		s: f32 = 1.0 / math.sqrt(v + EPSILON)
		for i in 0 ..< size {
			n := (s * (data(input)[offset + i] - m))
			o := n * data(weight)[i]
			data(output)[offset + i] = o
		}

		data(mean)[c] = m
		data(rstd)[c] = s
	}
}

layernorm_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Layernorm)
	weight  := variant.weight
	mean    := variant.mean
	rstd    := variant.rstd
	count   := variant.count
	size    := variant.size

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

softmax_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Softmax)
	size    := variant.size
	count   := variant.count

	for sample in 0 ..< count {
		// Find the maximum value for numerical stability.
		max_value := math.NEG_INF_F32
		for i in 0 ..< size {
			index := sample * size + i
			max_value = math.max(max_value, data(input)[index])
		}

		// Compute exp values and sum.
		sum: f32
		for i in 0 ..< size {
			index := sample * size + i
			exp_val := math.exp(data(input)[index] - max_value)
			data(output)[index] = exp_val
			sum += exp_val
		}

		// Normalize to get probabilities.
		for i in 0 ..< size {
			index := sample * size + i
			data(output)[index] /= sum
		}
	}
}

softmax_backward :: proc(op: ml.Operation) {
	count := op.variant.(ml.Softmax).count

	parallelize(count, count, op, proc(index: int, op: ml.Operation) {
		input, output := op.input, op.output
		size  := op.variant.(ml.Softmax).size
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
}

log_softmax_forward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(ml.Log_Softmax)
	size    := variant.size
	count   := variant.count

	for sample in 0 ..< count {
		// Find the maximum value for numerical stability.
		max_value := math.NEG_INF_F32
		for i in 0 ..< size {
			index := sample * size + i
			max_value = math.max(max_value, data(input)[index])
		}

		// Compute log_sum_exp for normalization.
		log_sum_exp: f32
		for i in 0 ..< size {
			index := sample * size + i
			log_sum_exp += math.exp(data(input)[index] - max_value)
		}
		log_sum_exp = math.ln(log_sum_exp) + max_value

		// Compute log probabilities.
		for i in 0 ..< size {
			index := sample * size + i
			data(output)[index] = data(input)[index] - log_sum_exp
		}
	}
}

log_softmax_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	variant := op.variant.(ml.Log_Softmax)
	size    := variant.size
	count   := variant.count

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
}

entropy_forward :: proc(op: ml.Operation) {
	probabilities := op.input
	output        := op.output
	variant       := op.variant.(ml.Entropy)
	size          := variant.size
	count         := variant.count

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
}

entropy_backward :: proc(op: ml.Operation) {
	probabilities, output := op.input, op.output

	variant := op.variant.(ml.Entropy)
	size    := variant.size
	count   := variant.count

	for sample in 0 ..< count {
		for i in 0 ..< size {
			index := sample * size + i
			p      := data(probabilities)[index]
			p_safe := math.max(p, 1e-8)

			grad := -(math.ln(p_safe) + 1.0)

			gradient(probabilities)[index] += gradient(output)[sample] * grad
		}
	}
}

mean_squared_error_forward :: proc(op: ml.Operation) {
	predictions := op.input
	output      := op.output
	variant     := op.variant.(ml.Mean_Squared_Error)
	targets     := variant.targets
	count       := variant.count
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

	variant := op.variant.(ml.Mean_Squared_Error)
	targets := variant.targets
	count   := variant.count

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
	class_size    := variant.class_size

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
	class_size    := variant.class_size

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

relu_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< ml.len(input) {
		if data(input)[i] < 0 {
			data(output)[i] = 0
		} else {
			data(output)[i] = data(input)[i]
		}
	}
}

relu_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	for i in 0 ..< ml.len(input) {
		if data(input)[i] > 0 {
			gradient(input)[i] += gradient(output)[i]
		}
	}
}

sigmoid_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< ml.len(input) {
		data(output)[i] = 1.0 / (1.0 + math.exp(-data(input)[i]))
	}
}

sigmoid_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	for i in 0 ..< ml.len(input) {
		sigmoid_value     := data(output)[i]
		gradient(input)[i] += gradient(output)[i] * sigmoid_value * (1.0 - sigmoid_value)
	}
}

GELU_SCALING_FACTOR :: 0.7978845608028654 // math.sqrt(f32(2) / math.PI)

gelu_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< ml.len(input) {
		x    := data(input)[i]
		cube := 0.044715 * x * x * x

		data(output)[i] = 0.5 * x * (1.0 + math.tanh(GELU_SCALING_FACTOR * (x + cube)))
	}
}

gelu_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	for i in 0 ..< ml.len(input) {
		x          := data(input)[i]
		cube       := 0.044715 * x * x * x
		tanh_arg   := GELU_SCALING_FACTOR * (x + cube)
		tanh_out   := math.tanh(tanh_arg)
		cosh_out   := math.cosh(tanh_arg)
		sech_out   := 1.0 / (cosh_out * cosh_out)
		local_grad := 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x)

		gradient(input)[i] += local_grad * gradient(output)[i]
	}
}

silu_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< ml.len(input) {
		sigmoid_val := 1.0 / (1.0 + math.exp(-data(input)[i]))
		data(output)[i] = data(input)[i] * sigmoid_val
	}
}

silu_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	for i in 0 ..< ml.len(input) {
		x           := data(input)[i]
		sigmoid_val := 1.0 / (1.0 + math.exp(-x))

		grad := sigmoid_val + x * sigmoid_val * (1.0 - sigmoid_val)

		gradient(input)[i] += gradient(output)[i] * grad
	}
}

tanh_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< ml.len(input) {
		data(output)[i] = math.tanh(data(input)[i])
	}
}

tanh_backward :: proc(op: ml.Operation) {
	input, output := op.input, op.output

	for i in 0 ..< ml.len(input) {
		tanh_value         := data(output)[i]
		gradient(input)[i] += gradient(output)[i] * (1.0 - tanh_value * tanh_value)
	}
}

batched_matmul_forward :: proc(op: ml.Operation) {
	variant := op.variant.(ml.Batched_Matmul)

	parallelize(variant.batch_count * variant.m, variant.batch_count * variant.m, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		variant := op.variant.(ml.Batched_Matmul)
		bt      := variant.b

		m := variant.m
		kk_count := variant.k
		n := variant.n

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
	variant := op.variant.(ml.Batched_Matmul)

	parallelize(variant.batch_count * variant.m, variant.batch_count * variant.m, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		variant := op.variant.(ml.Batched_Matmul)
		bt      := variant.b

		m := variant.m
		kk_count := variant.k
		n := variant.n

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

	parallelize(variant.batch_count * variant.k, variant.batch_count * variant.k, op, proc(idx: int, op: ml.Operation) {
		a       := op.input
		output  := op.output
		variant := op.variant.(ml.Batched_Matmul)
		bt      := variant.b

		m := variant.m
		kk_count := variant.k
		n := variant.n

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
}

permute_backward :: proc(op: ml.Operation) {
	input   := op.input
	output  := op.output
	axes    := op.variant.(ml.Permute).axes

	in_shape   := [3]int{input.shape [0],           input.shape [1], input.shape [2]}
	out_shape  := [3]int{output.shape[0],           output.shape[1], output.shape[2]}
	in_strides := [3]int{in_shape[1] * in_shape[2], in_shape[2],     1              }

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
}

causal_mask_forward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := ml.len(input) / block_size

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
}

causal_mask_backward :: proc(op: ml.Operation) {
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := ml.len(input) / block_size

	for blk in 0 ..< n_blocks {
		offset := blk * block_size
		for t1 in 0 ..< T {
			for t2 in 0 ..= t1 {
				idx := offset + t1 * T + t2
				gradient(input)[idx] += gradient(output)[idx]
			}
		}
	}
}