// CPU backend implementation. Owns:
//   - `Cpu_Storage` (per-tensor data + gradient + Adam state slices)
//   - The `_cpu_backend` instance and every `cpu_*` hook (alloc, free,
//     set_data, get_data, parameter_update, parameter_copy,
//     fill_gradient_with_ones, clear_storage, forward, backward).
//   - Every CPU op kernel — `cpu_*_forward` and `*_backward`.
//   - SIMD primitives (`_simd_dot_f32`, `_simd_axpy_f32`).
//   - The persistent worker pool that backs `parallelize`.
//   - Public accessors `data(t)` / `gradient(t)` (return CPU-side slices).
//
// `ml.odin` holds the backend-agnostic surface (`Backend`, `Tensor`,
// `Data_Type`, `Context`, `Operation`, op variants + public ops, autograd
// tape, optimizer). Adding a new op means: add a variant + public op proc
// in ml.odin, then add `cpu_X_forward` + `X_backward` to this file and
// register them in the dispatch switches below.
package machine_learning

import "base:builtin"
import "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:simd"
import "core:sync"
import "core:thread"

@(thread_local)
_global_odin_context: runtime.Context

when thread.IS_SUPPORTED {
	// Persistent worker pool. Each worker parks on its own semaphore and is
	// signaled directly per parallelize call — no per-call allocation, no
	// global queue, no busy-wait on completion.
	//
	// Worker 0 is the calling (main) thread; spawned workers serve task ids
	// 1 .. _thread_count-1.

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
	// Serializes parallelize fan-outs across host threads. Each fan-out claims
	// the entire pool for its duration, so concurrent callers wait their turn
	// rather than oversubscribing the cores.
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
			w := builtin.new(Worker)
			w.id = i + 1
			w.thread = thread.create(_worker_proc)
			w.thread.data = w
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

	// Get the current thread count.
	thread_count :: #force_inline proc() -> int {
		return _thread_count
	}

	// Set the thread count for parallelized calculations.
	// Should only be called from the main thread.
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

	// Parallelize a job over `task_count` workers. Each worker invokes `job`
	// once per index in its chunk of [0, job_count). The thunk type-erases
	// `data` and the user `job` pointer so the dispatch state is non-generic.
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
		if n > _thread_count do n = _thread_count
		if n > job_count     do n = job_count

		if n <= 1 {
			for i in 0 ..< job_count {
				job(i, data)
			}
			return
		}

		td := Thunk_Data{data = data, job = job}

		// Claim the pool for this fan-out so concurrent callers from other
		// host threads don't trample each other's _dispatch / _done_wg.
		sync.mutex_lock(&_pool_mutex)
		defer sync.mutex_unlock(&_pool_mutex)

		_dispatch = Dispatch{
			chunk_proc = thunk,
			data       = &td,
			job_count  = job_count,
			task_count = n,
		}

		// Wake n-1 spawned workers; main thread runs slice 0 itself.
		sync.wait_group_add(&_done_wg, n - 1)
		for i in 0 ..< n - 1 {
			sync.sema_post(&_workers[i].start_sem)
		}

		chunk := (job_count + n - 1) / n
		end := chunk
		if end > job_count do end = job_count
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

// CPU-resident storage. Activations populate just `data` and `gradient`;
// parameters also populate `adam_m` / `adam_v`. The storage struct lives
// in the active context's arena (for activations) or on the heap (for
// persistent / parameter allocations); the slices it points to live in
// the same place.
Cpu_Storage :: struct {
	data:     []f32,
	gradient: []f32,
	adam_m:   []f32, // empty unless this is a parameter (extra_buffers >= 2)
	adam_v:   []f32,
}

// Slice accessors for CPU tensors. Assert the tensor is CPU-resident;
// GPU code uses `set_data` / `get_data` (or backend-specific helpers).
@(require_results)
data :: #force_inline proc(t: Tensor) -> []f32 {
	return (cast(^Cpu_Storage)t.storage).data
}
@(require_results)
gradient :: #force_inline proc(t: Tensor) -> []f32 {
	return (cast(^Cpu_Storage)t.storage).gradient
}

// CPU backend singleton. `cpu_forward` / `cpu_backward` and the alloc /
// clear hooks are defined further down the file, alongside the op procs
// they dispatch into.
_cpu_backend := Backend{
	alloc                   = cpu_alloc,
	free                    = cpu_free,
	clear_storage           = cpu_clear_storage,
	set_data                = cpu_set_data,
	get_data                = cpu_get_data,
	parameter_update        = cpu_parameter_update,
	parameter_copy          = cpu_parameter_copy,
	fill_gradient_with_ones = cpu_fill_gradient_with_ones,
	forward                 = cpu_forward,
	backward                = cpu_backward,
}

// Allocate a `Cpu_Storage` for `t` with `n`-element data + gradient, and
// optional `extra_buffers` more (2 for parameters → adam_m + adam_v).
// `persistent=false` allocates everything from the active context's arena
// (reset in bulk by `cpu_clear_storage`); `persistent=true` heap-allocates
// (freed by `cpu_free`).
cpu_alloc :: proc(t: ^Tensor, n: int, persistent: bool, extra_buffers: int) {
	fmt.assertf(t.type == .F32, "cpu_alloc: only F32 is supported (got %v)", t.type)
	fmt.assertf(extra_buffers == 0 || extra_buffers == 2, "cpu_alloc: extra_buffers must be 0 or 2 (got %v)", extra_buffers)

	allocator := persistent ? context.allocator : arena_allocator()

	storage, serr := builtin.new(Cpu_Storage, allocator=allocator)
	fmt.assertf(serr == nil, "cpu_alloc: failed to allocate Cpu_Storage: %v", serr)

	derr: mem.Allocator_Error
	storage.data,     derr = builtin.make([]f32, n, allocator=allocator)
	fmt.assertf(derr == nil, "cpu_alloc: failed to allocate data: %v", derr)
	storage.gradient, derr = builtin.make([]f32, n, allocator=allocator)
	fmt.assertf(derr == nil, "cpu_alloc: failed to allocate gradient: %v", derr)

	if extra_buffers >= 2 {
		storage.adam_m, derr = builtin.make([]f32, n, allocator=allocator)
		fmt.assertf(derr == nil, "cpu_alloc: failed to allocate adam_m: %v", derr)
		storage.adam_v, derr = builtin.make([]f32, n, allocator=allocator)
		fmt.assertf(derr == nil, "cpu_alloc: failed to allocate adam_v: %v", derr)
	}

	t.storage = storage
}

// Free a heap-allocated CPU storage (persistent / parameter). No-op for
// arena-resident storage — that's reclaimed in bulk by `cpu_clear_storage`.
cpu_free :: proc(t: ^Tensor) {
	if t.storage == nil { return }
	s := cast(^Cpu_Storage)t.storage
	builtin.delete(s.data)
	builtin.delete(s.gradient)
	if s.adam_m != nil do builtin.delete(s.adam_m)
	if s.adam_v != nil do builtin.delete(s.adam_v)
	builtin.free(s)
	t.storage = nil
}

cpu_clear_storage :: proc() {
	mem.arena_free_all(&_current_ctx.arena)
}

cpu_set_data :: proc(t: ^Tensor, src: []f32) {
	d := data(t^)
	fmt.assertf(builtin.len(src) == builtin.len(d), "cpu_set_data size mismatch: src=%v data=%v", builtin.len(src), builtin.len(d))
	builtin.copy(d, src)
}

cpu_get_data :: proc(t: ^Tensor, dst: []f32) {
	d := data(t^)
	fmt.assertf(builtin.len(dst) == builtin.len(d), "cpu_get_data size mismatch: dst=%v data=%v", builtin.len(dst), builtin.len(d))
	builtin.copy(dst, d)
}

cpu_parameter_update :: proc(opt: Optimizer, p: ^Tensor) {
	s := cast(^Cpu_Storage)p.storage
	fmt.assertf(s != nil && s.adam_m != nil, "cpu_parameter_update: parameter has no Adam storage — was it allocated with extra_buffers=2?")

	for i in 0 ..< len(p^) {
		grad := s.gradient[i]

		s.adam_m[i] = opt.beta1 * s.adam_m[i] + (1 - opt.beta1) * grad
		s.adam_v[i] = opt.beta2 * s.adam_v[i] + (1 - opt.beta2) * grad * grad

		m_hat := s.adam_m[i] / opt.bias_correction1
		v_hat := s.adam_v[i] / opt.bias_correction2

		s.data[i] = s.data[i] * (1 - opt.learning_rate * opt.weight_decay) - opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)

		s.gradient[i] = 0
	}
}

cpu_parameter_copy :: proc(dst, src: ^Tensor) {
	d := cast(^Cpu_Storage)dst.storage
	s := cast(^Cpu_Storage)src.storage
	builtin.copy(d.data,     s.data)
	builtin.copy(d.gradient, s.gradient)
	if s.adam_m != nil && d.adam_m != nil {
		builtin.copy(d.adam_m, s.adam_m)
		builtin.copy(d.adam_v, s.adam_v)
	}
}

cpu_fill_gradient_with_ones :: proc(t: ^Tensor) {
	g := gradient(t^)
	for i in 0 ..< builtin.len(g) {
		g[i] = 1
	}
}

// Public accessor for the CPU backend, useful for explicitly creating a
// Context on the CPU (the default if you don't pass a backend).
@(require_results)
cpu_backend :: #force_inline proc() -> ^Backend {
	return &_cpu_backend
}

// CPU backend: forward dispatch. Exhaustive — the compiler warns if a
// future Operation_Variant case is added without a matching forward.
cpu_forward :: proc(op: Operation) {
	switch _ in op.variant {
	case Add:                cpu_add_forward                (op)
	case Sub:                cpu_sub_forward                (op)
	case Mul:                cpu_mul_forward                (op)
	case Div:                cpu_div_forward                (op)
	case Exp:                cpu_exp_forward                (op)
	case Clamp:              cpu_clamp_forward              (op)
	case Min:                cpu_min_forward                (op)
	case Max:                cpu_max_forward                (op)
	case Mean:               cpu_mean_forward               (op)
	case Transpose:          cpu_transpose_forward          (op)
	case Select:             cpu_select_forward             (op)
	case Slice:              cpu_slice_forward              (op)
	case Slice_Trailing:     cpu_slice_trailing_forward     (op)
	case Concat:             cpu_concat_forward             (op)
	case Linear:             cpu_linear_forward             (op)
	case Rope:               cpu_rope_forward               (op)
	case Layernorm:          cpu_layernorm_forward          (op)
	case Softmax:            cpu_softmax_forward            (op)
	case Entropy:            cpu_entropy_forward            (op)
	case Log_Softmax:        cpu_log_softmax_forward        (op)
	case Mean_Squared_Error: cpu_mean_squared_error_forward (op)
	case Cross_Entropy:      cpu_cross_entropy_forward      (op)
	case Relu:               cpu_relu_forward               (op)
	case Sigmoid:            cpu_sigmoid_forward            (op)
	case Gelu:               cpu_gelu_forward               (op)
	case Silu:               cpu_silu_forward               (op)
	case Tanh:               cpu_tanh_forward               (op)
	case Batched_Matmul:     cpu_batched_matmul_forward     (op)
	case Permute:            cpu_permute_forward            (op)
	case Causal_Mask:        cpu_causal_mask_forward        (op)
	}
}

// CPU backend: backward dispatch. Every op already has a per-variant
// `_backward` proc defined elsewhere in this file, so this switch is
// exhaustive — the compiler warns if a new variant is added without a
// matching case here.
cpu_backward :: proc(op: Operation) {
	switch _ in op.variant {
	case Add:                add_backward               (op)
	case Sub:                sub_backward               (op)
	case Mul:                mul_backward               (op)
	case Div:                div_backward               (op)
	case Exp:                exp_backward               (op)
	case Clamp:              clamp_backward             (op)
	case Min:                min_backward               (op)
	case Max:                max_backward               (op)
	case Mean:               mean_backward              (op)
	case Transpose:          transpose_backward         (op)
	case Select:             select_backward            (op)
	case Slice:              slice_backward             (op)
	case Slice_Trailing:     slice_trailing_backward    (op)
	case Concat:             concat_backward            (op)
	case Linear:             linear_backward            (op)
	case Rope:               rope_backward              (op)
	case Layernorm:          layernorm_backward         (op)
	case Softmax:            softmax_backward           (op)
	case Entropy:            entropy_backward           (op)
	case Log_Softmax:        log_softmax_backward       (op)
	case Mean_Squared_Error: mean_squared_error_backward(op)
	case Cross_Entropy:      cross_entropy_backward     (op)
	case Relu:               relu_backward              (op)
	case Sigmoid:            sigmoid_backward           (op)
	case Gelu:               gelu_backward              (op)
	case Silu:               silu_backward              (op)
	case Tanh:               tanh_backward              (op)
	case Batched_Matmul:     batched_matmul_backward    (op)
	case Permute:            permute_backward           (op)
	case Causal_Mask:        causal_mask_backward       (op)
	}
}


cpu_add_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Add)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			data(output)[o] = data(a)[o] + data(b)[j]
		}
	}
}

add_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Add)
	b       := variant.b

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			gradient(a)[o] += gradient(output)[o]
			gradient(b)[j] += gradient(output)[o]
		}
	}
}

cpu_sub_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Sub)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			data(output)[o] = data(a)[o] - data(b)[j]
		}
	}
}

sub_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Sub)
	b       := variant.b

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			gradient(a)[o] += gradient(output)[o]
			gradient(b)[j] -= gradient(output)[o]
		}
	}
}

cpu_mul_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Mul)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			data(output)[o] = data(a)[o] * data(b)[j]
		}
	}
}

mul_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Mul)
	b       := variant.b

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			gradient(a)[o] += gradient(output)[o] * data(b)[j]
			gradient(b)[j] += gradient(output)[o] * data(a)[o]
		}
	}
}

cpu_div_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Div)
	b       := variant.b
	stride  := variant.stride

	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			data(output)[o] = data(a)[o] / data(b)[j]
		}
	}
}

div_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Div)
	b       := variant.b

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			gradient(a)[o] += gradient(output)[o] / data(b)[j]
			gradient(b)[j] += gradient(output)[o] * (-data(a)[o] / (data(b)[j] * data(b)[j]))
		}
	}
}

cpu_exp_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		data(output)[i] = math.exp(data(input)[i])
	}
}

exp_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		gradient(input)[i] += data(output)[i] * gradient(output)[i]
	}
}

cpu_clamp_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< len(input) {
		data(output)[i] = math.clamp(data(input)[i], min_val, max_val)
	}
}

clamp_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< len(input) {
		if data(input)[i] >= min_val && data(input)[i] <= max_val {
			gradient(input)[i] += gradient(output)[i]
		}
	}
}

cpu_min_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Min)
	b       := variant.b

	for i in 0 ..< len(a) {
		data(output)[i] = math.min(data(a)[i], data(b)[i])
	}
}

min_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Min)
	b       := variant.b

	for i in 0 ..< len(a) {
		if data(a)[i] <= data(b)[i] {
			gradient(a)[i] += gradient(output)[i]
		} else {
			gradient(b)[i] += gradient(output)[i]
		}
	}
}

cpu_max_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Max)
	b       := variant.b

	for i in 0 ..< len(a) {
		data(output)[i] = math.max(data(a)[i], data(b)[i])
	}
}

max_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Max)
	b       := variant.b

	for i in 0 ..< len(a) {
		if data(a)[i] >= data(b)[i] {
			gradient(a)[i] += gradient(output)[i]
		} else {
			gradient(b)[i] += gradient(output)[i]
		}
	}
}

cpu_mean_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Mean)
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

mean_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Mean)
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

cpu_transpose_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Transpose)
	rows    := variant.rows
	columns := len(input) / rows

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			data(output)[j * rows + i] = data(input)[i * columns + j]
		}
	}
}

transpose_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Transpose)
	rows    := variant.rows

	columns := len(input) / rows

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			gradient(input)[i * columns + j] += gradient(output)[j * rows + i]
		}
	}
}

cpu_select_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Select)
	indices := variant.indices
	size    := variant.size

	for i in 0 ..< builtin.len(indices) {
		for j in 0 ..< size {
			data(output)[i * size + j] = data(input)[indices[i] * size + j]
		}
	}
}

select_backward :: proc(op: Operation, loc := #caller_location) {
	weight, output := op.input, op.output

	variant := op.variant.(Select)
	indices := variant.indices
	size    := variant.size

	for i in 0 ..< builtin.len(indices) {
		for j in 0 ..< size {
			gradient(weight)[indices[i] * size + j] += gradient(output)[i * size + j]
		}
	}
}

cpu_slice_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Slice)
	start   := variant.start
	end     := variant.end

	builtin.copy(data(output), data(input)[start:end])
}

slice_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Slice)
	start   := variant.start

	for i in 0 ..< len(output) {
		gradient(input)[start + i] += gradient(output)[i]
	}
}

cpu_slice_trailing_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Slice_Trailing)
	start   := variant.start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := _leading_count(input)

	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			data(output)[out_off + i] = data(input)[in_off + i]
		}
	}
}

slice_trailing_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Slice_Trailing)
	start   := variant.start

	trailing     := input.shape[input.rank - 1]
	new_trailing := output.shape[output.rank - 1]
	leading      := _leading_count(input)

	for r in 0 ..< leading {
		in_off  := r * trailing + start
		out_off := r * new_trailing
		for i in 0 ..< new_trailing {
			gradient(input)[in_off + i] += gradient(output)[out_off + i]
		}
	}
}

cpu_concat_forward :: proc(op: Operation) {
	output  := op.output
	variant := op.variant.(Concat)
	inputs  := variant.inputs

	leading      := _leading_count(inputs[0])
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

concat_backward :: proc(op: Operation, loc := #caller_location) {
	output := op.output

	variant := op.variant.(Concat)
	inputs  := variant.inputs

	leading      := _leading_count(inputs[0])
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

// Use 8-lane (256-bit / AVX) SIMD when the target supports AVX; otherwise fall
// back to plain scalar loops that LLVM auto-vectorizes for whatever the target
// does support (typically 4-lane SSE). Forcing 256-bit ops on a non-AVX target
// makes LLVM emit 2x128 SSE pairs plus a software fma (mul+add), which ends up
// noticeably slower than letting the compiler auto-vectorize the scalar form.
HAS_AVX :: intrinsics.has_target_feature("avx")

when HAS_AVX {
	SIMD_LANES :: 8
	F32x8      :: #simd[SIMD_LANES]f32

	// sum(a[i] * b[i]) for i in 0..<n. Uses FMA when available.
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

	// y[i] += a * x[i] for i in 0..<n. Standard SAXPY.
	_simd_axpy_f32 :: #force_inline proc "contextless" (y, x: [^]f32, a: f32, n: int) {
		av := F32x8(a)
		i := 0
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
	// Use 4 independent accumulators so LLVM can auto-vectorize the reduction
	// into parallel SSE adds. A single accumulator forces a serial fp dep chain
	// (fp add isn't associative without -ffast-math) and the loop ends up
	// running scalar.
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

cpu_linear_forward :: proc(op: Operation) {
	count := op.variant.(Linear).count

	parallelize(count, count, op, proc(index: int, op: Operation) {
		input, output := op.input, op.output

		variant     := op.variant.(Linear)
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

linear_backward :: proc(op: Operation, loc := #caller_location) {
	count := op.variant.(Linear).count

	parallelize(count, count, op, proc(index: int, op: Operation) {
		input, output := op.input, op.output

		variant     := op.variant.(Linear)
		weight      := variant.weight
		input_size  := variant.input_size
		output_size := variant.output_size

		input_data_ptr  := ([^]f32)(raw_data(data(input)))
		input_grad_ptr  := ([^]f32)(raw_data(gradient(input)))
		output_grad_ptr := ([^]f32)(raw_data(gradient(output)))
		weight_data_ptr := ([^]f32)(raw_data(data(weight)))
		weight_grad_ptr := ([^]f32)(raw_data(gradient(weight)))

		x      := input_data_ptr [index * input_size:]
		dx     := input_grad_ptr [index * input_size:]
		dy     := output_grad_ptr[index * output_size:]

		for o in 0 ..< output_size {
			dout := dy[o]
			if dout == 0 do continue

			w_data := weight_data_ptr[o * input_size:]
			w_grad := weight_grad_ptr[o * input_size:]

			// gradient(weight)[o, :] += x * dout
			_simd_axpy_f32(w_grad, x, dout, input_size)
			// gradient(input)[c, :] += weight[o, :] * dout
			_simd_axpy_f32(dx, w_data, dout, input_size)
		}
	})
}

// Multi-head scaled dot product attention. Input is stacked QKV with shape
// [token_count, 3 * embedding] — Q occupies the first `embedding` columns,
// K the next `embedding`, V the last `embedding`. Output shape is
// [token_count, embedding]. `head_count` stays explicit because it isn't
// derivable from the storage shape.
//
// This is a thin wrapper that decomposes attention into primitive ops:
// slice → reshape → permute → batched_matmul → mul → causal_mask → softmax →
// batched_matmul → permute → reshape. There is no Attention op variant on
// the tape; backward falls out of autograd over the primitives.
@(require_results)
attention :: proc(input: Tensor, head_count: int, causal := true, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 2, "attention requires a 2-D tensor [tokens, 3*embedding]", loc=loc)

	token_count := input.shape[0]
	input_size  := input.shape[1]
	assert(input_size % 3 == 0, "Trailing dim must be divisible by 3 (for Q, K, V)", loc=loc)

	output_size := input_size / 3
	assert(output_size % head_count == 0, "Output size must be divisible by head count", loc=loc)

	head_size := output_size / head_count

	q_flat := slice_trailing(input, 0,               output_size,     loc=loc)
	k_flat := slice_trailing(input, output_size,     output_size * 2, loc=loc)
	v_flat := slice_trailing(input, output_size * 2, output_size * 3, loc=loc)

	q := reshape(q_flat, token_count, head_count, head_size, loc=loc)
	k := reshape(k_flat, token_count, head_count, head_size, loc=loc)
	v := reshape(v_flat, token_count, head_count, head_size, loc=loc)

	q_t := permute(q, {1, 0, 2}, loc=loc)
	k_t := permute(k, {1, 0, 2}, loc=loc)
	v_t := permute(v, {1, 0, 2}, loc=loc)

	k_t_T  := permute(k_t, {0, 2, 1}, loc=loc)
	raw    := batched_matmul(q_t, k_t_T, loc=loc)
	scaled := mul(raw, scalar(1.0 / math.sqrt(f32(head_size)), loc=loc), loc=loc)

	masked := causal ? causal_mask(scaled, loc=loc) : scaled
	attn   := softmax(masked, loc=loc)

	out_per_head := batched_matmul(attn, v_t, loc=loc)
	out          := permute(out_per_head, {1, 0, 2}, loc=loc)
	output       =  reshape(out, token_count, output_size, loc=loc)
	return
}

cpu_rope_forward :: proc(op: Operation) {
	input       := op.input
	output      := op.output
	variant     := op.variant.(Rope)
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

rope_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant     := op.variant.(Rope)
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

cpu_layernorm_forward :: proc(op: Operation) {
	EPSILON :: 1e-5

	input   := op.input
	output  := op.output
	variant := op.variant.(Layernorm)
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

layernorm_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Layernorm)
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

cpu_softmax_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Softmax)
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

softmax_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Softmax)
	size    := variant.size
	count   := variant.count

	for sample in 0 ..< count {
		for i in 0 ..< size {
			input_index := sample * size + i

			gradient_sum: f32

			for j in 0 ..< size {
				output_index := sample * size + j
				if i == j {
					gradient_sum += gradient(output)[output_index] * data(output)[input_index] * (1 - data(output)[input_index])
				} else {
					gradient_sum += gradient(output)[output_index] * (-data(output)[input_index] * data(output)[output_index])
				}
			}

			gradient(input)[input_index] += gradient_sum
		}
	}
}

cpu_log_softmax_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Log_Softmax)
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

log_softmax_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Log_Softmax)
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

cpu_entropy_forward :: proc(op: Operation) {
	probabilities := op.input
	output        := op.output
	variant       := op.variant.(Entropy)
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

entropy_backward :: proc(op: Operation, loc := #caller_location) {
	probabilities, output := op.input, op.output

	variant := op.variant.(Entropy)
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

cpu_mean_squared_error_forward :: proc(op: Operation) {
	predictions := op.input
	output      := op.output
	variant     := op.variant.(Mean_Squared_Error)
	targets     := variant.targets
	count       := variant.count
	sample_size := len(predictions) / count

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

mean_squared_error_backward :: proc(op: Operation, loc := #caller_location) {
	predictions, output := op.input, op.output

	variant := op.variant.(Mean_Squared_Error)
	targets := variant.targets
	count   := variant.count

	sample_size := len(predictions) / count

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

cpu_cross_entropy_forward :: proc(op: Operation) {
	input         := op.input
	output        := op.output
	variant       := op.variant.(Cross_Entropy)
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

cross_entropy_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant       := op.variant.(Cross_Entropy)
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

cpu_relu_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		if data(input)[i] < 0 {
			data(output)[i] = 0
		} else {
			data(output)[i] = data(input)[i]
		}
	}
}

relu_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		if data(input)[i] > 0 {
			gradient(input)[i] += gradient(output)[i]
		}
	}
}

cpu_sigmoid_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		data(output)[i] = 1.0 / (1.0 + math.exp(-data(input)[i]))
	}
}

sigmoid_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		sigmoid_value     := data(output)[i]
		gradient(input)[i] += gradient(output)[i] * sigmoid_value * (1.0 - sigmoid_value)
	}
}

GELU_SCALING_FACTOR :: 0.7978845608028654 // math.sqrt(f32(2) / math.PI)

cpu_gelu_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		x    := data(input)[i]
		cube := 0.044715 * x * x * x

		data(output)[i] = 0.5 * x * (1.0 + math.tanh(GELU_SCALING_FACTOR * (x + cube)))
	}
}

gelu_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
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

cpu_silu_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		sigmoid_val := 1.0 / (1.0 + math.exp(-data(input)[i]))
		data(output)[i] = data(input)[i] * sigmoid_val
	}
}

silu_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		x           := data(input)[i]
		sigmoid_val := 1.0 / (1.0 + math.exp(-x))

		grad := sigmoid_val + x * sigmoid_val * (1.0 - sigmoid_val)

		gradient(input)[i] += gradient(output)[i] * grad
	}
}

cpu_tanh_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		data(output)[i] = math.tanh(data(input)[i])
	}
}

tanh_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		tanh_value        := data(output)[i]
		gradient(input)[i] += gradient(output)[i] * (1.0 - tanh_value * tanh_value)
	}
}

// Batched_Matmul — batched matrix multiply. C[b, i, j] = sum_k A[b, i, k] * B[b, k, j].
// Both inputs are rank-3, output is rank-3. Used to decompose attention
// into primitives without baking matmul-with-batch into a single fused op.
cpu_batched_matmul_forward :: proc(op: Operation) {
	variant := op.variant.(Batched_Matmul)

	parallelize(variant.batch_count * variant.m, variant.batch_count * variant.m, op, proc(idx: int, op: Operation) {
		a       := op.input
		output  := op.output
		variant := op.variant.(Batched_Matmul)
		bt      := variant.b

		m := variant.m
		kk_count := variant.k
		n := variant.n

		bi := idx / m
		i  := idx % m

		a_ptr := ([^]f32)(raw_data(data(a)))
		b_ptr := ([^]f32)(raw_data(data(bt)))
		c_ptr := ([^]f32)(raw_data(data(output)))

		a_row := a_ptr[bi * m * kk_count + i * kk_count:]   // [k]
		c_row := c_ptr[bi * m * n + i * n:]                 // [n]; pre-zeroed

		for kk in 0 ..< kk_count {
			b_row := b_ptr[bi * kk_count * n + kk * n:]      // [n]
			_simd_axpy_f32(c_row, b_row, a_row[kk], n)
		}
	})
}

batched_matmul_backward :: proc(op: Operation, loc := #caller_location) {
	variant := op.variant.(Batched_Matmul)

	// dA[bi, i, k] += sum_j dC[bi, i, j] * B[bi, k, j]
	parallelize(variant.batch_count * variant.m, variant.batch_count * variant.m, op, proc(idx: int, op: Operation) {
		a       := op.input
		output  := op.output
		variant := op.variant.(Batched_Matmul)
		bt      := variant.b

		m := variant.m
		kk_count := variant.k
		n := variant.n

		bi := idx / m
		i  := idx % m

		a_grad_ptr := ([^]f32)(raw_data(gradient(a)))
		b_data_ptr := ([^]f32)(raw_data(data(bt)))
		c_grad_ptr := ([^]f32)(raw_data(gradient(output)))

		dc_row := c_grad_ptr[bi * m * n + i * n:]            // [n]
		da_row := a_grad_ptr[bi * m * kk_count + i * kk_count:] // [k]

		for kk in 0 ..< kk_count {
			b_row := b_data_ptr[bi * kk_count * n + kk * n:]  // [n]
			da_row[kk] += _simd_dot_f32(dc_row, b_row, n)
		}
	})

	// dB[bi, k, j] += sum_i A[bi, i, k] * dC[bi, i, j]
	// Parallel over (bi, k); inner i is serial axpy into dB row k. No race
	// because each worker writes to a unique (bi, k) row.
	parallelize(variant.batch_count * variant.k, variant.batch_count * variant.k, op, proc(idx: int, op: Operation) {
		a       := op.input
		output  := op.output
		variant := op.variant.(Batched_Matmul)
		bt      := variant.b

		m := variant.m
		kk_count := variant.k
		n := variant.n

		bi := idx / kk_count
		kk := idx % kk_count

		a_data_ptr := ([^]f32)(raw_data(data(a)))
		b_grad_ptr := ([^]f32)(raw_data(gradient(bt)))
		c_grad_ptr := ([^]f32)(raw_data(gradient(output)))

		db_row := b_grad_ptr[bi * kk_count * n + kk * n:]    // [n]

		for ii in 0 ..< m {
			a_ik   := a_data_ptr[bi * m * kk_count + ii * kk_count + kk]
			dc_row := c_grad_ptr[bi * m * n + ii * n:]        // [n]
			_simd_axpy_f32(db_row, dc_row, a_ik, n)
		}
	})
}

// Permute — reorder the axes of a rank-3 tensor. `axes[i]` is the source
// axis for the output's axis `i`. Output shape =
// (input.shape[axes[0]], input.shape[axes[1]], input.shape[axes[2]]).
//
// Currently rank-3 only. Generalize to N-D when an op needs it.
cpu_permute_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	axes    := op.variant.(Permute).axes

	in_shape  := [3]int{ input.shape [0], input.shape [1], input.shape [2] }
	out_shape := [3]int{ output.shape[0], output.shape[1], output.shape[2] }
	in_strides := [3]int{ in_shape[1] * in_shape[2], in_shape[2], 1 }

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

permute_backward :: proc(op: Operation, loc := #caller_location) {
	input   := op.input
	output  := op.output
	axes    := op.variant.(Permute).axes

	in_shape  := [3]int{ input.shape [0], input.shape [1], input.shape [2] }
	out_shape := [3]int{ output.shape[0], output.shape[1], output.shape[2] }
	in_strides := [3]int{ in_shape[1] * in_shape[2], in_shape[2], 1 }

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

// Causal_Mask — given a tensor whose trailing two dims are [T, T], replace
// upper-triangle entries (t2 > t1) with -inf, leave the rest untouched.
// Composes with `softmax` to give the "softmax over preceding tokens only"
// semantics that causal attention needs, without baking masking into the
// softmax kernel.
cpu_causal_mask_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := len(input) / block_size

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

causal_mask_backward :: proc(op: Operation, loc := #caller_location) {
	input  := op.input
	output := op.output

	T          := input.shape[input.rank - 1]
	block_size := T * T
	n_blocks   := len(input) / block_size

	for blk in 0 ..< n_blocks {
		offset := blk * block_size
		for t1 in 0 ..< T {
			for t2 in 0 ..= t1 {
				idx := offset + t1 * T + t2
				gradient(input)[idx] += gradient(output)[idx]
			}
			// Masked positions (t2 > t1): gradient blocked.
		}
	}
}
