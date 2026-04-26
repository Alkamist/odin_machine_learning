// This library was designed with the goal of exploring and understanding machine
// learning. Some of the main goals are simplicity and understandability.
//
// The main working unit is the Tensor: a contiguous, row-major slice of data
// and matching gradient buffer, plus an inline shape (rank-N, capped at
// MAX_TENSOR_RANK). Activations live in a global thread-local arena;
// Parameters embed a Tensor (via `using`) plus Adam state and own their own
// buffers, so they pass anywhere a Tensor is expected.
//
// Operations that are performed are stored in a global thread-local buffer,
// so that they can be backpropagated to calculate gradients. The workflow is
// to call clear, do your calculations, call backward to accumulate the
// gradients, update parameters, and repeat.
//
// One downside to doing all of this from scratch is that this library isn't
// particularly optimized. Some calculations are parallelized, but they can definitely
// be improved. I'm not sure if my approach to parallelization is very good.

package machine_learning

import "base:runtime"
import "base:builtin"
import "base:intrinsics"
import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:simd"
import "core:sync"
import "core:thread"

MAX_OPERATIONS :: 4096

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

// Maximum number of dimensions a Tensor can have. Shapes are stored inline as
// a fixed-size array so Tensor stays a value type with no extra allocation.
// Bump this if a future op needs more.
MAX_TENSOR_RANK :: 6

// The main working unit of the library. Always contiguous, row-major. No
// views, no strides, no transpose-aliasing — `reshape` is the only operation
// that shares storage, and it requires the element count to match exactly.
//
// `backend` identifies which Backend owns this tensor's storage. CPU
// backend uses `data` and `gradient` (slices into the context arena, or
// into a Parameter's owned buffers). GPU and other backends store an
// opaque pointer in `storage` and leave the slices empty; their ops cast
// `storage` to a backend-specific struct (e.g. one holding vk.Buffer
// handles).
//
// `count` is the total element count (product of shape dims). Cached so
// `len(t)` doesn't have to scan `shape` and works regardless of backend
// (CPU's slice length is fine; GPU's slice is empty).
Tensor :: struct {
	backend:  ^Backend,
	storage:  rawptr,

	data:     []f32,
	gradient: []f32,

	shape:    [MAX_TENSOR_RANK]int,
	rank:     int,
	count:    int,
}

// Trainable values. A Parameter is a Tensor (its data/gradient are
// user-allocated rather than arena-allocated) plus Adam optimizer state.
// `using tensor` lets a Parameter pass anywhere a Tensor is expected.
Parameter :: struct {
	using tensor: Tensor,

	adam_m: []f32,
	adam_v: []f32,
}

// Op execution interface. Each Backend has one `forward` proc that runs the
// math for a freshly built Operation, and one `backward` proc that runs its
// gradient computation. Both procs are expected to switch on `op.variant`
// internally and call into backend-specific kernels. Adding a new op means
// adding a variant to Operation_Variant and one case to each backend's
// dispatch switch — no new field on Backend.
//
// `data` is opaque backend-specific state (e.g., a GPU device handle).
//
// `alloc` is called by `zeros` and friends to allocate activation storage
// for `n` f32 elements; the backend fills in whatever Tensor fields it
// owns (CPU sets `data`/`gradient`; GPU sets `storage`).
//
// `clear_storage` is called by `clear` to release activation storage in
// bulk (CPU resets the arena; GPU resets the activation pool).
Backend :: struct {
	name: string,
	data: rawptr,

	alloc:         proc(t: ^Tensor, n: int),
	clear_storage: proc(),

	// Per-Context backend-specific state. `context_alloc` is called by
	// `ml.context_create` and the result is stashed on the Context as
	// `backend_data`; `context_free` is called by `ml.context_destroy`.
	// `context_begin` / `context_end` mirror `ml.context_begin` /
	// `ml.context_end` so backends with their own per-thread stack
	// (e.g. the GPU's `Gpu_Context`) can keep it in sync.
	//
	// CPU leaves all four nil. GPU implements them so a single
	// `ml.context_create(..., gpu.backend())` is enough — callers don't
	// have to touch `gpu.context_create` / `gpu.context_scope` directly.
	context_alloc: proc() -> rawptr,
	context_free:  proc(data: rawptr),
	context_begin: proc(data: rawptr),
	context_end:   proc(),

	// Synchronize any pending GPU work and release transient resources.
	// Called by `ml.clear` before activation storage is recycled, and
	// before any host read of tensor data, so callers don't need to
	// open / close batches manually. CPU leaves it nil.
	flush: proc(),

	// Allocate a tensor whose lifetime is NOT tied to the active context's
	// activation pool — `clear_storage` won't free it. CPU uses
	// `context.allocator` (heap), GPU creates a Gpu_Storage that isn't
	// tracked on the gctx allocations list. Pair with `persistent_free`.
	// This is the path for long-lived weights (parameters, inference
	// models, etc.) that need to survive `ml.clear()`.
	persistent_alloc: proc(t: ^Tensor, n: int),
	persistent_free:  proc(t: ^Tensor),

	// Copy `src` into `t`'s data storage. CPU does a slice copy; GPU does
	// a host-visible-stage upload. Used by `tensor` / `scalar` so callers
	// don't need a backend-specific path to seed inputs.
	set_data: proc(t: ^Tensor, src: []f32),

	// Read `t`'s data storage into `dst`. CPU does a slice copy; GPU does
	// a host-visible-stage download.
	get_data: proc(t: ^Tensor, dst: []f32),

	// Allocate the four buffers that back a Parameter (data + gradient +
	// adam_m + adam_v). CPU heap-allocates four []f32 slices; GPU creates
	// four DEVICE_LOCAL buffers and hangs them off the embedded Tensor's
	// storage. Pair with `parameter_free`.
	parameter_alloc: proc(p: ^Parameter, n: int),
	parameter_free:  proc(p: ^Parameter),

	// Apply one Adam(W) step + zero gradient on `p`. CPU does the scalar
	// loop; GPU dispatches `opt_step_adam.spv`.
	parameter_update: proc(opt: Optimizer, p: ^Parameter),

	// Copy data + gradient + adam_m + adam_v from `src` to `dst`. Used by
	// algorithms that snapshot weights (e.g. PPO target networks).
	parameter_copy: proc(dst, src: ^Parameter),

	// Fill `t`'s gradient with 1.0 for every element. Called by `backward`
	// to seed the final op's output gradient before the reverse walk; CPU
	// writes to the slice, GPU dispatches a CmdFillBuffer.
	fill_gradient_with_ones: proc(t: ^Tensor),

	forward:  proc(op: Operation),
	backward: proc(op: Operation),
}

// Per-thread state for tensor allocation, the autograd tape, and the current
// op-execution backend. Multiple Contexts can be active on a thread; the
// current one is tracked via a thread-local stack threaded through
// `previous_ctx`.
//
// A Context's address must stay stable while it is on the stack (the linked
// list stores it by pointer), so always heap-allocate via `context_create`
// or embed it in a long-lived struct. Don't put a Context on a stack frame
// that can return while the Context is still pushed.
Context :: struct {
	backend: ^Backend,

	// Opaque backend-owned state allocated by `Backend.context_alloc` in
	// `context_create`. CPU leaves it nil. GPU stores a `^Gpu_Context`
	// here so the GPU's per-thread context lifecycle is bound to this
	// `ml.Context`'s lifecycle — one `ml.context_create` is enough.
	backend_data: rawptr,

	arena: mem.Arena,

	operation_count: int,
	operations:      [MAX_OPERATIONS]Operation,

	previous_ctx: ^Context,
}

@(thread_local)
_global_odin_context: runtime.Context

// Top of the thread-local context stack. nil means no active context.
@(thread_local)
_current_ctx: ^Context

// CPU backend singleton. `cpu_forward` / `cpu_backward` and the alloc /
// clear hooks are defined further down the file, alongside the op procs
// they dispatch into.
_cpu_backend := Backend{
	name                    = "cpu",
	alloc                   = cpu_alloc,
	clear_storage           = cpu_clear_storage,
	persistent_alloc        = cpu_persistent_alloc,
	persistent_free         = cpu_persistent_free,
	set_data                = cpu_set_data,
	get_data                = cpu_get_data,
	parameter_alloc         = cpu_parameter_alloc,
	parameter_free          = cpu_parameter_free,
	parameter_update        = cpu_parameter_update,
	parameter_copy          = cpu_parameter_copy,
	fill_gradient_with_ones = cpu_fill_gradient_with_ones,
	forward                 = cpu_forward,
	backward                = cpu_backward,
}

// CPU activation alloc: take both data and gradient slices out of the
// active context's arena. Matches the pre-Backend behavior of `zeros`.
cpu_alloc :: proc(t: ^Tensor, n: int) {
	derr: mem.Allocator_Error
	t.data, derr = builtin.make([]f32, n, allocator=arena_allocator())
	fmt.assertf(derr == nil, "Failed to allocate tensor data in arena: %v", derr)

	gerr: mem.Allocator_Error
	t.gradient, gerr = builtin.make([]f32, n, allocator=arena_allocator())
	fmt.assertf(gerr == nil, "Failed to allocate tensor gradient in arena: %v", gerr)
}

// CPU activation reset: free everything the arena holds.
cpu_clear_storage :: proc() {
	mem.arena_free_all(&_current_ctx.arena)
}

// Heap-allocate persistent CPU storage. Mirrors `ml.make`'s alloc path —
// `context.allocator` so the slice lives until explicitly freed.
cpu_persistent_alloc :: proc(t: ^Tensor, n: int) {
	derr: mem.Allocator_Error
	t.data, derr = builtin.make([]f32, n)
	fmt.assertf(derr == nil, "Failed to allocate persistent tensor data: %v", derr)

	gerr: mem.Allocator_Error
	t.gradient, gerr = builtin.make([]f32, n)
	fmt.assertf(gerr == nil, "Failed to allocate persistent tensor gradient: %v", gerr)
}

cpu_persistent_free :: proc(t: ^Tensor) {
	builtin.delete(t.data)
	builtin.delete(t.gradient)
	t.data     = nil
	t.gradient = nil
}

cpu_set_data :: proc(t: ^Tensor, src: []f32) {
	fmt.assertf(builtin.len(src) == builtin.len(t.data), "cpu_set_data size mismatch: src=%v t.data=%v", builtin.len(src), builtin.len(t.data))
	builtin.copy(t.data, src)
}

cpu_get_data :: proc(t: ^Tensor, dst: []f32) {
	fmt.assertf(builtin.len(dst) == builtin.len(t.data), "cpu_get_data size mismatch: dst=%v t.data=%v", builtin.len(dst), builtin.len(t.data))
	builtin.copy(dst, t.data)
}

cpu_parameter_alloc :: proc(p: ^Parameter, n: int) {
	derr1, derr2, derr3, derr4: mem.Allocator_Error
	p.data,     derr1 = builtin.make([]f32, n)
	p.gradient, derr2 = builtin.make([]f32, n)
	p.adam_m,   derr3 = builtin.make([]f32, n)
	p.adam_v,   derr4 = builtin.make([]f32, n)
	fmt.assertf(derr1 == nil && derr2 == nil && derr3 == nil && derr4 == nil,
		"Failed to allocate parameter: %v %v %v %v", derr1, derr2, derr3, derr4)
}

cpu_parameter_free :: proc(p: ^Parameter) {
	builtin.delete(p.data)
	builtin.delete(p.gradient)
	builtin.delete(p.adam_m)
	builtin.delete(p.adam_v)
	p.data     = nil
	p.gradient = nil
	p.adam_m   = nil
	p.adam_v   = nil
}

cpu_parameter_update :: proc(opt: Optimizer, p: ^Parameter) {
	for i in 0 ..< len(p^) {
		grad := p.gradient[i]

		p.adam_m[i] = opt.beta1 * p.adam_m[i] + (1 - opt.beta1) * grad
		p.adam_v[i] = opt.beta2 * p.adam_v[i] + (1 - opt.beta2) * grad * grad

		m_hat := p.adam_m[i] / opt.bias_correction1
		v_hat := p.adam_v[i] / opt.bias_correction2

		p.data[i] = p.data[i] * (1 - opt.learning_rate * opt.weight_decay) - opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)

		p.gradient[i] = 0
	}
}

cpu_parameter_copy :: proc(dst, src: ^Parameter) {
	builtin.copy(dst.data,     src.data)
	builtin.copy(dst.gradient, src.gradient)
	builtin.copy(dst.adam_m,   src.adam_m)
	builtin.copy(dst.adam_v,   src.adam_v)
}

cpu_fill_gradient_with_ones :: proc(t: ^Tensor) {
	for i in 0 ..< builtin.len(t.gradient) {
		t.gradient[i] = 1
	}
}

// Public accessor for the CPU backend, useful for explicitly creating a
// Context on the CPU (the default if you don't pass a backend).
@(require_results)
cpu_backend :: #force_inline proc() -> ^Backend {
	return &_cpu_backend
}

// Heap-allocate and initialize a Context. Pair with `context_destroy`. The
// Context's backend is the CPU backend by default; pass an explicit
// `backend` (e.g. from `gpu.backend()`) to run ops on a different one.
@(require_results)
context_create :: proc(size: int, backend: ^Backend = nil, allocator := context.allocator, loc := #caller_location) -> ^Context {
	ctx, cerr := builtin.new(Context, allocator=allocator, loc=loc)
	assert(cerr == nil, "Failed to allocate Context", loc=loc)

	data, derr := builtin.make([]byte, size, allocator=allocator, loc=loc)
	assert(derr == nil, "Failed to allocate context arena data", loc=loc)
	mem.arena_init(&ctx.arena, data)

	ctx.backend = backend != nil ? backend : &_cpu_backend
	if ctx.backend.context_alloc != nil {
		ctx.backend_data = ctx.backend.context_alloc()
	}

	return ctx
}

// Destroy and free a Context allocated by `context_create`. The Context must
// not be on the active stack when destroyed.
context_destroy :: proc(ctx: ^Context, allocator := context.allocator, loc := #caller_location) {
	assert(_current_ctx != ctx, "context_destroy called on the active context", loc=loc)
	if ctx.backend.context_free != nil && ctx.backend_data != nil {
		ctx.backend.context_free(ctx.backend_data)
		ctx.backend_data = nil
	}
	if ctx.arena.data != nil {
		builtin.delete(ctx.arena.data, allocator=allocator, loc=loc)
	}
	builtin.free(ctx, allocator=allocator, loc=loc)
}

// Push a user-owned context onto the thread-local stack.
context_begin :: proc(ctx: ^Context) {
	ctx.previous_ctx = _current_ctx
	_current_ctx = ctx
	if ctx.backend.context_begin != nil {
		ctx.backend.context_begin(ctx.backend_data)
	}
}

// Pop the current context off the stack.
context_end :: proc() {
	assert(_current_ctx != nil, "context_end called with no active context")
	if _current_ctx.backend.context_end != nil {
		_current_ctx.backend.context_end()
	}
	_current_ctx = _current_ctx.previous_ctx
}

// Scoped push: paired with `defer`-style pop on scope exit via `deferred_none`.
@(deferred_none=context_end)
context_scope :: proc(ctx: ^Context) {
	context_begin(ctx)
}

// Get the active thread-local context. Asserts one is active.
@(require_results)
current_context :: #force_inline proc(loc := #caller_location) -> ^Context {
	assert(_current_ctx != nil, "no active context — call init or context_begin", loc=loc)
	return _current_ctx
}

// Clear the active context's activation storage and operation tape.
// Activation storage release is delegated to the backend (CPU resets the
// arena; GPU would reset the activation pool).
clear :: proc(loc := #caller_location) {
	assert(_current_ctx != nil && _current_ctx.backend != nil, "Did you forget to call context_create / context_scope?", loc=loc)
	if _current_ctx.backend.flush != nil {
		_current_ctx.backend.flush()
	}
	_current_ctx.backend.clear_storage()
	_current_ctx.operation_count = 0
}

// Force the active backend to finish all queued work. CPU is synchronous
// already so this is a no-op; GPU submits + waits any open command-buffer
// batch. `ml.clear` calls this internally before recycling activations,
// and `Backend.get_data` calls it before reading host memory, so most
// callers don't need it. Use it explicitly when you need to time a
// single step on GPU without rolling its completion into the next
// iteration's `ml.clear`.
sync :: proc() {
	if _current_ctx == nil || _current_ctx.backend.flush == nil { return }
	_current_ctx.backend.flush()
}

// Get the active context's arena allocator.
arena_allocator :: proc() -> mem.Allocator {
	return mem.arena_allocator(&_current_ctx.arena)
}

// Total element count of a tensor (product of all dims).
@(require_results)
len :: #force_inline proc(t: Tensor) -> int {
	return t.count
}

// Total element count implied by a shape.
@(require_results)
shape_element_count :: proc(shape: []int) -> int {
	n := 1
	for d in shape {
		n *= d
	}
	return n
}

// Allocate a tensor in the global arena initialized with zeros. Variadic
// `shape` gives the dimensions in row-major order (outermost first); a
// single int produces a 1-D tensor.
@(require_results)
zeros :: proc(shape: ..int, loc := #caller_location) -> (t: Tensor) {
	assert(_current_ctx != nil && _current_ctx.backend != nil, "Did you forget to call context_create / context_scope?", loc=loc)
	assert(builtin.len(shape) > 0, "Tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Tensor rank exceeds MAX_TENSOR_RANK", loc=loc)

	n := shape_element_count(shape)
	assert(n > 0, "Tensor element count must be positive", loc=loc)

	t.backend = _current_ctx.backend
	t.backend.alloc(&t, n)

	t.count = n
	t.rank  = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "Tensor dimension must be positive", loc=loc)
		t.shape[i] = d
	}
	return
}

// Allocate a tensor whose storage survives `ml.clear()`. Use this for
// long-lived weights / inference models that are reused across many
// activation cycles. Pair with `persistent_destroy`.
//
// Counterpart of `zeros` for the persistent lifetime path. Routes
// through `Backend.persistent_alloc` so it works on either backend.
@(require_results)
persistent_zeros :: proc(shape: ..int, loc := #caller_location) -> (t: Tensor) {
	assert(_current_ctx != nil && _current_ctx.backend != nil, "Did you forget to call context_create / context_scope?", loc=loc)
	assert(builtin.len(shape) > 0, "Tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Tensor rank exceeds MAX_TENSOR_RANK", loc=loc)

	n := shape_element_count(shape)
	assert(n > 0, "Tensor element count must be positive", loc=loc)

	t.backend = _current_ctx.backend
	t.backend.persistent_alloc(&t, n)

	t.count = n
	t.rank  = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "Tensor dimension must be positive", loc=loc)
		t.shape[i] = d
	}
	return
}

// Free a tensor allocated by `persistent_zeros`. Safe to call regardless
// of the active context's backend, as long as `t.backend` matches.
persistent_destroy :: proc(t: Tensor) {
	if t.backend == nil { return }
	tt := t
	t.backend.persistent_free(&tt)
}

// Allocate a same-shape zeroed tensor.
@(require_results)
zeros_like :: proc(src: Tensor, loc := #caller_location) -> Tensor {
	shape := src.shape
	return zeros(..shape[:src.rank], loc=loc)
}

// Copy data to the global arena as a 1-D tensor.
@(require_results)
tensor :: proc(data: []f32, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(data) > 0, "Length must be at least 1", loc=loc)

	t = zeros(builtin.len(data), loc=loc)
	t.backend.set_data(&t, data)
	return
}

// Single-value 1-D tensor in the global arena.
@(require_results)
scalar :: proc(value: f32, loc := #caller_location) -> (t: Tensor) {
	t = zeros(1, loc=loc)
	src := [1]f32{value}
	t.backend.set_data(&t, src[:])
	return
}

// Allocate a zeroed tensor whose shape is `src.shape` with the trailing dim
// dropped. If src is rank 1, the output is a 1-D tensor of length 1.
@(require_results)
_zeros_drop_last :: proc(src: Tensor, loc := #caller_location) -> Tensor {
	if src.rank <= 1 {
		return zeros(1, loc=loc)
	}
	shape := src.shape
	return zeros(..shape[:src.rank - 1], loc=loc)
}

// Allocate a zeroed tensor whose shape is `src.shape` with the trailing dim
// replaced by `new_trailing`.
@(require_results)
_zeros_replace_trailing :: proc(src: Tensor, new_trailing: int, loc := #caller_location) -> Tensor {
	new_shape: [MAX_TENSOR_RANK]int = src.shape
	new_shape[src.rank - 1] = new_trailing
	return zeros(..new_shape[:src.rank], loc=loc)
}

// Product of leading dimensions (rank-1 dims). For rank 1, returns 1.
@(require_results)
_leading_count :: proc(t: Tensor) -> int {
	n := 1
	for i in 0 ..< t.rank - 1 {
		n *= t.shape[i]
	}
	return n
}

// Reinterpret a Tensor under a new shape. Pure header change — shares storage
// with src. The new shape's element count must equal the source's.
@(require_results)
reshape :: proc(src: Tensor, shape: ..int, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(shape) > 0, "Tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Tensor rank exceeds MAX_TENSOR_RANK", loc=loc)
	assert(shape_element_count(shape) == len(src), "Reshape element count mismatch", loc=loc)

	t.backend  = src.backend
	t.storage  = src.storage
	t.data     = src.data
	t.gradient = src.gradient
	t.count    = src.count
	t.rank     = builtin.len(shape)
	for d, i in shape {
		t.shape[i] = d
	}
	return
}

// Allocate a parameter initialized with zeros. Variadic `shape` gives the
// dimensions in row-major order (outermost first); a single int produces a
// 1-D parameter. Because `Parameter` embeds `Tensor` via `using`, a Parameter
// passes anywhere a Tensor is expected — no explicit conversion needed.
@(require_results)
make :: proc(shape: ..int, allocator := context.allocator, loc := #caller_location) -> (parameter: Parameter, err: mem.Allocator_Error) #optional_allocator_error {
	assert(builtin.len(shape) > 0, "Parameter must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Parameter rank exceeds MAX_TENSOR_RANK", loc=loc)

	n := shape_element_count(shape)
	assert(n > 0, "Parameter element count must be positive", loc=loc)

	parameter.backend = _current_ctx != nil ? _current_ctx.backend : &_cpu_backend
	parameter.backend.parameter_alloc(&parameter, n)

	parameter.count = n
	parameter.rank  = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "Parameter dimension must be positive", loc=loc)
		parameter.shape[i] = d
	}

	return parameter, nil
}

// Destroy an allocated parameter.
destroy :: proc(parameter: Parameter, loc := #caller_location) {
	if parameter.backend == nil { return }
	p := parameter
	parameter.backend.parameter_free(&p)
}

// Copy parameter data from src to dst.
copy :: proc(dst, src: Parameter, loc := #caller_location) {
	assert(len(dst) == len(src), "Parameter lengths need to be equal", loc=loc)
	assert(dst.backend == src.backend, "Parameter copy across backends not supported", loc=loc)
	d, s := dst, src
	dst.backend.parameter_copy(&d, &s)
}

// Fill tensor data with normally distributed random numbers. Works on
// either backend — for non-CPU backends we fill into a temp_allocator
// host buffer and upload via `Backend.set_data`.
fill_normal :: proc(t: Tensor, mean, std: f32) {
	if t.backend == nil || t.backend == &_cpu_backend {
		for &v in t.data {
			v = rand.float32_normal(mean, std)
		}
		return
	}
	n := len(t)
	buf := builtin.make([]f32, n, allocator=context.temp_allocator)
	for i in 0 ..< n {
		buf[i] = rand.float32_normal(mean, std)
	}
	tt := t
	t.backend.set_data(&tt, buf)
}

// Fill tensor data with a single value. Backend-aware.
fill_value :: proc(t: Tensor, value: f32) {
	if t.backend == nil || t.backend == &_cpu_backend {
		for &v in t.data {
			v = value
		}
		return
	}
	n := len(t)
	buf := builtin.make([]f32, n, allocator=context.temp_allocator)
	for i in 0 ..< n {
		buf[i] = value
	}
	tt := t
	t.backend.set_data(&tt, buf)
}

// Perform He initialization.
he_initialization :: proc(t: Tensor, input_features: int) {
	fill_normal(t, 0, math.sqrt(2 / f32(input_features)))
}

// Perform Xavier/Glorot initialization.
xavier_initialization :: proc(t: Tensor, input_features, output_features: int) {
	fill_normal(t, 0, math.sqrt(2 / f32(input_features + output_features)))
}

Optimizer :: struct {
	iteration:      u64,
	period_counter: int,

	learning_rate: f32,
	beta1:         f32,
	beta2:         f32,
	epsilon:       f32,
	weight_decay:  f32,

	bias_correction1: f32,
	bias_correction2: f32,
}

// Check to see if an optimizer step should occur based on the period,
// then set the optimizer hyperparameters and increment the iteration.
// This is meant to be used in an if statement with parameter updates
// inside the scope.
@(require_results)
optimize :: proc(
	opt:           ^Optimizer,
	period:        int = 128,
	learning_rate: f32 = 0.001,
	beta1:         f32 = 0.9,
	beta2:         f32 = 0.999,
	epsilon:       f32 = 1e-8,
	weight_decay:  f32 = 0,
) -> bool {
	opt.period_counter += 1
	if opt.period_counter < period {
		return false
	}
	opt.period_counter = 0

	opt.iteration += 1

	opt.learning_rate = learning_rate
	opt.beta1         = beta1
	opt.beta2         = beta2
	opt.epsilon       = epsilon
	opt.weight_decay  = weight_decay

	opt.bias_correction1 = 1 - math.pow(opt.beta1, f32(opt.iteration))
	opt.bias_correction2 = 1 - math.pow(opt.beta2, f32(opt.iteration))

	return true
}

// Update a parameter's data and zero its gradients.
// This is meant to be called inside the scope of optimize.
update :: proc(opt: Optimizer, parameter: Parameter) {
	p := parameter
	parameter.backend.parameter_update(opt, &p)
}

Operation_Variant :: union {
	Add,
	Sub,
	Mul,
	Div,
	Exp,
	Clamp,
	Min,
	Max,
	Mean,
	Transpose,
	Select,
	Slice,
	Slice_Trailing,
	Concat,
	Linear,
	Rope,
	Layernorm,
	Softmax,
	Entropy,
	Log_Softmax,
	Mean_Squared_Error,
	Cross_Entropy,
	Relu,
	Sigmoid,
	Gelu,
	Silu,
	Tanh,
	Batched_Matmul,
	Permute,
	Causal_Mask,
}

Operation :: struct {
	input:   Tensor,
	output:  Tensor,
	variant: Operation_Variant,
}

// Append an operation to the global context for backpropagation.
append_operation :: proc(op: Operation, loc := #caller_location) {
	assert(_current_ctx.operation_count < MAX_OPERATIONS, "Maximum operations exceeded, did you forget to call clear?", loc=loc)
	_current_ctx.operations[_current_ctx.operation_count] = op
	_current_ctx.operation_count += 1
}

// Iterate backwards through all operations and accumulate gradients through
// tensors. Only the final operation's output gradient is initialized to 1,
// which means gradients flow backward from the final operation. Gradients
// won't flow properly if you have multiple final operations. I'm not sure
// of the best way to solve that problem.
backward :: proc(loc := #caller_location) {
	if _current_ctx == nil || _current_ctx.operation_count <= 0 {
		return
	}

	backend := _current_ctx.backend

	// Seed the final op's output gradient with all 1.0s. Backend-specific
	// because the gradient lives on host memory (CPU) or in a Vulkan buffer
	// (GPU).
	final_op := &_current_ctx.operations[_current_ctx.operation_count - 1]
	backend.fill_gradient_with_ones(&final_op.output)

	for i := _current_ctx.operation_count - 1; i >= 0; i -= 1 {
		backend.backward(_current_ctx.operations[i])
	}
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

Add :: struct {
	b:      Tensor,
	stride: int,
}

// Add two tensors, b is broadcasted into a if necessary.
@(require_results)
add :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Add{
			b      = b,
			stride = len(a) / len(b),
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			output.data[o] = a.data[o] + b.data[j]
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
			a.gradient[o] += output.gradient[o]
			b.gradient[j] += output.gradient[o]
		}
	}
}

Sub :: struct {
	b:      Tensor,
	stride: int,
}

// Subtract two tensors, b is broadcasted into a if necessary.
@(require_results)
sub :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Sub{
			b      = b,
			stride = len(a) / len(b),
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			output.data[o] = a.data[o] - b.data[j]
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
			a.gradient[o] += output.gradient[o]
			b.gradient[j] -= output.gradient[o]
		}
	}
}

Mul :: struct {
	b:      Tensor,
	stride: int,
}

// Multiply two tensors, b is broadcasted into a if necessary.
@(require_results)
mul :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Mul{
			b      = b,
			stride = len(a) / len(b),
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			output.data[o] = a.data[o] * b.data[j]
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
			a.gradient[o] += output.gradient[o] * b.data[j]
			b.gradient[j] += output.gradient[o] * a.data[o]
		}
	}
}

Div :: struct {
	b:      Tensor,
	stride: int,
}

// Divide two tensors, b is broadcasted into a if necessary.
@(require_results)
div :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Div{
			b      = b,
			stride = len(a) / len(b),
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			output.data[o] = a.data[o] / b.data[j]
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
			a.gradient[o] += output.gradient[o] / b.data[j]
			b.gradient[j] += output.gradient[o] * (-a.data[o] / (b.data[j] * b.data[j]))
		}
	}
}

Exp :: struct {
}

@(require_results)
exp :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Exp{},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_exp_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		output.data[i] = math.exp(input.data[i])
	}
}

exp_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		input.gradient[i] += output.data[i] * output.gradient[i]
	}
}

Clamp :: struct {
	min_val: f32,
	max_val: f32,
}

@(require_results)
clamp :: proc(input: Tensor, min_val, max_val: f32, loc := #caller_location) -> (output: Tensor) {
	assert(min_val <= max_val, "Requires min_val <= max_val", loc=loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Clamp{
			min_val = min_val,
			max_val = max_val,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_clamp_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< len(input) {
		output.data[i] = math.clamp(input.data[i], min_val, max_val)
	}
}

clamp_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Clamp)
	min_val := variant.min_val
	max_val := variant.max_val

	for i in 0 ..< len(input) {
		if input.data[i] >= min_val && input.data[i] <= max_val {
			input.gradient[i] += output.gradient[i]
		}
	}
}

Min :: struct {
	b: Tensor,
}

@(require_results)
min :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(len(a) == len(b), "Requires inputs of equal length", loc=loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Min{
			b = b,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_min_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Min)
	b       := variant.b

	for i in 0 ..< len(a) {
		output.data[i] = math.min(a.data[i], b.data[i])
	}
}

min_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Min)
	b       := variant.b

	for i in 0 ..< len(a) {
		if a.data[i] <= b.data[i] {
			a.gradient[i] += output.gradient[i]
		} else {
			b.gradient[i] += output.gradient[i]
		}
	}
}

Max :: struct {
	b: Tensor,
}

@(require_results)
max :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(len(a) == len(b), "Requires inputs of equal length", loc=loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Max{
			b = b,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_max_forward :: proc(op: Operation) {
	a       := op.input
	output  := op.output
	variant := op.variant.(Max)
	b       := variant.b

	for i in 0 ..< len(a) {
		output.data[i] = math.max(a.data[i], b.data[i])
	}
}

max_backward :: proc(op: Operation, loc := #caller_location) {
	a, output := op.input, op.output

	variant := op.variant.(Max)
	b       := variant.b

	for i in 0 ..< len(a) {
		if a.data[i] >= b.data[i] {
			a.gradient[i] += output.gradient[i]
		} else {
			b.gradient[i] += output.gradient[i]
		}
	}
}

Mean :: struct {
	size:  int,
	count: int,
}

@(require_results)
mean :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	count := _leading_count(input)
	size  := len(input) / count
	output = _zeros_drop_last(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Mean{
			size  = size,
			count = count,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			sum += input.data[index]
		}
		output.data[sample] = sum / f32(size)
	}
}

mean_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Mean)
	size    := variant.size
	count   := variant.count

	for sample in 0 ..< count {
		gradient_per_element := output.gradient[sample] / f32(size)

		for i in 0 ..< size {
			input_index := sample * size + i
			input.gradient[input_index] += gradient_per_element
		}
	}
}

Transpose :: struct {
	rows: int,
}

// Transpose a 2-D tensor: [rows, cols] -> [cols, rows].
@(require_results)
transpose :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 2, "transpose requires a 2-D tensor", loc=loc)

	rows    := input.shape[0]
	columns := input.shape[1]

	output = zeros(columns, rows, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Transpose{
			rows = rows,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_transpose_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Transpose)
	rows    := variant.rows
	columns := len(input) / rows

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			output.data[j * rows + i] = input.data[i * columns + j]
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
			input.gradient[i * columns + j] += output.gradient[j * rows + i]
		}
	}
}

Select :: struct {
	indices: []int,
	size:    int,
}

// Select rows from a tensor by index. Input shape [N, ...rest]; output shape
// [len(indices), ...rest]. The "row size" is the product of trailing dims.
@(require_results)
select :: proc(input: Tensor, indices: []int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "select input must have rank >= 1", loc=loc)

	size := 1
	for i in 1 ..< input.rank {
		size *= input.shape[i]
	}

	indices_copy := builtin.make([]int, builtin.len(indices), allocator=arena_allocator())
	for i in 0 ..< builtin.len(indices) {
		indices_copy[i] = indices[i]
	}

	out_shape: [MAX_TENSOR_RANK]int = input.shape
	out_shape[0] = builtin.len(indices)
	output = zeros(..out_shape[:input.rank], loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Select{
			indices  = indices_copy,
			size     = size,
		}
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_select_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Select)
	indices := variant.indices
	size    := variant.size

	for i in 0 ..< builtin.len(indices) {
		for j in 0 ..< size {
			output.data[i * size + j] = input.data[indices[i] * size + j]
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
			weight.gradient[indices[i] * size + j] += output.gradient[i * size + j]
		}
	}
}

Slice :: struct {
	start: int,
	end:   int,
}

// Slice an input tensor. Copies the data.
@(require_results)
slice :: proc(input: Tensor, start, end: int, loc := #caller_location) -> (output: Tensor) {
	fmt.assertf(start >= 0 && end <= len(input) && start <= end, "Slice indices out of bounds %v:%v", start, end, loc=loc)

	output = zeros(end - start, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Slice{
			start = start,
			end   = end,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_slice_forward :: proc(op: Operation) {
	input   := op.input
	output  := op.output
	variant := op.variant.(Slice)
	start   := variant.start
	end     := variant.end

	builtin.copy(output.data, input.data[start:end])
}

slice_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Slice)
	start   := variant.start

	for i in 0 ..< len(output) {
		input.gradient[start + i] += output.gradient[i]
	}
}

Slice_Trailing :: struct {
	start, end: int,
}

// Slice along the trailing dim, preserving rank. Output shape =
// input.shape with the trailing dim replaced by (end - start).
@(require_results)
slice_trailing :: proc(input: Tensor, start, end: int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "slice_trailing input must have rank >= 1", loc=loc)
	trailing := input.shape[input.rank - 1]
	fmt.assertf(start >= 0 && end <= trailing && start <= end, "slice_trailing indices out of bounds %v:%v (trailing=%v)", start, end, trailing, loc=loc)

	new_trailing := end - start
	output = _zeros_replace_trailing(input, new_trailing, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Slice_Trailing{
			start = start,
			end   = end,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			output.data[out_off + i] = input.data[in_off + i]
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
			input.gradient[in_off + i] += output.gradient[out_off + i]
		}
	}
}

Concat :: struct {
	inputs: []Tensor,
}

// Concatenate multiple tensors along the trailing dim. All inputs must share
// rank and match in every dim except the trailing one. Output shape =
// inputs[0].shape with the trailing dim replaced by the sum of trailings.
@(require_results)
concat :: proc(inputs: ..Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(builtin.len(inputs) > 0, "Requires at least one input", loc=loc)

	first := inputs[0]
	trailing_sum := first.shape[first.rank - 1]
	for i in 1 ..< builtin.len(inputs) {
		assert(inputs[i].rank == first.rank, "All concat inputs must have the same rank", loc=loc)
		for d in 0 ..< first.rank - 1 {
			assert(inputs[i].shape[d] == first.shape[d], "All concat inputs must match in non-trailing dims", loc=loc)
		}
		trailing_sum += inputs[i].shape[inputs[i].rank - 1]
	}

	inputs_copy := builtin.make([]Tensor, builtin.len(inputs), allocator=arena_allocator())
	for input, i in inputs {
		inputs_copy[i] = input
	}

	output = _zeros_replace_trailing(first, trailing_sum, loc=loc)

	op := Operation{
		input   = {},
		output  = output,
		variant = Concat{
			inputs = inputs_copy,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
				output.data[out_off + i] = input.data[in_off + i]
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
				input.gradient[in_off + i] += output.gradient[out_off + i]
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

Linear :: struct {
	weight:      Tensor,
	input_size:  int,
	output_size: int,
	count:       int,
}

// Linear transformation. weight is [output_size, input_size]; input has
// trailing dim equal to input_size. Output shape = input.shape with the
// trailing dim replaced by output_size. `count` (the number of input rows
// to project) is the product of input's leading dims.
@(require_results)
linear :: proc(input, weight: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank  >= 1, "Linear input must have rank >= 1",  loc=loc)
	assert(weight.rank == 2, "Linear weight must be a 2-D tensor [output_size, input_size]", loc=loc)

	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	assert(input.shape[input.rank - 1] == input_size, "Input trailing dim must equal weight's input dim", loc=loc)

	count := _leading_count(input)
	output = _zeros_replace_trailing(input, output_size, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Linear{
			weight      = weight,
			input_size  = input_size,
			output_size = output_size,
			count       = count,
		}
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_linear_forward :: proc(op: Operation) {
	count := op.variant.(Linear).count

	parallelize(count, count, op, proc(index: int, op: Operation) {
		input, output := op.input, op.output

		variant     := op.variant.(Linear)
		weight      := variant.weight
		input_size  := variant.input_size
		output_size := variant.output_size

		input_ptr  := ([^]f32)(raw_data(input.data))
		output_ptr := ([^]f32)(raw_data(output.data))
		weight_ptr := ([^]f32)(raw_data(weight.data))

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

		input_data_ptr  := ([^]f32)(raw_data(input.data))
		input_grad_ptr  := ([^]f32)(raw_data(input.gradient))
		output_grad_ptr := ([^]f32)(raw_data(output.gradient))
		weight_data_ptr := ([^]f32)(raw_data(weight.data))
		weight_grad_ptr := ([^]f32)(raw_data(weight.gradient))

		x      := input_data_ptr [index * input_size:]
		dx     := input_grad_ptr [index * input_size:]
		dy     := output_grad_ptr[index * output_size:]

		for o in 0 ..< output_size {
			dout := dy[o]
			if dout == 0 do continue

			w_data := weight_data_ptr[o * input_size:]
			w_grad := weight_grad_ptr[o * input_size:]

			// weight.gradient[o, :] += x * dout
			_simd_axpy_f32(w_grad, x, dout, input_size)
			// input.gradient[c, :] += weight[o, :] * dout
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

Rope :: struct {
	token_count: int,
	head_count:  int,
	head_size:   int,
	base:        f32,

	cos_cache: Tensor,
	sin_cache: Tensor,
}

// Rotary position embedding. Preserves input shape; reads `token_count` from
// input.shape[0]. `head_count` stays explicit.
@(require_results)
rope :: proc(input: Tensor, head_count: int, base: f32 = 10000, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 2, "rope requires rank >= 2", loc=loc)

	token_count := input.shape[0]
	input_size  := input.shape[input.rank - 1]
	assert(input_size % head_count == 0, "Trailing dim must be divisible by head count", loc=loc)

	head_size := input_size / head_count
	assert(head_size % 2 == 0, "Head size must be even", loc=loc)

	output = zeros_like(input, loc=loc)

	cos_cache := zeros(token_count * head_size / 2, loc=loc)
	sin_cache := zeros(token_count * head_size / 2, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Rope{
			token_count = token_count,
			head_count  = head_count,
			head_size   = head_size,
			base        = base,
			cos_cache   = cos_cache,
			sin_cache   = sin_cache,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

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
			cos_cache.data[cache_idx] = math.cos(theta)
			sin_cache.data[cache_idx] = math.sin(theta)
		}
	}

	for t in 0 ..< token_count {
		for h in 0 ..< head_count {
			head_offset := t * head_count * head_size + h * head_size

			for i in 0 ..< head_size / 2 {
				cache_idx := t * (head_size / 2) + i
				cos_val := cos_cache.data[cache_idx]
				sin_val := sin_cache.data[cache_idx]

				x := input.data[head_offset + i * 2]
				y := input.data[head_offset + i * 2 + 1]

				output.data[head_offset + i * 2]     = x * cos_val - y * sin_val
				output.data[head_offset + i * 2 + 1] = x * sin_val + y * cos_val
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
				cos_val := cos_cache.data[cache_idx]
				sin_val := sin_cache.data[cache_idx]

				grad_x := output.gradient[head_offset + i * 2]
				grad_y := output.gradient[head_offset + i * 2 + 1]

				input.gradient[head_offset + i * 2]     +=  grad_x * cos_val + grad_y * sin_val
				input.gradient[head_offset + i * 2 + 1] += -grad_x * sin_val + grad_y * cos_val
			}
		}
	}
}

Layernorm :: struct {
	weight: Tensor,
	mean:   Tensor,
	rstd:   Tensor,
	count:  int,
	size:   int,
}

@(require_results)
layernorm :: proc(input, weight: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(weight.rank == 1, "layernorm weight must be 1-D", loc=loc)
	assert(weight.shape[0] == input.shape[input.rank - 1], "layernorm weight length must equal input's trailing dim", loc=loc)

	count := _leading_count(input)
	size  := input.shape[input.rank - 1]

	mean := zeros(count, loc=loc)
	rstd := zeros(count, loc=loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Layernorm{
			weight = weight,
			mean   = mean,
			rstd   = rstd,
			count  = count,
			size   = size,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			m += input.data[offset + i]
		}
		m /= f32(size)

		v: f32
		for i in 0 ..< size {
			x_shift := input.data[offset + i] - m
			v += x_shift * x_shift
		}
		v /= f32(size)

		s: f32 = 1.0 / math.sqrt(v + EPSILON)
		for i in 0 ..< size {
			n := (s * (input.data[offset + i] - m))
			o := n * weight.data[i]
			output.data[offset + i] = o
		}

		mean.data[c] = m
		rstd.data[c] = s
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
			norm  := (input.data[offset + i] - mean.data[c]) * rstd.data[c]
			dnorm := weight.data[i] * output.gradient[offset + i]
			dnorm_mean      += dnorm
			dnorm_norm_mean += dnorm * norm
		}
		dnorm_mean      /= f32(size)
		dnorm_norm_mean /= f32(size)

		for i in 0 ..< size {
			norm  := (input.data[offset + i] - mean.data[c]) * rstd.data[c]
			dnorm := weight.data[i] * output.gradient[offset + i]

			weight.gradient[i] += norm * output.gradient[offset + i]

			gradient: f32
			gradient += dnorm
			gradient -= dnorm_mean
			gradient -= norm * dnorm_norm_mean
			gradient *= rstd.data[c]

			input.gradient[offset + i] += gradient
		}
	}
}

Softmax :: struct {
	size:  int,
	count: int,
}

@(require_results)
softmax :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	count := _leading_count(input)
	size  := input.shape[input.rank - 1]

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Softmax{
			size  = size,
			count = count,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			max_value = math.max(max_value, input.data[index])
		}

		// Compute exp values and sum.
		sum: f32
		for i in 0 ..< size {
			index := sample * size + i
			exp_val := math.exp(input.data[index] - max_value)
			output.data[index] = exp_val
			sum += exp_val
		}

		// Normalize to get probabilities.
		for i in 0 ..< size {
			index := sample * size + i
			output.data[index] /= sum
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
					gradient_sum += output.gradient[output_index] * output.data[input_index] * (1 - output.data[input_index])
				} else {
					gradient_sum += output.gradient[output_index] * (-output.data[input_index] * output.data[output_index])
				}
			}

			input.gradient[input_index] += gradient_sum
		}
	}
}

Log_Softmax :: struct {
	size:  int,
	count: int,
}

@(require_results)
log_softmax :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	count := _leading_count(input)
	size  := input.shape[input.rank - 1]

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Log_Softmax{
			size  = size,
			count = count,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			max_value = math.max(max_value, input.data[index])
		}

		// Compute log_sum_exp for normalization.
		log_sum_exp: f32
		for i in 0 ..< size {
			index := sample * size + i
			log_sum_exp += math.exp(input.data[index] - max_value)
		}
		log_sum_exp = math.ln(log_sum_exp) + max_value

		// Compute log probabilities.
		for i in 0 ..< size {
			index := sample * size + i
			output.data[index] = input.data[index] - log_sum_exp
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
			gradient_sum += output.gradient[output_index]
		}

		for i in 0 ..< size {
			index := sample * size + i
			input.gradient[index] += output.gradient[index] - math.exp(output.data[index]) * gradient_sum
		}
	}
}

Entropy :: struct {
	size:  int,
	count: int,
}

@(require_results)
entropy :: proc(probabilities: Tensor, loc := #caller_location) -> (output: Tensor) {
	count := _leading_count(probabilities)
	size  := probabilities.shape[probabilities.rank - 1]

	output = _zeros_drop_last(probabilities, loc=loc)

	op := Operation{
		input   = probabilities,
		output  = output,
		variant = Entropy{
			size  = size,
			count = count,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			p      := probabilities.data[index]
			p_safe := math.max(p, 1e-8)

			entropy_value -= p * math.ln(p_safe)
		}

		output.data[sample] = entropy_value
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
			p      := probabilities.data[index]
			p_safe := math.max(p, 1e-8)

			gradient := -(math.ln(p_safe) + 1.0)

			probabilities.gradient[index] += output.gradient[sample] * gradient
		}
	}
}

Mean_Squared_Error :: struct {
	targets: Tensor,
	count:   int,
}

@(require_results)
mean_squared_error :: proc(predictions, targets: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(len(predictions) == len(targets), "Predictions and targets must have same length", loc=loc)

	count := _leading_count(predictions)

	output = _zeros_drop_last(predictions, loc=loc)

	op := Operation{
		input   = predictions,
		output  = output,
		variant = Mean_Squared_Error{
			targets = targets,
			count   = count,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			diff  := predictions.data[index] - targets.data[index]
			sum_squared_error += diff * diff
		}

		output.data[sample] = sum_squared_error / f32(sample_size)
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

		upstream_gradient := output.gradient[sample]

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			gradient := scale * (predictions.data[index] - targets.data[index])
			predictions.gradient[index] += gradient * upstream_gradient
		}
	}
}

Cross_Entropy :: struct {
	probabilities: Tensor,
	targets:       []int,
	class_size:    int,
}

// Cross entropy performs softmax internally, so it expects the input
// to not already be softmaxed.
@(require_results)
cross_entropy :: proc(input: Tensor, targets: []int, loc := #caller_location) -> (output: Tensor) {
	sample_count := builtin.len(targets)
	assert(sample_count > 0, "Must have at least one target", loc=loc)
	assert(input.rank >= 1, "cross_entropy input must have rank >= 1", loc=loc)
	assert(_leading_count(input) == sample_count, "Input leading-dim product must equal number of targets", loc=loc)

	class_size := input.shape[input.rank - 1]

	targets_copy := builtin.make([]int, sample_count, allocator=arena_allocator())
	for target, i in targets {
		assert(target >= 0 && target < class_size, "Target is out of bounds", loc=loc)
		targets_copy[i] = target
	}

	probabilities := zeros_like(input, loc=loc)
	output         = _zeros_drop_last(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Cross_Entropy{
			probabilities = probabilities,
			targets       = targets_copy,
			class_size    = class_size,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
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
			max_value = math.max(max_value, input.data[index])
		}

		// Compute exponentials and sum for softmax denominator.
		sum: f32
		for i in 0 ..< class_size {
			index := offset + i
			exp_val := math.exp(input.data[index] - max_value)
			probabilities.data[index] = exp_val
			sum += exp_val
		}

		// Normalize to get actual probabilities.
		for i in 0 ..< class_size {
			index := offset + i
			probabilities.data[index] /= sum
		}

		// Compute negative log likelihood.
		target_index := offset + target
		output.data[sample] = -input.data[target_index] + max_value + math.ln(sum)
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

		upstream_gradient := output.gradient[sample]

		for i in 0 ..< class_size {
			index := offset + i
			target_value: f32 = i == target ? 1 : 0

			gradient := (probabilities.data[index] - target_value) * upstream_gradient

			input.gradient[index] += gradient
		}
	}
}

Relu :: struct {
}

@(require_results)
relu :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Relu{},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_relu_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		if input.data[i] < 0 {
			output.data[i] = 0
		} else {
			output.data[i] = input.data[i]
		}
	}
}

relu_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		if input.data[i] > 0 {
			input.gradient[i] += output.gradient[i]
		}
	}
}

Sigmoid :: struct {
}

@(require_results)
sigmoid :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Sigmoid{},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_sigmoid_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		output.data[i] = 1.0 / (1.0 + math.exp(-input.data[i]))
	}
}

sigmoid_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		sigmoid_value     := output.data[i]
		input.gradient[i] += output.gradient[i] * sigmoid_value * (1.0 - sigmoid_value)
	}
}

GELU_SCALING_FACTOR :: 0.7978845608028654 // math.sqrt(f32(2) / math.PI)

Gelu :: struct {
}

@(require_results)
gelu :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Gelu{},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_gelu_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		x    := input.data[i]
		cube := 0.044715 * x * x * x

		output.data[i] = 0.5 * x * (1.0 + math.tanh(GELU_SCALING_FACTOR * (x + cube)))
	}
}

gelu_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		x          := input.data[i]
		cube       := 0.044715 * x * x * x
		tanh_arg   := GELU_SCALING_FACTOR * (x + cube)
		tanh_out   := math.tanh(tanh_arg)
		cosh_out   := math.cosh(tanh_arg)
		sech_out   := 1.0 / (cosh_out * cosh_out)
		local_grad := 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x)

		input.gradient[i] += local_grad * output.gradient[i]
	}
}

Silu :: struct {
}

@(require_results)
silu :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Silu{},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_silu_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		sigmoid_val := 1.0 / (1.0 + math.exp(-input.data[i]))
		output.data[i] = input.data[i] * sigmoid_val
	}
}

silu_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		x           := input.data[i]
		sigmoid_val := 1.0 / (1.0 + math.exp(-x))

		gradient := sigmoid_val + x * sigmoid_val * (1.0 - sigmoid_val)

		input.gradient[i] += output.gradient[i] * gradient
	}
}

Tanh :: struct {
}

@(require_results)
tanh :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Tanh{},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)

	return
}

cpu_tanh_forward :: proc(op: Operation) {
	input  := op.input
	output := op.output

	for i in 0 ..< len(input) {
		output.data[i] = math.tanh(input.data[i])
	}
}

tanh_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		tanh_value        := output.data[i]
		input.gradient[i] += output.gradient[i] * (1.0 - tanh_value * tanh_value)
	}
}

// Batched_Matmul — batched matrix multiply. C[b, i, j] = sum_k A[b, i, k] * B[b, k, j].
// Both inputs are rank-3, output is rank-3. Used to decompose attention
// into primitives without baking matmul-with-batch into a single fused op.
Batched_Matmul :: struct {
	b:           Tensor,
	batch_count: int,
	m:           int,
	k:           int,
	n:           int,
}

@(require_results)
batched_matmul :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(a.rank == 3 && b.rank == 3, "batched_matmul requires rank-3 inputs", loc=loc)
	assert(a.shape[0] == b.shape[0], "batched_matmul batch dims must match", loc=loc)
	assert(a.shape[2] == b.shape[1], "batched_matmul inner dim must match: a.shape[2] == b.shape[1]", loc=loc)

	batch_count := a.shape[0]
	m           := a.shape[1]
	k           := a.shape[2]
	n           := b.shape[2]

	output = zeros(batch_count, m, n, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Batched_Matmul{
			b           = b,
			batch_count = batch_count,
			m           = m,
			k           = k,
			n           = n,
		},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)
	return
}

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

		a_ptr := ([^]f32)(raw_data(a.data))
		b_ptr := ([^]f32)(raw_data(bt.data))
		c_ptr := ([^]f32)(raw_data(output.data))

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

		a_grad_ptr := ([^]f32)(raw_data(a.gradient))
		b_data_ptr := ([^]f32)(raw_data(bt.data))
		c_grad_ptr := ([^]f32)(raw_data(output.gradient))

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

		a_data_ptr := ([^]f32)(raw_data(a.data))
		b_grad_ptr := ([^]f32)(raw_data(bt.gradient))
		c_grad_ptr := ([^]f32)(raw_data(output.gradient))

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
Permute :: struct {
	axes: [3]int,
}

@(require_results)
permute :: proc(input: Tensor, axes: [3]int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 3, "permute is rank-3 only for now", loc=loc)
	seen := [3]bool{}
	for ax in axes {
		assert(ax >= 0 && ax < 3, "permute axis out of range", loc=loc)
		assert(!seen[ax], "permute axes must be a permutation of (0, 1, 2)", loc=loc)
		seen[ax] = true
	}

	out_shape := [3]int{
		input.shape[axes[0]],
		input.shape[axes[1]],
		input.shape[axes[2]],
	}
	output = zeros(out_shape[0], out_shape[1], out_shape[2], loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Permute{ axes = axes },
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)
	return
}

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

				output.data[dst_idx] = input.data[src_idx]
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

				input.gradient[src_idx] += output.gradient[dst_idx]
			}
		}
	}
}

// Causal_Mask — given a tensor whose trailing two dims are [T, T], replace
// upper-triangle entries (t2 > t1) with -inf, leave the rest untouched.
// Composes with `softmax` to give the "softmax over preceding tokens only"
// semantics that causal attention needs, without baking masking into the
// softmax kernel.
Causal_Mask :: struct {}

@(require_results)
causal_mask :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 2, "causal_mask requires rank >= 2", loc=loc)
	T := input.shape[input.rank - 1]
	assert(input.shape[input.rank - 2] == T, "causal_mask requires square trailing 2D ([..., T, T])", loc=loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Causal_Mask{},
	}
	_current_ctx.backend.forward(op)
	append_operation(op, loc=loc)
	return
}

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
					output.data[idx] = input.data[idx]
				} else {
					output.data[idx] = math.NEG_INF_F32
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
				input.gradient[idx] += output.gradient[idx]
			}
			// Masked positions (t2 > t1): gradient blocked.
		}
	}
}