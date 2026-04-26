// This library is a from-scratch machine learning library focused on
// simplicity. The main working unit is the `Tensor`: a contiguous,
// row-major value with an inline shape (rank-N, capped at
// `MAX_TENSOR_RANK`) and an opaque `storage` pointer that the active
// `Backend` interprets. CPU stores the data + gradient (and optional
// Adam state) as f32 slices in a `Cpu_Storage` struct; GPU stores them
// as `vk.Buffer` handles in a `Gpu_Storage`. Activations live in a
// per-context arena (CPU) or pool (GPU); trainable tensors allocated
// via `make` survive `clear` and carry Adam state.
//
// Operations are recorded onto a thread-local autograd tape during
// forward; `backward` walks the tape in reverse to accumulate
// gradients. Workflow: `clear`, run forward ops, `backward`,
// `optimize` + `update`, repeat.
//
// This file holds the backend-agnostic surface — `Tensor`, `Backend`,
// `Context`, `Operation` and its variants, every public op
// constructor, the autograd walk, the Adam optimizer, and the public
// alloc / data-transfer / accessor procs. The CPU backend
// implementation (kernels, SIMD, worker pool, `cpu_*` hooks) lives in
// `cpu.odin`. The GPU backend lives in `gpu/`.

package machine_learning

import "base:builtin"
import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"

MAX_OPERATIONS :: 4096


// Maximum number of dimensions a Tensor can have. Shapes are stored inline as
// a fixed-size array so Tensor stays a value type with no extra allocation.
// Bump this if a future op needs more.
MAX_TENSOR_RANK :: 6

// The main working unit of the library. Always contiguous, row-major. No
// views, no strides, no transpose-aliasing — `reshape` is the only operation
// that shares storage, and it requires the element count to match exactly.
//
// `backend` identifies which Backend owns this tensor's storage.
// `storage` is an opaque pointer the backend casts to its own struct
// (CPU: ^Cpu_Storage holding f32 slices; GPU: ^Gpu_Storage holding
// vk.Buffer handles). `data(t)` / `gradient(t)` return slices for CPU
// callers; GPU code goes through `set_data` / `get_data` instead.
//
// `type` is the element type. Currently only F32 is implemented and
// asserted at allocation; the field exists so future precisions
// (F16, BF16, etc.) plug in without breaking the API.
//
// `count` is the total element count (product of shape dims). Cached so
// `len(t)` doesn't have to scan `shape`.
Tensor :: struct {
	backend: ^Backend,
	storage: rawptr,

	type:    Data_Type,
	shape:   [MAX_TENSOR_RANK]int,
	rank:    int,
	count:   int,
}

Data_Type :: enum u8 {
	F32 = 0,
}

@(require_results)
data_type_size :: #force_inline proc(t: Data_Type) -> int {
	switch t {
	case .F32: return size_of(f32)
	}
	return 0
}


// Backend-agnostic host-data transfer. CPU does a slice copy; GPU does
// a host-visible-stage upload / download.
set_data :: #force_inline proc(t: Tensor, src: []f32) {
	tt := t
	t.backend.set_data(&tt, src)
}
get_data :: #force_inline proc(t: Tensor, dst: []f32) {
	tt := t
	t.backend.get_data(&tt, dst)
}

// Op execution interface. Each Backend has one `forward` proc that runs the
// math for a freshly built Operation, and one `backward` proc that runs its
// gradient computation. Both procs are expected to switch on `op.variant`
// internally and call into backend-specific kernels. Adding a new op means
// adding a variant to Operation_Variant and one case to each backend's
// dispatch switch — no new field on Backend.
//
// Allocation hooks. Two parameters control all variants:
//
//   `persistent`: false → tracked in the active context's activation
//                         pool, freed in bulk by `clear_storage`.
//                 true  → survives `clear_storage`; explicit `free`.
//
//   `extra_buffers`: count of additional same-shape buffers allocated
//                    alongside data + gradient. 0 for activations and
//                    most persistent tensors; 2 for Adam parameters
//                    (adam_m + adam_v).
//
// `clear_storage` is called by `clear` to bulk-reset the activation pool
// (CPU resets the arena; GPU recycles into the per-context buffer pool).
Backend :: struct {
	alloc:         proc(t: ^Tensor, n: int, persistent: bool, extra_buffers: int),
	free:          proc(t: ^Tensor),
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

	// Copy `src` into `t`'s data storage. CPU does a slice copy; GPU does
	// a host-visible-stage upload. Used by `tensor` / `scalar` so callers
	// don't need a backend-specific path to seed inputs.
	set_data: proc(t: ^Tensor, src: []f32),

	// Read `t`'s data storage into `dst`. CPU does a slice copy; GPU does
	// a host-visible-stage download.
	get_data: proc(t: ^Tensor, dst: []f32),

	// Apply one Adam(W) step + zero gradient on `p`. CPU does the scalar
	// loop; GPU dispatches `opt_step_adam.spv`. `p` must have been
	// allocated with extra_buffers=2.
	parameter_update: proc(opt: Optimizer, p: ^Tensor),

	// Copy data + gradient + adam_m + adam_v from `src` to `dst`. Both
	// must have been allocated with extra_buffers=2.
	parameter_copy: proc(dst, src: ^Tensor),

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

// Top of the thread-local context stack. nil means no active context.
@(thread_local)
_current_ctx: ^Context


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
	return _alloc_tensor(shape, persistent=false, extra_buffers=0, loc=loc)
}

// Allocate a tensor whose storage survives `ml.clear()`. Use this for
// long-lived inputs / lookup tables / etc. that are reused across many
// activation cycles. Pair with `persistent_destroy`.
@(require_results)
persistent_zeros :: proc(shape: ..int, loc := #caller_location) -> (t: Tensor) {
	return _alloc_tensor(shape, persistent=true, extra_buffers=0, loc=loc)
}

// Free a tensor allocated by `persistent_zeros` or `make`.
persistent_destroy :: proc(t: Tensor) {
	if t.backend == nil { return }
	tt := t
	t.backend.free(&tt)
}

@(require_results, private)
_alloc_tensor :: proc(shape: []int, persistent: bool, extra_buffers: int, loc := #caller_location) -> (t: Tensor) {
	assert(_current_ctx != nil && _current_ctx.backend != nil, "Did you forget to call context_create / context_scope?", loc=loc)
	assert(builtin.len(shape) > 0, "Tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Tensor rank exceeds MAX_TENSOR_RANK", loc=loc)

	n := shape_element_count(shape)
	assert(n > 0, "Tensor element count must be positive", loc=loc)

	t.backend = _current_ctx.backend
	t.type    = .F32
	t.count   = n
	t.rank    = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "Tensor dimension must be positive", loc=loc)
		t.shape[i] = d
	}
	t.backend.alloc(&t, n, persistent, extra_buffers)
	return
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

	t.backend = src.backend
	t.storage = src.storage
	t.type    = src.type
	t.count   = src.count
	t.rank    = builtin.len(shape)
	for d, i in shape {
		t.shape[i] = d
	}
	return
}

// Allocate a trainable tensor — a `Tensor` with the Adam optimizer
// state slots populated (`extra_buffers=2` on the alloc hook). Survives
// `ml.clear()`. Pair with `destroy`.
@(require_results)
make :: proc(shape: ..int, loc := #caller_location) -> (t: Tensor, err: mem.Allocator_Error) #optional_allocator_error {
	t = _alloc_tensor(shape, persistent=true, extra_buffers=2, loc=loc)
	return t, nil
}

// Destroy a tensor allocated by `make`.
destroy :: proc(t: Tensor, loc := #caller_location) {
	if t.backend == nil { return }
	tt := t
	t.backend.free(&tt)
}

// Copy data + gradient + adam_m + adam_v from `src` to `dst`. Both must
// have been allocated with `make`.
copy :: proc(dst, src: Tensor, loc := #caller_location) {
	assert(len(dst) == len(src), "Tensor lengths must be equal", loc=loc)
	assert(dst.backend == src.backend, "Tensor copy across backends not supported", loc=loc)
	d, s := dst, src
	dst.backend.parameter_copy(&d, &s)
}

// Fill tensor data with normally distributed random numbers. Works on
// either backend — for non-CPU backends we fill into a temp_allocator
// host buffer and upload via `Backend.set_data`.
fill_normal :: proc(t: Tensor, mean, std: f32) {
	n := len(t)
	if t.backend == &_cpu_backend {
		d := data(t)
		for &v in d {
			v = rand.float32_normal(mean, std)
		}
		return
	}
	buf := builtin.make([]f32, n, allocator=context.temp_allocator)
	for i in 0 ..< n {
		buf[i] = rand.float32_normal(mean, std)
	}
	tt := t
	t.backend.set_data(&tt, buf)
}

// Fill tensor data with a single value. Backend-aware.
fill_value :: proc(t: Tensor, value: f32) {
	n := len(t)
	if t.backend == &_cpu_backend {
		d := data(t)
		for &v in d {
			v = value
		}
		return
	}
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

// Apply one Adam(W) step to `t` and zero its gradient. `t` must have
// been allocated with `make`. This is meant to be called inside the
// scope of `optimize`.
update :: proc(opt: Optimizer, t: Tensor) {
	tt := t
	t.backend.parameter_update(opt, &tt)
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

