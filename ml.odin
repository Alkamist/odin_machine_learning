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

	_Worker :: struct {
		thread:    ^thread.Thread,
		id:        int,
		start_sem: sync.Sema,
	}

	_Dispatch :: struct {
		chunk_proc: proc(start, end: int, raw: rawptr),
		data:       rawptr,
		job_count:  int,
		task_count: int,
	}

	_thread_count: int = 1

	_workers:  []^_Worker
	_shutdown: bool
	_dispatch: _Dispatch
	_done_wg:  sync.Wait_Group

	_worker_proc :: proc(t: ^thread.Thread) {
		w := cast(^_Worker)t.data

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
		_shutdown = false
		n := thread_count - 1
		_workers = builtin.make([]^_Worker, n)
		for i in 0 ..< n {
			w := builtin.new(_Worker)
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

		_dispatch = _Dispatch{
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
// `data` and `gradient` are slices into the global thread-local arena (for
// activations) or into a Parameter's owned buffers (for trainable values).
Tensor :: struct {
	data:     []f32,
	gradient: []f32,
	shape:    [MAX_TENSOR_RANK]int,
	rank:     int,
}

// Trainable values. A Parameter is a Tensor (its data/gradient are
// user-allocated rather than arena-allocated) plus Adam optimizer state.
// `using tensor` lets a Parameter pass anywhere a Tensor is expected.
Parameter :: struct {
	using tensor: Tensor,

	adam_m: []f32,
	adam_v: []f32,
}

Context :: struct {
	arena: mem.Arena,

	operation_count: int,
	operations:      [MAX_OPERATIONS]Operation,
}

@(thread_local)
_global_odin_context: runtime.Context

@(thread_local)
_ctx: Context

@(init)
init_global_context_cleaner :: proc "contextless" () {
	runtime.add_thread_local_cleaner(destroy_global_context)
}

// Initialize the global context.
init :: proc(size: int, allocator := context.allocator, loc := #caller_location) {
	_global_odin_context = context

	destroy_global_context()

	data, err := builtin.make([]byte, size, allocator=allocator, loc=loc)
	assert(err == nil, "Failed to allocate global context arena data", loc=loc)
	mem.arena_init(&_ctx.arena, data)
}

// Destroy the global context. Called automatically.
@(fini)
destroy_global_context :: proc "contextless" () {
	if _ctx.arena.data == nil {
		_ctx = {}
		return
	}

	context = _global_odin_context

	builtin.delete(_ctx.arena.data)
	_ctx = {}
}

// Clear the global arena and operations.
clear :: proc(loc := #caller_location) {
	assert(_ctx.arena.data != nil, "Did you forget to call init?", loc=loc)
	mem.arena_free_all(&_ctx.arena)
	_ctx.operation_count = 0
}

// Get the global arena's allocator.
arena_allocator :: proc() -> mem.Allocator {
	return mem.arena_allocator(&_ctx.arena)
}

// Total element count of a tensor (product of all dims). Equivalent to
// `builtin.len(t.data)`.
@(require_results)
len :: #force_inline proc(t: Tensor) -> int {
	return builtin.len(t.data)
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
	assert(_ctx.arena.data != nil, "Did you forget to call init?", loc=loc)
	assert(builtin.len(shape) > 0, "Tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Tensor rank exceeds MAX_TENSOR_RANK", loc=loc)

	n := shape_element_count(shape)
	assert(n > 0, "Tensor element count must be positive", loc=loc)

	err: mem.Allocator_Error
	t.data, err = builtin.make([]f32, n, allocator=arena_allocator(), loc=loc)
	fmt.assertf(err == nil, "Failed to allocate tensor data in global arena: %v", err, loc=loc)

	t.gradient, err = builtin.make([]f32, n, allocator=arena_allocator(), loc=loc)
	fmt.assertf(err == nil, "Failed to allocate tensor gradient in global arena: %v", err, loc=loc)

	t.rank = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "Tensor dimension must be positive", loc=loc)
		t.shape[i] = d
	}
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
	for v, i in data {
		t.data[i] = v
	}
	return
}

// Single-value 1-D tensor in the global arena.
@(require_results)
scalar :: proc(value: f32, loc := #caller_location) -> (t: Tensor) {
	t = zeros(1, loc=loc)
	t.data[0] = value
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

	t.data     = src.data
	t.gradient = src.gradient
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

	parameter.data     = builtin.make([]f32, n, allocator=allocator, loc=loc) or_return
	parameter.gradient = builtin.make([]f32, n, allocator=allocator, loc=loc) or_return
	parameter.adam_m   = builtin.make([]f32, n, allocator=allocator, loc=loc) or_return
	parameter.adam_v   = builtin.make([]f32, n, allocator=allocator, loc=loc) or_return

	parameter.rank = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "Parameter dimension must be positive", loc=loc)
		parameter.shape[i] = d
	}

	return parameter, nil
}

// Destroy an allocated parameter.
destroy :: proc(parameter: Parameter, loc := #caller_location) {
	builtin.delete(parameter.data,     loc=loc)
	builtin.delete(parameter.gradient, loc=loc)
	builtin.delete(parameter.adam_m,   loc=loc)
	builtin.delete(parameter.adam_v,   loc=loc)
}

// Copy parameter data from src to dst.
copy :: proc(dst, src: Parameter, loc := #caller_location) {
	assert(len(dst) == len(src), "Parameter lengths need to be equal", loc=loc)
	builtin.copy(dst.data,     src.data)
	builtin.copy(dst.gradient, src.gradient)
	builtin.copy(dst.adam_m,   src.adam_m)
	builtin.copy(dst.adam_v,   src.adam_v)
	return
}

// Fill tensor data with normally distributed random numbers.
fill_normal :: proc(t: Tensor, mean, std: f32) {
	for &v in t.data {
		v = rand.float32_normal(mean, std)
	}
}

// Fill tensor data with a single value.
fill_value :: proc(t: Tensor, value: f32) {
	for &v in t.data {
		v = value
	}
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
	for i in 0 ..< len(parameter) {
		grad := parameter.gradient[i]

		parameter.adam_m[i] = opt.beta1 * parameter.adam_m[i] + (1 - opt.beta1) * grad
		parameter.adam_v[i] = opt.beta2 * parameter.adam_v[i] + (1 - opt.beta2) * grad * grad

		m_hat := parameter.adam_m[i] / opt.bias_correction1
		v_hat := parameter.adam_v[i] / opt.bias_correction2

		parameter.data[i] = parameter.data[i] * (1 - opt.learning_rate * opt.weight_decay) - opt.learning_rate * m_hat / (math.sqrt(v_hat) + opt.epsilon)

		parameter.gradient[i] = 0
	}
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
	Concat,
	Interleave,
	Deinterleave,
	Linear,
	Attention,
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
}

Operation :: struct {
	input:   Tensor,
	output:  Tensor,
	variant: Operation_Variant,
}

// Append an operation to the global context for backpropagation.
append_operation :: proc(op: Operation, loc := #caller_location) {
	assert(_ctx.operation_count < MAX_OPERATIONS, "Maximum operations exceeded, did you forget to call clear?", loc=loc)
	_ctx.operations[_ctx.operation_count] = op
	_ctx.operation_count += 1
}

// Iterate backwards through all operations and accumulate gradients through tensors.
// Only the final operation's output gradient is initialized to 1, which means
// that gradients flow backward from the final operation. Gradients won't
// flow properly if you have multiple final operations. I'm not sure of the
// best way to solve that problem.
backward :: proc(loc := #caller_location) {
	if _ctx.operation_count <= 0 {
		return
	}

	// The final gradient needs to be set to 1.
	final_op := _ctx.operations[_ctx.operation_count - 1]
	for i in 0 ..< len(final_op.output) {
		final_op.output.gradient[i] = 1
	}

	for i := _ctx.operation_count - 1; i >= 0; i -= 1 {
		op := _ctx.operations[i]
		switch _ in op.variant {
		case Add:                add_backward               (op, loc=loc)
		case Sub:                sub_backward               (op, loc=loc)
		case Mul:                mul_backward               (op, loc=loc)
		case Div:                div_backward               (op, loc=loc)
		case Exp:                exp_backward               (op, loc=loc)
		case Clamp:              clamp_backward             (op, loc=loc)
		case Min:                min_backward               (op, loc=loc)
		case Max:                max_backward               (op, loc=loc)
		case Mean:               mean_backward              (op, loc=loc)
		case Transpose:          transpose_backward         (op, loc=loc)
		case Select:             select_backward            (op, loc=loc)
		case Slice:              slice_backward             (op, loc=loc)
		case Concat:             concat_backward            (op, loc=loc)
		case Interleave:         interleave_backward        (op, loc=loc)
		case Deinterleave:       deinterleave_backward      (op, loc=loc)
		case Linear:             linear_backward            (op, loc=loc)
		case Attention:          attention_backward         (op, loc=loc)
		case Rope:               rope_backward              (op, loc=loc)
		case Layernorm:          layernorm_backward         (op, loc=loc)
		case Softmax:            softmax_backward           (op, loc=loc)
		case Entropy:            entropy_backward           (op, loc=loc)
		case Log_Softmax:        log_softmax_backward       (op, loc=loc)
		case Mean_Squared_Error: mean_squared_error_backward(op, loc=loc)
		case Cross_Entropy:      cross_entropy_backward     (op, loc=loc)
		case Relu:               relu_backward              (op, loc=loc)
		case Sigmoid:            sigmoid_backward           (op, loc=loc)
		case Gelu:               gelu_backward              (op, loc=loc)
		case Silu:               silu_backward              (op, loc=loc)
		case Tanh:               tanh_backward              (op, loc=loc)
		}
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

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			output.data[o] = a.data[o] + b.data[j]
		}
	}

	append_operation({
		input   = a,
		output  = output,
		variant = Add{
			b      = b,
			stride = stride,
		},
	}, loc=loc)

	return
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

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			output.data[o] = a.data[o] - b.data[j]
		}
	}

	append_operation({
		input   = a,
		output  = output,
		variant = Sub{
			b      = b,
			stride = stride,
		},
	}, loc=loc)

	return
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

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			output.data[o] = a.data[o] * b.data[j]
		}
	}

	append_operation({
		input   = a,
		output  = output,
		variant = Mul{
			b      = b,
			stride = stride,
		},
	}, loc=loc)

	return
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

	stride := len(a) / len(b)
	for i in 0 ..< stride {
		for j in 0 ..< len(b) {
			o := i * len(b) + j
			output.data[o] = a.data[o] / b.data[j]
		}
	}

	append_operation({
		input   = a,
		output  = output,
		variant = Div{
			b      = b,
			stride = stride,
		},
	}, loc=loc)

	return
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

	for i in 0 ..< len(input) {
		output.data[i] = math.exp(input.data[i])
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Exp{},
	}, loc=loc)

	return
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

	for i in 0 ..< len(input) {
		output.data[i] = math.clamp(input.data[i], min_val, max_val)
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Clamp{
			min_val = min_val,
			max_val = max_val,
		},
	}, loc=loc)

	return
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

	for i in 0 ..< len(a) {
		output.data[i] = math.min(a.data[i], b.data[i])
	}

	append_operation({
		input   = a,
		output  = output,
		variant = Min{
			b = b,
		},
	}, loc=loc)

	return
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

	for i in 0 ..< len(a) {
		output.data[i] = math.max(a.data[i], b.data[i])
	}

	append_operation({
		input   = a,
		output  = output,
		variant = Max{
			b = b,
		},
	}, loc=loc)

	return
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

	for sample in 0 ..< count {
		sum: f32
		for i in 0 ..< size {
			index := sample * size + i
			sum += input.data[index]
		}
		output.data[sample] = sum / f32(size)
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Mean{
			size  = size,
			count = count,
		},
	}, loc=loc)

	return
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

	for i in 0 ..< rows {
		for j in 0 ..< columns {
			output.data[j * rows + i] = input.data[i * columns + j]
		}
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Transpose{
			rows = rows,
		},
	}, loc=loc)

	return
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

	out_shape: [MAX_TENSOR_RANK]int = input.shape
	out_shape[0] = builtin.len(indices)
	output = zeros(..out_shape[:input.rank], loc=loc)

	for i in 0 ..< builtin.len(indices) {
		indices_copy[i] = indices[i]
		for j in 0 ..< size {
			output.data[i * size + j] = input.data[indices[i] * size + j]
		}
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Select{
			indices  = indices_copy,
			size     = size,
		}
	}, loc=loc)

	return
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

	builtin.copy(output.data, input.data[start:end])

	append_operation({
		input   = input,
		output  = output,
		variant = Slice{
			start = start,
			end   = end,
		},
	}, loc=loc)

	return
}

slice_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant := op.variant.(Slice)
	start   := variant.start

	for i in 0 ..< len(output) {
		input.gradient[start + i] += output.gradient[i]
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

	start := 0
	for input in inputs_copy {
		builtin.copy(output.data[start:][:len(input)], input.data)
		start += len(input)
	}

	append_operation({
		input   = {},
		output  = output,
		variant = Concat{
			inputs = inputs_copy,
		},
	}, loc=loc)

	return
}

concat_backward :: proc(op: Operation, loc := #caller_location) {
	output := op.output

	variant := op.variant.(Concat)
	inputs  := variant.inputs

	start := 0
	for input in inputs {
		for i in 0 ..< len(input) {
			input.gradient[i] += output.gradient[start + i]
		}
		start += len(input)
	}
}

Interleave :: struct {
	inputs: []Tensor,
}

// Interleave multiple tensors. All inputs must share shape. Output shape =
// inputs[0].shape with the trailing dim multiplied by len(inputs).
@(require_results)
interleave :: proc(inputs: ..Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(builtin.len(inputs) > 1, "Must have at least 2 inputs", loc=loc)

	first := inputs[0]
	for i in 1 ..< builtin.len(inputs) {
		assert(inputs[i].rank == first.rank, "All interleave inputs must have the same rank", loc=loc)
		for d in 0 ..< first.rank {
			assert(inputs[i].shape[d] == first.shape[d], "All interleave inputs must have the same shape", loc=loc)
		}
	}

	inputs_copy := builtin.make([]Tensor, builtin.len(inputs), allocator=arena_allocator())
	for input, i in inputs {
		inputs_copy[i] = input
	}

	length := len(first)
	output = _zeros_replace_trailing(first, first.shape[first.rank - 1] * builtin.len(inputs), loc=loc)

	for i in 0 ..< length {
		for j in 0 ..< builtin.len(inputs) {
			output.data[i * builtin.len(inputs) + j] = inputs[j].data[i]
		}
	}

	append_operation({
		input   = {},
		output  = output,
		variant = Interleave{
			inputs = inputs_copy,
		},
	}, loc=loc)

	return
}

interleave_backward :: proc(op: Operation, loc := #caller_location) {
	output  := op.output

	variant := op.variant.(Interleave)
	inputs  := variant.inputs

	length := len(inputs[0])

	for i in 0 ..< length {
		for j in 0 ..< builtin.len(inputs) {
			inputs[j].gradient[i] += output.gradient[i * builtin.len(inputs) + j]
		}
	}
}

Deinterleave :: struct {
	column:       int,
	column_count: int,
}

// Extract one of `column_count` interleaved channels from a tensor's
// trailing dim. Output shape = input.shape with trailing / column_count.
@(require_results)
deinterleave :: proc(input: Tensor, column, column_count: int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "deinterleave input must have rank >= 1", loc=loc)
	trailing := input.shape[input.rank - 1]
	assert(trailing % column_count == 0, "Trailing dim must be divisible by column_count", loc=loc)

	output = _zeros_replace_trailing(input, trailing / column_count, loc=loc)

	for i in 0 ..< len(output) {
		output.data[i] = input.data[i * column_count + column]
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Deinterleave{
			column       = column,
			column_count = column_count,
		},
	}, loc=loc)

	return
}

deinterleave_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	variant      := op.variant.(Deinterleave)
	column       := variant.column
	column_count := variant.column_count

	for i in 0 ..< len(output) {
		input.gradient[i * column_count + column] += output.gradient[i]
	}
}

// Use 8-lane (256-bit / AVX) SIMD when the target supports AVX; otherwise fall
// back to plain scalar loops that LLVM auto-vectorizes for whatever the target
// does support (typically 4-lane SSE). Forcing 256-bit ops on a non-AVX target
// makes LLVM emit 2x128 SSE pairs plus a software fma (mul+add), which ends up
// noticeably slower than letting the compiler auto-vectorize the scalar form.
_HAS_AVX :: intrinsics.has_target_feature("avx")

when _HAS_AVX {
	_SIMD_LANES :: 8
	_F32x8      :: #simd[_SIMD_LANES]f32

	// sum(a[i] * b[i]) for i in 0..<n. Uses FMA when available.
	_simd_dot_f32 :: #force_inline proc "contextless" (a, b: [^]f32, n: int) -> f32 {
		acc: _F32x8
		i := 0
		for ; i + _SIMD_LANES <= n; i += _SIMD_LANES {
			av := intrinsics.unaligned_load((^_F32x8)(&a[i]))
			bv := intrinsics.unaligned_load((^_F32x8)(&b[i]))
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
		av := _F32x8(a)
		i := 0
		for ; i + _SIMD_LANES <= n; i += _SIMD_LANES {
			xv := intrinsics.unaligned_load((^_F32x8)(&x[i]))
			yv := intrinsics.unaligned_load((^_F32x8)(&y[i]))
			intrinsics.unaligned_store((^_F32x8)(&y[i]), simd.fma(xv, av, yv))
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

	append_operation(op, loc=loc)

	return
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

Attention :: struct {
	input_size:  int,
	output_size: int,
	token_count: int,
	head_count:  int,
	head_size:   int,
	scale:       f32,
	causal:      bool,

	pre_attention_scores:  Tensor,
	post_attention_scores: Tensor,
}

// Multi-head scaled dot product attention. Input is interleaved QKV with
// shape [token_count, 3 * embedding]. Output shape is [token_count, embedding].
// `head_count` stays explicit because it isn't derivable from the storage shape.
@(require_results)
attention :: proc(input: Tensor, head_count: int, causal := true, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 2, "attention requires a 2-D tensor [tokens, 3*embedding]", loc=loc)

	token_count := input.shape[0]
	input_size  := input.shape[1]
	assert(input_size % 3 == 0, "Trailing dim must be divisible by 3 (for Q, K, V)", loc=loc)

	output_size := input_size / 3
	assert(output_size % head_count == 0, "Output size must be divisible by head count", loc=loc)

	pre_attention_scores  := zeros(head_count * token_count * token_count, loc=loc)
	post_attention_scores := zeros(head_count * token_count * token_count, loc=loc)

	output = zeros(token_count, output_size, loc=loc)

	head_size := output_size / head_count
	scale     := 1.0 / math.sqrt(f32(head_size))

	op := Operation{
		input =   input,
		output =  output,
		variant = Attention{
			input_size  = input_size,
			output_size = output_size,
			token_count = token_count,
			head_count  = head_count,
			head_size   = head_size,
			scale       = scale,
			causal      = causal,

			pre_attention_scores  = pre_attention_scores,
			post_attention_scores = post_attention_scores,
		}
	}

	parallelize(token_count, thread_count(), op, proc(index: int, op: Operation) {
		input, output := op.input, op.output

		variant               := op.variant.(Attention)
		input_size            := variant.input_size
		output_size           := variant.output_size
		token_count           := variant.token_count
		head_count            := variant.head_count
		head_size             := variant.head_size
		scale                 := variant.scale
		causal                := variant.causal
		pre_attention_scores  := variant.pre_attention_scores
		post_attention_scores := variant.post_attention_scores

		t := index

		input_ptr  := ([^]f32)(raw_data(input.data))
		output_ptr := ([^]f32)(raw_data(output.data))

		for h in 0 ..< head_count {
			query_offset := t * input_size + h * head_size
			score_offset := h * token_count * token_count + t * token_count

			max_t2 := causal ? t : token_count - 1

			max_value := math.NEG_INF_F32

			// Compute raw attention scores: dot(Q[t,h], K[t2,h]) * scale.
			for t2 in 0 ..= max_t2 {
				key_offset := t2 * input_size + h * head_size + output_size

				value := _simd_dot_f32(input_ptr[query_offset:], input_ptr[key_offset:], head_size)
				value *= scale

				if value > max_value {
					max_value = value
				}

				pre_attention_scores.data[score_offset + t2] = value
			}

			// Apply softmax to get attention weights.
			exp_sum: f32
			for t2 in 0 ..= max_t2 {
				exp_v := math.exp(pre_attention_scores.data[score_offset + t2] - max_value)
				exp_sum += exp_v
				post_attention_scores.data[score_offset + t2] = exp_v
			}
			exp_sum_inv: f32 = exp_sum == 0 ? 0 : 1 / exp_sum

			// Apply normalization and causal masking.
			for t2 in 0 ..< token_count {
				if t2 <= max_t2 {
					post_attention_scores.data[score_offset + t2] *= exp_sum_inv
				} else {
					post_attention_scores.data[score_offset + t2] = 0
				}
			}

			output_offset := t * output_size + h * head_size

			// Accumulate weighted values: output[t,h] += sum_t2 score[t,t2] * V[t2,h].
			for t2 in 0 ..= max_t2 {
				value_offset := t2 * input_size + h * head_size + output_size * 2
				score        := post_attention_scores.data[score_offset + t2]
				_simd_axpy_f32(output_ptr[output_offset:], input_ptr[value_offset:], score, head_size)
			}
		}
	})

	append_operation(op, loc=loc)

	return
}

attention_backward :: proc(op: Operation, loc := #caller_location) {
	token_count := op.variant.(Attention).token_count

	parallelize(token_count, token_count, op, proc(index: int, op: Operation) {
		input, output := op.input, op.output

		variant               := op.variant.(Attention)
		input_size            := variant.input_size
		output_size           := variant.output_size
		token_count           := variant.token_count
		head_count            := variant.head_count
		head_size             := variant.head_size
		scale                 := variant.scale
		causal                := variant.causal
		pre_attention_scores  := variant.pre_attention_scores
		post_attention_scores := variant.post_attention_scores

		t := index

		for h in 0 ..< head_count {
			score_offset  := h * token_count * token_count + t * token_count
			query_offset  := t * input_size + h * head_size
			output_offset := t * output_size + h * head_size

			max_t2 := causal ? t : token_count - 1

			input_data_ptr  := ([^]f32)(raw_data(input.data))
			input_grad_ptr  := ([^]f32)(raw_data(input.gradient))
			output_grad_ptr := ([^]f32)(raw_data(output.gradient))

			// Backpropagate through weighted sum of values.
			//   post_attn_grad[t2] += dot(V[t2,h], dout[t,h])
			//   input_grad[V[t2,h]] += post_attn[t2] * dout[t,h]
			for t2 in 0 ..= max_t2 {
				value_offset := t2 * input_size + h * head_size + output_size * 2

				post_attention_scores.gradient[score_offset + t2] += _simd_dot_f32(
					input_data_ptr[value_offset:],
					output_grad_ptr[output_offset:],
					head_size,
				)

				score := post_attention_scores.data[score_offset + t2]
				_simd_axpy_f32(input_grad_ptr[value_offset:], output_grad_ptr[output_offset:], score, head_size)
			}

			// Backpropagate through softmax.
			for t2 in 0 ..= max_t2 {
				for t3 in 0 ..= max_t2 {
					indicator: f32 = t2 == t3 ? 1 : 0
					local_derivative := post_attention_scores.data[score_offset + t2] * (indicator - post_attention_scores.data[score_offset + t3])
					pre_attention_scores.gradient[score_offset + t3] += local_derivative * post_attention_scores.gradient[score_offset + t2]
				}
			}

			// Backpropagate through scaled dot product.
			//   input_grad[Q[t,h]] += factor * K[t2,h]
			//   input_grad[K[t2,h]] += factor * Q[t,h]
			// where factor = pre_attn_grad[t2] * scale.
			for t2 in 0 ..= max_t2 {
				key_offset := t2 * input_size + h * head_size + output_size
				factor     := pre_attention_scores.gradient[score_offset + t2] * scale

				_simd_axpy_f32(input_grad_ptr[query_offset:], input_data_ptr[key_offset:],   factor, head_size)
				_simd_axpy_f32(input_grad_ptr[key_offset:],   input_data_ptr[query_offset:], factor, head_size)
			}
		}
	})
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

	append_operation({
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
	}, loc=loc)

	return
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

	EPSILON :: 1e-5

	count := _leading_count(input)
	size  := input.shape[input.rank - 1]

	mean := zeros(count, loc=loc)
	rstd := zeros(count, loc=loc)

	output = zeros_like(input, loc=loc)

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

	append_operation({
		input   = input,
		output  = output,
		variant = Layernorm{
			weight = weight,
			mean   = mean,
			rstd   = rstd,
			count  = count,
			size   = size,
		},
	}, loc=loc)

	return
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

	append_operation({
		input   = input,
		output  = output,
		variant = Softmax{
			size  = size,
			count = count,
		},
	}, loc=loc)

	return
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

	append_operation({
		input   = input,
		output  = output,
		variant = Log_Softmax{
			size  = size,
			count = count,
		},
	}, loc=loc)

	return
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

	append_operation({
		input   = probabilities,
		output  = output,
		variant = Entropy{
			size  = size,
			count = count,
		},
	}, loc=loc)

	return
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

	count       := _leading_count(predictions)
	sample_size := predictions.shape[predictions.rank - 1]

	output = _zeros_drop_last(predictions, loc=loc)

	for sample in 0 ..< count {
		sum_squared_error: f32

		for i in 0 ..< sample_size {
			index := sample * sample_size + i
			diff  := predictions.data[index] - targets.data[index]
			sum_squared_error += diff * diff
		}

		output.data[sample] = sum_squared_error / f32(sample_size)
	}

	append_operation({
		input   = predictions,
		output  = output,
		variant = Mean_Squared_Error{
			targets = targets,
			count   = count,
		},
	}, loc=loc)

	return
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

	for sample in 0 ..< sample_count {
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

	append_operation({
		input   = input,
		output  = output,
		variant = Cross_Entropy{
			probabilities = probabilities,
			targets       = targets_copy,
			class_size    = class_size,
		},
	}, loc=loc)

	return
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

	for i in 0 ..< len(input) {
		if input.data[i] < 0 {
			output.data[i] = 0
		} else {
			output.data[i] = input.data[i]
		}
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Relu{},
	}, loc=loc)

	return
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

	for i in 0 ..< len(input) {
		output.data[i] = 1.0 / (1.0 + math.exp(-input.data[i]))
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Sigmoid{},
	}, loc=loc)

	return
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

	for i in 0 ..< len(input) {
		x    := input.data[i]
		cube := 0.044715 * x * x * x

		output.data[i] = 0.5 * x * (1.0 + math.tanh(GELU_SCALING_FACTOR * (x + cube)))
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Gelu{},
	}, loc=loc)

	return
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

	for i in 0 ..< len(input) {
		sigmoid_val := 1.0 / (1.0 + math.exp(-input.data[i]))
		output.data[i] = input.data[i] * sigmoid_val
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Silu{},
	}, loc=loc)

	return
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

	for i in 0 ..< len(input) {
		output.data[i] = math.tanh(input.data[i])
	}

	append_operation({
		input   = input,
		output  = output,
		variant = Tanh{},
	}, loc=loc)

	return
}

tanh_backward :: proc(op: Operation, loc := #caller_location) {
	input, output := op.input, op.output

	for i in 0 ..< len(input) {
		tanh_value        := output.data[i]
		input.gradient[i] += output.gradient[i] * (1.0 - tanh_value * tanh_value)
	}
}