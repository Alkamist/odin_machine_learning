package machine_learning

import "base:runtime"
import "base:builtin"
import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:thread"


// This library was designed with the goal of exploring and understanding machine
// learning. Some of the main goals are simplicity and understandability.
//
// The main working units of the library are Arrays, rather than Tensors. Shape
// information is inferred when possible, and passed in as function arguments
// when not. Arrays are just slices of data and gradients that are stored in a
// global thread local arena. Parameters are an extension of Arrays, and are
// allocated by the user; these are values that can be trained.
//
// Operations that are performed are stored in a global thread local buffer,
// so that they can be backpropagated to calculate gradients. So basically,
// the workflow is to call clear, do your calculations, call backward to accumulate
// the gradients, update parameters, and repeat.
//
// One downside to doing all of this from scratch is that this library isn't
// particularly optimized. Some calculations are parallelized, but they can definitely
// be improved. I'm not sure if my approach to parallelization is very good.


MAX_OPERATIONS :: 4096

_thread_count := 1
_thread_pool: thread.Pool

_startup_thread_pool :: proc(thread_count: int) {
	thread.pool_init(&_thread_pool, context.allocator, thread_count)
	thread.pool_start(&_thread_pool)
}

_cleanup_thread_pool :: proc() {
	thread.pool_join(&_thread_pool)
	thread.pool_destroy(&_thread_pool)
	_thread_pool = {}
}

// Get the current thread count.
thread_count :: #force_inline proc() -> int {
	return _thread_count
}

// Set the thread count for parallelized calculations.
// Should only be called from the main thread.
set_thread_count :: proc(count: int, loc := #caller_location) {
	assert(count > 0, "Thread count must be at least 1", loc=loc)

	// If the thread count hasn't changed, there's no need to do anything.
	if count == _thread_count {
		return
	}

	// Cleanup the thread pool if necessary.
	if _thread_count > 1 {
		_cleanup_thread_pool()
	}

	// Single-threaded, no need to startup a new thread pool.
	if count == 1 {
		_thread_count = 1
		return
	}

	// Multi-threaded, so start up a new thread pool.
	_startup_thread_pool(count)
	_thread_count = count
}

@(fini)
thread_pool_fini :: proc "contextless" () {
	// If thread count is 1 or less, there's no need to clean up anything.
	if _thread_count <= 1 {
		return
	}

	context = _global_odin_context

	_cleanup_thread_pool()
}

// Parallelize a job within the given amount of parallel tasks.
parallelize :: proc(job_count, task_count: int, data: $Data, job: proc(index: int, data: Data)) {
	if job_count <= 1 {
		job(0, data)
		return
	}

	if task_count <= 1 || _thread_count <= 1 {
		for i in 0 ..< job_count {
			job(i, data)
		}
		return
	}

	Task_Data :: struct {
		job_count:  int,
		task_count: int,
		data:       Data,
		job:        proc(index: int, data: Data),
	}

	task_data := Task_Data{
		job_count  = job_count,
		task_count = task_count,
		data       = data,
		job        = job,
	}

	task_proc :: proc(task: thread.Task) {
		task_data := cast(^Task_Data)task.data

		// Calculate bounds for the task.
		jobs_per_task := (task_data.job_count + task_data.task_count - 1) / task_data.task_count
		start         := task.user_index * jobs_per_task
		end           := math.min(start + jobs_per_task, task_data.job_count)

		// Execute jobs within calculated bounds.
		for i in start ..< end {
			task_data.job(i, task_data.data)
		}
	}

	for i in 0 ..< task_count {
		// Quick bounds check to avoid creating unnecessary tasks.
		jobs_per_task := (job_count + task_count - 1) / task_count
		start         := i * jobs_per_task
		if start >= job_count {
			break
		}

		thread.pool_add_task(&_thread_pool, context.allocator, task_proc, &task_data, i)
	}

	// Wait for all of the tasks to finish.
	for !thread.pool_is_empty(&_thread_pool) {
		thread.pool_pop_done(&_thread_pool)
	}
}

// The main working unit of the library.
Array :: struct {
	data:     []f32,
	gradient: []f32,
}

// Trainable values.
Parameter :: struct {
	using array: Array,

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

// Get the length of an array.
@(require_results)
len :: #force_inline proc(arr: Array) -> int {
	return builtin.len(arr.data)
}

// Allocate an array in the global arena initialized with zeros.
@(require_results)
zeros :: proc(len: int, loc := #caller_location) -> (arr: Array) {
	assert(_ctx.arena.data != nil, "Did you forget to call init?", loc=loc)
	assert(len > 0, "Length must be at least 1", loc=loc)

	err: mem.Allocator_Error

	arr.data, err = builtin.make([]f32, len, allocator=arena_allocator(), loc=loc)
	fmt.assertf(err == nil, "Failed to allocate array data in global arena: %v", err, loc=loc)

	arr.gradient, err = builtin.make([]f32, len, allocator=arena_allocator(), loc=loc)
	fmt.assertf(err == nil, "Failed to allocate array gradient in global arena: %v", err, loc=loc)

	return
}

// Copy data to the global arena as an array.
@(require_results)
array :: proc(data: []f32, loc := #caller_location) -> (arr: Array) {
	assert(builtin.len(data) > 0, "Length must be at least 1", loc=loc)

	arr = zeros(builtin.len(data), loc=loc)
	for i in 0 ..< len(arr) {
		arr.data[i] = data[i]
	}

	return
}

// Copy a single value to the global arena as an array.
@(require_results)
scalar :: proc(value: f32, loc := #caller_location) -> (arr: Array) {
	arr = zeros(1, loc=loc)
	arr.data[0] = value
	return
}

// Allocate a parameter initialized with zeros.
@(require_results)
make :: proc(len: int, allocator := context.allocator, loc := #caller_location) -> (parameter: Parameter, err: mem.Allocator_Error) #optional_allocator_error {
	assert(len > 0, "Length must be at least 1", loc=loc)

	parameter.data     = builtin.make([]f32, len, allocator=allocator, loc=loc) or_return
	parameter.gradient = builtin.make([]f32, len, allocator=allocator, loc=loc) or_return
	parameter.adam_m   = builtin.make([]f32, len, allocator=allocator, loc=loc) or_return
	parameter.adam_v   = builtin.make([]f32, len, allocator=allocator, loc=loc) or_return

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

// Fill array data with normally distributed random numbers.
fill_normal :: proc(arr: Array, mean, std: f32) {
	for &v in arr.data {
		v = rand.float32_normal(mean, std)
	}
}

// Fill array data with a single value.
fill_value :: proc(arr: Array, value: f32) {
	for &v in arr.data {
		v = value
	}
}

// Perform He initialization on an array.
he_initialization :: proc(arr: Array, input_features: int) {
	fill_normal(arr, 0, math.sqrt(2 / f32(input_features)))
}

// Perform Xavier/Glorot initialization on an array.
xavier_initialization :: proc(arr: Array, input_features, output_features: int) {
	fill_normal(arr, 0, math.sqrt(2 / f32(input_features + output_features)))
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
	input:   Array,
	output:  Array,
	variant: Operation_Variant,
}

// Append an operation to the global context for backpropagation.
append_operation :: proc(op: Operation, loc := #caller_location) {
	assert(_ctx.operation_count < MAX_OPERATIONS, "Maximum operations exceeded, did you forget to call clear?", loc=loc)
	_ctx.operations[_ctx.operation_count] = op
	_ctx.operation_count += 1
}

// Iterate backwards through all operations and accumulate gradients through arrays.
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
	b:      Array,
	stride: int,
}

// Add two arrays, b is broadcasted into a if necessary.
@(require_results)
add :: proc(a, b: Array, loc := #caller_location) -> (output: Array) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros(len(a), loc=loc)

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
	b:      Array,
	stride: int,
}

// Subtract two arrays, b is broadcasted into a if necessary.
@(require_results)
sub :: proc(a, b: Array, loc := #caller_location) -> (output: Array) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros(len(a), loc=loc)

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
	b:      Array,
	stride: int,
}

// Multiply two arrays, b is broadcasted into a if necessary.
@(require_results)
mul :: proc(a, b: Array, loc := #caller_location) -> (output: Array) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros(len(a), loc=loc)

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
	b:      Array,
	stride: int,
}

// Divide two arrays, b is broadcasted into a if necessary.
@(require_results)
div :: proc(a, b: Array, loc := #caller_location) -> (output: Array) {
	assert(len(a) % len(b) == 0, "A length must be divisible by B length", loc=loc)

	output = zeros(len(a), loc=loc)

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
exp :: proc(input: Array, loc := #caller_location) -> (output: Array) {
	output = zeros(len(input), loc=loc)

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
clamp :: proc(input: Array, min_val, max_val: f32, loc := #caller_location) -> (output: Array) {
	assert(min_val <= max_val, "Requires min_val <= max_val", loc=loc)

	output = zeros(len(input), loc=loc)

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
	b: Array,
}

@(require_results)
min :: proc(a, b: Array, loc := #caller_location) -> (output: Array) {
	assert(len(a) == len(b), "Requires inputs of equal length", loc=loc)

	output = zeros(len(a), loc=loc)

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
	b: Array,
}

@(require_results)
max :: proc(a, b: Array, loc := #caller_location) -> (output: Array) {
	assert(len(a) == len(b), "Requires inputs of equal length", loc=loc)

	output = zeros(len(a), loc=loc)

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
mean :: proc(input: Array, count := 1, loc := #caller_location) -> (output: Array) {
	assert(count > 0, "Count must be at least 1", loc=loc)
	assert(len(input) % count == 0, "Input length must be divisible by count", loc=loc)

	size := len(input) / count
	output = zeros(count, loc=loc)

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

// Transpose the data of an array with the given row count.
@(require_results)
transpose :: proc(input: Array, rows: int, loc := #caller_location) -> (output: Array) {
	assert(len(input) % rows == 0, "Input length must be divisible by rows", loc=loc)

	columns := len(input) / rows

	output = zeros(len(input), loc=loc)

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

// Select rows from an array based on indices and size.
@(require_results)
select :: proc(input: Array, indices: []int, size := 1, loc := #caller_location) -> (output: Array) {
	assert(len(input) % size == 0, "Input length must be divisible by size", loc=loc)

	indices_copy := builtin.make([]int, builtin.len(indices), allocator=arena_allocator())

	output = zeros(size * builtin.len(indices), loc=loc)

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

// Slice an input array. Copies the data.
@(require_results)
slice :: proc(input: Array, start, end: int, loc := #caller_location) -> (output: Array) {
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
	inputs: []Array,
}

// Concatenate multiple arrays.
@(require_results)
concat :: proc(inputs: ..Array, loc := #caller_location) -> (output: Array) {
	assert(builtin.len(inputs) > 0, "Requires at least one input", loc=loc)

	inputs_copy := builtin.make([]Array, builtin.len(inputs), allocator=arena_allocator())

	output_length := 0
	for input, i in inputs {
		inputs_copy[i] = input
		output_length += len(input)
	}

	output = zeros(output_length, loc=loc)

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
	inputs: []Array,
}

// Interleave multiple arrays.
@(require_results)
interleave :: proc(inputs: ..Array, loc := #caller_location) -> (output: Array) {
	assert(builtin.len(inputs) > 1, "Must have at least 2 inputs", loc=loc)

	inputs_copy := builtin.make([]Array, builtin.len(inputs), allocator=arena_allocator())

	length := len(inputs[0])
	for i in 0 ..< builtin.len(inputs) {
		assert(len(inputs[i]) == length, "All inputs must have the same length", loc=loc)
		inputs_copy[i] = inputs[i]
	}

	output = zeros(length * builtin.len(inputs), loc=loc)

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

// Extract the desired column from an interleaved array.
@(require_results)
deinterleave :: proc(input: Array, column, column_count: int, loc := #caller_location) -> (output: Array) {
	assert(len(input) % column_count == 0, "Input length must be divisible by column count", loc=loc)

	output = zeros(len(input) / column_count, loc=loc)

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

Linear :: struct {
	weight:      Array,
	input_size:  int,
	output_size: int,
	count:       int,
}

// Perform a linear transformation. Basically the matrix vector dot product
// when count is 1, and matrix multiplication when count is greater than 1.
@(require_results)
linear :: proc(input, weight: Array, count := 1, loc := #caller_location) -> (output: Array) {
	assert(count > 0, "Count must be at least 1", loc=loc)
	assert(len(input) % count == 0, "Input length must be divisible by count", loc=loc)

	input_size := len(input) / count
	assert(len(weight) % input_size == 0, "Weight length must be divisible by input size", loc=loc)

	output_size := len(weight) / input_size
	output = zeros(count * output_size, loc=loc)

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
		count       := variant.count

		BLOCK_SIZE :: 8

		input_offset  := index * input_size
		output_offset := index * output_size

		for o_block := 0; o_block < output_size; o_block += BLOCK_SIZE {
			o_end := math.min(o_block + BLOCK_SIZE, output_size)

			for i_block := 0; i_block < input_size; i_block += BLOCK_SIZE {
				i_end := math.min(i_block + BLOCK_SIZE, input_size)

				for o in o_block ..< o_end {
					sum: f32
					for i in i_block ..< i_end {
						sum += weight.data[o * input_size + i] * input.data[input_offset + i]
					}
					output.data[output_offset + o] += sum
				}
			}
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

		input_offset  := index * input_size
		output_offset := index * output_size

		for o in 0 ..< output_size {
			output_gradient := output.gradient[output_offset + o]

			if output_gradient == 0 do continue

			w := o * input_size

			for i in 0 ..< input_size {
				weight.gradient[w + i] += input.data[input_offset + i] * output_gradient
			}

			for i in 0 ..< input_size {
				input.gradient[input_offset + i] += weight.data[w + i] * output_gradient
			}
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

	pre_attention_scores:  Array,
	post_attention_scores: Array,
}

// Perform multi-head scaled dot product attention. Input is an interleaved qkv array.
@(require_results)
attention :: proc(input: Array, token_count, head_count: int, causal := true, loc := #caller_location) -> (output: Array) {
	assert(len(input) % token_count == 0, "Input length must be divisible by token count")

	input_size := len(input) / token_count
	assert(input_size % 3 == 0, "Input size must be divisible by 3 (for Q, K, V)", loc=loc)

	output_size := input_size / 3
	assert(output_size % head_count == 0, "Output size must be divisible by head count", loc=loc)

	pre_attention_scores  := zeros(head_count * token_count * token_count, loc=loc)
	post_attention_scores := zeros(head_count * token_count * token_count, loc=loc)

	output = zeros(token_count * output_size, loc=loc)

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

		for h in 0 ..< head_count {
			query_offset := t * input_size + h * head_size
			score_offset := h * token_count * token_count + t * token_count

			max_t2 := causal ? t : token_count - 1

			max_value := math.NEG_INF_F32

			// Compute raw attention scores.
			for t2 in 0 ..= max_t2 {
				key_offset := t2 * input_size + h * head_size + output_size

				// Compute dot product between query and key.
				value: f32
				for i in 0 ..< head_size {
					value += input.data[query_offset + i] * input.data[key_offset + i]
				}

				// Apply scaling factor.
				value *= scale

				// Track maximum for numerical stability.
				if value > max_value {
					max_value = value
				}

				// Store raw attention score.
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

			// Accumulate weighted values.
			for t2 in 0 ..= max_t2 {
				value_offset := t2 * input_size + h * head_size + output_size * 2
				score        := post_attention_scores.data[score_offset + t2]
				for i in 0 ..< head_size {
					output.data[output_offset + i] += score * input.data[value_offset + i]
				}
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

			// Backpropagate through weighted sum of values.
			for t2 in 0 ..= max_t2 {
				value_offset := t2 * input_size + h * head_size + output_size * 2
				for i in 0 ..< head_size {
					post_attention_scores.gradient[score_offset + t2] += input.data[value_offset + i] * output.gradient[output_offset + i]
					input.gradient[value_offset + i] += post_attention_scores.data[score_offset + t2] * output.gradient[output_offset + i]
				}
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
			for t2 in 0 ..= max_t2 {
				key_offset := t2 * input_size + h * head_size + output_size
				for i in 0 ..< head_size {
					input.gradient[query_offset + i] += input.data[key_offset + i] * pre_attention_scores.gradient[score_offset + t2] * scale
					input.gradient[key_offset + i]   += input.data[query_offset + i] * pre_attention_scores.gradient[score_offset + t2] * scale
				}
			}
		}
	})
}

Rope :: struct {
	token_count: int,
	head_count:  int,
	head_size:   int,
	base:        f32,

	cos_cache: Array,
	sin_cache: Array,
}

// Perform rotary position embedding.
@(require_results)
rope :: proc(input: Array, token_count, head_count: int, base: f32 = 10000, loc := #caller_location) -> (output: Array) {
	assert(len(input) % token_count == 0, "Input length must be divisible by token count", loc=loc)

	input_size := len(input) / token_count
	assert(input_size % head_count == 0, "Input size must be divisible by head count", loc=loc)

	head_size := input_size / head_count
	assert(head_size % 2 == 0, "Head size must be even", loc=loc)

	output = zeros(len(input), loc=loc)

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
	weight: Array,
	mean:   Array,
	rstd:   Array,
	count:  int,
	size:   int,
}

@(require_results)
layernorm :: proc(input, weight: Array, count := 1, loc := #caller_location) -> (output: Array) {
	assert(count > 0, "Count must be at least 1", loc=loc)
	assert(len(input) % count == 0, "Input length must be divisible by count", loc=loc)

	EPSILON :: 1e-5

	mean := zeros(count, loc=loc)
	rstd := zeros(count, loc=loc)

	output = zeros(len(input), loc=loc)

	size := len(input) / count

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
softmax :: proc(input: Array, count := 1, loc := #caller_location) -> (output: Array) {
	assert(count > 0, "Count must be at least 1", loc=loc)
	assert(len(input) % count == 0, "Input length must be divisible by count", loc=loc)

	output = zeros(len(input), loc=loc)

	size := len(input) / count

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
log_softmax :: proc(input: Array, count := 1, loc := #caller_location) -> (output: Array) {
	assert(count > 0, "Count must be at least 1", loc=loc)
	assert(len(input) % count == 0, "Input length must be divisible by count", loc=loc)

	output = zeros(len(input), loc=loc)

	size := len(input) / count

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
entropy :: proc(probabilities: Array, count := 1, loc := #caller_location) -> (output: Array) {
	assert(count > 0, "Count must be at least 1", loc=loc)
	assert(len(probabilities) % count == 0, "Input length must be divisible by count", loc=loc)

	output = zeros(count, loc=loc)

	size := len(probabilities) / count

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
	targets: Array,
	count:   int,
}

@(require_results)
mean_squared_error :: proc(predictions, targets: Array, count := 1, loc := #caller_location) -> (output: Array) {
	assert(len(predictions) == len(targets), "Predictions and targets must have same length", loc=loc)
	assert(count > 0, "Count must be at least 1", loc=loc)
	assert(len(predictions) % count == 0, "Input length must be divisible by count", loc=loc)

	sample_size := len(predictions) / count

	output = zeros(count, loc=loc)

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
	probabilities: Array,
	targets:       []int,
	class_size:    int,
}

// Cross entropy performs softmax internally, so it expects the input
// to not already be softmaxed.
@(require_results)
cross_entropy :: proc(input: Array, targets: []int, loc := #caller_location) -> (output: Array) {
	sample_count := builtin.len(targets)
	assert(sample_count > 0, "Must have at least one target", loc=loc)
	assert(len(input) % sample_count == 0, "Input length must be divisible by number of targets", loc=loc)

	class_size := len(input) / sample_count

	targets_copy := builtin.make([]int, sample_count, allocator=arena_allocator())

	for target, i in targets {
		assert(target >= 0 && target < class_size, "Target is out of bounds", loc=loc)
		targets_copy[i] = target
	}

	probabilities := zeros(len(input), loc=loc)
	output         = zeros(sample_count, loc=loc)

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
relu :: proc(input: Array, loc := #caller_location) ->(output: Array) {
	output = zeros(len(input), loc=loc)

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
sigmoid :: proc(input: Array, loc := #caller_location) -> (output: Array) {
	output = zeros(len(input), loc=loc)

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
gelu :: proc(input: Array, loc := #caller_location) -> (output: Array) {
	output = zeros(len(input), loc=loc)

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
silu :: proc(input: Array, loc := #caller_location) -> (output: Array) {
	output = zeros(len(input), loc=loc)

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
tanh :: proc(input: Array, loc := #caller_location) -> (output: Array) {
	output = zeros(len(input), loc=loc)

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