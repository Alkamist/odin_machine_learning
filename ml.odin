package machine_learning

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"

MAX_OPERATIONS          :: 4096
MAX_TENSOR_RANK         :: 6
BACKEND_BUFFER_MAX_SIZE :: 16

OP_ARENA_DEFAULT_SIZE   :: 1 * 1024 * 1024

Data_Type :: enum u8 {
	F32,
}

@(require_results)
data_type_size :: #force_inline proc(t: Data_Type) -> int {
	switch t {
	case .F32: return size_of(f32)
	}
	return 0
}

Buffer_Kind :: enum u8 {
	Data,
	Gradient,
	Adam_M,
	Adam_V,
}
Buffer_Set :: bit_set[Buffer_Kind; u8]

Backend_Buffer :: [BACKEND_BUFFER_MAX_SIZE]byte

Tensor :: struct {
	vtable:  ^Backend_VTable,
	buffers: [Buffer_Kind]Backend_Buffer,
	type:    Data_Type,
	shape:   [MAX_TENSOR_RANK]int,
	rank:    int,
	count:   int,
}

Backend_VTable :: struct {
	init:     proc(ctx: ^Context, size: int, loc: runtime.Source_Code_Location),
	destroy:  proc(ctx: ^Context, loc: runtime.Source_Code_Location),

	clear:    proc(loc: runtime.Source_Code_Location),
	forward:  proc(op: Operation, loc: runtime.Source_Code_Location),
	backward: proc(op: Operation, loc: runtime.Source_Code_Location),
	update:   proc(opt: Optimizer, t: ^Tensor, loc: runtime.Source_Code_Location),

	buffer_alloc: proc(len: int, persist: bool, loc: runtime.Source_Code_Location) -> Backend_Buffer,
	buffer_free:  proc(buffer: Backend_Buffer, loc: runtime.Source_Code_Location),
	buffer_get:   proc(buffer: Backend_Buffer, data: []f32, loc: runtime.Source_Code_Location),
	buffer_set:   proc(buffer: Backend_Buffer, data: []f32, loc: runtime.Source_Code_Location),
	buffer_copy:  proc(dst, src: Backend_Buffer, loc: runtime.Source_Code_Location),
}

Context :: struct {
	vtable:       ^Backend_VTable,
	backend_data: rawptr,

	op_arena:      mem.Arena,
	_op_arena_buf: []byte,

	operation_count: int,
	operations:      [MAX_OPERATIONS]Operation,

	previous_ctx: ^Context,
}

@(thread_local)
_current_ctx: ^Context

@(require_results)
context_create :: proc(size: int, vtable: ^Backend_VTable, allocator := context.allocator, loc := #caller_location) -> ^Context {
	assert(vtable != nil, "Backend vtable must not be nil", loc=loc)

	ctx, ctx_err := builtin.new(Context, allocator=allocator, loc=loc)
	assert(ctx_err == nil, "Failed to allocate Context", loc=loc)

	ctx.vtable = vtable

	op_arena_buf, op_arena_err := builtin.make([]byte, OP_ARENA_DEFAULT_SIZE, allocator=allocator, loc=loc)
	assert(op_arena_err == nil, "Failed to allocate op-metadata arena", loc=loc)
	ctx._op_arena_buf = op_arena_buf
	mem.arena_init(&ctx.op_arena, op_arena_buf)

	vtable.init(ctx, size, loc)

	return ctx
}

context_destroy :: proc(ctx: ^Context, allocator := context.allocator, loc := #caller_location) {
	assert(_current_ctx != ctx, "context_destroy called on the active context", loc=loc)

	if ctx.vtable != nil && ctx.vtable.destroy != nil {
		ctx.vtable.destroy(ctx, loc)
	}

	if ctx._op_arena_buf != nil {
		builtin.delete(ctx._op_arena_buf, allocator=allocator, loc=loc)
	}
	builtin.free(ctx, allocator=allocator, loc=loc)
}

context_begin :: proc(ctx: ^Context) {
	ctx.previous_ctx = _current_ctx
	_current_ctx     = ctx
}

context_end :: proc() {
	assert(_current_ctx != nil, "context_end called with no active context")
	_current_ctx = _current_ctx.previous_ctx
}

@(deferred_none=context_end)
context_scope :: proc(ctx: ^Context) {
	context_begin(ctx)
}

@(require_results)
current_context :: #force_inline proc(loc := #caller_location) -> ^Context {
	assert(_current_ctx != nil, "Called current_context with no active context", loc=loc)
	return _current_ctx
}

clear :: proc(loc := #caller_location) {
	assert(_current_ctx != nil && _current_ctx.vtable != nil, "Did you forget to call context_create or context_scope?", loc=loc)

	_current_ctx.vtable.clear(loc)
	mem.arena_free_all(&_current_ctx.op_arena)
	_current_ctx.operation_count = 0
}

op_arena_allocator :: proc() -> mem.Allocator {
	return mem.arena_allocator(&_current_ctx.op_arena)
}

@(require_results)
len :: #force_inline proc(t: Tensor) -> int {
	return t.count
}

@(require_results)
shape_element_count :: proc(shape: []int) -> int {
	n := 1
	for d in shape {
		n *= d
	}
	return n
}

DEFAULT_ACTIVATION_BUFFERS :: Buffer_Set{.Data, .Gradient}
DEFAULT_PARAMETER_BUFFERS  :: Buffer_Set{.Data, .Gradient, .Adam_M, .Adam_V}

@(require_results)
alloc :: proc(shape: []int, persistent: bool, buffers: Buffer_Set, loc := #caller_location) -> (t: Tensor) {
	assert(_current_ctx != nil && _current_ctx.vtable != nil, "Did you forget to call context_create / context_scope?", loc=loc)
	assert(builtin.len(shape) > 0, "Tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Tensor rank exceeds MAX_TENSOR_RANK", loc=loc)

	element_count := shape_element_count(shape)
	assert(element_count > 0, "Tensor element count must be positive", loc=loc)

	t.vtable = _current_ctx.vtable
	t.type   = .F32
	t.count  = element_count
	t.rank   = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "Tensor dimension must be positive", loc=loc)
		t.shape[i] = d
	}

	for kind in Buffer_Kind {
		if kind in buffers {
			t.buffers[kind] = t.vtable.buffer_alloc(element_count, persistent, loc)
		}
	}

	return
}

@(require_results)
zeros :: proc(shape: []int, loc := #caller_location) -> (t: Tensor) {
	return alloc(shape, persistent=false, buffers=DEFAULT_ACTIVATION_BUFFERS, loc=loc)
}

@(require_results)
zeros_like :: proc(src: Tensor, loc := #caller_location) -> Tensor {
	shape := src.shape
	return zeros(shape[:src.rank], loc=loc)
}

@(require_results)
tensor :: proc(data: []f32, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(data) > 0, "Length must be at least 1", loc=loc)
	shape := [1]int{builtin.len(data)}
	t = zeros(shape[:], loc=loc)
	t.vtable.buffer_set(t.buffers[.Data], data, loc)
	return
}

@(require_results)
scalar :: proc(value: f32, loc := #caller_location) -> (t: Tensor) {
	shape := [1]int{1}
	t = zeros(shape[:], loc=loc)
	src := [1]f32{value}
	t.vtable.buffer_set(t.buffers[.Data], src[:], loc)
	return
}

@(require_results)
_zeros_drop_last :: proc(src: Tensor, loc := #caller_location) -> Tensor {
	if src.rank <= 1 {
		shape := [1]int{1}
		return zeros(shape[:], loc=loc)
	}
	shape := src.shape
	return zeros(shape[:src.rank - 1], loc=loc)
}

@(require_results)
_zeros_replace_trailing :: proc(src: Tensor, new_trailing: int, loc := #caller_location) -> Tensor {
	new_shape: [MAX_TENSOR_RANK]int = src.shape
	new_shape[src.rank - 1] = new_trailing
	return zeros(new_shape[:src.rank], loc=loc)
}

@(require_results)
_leading_count :: proc(t: Tensor) -> int {
	n := 1
	for i in 0 ..< t.rank - 1 {
		n *= t.shape[i]
	}
	return n
}

@(require_results)
reshape :: proc(src: Tensor, shape: []int, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(shape) > 0, "Tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "Tensor rank exceeds MAX_TENSOR_RANK", loc=loc)
	assert(shape_element_count(shape) == len(src), "Reshape element count mismatch", loc=loc)

	t.vtable  = src.vtable
	t.buffers = src.buffers
	t.type    = src.type
	t.count   = src.count
	t.rank    = builtin.len(shape)
	for d, i in shape {
		t.shape[i] = d
	}
	return
}

@(require_results)
make :: proc(shape: []int, loc := #caller_location) -> (t: Tensor, err: mem.Allocator_Error) #optional_allocator_error {
	t = alloc(shape, persistent=true, buffers=DEFAULT_PARAMETER_BUFFERS, loc=loc)
	return t, nil
}

destroy :: proc(t: Tensor, loc := #caller_location) {
	if t.vtable == nil { return }
	for kind in Buffer_Kind {
		t.vtable.buffer_free(t.buffers[kind], loc)
	}
}

copy :: proc(dst, src: Tensor, loc := #caller_location) {
	assert(len(dst) == len(src), "Tensor lengths must be equal", loc=loc)
	assert(dst.vtable == src.vtable, "Tensor copy across backends not supported", loc=loc)
	for kind in Buffer_Kind {
		dst.vtable.buffer_copy(dst.buffers[kind], src.buffers[kind], loc)
	}
}

get_data :: proc(t: Tensor, data: []f32, loc := #caller_location) {
	t.vtable.buffer_get(t.buffers[.Data], data, loc)
}

fill_normal :: proc(t: Tensor, mean, std: f32, loc := #caller_location) {
	n   := len(t)
	buf := builtin.make([]f32, n, allocator=context.temp_allocator)
	for i in 0 ..< n {
		buf[i] = rand.float32_normal(mean, std)
	}
	t.vtable.buffer_set(t.buffers[.Data], buf, loc)
}

fill_value :: proc(t: Tensor, value: f32, loc := #caller_location) {
	n   := len(t)
	buf := builtin.make([]f32, n, allocator=context.temp_allocator)
	for i in 0 ..< n {
		buf[i] = value
	}
	t.vtable.buffer_set(t.buffers[.Data], buf, loc)
}

he_initialization :: proc(t: Tensor, input_features: int) {
	fill_normal(t, 0, math.sqrt(2 / f32(input_features)))
}

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

update :: proc(opt: Optimizer, t: Tensor, loc := #caller_location) {
	t := t
	t.vtable.update(opt, &t, loc)
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

append_operation :: proc(op: Operation, loc := #caller_location) {
	assert(_current_ctx.operation_count < MAX_OPERATIONS, "Maximum operations exceeded, did you forget to call clear?", loc=loc)
	_current_ctx.operations[_current_ctx.operation_count] = op
	_current_ctx.operation_count += 1
}

backward :: proc(loc := #caller_location) {
	if _current_ctx == nil || _current_ctx.operation_count <= 0 {
		return
	}

	vtable := _current_ctx.vtable

	final_op := &_current_ctx.operations[_current_ctx.operation_count - 1]

	ones := builtin.make([]f32, final_op.output.count, allocator=context.temp_allocator)
	for &v in ones {
		v = 1
	}
	vtable.buffer_set(final_op.output.buffers[.Gradient], ones, loc)

	for i := _current_ctx.operation_count - 1; i >= 0; i -= 1 {
		vtable.backward(_current_ctx.operations[i], loc)
	}
}

@(require_results)
attention :: proc(input: Tensor, head_count: int, causal := true, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 2, "attention requires a 2-D tensor [tokens, 3 * embedding]", loc=loc)

	token_count := input.shape[0]
	input_size  := input.shape[1]
	assert(input_size % 3 == 0, "Trailing dim must be divisible by 3 (for Q, K, V)", loc=loc)

	output_size := input_size / 3
	assert(output_size % head_count == 0, "Output size must be divisible by head count", loc=loc)

	head_size := output_size / head_count

	q_flat := slice_trailing(input, 0,               output_size,     loc=loc)
	k_flat := slice_trailing(input, output_size,     output_size * 2, loc=loc)
	v_flat := slice_trailing(input, output_size * 2, output_size * 3, loc=loc)

	q := reshape(q_flat, {token_count, head_count, head_size}, loc=loc)
	k := reshape(k_flat, {token_count, head_count, head_size}, loc=loc)
	v := reshape(v_flat, {token_count, head_count, head_size}, loc=loc)

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
	output        = reshape(out, {token_count, output_size}, loc=loc)

	return
}

Add :: struct {
	b:      Tensor,
	stride: int,
}

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
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Sub :: struct {
	b:      Tensor,
	stride: int,
}

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
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Mul :: struct {
	b:      Tensor,
	stride: int,
}

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
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Div :: struct {
	b:      Tensor,
	stride: int,
}

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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Transpose :: struct {
	rows: int,
}

@(require_results)
transpose :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 2, "transpose requires a 2-D tensor", loc=loc)

	rows    := input.shape[0]
	columns := input.shape[1]

	output = zeros({columns, rows}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Transpose{
			rows = rows,
		},
	}
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Select :: struct {
	indices: []int,
	size:    int,
}

@(require_results)
select :: proc(input: Tensor, indices: []int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "select input must have rank >= 1", loc=loc)

	size := 1
	for i in 1 ..< input.rank {
		size *= input.shape[i]
	}

	indices_copy := builtin.make([]int, builtin.len(indices), allocator=op_arena_allocator())
	for i in 0 ..< builtin.len(indices) {
		indices_copy[i] = indices[i]
	}

	out_shape: [MAX_TENSOR_RANK]int = input.shape
	out_shape[0] = builtin.len(indices)
	output = zeros(out_shape[:input.rank], loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Select{
			indices  = indices_copy,
			size     = size,
		}
	}
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Slice :: struct {
	start: int,
	end:   int,
}

@(require_results)
slice :: proc(input: Tensor, start, end: int, loc := #caller_location) -> (output: Tensor) {
	fmt.assertf(start >= 0 && end <= len(input) && start <= end, "Slice indices out of bounds %v:%v", start, end, loc=loc)

	output = zeros({end - start}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Slice{
			start = start,
			end   = end,
		},
	}
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Slice_Trailing :: struct {
	start, end: int,
}

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
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Concat :: struct {
	inputs: []Tensor,
}

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

	inputs_copy := builtin.make([]Tensor, builtin.len(inputs), allocator=op_arena_allocator())
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
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}

Linear :: struct {
	weight:      Tensor,
	input_size:  int,
	output_size: int,
	count:       int,
}

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
	_current_ctx.vtable.forward(op, loc)
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

@(require_results)
rope :: proc(input: Tensor, head_count: int, base: f32 = 10000, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 2, "rope requires rank >= 2", loc=loc)

	token_count := input.shape[0]
	input_size  := input.shape[input.rank - 1]
	assert(input_size % head_count == 0, "Trailing dim must be divisible by head count", loc=loc)

	head_size := input_size / head_count
	assert(head_size % 2 == 0, "Head size must be even", loc=loc)

	output = zeros_like(input, loc=loc)

	cos_cache := zeros({token_count * head_size / 2}, loc=loc)
	sin_cache := zeros({token_count * head_size / 2}, loc=loc)

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
	_current_ctx.vtable.forward(op, loc)
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

	mean := zeros({count}, loc=loc)
	rstd := zeros({count}, loc=loc)

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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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

	targets_copy := builtin.make([]int, sample_count, allocator=op_arena_allocator())
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
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

	output = zeros({batch_count, m, n}, loc=loc)

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
	_current_ctx.vtable.forward(op, loc)
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
	output = zeros({out_shape[0], out_shape[1], out_shape[2]}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Permute{axes = axes},
	}
	_current_ctx.vtable.forward(op, loc)
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
	_current_ctx.vtable.forward(op, loc)
	append_operation(op, loc=loc)

	return
}