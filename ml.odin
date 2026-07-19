package machine_learning

import "base:builtin"
import "base:intrinsics"
import "base:runtime"

import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:reflect"
import "core:sync"

MAX_OPERATIONS               :: 16384
MAX_TENSOR_RANK              :: 6
BACKEND_BUFFER_MAX_SIZE      :: 16
OPERATION_ARENA_DEFAULT_SIZE :: 16 * 1024 * 1024
K_QUANT_BLOCK_SIZE           :: 256
Q4_K_BLOCK_BYTES             :: 144
Q6_K_BLOCK_BYTES             :: 210

Data_Type :: enum u8 {
	Bf16,
	F32,
	I32,
	Q4_K,
	Q6_K,
}

@(require_results)
data_type_size :: #force_inline proc(t: Data_Type) -> int {
	switch t {
	case .Bf16: return size_of(Bf16)
	case .F32:  return size_of(f32)
	case .I32:  return size_of(i32)
	case .Q4_K: return 0 // packed; see `_data_byte_count`
	case .Q6_K: return 0 // packed; see `_data_byte_count`
	}
	return 0
}

@(require_results)
_data_byte_count :: #force_inline proc(t: Data_Type, element_count: int, loc := #caller_location) -> int {
	#partial switch t {
	case .Q4_K:
		assert(element_count % K_QUANT_BLOCK_SIZE == 0, "Q4_K tensor element count must be a multiple of 256", loc=loc)
		return (element_count / K_QUANT_BLOCK_SIZE) * Q4_K_BLOCK_BYTES
	case .Q6_K:
		assert(element_count % K_QUANT_BLOCK_SIZE == 0, "Q6_K tensor element count must be a multiple of 256", loc=loc)
		return (element_count / K_QUANT_BLOCK_SIZE) * Q6_K_BLOCK_BYTES
	}
	return element_count * data_type_size(t)
}

Bf16 :: distinct u16

@(require_results)
bf16_from_f32 :: #force_inline proc "contextless" (x: f32) -> Bf16 {
	bits := transmute(u32)x
	if bits & 0x7fff_ffff > 0x7f80_0000 {
		return Bf16(0x7fc0)
	}
	rounded := bits + 0x7fff + ((bits >> 16) & 1)
	return Bf16(rounded >> 16)
}

@(require_results)
bf16_to_f32 :: #force_inline proc "contextless" (x: Bf16) -> f32 {
	return transmute(f32)(u32(x) << 16)
}

Buffer_Kind :: enum u8 {
	Data,
	Gradient,
}
Buffer_Set :: bit_set[Buffer_Kind; u8]

Backend_Buffer :: [BACKEND_BUFFER_MAX_SIZE]byte

Tensor :: struct {
	backend: ^Backend,
	buffers: [Buffer_Kind]Backend_Buffer,
	type:    Data_Type,
	shape:   [MAX_TENSOR_RANK]int,
	rank:    int,
	count:   int,
}

Backend :: struct #all_or_none {
	clear:    proc(loc: runtime.Source_Code_Location),
	forward:  proc(op: ^Operation, loc: runtime.Source_Code_Location),
	backward: proc(op: Operation, loc: runtime.Source_Code_Location),
	update:   proc(opt: Optimizer, t: Tensor, m, v: Backend_Buffer, loc: runtime.Source_Code_Location),

	buffer_alloc: proc(byte_count: int, kind: Buffer_Kind, persist: bool, loc: runtime.Source_Code_Location) -> Backend_Buffer,
	buffer_free:  proc(buffer: Backend_Buffer, loc: runtime.Source_Code_Location),
	buffer_get:   proc(buffer: Backend_Buffer, dst: []byte, loc: runtime.Source_Code_Location),
	buffer_set:   proc(buffer: Backend_Buffer, src: []byte, loc: runtime.Source_Code_Location),
	buffer_copy:  proc(dst, src: Backend_Buffer, loc: runtime.Source_Code_Location),

	buffer_sq_sum_accumulate: proc(buffer: Backend_Buffer, count: int, accumulator: Backend_Buffer, loc: runtime.Source_Code_Location),
	buffer_scale:             proc(buffer: Backend_Buffer, count: int, scale: f32, loc: runtime.Source_Code_Location),

	forward_ops:  Operation_Set,
	backward_ops: Operation_Set,
}

Context :: struct {
	backend: ^Backend,

	op_arena:      mem.Arena,
	_op_arena_buf: []byte,

	operation_count: int,
	operations:      [MAX_OPERATIONS]Operation,

	training: bool,

	owner_thread_id: int,

	grad_norm_accumulator: Backend_Buffer,
}

@(thread_local)
_current_ctx: ^Context

_context_init :: proc(ctx: ^Context, backend: ^Backend, allocator: mem.Allocator, loc: runtime.Source_Code_Location) {
	_assert_operation_variant_order()

	ctx.backend  = backend
	ctx.training = true

	op_arena_buf, op_arena_err := builtin.make([]byte, OPERATION_ARENA_DEFAULT_SIZE, allocator=allocator, loc=loc)
	assert(op_arena_err == nil, "failed to allocate op-metadata arena", loc=loc)
	ctx._op_arena_buf = op_arena_buf
	mem.arena_init(&ctx.op_arena, op_arena_buf)
}

_context_destroy :: proc(ctx: ^Context, loc: runtime.Source_Code_Location) {
	assert(_current_ctx != ctx, "cannot destroy the active context", loc=loc)
	assert(sync.atomic_load(&ctx.owner_thread_id) == 0, "cannot destroy a context that is active on a thread", loc=loc)
	builtin.delete(ctx._op_arena_buf, loc=loc)
}

@(require_results)
context_begin :: proc(ctx: ^Context, loc := #caller_location) -> (previous: ^Context) {
	thread_id := int(sync.current_thread_id())
	owner, exchanged := sync.atomic_compare_exchange_strong(&ctx.owner_thread_id, 0, thread_id)
	fmt.assertf(exchanged || owner == thread_id, "context is active on thread %v; a context may only be used by one thread at a time", owner, loc=loc)
	previous     = _current_ctx
	_current_ctx = ctx
	return
}

context_end :: proc(previous: ^Context) {
	if _current_ctx != nil && _current_ctx != previous {
		sync.atomic_store(&_current_ctx.owner_thread_id, 0)
	}
	_current_ctx = previous
}

@(deferred_out=context_end)
context_scope :: proc(ctx: ^Context, loc := #caller_location) -> ^Context {
	return context_begin(ctx, loc=loc)
}

@(require_results)
current_context :: #force_inline proc(loc := #caller_location) -> ^Context {
	assert(_current_ctx != nil, "no active context", loc=loc)
	return _current_ctx
}

clear :: proc(training := false, loc := #caller_location) {
	assert(_current_ctx != nil, "no active context", loc=loc)

	_current_ctx.training = training
	_current_ctx.backend.clear(loc)

	mem.arena_free_all(&_current_ctx.op_arena)
	_current_ctx.operation_count = 0
}

@(require_results)
is_training :: #force_inline proc(loc := #caller_location) -> bool {
	assert(_current_ctx != nil, "no active context", loc=loc)
	return _current_ctx.training
}

op_arena_allocator :: proc() -> mem.Allocator {
	return mem.arena_allocator(&_current_ctx.op_arena)
}

@(require_results)
_op_arena_make :: proc($T: typeid, count: int, loc := #caller_location) -> []T {
	slice, err := builtin.make([]T, count, allocator=op_arena_allocator(), loc=loc)
	fmt.assertf(
		err == nil && builtin.len(slice) == count, 
		"op arena exhausted allocating %d x %v (%d bytes) - raise OP_ARENA_DEFAULT_SIZE", 
		count, typeid_of(T), count * size_of(T), loc=loc,
	)
	return slice
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
DEFAULT_PARAMETER_BUFFERS  :: Buffer_Set{.Data, .Gradient}

@(require_results)
buffer_dtype :: #force_inline proc(tensor_type: Data_Type, kind: Buffer_Kind) -> Data_Type {
	if kind == .Data {
		return tensor_type
	}
	return .F32
}

@(require_results)
alloc :: proc(type: Data_Type, shape: []int, persistent := false, buffers := DEFAULT_ACTIVATION_BUFFERS, loc := #caller_location) -> (t: Tensor) {
	assert(_current_ctx != nil, "no active context", loc=loc)
	assert(builtin.len(shape) > 0, "tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "tensor rank exceeds MAX_TENSOR_RANK", loc=loc)

	element_count := shape_element_count(shape)
	assert(element_count > 0, "tensor element count must be positive", loc=loc)

	assert(type != .I32 || .Gradient not_in buffers, "I32 tensors cannot have gradient buffers", loc=loc)

	t.backend = _current_ctx.backend
	t.type    = type
	t.count   = element_count
	t.rank    = builtin.len(shape)
	for d, i in shape {
		assert(d > 0, "tensor dimension must be positive", loc=loc)
		t.shape[i] = d
	}

	for kind in Buffer_Kind {
		if kind in buffers {
			kind_type  := buffer_dtype(type, kind)
			byte_count := _data_byte_count(kind_type, element_count)
			assert(byte_count > 0, "tensor byte count must be positive", loc=loc)
			byte_count = (byte_count + 3) & ~int(3)
			t.buffers[kind] = t.backend.buffer_alloc(byte_count, kind, persistent, loc)
		}
	}

	return
}

@(require_results)
zeros :: proc(type: Data_Type, shape: []int, loc := #caller_location) -> (t: Tensor) {
	buffers := DEFAULT_ACTIVATION_BUFFERS
	if _current_ctx != nil && !_current_ctx.training {
		buffers = Buffer_Set{.Data}
	}
	if type == .I32 {
		buffers -= {.Gradient}
	}
	return alloc(type, shape, persistent=false, buffers=buffers, loc=loc)
}

@(require_results)
zeros_like :: proc(src: Tensor, loc := #caller_location) -> Tensor {
	shape := src.shape
	return zeros(src.type, shape[:src.rank], loc=loc)
}

@(require_results)
scratch :: proc(type: Data_Type, shape: []int, loc := #caller_location) -> Tensor {
	return alloc(type, shape, persistent=false, buffers={.Data}, loc=loc)
}

tensor :: proc{_tensor_flat, _tensor_shaped, _tensor_flat_i32, _tensor_shaped_i32}

@(require_results)
_tensor_flat :: proc(data: []f32, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(data) > 0, "length must be at least 1", loc=loc)
	shape := [1]int{builtin.len(data)}
	t = zeros(.F32, shape[:], loc=loc)
	t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(data), loc)
	return
}

@(require_results)
_tensor_shaped :: proc(data: []f32, shape: []int, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(data) > 0, "length must be at least 1", loc=loc)
	assert(shape_element_count(shape) == builtin.len(data), "tensor shape element count must match data length", loc=loc)
	t = zeros(.F32, shape, loc=loc)
	t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(data), loc)
	return
}

@(require_results)
_tensor_flat_i32 :: proc(data: []i32, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(data) > 0, "length must be at least 1", loc=loc)
	shape := [1]int{builtin.len(data)}
	t = zeros(.I32, shape[:], loc=loc)
	t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(data), loc)
	return
}

@(require_results)
_tensor_shaped_i32 :: proc(data: []i32, shape: []int, loc := #caller_location) -> (t: Tensor) {
	assert(builtin.len(data) > 0, "length must be at least 1", loc=loc)
	assert(shape_element_count(shape) == builtin.len(data), "tensor shape element count must match data length", loc=loc)
	t = zeros(.I32, shape, loc=loc)
	t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(data), loc)
	return
}

@(require_results)
scalar :: proc(type: Data_Type, value: f32, persistent := false, loc := #caller_location) -> (t: Tensor) {
	if persistent {
		t = alloc(type, {1}, persistent=true, buffers={.Data}, loc=loc)
	} else {
		shape := [1]int{1}
		t = zeros(type, shape[:], loc=loc)
	}
	switch type {
	case .F32:
		src := [1]f32{value}
		t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(src[:]), loc)
	case .Bf16:
		src := [1]Bf16{bf16_from_f32(value)}
		t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(src[:]), loc)
	case .I32, .Q4_K, .Q6_K:
		fmt.panicf("scalar does not support dtype %v", type, loc=loc)
	}
	return
}

@(require_results)
_zeros_drop_last :: proc(src: Tensor, loc := #caller_location) -> Tensor {
	if src.rank <= 1 {
		shape := [1]int{1}
		return zeros(src.type, shape[:], loc=loc)
	}
	shape := src.shape
	return zeros(src.type, shape[:src.rank - 1], loc=loc)
}

@(require_results)
_zeros_replace_trailing :: proc(src: Tensor, new_trailing: int, loc := #caller_location) -> Tensor {
	new_shape: [MAX_TENSOR_RANK]int = src.shape
	new_shape[src.rank - 1] = new_trailing
	return zeros(src.type, new_shape[:src.rank], loc=loc)
}

@(require_results)
_zeros_replace_leading :: proc(src: Tensor, new_leading: int, loc := #caller_location) -> Tensor {
	new_shape: [MAX_TENSOR_RANK]int = src.shape
	new_shape[0] = new_leading
	return zeros(src.type, new_shape[:src.rank], loc=loc)
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
	assert(builtin.len(shape) > 0, "tensor must have at least one dimension", loc=loc)
	assert(builtin.len(shape) <= MAX_TENSOR_RANK, "tensor rank exceeds MAX_TENSOR_RANK", loc=loc)
	assert(shape_element_count(shape) == len(src), "reshape element count mismatch", loc=loc)

	t.backend = src.backend
	t.buffers = src.buffers
	t.type    = src.type
	t.count   = src.count
	t.rank    = builtin.len(shape)
	for d, i in shape {
		t.shape[i] = d
	}
	return
}

destroy :: proc(t: Tensor, loc := #caller_location) {
	if t.backend == nil { return }
	for kind in Buffer_Kind {
		t.backend.buffer_free(t.buffers[kind], loc)
	}
}

copy :: proc(dst, src: Tensor, loc := #caller_location) {
	assert(dst.type == src.type, "tensor types must be equal", loc=loc)
	assert(dst.rank == src.rank && dst.shape == src.shape, "tensor shapes must be equal", loc=loc)
	assert(dst.backend == src.backend, "tensor copy across backends not supported", loc=loc)
	for kind in Buffer_Kind {
		dst.backend.buffer_copy(dst.buffers[kind], src.buffers[kind], loc)
	}
}

get_data :: proc{_get_data_f32, _get_data_i32}

_get_data_f32 :: proc(t: Tensor, data: []f32, loc := #caller_location) {
	assert(t.type == .F32, "get_data with []f32 requires an F32 tensor", loc=loc)
	t.backend.buffer_get(t.buffers[.Data], mem.slice_to_bytes(data), loc)
}

_get_data_i32 :: proc(t: Tensor, data: []i32, loc := #caller_location) {
	assert(t.type == .I32, "get_data with []i32 requires an I32 tensor", loc=loc)
	t.backend.buffer_get(t.buffers[.Data], mem.slice_to_bytes(data), loc)
}

set_data :: proc{_set_data_f32, _set_data_i32}

_set_data_f32 :: proc(t: Tensor, data: []f32, loc := #caller_location) {
	assert(t.type == .F32, "set_data with []f32 requires an F32 tensor", loc=loc)
	t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(data), loc)
}

_set_data_i32 :: proc(t: Tensor, data: []i32, loc := #caller_location) {
	assert(t.type == .I32, "set_data with []i32 requires an I32 tensor", loc=loc)
	t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(data), loc)
}

argmax :: proc(t: Tensor, results: []int, loc := #caller_location) {
	assert(t.type == .F32, "argmax requires an F32 tensor", loc=loc)
	assert(t.rank >= 1, "argmax input must have rank >= 1", loc=loc)

	trailing := t.shape[t.rank - 1]
	leading  := _leading_count(t)
	assert(builtin.len(results) == leading, "argmax results length must equal the leading-dim product", loc=loc)

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	buffer := builtin.make([]f32, len(t), allocator=context.temp_allocator, loc=loc)
	get_data(t, buffer, loc)

	for row in 0 ..< leading {
		values := buffer[row * trailing:][:trailing]
		best   := 0
		for i in 1 ..< trailing {
			if values[i] > values[best] {
				best = i
			}
		}
		results[row] = best
	}
}

get_gradient :: proc(t: Tensor, data: []f32, loc := #caller_location) {
	assert(t.type == .F32, "get_gradient with []f32 requires an F32 tensor", loc=loc)
	t.backend.buffer_get(t.buffers[.Gradient], mem.slice_to_bytes(data), loc)
}

get_bytes :: proc(t: Tensor, kind: Buffer_Kind, dst: []byte, loc := #caller_location) {
	t.backend.buffer_get(t.buffers[kind], dst, loc)
}

set_bytes :: proc(t: Tensor, kind: Buffer_Kind, src: []byte, loc := #caller_location) {
	t.backend.buffer_set(t.buffers[kind], src, loc)
}

@(require_results)
has_buffer :: #force_inline proc(t: Tensor, kind: Buffer_Kind) -> bool {
	return t.buffers[kind] != Backend_Buffer{}
}

@(require_results)
has_gradient :: #force_inline proc(t: Tensor) -> bool {
	return t.buffers[.Gradient] != Backend_Buffer{}
}

@(require_results)
buffer_byte_count :: proc(t: Tensor, kind: Buffer_Kind) -> int {
	return _data_byte_count(buffer_dtype(t.type, kind), t.count)
}

fill_normal :: proc(t: Tensor, mean, std: f32, loc := #caller_location) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	n := len(t)
	switch t.type {
	case .F32:
		buf := builtin.make([]f32, n, allocator=context.temp_allocator)
		for i in 0 ..< n {
			buf[i] = rand.float32_normal(mean, std)
		}
		t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(buf), loc)
	case .Bf16:
		buf := builtin.make([]Bf16, n, allocator=context.temp_allocator)
		for i in 0 ..< n {
			buf[i] = bf16_from_f32(rand.float32_normal(mean, std))
		}
		t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(buf), loc)
	case .I32, .Q4_K, .Q6_K:
		fmt.panicf("fill_normal does not support dtype %v", t.type, loc=loc)
	}
}

fill_value :: proc(t: Tensor, value: f32, loc := #caller_location) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	n := len(t)
	switch t.type {
	case .F32:
		buf := builtin.make([]f32, n, allocator=context.temp_allocator)
		for i in 0 ..< n {
			buf[i] = value
		}
		t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(buf), loc)
	case .Bf16:
		value_bf := bf16_from_f32(value)
		buf      := builtin.make([]Bf16, n, allocator=context.temp_allocator)
		for i in 0 ..< n {
			buf[i] = value_bf
		}
		t.backend.buffer_set(t.buffers[.Data], mem.slice_to_bytes(buf), loc)
	case .I32, .Q4_K, .Q6_K:
		fmt.panicf("fill_value does not support dtype %v", t.type, loc=loc)
	}
}

he_initialization :: proc(t: Tensor, input_features: int) {
	fill_normal(t, 0, math.sqrt(2 / f32(input_features)))
}

xavier_initialization :: proc(t: Tensor, input_features, output_features: int) {
	fill_normal(t, 0, math.sqrt(2 / f32(input_features + output_features)))
}

Optimizer_State :: struct {
	m: Backend_Buffer,
	v: Backend_Buffer,
}

Optimizer_Kind :: enum {
	Adam_W,
}

Optimizer :: struct {
	kind: Optimizer_Kind,

	learning_rate:      f32,
	beta1:              f32,
	beta2:              f32,
	epsilon:            f32,
	weight_decay:       f32,
	accumulation_steps: int,

	iteration:            u64,
	accumulation_counter: int,

	bias_correction1: f32,
	bias_correction2: f32,

	backend: ^Backend,
	state:   map[Backend_Buffer]Optimizer_State,
}

@(require_results)
optimizer_make :: proc(
	learning_rate:      f32 = 0.001,
	beta1:              f32 = 0.9,
	beta2:              f32 = 0.999,
	epsilon:            f32 = 1e-8,
	weight_decay:       f32 = 0,
	accumulation_steps: int = 1,
	kind:               Optimizer_Kind = .Adam_W,
	loc := #caller_location,
) -> (opt: Optimizer) {
	assert(accumulation_steps >= 1, "accumulation_steps must be at least 1", loc=loc)
	opt = {
		kind               = kind,
		learning_rate      = learning_rate,
		beta1              = beta1,
		beta2              = beta2,
		epsilon            = epsilon,
		weight_decay       = weight_decay,
		accumulation_steps = accumulation_steps,
	}
	return
}

@(require_results)
_optimizer_state :: proc(opt: ^Optimizer, t: Tensor, loc := #caller_location) -> Optimizer_State {
	key := t.buffers[.Data]
	if existing, ok := opt.state[key]; ok {
		return existing
	}

	if opt.state == nil {
		opt.state = builtin.make(map[Backend_Buffer]Optimizer_State)
	}
	opt.backend = t.backend

	byte_count := _data_byte_count(.F32, t.count)
	byte_count = (byte_count + 3) & ~int(3)

	state := Optimizer_State{
		m = t.backend.buffer_alloc(byte_count, .Gradient, true, loc),
		v = t.backend.buffer_alloc(byte_count, .Gradient, true, loc),
	}
	opt.state[key] = state
	return state
}

@(require_results)
_optimizer_state_lookup :: proc(opt: ^Optimizer, t: Tensor) -> (Optimizer_State, bool) {
	if opt == nil {
		return {}, false
	}
	state, ok := opt.state[t.buffers[.Data]]
	return state, ok
}

optimizer_destroy :: proc(opt: ^Optimizer, loc := #caller_location) {
	if opt.backend == nil {
		return
	}
	for _, state in opt.state {
		opt.backend.buffer_free(state.m, loc)
		opt.backend.buffer_free(state.v, loc)
	}
	builtin.delete(opt.state)
	opt.state   = nil
	opt.backend = nil
}

@(require_results)
optimizer_step :: proc(opt: ^Optimizer) -> bool {
	opt.accumulation_counter += 1
	if opt.accumulation_counter < opt.accumulation_steps {
		return false
	}
	opt.accumulation_counter = 0

	opt.iteration += 1

	opt.bias_correction1 = 1 - math.pow(opt.beta1, f32(opt.iteration))
	opt.bias_correction2 = 1 - math.pow(opt.beta2, f32(opt.iteration))

	return true
}

update :: proc(opt: ^Optimizer, t: Tensor, loc := #caller_location) {
	state := _optimizer_state(opt, t, loc)
	t.backend.update(opt^, t, state.m, state.v, loc)
}

clip_gradient_norm :: proc(r: ^Registry, max_norm: f32, loc := #caller_location) -> (norm: f32) {
	trainable_count := 0
	for parameter in r.parameters {
		if .Train in parameter.flags {
			trainable_count += 1
		}
	}
	if trainable_count == 0 {
		return
	}

	ctx     := current_context(loc=loc)
	backend := ctx.backend

	if ctx.grad_norm_accumulator == (Backend_Buffer{}) {
		ctx.grad_norm_accumulator = backend.buffer_alloc(size_of(f64), .Gradient, true, loc)
	}
	accumulator := ctx.grad_norm_accumulator

	zero_bytes: [size_of(f64)]byte
	backend.buffer_set(accumulator, zero_bytes[:], loc)

	for parameter in r.parameters {
		if .Train not_in parameter.flags {
			continue
		}
		assert(parameter.tensor.buffers[.Gradient] != Backend_Buffer{}, "clip_gradient_norm requires parameters with a gradient buffer", loc=loc)
		backend.buffer_sq_sum_accumulate(parameter.tensor.buffers[.Gradient], parameter.tensor.count, accumulator, loc)
	}

	total_sq: [1]f64
	backend.buffer_get(accumulator, mem.slice_to_bytes(total_sq[:]), loc)

	norm = f32(math.sqrt(total_sq[0]))
	if norm <= max_norm || norm == 0 {
		return
	}

	scale := max_norm / norm
	for parameter in r.parameters {
		if .Train not_in parameter.flags {
			continue
		}
		backend.buffer_scale(parameter.tensor.buffers[.Gradient], parameter.tensor.count, scale, loc)
	}
	return
}

@(require_results)
linear_schedule :: proc(step, total_steps: int, start_value, end_value: f32) -> f32 {
	if total_steps <= 0 {
		return end_value
	}
	t := builtin.clamp(f32(step) / f32(total_steps), 0, 1)
	return start_value + (end_value - start_value) * t
}

@(require_results)
warmup_cosine_schedule :: proc(step, total_steps, warmup_steps: int, peak_value, final_value: f32) -> f32 {
	if warmup_steps > 0 && step < warmup_steps {
		return peak_value * f32(step) / f32(warmup_steps)
	}
	if total_steps <= warmup_steps {
		return final_value
	}
	t := builtin.clamp(f32(step - warmup_steps) / f32(total_steps - warmup_steps), 0, 1)
	return final_value + (peak_value - final_value) * 0.5 * (1 + math.cos(math.PI * t))
}

Operation_Variant :: union {
	Add,
	Sub,
	Mul,
	Div,
	Exp,
	Sqrt,
	Clamp,
	Min,
	Max,
	Mean,
	Sum,
	Max_Reduce,
	Im2col,
	Max_Pool2d,
	Avg_Pool2d,
	Transpose,
	Select,
	Slice,
	Slice_Trailing,
	Slice_Leading,
	Concat,
	Linear,
	Linear_Q4_K,
	Linear_Q4_K_Gate_Up_Geglu,
	Linear_Q6_K,
	Rope,
	Layernorm,
	Rmsnorm,
	Rmsnorm_Rope,
	Rmsnorm_Rope_Write_Cache,
	Add_Rmsnorm,
	Softmax,
	Entropy,
	Log_Softmax,
	Mean_Squared_Error,
	Smooth_L1,
	Cross_Entropy,
	Relu,
	Sigmoid,
	Gelu,
	Gelu_Mul,
	Silu,
	Tanh,
	Batched_Matmul,
	Permute,
	Causal_Mask,
	Attention,
	Attention_Cache,
	Cast,
	Lerp_Assign,
	Accumulate_Mean,
}

Operation_Kind :: enum {
	Add,
	Sub,
	Mul,
	Div,
	Exp,
	Sqrt,
	Clamp,
	Min,
	Max,
	Mean,
	Sum,
	Max_Reduce,
	Im2col,
	Max_Pool2d,
	Avg_Pool2d,
	Transpose,
	Select,
	Slice,
	Slice_Trailing,
	Slice_Leading,
	Concat,
	Linear,
	Linear_Q4_K,
	Linear_Q4_K_Gate_Up_Geglu,
	Linear_Q6_K,
	Rope,
	Layernorm,
	Rmsnorm,
	Rmsnorm_Rope,
	Rmsnorm_Rope_Write_Cache,
	Add_Rmsnorm,
	Softmax,
	Entropy,
	Log_Softmax,
	Mean_Squared_Error,
	Smooth_L1,
	Cross_Entropy,
	Relu,
	Sigmoid,
	Gelu,
	Gelu_Mul,
	Silu,
	Tanh,
	Batched_Matmul,
	Permute,
	Causal_Mask,
	Attention,
	Attention_Cache,
	Cast,
	Lerp_Assign,
	Accumulate_Mean,
}
Operation_Set :: bit_set[Operation_Kind]

OPERATION_SET_ALL :: ~Operation_Set{}

#assert(builtin.len(Operation_Kind) == intrinsics.type_union_variant_count(Operation_Variant))

_assert_operation_variant_order :: proc() {
	union_info := runtime.type_info_base(type_info_of(Operation_Variant)).variant.(runtime.Type_Info_Union)
	enum_info  := runtime.type_info_base(type_info_of(Operation_Kind)).variant.(runtime.Type_Info_Enum)
	for variant_type, i in union_info.variants {
		named := variant_type.variant.(runtime.Type_Info_Named)
		fmt.assertf(named.name == enum_info.names[i], "Operation_Variant[%v] is %v but Operation_Kind[%v] is %v; the two lists must be in the same order", i, named.name, i, enum_info.names[i])
	}
}

@(require_results)
operation_kind :: proc(variant: Operation_Variant, loc := #caller_location) -> Operation_Kind {
	tag := reflect.get_union_variant_raw_tag(variant)
	assert(tag > 0, "nil Operation_Variant", loc=loc)
	return Operation_Kind(tag - 1)
}

_run_forward :: proc(op: ^Operation, loc: runtime.Source_Code_Location) {
	kind := operation_kind(op.variant)
	fmt.assertf(kind in _current_ctx.backend.forward_ops, "backend does not support %v in forward_ops", kind, loc=loc)
	_current_ctx.backend.forward(op, loc)
}

Operation :: struct {
	input:   Tensor,
	output:  Tensor,
	variant: Operation_Variant,
}

_record_forward :: proc(op: Operation, loc := #caller_location) {
	assert(_current_ctx.operation_count < MAX_OPERATIONS, "maximum operations exceeded; call clear to reset", loc=loc)
	slot := &_current_ctx.operations[_current_ctx.operation_count]
	slot^ = op
	_current_ctx.operation_count += 1
	_run_forward(slot, loc)
}

backward :: proc(loss: Tensor, loc := #caller_location) {
	if _current_ctx == nil || _current_ctx.operation_count <= 0 {
		return
	}

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	backend := _current_ctx.backend

	assert(_current_ctx.training, "backward called on an inference pass; call clear(training=true) first", loc=loc)
	assert(loss.backend == _current_ctx.backend, "loss tensor is not from the active context", loc=loc)
	assert(loss.type == .F32, "backward requires an F32 loss tensor", loc=loc)
	assert(loss.count == 1, "backward requires a scalar loss (reduce with mean first)", loc=loc)
	assert(loss.buffers[.Gradient] != Backend_Buffer{}, "loss tensor has no gradient buffer", loc=loc)

	unsupported: Operation_Set
	for i in 0 ..< _current_ctx.operation_count {
		kind := operation_kind(_current_ctx.operations[i].variant)
		if kind not_in backend.backward_ops {
			unsupported += {kind}
		}
	}
	fmt.assertf(unsupported == {}, "this backend cannot differentiate %v", unsupported, loc=loc)

	ones := builtin.make([]f32, loss.count, allocator=context.temp_allocator)
	for &v in ones {
		v = 1
	}
	backend.buffer_set(loss.buffers[.Gradient], mem.slice_to_bytes(ones), loc)

	for i := _current_ctx.operation_count - 1; i >= 0; i -= 1 {
		backend.backward(_current_ctx.operations[i], loc)
	}
}

Cast :: struct {}

Lerp_Assign :: struct {
	source: Tensor,
	alpha:  f32,
}

lerp_assign :: proc(dst, source: Tensor, alpha: f32, loc := #caller_location) {
	assert(len(dst) == len(source), "dst and source must have the same length", loc=loc)
	assert(dst.type == .F32 && source.type == .F32, "lerp_assign requires F32 tensors", loc=loc)

	op := Operation{
		input   = dst,
		output  = dst,
		variant = Lerp_Assign{source = source, alpha = alpha},
	}
	_run_forward(&op, loc)
}

Accumulate_Mean :: struct {}

accumulate_mean :: proc(dst, source: Tensor, loc := #caller_location) {
	assert(len(dst) == 1, "dst must be a length-1 scalar", loc=loc)
	assert(dst.type == .F32 && source.type == .F32, "accumulate_mean requires F32 tensors", loc=loc)

	op := Operation{
		input   = source,
		output  = dst,
		variant = Accumulate_Mean{},
	}
	_run_forward(&op, loc)
}

@(require_results)
cast_to :: proc(input: Tensor, target_type: Data_Type, loc := #caller_location) -> (output: Tensor) {
	assert(input.type == .F32 || input.type == .Bf16, "cast_to input must be F32 or Bf16", loc=loc)
	assert(target_type == .F32 || target_type == .Bf16, "cast_to target must be F32 or Bf16", loc=loc)
	shape := input.shape
	output = zeros(target_type, shape[:input.rank], loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Cast{},
	}
	_record_forward(op, loc=loc)

	return
}

Attention :: struct {
	n_q_heads:  int,
	n_kv_heads: int,
	causal:     bool,
	window:     int,
	key:        Tensor,
	value:      Tensor,

	softmax_outputs: Tensor,
	lse:             Tensor,
	d_p_scratch:     Tensor,
}

@(require_results)
attention :: proc(
	query:     Tensor,
	key:       Tensor,
	value:     Tensor,
	n_q_heads: int,
	n_kv_heads := 0,
	causal     := true,
	window     := 0,
	loc        := #caller_location,
) -> (output: Tensor) {
	kv_heads := n_kv_heads if n_kv_heads > 0 else n_q_heads

	assert(query.rank == 2, "attention query must be 2-D [tokens, n_q_heads * head_size]", loc=loc)
	assert(key.rank == 2, "attention key must be 2-D [tokens, n_kv_heads * head_size]", loc=loc)
	assert(value.rank == 2, "attention value must be 2-D [tokens, n_kv_heads * head_size]", loc=loc)

	token_count := query.shape[0]
	assert(key.shape[0] == token_count, "attention key token count must match query", loc=loc)
	assert(value.shape[0] == token_count, "attention value token count must match query", loc=loc)

	q_size  := query.shape[1]
	kv_size := key.shape[1]
	assert(value.shape[1] == kv_size, "attention key and value must have same trailing dim", loc=loc)
	assert(q_size  % n_q_heads == 0, "query trailing dim must be divisible by n_q_heads", loc=loc)
	assert(kv_size % kv_heads == 0, "key/value trailing dim must be divisible by n_kv_heads", loc=loc)

	head_size := q_size / n_q_heads
	assert(kv_size / kv_heads == head_size, "head_size must match between query and key/value", loc=loc)
	assert(n_q_heads % kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads", loc=loc)

	assert(query.type == key.type && key.type == value.type, "attention Q/K/V must share dtype", loc=loc)
	assert(query.type == .F32 || query.type == .Bf16, "attention requires F32 or Bf16 input", loc=loc)
	assert(window >= 0, "attention window must be non-negative (0 means full attention)", loc=loc)
	assert(window == 0 || causal, "attention window > 0 requires causal=true", loc=loc)

	output = zeros(query.type, {token_count, q_size}, loc=loc)

	op := Operation{
		input   = query,
		output  = output,
		variant = Attention{
			n_q_heads  = n_q_heads,
			n_kv_heads = kv_heads,
			causal     = causal,
			window     = window,
			key        = key,
			value      = value,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Attention_Cache :: struct {
	n_q_heads:      int,
	n_kv_heads:     int,
	cache_position: int,
	window:         int,

	key:     Tensor,
	value:   Tensor,
	k_cache: Tensor,
	v_cache: Tensor,
}

@(require_results)
attention_with_cache :: proc(
	query:          Tensor,
	key:            Tensor,
	value:          Tensor,
	k_cache:        Tensor,
	v_cache:        Tensor,
	cache_position: int,
	n_q_heads:      int,
	n_kv_heads      := 0,
	window          := 0,
	loc             := #caller_location,
) -> (output: Tensor) {
	kv_heads := n_kv_heads if n_kv_heads > 0 else n_q_heads

	assert(query.rank == 2, "attention_with_cache query must be 2-D [tokens, n_q_heads * head_size]", loc=loc)
	assert(key.rank == 2, "attention_with_cache key must be 2-D [tokens, n_kv_heads * head_size]", loc=loc)
	assert(value.rank == 2, "attention_with_cache value must be 2-D [tokens, n_kv_heads * head_size]", loc=loc)
	assert(k_cache.rank == 2, "attention_with_cache k_cache must be 2-D [t_max, n_kv_heads * head_size]", loc=loc)
	assert(v_cache.rank == 2, "attention_with_cache v_cache must be 2-D [t_max, n_kv_heads * head_size]", loc=loc)

	token_count := query.shape[0]
	assert(key.shape[0] == token_count, "attention_with_cache key token count must match query", loc=loc)
	assert(value.shape[0] == token_count, "attention_with_cache value token count must match query", loc=loc)

	q_size  := query.shape[1]
	kv_size := key.shape[1]
	assert(value.shape[1] == kv_size, "attention_with_cache key/value trailing dim mismatch", loc=loc)
	assert(k_cache.shape[1] == kv_size, "attention_with_cache k_cache trailing dim must match key", loc=loc)
	assert(v_cache.shape[1] == kv_size, "attention_with_cache v_cache trailing dim must match value", loc=loc)
	assert(q_size  % n_q_heads == 0, "query trailing dim must be divisible by n_q_heads", loc=loc)
	assert(kv_size % kv_heads  == 0, "key/value trailing dim must be divisible by n_kv_heads", loc=loc)

	head_size := q_size / n_q_heads
	assert(kv_size / kv_heads == head_size, "head_size must match between query and key/value", loc=loc)
	assert(n_q_heads % kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads", loc=loc)

	assert(query.type == value.type, "attention_with_cache Q/V must share dtype", loc=loc)
	assert(k_cache.type == v_cache.type, "attention_with_cache k_cache/v_cache must share dtype", loc=loc)
	assert(query.type == .F32 || query.type == .Bf16, "attention_with_cache requires F32 or Bf16 activations", loc=loc)

	t_capacity := k_cache.shape[0]
	assert(cache_position >= 0, "cache_position must be non-negative", loc=loc)
	assert(window >= 0, "attention_with_cache window must be non-negative (0 means full attention)", loc=loc)
	if window == 0 {
		assert(cache_position + token_count <= t_capacity, "cache overflow: cache_position + token_count > t_max", loc=loc)
	} else {
		assert(t_capacity >= window, "sliding cache capacity must be >= window", loc=loc)
		assert(token_count <= t_capacity, "token_count exceeds ring capacity", loc=loc)
	}

	output = zeros(query.type, {token_count, q_size}, loc=loc)

	op := Operation{
		input   = query,
		output  = output,
		variant = Attention_Cache{
			n_q_heads      = n_q_heads,
			n_kv_heads     = kv_heads,
			cache_position = cache_position,
			window         = window,
			key            = key,
			value          = value,
			k_cache        = k_cache,
			v_cache        = v_cache,
		},
	}
	_record_forward(op, loc=loc)

	return
}

_assert_float :: proc(t: Tensor, name: string, loc: runtime.Source_Code_Location) {
	fmt.assertf(t.type == .F32 || t.type == .Bf16, "%s requires F32 or Bf16 input (got %v)", name, t.type, loc=loc)
}

_assert_broadcastable :: proc(a, b: Tensor, loc: runtime.Source_Code_Location) {
	assert(a.type == b.type, "broadcast inputs must have the same dtype", loc=loc)
	if b.count == 1 {
		return
	}
	assert(b.rank <= a.rank, "broadcast: b's rank must not exceed a's", loc=loc)
	for d in 0 ..< b.rank {
		assert(b.shape[b.rank - 1 - d] == a.shape[a.rank - 1 - d], "broadcast: b's shape must equal a's trailing shape", loc=loc)
	}
}

Add :: struct {
	b: Tensor,
}

@(require_results)
add :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(a, "add", loc)
	_assert_broadcastable(a, b, loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Add{b=b},
	}
	_record_forward(op, loc=loc)

	return
}

Sub :: struct {
	b: Tensor,
}

@(require_results)
sub :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(a, "sub", loc)
	_assert_broadcastable(a, b, loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Sub{b=b},
	}
	_record_forward(op, loc=loc)

	return
}

Mul :: struct {
	b: Tensor,
}

@(require_results)
mul :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(a, "mul", loc)
	_assert_broadcastable(a, b, loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Mul{b=b},
	}
	_record_forward(op, loc=loc)

	return
}

Gelu_Mul :: struct {
	b: Tensor,
}

@(require_results)
gelu_mul :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(a, "gelu_mul", loc)
	_assert_broadcastable(a, b, loc)

	if a.type == .F32 || _current_ctx.training || .Gelu_Mul not_in _current_ctx.backend.forward_ops {
		return mul(gelu(a, loc=loc), b, loc=loc)
	}

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Gelu_Mul{b=b},
	}
	_record_forward(op, loc=loc)

	return
}

Div :: struct {
	b: Tensor,
}

@(require_results)
div :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(a, "div", loc)
	_assert_broadcastable(a, b, loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Div{b=b},
	}
	_record_forward(op, loc=loc)

	return
}

Exp :: struct {
}

@(require_results)
exp :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "exp", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Exp{},
	}
	_record_forward(op, loc=loc)

	return
}

Sqrt :: struct {
}

@(require_results)
sqrt :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.type == .F32, "sqrt is F32-only", loc=loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Sqrt{},
	}
	_record_forward(op, loc=loc)

	return
}

Clamp :: struct {
	min_val: f32,
	max_val: f32,
}

@(require_results)
clamp :: proc(input: Tensor, min_val, max_val: f32, loc := #caller_location) -> (output: Tensor) {
	assert(input.type == .F32, "clamp is F32-only", loc=loc)
	assert(min_val <= max_val, "requires min_val <= max_val", loc=loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Clamp{
			min_val = min_val,
			max_val = max_val,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Min :: struct {
	b: Tensor,
}

@(require_results)
min :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(a.type == .F32 && b.type == .F32, "min is F32-only", loc=loc)
	_assert_broadcastable(a, b, loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Min{
			b = b,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Max :: struct {
	b: Tensor,
}

@(require_results)
max :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(a.type == .F32 && b.type == .F32, "max is F32-only", loc=loc)
	_assert_broadcastable(a, b, loc)

	output = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Max{
			b = b,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Mean :: struct {}
Sum :: struct {}
Max_Reduce :: struct {}

@(require_results)
_reduce_trailing_record :: proc(input: Tensor, variant: Operation_Variant, loc: runtime.Source_Code_Location) -> (output: Tensor) {
	output = _zeros_drop_last(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = variant,
	}
	_record_forward(op, loc=loc)

	return
}

@(require_results)
_reduce_axis :: proc(input: Tensor, axis: int, variant: Operation_Variant, loc: runtime.Source_Code_Location) -> (output: Tensor) {
	_assert_float(input, "reduce", loc)
	rank := input.rank
	target := axis < 0 ? rank - 1 : axis
	assert(target >= 0 && target < rank, "reduce axis out of range", loc=loc)

	if rank == 1 || target == rank - 1 {
		return _reduce_trailing_record(input, variant, loc=loc)
	}

	leading := 1
	for i in 0 ..< target {
		leading *= input.shape[i]
	}
	reduced_dim := input.shape[target]
	trailing := 1
	for i in target + 1 ..< rank {
		trailing *= input.shape[i]
	}

	collapsed := reshape(input, {leading, reduced_dim, trailing}, loc=loc)
	swapped   := permute(collapsed, {0, 2, 1}, loc=loc)
	reduced   := _reduce_trailing_record(swapped, variant, loc=loc)

	out_shape: [MAX_TENSOR_RANK]int
	out_rank := 0
	for i in 0 ..< rank {
		if i == target {
			continue
		}
		out_shape[out_rank] = input.shape[i]
		out_rank += 1
	}
	return reshape(reduced, out_shape[:out_rank], loc=loc)
}

@(require_results)
mean :: proc(input: Tensor, axis := -1, loc := #caller_location) -> Tensor {
	return _reduce_axis(input, axis, Mean{}, loc=loc)
}

@(require_results)
sum :: proc(input: Tensor, axis := -1, loc := #caller_location) -> Tensor {
	return _reduce_axis(input, axis, Sum{}, loc=loc)
}

@(require_results)
max_reduce :: proc(input: Tensor, axis := -1, loc := #caller_location) -> Tensor {
	return _reduce_axis(input, axis, Max_Reduce{}, loc=loc)
}

Transpose :: struct {}

@(require_results)
transpose :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 2, "transpose requires a 2-D tensor", loc=loc)
	assert(input.type == .F32, "transpose is F32-only", loc=loc)

	rows    := input.shape[0]
	columns := input.shape[1]

	output = zeros(.F32, {columns, rows}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Transpose{},
	}
	_record_forward(op, loc=loc)

	return
}

Select :: struct {
	indices: Tensor,
}

select :: proc{_select_tensor, _select_ints}

@(require_results)
_select_tensor :: proc(input: Tensor, indices: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "select input must have rank >= 1", loc=loc)
	assert(indices.type == .I32, "select indices must be an I32 tensor", loc=loc)
	assert(indices.rank == 1, "select indices must be rank-1", loc=loc)
	assert(len(indices) > 0, "select requires at least one index", loc=loc)

	out_shape: [MAX_TENSOR_RANK]int = input.shape
	out_shape[0] = len(indices)
	output = zeros(input.type, out_shape[:input.rank], loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Select{
			indices = indices,
		}
	}
	_record_forward(op, loc=loc)

	return
}

@(require_results)
_select_ints :: proc(input: Tensor, indices: []int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "select input must have rank >= 1", loc=loc)
	assert(builtin.len(indices) > 0, "select requires at least one index", loc=loc)

	for index in indices {
		assert(index >= 0 && index < input.shape[0], "select index out of bounds", loc=loc)
	}

	index_tensor := scratch(.I32, {builtin.len(indices)}, loc=loc)
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		host := builtin.make([]i32, builtin.len(indices), allocator=context.temp_allocator, loc=loc)
		for index, i in indices {
			host[i] = i32(index)
		}
		index_tensor.backend.buffer_set(index_tensor.buffers[.Data], mem.slice_to_bytes(host), loc)
	}

	return _select_tensor(input, index_tensor, loc=loc)
}

Slice :: struct {
	start: int,
	end:   int,
}

@(require_results)
slice :: proc(input: Tensor, start, end: int, loc := #caller_location) -> (output: Tensor) {
	fmt.assertf(start >= 0 && end <= len(input) && start <= end, "slice indices out of bounds %v:%v", start, end, loc=loc)

	output = zeros(input.type, {end - start}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Slice{
			start = start,
			end   = end,
		},
	}
	_record_forward(op, loc=loc)

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
	_record_forward(op, loc=loc)

	return
}

Slice_Leading :: struct {
	start, end: int,
}

@(require_results)
slice_leading :: proc(input: Tensor, start, end: int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "slice_leading input must have rank >= 1", loc=loc)
	leading := input.shape[0]
	fmt.assertf(start >= 0 && start < end && end <= leading, "slice_leading indices out of bounds %v:%v (leading=%v)", start, end, leading, loc=loc)

	new_leading := end - start
	output = _zeros_replace_leading(input, new_leading, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Slice_Leading{
			start = start,
			end   = end,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Concat :: struct {
	inputs: []Tensor,
}

@(require_results)
concat :: proc(inputs: ..Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(builtin.len(inputs) > 0, "requires at least one input", loc=loc)

	first := inputs[0]
	trailing_sum := first.shape[first.rank - 1]
	for i in 1 ..< builtin.len(inputs) {
		assert(inputs[i].rank == first.rank, "all concat inputs must have the same rank", loc=loc)
		for d in 0 ..< first.rank - 1 {
			assert(inputs[i].shape[d] == first.shape[d], "all concat inputs must match in non-trailing dims", loc=loc)
		}
		trailing_sum += inputs[i].shape[inputs[i].rank - 1]
	}

	inputs_copy := _op_arena_make(Tensor, builtin.len(inputs), loc=loc)
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
	_record_forward(op, loc=loc)

	return
}

Linear :: struct {
	weight: Tensor,
}

@(require_results)
linear :: proc(input, weight: Tensor, bias: Maybe(Tensor) = nil, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "linear input must have rank >= 1",  loc=loc)
	assert(weight.rank == 2, "linear weight must be a 2-D tensor [output_size, input_size]", loc=loc)

	output_size := weight.shape[0]
	input_size  := weight.shape[1]
	assert(input.shape[input.rank - 1] == input_size, "input trailing dim must equal weight's input dim", loc=loc)
	_assert_float(input, "linear", loc)

	#partial switch weight.type {
	case .Q4_K, .Q6_K:
		assert(bias == nil, "quantized linear does not support bias", loc=loc)
		assert(input.type == .Bf16, "quantized linear input must be Bf16", loc=loc)
		assert(input_size % K_QUANT_BLOCK_SIZE == 0, "quantized linear input dim must be a multiple of 256", loc=loc)
	}

	new_shape                := input.shape
	new_shape[input.rank - 1] = output_size
	output = zeros(input.type, new_shape[:input.rank], loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Linear{
			weight = weight,
		}
	}
	#partial switch weight.type {
	case .Q4_K: op.variant = Linear_Q4_K{weight=weight}
	case .Q6_K: op.variant = Linear_Q6_K{weight=weight}
	}
	_record_forward(op, loc=loc)

	if b, has_bias := bias.?; has_bias {
		assert(b.rank == 1 && b.shape[0] == output_size, "linear bias must be a 1-D [output_size] tensor", loc=loc)
		output = add(output, b, loc=loc)
	}

	return
}

embedding :: proc{_embedding_tensor, _embedding_ints}

@(require_results)
_embedding_tensor :: proc(table: Tensor, ids: Tensor, loc := #caller_location) -> Tensor {
	return _select_tensor(table, ids, loc=loc)
}

@(require_results)
_embedding_ints :: proc(table: Tensor, ids: []int, loc := #caller_location) -> Tensor {
	return _select_ints(table, ids, loc=loc)
}

@(require_results)
dropout :: proc(input: Tensor, rate: f32, loc := #caller_location) -> Tensor {
	assert(rate >= 0 && rate < 1, "dropout rate must be in [0, 1)", loc=loc)
	assert(input.type == .F32, "dropout is F32-only", loc=loc)
	if !is_training(loc=loc) || rate == 0 {
		return input
	}

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	scale := 1 / (1 - rate)
	host  := builtin.make([]f32, input.count, allocator=context.temp_allocator, loc=loc)
	for &value in host {
		value = 0 if rand.float32() < rate else scale
	}

	shape := input.shape
	mask := scratch(.F32, shape[:input.rank], loc=loc)
	set_data(mask, host, loc=loc)

	return mul(input, mask, loc=loc)
}

Linear_Q4_K :: struct {
	weight: Tensor, // .Q4_K logical shape [output_size, input_size]; input_size % 256 == 0
}

Linear_Q4_K_Gate_Up_Geglu :: struct {
	w_gate: Tensor,
	w_up:   Tensor,
}

@(require_results)
linear_q4_k_gate_up_geglu :: proc(input, w_gate, w_up: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 1, "linear_q4_k_gate_up_geglu input must have rank >= 1", loc=loc)
	assert(w_gate.rank == 2, "linear_q4_k_gate_up_geglu w_gate must be 2-D", loc=loc)
	assert(w_up.rank == 2, "linear_q4_k_gate_up_geglu w_up must be 2-D", loc=loc)
	assert(w_gate.type == .Q4_K && w_up.type == .Q4_K, "linear_q4_k_gate_up_geglu weights must be Q4_K", loc=loc)
	assert(input.type == .Bf16 || input.type == .F32, "linear_q4_k_gate_up_geglu input must be Bf16 or F32", loc=loc)
	assert(w_gate.shape[0] == w_up.shape[0], "gate/up output dims must match", loc=loc)
	assert(w_gate.shape[1] == w_up.shape[1], "gate/up input dims must match", loc=loc)

	output_size := w_gate.shape[0]
	input_size  := w_gate.shape[1]
	assert(input_size % K_QUANT_BLOCK_SIZE == 0, "input dim must be a multiple of 256", loc=loc)
	assert(input.shape[input.rank - 1] == input_size, "input trailing dim must equal weight's input dim", loc=loc)

	if _leading_count(input) == 1 && !_current_ctx.training && .Linear_Q4_K_Gate_Up_Geglu in _current_ctx.backend.forward_ops {
		new_shape: [MAX_TENSOR_RANK]int = input.shape
		new_shape[input.rank - 1] = output_size
		output = zeros(.Bf16, new_shape[:input.rank], loc=loc)
		op := Operation{
			input   = input,
			output  = output,
			variant = Linear_Q4_K_Gate_Up_Geglu{w_gate=w_gate, w_up=w_up},
		}
		_record_forward(op, loc=loc)
		return
	}

	gate  := linear(input, w_gate, loc=loc)
	up    := linear(input, w_up,   loc=loc)
	output = gelu_mul(gate, up, loc=loc)

	return
}

Linear_Q6_K :: struct {
	weight: Tensor, // .Q6_K logical shape [output_size, input_size]; input_size % 256 == 0
}

Rope :: struct {
	head_count:        int,
	base:              f32,
	position_offset:   int,
	rotate_pair_count: int, // pairs in [0, rotate_pair_count) are rotated; the rest pass through

	cos_cache: Tensor,
	sin_cache: Tensor,
}

@(require_results)
rope :: proc(input: Tensor, head_count: int, base: f32 = 10000, position_offset: int = 0, rope_fraction: f32 = 1.0, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 2, "rope requires rank >= 2", loc=loc)
	_assert_float(input, "rope", loc)
	assert(rope_fraction > 0 && rope_fraction <= 1, "rope_fraction must be in (0, 1]", loc=loc)

	input_size := input.shape[input.rank - 1]
	assert(input_size % head_count == 0, "trailing dim must be divisible by head count", loc=loc)

	head_size := input_size / head_count
	assert(head_size % 2 == 0, "head size must be even", loc=loc)

	half_head         := head_size / 2
	rotate_pair_count := int(rope_fraction * f32(head_size)) / 2
	assert(rotate_pair_count > 0 && rotate_pair_count <= half_head, "rope_fraction yields zero rotated pairs", loc=loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Rope{
			head_count        = head_count,
			base              = base,
			position_offset   = position_offset,
			rotate_pair_count = rotate_pair_count,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Layernorm :: struct {
	weight: Tensor,

	mean: Tensor,
	rstd: Tensor,
}

@(require_results)
layernorm :: proc(input, weight: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(weight.rank == 1, "layernorm weight must be 1-D", loc=loc)
	assert(weight.shape[0] == input.shape[input.rank - 1], "layernorm weight length must equal input's trailing dim", loc=loc)
	_assert_float(input, "layernorm", loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Layernorm{
			weight = weight,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Rmsnorm :: struct {
	weight: Tensor,
	eps:    f32,

	rstd: Tensor,
}

RMSNORM_DEFAULT_EPS :: 1e-5

@(require_results)
rmsnorm :: proc(input, weight: Tensor, eps: f32 = RMSNORM_DEFAULT_EPS, loc := #caller_location) -> (output: Tensor) {
	assert(weight.rank == 1, "rmsnorm weight must be 1-D", loc=loc)
	assert(weight.shape[0] == input.shape[input.rank - 1], "rmsnorm weight length must equal input's trailing dim", loc=loc)
	assert(eps > 0, "rmsnorm eps must be positive", loc=loc)
	_assert_float(input, "rmsnorm", loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Rmsnorm{
			weight = weight,
			eps    = eps,
		},
	}
	_record_forward(op, loc=loc)

	return
}

@(require_results)
per_head_rmsnorm :: proc(x, weight: Tensor, head_count: int, eps: f32 = RMSNORM_DEFAULT_EPS, loc := #caller_location) -> Tensor {
	assert(x.rank == 2, "per_head_rmsnorm requires a 2-D [tokens, heads * head_size] tensor", loc=loc)
	assert(x.shape[1] % head_count == 0, "per_head_rmsnorm trailing dim must be divisible by head_count", loc=loc)

	token_count := x.shape[0]
	head_size   := x.shape[1] / head_count
	view        := reshape(x, {token_count * head_count, head_size}, loc=loc)
	normed      := rmsnorm(view, weight, eps=eps, loc=loc)
	return reshape(normed, {token_count, head_count * head_size}, loc=loc)
}

Rmsnorm_Rope :: struct {
	weight:            Tensor,
	eps:               f32,
	head_count:        int,
	base:              f32,
	position_offset:   int,
	rotate_pair_count: int,
}

Rmsnorm_Rope_Write_Cache :: struct {
	weight:            Tensor,
	eps:               f32,
	head_count:        int,
	base:              f32,
	position_offset:   int,
	rotate_pair_count: int,
	cache:             Tensor,
	cache_capacity:    int,
}

@(require_results)
rmsnorm_rope :: proc(input, weight: Tensor, head_count: int, eps: f32 = RMSNORM_DEFAULT_EPS, base: f32 = 10000, position_offset := 0, rope_fraction: f32 = 1.0, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 2, "rmsnorm_rope requires rank-2 input [tokens, head_count*head_size]", loc=loc)
	_assert_float(input, "rmsnorm_rope", loc)
	assert(weight.rank == 1, "rmsnorm_rope weight must be 1-D", loc=loc)
	assert(eps > 0, "rmsnorm_rope eps must be positive", loc=loc)
	assert(rope_fraction > 0 && rope_fraction <= 1, "rope_fraction must be in (0, 1]", loc=loc)

	input_size := input.shape[1]
	assert(input_size % head_count == 0, "trailing dim must be divisible by head count", loc=loc)
	head_size := input_size / head_count
	assert(head_size % 2 == 0, "head size must be even", loc=loc)
	assert(weight.shape[0] == head_size, "rmsnorm_rope weight length must equal head_size", loc=loc)

	half_head         := head_size / 2
	rotate_pair_count := int(rope_fraction * f32(head_size)) / 2
	assert(rotate_pair_count > 0 && rotate_pair_count <= half_head, "rope_fraction yields zero rotated pairs", loc=loc)

	if input.type == .F32 || _current_ctx.training || .Rmsnorm_Rope not_in _current_ctx.backend.forward_ops {
		view   := reshape(input, []int{input.shape[0] * head_count, head_size}, loc=loc)
		normed := rmsnorm(view, weight, eps=eps, loc=loc)
		normed  = reshape(normed, []int{input.shape[0], head_count * head_size}, loc=loc)
		return rope(normed, head_count, base=base, position_offset=position_offset, rope_fraction=rope_fraction, loc=loc)
	}

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Rmsnorm_Rope{
			weight            = weight,
			eps               = eps,
			head_count        = head_count,
			base              = base,
			position_offset   = position_offset,
			rotate_pair_count = rotate_pair_count,
		},
	}
	_record_forward(op, loc=loc)

	return
}

@(require_results)
rmsnorm_rope_write_cache :: proc(
	input, weight, cache: Tensor,
	cache_capacity: int,
	head_count: int,
	eps: f32 = RMSNORM_DEFAULT_EPS, base: f32 = 10000,
	position_offset := 0, rope_fraction: f32 = 1.0,
	loc := #caller_location,
) -> (output: Tensor) {
	assert(input.rank == 2, "rmsnorm_rope_write_cache requires rank-2 input", loc=loc)
	_assert_float(input, "rmsnorm_rope_write_cache", loc)
	assert(weight.rank == 1, "rmsnorm_rope_write_cache weight must be 1-D", loc=loc)
	assert(cache.rank == 2, "rmsnorm_rope_write_cache cache must be 2-D [capacity, head_count*head_size]", loc=loc)
	assert(cache.shape[1] == input.shape[1], "rmsnorm_rope_write_cache cache trailing dim must equal input trailing dim", loc=loc)
	assert(cache_capacity == cache.shape[0], "rmsnorm_rope_write_cache cache_capacity must equal cache.shape[0]", loc=loc)
	assert(eps > 0, "rmsnorm_rope_write_cache eps must be positive", loc=loc)
	assert(rope_fraction > 0 && rope_fraction <= 1, "rope_fraction must be in (0, 1]", loc=loc)

	input_size := input.shape[1]
	assert(input_size % head_count == 0, "trailing dim must be divisible by head count", loc=loc)
	head_size := input_size / head_count
	assert(head_size % 2 == 0, "head size must be even", loc=loc)
	assert(weight.shape[0] == head_size, "rmsnorm_rope_write_cache weight length must equal head_size", loc=loc)

	half_head         := head_size / 2
	rotate_pair_count := int(rope_fraction * f32(head_size)) / 2
	assert(rotate_pair_count > 0 && rotate_pair_count <= half_head, "rope_fraction yields zero rotated pairs", loc=loc)

	if !_current_ctx.training && .Rmsnorm_Rope_Write_Cache in _current_ctx.backend.forward_ops {
		output = zeros_like(input, loc=loc)
		op := Operation{
			input   = input,
			output  = output,
			variant = Rmsnorm_Rope_Write_Cache{
				weight            = weight,
				eps               = eps,
				head_count        = head_count,
				base              = base,
				position_offset   = position_offset,
				rotate_pair_count = rotate_pair_count,
				cache             = cache,
				cache_capacity    = cache_capacity,
			},
		}
		_record_forward(op, loc=loc)
		return
	}

	output = rmsnorm_rope(input, weight, head_count, eps=eps, base=base, position_offset=position_offset, rope_fraction=rope_fraction, loc=loc)
	return
}

Add_Rmsnorm :: struct {
	b:            Tensor,
	weight:       Tensor,
	eps:          f32,
	residual_out: Tensor,
}

@(require_results)
add_rmsnorm :: proc(a, b, weight: Tensor, eps: f32 = RMSNORM_DEFAULT_EPS, loc := #caller_location) -> (residual_new, normed: Tensor) {
	assert(a.rank == b.rank, "add_rmsnorm a/b rank must match", loc=loc)
	assert(a.type == b.type, "add_rmsnorm a/b dtypes must match", loc=loc)
	_assert_float(a, "add_rmsnorm", loc)
	assert(weight.rank == 1, "add_rmsnorm weight must be 1-D", loc=loc)
	assert(weight.shape[0] == a.shape[a.rank - 1], "add_rmsnorm weight length must equal trailing dim", loc=loc)
	assert(eps > 0, "add_rmsnorm eps must be positive", loc=loc)
	for d in 0 ..< a.rank {
		assert(a.shape[d] == b.shape[d], "add_rmsnorm a/b shape must match", loc=loc)
	}

	if a.type == .F32 || _current_ctx.training || .Add_Rmsnorm not_in _current_ctx.backend.forward_ops {
		residual_new = add(a, b, loc=loc)
		normed       = rmsnorm(residual_new, weight, eps, loc=loc)
		return
	}

	residual_new = zeros_like(a, loc=loc)
	normed       = zeros_like(a, loc=loc)

	op := Operation{
		input   = a,
		output  = normed,
		variant = Add_Rmsnorm{
			b            = b,
			weight       = weight,
			eps          = eps,
			residual_out = residual_new,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Softmax :: struct {}

@(require_results)
softmax :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "softmax", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Softmax{},
	}
	_record_forward(op, loc=loc)

	return
}

Log_Softmax :: struct {}

@(require_results)
log_softmax :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "log_softmax", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Log_Softmax{},
	}
	_record_forward(op, loc=loc)

	return
}

Entropy :: struct {}

@(require_results)
entropy :: proc(probabilities: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(probabilities, "entropy", loc)
	output = _zeros_drop_last(probabilities, loc=loc)

	op := Operation{
		input   = probabilities,
		output  = output,
		variant = Entropy{},
	}
	_record_forward(op, loc=loc)

	return
}

Mean_Squared_Error :: struct {
	targets: Tensor,
}

@(require_results)
mean_squared_error :: proc(predictions, targets: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(predictions.type == .F32 && targets.type == .F32, "mean_squared_error is F32-only", loc=loc)
	assert(len(predictions) == len(targets), "predictions and targets must have same length", loc=loc)

	output = _zeros_drop_last(predictions, loc=loc)

	op := Operation{
		input   = predictions,
		output  = output,
		variant = Mean_Squared_Error{
			targets = targets,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Smooth_L1 :: struct {
	targets: Tensor,
	beta:    f32,
}

@(require_results)
smooth_l1 :: proc(predictions, targets: Tensor, beta: f32 = 1.0, loc := #caller_location) -> (output: Tensor) {
	assert(predictions.type == .F32 && targets.type == .F32, "smooth_l1 is F32-only", loc=loc)
	assert(len(predictions) == len(targets), "predictions and targets must have same length", loc=loc)
	assert(beta > 0, "smooth_l1 beta must be positive", loc=loc)

	output = _zeros_drop_last(predictions, loc=loc)

	op := Operation{
		input   = predictions,
		output  = output,
		variant = Smooth_L1{
			targets = targets,
			beta    = beta,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Cross_Entropy :: struct {
	targets: Tensor,

	probabilities: Tensor,
}

cross_entropy :: proc{_cross_entropy_tensor, _cross_entropy_ints}

@(require_results)
_cross_entropy_tensor :: proc(input: Tensor, targets: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.type == .F32, "cross_entropy is F32-only", loc=loc)
	assert(targets.type == .I32, "cross_entropy targets must be an I32 tensor", loc=loc)
	assert(targets.rank == 1, "cross_entropy targets must be rank-1", loc=loc)
	assert(input.rank >= 1, "cross_entropy input must have rank >= 1", loc=loc)
	assert(len(targets) > 0, "must have at least one target", loc=loc)
	assert(_leading_count(input) == len(targets), "input leading-dim product must equal number of targets", loc=loc)

	output = _zeros_drop_last(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Cross_Entropy{
			targets = targets,
		},
	}
	_record_forward(op, loc=loc)

	return
}

@(require_results)
_cross_entropy_ints :: proc(input: Tensor, targets: []int, loc := #caller_location) -> (output: Tensor) {
	sample_count := builtin.len(targets)
	assert(input.type == .F32, "cross_entropy is F32-only", loc=loc)
	assert(sample_count > 0, "must have at least one target", loc=loc)
	assert(input.rank >= 1, "cross_entropy input must have rank >= 1", loc=loc)
	assert(_leading_count(input) == sample_count, "input leading-dim product must equal number of targets", loc=loc)

	class_size := input.shape[input.rank - 1]

	target_tensor := scratch(.I32, {sample_count}, loc=loc)
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		host := builtin.make([]i32, sample_count, allocator=context.temp_allocator, loc=loc)
		for target, i in targets {
			assert(target >= 0 && target < class_size, "target is out of bounds", loc=loc)
			host[i] = i32(target)
		}
		target_tensor.backend.buffer_set(target_tensor.buffers[.Data], mem.slice_to_bytes(host), loc)
	}

	return _cross_entropy_tensor(input, target_tensor, loc=loc)
}

Relu :: struct {
}

@(require_results)
relu :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "relu", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Relu{},
	}
	_record_forward(op, loc=loc)

	return
}

Sigmoid :: struct {
}

@(require_results)
sigmoid :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "sigmoid", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Sigmoid{},
	}
	_record_forward(op, loc=loc)

	return
}

Gelu :: struct {
}

@(require_results)
gelu :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "gelu", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Gelu{},
	}
	_record_forward(op, loc=loc)

	return
}

Silu :: struct {
}

@(require_results)
silu :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "silu", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Silu{},
	}
	_record_forward(op, loc=loc)

	return
}

Tanh :: struct {
}

@(require_results)
tanh :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	_assert_float(input, "tanh", loc)
	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Tanh{},
	}
	_record_forward(op, loc=loc)

	return
}

Batched_Matmul :: struct {
	b: Tensor,
}

@(require_results)
batched_matmul :: proc(a, b: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(a.rank == 3 && b.rank == 3, "batched_matmul requires rank-3 inputs", loc=loc)
	assert(a.shape[0] == b.shape[0], "batched_matmul batch dims must match", loc=loc)
	assert(a.shape[2] == b.shape[1], "batched_matmul inner dim must match: a.shape[2] == b.shape[1]", loc=loc)
	assert(a.type == b.type, "batched_matmul inputs must have the same dtype", loc=loc)
	_assert_float(a, "batched_matmul", loc)

	batch_count := a.shape[0]
	m           := a.shape[1]
	n           := b.shape[2]

	output = zeros(a.type, {batch_count, m, n}, loc=loc)

	op := Operation{
		input   = a,
		output  = output,
		variant = Batched_Matmul{
			b=b,
		},
	}
	_record_forward(op, loc=loc)

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
	output = zeros(input.type, {out_shape[0], out_shape[1], out_shape[2]}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Permute{axes=axes},
	}
	_record_forward(op, loc=loc)

	return
}

Im2col :: struct {
	kernel_h: int,
	kernel_w: int,
	stride_h: int,
	stride_w: int,
	pad_h:    int,
	pad_w:    int,
	out_h:    int,
	out_w:    int,
}

@(require_results)
im2col :: proc(input: Tensor, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w: int, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 4, "im2col input must be 4-D [n, h, w, c]", loc=loc)
	assert(input.type == .F32 || input.type == .Bf16, "im2col requires F32 or Bf16 input", loc=loc)
	assert(kernel_h >= 1 && kernel_w >= 1, "im2col kernel dims must be >= 1", loc=loc)
	assert(stride_h >= 1 && stride_w >= 1, "im2col stride must be >= 1", loc=loc)
	assert(pad_h >= 0 && pad_w >= 0, "im2col padding must be >= 0", loc=loc)

	h := input.shape[1]
	w := input.shape[2]
	c := input.shape[3]

	out_h := (h + 2 * pad_h - kernel_h) / stride_h + 1
	out_w := (w + 2 * pad_w - kernel_w) / stride_w + 1
	assert(out_h >= 1 && out_w >= 1, "im2col output dims must be >= 1", loc=loc)

	output = zeros(input.type, {input.shape[0] * out_h * out_w, kernel_h * kernel_w * c}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Im2col{
			kernel_h = kernel_h,
			kernel_w = kernel_w,
			stride_h = stride_h,
			stride_w = stride_w,
			pad_h    = pad_h,
			pad_w    = pad_w,
			out_h    = out_h,
			out_w    = out_w,
		},
	}
	_record_forward(op, loc=loc)

	return
}

@(require_results)
conv2d :: proc(input, weight: Tensor, bias: Maybe(Tensor) = nil, stride := 1, padding := 0, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 4, "conv2d input must be 4-D [n, h, w, c]", loc=loc)
	assert(weight.rank == 4, "conv2d weight must be 4-D [out_c, kh, kw, in_c]", loc=loc)
	assert(input.type == .F32 || input.type == .Bf16, "conv2d requires F32 or Bf16 input", loc=loc)
	assert(stride >= 1, "conv2d stride must be >= 1", loc=loc)
	assert(padding >= 0, "conv2d padding must be >= 0", loc=loc)

	out_c := weight.shape[0]
	kernel_h := weight.shape[1]
	kernel_w := weight.shape[2]
	in_c     := weight.shape[3]
	assert(input.shape[3] == in_c, "conv2d weight in_c must match input channel count", loc=loc)

	input_n := input.shape[0]
	h       := input.shape[1]
	w       := input.shape[2]

	out_h := (h + 2 * padding - kernel_h) / stride + 1
	out_w := (w + 2 * padding - kernel_w) / stride + 1
	assert(out_h >= 1 && out_w >= 1, "conv2d output dims must be >= 1", loc=loc)

	weight_matrix := reshape(weight, {out_c, kernel_h * kernel_w * in_c}, loc=loc)
	columns       := im2col(input, kernel_h, kernel_w, stride, stride, padding, padding, loc=loc)
	result        := linear(columns, weight_matrix, bias, loc=loc)
	output         = reshape(result, {input_n, out_h, out_w, out_c}, loc=loc)

	return
}

@(require_results)
conv1d :: proc(input, weight: Tensor, bias: Maybe(Tensor) = nil, stride := 1, padding := 0, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank == 3, "conv1d input must be 3-D [n, w, c]", loc=loc)
	assert(weight.rank == 3, "conv1d weight must be 3-D [out_c, kw, in_c]", loc=loc)
	assert(input.type == .F32 || input.type == .Bf16, "conv1d requires F32 or Bf16 input", loc=loc)
	assert(stride >= 1, "conv1d stride must be >= 1", loc=loc)
	assert(padding >= 0, "conv1d padding must be >= 0", loc=loc)

	input_n := input.shape[0]
	w       := input.shape[1]
	c       := input.shape[2]
	out_c   := weight.shape[0]
	kernel_w := weight.shape[1]
	in_c     := weight.shape[2]
	assert(c == in_c, "conv1d weight in_c must match input channel count", loc=loc)

	out_w := (w + 2 * padding - kernel_w) / stride + 1
	assert(out_w >= 1, "conv1d output dim must be >= 1", loc=loc)

	input_2d      := reshape(input, {input_n, 1, w, c}, loc=loc)
	weight_matrix := reshape(weight, {out_c, kernel_w * in_c}, loc=loc)
	columns       := im2col(input_2d, 1, kernel_w, stride, stride, 0, padding, loc=loc)
	result        := linear(columns, weight_matrix, bias, loc=loc)
	output         = reshape(result, {input_n, out_w, out_c}, loc=loc)

	return
}

Max_Pool2d :: struct {
	kernel_h: int,
	kernel_w: int,
	stride_h: int,
	stride_w: int,
}

Avg_Pool2d :: struct {
	kernel_h: int,
	kernel_w: int,
	stride_h: int,
	stride_w: int,
}

@(require_results)
_pool2d_shape :: proc(input: Tensor, kernel_h, kernel_w, stride_h, stride_w: int, loc: runtime.Source_Code_Location) -> (out_h, out_w: int) {
	assert(input.rank == 4, "pool2d input must be 4-D [n, h, w, c]", loc=loc)
	assert(input.type == .F32 || input.type == .Bf16, "pool2d requires F32 or Bf16 input", loc=loc)
	assert(kernel_h >= 1 && kernel_w >= 1, "pool2d kernel dims must be >= 1", loc=loc)
	assert(stride_h >= 1 && stride_w >= 1, "pool2d stride must be >= 1", loc=loc)

	h := input.shape[1]
	w := input.shape[2]
	assert(h >= kernel_h && w >= kernel_w, "pool2d window must fit inside the input", loc=loc)
	assert((h - kernel_h) % stride_h == 0, "pool2d height must tile the window exactly", loc=loc)
	assert((w - kernel_w) % stride_w == 0, "pool2d width must tile the window exactly", loc=loc)

	out_h = (h - kernel_h) / stride_h + 1
	out_w = (w - kernel_w) / stride_w + 1
	return
}

@(require_results)
max_pool2d :: proc(input: Tensor, size := 2, stride := -1, loc := #caller_location) -> (output: Tensor) {
	actual_stride := stride if stride > 0 else size
	out_h, out_w := _pool2d_shape(input, size, size, actual_stride, actual_stride, loc)

	output = zeros(input.type, {input.shape[0], out_h, out_w, input.shape[3]}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Max_Pool2d{
			kernel_h = size,
			kernel_w = size,
			stride_h = actual_stride,
			stride_w = actual_stride,
		},
	}
	_record_forward(op, loc=loc)

	return
}

@(require_results)
avg_pool2d :: proc(input: Tensor, size := 2, stride := -1, loc := #caller_location) -> (output: Tensor) {
	actual_stride := stride if stride > 0 else size
	out_h, out_w := _pool2d_shape(input, size, size, actual_stride, actual_stride, loc)

	output = zeros(input.type, {input.shape[0], out_h, out_w, input.shape[3]}, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Avg_Pool2d{
			kernel_h = size,
			kernel_w = size,
			stride_h = actual_stride,
			stride_w = actual_stride,
		},
	}
	_record_forward(op, loc=loc)

	return
}

Causal_Mask :: struct {}

@(require_results)
causal_mask :: proc(input: Tensor, loc := #caller_location) -> (output: Tensor) {
	assert(input.rank >= 2, "causal_mask requires rank >= 2", loc=loc)
	_assert_float(input, "causal_mask", loc)
	T := input.shape[input.rank - 1]
	assert(input.shape[input.rank - 2] == T, "causal_mask requires square trailing 2D ([..., T, T])", loc=loc)

	output = zeros_like(input, loc=loc)

	op := Operation{
		input   = input,
		output  = output,
		variant = Causal_Mask{},
	}
	_record_forward(op, loc=loc)

	return
}
