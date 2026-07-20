package machine_learning

import "base:builtin"
import "base:intrinsics"
import "base:runtime"

import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
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

	training:  bool,
	pass_open: bool,

	owner_thread_id: int,

	grad_norm_accumulator: Backend_Buffer,
}

@(thread_local)
_current_ctx: ^Context

_context_init :: proc(ctx: ^Context, backend: ^Backend, allocator: mem.Allocator, loc: runtime.Source_Code_Location) {
	_assert_operation_variant_order()

	ctx.backend   = backend
	ctx.training  = true
	ctx.pass_open = true

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

	_current_ctx.training  = training
	_current_ctx.pass_open = true
	_current_ctx.backend.clear(loc)

	mem.arena_free_all(&_current_ctx.op_arena)
	_current_ctx.operation_count = 0
}

_pass_end :: proc() {
	_current_ctx.pass_open = false
}

@(deferred_none=_pass_end)
pass :: proc(training := false, loc := #caller_location) -> bool {
	clear(training=training, loc=loc)
	return true
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
		"op arena exhausted allocating %d x %v (%d bytes) - raise OPERATION_ARENA_DEFAULT_SIZE",
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

