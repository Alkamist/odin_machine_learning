package machine_learning

import "base:builtin"
import "base:intrinsics"
import "base:runtime"

import "core:fmt"
import "core:mem"
import "core:reflect"

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

