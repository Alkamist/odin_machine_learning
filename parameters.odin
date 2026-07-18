package machine_learning

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:strings"

Init_He     :: struct {}
Init_Xavier :: struct {}
Init_Normal :: struct {
	mean: f32,
	std:  f32,
}
Init_Value :: struct {
	value: f32,
}

Init :: union {
	Init_He,
	Init_Xavier,
	Init_Normal,
	Init_Value,
}

Parameter_Info :: struct {
	name:      string,
	tensor:    Tensor,
	init:      Init,
	trainable: bool,
}

register :: proc(list: ^[dynamic]Parameter_Info, prefix, name: string, tensor: Tensor, init: Init = nil, trainable := true, loc := #caller_location) {
	assert(!trainable || has_gradient(tensor), "trainable parameter requires a gradient buffer", loc=loc)

	if list.allocator.procedure == nil {
		list.allocator = context.allocator
	}

	full: string
	if prefix == "" {
		full = strings.clone(name, allocator=list.allocator)
	} else {
		full = fmt.aprintf("%s.%s", prefix, name, allocator=list.allocator)
	}
	append(list, Parameter_Info{name=full, tensor=tensor, init=init, trainable=trainable})
}

@(require_results)
parameter_make :: proc(list: ^[dynamic]Parameter_Info, prefix, name: string, type: Data_Type, shape: []int, init: Init = nil, trainable := true, loc := #caller_location) -> (t: Tensor) {
	buffers := DEFAULT_PARAMETER_BUFFERS if trainable else Buffer_Set{.Data}
	t = alloc(type, shape, persistent=true, buffers=buffers, loc=loc)
	register(list, prefix, name, t, init=init, trainable=trainable, loc=loc)
	return
}

registry_destroy :: proc(list: ^[dynamic]Parameter_Info, loc := #caller_location) {
	for info in list {
		destroy(info.tensor, loc=loc)
		builtin.delete(info.name, allocator=list.allocator, loc=loc)
	}
	builtin.delete(list^, loc=loc)
	list^ = nil
}

registry_clear :: proc(list: ^[dynamic]Parameter_Info, loc := #caller_location) {
	for info in list {
		builtin.delete(info.name, allocator=list.allocator, loc=loc)
	}
	builtin.clear(list)
}

registry_randomize :: proc(list: []Parameter_Info, loc := #caller_location) {
	for info in list {
		switch spec in info.init {
		case Init_He:
			assert(info.tensor.rank == 2, "He initialization requires a 2-D [out, in] tensor", loc=loc)
			he_initialization(info.tensor, info.tensor.shape[1])
		case Init_Xavier:
			assert(info.tensor.rank == 2, "Xavier initialization requires a 2-D [out, in] tensor", loc=loc)
			xavier_initialization(info.tensor, info.tensor.shape[1], info.tensor.shape[0])
		case Init_Normal:
			fill_normal(info.tensor, spec.mean, spec.std, loc=loc)
		case Init_Value:
			fill_value(info.tensor, spec.value, loc=loc)
		}
	}
}

registry_update :: proc(opt: ^Optimizer, list: []Parameter_Info, loc := #caller_location) {
	for info in list {
		if info.trainable {
			update(opt, info.tensor, loc=loc)
		}
	}
}

registry_copy :: proc(dst, src: []Parameter_Info, loc := #caller_location) {
	assert(builtin.len(dst) == builtin.len(src), "registries must have the same parameter count", loc=loc)
	for info, i in src {
		assert(dst[i].name == info.name, "registries must have matching parameter names", loc=loc)
		copy(dst[i].tensor, info.tensor, loc=loc)
	}
}

registry_parameters :: proc(list: []Parameter_Info, out: ^[dynamic]Parameter, prefix := "") {
	for info in list {
		if info.trainable {
			parameter_append(out, prefix, info.name, info.tensor)
		}
	}
}

@(require_results)
registry_parameter_count :: proc(list: []Parameter_Info) -> (total: int) {
	for info in list {
		if info.trainable {
			total += info.tensor.count
		}
	}
	return
}
