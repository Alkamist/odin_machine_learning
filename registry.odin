package machine_learning

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:strings"

Parameter_Flag  :: enum {
	Train,
	Checkpoint,
	Owned,
}
Parameter_Flags :: bit_set[Parameter_Flag]
PARAMETER_DEFAULT_FLAGS :: Parameter_Flags{.Train, .Checkpoint}

Init_None   :: struct {}
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
	Init_None,
	Init_He,
	Init_Xavier,
	Init_Normal,
	Init_Value,
}

Parameter :: struct {
	name:   string,
	tensor: Tensor,
	init:   Init,
	flags:  Parameter_Flags,
}

Registry :: struct {
	parameters: [dynamic]Parameter,
}

_registry_clone_name :: proc(r: ^Registry, prefix, name: string) -> string {
	if r.parameters.allocator.procedure == nil {
		r.parameters.allocator = context.allocator
	}
	if prefix == "" {
		return strings.clone(name, allocator=r.parameters.allocator)
	}
	return fmt.aprintf("%s.%s", prefix, name, allocator=r.parameters.allocator)
}

_registry_normalize_init :: proc(init: Init, flags: Parameter_Flags, loc: runtime.Source_Code_Location) -> Init {
	if init == nil {
		assert(.Train not_in flags, "trainable parameter requires an init; pass Init_None if it is filled by a loader or by hand", loc=loc)
		return Init_None{}
	}
	return init
}

@(require_results)
parameter_make :: proc(r: ^Registry, prefix, name: string, type: Data_Type, shape: []int, init: Init = nil, flags := PARAMETER_DEFAULT_FLAGS, loc := #caller_location) -> (t: Tensor) {
	normalized := _registry_normalize_init(init, flags, loc)
	buffers    := DEFAULT_PARAMETER_BUFFERS if .Train in flags else Buffer_Set{.Data}
	t = alloc(type, shape, persistent=true, buffers=buffers, loc=loc)
	append(&r.parameters, Parameter{name=_registry_clone_name(r, prefix, name), tensor=t, init=normalized, flags=flags + {.Owned}})
	return
}

parameter_register :: proc(r: ^Registry, prefix, name: string, tensor: Tensor, init: Init = nil, flags := PARAMETER_DEFAULT_FLAGS, loc := #caller_location) {
	assert(.Train not_in flags || has_gradient(tensor), "trainable parameter requires a gradient buffer", loc=loc)
	normalized := _registry_normalize_init(init, flags, loc)
	append(&r.parameters, Parameter{name=_registry_clone_name(r, prefix, name), tensor=tensor, init=normalized, flags=flags})
}

registry_destroy :: proc(r: ^Registry, loc := #caller_location) {
	for parameter in r.parameters {
		if .Owned in parameter.flags {
			destroy(parameter.tensor, loc=loc)
		}
		builtin.delete(parameter.name, allocator=r.parameters.allocator, loc=loc)
	}
	builtin.delete(r.parameters, loc=loc)
	r^ = {}
}

registry_clear :: proc(r: ^Registry, loc := #caller_location) {
	for parameter in r.parameters {
		builtin.delete(parameter.name, allocator=r.parameters.allocator, loc=loc)
	}
	builtin.clear(&r.parameters)
}

registry_randomize :: proc(r: ^Registry, loc := #caller_location) {
	for parameter in r.parameters {
		switch spec in parameter.init {
		case Init_None:
		case Init_He:
			assert(parameter.tensor.rank == 2, "He initialization requires a 2-D [out, in] tensor", loc=loc)
			he_initialization(parameter.tensor, parameter.tensor.shape[1])
		case Init_Xavier:
			assert(parameter.tensor.rank == 2, "Xavier initialization requires a 2-D [out, in] tensor", loc=loc)
			xavier_initialization(parameter.tensor, parameter.tensor.shape[1], parameter.tensor.shape[0])
		case Init_Normal:
			fill_normal(parameter.tensor, spec.mean, spec.std, loc=loc)
		case Init_Value:
			fill_value(parameter.tensor, spec.value, loc=loc)
		}
	}
}

registry_update :: proc(opt: ^Optimizer, r: ^Registry, loc := #caller_location) {
	for parameter in r.parameters {
		if .Train in parameter.flags {
			update(opt, parameter.tensor, loc=loc)
		}
	}
}

registry_copy :: proc(dst, src: ^Registry, loc := #caller_location) {
	assert(builtin.len(dst.parameters) == builtin.len(src.parameters), "registries must have the same parameter count", loc=loc)
	for parameter, i in src.parameters {
		assert(dst.parameters[i].name == parameter.name, "registries must have matching parameter names", loc=loc)
		copy(dst.parameters[i].tensor, parameter.tensor, loc=loc)
	}
}

registry_gather :: proc(dst, src: ^Registry, prefix := "", loc := #caller_location) {
	for parameter in src.parameters {
		append(&dst.parameters, Parameter{name=_registry_clone_name(dst, prefix, parameter.name), tensor=parameter.tensor, init=parameter.init, flags=parameter.flags - {.Owned}})
	}
}

@(require_results)
registry_element_count :: proc(r: ^Registry, flags := Parameter_Flags{.Train}) -> (total: int) {
	for parameter in r.parameters {
		if flags <= parameter.flags {
			total += parameter.tensor.count
		}
	}
	return
}

registry_read :: proc(r: ^Registry, dst: []f32, loc := #caller_location) {
	assert(builtin.len(dst) == registry_element_count(r, flags=Parameter_Flags{.Train}), "dst length must equal the trainable element count", loc=loc)
	index := 0
	for parameter in r.parameters {
		if .Train in parameter.flags {
			get_data(parameter.tensor, dst[index:index + parameter.tensor.count], loc=loc)
			index += parameter.tensor.count
		}
	}
}

registry_write :: proc(r: ^Registry, src: []f32, loc := #caller_location) {
	assert(builtin.len(src) == registry_element_count(r, flags=Parameter_Flags{.Train}), "src length must equal the trainable element count", loc=loc)
	index := 0
	for parameter in r.parameters {
		if .Train in parameter.flags {
			set_data(parameter.tensor, src[index:index + parameter.tensor.count], loc=loc)
			index += parameter.tensor.count
		}
	}
}
