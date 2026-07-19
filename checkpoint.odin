package machine_learning

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:log"
import "core:strconv"
import "core:strings"

import st "loaders/safetensors"

CHECKPOINT_VERSION :: "1"

@(require_results)
checkpoint_save :: proc(path: string, r: ^Registry, opt: ^Optimizer, metadata: map[string]string, loc := #caller_location) -> bool {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	entries := builtin.make([dynamic]st.Entry, allocator=context.temp_allocator)

	for parameter in r.parameters {
		if .Checkpoint not_in parameter.flags {
			continue
		}
		tensor := parameter.tensor
		assert(
			tensor.type == .F32 || tensor.type == .Bf16,
			"only F32/BF16 tensors are checkpointable (quantized base weights are frozen state, saved by reference)",
			loc = loc,
		)

		shape := builtin.make([]int, tensor.rank, allocator=context.temp_allocator)
		builtin.copy(shape, tensor.shape[:tensor.rank])

		data_bytes := builtin.make([]byte, buffer_byte_count(tensor, .Data), allocator=context.temp_allocator)
		get_bytes(tensor, .Data, data_bytes, loc=loc)
		append(&entries, st.Entry{
			name  = parameter.name,
			dtype = _checkpoint_dtype_string(tensor.type),
			shape = shape,
			bytes = data_bytes,
		})

		if state, present := _optimizer_state_lookup(opt, tensor); present {
			moment_byte_count := _data_byte_count(.F32, tensor.count)
			m_bytes := builtin.make([]byte, moment_byte_count, allocator=context.temp_allocator)
			v_bytes := builtin.make([]byte, moment_byte_count, allocator=context.temp_allocator)
			tensor.backend.buffer_get(state.m, m_bytes, loc)
			tensor.backend.buffer_get(state.v, v_bytes, loc)
			append(&entries, st.Entry{name=fmt.tprintf("%s.adam_m", parameter.name), dtype="F32", shape=shape, bytes=m_bytes})
			append(&entries, st.Entry{name=fmt.tprintf("%s.adam_v", parameter.name), dtype="F32", shape=shape, bytes=v_bytes})
		}
	}

	full_metadata := builtin.make(map[string]string, allocator=context.temp_allocator)
	for key, value in metadata {
		full_metadata[key] = value
	}
	full_metadata["ml.checkpoint_version"] = CHECKPOINT_VERSION
	if opt != nil {
		full_metadata["ml.optimizer_iteration"] = fmt.tprintf("%v", opt.iteration)
	}

	return st.save(path, entries[:], full_metadata, loc=loc)
}

@(require_results)
checkpoint_load :: proc(path: string, r: ^Registry, opt: ^Optimizer, loc := #caller_location) -> (metadata: map[string]string, ok: bool) {
	loader, load_ok := st.load(path, loc=loc)
	if !load_ok {
		return
	}
	defer st.destroy(loader)

	for parameter in r.parameters {
		if .Checkpoint not_in parameter.flags {
			continue
		}
		tensor := parameter.tensor

		info, present := st.get_info(loader, parameter.name)
		if !present {
			log.errorf("missing tensor %q in %v", parameter.name, path, location=loc)
			return
		}
		expected_dtype := _checkpoint_dtype_string(tensor.type)
		if info.dtype != expected_dtype {
			log.errorf("tensor %q dtype mismatch (file %v, model %v)", parameter.name, info.dtype, expected_dtype, location=loc)
			return
		}
		if !st.shapes_match(info.shape, tensor.shape[:tensor.rank]) {
			log.errorf("tensor %q shape mismatch (file %v, model %v)", parameter.name, info.shape, tensor.shape[:tensor.rank], location=loc)
			return
		}
		file_bytes, _ := st.get_bytes(loader, parameter.name)
		expected_bytes := buffer_byte_count(tensor, .Data)
		if builtin.len(file_bytes) != expected_bytes {
			log.errorf("tensor %q byte count mismatch (file %v, expected %v)", parameter.name, builtin.len(file_bytes), expected_bytes, location=loc)
			return
		}

		if opt != nil {
			moment_byte_count := _data_byte_count(.F32, tensor.count)
			for suffix in ([]string{"adam_m", "adam_v"}) {
				moment_name := fmt.tprintf("%s.%s", parameter.name, suffix)
				moment_info, moment_present := st.get_info(loader, moment_name)
				if !moment_present {
					continue
				}
				if moment_info.dtype != "F32" {
					log.errorf("moment %q must be F32, got %v", moment_name, moment_info.dtype, location=loc)
					return
				}
				moment_bytes, _ := st.get_bytes(loader, moment_name)
				if builtin.len(moment_bytes) != moment_byte_count {
					log.errorf("moment %q byte count mismatch (file %v, expected %v)", moment_name, builtin.len(moment_bytes), moment_byte_count, location=loc)
					return
				}
			}
		}
	}

	for parameter in r.parameters {
		if .Checkpoint not_in parameter.flags {
			continue
		}
		tensor := parameter.tensor
		file_bytes, _ := st.get_bytes(loader, parameter.name)
		set_bytes(tensor, .Data, file_bytes, loc=loc)

		if opt != nil {
			m_bytes, m_ok := st.get_bytes(loader, fmt.tprintf("%s.adam_m", parameter.name))
			v_bytes, v_ok := st.get_bytes(loader, fmt.tprintf("%s.adam_v", parameter.name))
			if m_ok && v_ok {
				state := _optimizer_state(opt, tensor, loc)
				tensor.backend.buffer_set(state.m, m_bytes, loc)
				tensor.backend.buffer_set(state.v, v_bytes, loc)
			}
		}
	}

	if opt != nil {
		opt.iteration = checkpoint_metadata_u64(loader.metadata, "ml.optimizer_iteration", fallback=opt.iteration)
	}

	metadata = builtin.make(map[string]string)
	for key, value in loader.metadata {
		metadata[strings.clone(key)] = strings.clone(value)
	}
	ok = true
	return
}

checkpoint_metadata_destroy :: proc(metadata: map[string]string) {
	metadata := metadata
	for key, value in metadata {
		delete(key)
		delete(value)
	}
	delete(metadata)
}

@(require_results)
checkpoint_metadata_u64 :: proc(metadata: map[string]string, key: string, fallback: u64 = 0) -> u64 {
	if value, present := metadata[key]; present {
		if parsed, parse_ok := strconv.parse_u64(value); parse_ok {
			return parsed
		}
	}
	return fallback
}

@(require_results)
_checkpoint_dtype_string :: proc(type: Data_Type) -> string {
	#partial switch type {
	case .F32:  return "F32"
	case .Bf16: return "BF16"
	}
	return ""
}
