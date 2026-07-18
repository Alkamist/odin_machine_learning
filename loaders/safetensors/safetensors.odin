// Generic safetensors v0.3 loader
//
//   8 bytes   little-endian u64 = header_byte_count
//   N bytes   UTF-8 JSON object: {tensor_name: {dtype, shape, data_offsets}, "__metadata__": {...}}
//   ...       contiguous tensor data, indexed by data_offsets relative to the
//             byte right after the header.
//
// Tensor data is dtype-native; for BF16/F16 the bytes are little-endian
// packed u16s. The loader reads the file once, holds the bytes, and exposes
// `get_bytes` / `get_info` lookups by tensor name.
package safetensors

import "core:encoding/json"
import "core:fmt"
import "core:log"
import "core:os"
import "core:slice"
import "core:strings"

Tensor_Info :: struct {
	dtype:        string,
	shape:        []int,
	data_offsets: [2]int,
}

Loader :: struct {
	bytes:      []byte,
	data_start: int,
	tensors:    map[string]Tensor_Info,
	metadata:   map[string]string,
	_json_root: json.Value,
}

@(require_results)
load :: proc(path: string, allocator := context.allocator, loc := #caller_location) -> (loader: Loader, ok: bool) {
	context.allocator = allocator

	file_bytes, read_err := os.read_entire_file_from_path(path, allocator)
	if read_err != nil {
		log.errorf("failed to read %v: %v", path, read_err, location=loc)
		return {}, false
	}

	if len(file_bytes) < 8 {
		log.errorf("%v is shorter than the 8-byte header prefix", path, location=loc)
		delete(file_bytes)
		return {}, false
	}

	header_len_u64 := (^u64le)(raw_data(file_bytes))^
	if u64(header_len_u64) > u64(len(file_bytes) - 8) {
		log.errorf("header_len %v overruns file size %v", header_len_u64, len(file_bytes), location=loc)
		delete(file_bytes)
		return {}, false
	}
	header_len := int(header_len_u64)
	data_start := 8 + header_len

	header_bytes := file_bytes[8:8 + header_len]
	root, parse_err := json.parse(header_bytes, parse_integers = true)
	if parse_err != .None {
		log.errorf("JSON parse error %v in %v", parse_err, path, location=loc)
		delete(file_bytes)
		return {}, false
	}

	root_object, object_ok := root.(json.Object)
	if !object_ok {
		log.errorf("header root is not a JSON object", location=loc)
		json.destroy_value(root)
		delete(file_bytes)
		return {}, false
	}

	tensors: map[string]Tensor_Info
	for name, entry in root_object {
		if name == "__metadata__" {
			continue
		}

		entry_object, entry_ok := entry.(json.Object)
		if !entry_ok {
			log.errorf("tensor %q metadata is not an object", name, location=loc)
			_destroy_partial(root, file_bytes, tensors)
			return {}, false
		}

		dtype_value, dtype_present       := entry_object["dtype"]
		shape_value, shape_present       := entry_object["shape"]
		offsets_value, offsets_present   := entry_object["data_offsets"]
		if !(dtype_present && shape_present && offsets_present) {
			log.errorf("tensor %q missing dtype/shape/data_offsets", name, location=loc)
			_destroy_partial(root, file_bytes, tensors)
			return {}, false
		}

		dtype_string, dtype_string_ok := dtype_value.(string)
		shape_array,  shape_array_ok  := shape_value.(json.Array)
		offsets_array, offsets_array_ok := offsets_value.(json.Array)
		if !(dtype_string_ok && shape_array_ok && offsets_array_ok) {
			log.errorf("tensor %q has unexpected metadata types", name, location=loc)
			_destroy_partial(root, file_bytes, tensors)
			return {}, false
		}

		shape := make([]int, len(shape_array))
		element_count := i64(1)
		for axis_value, axis_index in shape_array {
			axis_int, axis_int_ok := axis_value.(json.Integer)
			if !axis_int_ok || axis_int < 0 || (axis_int > 0 && element_count > (1 << 62) / axis_int) {
				log.errorf("tensor %q shape[%v] is not a valid dimension", name, axis_index, location=loc)
				delete(shape)
				_destroy_partial(root, file_bytes, tensors)
				return {}, false
			}
			shape[axis_index] = int(axis_int)
			element_count *= axis_int
		}

		if len(offsets_array) != 2 {
			log.errorf("tensor %q data_offsets has %v entries, expected 2", name, len(offsets_array), location=loc)
			delete(shape)
			_destroy_partial(root, file_bytes, tensors)
			return {}, false
		}
		start_int, start_ok := offsets_array[0].(json.Integer)
		end_int,   end_ok   := offsets_array[1].(json.Integer)
		if !(start_ok && end_ok) {
			log.errorf("tensor %q data_offsets entries are not integers", name, location=loc)
			delete(shape)
			_destroy_partial(root, file_bytes, tensors)
			return {}, false
		}

		start := int(start_int)
		end   := int(end_int)
		if start < 0 || end < start || end > len(file_bytes) - data_start {
			log.errorf("tensor %q data_offsets [%v, %v) out of bounds", name, start, end, location=loc)
			delete(shape)
			_destroy_partial(root, file_bytes, tensors)
			return {}, false
		}
		if dtype_size, dtype_size_known := _dtype_size(dtype_string); dtype_size_known {
			if i64(end - start) != element_count * i64(dtype_size) {
				log.errorf("tensor %q byte range %v does not match shape element count %v x dtype size %v", name, end - start, element_count, dtype_size, location=loc)
				delete(shape)
				_destroy_partial(root, file_bytes, tensors)
				return {}, false
			}
		}

		tensors[name] = Tensor_Info{
			dtype        = dtype_string,
			shape        = shape,
			data_offsets = {start, end},
		}
	}

	metadata := make(map[string]string)
	if metadata_value, has_metadata := root_object["__metadata__"]; has_metadata {
		if metadata_object, metadata_ok := metadata_value.(json.Object); metadata_ok {
			for metadata_key, metadata_entry in metadata_object {
				if metadata_string, string_ok := metadata_entry.(string); string_ok {
					metadata[metadata_key] = metadata_string
				}
			}
		}
	}

	loader.bytes      = file_bytes
	loader.data_start = data_start
	loader.tensors    = tensors
	loader.metadata   = metadata
	loader._json_root = root

	return loader, true
}

@(require_results)
_dtype_size :: proc(dtype: string) -> (int, bool) {
	switch dtype {
	case "F64", "I64", "U64":         return 8, true
	case "F32", "I32", "U32":         return 4, true
	case "F16", "BF16", "I16", "U16": return 2, true
	case "I8", "U8", "BOOL":          return 1, true
	}
	return 0, false
}

_destroy_partial :: proc(root: json.Value, file_bytes: []byte, tensors: map[string]Tensor_Info) {
	tensors := tensors
	for _, info in tensors {
		delete(info.shape)
	}
	delete(tensors)
	json.destroy_value(root)
	delete(file_bytes)
}

destroy :: proc(loader: Loader) {
	for _, info in loader.tensors {
		delete(info.shape)
	}
	tensors := loader.tensors
	delete(tensors)
	metadata := loader.metadata
	delete(metadata)
	json.destroy_value(loader._json_root)
	delete(loader.bytes)
}

@(require_results)
get_info :: proc(loader: Loader, name: string) -> (info: Tensor_Info, ok: bool) {
	info, ok = loader.tensors[name]
	return
}

@(require_results)
get_bytes :: proc(loader: Loader, name: string) -> ([]byte, bool) {
	info, ok := loader.tensors[name]
	if !ok {
		return nil, false
	}
	return loader.bytes[loader.data_start + info.data_offsets[0] : loader.data_start + info.data_offsets[1]], true
}

@(require_results)
shapes_match :: proc(a, b: []int) -> bool {
	return slice.equal(a, b)
}

Entry :: struct {
	name:  string,
	dtype: string, // "F32", "BF16", ...
	shape: []int,
	bytes: []byte,
}

@(require_results)
encode :: proc(entries: []Entry, metadata: map[string]string, allocator := context.allocator) -> []byte {
	header_builder := strings.builder_make(allocator=context.temp_allocator)

	strings.write_byte(&header_builder, '{')
	data_cursor := 0
	for entry, index in entries {
		start := data_cursor
		end   := data_cursor + len(entry.bytes)
		data_cursor = end

		if index > 0 {
			strings.write_byte(&header_builder, ',')
		}
		_write_json_string(&header_builder, entry.name)
		strings.write_string(&header_builder, ":{\"dtype\":")
		_write_json_string(&header_builder, entry.dtype)
		strings.write_string(&header_builder, ",\"shape\":[")
		for dimension, axis in entry.shape {
			if axis > 0 {
				strings.write_byte(&header_builder, ',')
			}
			strings.write_int(&header_builder, dimension)
		}
		strings.write_string(&header_builder, "],\"data_offsets\":[")
		strings.write_int(&header_builder, start)
		strings.write_byte(&header_builder, ',')
		strings.write_int(&header_builder, end)
		strings.write_string(&header_builder, "]}")
	}
	if len(metadata) > 0 {
		strings.write_string(&header_builder, ",\"__metadata__\":{")
		first := true
		for key, value in metadata {
			if !first {
				strings.write_byte(&header_builder, ',')
			}
			first = false
			_write_json_string(&header_builder, key)
			strings.write_byte(&header_builder, ':')
			_write_json_string(&header_builder, value)
		}
		strings.write_byte(&header_builder, '}')
	}
	strings.write_byte(&header_builder, '}')

	header     := strings.to_string(header_builder)
	header_len := len(header)

	total := 8 + header_len + data_cursor
	result := make([]byte, total, allocator=allocator)
	(^u64le)(raw_data(result))^ = u64le(header_len)
	copy(result[8:], header)

	offset := 8 + header_len
	for entry in entries {
		copy(result[offset:], entry.bytes)
		offset += len(entry.bytes)
	}
	return result
}

@(require_results)
save :: proc(path: string, entries: []Entry, metadata: map[string]string, loc := #caller_location) -> bool {
	bytes := encode(entries, metadata, allocator=context.temp_allocator)

	tmp_path := strings.concatenate({path, ".tmp"}, allocator=context.temp_allocator)
	if write_err := os.write_entire_file_from_bytes(tmp_path, bytes); write_err != nil {
		log.errorf("failed to write %v: %v", tmp_path, write_err, location=loc)
		return false
	}

	os.remove(path)
	if rename_err := os.rename(tmp_path, path); rename_err != nil {
		log.errorf("failed to rename %v -> %v: %v", tmp_path, path, rename_err, location=loc)
		os.remove(tmp_path)
		return false
	}
	return true
}

_write_json_string :: proc(builder: ^strings.Builder, value: string) {
	strings.write_byte(builder, '"')
	for index in 0 ..< len(value) {
		character := value[index]
		switch character {
		case '"':  strings.write_string(builder, "\\\"")
		case '\\': strings.write_string(builder, "\\\\")
		case '\n': strings.write_string(builder, "\\n")
		case '\r': strings.write_string(builder, "\\r")
		case '\t': strings.write_string(builder, "\\t")
		case:
			if character < 0x20 {
				fmt.sbprintf(builder, "\\u%04x", character)
			} else {
				strings.write_byte(builder, character)
			}
		}
	}
	strings.write_byte(builder, '"')
}
