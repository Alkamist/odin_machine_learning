package safetensors

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:slice"

// Generic safetensors v0.3 loader. The on-disk format is:
//
//   8 bytes   little-endian u64 = header_byte_count
//   N bytes   UTF-8 JSON object: {tensor_name: {dtype, shape, data_offsets}, "__metadata__": {...}}
//   ...       contiguous tensor data, indexed by data_offsets relative to the
//             byte right after the header.
//
// Tensor data is dtype-native; for BF16/F16 the bytes are little-endian
// packed u16s. The loader reads the file once, holds the bytes, and exposes
// `get_bytes` / `get_info` lookups by tensor name.

Tensor_Info :: struct {
	dtype:        string,
	shape:        []int,
	data_offsets: [2]int,
}

Loader :: struct {
	bytes:      []byte,
	data_start: int,
	tensors:    map[string]Tensor_Info,
	// Parsed JSON tree that owns the dtype/shape allocations referenced by
	// `tensors`; freed in `destroy`.
	_json_root: json.Value,
}

@(require_results)
load :: proc(path: string, allocator := context.allocator) -> (loader: Loader, ok: bool) {
	context.allocator = allocator

	file_bytes, read_err := os.read_entire_file_from_path(path, allocator)
	if read_err != nil {
		fmt.eprintfln("safetensors.load: failed to read %v: %v", path, read_err)
		return {}, false
	}

	if len(file_bytes) < 8 {
		fmt.eprintfln("safetensors.load: %v is shorter than the 8-byte header prefix", path)
		delete(file_bytes)
		return {}, false
	}

	header_len := int((^u64le)(raw_data(file_bytes))^)
	if header_len + 8 > len(file_bytes) {
		fmt.eprintfln("safetensors.load: header_len %v overruns file size %v", header_len, len(file_bytes))
		delete(file_bytes)
		return {}, false
	}

	header_bytes := file_bytes[8:8 + header_len]
	root, parse_err := json.parse(header_bytes, parse_integers = true)
	if parse_err != .None {
		fmt.eprintfln("safetensors.load: JSON parse error %v in %v", parse_err, path)
		delete(file_bytes)
		return {}, false
	}

	root_object, object_ok := root.(json.Object)
	if !object_ok {
		fmt.eprintfln("safetensors.load: header root is not a JSON object")
		json.destroy_value(root)
		delete(file_bytes)
		return {}, false
	}

	tensors: map[string]Tensor_Info
	for name, entry in root_object {
		if name == "__metadata__" do continue

		entry_object, entry_ok := entry.(json.Object)
		if !entry_ok {
			fmt.eprintfln("safetensors.load: tensor %q metadata is not an object", name)
			json.destroy_value(root)
			delete(file_bytes)
			delete(tensors)
			return {}, false
		}

		dtype_value, dtype_present       := entry_object["dtype"]
		shape_value, shape_present       := entry_object["shape"]
		offsets_value, offsets_present   := entry_object["data_offsets"]
		if !(dtype_present && shape_present && offsets_present) {
			fmt.eprintfln("safetensors.load: tensor %q missing dtype/shape/data_offsets", name)
			json.destroy_value(root)
			delete(file_bytes)
			delete(tensors)
			return {}, false
		}

		dtype_string, dtype_string_ok := dtype_value.(string)
		shape_array,  shape_array_ok  := shape_value.(json.Array)
		offsets_array, offsets_array_ok := offsets_value.(json.Array)
		if !(dtype_string_ok && shape_array_ok && offsets_array_ok) {
			fmt.eprintfln("safetensors.load: tensor %q has unexpected metadata types", name)
			json.destroy_value(root)
			delete(file_bytes)
			delete(tensors)
			return {}, false
		}

		shape := make([]int, len(shape_array))
		for axis_value, axis_index in shape_array {
			axis_int, axis_int_ok := axis_value.(json.Integer)
			if !axis_int_ok {
				fmt.eprintfln("safetensors.load: tensor %q shape[%v] is not an integer", name, axis_index)
				delete(shape)
				json.destroy_value(root)
				delete(file_bytes)
				delete(tensors)
				return {}, false
			}
			shape[axis_index] = int(axis_int)
		}

		if len(offsets_array) != 2 {
			fmt.eprintfln("safetensors.load: tensor %q data_offsets has %v entries, expected 2", name, len(offsets_array))
			delete(shape)
			json.destroy_value(root)
			delete(file_bytes)
			delete(tensors)
			return {}, false
		}
		start_int, start_ok := offsets_array[0].(json.Integer)
		end_int,   end_ok   := offsets_array[1].(json.Integer)
		if !(start_ok && end_ok) {
			fmt.eprintfln("safetensors.load: tensor %q data_offsets entries are not integers", name)
			delete(shape)
			json.destroy_value(root)
			delete(file_bytes)
			delete(tensors)
			return {}, false
		}

		tensors[name] = Tensor_Info{
			dtype        = dtype_string,
			shape        = shape,
			data_offsets = {int(start_int), int(end_int)},
		}
	}

	loader.bytes      = file_bytes
	loader.data_start = 8 + header_len
	loader.tensors    = tensors
	loader._json_root = root
	return loader, true
}

destroy :: proc(loader: Loader) {
	for _, info in loader.tensors {
		delete(info.shape)
	}
	tensors := loader.tensors
	delete(tensors)
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
	if !ok do return nil, false
	return loader.bytes[loader.data_start + info.data_offsets[0] : loader.data_start + info.data_offsets[1]], true
}

@(require_results)
shapes_match :: proc(a, b: []int) -> bool {
	return slice.equal(a, b)
}
