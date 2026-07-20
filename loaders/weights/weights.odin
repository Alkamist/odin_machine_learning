package weights

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:log"
import "core:slice"

import ml "../../"
import    "../../loaders/gguf"
import    "../../loaders/safetensors"

Data_Type :: enum {
	F32,
	F16,
	BF16,
	Q4_K,
	Q6_K,
	Other,
}

Info :: struct {
	type:  Data_Type,
	shape: []int, // row-major, already normalized
}

Source :: struct {
	get_info:  proc(data: rawptr, name: string) -> (Info, bool),
	get_bytes: proc(data: rawptr, name: string) -> ([]byte, bool),
	data:      rawptr,
}

Transform :: enum {
	None,
	Rope_Permute,
}

@(require_results)
from_safetensors :: proc(loader: ^safetensors.Loader) -> Source {
	return Source{
		get_info  = _safetensors_get_info,
		get_bytes = _safetensors_get_bytes,
		data      = loader,
	}
}

@(require_results)
from_gguf :: proc(loader: ^gguf.Loader) -> Source {
	return Source{
		get_info  = _gguf_get_info,
		get_bytes = _gguf_get_bytes,
		data      = loader,
	}
}

@(require_results)
write_tensor :: proc(target: ^ml.Tensor, source: Source, name: string, transform := Transform.None, head_count := 0, head_size := 0, loc := #caller_location) -> bool {
	info, info_ok := source.get_info(source.data, name)
	if !info_ok {
		log.errorf("missing tensor %q", name, location=loc)
		return false
	}

	target_shape := target.shape[:target.rank]
	if !slice.equal(info.shape, target_shape) {
		log.errorf("tensor %q source shape %v does not match model tensor shape %v", name, info.shape, target_shape, location=loc)
		return false
	}

	raw, bytes_ok := source.get_bytes(source.data, name)
	if !bytes_ok {
		log.errorf("tensor %q has no data bytes", name, location=loc)
		return false
	}

	switch info.type {
	case .Q4_K, .Q6_K:
		return _write_quant(target, info.type, raw, name, transform, head_count, head_size, loc)
	case .F32, .BF16:
		return _write_float(target^, info.type, raw, name, transform, head_count, head_size, loc)
	case .F16, .Other:
		log.errorf("tensor %q has unsupported source dtype (only F32/BF16/Q4_K/Q6_K implemented)", name, location=loc)
		return false
	}
	return false
}

_write_quant :: proc(target: ^ml.Tensor, dtype: Data_Type, raw: []byte, name: string, transform: Transform, head_count, head_size: int, loc: runtime.Source_Code_Location) -> bool {
	ml_dtype:    ml.Data_Type
	block_bytes: int
	#partial switch dtype {
	case .Q4_K: ml_dtype = .Q4_K; block_bytes = ml.Q4_K_BLOCK_BYTES
	case .Q6_K: ml_dtype = .Q6_K; block_bytes = ml.Q6_K_BLOCK_BYTES
	case:       return false
	}

	target_shape := target.shape[:target.rank]
	if target.count % ml.K_QUANT_BLOCK_SIZE != 0 {
		log.errorf("tensor %q element count %v is not a multiple of 256", name, target.count, location=loc)
		return false
	}
	expected := (target.count / ml.K_QUANT_BLOCK_SIZE) * block_bytes
	if builtin.len(raw) != expected {
		log.errorf("tensor %q quant byte count %v != expected %v", name, builtin.len(raw), expected, location=loc)
		return false
	}

	data := raw
	if transform == .Rope_Permute {
		if head_count <= 0 || head_size <= 0 {
			log.errorf("tensor %q rope permute requires head_count/head_size", name, location=loc)
			return false
		}
		if target.rank != 2 || target_shape[0] != head_count * head_size {
			log.errorf("tensor %q rope permute expected [%v, embed], got %v", name, head_count * head_size, target_shape, location=loc)
			return false
		}
		embedding := target_shape[1]
		if embedding % ml.K_QUANT_BLOCK_SIZE != 0 {
			log.errorf("tensor %q embedding %v is not a multiple of 256", name, embedding, location=loc)
			return false
		}
		row_bytes := (embedding / ml.K_QUANT_BLOCK_SIZE) * block_bytes
		permuted  := builtin.make([]byte, expected, context.temp_allocator)
		half      := head_size / 2
		for h in 0 ..< head_count {
			head_offset := h * head_size * row_bytes
			for i in 0 ..< half {
				even_dst := head_offset + (2 * i + 0)     * row_bytes
				odd_dst  := head_offset + (2 * i + 1)     * row_bytes
				even_src := head_offset + (i)             * row_bytes
				odd_src  := head_offset + (half + i)      * row_bytes
				builtin.copy(permuted[even_dst:even_dst + row_bytes], data[even_src:even_src + row_bytes])
				builtin.copy(permuted[odd_dst :odd_dst  + row_bytes], data[odd_src :odd_src  + row_bytes])
			}
		}
		data = permuted
	}

	ml.destroy(target^, loc)
	new_target := ml.alloc(ml_dtype, target_shape, persistent=true, buffers=ml.Buffer_Set{.Data}, loc=loc)
	ml.set_bytes(new_target, .Data, data, loc)
	target^ = new_target
	return true
}

_write_float :: proc(target: ml.Tensor, dtype: Data_Type, raw: []byte, name: string, transform: Transform, head_count, head_size: int, loc: runtime.Source_Code_Location) -> bool {
	floats := builtin.make([]f32, target.count, context.temp_allocator)
	if !_decode_floats(dtype, raw, floats, name, loc) {
		return false
	}

	if transform == .Rope_Permute {
		shape_buffer := target.shape
		target_shape := shape_buffer[:target.rank]
		if head_count <= 0 || head_size <= 0 {
			log.errorf("tensor %q rope permute requires head_count/head_size", name, location=loc)
			return false
		}
		if head_size % 2 != 0 {
			log.errorf("tensor %q rope permute requires even head_size, got %v", name, head_size, location=loc)
			return false
		}
		if target.rank != 2 || target_shape[0] != head_count * head_size {
			log.errorf("tensor %q rope permute expected [%v, embed], got %v", name, head_count * head_size, target_shape, location=loc)
			return false
		}
		embedding := target_shape[1]
		permuted  := builtin.make([]f32, target.count, context.temp_allocator)
		half      := head_size / 2
		for h in 0 ..< head_count {
			head_offset := h * head_size * embedding
			for i in 0 ..< half {
				even_dst := head_offset + (2 * i + 0)     * embedding
				odd_dst  := head_offset + (2 * i + 1)     * embedding
				even_src := head_offset + (i)             * embedding
				odd_src  := head_offset + (half + i)      * embedding
				builtin.copy(permuted[even_dst:even_dst + embedding], floats[even_src:even_src + embedding])
				builtin.copy(permuted[odd_dst :odd_dst  + embedding], floats[odd_src :odd_src  + embedding])
			}
		}
		floats = permuted
	}

	return set_floats(target, floats, loc)
}

set_floats :: proc(target: ml.Tensor, floats: []f32, loc := #caller_location) -> bool {
	#partial switch target.type {
	case .F32:
		ml.set_data(target, floats, loc=loc)
	case .Bf16:
		bytes := builtin.make([]byte, target.count * 2, context.temp_allocator)
		bf    := ([^]ml.Bf16)(raw_data(bytes))[:target.count]
		for value, index in floats {
			bf[index] = ml.bf16_from_f32(value)
		}
		ml.set_bytes(target, .Data, bytes, loc=loc)
	case:
		log.errorf("set_floats does not support target dtype %v", target.type, location=loc)
		return false
	}
	return true
}

@(require_results)
read_floats :: proc(source: Source, name: string, loc := #caller_location) -> (values: []f32, info: Info, ok: bool) {
	info, ok = source.get_info(source.data, name)
	if !ok {
		log.errorf("missing tensor %q", name, location=loc)
		return
	}
	raw, bytes_ok := source.get_bytes(source.data, name)
	if !bytes_ok {
		log.errorf("tensor %q has no data bytes", name, location=loc)
		return {}, info, false
	}
	count := 1
	for dimension in info.shape {
		count *= dimension
	}
	values = builtin.make([]f32, count, context.temp_allocator)
	if !_decode_floats(info.type, raw, values, name, loc) {
		return {}, info, false
	}
	return values, info, true
}

_decode_floats :: proc(dtype: Data_Type, raw: []byte, dst: []f32, name: string, loc: runtime.Source_Code_Location) -> bool {
	count := builtin.len(dst)
	#partial switch dtype {
	case .F32:
		if builtin.len(raw) != count * 4 {
			log.errorf("tensor %q F32 byte count %v != expected %v", name, builtin.len(raw), count * 4, location=loc)
			return false
		}
		builtin.copy(dst, slice.from_ptr((^f32)(raw_data(raw)), count))
	case .BF16:
		if builtin.len(raw) != count * 2 {
			log.errorf("tensor %q BF16 byte count %v != expected %v", name, builtin.len(raw), count * 2, location=loc)
			return false
		}
		source := slice.from_ptr((^ml.Bf16)(raw_data(raw)), count)
		for value, index in source {
			dst[index] = ml.bf16_to_f32(value)
		}
	case:
		log.errorf("tensor %q cannot decode source dtype to f32", name, location=loc)
		return false
	}
	return true
}

_safetensors_get_info :: proc(data: rawptr, name: string) -> (Info, bool) {
	loader := (^safetensors.Loader)(data)
	info, ok := safetensors.get_info(loader^, name)
	if !ok {
		return {}, false
	}
	return Info{type = _safetensors_dtype(info.dtype), shape = info.shape}, true
}

_safetensors_get_bytes :: proc(data: rawptr, name: string) -> ([]byte, bool) {
	loader := (^safetensors.Loader)(data)
	return safetensors.get_bytes(loader^, name)
}

_safetensors_dtype :: proc(dtype: string) -> Data_Type {
	switch dtype {
	case "F32":  return .F32
	case "F16":  return .F16
	case "BF16": return .BF16
	}
	return .Other
}

_gguf_get_info :: proc(data: rawptr, name: string) -> (Info, bool) {
	loader := (^gguf.Loader)(data)
	info, ok := gguf.get_info(loader^, name)
	if !ok {
		return {}, false
	}
	normalized := builtin.make([]int, builtin.len(info.shape), context.temp_allocator)
	n := builtin.len(info.shape)
	for i in 0 ..< n {
		normalized[i] = info.shape[n - 1 - i]
	}
	return Info{type = _gguf_dtype(info.type), shape = normalized}, true
}

_gguf_get_bytes :: proc(data: rawptr, name: string) -> ([]byte, bool) {
	loader := (^gguf.Loader)(data)
	return gguf.get_bytes(loader^, name)
}

_gguf_dtype :: proc(type: gguf.Tensor_Type) -> Data_Type {
	#partial switch type {
	case .F32:  return .F32
	case .F16:  return .F16
	case .BF16: return .BF16
	case .Q4_K: return .Q4_K
	case .Q6_K: return .Q6_K
	}
	return .Other
}
