package machine_learning_network_llama

import "base:builtin"
import "core:fmt"
import "core:os"
import "core:slice"

import ml          "../../"
import safetensors "../../loaders/safetensors"

load_safetensors :: proc(model: Llama, path: string) -> bool {
	loader, load_ok := safetensors.load(path)
	if !load_ok {
		return false
	}
	defer safetensors.destroy(loader)

	if !_load_named(loader, model.token_embeddings, "model.embed_tokens.weight") {
		return false
	}

	for layer, i in model.layers {
		ok := _load_named(loader, layer.input_norm_weight,     fmt.tprintf("model.layers.%v.input_layernorm.weight",          i)) &&
		      _load_rope_permuted(loader, layer.q_proj_weight, fmt.tprintf("model.layers.%v.self_attn.q_proj.weight",         i), model.config.n_q_heads,  model.config.head_size) &&
		      _load_rope_permuted(loader, layer.k_proj_weight, fmt.tprintf("model.layers.%v.self_attn.k_proj.weight",         i), model.config.n_kv_heads, model.config.head_size) &&
		      _load_named(loader, layer.v_proj_weight,         fmt.tprintf("model.layers.%v.self_attn.v_proj.weight",         i)) &&
		      _load_named(loader, layer.o_proj_weight,         fmt.tprintf("model.layers.%v.self_attn.o_proj.weight",         i)) &&
		      _load_named(loader, layer.post_attn_norm_weight, fmt.tprintf("model.layers.%v.post_attention_layernorm.weight", i)) &&
		      _load_named(loader, layer.gate_proj_weight,      fmt.tprintf("model.layers.%v.mlp.gate_proj.weight",            i)) &&
		      _load_named(loader, layer.up_proj_weight,        fmt.tprintf("model.layers.%v.mlp.up_proj.weight",              i)) &&
		      _load_named(loader, layer.down_proj_weight,      fmt.tprintf("model.layers.%v.mlp.down_proj.weight",            i))
		if !ok {
			return false
		}
	}

	if !_load_named(loader, model.output_norm_weight, "model.norm.weight") {
		return false
	}

	if _, has_lm_head := safetensors.get_info(loader, "lm_head.weight"); has_lm_head {
		if !model.config.tied_embeddings {
			if !_load_named(loader, model.lm_head_weight, "lm_head.weight") {
				return false
			}
		}
	}

	return true
}

_load_rope_permuted :: proc(loader: safetensors.Loader, target: ml.Tensor, name: string, head_count, head_size: int) -> bool {
	info, info_ok := safetensors.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("llama.load_safetensors: missing tensor %q", name)
		return false
	}

	target_shape_buffer := target.shape
	target_shape := target_shape_buffer[:target.rank]
	if !slice.equal(info.shape, target_shape) {
		fmt.eprintfln("llama.load_safetensors: %q shape %v doesn't match model tensor shape %v",
			name, info.shape, target_shape)
		return false
	}
	if target.rank != 2 || target.shape[0] != head_count * head_size {
		fmt.eprintfln("llama.load_safetensors: %q expected [%v, embed], got %v", name, head_count * head_size, target_shape)
		return false
	}

	embedding_size := target.shape[1]
	half_size      := head_size / 2

	raw_bytes, bytes_ok := safetensors.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	source_floats := builtin.make([]f32, ml.len(target), context.temp_allocator)
	if !_decode_dtype_bytes(info, raw_bytes, source_floats) {
		return false
	}

	permuted := builtin.make([]f32, ml.len(target), context.temp_allocator)
	for h in 0 ..< head_count {
		head_offset := h * head_size * embedding_size
		for i in 0 ..< half_size {
			even_dst := head_offset + (2 * i + 0) * embedding_size
			odd_dst  := head_offset + (2 * i + 1) * embedding_size
			even_src := head_offset + (i)             * embedding_size
			odd_src  := head_offset + (half_size + i) * embedding_size
			builtin.copy(permuted[even_dst:even_dst + embedding_size], source_floats[even_src:even_src + embedding_size])
			builtin.copy(permuted[odd_dst :odd_dst  + embedding_size], source_floats[odd_src :odd_src  + embedding_size])
		}
	}

	_set_target_from_floats(target, permuted)
	return true
}

_decode_dtype_bytes :: proc(info: safetensors.Tensor_Info, raw_bytes: []byte, dst: []f32) -> bool {
	count := builtin.len(dst)
	switch info.dtype {
	case "F32":
		if builtin.len(raw_bytes) != count * 4 {
			fmt.eprintfln("llama.load_safetensors: F32 byte count %v != expected %v", builtin.len(raw_bytes), count * 4)
			return false
		}
		builtin.copy(dst, slice.from_ptr((^f32)(raw_data(raw_bytes)), count))
	case "BF16":
		if builtin.len(raw_bytes) != count * 2 {
			fmt.eprintfln("llama.load_safetensors: BF16 byte count %v != expected %v", builtin.len(raw_bytes), count * 2)
			return false
		}
		bf := slice.from_ptr((^ml.Bf16)(raw_data(raw_bytes)), count)
		for value, index in bf {
			dst[index] = ml.bf16_to_f32(value)
		}
	case:
		fmt.eprintfln("llama.load_safetensors: unsupported dtype %q (only F32 and BF16 implemented)", info.dtype)
		return false
	}
	return true
}

_load_named :: proc(loader: safetensors.Loader, target: ml.Tensor, name: string) -> bool {
	info, info_ok := safetensors.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("llama.load_safetensors: missing tensor %q", name)
		return false
	}

	shape_buffer := target.shape
	target_shape := shape_buffer[:target.rank]
	if !slice.equal(info.shape, target_shape) {
		fmt.eprintfln("llama.load_safetensors: %q shape %v doesn't match model tensor shape %v",
			name, info.shape, target_shape)
		return false
	}

	raw_bytes, bytes_ok := safetensors.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	floats := builtin.make([]f32, ml.len(target), context.temp_allocator)
	if !_decode_dtype_bytes(info, raw_bytes, floats) {
		return false
	}
	_set_target_from_floats(target, floats)
	return true
}

// Writes f32 source values into the target tensor regardless of its dtype.
// For Bf16 targets, converts element-wise via bf16_from_f32.
_set_target_from_floats :: proc(target: ml.Tensor, floats: []f32, loc := #caller_location) {
	switch target.type {
	case .F32:
		ml.set_data(target, floats)
	case .Bf16:
		count := builtin.len(floats)
		byte_buf := builtin.make([]byte, count * 2, context.temp_allocator)
		bf := ([^]ml.Bf16)(raw_data(byte_buf))[:count]
		for i in 0 ..< count {
			bf[i] = ml.bf16_from_f32(floats[i])
		}
		ml.set_data_bytes(target, byte_buf)
	case .Q4_K, .Q6_K:
		fmt.panicf("llama loader: target dtype %v not supported", target.type, loc=loc)
	}
}

// Loads weights saved by examples/reascript_smollm in the SMLW0001 format.
//
// File layout (LE):
//   magic "SMLW0001" (8 bytes)
//   tensor_count (i32)
//   per tensor:
//     i32 name_len + name bytes
//     i32 dtype (0=F32, 1=BF16)
//     i32 rank + rank * i32 shape
//     payload (count * dtype_bytes)
//
// Tensor names follow the same scheme as load_safetensors. Q/K bytes are
// already in the model's pair-interleaved order (saved straight from the
// trained tensors), so no rope-permutation is applied here.
load_smlw :: proc(model: Llama, path: string) -> bool {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("llama.load_smlw: could not read %v: %v", path, err)
		return false
	}
	defer delete(bytes)

	if builtin.len(bytes) < 12 || string(bytes[:8]) != "SMLW0001" {
		fmt.eprintfln("llama.load_smlw: %v is not an SMLW0001 file", path)
		return false
	}

	tensor_count := int((^i32)(&bytes[8])^)
	cursor := 12

	tensors_by_name: map[string][]byte
	tensors_by_name.allocator = context.temp_allocator
	tensor_dtypes:    map[string]i32
	tensor_dtypes.allocator   = context.temp_allocator

	for i in 0 ..< tensor_count {
		if cursor + 4 > builtin.len(bytes) {
			fmt.eprintfln("llama.load_smlw: truncated at tensor %v name_len", i)
			return false
		}
		name_len := int((^i32)(&bytes[cursor])^)
		cursor += 4
		if cursor + name_len > builtin.len(bytes) {
			fmt.eprintfln("llama.load_smlw: truncated at tensor %v name", i)
			return false
		}
		name := string(bytes[cursor : cursor + name_len])
		cursor += name_len

		if cursor + 8 > builtin.len(bytes) {
			fmt.eprintfln("llama.load_smlw: truncated at tensor %q dtype/rank", name)
			return false
		}
		dtype := (^i32)(&bytes[cursor])^
		cursor += 4
		rank := int((^i32)(&bytes[cursor])^)
		cursor += 4

		if cursor + rank * 4 > builtin.len(bytes) {
			fmt.eprintfln("llama.load_smlw: truncated at tensor %q shape", name)
			return false
		}
		count := 1
		for d in 0 ..< rank {
			dim := int((^i32)(&bytes[cursor])^)
			count *= dim
			cursor += 4
		}

		bytes_per_elem := 4 if dtype == 0 else 2
		payload_bytes  := count * bytes_per_elem
		if cursor + payload_bytes > builtin.len(bytes) {
			fmt.eprintfln("llama.load_smlw: truncated at tensor %q payload", name)
			return false
		}
		tensors_by_name[name] = bytes[cursor : cursor + payload_bytes]
		tensor_dtypes[name]   = dtype
		cursor += payload_bytes
	}

	apply :: proc(tensors: map[string][]byte, dtypes: map[string]i32, target: ml.Tensor, name: string) -> bool {
		raw, ok := tensors[name]
		if !ok {
			fmt.eprintfln("llama.load_smlw: missing tensor %q", name)
			return false
		}
		dtype := dtypes[name]
		count := ml.len(target)
		floats := builtin.make([]f32, count, context.temp_allocator)
		switch dtype {
		case 0:
			if builtin.len(raw) != count * 4 {
				fmt.eprintfln("llama.load_smlw: %q F32 byte count mismatch", name)
				return false
			}
			builtin.copy(floats, slice.from_ptr((^f32)(raw_data(raw)), count))
		case 1:
			if builtin.len(raw) != count * 2 {
				fmt.eprintfln("llama.load_smlw: %q BF16 byte count mismatch", name)
				return false
			}
			bf := slice.from_ptr((^ml.Bf16)(raw_data(raw)), count)
			for v, idx in bf {
				floats[idx] = ml.bf16_to_f32(v)
			}
		case:
			fmt.eprintfln("llama.load_smlw: %q unsupported dtype %v", name, dtype)
			return false
		}
		_set_target_from_floats(target, floats)
		return true
	}

	if !apply(tensors_by_name, tensor_dtypes, model.token_embeddings, "model.embed_tokens.weight") {
		return false
	}
	for layer, i in model.layers {
		ok := apply(tensors_by_name, tensor_dtypes, layer.input_norm_weight,     fmt.tprintf("model.layers.%v.input_layernorm.weight",          i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.q_proj_weight,         fmt.tprintf("model.layers.%v.self_attn.q_proj.weight",         i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.k_proj_weight,         fmt.tprintf("model.layers.%v.self_attn.k_proj.weight",         i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.v_proj_weight,         fmt.tprintf("model.layers.%v.self_attn.v_proj.weight",         i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.o_proj_weight,         fmt.tprintf("model.layers.%v.self_attn.o_proj.weight",         i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.post_attn_norm_weight, fmt.tprintf("model.layers.%v.post_attention_layernorm.weight", i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.gate_proj_weight,      fmt.tprintf("model.layers.%v.mlp.gate_proj.weight",            i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.up_proj_weight,        fmt.tprintf("model.layers.%v.mlp.up_proj.weight",              i)) &&
		      apply(tensors_by_name, tensor_dtypes, layer.down_proj_weight,      fmt.tprintf("model.layers.%v.mlp.down_proj.weight",            i))
		if !ok {
			return false
		}
	}
	if !apply(tensors_by_name, tensor_dtypes, model.output_norm_weight, "model.norm.weight") {
		return false
	}
	if !model.config.tied_embeddings {
		if _, has := tensors_by_name["lm_head.weight"]; has {
			if !apply(tensors_by_name, tensor_dtypes, model.lm_head_weight, "lm_head.weight") {
				return false
			}
		}
	}

	return true
}