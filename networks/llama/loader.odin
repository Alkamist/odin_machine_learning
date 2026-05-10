package machine_learning_network_llama

import "base:builtin"
import "core:fmt"
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

	ml.set_data(target, permuted)
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
	ml.set_data(target, floats)
	return true
}