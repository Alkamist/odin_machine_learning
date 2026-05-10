package machine_learning_network_gemma

import "base:builtin"
import "core:fmt"
import "core:math"
import "core:slice"

import ml          "../../"
import safetensors "../../loaders/safetensors"

@(require_results)
load_safetensors :: proc(model: Gemma, path: string) -> bool {
	loader, load_ok := safetensors.load(path)
	if !load_ok {
		return false
	}
	defer safetensors.destroy(loader)

	if !_load_named(loader, model.embed_tokens_weight, "model.language_model.embed_tokens.weight") {
		return false
	}
	if !_load_named(loader, model.output_norm_weight,  "model.language_model.norm.weight") {
		return false
	}
	if !_load_per_layer_embedding_host(loader, model) {
		return false
	}
	if !_load_named(loader, model.per_layer_model_projection_weight, "model.language_model.per_layer_model_projection.weight") {
		return false
	}
	if !_load_named(loader, model.per_layer_projection_norm_weight,  "model.language_model.per_layer_projection_norm.weight") {
		return false
	}

	cfg := model.config
	for layer, layer_idx in model.layers {
		head_dim := config_head_dim(cfg, layer_idx)
		prefix   := fmt.tprintf("model.language_model.layers.%v", layer_idx)

		ok := _load_named         (loader, layer.input_norm_weight,                fmt.tprintf("%v.input_layernorm.weight",            prefix)) &&
		      _load_named         (loader, layer.post_attention_norm_weight,       fmt.tprintf("%v.post_attention_layernorm.weight",   prefix)) &&
		      _load_named         (loader, layer.pre_feedforward_norm_weight,      fmt.tprintf("%v.pre_feedforward_layernorm.weight",  prefix)) &&
		      _load_named         (loader, layer.post_feedforward_norm_weight,     fmt.tprintf("%v.post_feedforward_layernorm.weight", prefix)) &&
		      _load_rope_permuted (loader, layer.q_proj_weight,                    fmt.tprintf("%v.self_attn.q_proj.weight",           prefix), cfg.num_attention_heads, head_dim) &&
		      _load_named         (loader, layer.o_proj_weight,                    fmt.tprintf("%v.self_attn.o_proj.weight",           prefix)) &&
		      _load_per_head_dim_permuted(loader, layer.q_norm_weight,             fmt.tprintf("%v.self_attn.q_norm.weight",           prefix), head_dim, math.sqrt(f32(head_dim))) &&
		      _load_named         (loader, layer.gate_proj_weight,                 fmt.tprintf("%v.mlp.gate_proj.weight",              prefix)) &&
		      _load_named         (loader, layer.up_proj_weight,                   fmt.tprintf("%v.mlp.up_proj.weight",                prefix)) &&
		      _load_named         (loader, layer.down_proj_weight,                 fmt.tprintf("%v.mlp.down_proj.weight",              prefix)) &&
		      _load_named         (loader, layer.per_layer_input_gate_weight,      fmt.tprintf("%v.per_layer_input_gate.weight",       prefix)) &&
		      _load_named         (loader, layer.per_layer_projection_weight,      fmt.tprintf("%v.per_layer_projection.weight",       prefix)) &&
		      _load_named         (loader, layer.post_per_layer_input_norm_weight, fmt.tprintf("%v.post_per_layer_input_norm.weight",  prefix)) &&
		      _load_named         (loader, layer.layer_scalar,                     fmt.tprintf("%v.layer_scalar",                      prefix))
		if !ok {
			return false
		}

		if !is_kv_shared_layer(cfg, layer_idx) {
			ok2 := _load_rope_permuted(loader, layer.k_proj_weight, fmt.tprintf("%v.self_attn.k_proj.weight", prefix), cfg.num_key_value_heads, head_dim) &&
			       _load_named        (loader, layer.v_proj_weight, fmt.tprintf("%v.self_attn.v_proj.weight", prefix)) &&
			       _load_per_head_dim_permuted(loader, layer.k_norm_weight, fmt.tprintf("%v.self_attn.k_norm.weight", prefix), head_dim, 1.0)
			if !ok2 {
				return false
			}
		}
	}

	if !cfg.tie_word_embeddings {
		if !_load_named(loader, model.lm_head_weight, "lm_head.weight") {
			return false
		}
	}

	return true
}

_load_per_layer_embedding_host :: proc(loader: safetensors.Loader, model: Gemma) -> bool {
	name := "model.language_model.embed_tokens_per_layer.weight"
	info, info_ok := safetensors.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load: missing %q", name)
		return false
	}
	cfg := model.config
	expected_shape := []int{cfg.vocab_size, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input}
	if !slice.equal(info.shape, expected_shape) {
		fmt.eprintfln("gemma.load: %q shape %v != expected %v", name, info.shape, expected_shape)
		return false
	}

	raw_bytes, bytes_ok := safetensors.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	count := cfg.vocab_size * cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
	#partial switch model.dtype {
	case .Bf16:
		switch info.dtype {
		case "BF16":
			if builtin.len(raw_bytes) != count * 2 {
				fmt.eprintfln("gemma.load: %q BF16 byte count mismatch", name)
				return false
			}
			builtin.copy(model.embed_tokens_per_layer_bytes, raw_bytes)
		case "F32":
			if builtin.len(raw_bytes) != count * 4 {
				fmt.eprintfln("gemma.load: %q F32 byte count mismatch", name)
				return false
			}
			src := slice.from_ptr((^f32)(raw_data(raw_bytes)), count)
			dst := ([^]ml.Bf16)(raw_data(model.embed_tokens_per_layer_bytes))
			for v, i in src {
				dst[i] = ml.bf16_from_f32(v)
			}
		case:
			fmt.eprintfln("gemma.load: %q unsupported source dtype %q", name, info.dtype)
			return false
		}
	case .F32:
		switch info.dtype {
		case "F32":
			if builtin.len(raw_bytes) != count * 4 {
				fmt.eprintfln("gemma.load: %q F32 byte count mismatch", name)
				return false
			}
			builtin.copy(model.embed_tokens_per_layer_bytes, raw_bytes)
		case "BF16":
			if builtin.len(raw_bytes) != count * 2 {
				fmt.eprintfln("gemma.load: %q BF16 byte count mismatch", name)
				return false
			}
			src := slice.from_ptr((^ml.Bf16)(raw_data(raw_bytes)), count)
			dst := ([^]f32)(raw_data(model.embed_tokens_per_layer_bytes))
			for v, i in src {
				dst[i] = ml.bf16_to_f32(v)
			}
		case:
			fmt.eprintfln("gemma.load: %q unsupported source dtype %q", name, info.dtype)
			return false
		}
	}
	return true
}

_load_named :: proc(loader: safetensors.Loader, target: ml.Tensor, name: string) -> bool {
	info, info_ok := safetensors.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_safetensors: missing tensor %q", name)
		return false
	}
	shape_buffer := target.shape
	target_shape := shape_buffer[:target.rank]
	if !slice.equal(info.shape, target_shape) {
		fmt.eprintfln("gemma.load_safetensors: %q shape %v doesn't match model tensor shape %v",
			name, info.shape, target_shape)
		return false
	}

	raw_bytes, bytes_ok := safetensors.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}
	return _write_target(target, info, raw_bytes, name)
}

_write_target :: proc(target: ml.Tensor, info: safetensors.Tensor_Info, raw_bytes: []byte, name: string) -> bool {
	count := ml.len(target)
	#partial switch target.type {
	case .F32:
		floats := builtin.make([]f32, count, context.temp_allocator)
		if !_decode_dtype_bytes(info, raw_bytes, floats) {
			return false
		}
		ml.set_data(target, floats)
	case .Bf16:
		bytes := builtin.make([]byte, count * 2, context.temp_allocator)
		bf    := ([^]ml.Bf16)(raw_data(bytes))
		switch info.dtype {
		case "BF16":
			if builtin.len(raw_bytes) != count * 2 {
				fmt.eprintfln("gemma.load: %q BF16 byte count %v != expected %v", name, builtin.len(raw_bytes), count * 2)
				return false
			}
			builtin.copy(bytes, raw_bytes)
		case "F32":
			if builtin.len(raw_bytes) != count * 4 {
				fmt.eprintfln("gemma.load: %q F32 byte count %v != expected %v", name, builtin.len(raw_bytes), count * 4)
				return false
			}
			src := slice.from_ptr((^f32)(raw_data(raw_bytes)), count)
			for v, i in src {
				bf[i] = ml.bf16_from_f32(v)
			}
		case:
			fmt.eprintfln("gemma.load: %q unsupported source dtype %q for Bf16 target", name, info.dtype)
			return false
		}
		ml.set_data_bytes(target, bytes)
	}
	return true
}

_load_rope_permuted :: proc(loader: safetensors.Loader, target: ml.Tensor, name: string, head_count, head_size: int) -> bool {
	info, info_ok := safetensors.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_safetensors: missing tensor %q", name)
		return false
	}
	target_shape_buffer := target.shape
	target_shape := target_shape_buffer[:target.rank]
	if !slice.equal(info.shape, target_shape) {
		fmt.eprintfln("gemma.load_safetensors: %q shape %v doesn't match model tensor shape %v",
			name, info.shape, target_shape)
		return false
	}
	if target.rank != 2 || target.shape[0] != head_count * head_size {
		fmt.eprintfln("gemma.load_safetensors: %q expected [%v, embed], got %v", name, head_count * head_size, target_shape)
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

	_write_floats_to_target(target, permuted, name)
	return true
}

_write_floats_to_target :: proc(target: ml.Tensor, src: []f32, name: string) {
	count := ml.len(target)
	assert(builtin.len(src) == count, name)
	#partial switch target.type {
	case .F32:
		ml.set_data(target, src)
	case .Bf16:
		bytes := builtin.make([]byte, count * 2, context.temp_allocator)
		bf    := ([^]ml.Bf16)(raw_data(bytes))
		for v, i in src {
			bf[i] = ml.bf16_from_f32(v)
		}
		ml.set_data_bytes(target, bytes)
	}
}

_load_per_head_dim_permuted :: proc(loader: safetensors.Loader, target: ml.Tensor, name: string, head_size: int, extra_scale: f32) -> bool {
	info, info_ok := safetensors.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_safetensors: missing tensor %q", name)
		return false
	}
	if target.rank != 1 || target.shape[0] != head_size {
		fmt.eprintfln("gemma.load_safetensors: %q expected 1-D [%v], got rank=%v shape[0]=%v", name, head_size, target.rank, target.shape[0])
		return false
	}
	if !slice.equal(info.shape, []int{head_size}) {
		fmt.eprintfln("gemma.load_safetensors: %q HF shape %v expected [%v]", name, info.shape, head_size)
		return false
	}

	raw_bytes, bytes_ok := safetensors.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	source := builtin.make([]f32, head_size, context.temp_allocator)
	if !_decode_dtype_bytes(info, raw_bytes, source) {
		return false
	}

	permuted := builtin.make([]f32, head_size, context.temp_allocator)
	half_size := head_size / 2
	for i in 0 ..< half_size {
		permuted[2 * i + 0] = source[i]              * extra_scale
		permuted[2 * i + 1] = source[half_size + i]  * extra_scale
	}

	_write_floats_to_target(target, permuted, name)
	return true
}

_decode_dtype_bytes :: proc(info: safetensors.Tensor_Info, raw_bytes: []byte, dst: []f32) -> bool {
	count := builtin.len(dst)
	switch info.dtype {
	case "F32":
		if builtin.len(raw_bytes) != count * 4 {
			fmt.eprintfln("gemma.load_safetensors: F32 byte count %v != expected %v", builtin.len(raw_bytes), count * 4)
			return false
		}
		builtin.copy(dst, slice.from_ptr((^f32)(raw_data(raw_bytes)), count))
	case "BF16":
		if builtin.len(raw_bytes) != count * 2 {
			fmt.eprintfln("gemma.load_safetensors: BF16 byte count %v != expected %v", builtin.len(raw_bytes), count * 2)
			return false
		}
		bf := slice.from_ptr((^ml.Bf16)(raw_data(raw_bytes)), count)
		for value, index in bf {
			dst[index] = ml.bf16_to_f32(value)
		}
	case:
		fmt.eprintfln("gemma.load_safetensors: unsupported dtype %q (only F32 and BF16 implemented)", info.dtype)
		return false
	}
	return true
}