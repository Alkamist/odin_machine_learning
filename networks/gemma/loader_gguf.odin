package gemma

import "base:builtin"
import "core:fmt"
import "core:log"
import "core:math"
import "core:slice"

import ml "../../"
import    "../../loaders/gguf"
import    "../../loaders/weights"

@(require_results)
load_gguf :: proc(model: ^Gemma, path: string) -> bool {
	loader, load_ok := gguf.load(path)
	if !load_ok {
		return false
	}
	defer gguf.destroy(loader)

	if !_validate_gguf_metadata(loader, model.config) {
		return false
	}

	source := weights.from_gguf(&loader)
	cfg    := model.config

	if !_load_dequant_to_bf16(loader, model.embed_tokens_weight, "token_embd.weight") {
		return false
	}
	if !_load_per_layer_token_embd(loader, model^) {
		return false
	}
	ok := weights.write_tensor(&model.output_norm_weight,                source, "output_norm.weight") &&
	      weights.write_tensor(&model.per_layer_model_projection_weight, source, "per_layer_model_proj.weight") &&
	      weights.write_tensor(&model.per_layer_projection_norm_weight,  source, "per_layer_proj_norm.weight")
	if !ok {
		return false
	}

	for &layer, layer_idx in model.layers {
		head_dim     := config_head_dim(cfg, layer_idx)
		q_norm_scale := math.sqrt(f32(head_dim))
		prefix       := fmt.tprintf("blk.%v", layer_idx)

		layer_ok := weights.write_tensor(&layer.input_norm_weight,                source, fmt.tprintf("%v.attn_norm.weight",           prefix)) &&
		            weights.write_tensor(&layer.post_attention_norm_weight,       source, fmt.tprintf("%v.post_attention_norm.weight", prefix)) &&
		            weights.write_tensor(&layer.pre_feedforward_norm_weight,      source, fmt.tprintf("%v.ffn_norm.weight",            prefix)) &&
		            weights.write_tensor(&layer.post_feedforward_norm_weight,     source, fmt.tprintf("%v.post_ffw_norm.weight",       prefix)) &&
		            weights.write_tensor(&layer.post_per_layer_input_norm_weight, source, fmt.tprintf("%v.post_norm.weight",           prefix)) &&
		            weights.write_tensor(&layer.layer_scalar,                     source, fmt.tprintf("%v.layer_output_scale.weight",  prefix)) &&
		            _load_norm_f32_to_dtype(loader, layer.q_norm_weight,          fmt.tprintf("%v.attn_q_norm.weight",         prefix), q_norm_scale) &&
		            weights.write_tensor(&layer.q_proj_weight,                    source, fmt.tprintf("%v.attn_q.weight",              prefix), .Rope_Permute, cfg.num_attention_heads, head_dim) &&
		            weights.write_tensor(&layer.o_proj_weight,                    source, fmt.tprintf("%v.attn_output.weight",         prefix)) &&
		            weights.write_tensor(&layer.gate_proj_weight,                 source, fmt.tprintf("%v.ffn_gate.weight",            prefix)) &&
		            weights.write_tensor(&layer.up_proj_weight,                   source, fmt.tprintf("%v.ffn_up.weight",              prefix)) &&
		            weights.write_tensor(&layer.down_proj_weight,                 source, fmt.tprintf("%v.ffn_down.weight",            prefix)) &&
		            weights.write_tensor(&layer.per_layer_input_gate_weight,      source, fmt.tprintf("%v.inp_gate.weight",            prefix)) &&
		            weights.write_tensor(&layer.per_layer_projection_weight,      source, fmt.tprintf("%v.proj.weight",               prefix))
		if !layer_ok {
			return false
		}

		if !is_kv_shared_layer(cfg, layer_idx) {
			kv_ok := weights.write_tensor(&layer.k_proj_weight, source, fmt.tprintf("%v.attn_k.weight", prefix), .Rope_Permute, cfg.num_key_value_heads, head_dim) &&
			         weights.write_tensor(&layer.v_proj_weight, source, fmt.tprintf("%v.attn_v.weight", prefix)) &&
			         _load_norm_f32_to_dtype(loader, layer.k_norm_weight, fmt.tprintf("%v.attn_k_norm.weight", prefix), 1.0)
			if !kv_ok {
				return false
			}
		}
	}

	if !cfg.tie_word_embeddings {
		log.errorf("untied lm_head not present in this GGUF; not implemented")
		return false
	}

	ml.registry_clear(&model.params)
	_register_parameters(model)

	return true
}

// Validates the GGUF's architecture metadata against the config we allocated
// for, so a mismatch fails here with a clear message instead of a tensor-shape
// crash deep in the load. Deriving the full config from the file is a stretch
// goal; this only checks the dimensions we can cheaply cross-reference.
_validate_gguf_metadata :: proc(loader: gguf.Loader, cfg: Config) -> bool {
	check :: proc(loader: gguf.Loader, key: string, expected: int) -> bool {
		value, ok := gguf.get_u32(loader, key)
		if !ok {
			return true // key absent: nothing to cross-check
		}
		if int(value) != expected {
			log.errorf("GGUF metadata %q = %v does not match config %v", key, value, expected)
			return false
		}
		return true
	}
	return check(loader, "gemma4.block_count",                        cfg.num_hidden_layers)           &&
	       check(loader, "gemma4.embedding_length",                   cfg.hidden_size)                 &&
	       check(loader, "gemma4.embedding_length_per_layer_input",   cfg.hidden_size_per_layer_input) &&
	       check(loader, "gemma4.feed_forward_length",                cfg.intermediate_size)           &&
	       check(loader, "gemma4.attention.head_count",               cfg.num_attention_heads)         &&
	       check(loader, "gemma4.attention.head_count_kv",            cfg.num_key_value_heads)
}

_load_norm_f32_to_dtype :: proc(loader: gguf.Loader, target: ml.Tensor, name: string, extra_scale: f32) -> bool {
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		log.errorf("missing tensor %q", name)
		return false
	}
	if info.type != .F32 {
		log.errorf("%q expected F32 norm, got %v", name, info.type)
		return false
	}
	shape_buf := target.shape
	target_shape := shape_buf[:target.rank]
	if !_shape_matches_reversed(info.shape, target_shape) {
		log.errorf("%q shape %v doesn't match target shape %v", name, info.shape, target_shape)
		return false
	}

	bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	count := ml.len(target)
	src   := slice.from_ptr((^f32)(raw_data(bytes)), count)

	#partial switch target.type {
	case .F32:
		if extra_scale == 1.0 {
			ml.set_data_bytes(target, bytes)
		} else {
			scaled := builtin.make([]f32, count, context.temp_allocator)
			for v, i in src {
				scaled[i] = v * extra_scale
			}
			ml.set_data(target, scaled)
		}
	case .Bf16:
		bytes_out := builtin.make([]byte, count * 2, context.temp_allocator)
		bf := ([^]ml.Bf16)(raw_data(bytes_out))
		for v, i in src {
			bf[i] = ml.bf16_from_f32(v * extra_scale)
		}
		ml.set_data_bytes(target, bytes_out)
	case:
		log.errorf("%q unsupported target dtype %v", name, target.type)
		return false
	}
	return true
}

_load_dequant_to_bf16 :: proc(loader: gguf.Loader, target: ml.Tensor, name: string) -> bool {
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		log.errorf("missing tensor %q", name)
		return false
	}
	shape_buf := target.shape
	target_shape := shape_buf[:target.rank]
	if !_shape_matches_reversed(info.shape, target_shape) {
		log.errorf("%q shape %v doesn't match target shape %v", name, info.shape, target_shape)
		return false
	}

	bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	count := ml.len(target)
	floats := builtin.make([]f32, count, context.temp_allocator)

	#partial switch info.type {
	case .Q6_K: ml.dequantize_q6_k(bytes, floats)
	case .Q4_K: ml.dequantize_q4_k(bytes, floats)
	case:
		log.errorf("%q expected Q4_K or Q6_K source for dequant load, got %v", name, info.type)
		return false
	}

	#partial switch target.type {
	case .F32:
		ml.set_data(target, floats)
	case .Bf16:
		bytes_out := builtin.make([]byte, count * 2, context.temp_allocator)
		bf := ([^]ml.Bf16)(raw_data(bytes_out))
		for v, i in floats {
			bf[i] = ml.bf16_from_f32(v)
		}
		ml.set_data_bytes(target, bytes_out)
	case:
		log.errorf("%q unsupported embed target dtype %v", name, target.type)
		return false
	}
	return true
}

_load_per_layer_token_embd :: proc(loader: gguf.Loader, model: Gemma) -> bool {
	name := "per_layer_token_embd.weight"
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		log.errorf("missing %q", name)
		return false
	}
	cfg := model.config
	expected_unreversed := []int{cfg.vocab_size, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input}
	if !_shape_matches_reversed(info.shape, expected_unreversed) {
		log.errorf("%q shape %v != expected (reversed of %v)", name, info.shape, expected_unreversed)
		return false
	}

	bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	count := cfg.vocab_size * cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
	#partial switch info.type {
	case .BF16:
		if model.dtype != .Bf16 {
			log.errorf("%q BF16 source needs Bf16 model dtype (got %v)", name, model.dtype)
			return false
		}
		if builtin.len(bytes) != count * 2 {
			log.errorf("%q BF16 byte count %v != expected %v", name, builtin.len(bytes), count * 2)
			return false
		}
		builtin.copy(model.embed_tokens_per_layer_bytes, bytes)
	case .F32:
		if model.dtype == .F32 {
			builtin.copy(model.embed_tokens_per_layer_bytes, bytes)
		} else {
			src := slice.from_ptr((^f32)(raw_data(bytes)), count)
			dst := ([^]ml.Bf16)(raw_data(model.embed_tokens_per_layer_bytes))
			for v, i in src {
				dst[i] = ml.bf16_from_f32(v)
			}
		}
	case:
		log.errorf("%q unsupported source dtype %v", name, info.type)
		return false
	}
	return true
}

_shape_matches_reversed :: proc(gguf_shape, target_shape: []int) -> bool {
	if builtin.len(gguf_shape) != builtin.len(target_shape) {
		return false
	}
	n := builtin.len(gguf_shape)
	for i in 0 ..< n {
		if gguf_shape[i] != target_shape[n - 1 - i] {
			return false
		}
	}
	return true
}
