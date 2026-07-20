package gemma

import "base:builtin"
import "core:fmt"
import "core:log"
import "core:math"
import "core:slice"

import ml "../../"
import    "../../loaders/safetensors"
import    "../../loaders/weights"

@(require_results)
load_safetensors :: proc(model: ^Gemma, path: string, loc := #caller_location) -> bool {
	loader, load_err := safetensors.load(path)
	if load_err != .None {
		return false
	}
	defer safetensors.destroy(loader)

	source := weights.from_safetensors(&loader)
	cfg    := model.config

	ok := weights.write_tensor(&model.embed_tokens_weight, source, "model.language_model.embed_tokens.weight") &&
	      weights.write_tensor(&model.output_norm_weight,  source, "model.language_model.norm.weight") &&
	      _load_per_layer_embedding_host(loader, model^, loc=loc) &&
	      weights.write_tensor(&model.per_layer_model_projection_weight, source, "model.language_model.per_layer_model_projection.weight") &&
	      weights.write_tensor(&model.per_layer_projection_norm_weight,  source, "model.language_model.per_layer_projection_norm.weight")
	if !ok {
		return false
	}
	if cfg.tied_embeddings {
		model.lm_head_weight = model.embed_tokens_weight
	}

	for &layer, layer_idx in model.layers {
		head_dim := config_head_dim(cfg, layer_idx)
		prefix   := fmt.tprintf("model.language_model.layers.%v", layer_idx)

		layer_ok := weights.write_tensor(&layer.input_norm_weight,                source, fmt.tprintf("%v.input_layernorm.weight",            prefix)) &&
		            weights.write_tensor(&layer.post_attention_norm_weight,       source, fmt.tprintf("%v.post_attention_layernorm.weight",   prefix)) &&
		            weights.write_tensor(&layer.pre_feedforward_norm_weight,      source, fmt.tprintf("%v.pre_feedforward_layernorm.weight",  prefix)) &&
		            weights.write_tensor(&layer.post_feedforward_norm_weight,     source, fmt.tprintf("%v.post_feedforward_layernorm.weight", prefix)) &&
		            weights.write_tensor(&layer.q_proj_weight,                    source, fmt.tprintf("%v.self_attn.q_proj.weight",           prefix), .Rope_Permute, cfg.n_q_heads, head_dim) &&
		            weights.write_tensor(&layer.o_proj_weight,                    source, fmt.tprintf("%v.self_attn.o_proj.weight",           prefix)) &&
		            _load_norm_permuted (source, layer.q_norm_weight,             fmt.tprintf("%v.self_attn.q_norm.weight",                   prefix), head_dim, math.sqrt(f32(head_dim)), loc=loc) &&
		            weights.write_tensor(&layer.gate_proj_weight,                 source, fmt.tprintf("%v.mlp.gate_proj.weight",              prefix)) &&
		            weights.write_tensor(&layer.up_proj_weight,                   source, fmt.tprintf("%v.mlp.up_proj.weight",                prefix)) &&
		            weights.write_tensor(&layer.down_proj_weight,                 source, fmt.tprintf("%v.mlp.down_proj.weight",              prefix)) &&
		            weights.write_tensor(&layer.per_layer_input_gate_weight,      source, fmt.tprintf("%v.per_layer_input_gate.weight",       prefix)) &&
		            weights.write_tensor(&layer.per_layer_projection_weight,      source, fmt.tprintf("%v.per_layer_projection.weight",       prefix)) &&
		            weights.write_tensor(&layer.post_per_layer_input_norm_weight, source, fmt.tprintf("%v.post_per_layer_input_norm.weight",  prefix)) &&
		            weights.write_tensor(&layer.layer_scalar,                     source, fmt.tprintf("%v.layer_scalar",                      prefix))
		if !layer_ok {
			return false
		}

		if !is_kv_shared_layer(cfg, layer_idx) {
			kv_ok := weights.write_tensor(&layer.k_proj_weight, source, fmt.tprintf("%v.self_attn.k_proj.weight", prefix), .Rope_Permute, cfg.n_kv_heads, head_dim) &&
			         weights.write_tensor(&layer.v_proj_weight, source, fmt.tprintf("%v.self_attn.v_proj.weight", prefix)) &&
			         _load_norm_permuted(source, layer.k_norm_weight, fmt.tprintf("%v.self_attn.k_norm.weight", prefix), head_dim, 1.0, loc=loc)
			if !kv_ok {
				return false
			}
		}
	}

	if !cfg.tied_embeddings {
		if !weights.write_tensor(&model.lm_head_weight, source, "lm_head.weight") {
			return false
		}
	}

	return true
}

// q_norm/k_norm weights are stored per-head-dim in the split (low half, high
// half) layout, so they need the same even/odd permutation as the projections,
// and q_norm bakes in the sqrt(head_dim) attention scale. This is specific to
// the safetensors export; the GGUF export stores them already permuted.
_load_norm_permuted :: proc(source: weights.Source, target: ml.Tensor, name: string, head_size: int, extra_scale: f32, loc := #caller_location) -> bool {
	values, info, ok := weights.read_floats(source, name)
	if !ok {
		return false
	}
	if !slice.equal(info.shape, []int{head_size}) {
		log.errorf("%q source shape %v expected [%v]", name, info.shape, head_size, location=loc)
		return false
	}
	if target.rank != 1 || target.shape[0] != head_size {
		log.errorf("%q expected 1-D [%v], got rank=%v shape[0]=%v", name, head_size, target.rank, target.shape[0], location=loc)
		return false
	}

	permuted  := builtin.make([]f32, head_size, context.temp_allocator)
	half_size := head_size / 2
	for i in 0 ..< half_size {
		permuted[2 * i + 0] = values[i]             * extra_scale
		permuted[2 * i + 1] = values[half_size + i] * extra_scale
	}
	return weights.set_floats(target, permuted)
}

_load_per_layer_embedding_host :: proc(loader: safetensors.Loader, model: Gemma, loc := #caller_location) -> bool {
	name := "model.language_model.embed_tokens_per_layer.weight"
	info, info_ok := safetensors.get_info(loader, name)
	if !info_ok {
		log.errorf("missing %q", name, location=loc)
		return false
	}
	cfg := model.config
	expected_shape := []int{cfg.vocab_size, cfg.layer_count * cfg.hidden_size_per_layer_input}
	if !slice.equal(info.shape, expected_shape) {
		log.errorf("%q shape %v != expected %v", name, info.shape, expected_shape, location=loc)
		return false
	}

	raw_bytes, bytes_ok := safetensors.get_bytes(loader, name)
	if !bytes_ok {
		return false
	}

	count := cfg.vocab_size * cfg.layer_count * cfg.hidden_size_per_layer_input
	#partial switch model.dtype {
	case .Bf16:
		switch info.dtype {
		case "BF16":
			if builtin.len(raw_bytes) != count * 2 {
				log.errorf("%q BF16 byte count mismatch", name, location=loc)
				return false
			}
			builtin.copy(model.embed_tokens_per_layer_bytes, raw_bytes)
		case "F32":
			if builtin.len(raw_bytes) != count * 4 {
				log.errorf("%q F32 byte count mismatch", name, location=loc)
				return false
			}
			src := slice.from_ptr((^f32)(raw_data(raw_bytes)), count)
			dst := ([^]ml.Bf16)(raw_data(model.embed_tokens_per_layer_bytes))
			for v, i in src {
				dst[i] = ml.bf16_from_f32(v)
			}
		case:
			log.errorf("%q unsupported source dtype %q", name, info.dtype, location=loc)
			return false
		}
	case .F32:
		switch info.dtype {
		case "F32":
			if builtin.len(raw_bytes) != count * 4 {
				log.errorf("%q F32 byte count mismatch", name, location=loc)
				return false
			}
			builtin.copy(model.embed_tokens_per_layer_bytes, raw_bytes)
		case "BF16":
			if builtin.len(raw_bytes) != count * 2 {
				log.errorf("%q BF16 byte count mismatch", name, location=loc)
				return false
			}
			src := slice.from_ptr((^ml.Bf16)(raw_data(raw_bytes)), count)
			dst := ([^]f32)(raw_data(model.embed_tokens_per_layer_bytes))
			for v, i in src {
				dst[i] = ml.bf16_to_f32(v)
			}
		case:
			log.errorf("%q unsupported source dtype %q", name, info.dtype, location=loc)
			return false
		}
	}
	return true
}
