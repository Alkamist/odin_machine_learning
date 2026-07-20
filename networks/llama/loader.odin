package llama

import "core:fmt"
import "core:log"

import "../../loaders/safetensors"
import "../../loaders/weights"

load_safetensors :: proc(model: ^Llama, path: string, loc := #caller_location) -> bool {
	loader, load_err := safetensors.load(path)
	if load_err != .None {
		return false
	}
	defer safetensors.destroy(loader)

	source := weights.from_safetensors(&loader)
	cfg    := model.config

	if !weights.write_tensor(&model.token_embeddings, source, "model.embed_tokens.weight") {
		return false
	}
	if cfg.tied_embeddings {
		model.lm_head_weight = model.token_embeddings
	}

	for &layer, i in model.layers {
		ok := weights.write_tensor(&layer.input_norm_weight,     source, fmt.tprintf("model.layers.%v.input_layernorm.weight",          i)) &&
		      weights.write_tensor(&layer.q_proj_weight,         source, fmt.tprintf("model.layers.%v.self_attn.q_proj.weight",         i), .Rope_Permute, cfg.n_q_heads,  cfg.head_size) &&
		      weights.write_tensor(&layer.k_proj_weight,         source, fmt.tprintf("model.layers.%v.self_attn.k_proj.weight",         i), .Rope_Permute, cfg.n_kv_heads, cfg.head_size) &&
		      weights.write_tensor(&layer.v_proj_weight,         source, fmt.tprintf("model.layers.%v.self_attn.v_proj.weight",         i)) &&
		      weights.write_tensor(&layer.o_proj_weight,         source, fmt.tprintf("model.layers.%v.self_attn.o_proj.weight",         i)) &&
		      weights.write_tensor(&layer.post_attn_norm_weight, source, fmt.tprintf("model.layers.%v.post_attention_layernorm.weight", i)) &&
		      weights.write_tensor(&layer.gate_proj_weight,      source, fmt.tprintf("model.layers.%v.mlp.gate_proj.weight",            i)) &&
		      weights.write_tensor(&layer.up_proj_weight,        source, fmt.tprintf("model.layers.%v.mlp.up_proj.weight",              i)) &&
		      weights.write_tensor(&layer.down_proj_weight,      source, fmt.tprintf("model.layers.%v.mlp.down_proj.weight",            i))
		if !ok {
			return false
		}
	}

	if !weights.write_tensor(&model.output_norm_weight, source, "model.norm.weight") {
		return false
	}

	if !cfg.tied_embeddings {
		if _, has_lm_head := safetensors.get_info(loader, "lm_head.weight"); !has_lm_head {
			log.errorf("model config is untied but %v has no lm_head.weight", path, location=loc)
			return false
		}
		if !weights.write_tensor(&model.lm_head_weight, source, "lm_head.weight") {
			return false
		}
	}

	return true
}
