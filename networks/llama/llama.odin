package llama

import "base:builtin"
import "core:fmt"
import "core:math"

import ml "../../"

Config :: struct {
	layer_count:       int,
	n_q_heads:         int,
	n_kv_heads:        int,
	head_size:         int,
	hidden_size:       int,
	intermediate_size: int,
	vocab_size:        int,
	rope_base:         f32,
	tied_embeddings:   bool,
}

SMOLLM2_135M_CONFIG :: Config{
	layer_count       = 30,
	n_q_heads         = 9,
	n_kv_heads        = 3,
	head_size         = 64,
	hidden_size       = 576,
	intermediate_size = 1536,
	vocab_size        = 49152,
	rope_base         = 100000,
	tied_embeddings   = true,
}

Layer :: struct {
	input_norm_weight:     ml.Tensor, // [hidden_size]
	q_proj_weight:         ml.Tensor, // [n_q_heads  * head_size, hidden_size]
	k_proj_weight:         ml.Tensor, // [n_kv_heads * head_size, hidden_size]
	v_proj_weight:         ml.Tensor, // [n_kv_heads * head_size, hidden_size]
	o_proj_weight:         ml.Tensor, // [hidden_size, n_q_heads * head_size]
	post_attn_norm_weight: ml.Tensor, // [hidden_size]
	gate_proj_weight:      ml.Tensor, // [intermediate_size, hidden_size]
	up_proj_weight:        ml.Tensor, // [intermediate_size, hidden_size]
	down_proj_weight:      ml.Tensor, // [hidden_size, intermediate_size]
}

Llama :: struct {
	config: Config,

	token_embeddings:   ml.Tensor, // [vocab_size, hidden_size]

	layers: []Layer,

	output_norm_weight: ml.Tensor, // [hidden_size]
	lm_head_weight:     ml.Tensor, // [vocab_size, hidden_size]; aliases token_embeddings when tied.

	params: ml.Registry,
}

make :: proc(config: Config, dtype: ml.Data_Type = .F32, trainable := true, allocator := context.allocator) -> (model: Llama) {
	context.allocator = allocator

	q_size  := config.n_q_heads  * config.head_size
	kv_size := config.n_kv_heads * config.head_size

	residual_scale := 0.02 / math.sqrt(f32(2 * config.layer_count))

	flags := ml.PARAMETER_DEFAULT_FLAGS if trainable else ml.Parameter_Flags{}

	model.config = config
	model.layers = builtin.make([]Layer, config.layer_count)

	model.token_embeddings = ml.parameter_make(&model.params, "", "model.embed_tokens.weight", dtype, {config.vocab_size, config.hidden_size}, init=ml.Init_Normal{mean=0, std=0.02}, flags=flags)

	for &layer, i in model.layers {
		prefix := fmt.tprintf("model.layers.%v", i)

		layer.input_norm_weight = ml.parameter_make(&model.params, prefix, "input_layernorm.weight",  dtype, {config.hidden_size}, init=ml.Init_Value{value=1}, flags=flags)
		layer.q_proj_weight     = ml.parameter_make(&model.params, prefix, "self_attn.q_proj.weight", dtype, {q_size, config.hidden_size}, init=ml.Init_Normal{mean=0, std=0.02}, flags=flags)
		layer.k_proj_weight     = ml.parameter_make(&model.params, prefix, "self_attn.k_proj.weight", dtype, {kv_size, config.hidden_size}, init=ml.Init_Normal{mean=0, std=0.02}, flags=flags)
		layer.v_proj_weight     = ml.parameter_make(&model.params, prefix, "self_attn.v_proj.weight", dtype, {kv_size, config.hidden_size}, init=ml.Init_Normal{mean=0, std=0.02}, flags=flags)
		layer.o_proj_weight     = ml.parameter_make(&model.params, prefix, "self_attn.o_proj.weight", dtype, {config.hidden_size, q_size}, init=ml.Init_Normal{mean=0, std=residual_scale}, flags=flags)

		layer.post_attn_norm_weight = ml.parameter_make(&model.params, prefix, "post_attention_layernorm.weight", dtype, {config.hidden_size}, init=ml.Init_Value{value=1}, flags=flags)
		layer.gate_proj_weight      = ml.parameter_make(&model.params, prefix, "mlp.gate_proj.weight", dtype, {config.intermediate_size, config.hidden_size}, init=ml.Init_Normal{mean=0, std=0.02}, flags=flags)
		layer.up_proj_weight        = ml.parameter_make(&model.params, prefix, "mlp.up_proj.weight",   dtype, {config.intermediate_size, config.hidden_size}, init=ml.Init_Normal{mean=0, std=0.02}, flags=flags)
		layer.down_proj_weight      = ml.parameter_make(&model.params, prefix, "mlp.down_proj.weight", dtype, {config.hidden_size, config.intermediate_size}, init=ml.Init_Normal{mean=0, std=residual_scale}, flags=flags)
	}

	model.output_norm_weight = ml.parameter_make(&model.params, "", "model.norm.weight", dtype, {config.hidden_size}, init=ml.Init_Value{value=1}, flags=flags)

	if config.tied_embeddings {
		model.lm_head_weight = model.token_embeddings
	} else {
		model.lm_head_weight = ml.parameter_make(&model.params, "", "lm_head.weight", dtype, {config.vocab_size, config.hidden_size}, init=ml.Init_Normal{mean=0, std=0.02}, flags=flags)
	}

	randomize(model)

	return
}

destroy :: proc(model: Llama) {
	model := model
	ml.registry_destroy(&model.params)
	delete(model.layers)
}

randomize :: proc(model: Llama) {
	model := model
	ml.registry_randomize(&model.params)
}

parameters :: proc(model: Llama, dst: ^ml.Registry) {
	model := model
	ml.registry_gather(dst, &model.params)
}

Cache :: ml.Kv_Cache

// Cache dtype tracks the model's weight dtype: K/V projections produce values
// in that dtype, so storing them otherwise would cost a per-token cast.
cache_make :: proc(model: Llama, t_max: int, allocator := context.allocator) -> (cache: Cache) {
	kv_size := model.config.n_kv_heads * model.config.head_size
	cache_type := model.token_embeddings.type

	cache.t_max  = t_max
	cache.length = 0
	cache.layers = builtin.make([]ml.Kv_Layer_Cache, len(model.layers), allocator)

	for &layer_cache in cache.layers {
		layer_cache.k = ml.alloc(cache_type, {t_max, kv_size}, persistent=true, buffers={.Data})
		layer_cache.v = ml.alloc(cache_type, {t_max, kv_size}, persistent=true, buffers={.Data})
	}

	return
}

@(require_results)
_forward :: proc(model: Llama, tokens: []int, cache: ^Cache = nil, logits_mode := ml.Logits_Mode.All, loc := #caller_location) -> (output: ml.Tensor) {
	position_offset := 0
	if cache != nil {
		ml.kv_cache_check(cache^, builtin.len(tokens), len(model.layers), loc=loc)
		position_offset = cache.length
	}

	output = ml.select(model.token_embeddings, tokens)

	residual := output

	for layer, i in model.layers {
		normed := ml.rmsnorm(residual, layer.input_norm_weight)

		q := ml.linear(normed, layer.q_proj_weight)
		k := ml.linear(normed, layer.k_proj_weight)
		v := ml.linear(normed, layer.v_proj_weight)

		q = ml.rope(q, model.config.n_q_heads,  base=model.config.rope_base, position_offset=position_offset)
		k = ml.rope(k, model.config.n_kv_heads, base=model.config.rope_base, position_offset=position_offset)

		attn_output: ml.Tensor
		if cache != nil {
			attn_output = ml.attention_with_cache(
				q, k, v,
				cache.layers[i].k, cache.layers[i].v,
				position_offset,
				model.config.n_q_heads,
				model.config.n_kv_heads,
			)
		} else {
			attn_output = ml.attention(q, k, v, model.config.n_q_heads, model.config.n_kv_heads)
		}
		attn_output = ml.linear(attn_output, layer.o_proj_weight)

		residual = ml.add(residual, attn_output)

		normed = ml.rmsnorm(residual, layer.post_attn_norm_weight)

		gate       := ml.linear(normed, layer.gate_proj_weight)
		up         := ml.linear(normed, layer.up_proj_weight)
		mlp_output := ml.linear(ml.mul(ml.silu(gate), up), layer.down_proj_weight)

		residual = ml.add(residual, mlp_output)
	}

	output = ml.rmsnorm(residual, model.output_norm_weight)
	token_count := builtin.len(tokens)
	if logits_mode == .Last && token_count > 1 {
		output = ml.slice_leading(output, token_count - 1, token_count)
	}
	output = ml.linear(output, model.lm_head_weight)
	if output.type != .F32 {
		output = ml.cast_to(output, .F32)
	}

	if cache != nil {
		cache.length += builtin.len(tokens)
	}

	return
}

@(require_results)
forward :: proc(model: Llama, tokens: []int, loc := #caller_location) -> (output: ml.Tensor) {
	return _forward(model, tokens, loc=loc)
}

@(require_results)
forward_cached :: proc(model: Llama, cache: ^Cache, new_tokens: []int, logits_mode := ml.Logits_Mode.All, loc := #caller_location) -> (output: ml.Tensor) {
	return _forward(model, new_tokens, cache=cache, logits_mode=logits_mode, loc=loc)
}

update :: proc(opt: ^ml.Optimizer, model: Llama) {
	model := model
	ml.registry_update(opt, &model.params)
}
