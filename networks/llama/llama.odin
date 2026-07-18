package llama

import "base:builtin"
import "core:math"

import ml "../../"

Config :: struct {
	layer_count:       int,
	n_q_heads:         int,
	n_kv_heads:        int,
	head_size:         int,
	embedding_size:    int,
	intermediate_size: int,
	vocabulary_size:   int,
	rope_base:         f32,
	tied_embeddings:   bool,
	use_qk_norm:       bool,
}

SMOLLM2_135M_CONFIG :: Config{
	layer_count       = 30,
	n_q_heads         = 9,
	n_kv_heads        = 3,
	head_size         = 64,
	embedding_size    = 576,
	intermediate_size = 1536,
	vocabulary_size   = 49152,
	rope_base         = 100000,
	tied_embeddings   = true,
}

Layer :: struct {
	input_norm_weight:     ml.Tensor, // [embedding_size]
	q_proj_weight:         ml.Tensor, // [n_q_heads  * head_size, embedding_size]
	k_proj_weight:         ml.Tensor, // [n_kv_heads * head_size, embedding_size]
	v_proj_weight:         ml.Tensor, // [n_kv_heads * head_size, embedding_size]
	o_proj_weight:         ml.Tensor, // [embedding_size, n_q_heads * head_size]
	q_norm_weight:         ml.Tensor, // [head_size]; only allocated when config.use_qk_norm
	k_norm_weight:         ml.Tensor, // [head_size]; only allocated when config.use_qk_norm
	post_attn_norm_weight: ml.Tensor, // [embedding_size]
	gate_proj_weight:      ml.Tensor, // [intermediate_size, embedding_size]
	up_proj_weight:        ml.Tensor, // [intermediate_size, embedding_size]
	down_proj_weight:      ml.Tensor, // [embedding_size, intermediate_size]
}

Llama :: struct {
	config: Config,

	token_embeddings:   ml.Tensor, // [vocabulary_size, embedding_size]

	layers: []Layer,

	output_norm_weight: ml.Tensor, // [embedding_size]
	lm_head_weight:     ml.Tensor, // [vocabulary_size, embedding_size]; aliases token_embeddings when tied.
}

make :: proc(config: Config, dtype: ml.Data_Type = .F32, allocator := context.allocator) -> (model: Llama) {
	q_size  := config.n_q_heads  * config.head_size
	kv_size := config.n_kv_heads * config.head_size

	model.config           = config
	model.layers           = builtin.make([]Layer, config.layer_count)
	model.token_embeddings = ml.make(dtype, {config.vocabulary_size, config.embedding_size})

	for &layer in model.layers {
		layer.input_norm_weight     = ml.make(dtype, {config.embedding_size})
		layer.q_proj_weight         = ml.make(dtype, {q_size,  config.embedding_size})
		layer.k_proj_weight         = ml.make(dtype, {kv_size, config.embedding_size})
		layer.v_proj_weight         = ml.make(dtype, {kv_size, config.embedding_size})
		layer.o_proj_weight         = ml.make(dtype, {config.embedding_size, q_size})
		if config.use_qk_norm {
			layer.q_norm_weight = ml.make(dtype, {config.head_size})
			layer.k_norm_weight = ml.make(dtype, {config.head_size})
		}
		layer.post_attn_norm_weight = ml.make(dtype, {config.embedding_size})
		layer.gate_proj_weight      = ml.make(dtype, {config.intermediate_size, config.embedding_size})
		layer.up_proj_weight        = ml.make(dtype, {config.intermediate_size, config.embedding_size})
		layer.down_proj_weight      = ml.make(dtype, {config.embedding_size,    config.intermediate_size})
	}

	model.output_norm_weight = ml.make(dtype, {config.embedding_size})
	if config.tied_embeddings {
		model.lm_head_weight = model.token_embeddings
	} else {
		model.lm_head_weight = ml.make(dtype, {config.vocabulary_size, config.embedding_size})
	}

	randomize(model)

	return
}

destroy :: proc(model: Llama) {
	ml.destroy(model.token_embeddings)

	for layer in model.layers {
		ml.destroy(layer.input_norm_weight)
		ml.destroy(layer.q_proj_weight)
		ml.destroy(layer.k_proj_weight)
		ml.destroy(layer.v_proj_weight)
		ml.destroy(layer.o_proj_weight)
		ml.destroy(layer.q_norm_weight)
		ml.destroy(layer.k_norm_weight)
		ml.destroy(layer.post_attn_norm_weight)
		ml.destroy(layer.gate_proj_weight)
		ml.destroy(layer.up_proj_weight)
		ml.destroy(layer.down_proj_weight)
	}

	ml.destroy(model.output_norm_weight)
	if !model.config.tied_embeddings {
		ml.destroy(model.lm_head_weight)
	}

	delete(model.layers)
}

copy :: proc(dst, src: Llama) {
	ml.copy(dst.token_embeddings, src.token_embeddings)

	for i in 0 ..< len(dst.layers) {
		ml.copy(dst.layers[i].input_norm_weight,     src.layers[i].input_norm_weight)
		ml.copy(dst.layers[i].q_proj_weight,         src.layers[i].q_proj_weight)
		ml.copy(dst.layers[i].k_proj_weight,         src.layers[i].k_proj_weight)
		ml.copy(dst.layers[i].v_proj_weight,         src.layers[i].v_proj_weight)
		ml.copy(dst.layers[i].o_proj_weight,         src.layers[i].o_proj_weight)
		if dst.config.use_qk_norm {
			ml.copy(dst.layers[i].q_norm_weight, src.layers[i].q_norm_weight)
			ml.copy(dst.layers[i].k_norm_weight, src.layers[i].k_norm_weight)
		}
		ml.copy(dst.layers[i].post_attn_norm_weight, src.layers[i].post_attn_norm_weight)
		ml.copy(dst.layers[i].gate_proj_weight,      src.layers[i].gate_proj_weight)
		ml.copy(dst.layers[i].up_proj_weight,        src.layers[i].up_proj_weight)
		ml.copy(dst.layers[i].down_proj_weight,      src.layers[i].down_proj_weight)
	}

	ml.copy(dst.output_norm_weight, src.output_norm_weight)
	if !dst.config.tied_embeddings {
		ml.copy(dst.lm_head_weight, src.lm_head_weight)
	}
}

randomize :: proc(model: Llama) {
	layer_count := len(model.layers)
	residual_scale := 0.02 / math.sqrt(f32(2 * layer_count))

	ml.fill_normal(model.token_embeddings, 0, 0.02)

	for &layer in model.layers {
		ml.fill_value(layer.input_norm_weight, 1)
		ml.fill_normal(layer.q_proj_weight, 0, 0.02)
		ml.fill_normal(layer.k_proj_weight, 0, 0.02)
		ml.fill_normal(layer.v_proj_weight, 0, 0.02)
		ml.fill_normal(layer.o_proj_weight, 0, residual_scale)

		if model.config.use_qk_norm {
			ml.fill_value(layer.q_norm_weight, 1)
			ml.fill_value(layer.k_norm_weight, 1)
		}

		ml.fill_value(layer.post_attn_norm_weight, 1)
		ml.fill_normal(layer.gate_proj_weight, 0, 0.02)
		ml.fill_normal(layer.up_proj_weight,   0, 0.02)
		ml.fill_normal(layer.down_proj_weight, 0, residual_scale)
	}

	ml.fill_value(model.output_norm_weight, 1)
	if !model.config.tied_embeddings {
		ml.fill_normal(model.lm_head_weight, 0, 0.02)
	}
}

Layer_Cache :: struct {
	k: ml.Tensor, // [t_max, n_kv_heads * head_size]
	v: ml.Tensor,
}

Cache :: struct {
	t_max:  int,
	length: int,
	layers: []Layer_Cache,
}

// Cache dtype tracks the model's weight dtype: K/V projections produce values
// in that dtype, so storing them otherwise would cost a per-token cast.
cache_make :: proc(model: Llama, t_max: int, allocator := context.allocator) -> (cache: Cache) {
	kv_size := model.config.n_kv_heads * model.config.head_size
	cache_type := model.token_embeddings.type

	cache.t_max  = t_max
	cache.length = 0
	cache.layers = builtin.make([]Layer_Cache, len(model.layers), allocator)

	for &layer_cache in cache.layers {
		layer_cache.k = ml.alloc(cache_type, {t_max, kv_size}, persistent=true, buffers={.Data})
		layer_cache.v = ml.alloc(cache_type, {t_max, kv_size}, persistent=true, buffers={.Data})
	}

	return
}

cache_destroy :: proc(cache: Cache) {
	for layer_cache in cache.layers {
		ml.destroy(layer_cache.k)
		ml.destroy(layer_cache.v)
	}
	delete(cache.layers)
}

cache_reset :: proc(cache: ^Cache) {
	cache.length = 0
}

@(require_results)
_per_head_rmsnorm :: proc(x: ml.Tensor, weight: ml.Tensor, head_count: int) -> ml.Tensor {
	token_count := x.shape[0]
	head_size   := x.shape[1] / head_count
	view        := ml.reshape(x, {token_count * head_count, head_size})
	normed      := ml.rmsnorm(view, weight)
	return ml.reshape(normed, {token_count, head_count * head_size})
}

@(require_results)
forward_cached :: proc(model: Llama, cache: ^Cache, new_tokens: []int) -> (output: ml.Tensor) {
	token_count := builtin.len(new_tokens)
	assert(token_count > 0,                          "forward_cached requires at least one new token")
	assert(cache.length + token_count <= cache.t_max, "forward_cached would overflow KV cache")
	assert(len(cache.layers) == len(model.layers),    "cache layer count must match model")

	cache_position := cache.length

	output = ml.select(model.token_embeddings, new_tokens)

	residual := output

	for layer, i in model.layers {
		normed := ml.rmsnorm(residual, layer.input_norm_weight)

		q := ml.linear(normed, layer.q_proj_weight)
		k := ml.linear(normed, layer.k_proj_weight)
		v := ml.linear(normed, layer.v_proj_weight)

		if model.config.use_qk_norm {
			q = _per_head_rmsnorm(q, layer.q_norm_weight, model.config.n_q_heads)
			k = _per_head_rmsnorm(k, layer.k_norm_weight, model.config.n_kv_heads)
		}

		q = ml.rope(q, model.config.n_q_heads,  model.config.rope_base, cache_position)
		k = ml.rope(k, model.config.n_kv_heads, model.config.rope_base, cache_position)

		attn_output := ml.attention_with_cache(
			q, k, v,
			cache.layers[i].k, cache.layers[i].v,
			cache_position,
			model.config.n_q_heads,
			model.config.n_kv_heads,
		)
		attn_output = ml.linear(attn_output, layer.o_proj_weight)

		residual = ml.add(residual, attn_output)

		normed = ml.rmsnorm(residual, layer.post_attn_norm_weight)

		gate       := ml.linear(normed, layer.gate_proj_weight)
		up         := ml.linear(normed, layer.up_proj_weight)
		mlp_output := ml.linear(ml.mul(ml.silu(gate), up), layer.down_proj_weight)

		residual = ml.add(residual, mlp_output)
	}

	output = ml.rmsnorm(residual, model.output_norm_weight)
	output = ml.linear(output, model.lm_head_weight)
	if output.type != .F32 {
		output = ml.cast_to(output, .F32)
	}

	cache.length += token_count

	return
}

@(require_results)
forward :: proc(model: Llama, tokens: []int) -> (output: ml.Tensor) {
	output = ml.select(model.token_embeddings, tokens)

	residual := output

	for layer in model.layers {
		normed := ml.rmsnorm(residual, layer.input_norm_weight)

		q := ml.linear(normed, layer.q_proj_weight)
		k := ml.linear(normed, layer.k_proj_weight)
		v := ml.linear(normed, layer.v_proj_weight)

		if model.config.use_qk_norm {
			q = _per_head_rmsnorm(q, layer.q_norm_weight, model.config.n_q_heads)
			k = _per_head_rmsnorm(k, layer.k_norm_weight, model.config.n_kv_heads)
		}

		q = ml.rope(q, model.config.n_q_heads,  model.config.rope_base)
		k = ml.rope(k, model.config.n_kv_heads, model.config.rope_base)

		attn_output := ml.attention(q, k, v, model.config.n_q_heads, model.config.n_kv_heads)
		attn_output  = ml.linear(attn_output, layer.o_proj_weight)

		residual = ml.add(residual, attn_output)

		normed = ml.rmsnorm(residual, layer.post_attn_norm_weight)

		gate       := ml.linear(normed, layer.gate_proj_weight)
		up         := ml.linear(normed, layer.up_proj_weight)
		mlp_output := ml.linear(ml.mul(ml.silu(gate), up), layer.down_proj_weight)

		residual = ml.add(residual, mlp_output)
	}

	output = ml.rmsnorm(residual, model.output_norm_weight)
	output = ml.linear(output, model.lm_head_weight)
	if output.type != .F32 {
		output = ml.cast_to(output, .F32)
	}
	return
}

update :: proc(opt: ml.Optimizer, model: Llama) {
	ml.update(opt, model.token_embeddings)

	for layer in model.layers {
		ml.update(opt, layer.input_norm_weight)
		ml.update(opt, layer.q_proj_weight)
		ml.update(opt, layer.k_proj_weight)
		ml.update(opt, layer.v_proj_weight)
		ml.update(opt, layer.o_proj_weight)
		if model.config.use_qk_norm {
			ml.update(opt, layer.q_norm_weight)
			ml.update(opt, layer.k_norm_weight)
		}
		ml.update(opt, layer.post_attn_norm_weight)
		ml.update(opt, layer.gate_proj_weight)
		ml.update(opt, layer.up_proj_weight)
		ml.update(opt, layer.down_proj_weight)
	}

	ml.update(opt, model.output_norm_weight)
	if !model.config.tied_embeddings {
		ml.update(opt, model.lm_head_weight)
	}
}
