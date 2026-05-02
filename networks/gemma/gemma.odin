package machine_learning_network_gemma

// Gemma 4 text-only forward pass. Mirrors `transformers.models.gemma4.modeling_gemma4`
// closely enough to match HF logits within bf16 tolerance. Only inference for now.
// See `tools/gemma_dump.py` for the reference logits the loader+forward target.

import "base:builtin"
import "core:math"

import ml "../../"

Layer_Type :: enum u8 {
	Sliding,
	Full,
}

Config :: struct {
	num_hidden_layers:           int,
	hidden_size:                 int,
	intermediate_size:           int,
	num_attention_heads:         int,
	num_key_value_heads:         int,
	head_dim_sliding:            int,
	head_dim_full:               int,
	vocab_size:                  int,
	max_position_embeddings:     int,
	sliding_window:              int,
	hidden_size_per_layer_input: int,
	num_kv_shared_layers:        int,
	rope_base_sliding:           f32,
	rope_base_full:              f32,
	rope_fraction_full:          f32,
	rms_norm_eps:                f32,
	final_logit_softcapping:     f32, // 0 disables; Gemma 4 uses 30.0 in `Gemma4ForCausalLM.forward`.
	tie_word_embeddings:         bool,

	layer_types: []Layer_Type, // length == num_hidden_layers
}

@(require_results)
make_e4b_config :: proc(allocator := context.allocator) -> Config {
	cfg := Config{
		num_hidden_layers           = 42,
		hidden_size                 = 2560,
		intermediate_size           = 10240,
		num_attention_heads         = 8,
		num_key_value_heads         = 2,
		head_dim_sliding            = 256,
		head_dim_full               = 512,
		vocab_size                  = 262144,
		max_position_embeddings     = 131072,
		sliding_window              = 512,
		hidden_size_per_layer_input = 256,
		num_kv_shared_layers        = 18,
		rope_base_sliding           = 10_000,
		rope_base_full              = 1_000_000,
		rope_fraction_full          = 0.25,
		rms_norm_eps                = 1e-6,
		final_logit_softcapping     = 30,
		tie_word_embeddings         = true,
	}

	// 5 sliding : 1 full repeating; full layers at 5, 11, 17, 23, 29, 35, 41.
	cfg.layer_types = builtin.make([]Layer_Type, cfg.num_hidden_layers, allocator)
	for i in 0 ..< cfg.num_hidden_layers {
		cfg.layer_types[i] = .Full if (i + 1) % 6 == 0 else .Sliding
	}
	return cfg
}

config_destroy :: proc(cfg: Config) {
	delete(cfg.layer_types)
}

@(require_results)
config_head_dim :: proc(cfg: Config, layer_idx: int) -> int {
	return cfg.head_dim_full if cfg.layer_types[layer_idx] == .Full else cfg.head_dim_sliding
}

@(require_results)
config_rope_base :: proc(cfg: Config, layer_idx: int) -> f32 {
	return cfg.rope_base_full if cfg.layer_types[layer_idx] == .Full else cfg.rope_base_sliding
}

@(require_results)
config_rope_fraction :: proc(cfg: Config, layer_idx: int) -> f32 {
	return cfg.rope_fraction_full if cfg.layer_types[layer_idx] == .Full else f32(1.0)
}

// `kv_source_layer_idx` returns the layer index that supplies K/V for `layer_idx`.
// Returns `layer_idx` itself for non-shared layers (i.e. they compute their own K/V).
// For shared layers it returns the last non-shared layer of the same attention type.
@(require_results)
kv_source_layer_idx :: proc(cfg: Config, layer_idx: int) -> int {
	first_shared := cfg.num_hidden_layers - cfg.num_kv_shared_layers
	if layer_idx < first_shared || cfg.num_kv_shared_layers == 0 {
		return layer_idx
	}
	target_type := cfg.layer_types[layer_idx]
	for i := first_shared - 1; i >= 0; i -= 1 {
		if cfg.layer_types[i] == target_type {
			return i
		}
	}
	return layer_idx
}

@(require_results)
is_kv_shared_layer :: proc(cfg: Config, layer_idx: int) -> bool {
	if cfg.num_kv_shared_layers == 0 do return false
	return layer_idx >= cfg.num_hidden_layers - cfg.num_kv_shared_layers
}

Layer :: struct {
	// Pre/post norms around attention and the feedforward.
	input_norm_weight:                  ml.Tensor, // [hidden_size]
	post_attention_norm_weight:         ml.Tensor, // [hidden_size]
	pre_feedforward_norm_weight:        ml.Tensor, // [hidden_size]
	post_feedforward_norm_weight:       ml.Tensor, // [hidden_size]

	// Attention. q/k/v_proj output widths follow the layer's head_dim.
	q_proj_weight:                      ml.Tensor, // [num_attention_heads * head_dim, hidden_size]
	q_norm_weight:                      ml.Tensor, // [head_dim] — pre-baked with sqrt(head_dim) for `scaling=1.0`
	k_proj_weight:                      ml.Tensor, // [num_kv_heads * head_dim, hidden_size] (omitted for shared layers)
	k_norm_weight:                      ml.Tensor, // [head_dim] (omitted for shared layers)
	v_proj_weight:                      ml.Tensor, // [num_kv_heads * head_dim, hidden_size] (omitted for shared layers)
	o_proj_weight:                      ml.Tensor, // [hidden_size, num_attention_heads * head_dim]

	// Feedforward (GeGLU with tanh-approx GeLU).
	gate_proj_weight:                   ml.Tensor, // [intermediate_size, hidden_size]
	up_proj_weight:                     ml.Tensor, // [intermediate_size, hidden_size]
	down_proj_weight:                   ml.Tensor, // [hidden_size, intermediate_size]

	// Per-Layer Embedding (PLE) block.
	per_layer_input_gate_weight:        ml.Tensor, // [hidden_size_per_layer_input, hidden_size]
	per_layer_projection_weight:        ml.Tensor, // [hidden_size, hidden_size_per_layer_input]
	post_per_layer_input_norm_weight:   ml.Tensor, // [hidden_size]

	// Per-layer trained scale applied to the block's residual at the very end.
	// HF: `hidden_states *= self.layer_scalar`. Shape `[1]` in the checkpoint;
	// stored as a scalar in our forward.
	layer_scalar:                       ml.Tensor, // [1]

	// Optional per-output-channel f32 scales paired with each projection above.
	// Populated by `quantize_for_inference`; backend == nil means the matching
	// `*_proj_weight` is still in its original Bf16/F32 dtype and `_linear`
	// dispatches to ml.linear instead of ml.linear_q8.
	q_proj_scales:                      ml.Tensor,
	k_proj_scales:                      ml.Tensor,
	v_proj_scales:                      ml.Tensor,
	o_proj_scales:                      ml.Tensor,
	gate_proj_scales:                   ml.Tensor,
	up_proj_scales:                     ml.Tensor,
	down_proj_scales:                   ml.Tensor,
	per_layer_input_gate_scales:        ml.Tensor,
	per_layer_projection_scales:        ml.Tensor,
}

Gemma :: struct {
	config: Config,
	dtype:  ml.Data_Type,

	// Token embedding (scaled by sqrt(hidden_size) at lookup time) and tied LM head.
	embed_tokens_weight: ml.Tensor, // [vocab_size, hidden_size]
	output_norm_weight:  ml.Tensor, // [hidden_size]
	lm_head_weight:      ml.Tensor, // tied to embed_tokens_weight

	layers: []Layer,

	// Per-Layer Embedding global pieces.
	//
	// `embed_tokens_per_layer` is `[vocab_size, num_hidden_layers * ple_dim]`
	// = ~5.6 GB at bf16 for E4B, which exceeds Vulkan's per-buffer
	// `maxStorageBufferRange` (2 GiB on most drivers). Since it's only used
	// for a per-token lookup across the prompt, we keep it as raw bytes on
	// the host and upload only the looked-up `[T, num_layers * ple_dim]`
	// slice to the GPU each forward.
	embed_tokens_per_layer_bytes:      []byte,
	embed_tokens_per_layer_row_bytes:  int, // bytes per vocab row
	per_layer_model_projection_weight: ml.Tensor, // [num_hidden_layers * ple_dim, hidden_size]
	per_layer_projection_norm_weight:  ml.Tensor, // [hidden_size_per_layer_input]

	// Optional Int8 quantization scales for the model-level linear projections
	// (see comment on Layer).
	per_layer_model_projection_scales: ml.Tensor,
	lm_head_scales:                    ml.Tensor,

	// V-norm has `with_scale=False` in HF — we feed `ml.rmsnorm` an all-ones constant
	// so we don't need a no-scale variant of the op. Created per head_dim used.
	v_norm_ones_sliding: ml.Tensor, // [head_dim_sliding]
	v_norm_ones_full:    ml.Tensor, // [head_dim_full]
}

@(require_results)
make :: proc(config: Config, dtype: ml.Data_Type = .F32, for_training: bool = false, allocator := context.allocator) -> (model: Gemma) {
	context.allocator = allocator

	model.config = config
	model.dtype  = dtype
	model.layers = builtin.make([]Layer, config.num_hidden_layers)

	// Inference allocates only `.Data`; training adds `.Gradient`, `.Adam_M`,
	// `.Adam_V` so the optimizer can store its moments. The 4× memory
	// difference matters: 8B × 2 bytes × 4 buffers = 64 GB and trips the
	// driver immediately, while inference only needs ~9 GB at bf16.
	buffers := ml.Buffer_Set{.Data}
	if for_training do buffers = ml.DEFAULT_PARAMETER_BUFFERS

	make_w :: proc(dtype: ml.Data_Type, shape: []int, buffers: ml.Buffer_Set) -> ml.Tensor {
		return ml.alloc(dtype, shape, persistent=true, buffers=buffers)
	}

	model.embed_tokens_weight = make_w(dtype, {config.vocab_size, config.hidden_size}, buffers)
	model.output_norm_weight  = make_w(dtype, {config.hidden_size}, buffers)
	if config.tie_word_embeddings {
		model.lm_head_weight = model.embed_tokens_weight
	} else {
		model.lm_head_weight = make_w(dtype, {config.vocab_size, config.hidden_size}, buffers)
	}

	ple_total := config.num_hidden_layers * config.hidden_size_per_layer_input
	dtype_bytes := dtype == .F32 ? 4 : 2
	model.embed_tokens_per_layer_row_bytes  = ple_total * dtype_bytes
	model.embed_tokens_per_layer_bytes      = builtin.make([]byte, config.vocab_size * model.embed_tokens_per_layer_row_bytes)
	model.per_layer_model_projection_weight = make_w(dtype, {ple_total, config.hidden_size}, buffers)
	model.per_layer_projection_norm_weight  = make_w(dtype, {config.hidden_size_per_layer_input}, buffers)

	model.v_norm_ones_sliding = make_w(dtype, {config.head_dim_sliding}, buffers)
	model.v_norm_ones_full    = make_w(dtype, {config.head_dim_full}, buffers)
	ml.fill_value(model.v_norm_ones_sliding, 1)
	ml.fill_value(model.v_norm_ones_full,    1)

	for &layer, layer_idx in model.layers {
		head_dim := config_head_dim(config, layer_idx)
		q_size   := config.num_attention_heads * head_dim
		kv_size  := config.num_key_value_heads * head_dim

		layer.input_norm_weight             = make_w(dtype, {config.hidden_size}, buffers)
		layer.post_attention_norm_weight    = make_w(dtype, {config.hidden_size}, buffers)
		layer.pre_feedforward_norm_weight   = make_w(dtype, {config.hidden_size}, buffers)
		layer.post_feedforward_norm_weight  = make_w(dtype, {config.hidden_size}, buffers)

		layer.q_proj_weight = make_w(dtype, {q_size, config.hidden_size}, buffers)
		layer.q_norm_weight = make_w(dtype, {head_dim}, buffers)
		layer.o_proj_weight = make_w(dtype, {config.hidden_size, q_size}, buffers)

		if !is_kv_shared_layer(config, layer_idx) {
			layer.k_proj_weight = make_w(dtype, {kv_size, config.hidden_size}, buffers)
			layer.k_norm_weight = make_w(dtype, {head_dim}, buffers)
			layer.v_proj_weight = make_w(dtype, {kv_size, config.hidden_size}, buffers)
		}

		layer.gate_proj_weight = make_w(dtype, {config.intermediate_size, config.hidden_size}, buffers)
		layer.up_proj_weight   = make_w(dtype, {config.intermediate_size, config.hidden_size}, buffers)
		layer.down_proj_weight = make_w(dtype, {config.hidden_size, config.intermediate_size}, buffers)

		layer.per_layer_input_gate_weight      = make_w(dtype, {config.hidden_size_per_layer_input, config.hidden_size}, buffers)
		layer.per_layer_projection_weight      = make_w(dtype, {config.hidden_size, config.hidden_size_per_layer_input}, buffers)
		layer.post_per_layer_input_norm_weight = make_w(dtype, {config.hidden_size}, buffers)
		layer.layer_scalar                     = make_w(dtype, {1}, buffers)
	}

	return
}

destroy :: proc(model: Gemma) {
	_destroy_if_set :: proc(t: ml.Tensor) {
		if t.backend != nil do ml.destroy(t)
	}

	ml.destroy(model.embed_tokens_weight)
	if !model.config.tie_word_embeddings do ml.destroy(model.lm_head_weight)
	// When tied, lm_head_weight points at embed_tokens_weight unless
	// quantize_for_inference replaced it with a fresh Int8 copy. The scales
	// tensor is the unambiguous tell: present iff lm_head was quantized into
	// its own buffer.
	if model.config.tie_word_embeddings && model.lm_head_scales.backend != nil {
		ml.destroy(model.lm_head_weight)
	}
	_destroy_if_set(model.lm_head_scales)
	ml.destroy(model.output_norm_weight)

	delete(model.embed_tokens_per_layer_bytes)
	ml.destroy(model.per_layer_model_projection_weight)
	_destroy_if_set(model.per_layer_model_projection_scales)
	ml.destroy(model.per_layer_projection_norm_weight)

	ml.destroy(model.v_norm_ones_sliding)
	ml.destroy(model.v_norm_ones_full)

	for layer, layer_idx in model.layers {
		ml.destroy(layer.input_norm_weight)
		ml.destroy(layer.post_attention_norm_weight)
		ml.destroy(layer.pre_feedforward_norm_weight)
		ml.destroy(layer.post_feedforward_norm_weight)

		ml.destroy(layer.q_proj_weight)
		ml.destroy(layer.q_norm_weight)
		ml.destroy(layer.o_proj_weight)
		_destroy_if_set(layer.q_proj_scales)
		_destroy_if_set(layer.o_proj_scales)

		if !is_kv_shared_layer(model.config, layer_idx) {
			ml.destroy(layer.k_proj_weight)
			ml.destroy(layer.k_norm_weight)
			ml.destroy(layer.v_proj_weight)
			_destroy_if_set(layer.k_proj_scales)
			_destroy_if_set(layer.v_proj_scales)
		}

		ml.destroy(layer.gate_proj_weight)
		ml.destroy(layer.up_proj_weight)
		ml.destroy(layer.down_proj_weight)
		_destroy_if_set(layer.gate_proj_scales)
		_destroy_if_set(layer.up_proj_scales)
		_destroy_if_set(layer.down_proj_scales)

		ml.destroy(layer.per_layer_input_gate_weight)
		ml.destroy(layer.per_layer_projection_weight)
		ml.destroy(layer.post_per_layer_input_norm_weight)
		ml.destroy(layer.layer_scalar)
		_destroy_if_set(layer.per_layer_input_gate_scales)
		_destroy_if_set(layer.per_layer_projection_scales)
	}
	delete(model.layers)
}

Quant_Mode :: enum {
	None,
	Int8,
	Int4,
	Q8_0,
}

// Dispatch a linear projection. If `scales` is set, the weight is quantized
// and we route to the matching ml.linear_q* op based on weight dtype + scales
// rank (Q8_0 uses Int8 weight + 2-D scales, Int8 uses Int8 weight + 1-D scales).
@(require_results)
_linear :: proc(input, weight, scales: ml.Tensor) -> ml.Tensor {
	if scales.backend == nil {
		return ml.linear(input, weight)
	}
	if weight.type == .Int4 {
		return ml.linear_q4(input, weight, scales)
	}
	if scales.rank == 2 {
		return ml.linear_q8_0(input, weight, scales)
	}
	return ml.linear_q8(input, weight, scales)
}

// Replace one Bf16 weight tensor with its quantized form + scales, freeing
// the original Bf16 buffer.
_quantize_in_place :: proc(mode: Quant_Mode, weight, scales: ^ml.Tensor) {
	q_w, q_s: ml.Tensor
	switch mode {
	case .None: return
	case .Int8: q_w, q_s = ml.quantize_int8(weight^)
	case .Int4: q_w, q_s = ml.quantize_int4(weight^)
	case .Q8_0: q_w, q_s = ml.quantize_q8_0(weight^)
	}
	ml.destroy(weight^)
	weight^ = q_w
	scales^ = q_s
}

// Convert all linear-projection weights in `model` from Bf16 to the requested
// quantized format (Int8 or Int4). Embeddings, RMSNorm weights, and the
// per-layer-input lookup table are left in their original dtype since they're
// either accessed via `select` (no matmul) or are tiny.
//
// If the model has tied embeddings, the LM head is first detached into a
// fresh Bf16 copy so it can be quantized independently of `embed_tokens_weight`
// (which `select` still uses at original precision).
quantize_for_inference :: proc(model: ^Gemma, mode: Quant_Mode) {
	if mode == .None do return
	assert(model.dtype == .Bf16, "quantize_for_inference currently expects a Bf16 model")

	if model.config.tie_word_embeddings {
		shape := model.embed_tokens_weight.shape
		copy_w := ml.alloc(.Bf16, shape[:model.embed_tokens_weight.rank], persistent=true, buffers=ml.Buffer_Set{.Data})
		raw_bytes := builtin.make([]byte, ml.len(model.embed_tokens_weight) * 2, context.temp_allocator)
		ml.get_data_bytes(model.embed_tokens_weight, raw_bytes)
		ml.set_data_bytes(copy_w, raw_bytes)
		model.lm_head_weight = copy_w
	}
	_quantize_in_place(mode, &model.lm_head_weight, &model.lm_head_scales)

	_quantize_in_place(mode, &model.per_layer_model_projection_weight, &model.per_layer_model_projection_scales)

	for &layer, layer_idx in model.layers {
		_quantize_in_place(mode, &layer.q_proj_weight,                &layer.q_proj_scales)
		_quantize_in_place(mode, &layer.o_proj_weight,                &layer.o_proj_scales)
		if !is_kv_shared_layer(model.config, layer_idx) {
			_quantize_in_place(mode, &layer.k_proj_weight,            &layer.k_proj_scales)
			_quantize_in_place(mode, &layer.v_proj_weight,            &layer.v_proj_scales)
		}
		_quantize_in_place(mode, &layer.gate_proj_weight,             &layer.gate_proj_scales)
		_quantize_in_place(mode, &layer.up_proj_weight,               &layer.up_proj_scales)
		_quantize_in_place(mode, &layer.down_proj_weight,             &layer.down_proj_scales)
		_quantize_in_place(mode, &layer.per_layer_input_gate_weight,  &layer.per_layer_input_gate_scales)
		_quantize_in_place(mode, &layer.per_layer_projection_weight,  &layer.per_layer_projection_scales)
	}
}

// Compute the per-layer-input table once per forward, mirroring
// `Gemma4TextModel.get_per_layer_inputs` + `project_per_layer_inputs`.
// Output is `[token_count, num_hidden_layers * ple_dim]` packed; layer `l`'s
// slice is `[:, l * ple_dim : (l + 1) * ple_dim]`.
@(require_results)
_per_layer_inputs :: proc(model: Gemma, tokens: []int, inputs_embeds: ml.Tensor) -> ml.Tensor {
	cfg         := model.config
	ple_dim     := cfg.hidden_size_per_layer_input
	token_count := builtin.len(tokens)
	ple_total   := cfg.num_hidden_layers * ple_dim

	// Token-identity component: lookup rows of the host-side per-layer
	// embedding table, upload as a small `[T, ple_total]` GPU tensor.
	row_bytes := model.embed_tokens_per_layer_row_bytes
	lookup_buf := builtin.make([]byte, token_count * row_bytes, context.temp_allocator)
	for tok, t in tokens {
		src := model.embed_tokens_per_layer_bytes[tok * row_bytes : (tok + 1) * row_bytes]
		copy(lookup_buf[t * row_bytes : (t + 1) * row_bytes], src)
	}
	token_identity := ml.alloc(model.dtype, {token_count, ple_total}, persistent=false, buffers={.Data})
	ml.set_data_bytes(token_identity, lookup_buf)
	token_identity = ml.mul(token_identity, ml.scalar(math.sqrt(f32(ple_dim)), model.dtype))

	// Context-aware component: project inputs_embeds, scale by 1/sqrt(hidden), reshape, RMSNorm.
	ctx_proj := _linear(inputs_embeds, model.per_layer_model_projection_weight, model.per_layer_model_projection_scales)
	ctx_proj  = ml.mul(ctx_proj, ml.scalar(1.0 / math.sqrt(f32(cfg.hidden_size)), model.dtype))

	// rmsnorm operates on the trailing dim. View as [T*num_layers, ple_dim] so the
	// norm runs across each layer-slice independently.
	flat_shape := []int{token_count * cfg.num_hidden_layers, ple_dim}
	ctx_proj    = ml.reshape(ctx_proj, flat_shape)
	ctx_proj    = ml.rmsnorm(ctx_proj, model.per_layer_projection_norm_weight, false, cfg.rms_norm_eps)
	ctx_proj    = ml.reshape(ctx_proj, []int{token_count, ple_total})

	combined := ml.add(ctx_proj, token_identity)
	combined  = ml.mul(combined, ml.scalar(1.0 / math.sqrt(f32(2)), model.dtype))
	return combined
}

// Apply Q-norm (or K-norm) across each head independently. Input is
// `[T, n_heads * head_dim]`; we reshape to `[T*n_heads, head_dim]` so rmsnorm
// runs per head, then reshape back. Q-norm weights are pre-baked at load time
// with `sqrt(head_dim)` so the resulting Q absorbs the `1/sqrt(head_dim)`
// scaling that `ml.attention` applies internally.
@(require_results)
_qkv_norm :: proc(model: Gemma, x: ml.Tensor, weight: ml.Tensor, n_heads, head_dim: int, eps: f32) -> ml.Tensor {
	token_count := x.shape[0]
	flat_shape  := []int{token_count * n_heads, head_dim}
	view        := ml.reshape(x, flat_shape)
	normed      := ml.rmsnorm(view, weight, false, eps)
	out_shape   := []int{token_count, n_heads * head_dim}
	return ml.reshape(normed, out_shape)
}

// Per-layer K/V cache for incremental decoding. Shared layers (the last
// `num_kv_shared_layers`) carry empty handles — `forward_cached` reuses the
// source layer's K/V tensors directly within a single forward.
//
// Sliding-attention layers store only `sliding_window` rows and are written
// as a ring buffer (the attention op modulo-indexes by `t_capacity`). At
// 128k context this cuts each sliding layer's cache from ~256MB to ~1MB.
Layer_Cache :: struct {
	k: ml.Tensor, // [t_capacity, num_kv_heads * head_dim] (t_capacity = sliding_window for sliding layers, t_max otherwise)
	v: ml.Tensor,
}

Cache :: struct {
	t_max:  int,
	length: int,
	layers: []Layer_Cache,
}

@(require_results)
cache_make :: proc(model: Gemma, t_max: int, allocator := context.allocator) -> (cache: Cache) {
	cfg := model.config
	cache.t_max  = t_max
	cache.length = 0
	cache.layers = builtin.make([]Layer_Cache, cfg.num_hidden_layers, allocator)

	for i in 0 ..< cfg.num_hidden_layers {
		if is_kv_shared_layer(cfg, i) do continue
		head_dim   := config_head_dim(cfg, i)
		kv_size    := cfg.num_key_value_heads * head_dim
		is_sliding := cfg.layer_types[i] == .Sliding
		t_cap      := cfg.sliding_window if is_sliding else t_max
		cache.layers[i].k = ml.alloc(model.dtype, {t_cap, kv_size}, persistent=true, buffers={.Data})
		cache.layers[i].v = ml.alloc(model.dtype, {t_cap, kv_size}, persistent=true, buffers={.Data})
	}
	return
}

cache_destroy :: proc(cache: Cache) {
	for layer_cache in cache.layers {
		if layer_cache.k.rank > 0 do ml.destroy(layer_cache.k)
		if layer_cache.v.rank > 0 do ml.destroy(layer_cache.v)
	}
	delete(cache.layers)
}

cache_reset :: proc(cache: ^Cache) {
	cache.length = 0
}

@(require_results)
forward :: proc(model: Gemma, tokens: []int) -> (logits: ml.Tensor) {
	logits, _ = forward_with_hidden(model, tokens)
	return
}

// Incremental forward: extends `cache` with `new_tokens` (a prompt prefill on
// the first call, then 1-token decodes). Position offset is taken from
// `cache.length`. Caller must ensure `cache.length + len(new_tokens) <= t_max`.
@(require_results)
forward_cached :: proc(model: Gemma, cache: ^Cache, new_tokens: []int) -> (logits: ml.Tensor) {
	cfg := model.config
	token_count := builtin.len(new_tokens)
	assert(token_count > 0, "forward_cached requires at least one new token")
	assert(cache.length + token_count <= cache.t_max, "forward_cached would overflow KV cache")
	assert(builtin.len(cache.layers) == cfg.num_hidden_layers, "cache layer count must match model")

	cache_position := cache.length

	embeds := ml.select(model.embed_tokens_weight, new_tokens)
	embeds  = ml.mul(embeds, ml.scalar(math.sqrt(f32(cfg.hidden_size)), model.dtype))
	inputs_embeds := embeds

	per_layer_inputs := _per_layer_inputs(model, new_tokens, inputs_embeds)
	ple_dim := cfg.hidden_size_per_layer_input

	// Per-layer K/V tensors produced in this forward. Shared layers reuse
	// their source layer's K/V directly instead of slicing it back out of
	// the (possibly ring-wrapped) cache buffer.
	Step_KV :: struct { k, v: ml.Tensor }
	step_kvs := builtin.make([]Step_KV, cfg.num_hidden_layers, context.temp_allocator)

	residual := embeds

	for layer, layer_idx in model.layers {
		head_dim      := config_head_dim(cfg, layer_idx)
		rope_base     := config_rope_base(cfg, layer_idx)
		rope_fraction := config_rope_fraction(cfg, layer_idx)
		is_sliding    := cfg.layer_types[layer_idx] == .Sliding
		window        := cfg.sliding_window if is_sliding else 0

		hidden := ml.rmsnorm(residual, layer.input_norm_weight, false, cfg.rms_norm_eps)

		q := _linear(hidden, layer.q_proj_weight, layer.q_proj_scales)
		q  = _qkv_norm(model, q, layer.q_norm_weight, cfg.num_attention_heads, head_dim, cfg.rms_norm_eps)
		q  = ml.rope(q, cfg.num_attention_heads, rope_base, cache_position, rope_fraction)

		// Shared layers reuse the source layer's K/V tensors (computed
		// earlier in this same forward) and write into the source layer's
		// cache buffer — `attention_with_cache` re-writes the same rows
		// idempotently.
		cache_layer_idx := layer_idx if !is_kv_shared_layer(cfg, layer_idx) else kv_source_layer_idx(cfg, layer_idx)
		k_cache := cache.layers[cache_layer_idx].k
		v_cache := cache.layers[cache_layer_idx].v

		k, v: ml.Tensor
		if is_kv_shared_layer(cfg, layer_idx) {
			source := kv_source_layer_idx(cfg, layer_idx)
			k = step_kvs[source].k
			v = step_kvs[source].v
		} else {
			k = _linear(hidden, layer.k_proj_weight, layer.k_proj_scales)
			k = _qkv_norm(model, k, layer.k_norm_weight, cfg.num_key_value_heads, head_dim, cfg.rms_norm_eps)
			k = ml.rope(k, cfg.num_key_value_heads, rope_base, cache_position, rope_fraction)

			v_norm_ones := model.v_norm_ones_full if head_dim == cfg.head_dim_full else model.v_norm_ones_sliding
			v = _linear(hidden, layer.v_proj_weight, layer.v_proj_scales)
			v = _qkv_norm(model, v, v_norm_ones, cfg.num_key_value_heads, head_dim, cfg.rms_norm_eps)

			step_kvs[layer_idx] = Step_KV{k = k, v = v}
		}

		attn := ml.attention_with_cache(q, k, v, k_cache, v_cache, cache_position, cfg.num_attention_heads, cfg.num_key_value_heads, window)
		attn  = _linear(attn, layer.o_proj_weight, layer.o_proj_scales)
		attn  = ml.rmsnorm(attn, layer.post_attention_norm_weight, false, cfg.rms_norm_eps)
		residual = ml.add(residual, attn)

		mlp_in := ml.rmsnorm(residual, layer.pre_feedforward_norm_weight, false, cfg.rms_norm_eps)
		gate   := _linear(mlp_in, layer.gate_proj_weight, layer.gate_proj_scales)
		up     := _linear(mlp_in, layer.up_proj_weight,   layer.up_proj_scales)
		mlp    := _linear(ml.mul(ml.gelu(gate), up), layer.down_proj_weight, layer.down_proj_scales)
		mlp     = ml.rmsnorm(mlp, layer.post_feedforward_norm_weight, false, cfg.rms_norm_eps)
		residual = ml.add(residual, mlp)

		ple_input := ml.slice_trailing(per_layer_inputs, layer_idx * ple_dim, (layer_idx + 1) * ple_dim)
		ple       := _linear(residual, layer.per_layer_input_gate_weight, layer.per_layer_input_gate_scales)
		ple        = ml.mul(ml.gelu(ple), ple_input)
		ple        = _linear(ple, layer.per_layer_projection_weight, layer.per_layer_projection_scales)
		ple        = ml.rmsnorm(ple, layer.post_per_layer_input_norm_weight, false, cfg.rms_norm_eps)
		residual   = ml.add(residual, ple)

		residual = ml.mul(residual, layer.layer_scalar)
	}

	final_hidden := ml.rmsnorm(residual, model.output_norm_weight, false, cfg.rms_norm_eps)
	logits = _linear(final_hidden, model.lm_head_weight, model.lm_head_scales)
	if cfg.final_logit_softcapping > 0 {
		cap := cfg.final_logit_softcapping
		logits = ml.mul(logits, ml.scalar(1.0 / cap, model.dtype))
		logits = ml.tanh(logits)
		logits = ml.mul(logits, ml.scalar(cap, model.dtype))
	}
	if model.dtype != .F32 do logits = ml.cast_to(logits, .F32)

	cache.length += token_count
	return
}

// Variant that also returns the post-final-norm hidden state (the value that
// feeds into `lm_head`). Useful for parity tests against HF reference.
@(require_results)
forward_with_hidden :: proc(model: Gemma, tokens: []int) -> (logits, final_hidden: ml.Tensor) {
	cfg := model.config

	// Scaled token embedding.
	embeds := ml.select(model.embed_tokens_weight, tokens)
	embeds  = ml.mul(embeds, ml.scalar(math.sqrt(f32(cfg.hidden_size)), model.dtype))
	inputs_embeds := embeds

	// Per-Layer Embedding inputs, shared across the layer loop.
	per_layer_inputs := _per_layer_inputs(model, tokens, inputs_embeds)
	ple_dim := cfg.hidden_size_per_layer_input

	residual := embeds

	// Cache K/V from the last non-shared layer of each attention type so shared
	// layers can reuse them. Indexed by source layer index.
	shared_keys:   map[int]ml.Tensor
	shared_values: map[int]ml.Tensor
	shared_keys.allocator   = context.temp_allocator
	shared_values.allocator = context.temp_allocator
	defer delete(shared_keys)
	defer delete(shared_values)

	for layer, layer_idx in model.layers {
		head_dim      := config_head_dim(cfg, layer_idx)
		rope_base     := config_rope_base(cfg, layer_idx)
		rope_fraction := config_rope_fraction(cfg, layer_idx)
		is_sliding    := cfg.layer_types[layer_idx] == .Sliding
		window        := cfg.sliding_window if is_sliding else 0

		// Pre-attn norm.
		hidden := ml.rmsnorm(residual, layer.input_norm_weight, false, cfg.rms_norm_eps)

		q := _linear(hidden, layer.q_proj_weight, layer.q_proj_scales)
		q  = _qkv_norm(model, q, layer.q_norm_weight, cfg.num_attention_heads, head_dim, cfg.rms_norm_eps)
		q  = ml.rope(q, cfg.num_attention_heads, rope_base, 0, rope_fraction)

		k, v: ml.Tensor
		if is_kv_shared_layer(cfg, layer_idx) {
			source := kv_source_layer_idx(cfg, layer_idx)
			k = shared_keys  [source]
			v = shared_values[source]
		} else {
			k = _linear(hidden, layer.k_proj_weight, layer.k_proj_scales)
			k = _qkv_norm(model, k, layer.k_norm_weight, cfg.num_key_value_heads, head_dim, cfg.rms_norm_eps)
			k = ml.rope(k, cfg.num_key_value_heads, rope_base, 0, rope_fraction)

			v_norm_ones := model.v_norm_ones_full if head_dim == cfg.head_dim_full else model.v_norm_ones_sliding
			v = _linear(hidden, layer.v_proj_weight, layer.v_proj_scales)
			v = _qkv_norm(model, v, v_norm_ones, cfg.num_key_value_heads, head_dim, cfg.rms_norm_eps)

			// Stash for any later shared layer of the same type.
			shared_keys  [layer_idx] = k
			shared_values[layer_idx] = v
		}

		attn := ml.attention(q, k, v, cfg.num_attention_heads, cfg.num_key_value_heads, true, window)
		attn  = _linear(attn, layer.o_proj_weight, layer.o_proj_scales)
		attn  = ml.rmsnorm(attn, layer.post_attention_norm_weight, false, cfg.rms_norm_eps)
		residual = ml.add(residual, attn)

		// Feedforward.
		mlp_in := ml.rmsnorm(residual, layer.pre_feedforward_norm_weight, false, cfg.rms_norm_eps)
		gate   := _linear(mlp_in, layer.gate_proj_weight, layer.gate_proj_scales)
		up     := _linear(mlp_in, layer.up_proj_weight,   layer.up_proj_scales)
		mlp    := _linear(ml.mul(ml.gelu(gate), up), layer.down_proj_weight, layer.down_proj_scales)
		mlp     = ml.rmsnorm(mlp, layer.post_feedforward_norm_weight, false, cfg.rms_norm_eps)
		residual = ml.add(residual, mlp)

		// Per-Layer Embedding residual block.
		ple_input := ml.slice_trailing(per_layer_inputs, layer_idx * ple_dim, (layer_idx + 1) * ple_dim)
		ple       := _linear(residual, layer.per_layer_input_gate_weight, layer.per_layer_input_gate_scales)
		ple        = ml.mul(ml.gelu(ple), ple_input)
		ple        = _linear(ple, layer.per_layer_projection_weight, layer.per_layer_projection_scales)
		ple        = ml.rmsnorm(ple, layer.post_per_layer_input_norm_weight, false, cfg.rms_norm_eps)
		residual   = ml.add(residual, ple)

		// Per-layer trained scale.
		residual = ml.mul(residual, layer.layer_scalar)
	}

	final_hidden = ml.rmsnorm(residual, model.output_norm_weight, false, cfg.rms_norm_eps)
	logits        = _linear(final_hidden, model.lm_head_weight, model.lm_head_scales)
	if cfg.final_logit_softcapping > 0 {
		cap := cfg.final_logit_softcapping
		logits = ml.mul(logits, ml.scalar(1.0 / cap, model.dtype))
		logits = ml.tanh(logits)
		logits = ml.mul(logits, ml.scalar(cap, model.dtype))
	}
	if model.dtype != .F32 {
		final_hidden = ml.cast_to(final_hidden, .F32)
		logits       = ml.cast_to(logits,       .F32)
	}
	return
}
