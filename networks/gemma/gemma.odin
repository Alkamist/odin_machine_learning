package gemma

import "base:builtin"

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

import ml   "../../"
import lora "../../networks/lora"

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
make_e4b_config :: proc(allocator := context.allocator) -> (cfg: Config) {
	cfg = {
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
		rope_base_sliding           = 10000,
		rope_base_full              = 1000000,
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

	return
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
	if cfg.num_kv_shared_layers == 0 {
		return false
	}
	return layer_idx >= cfg.num_hidden_layers - cfg.num_kv_shared_layers
}

LoRA_Target :: enum {
	Q,
	K,
	V,
	O,
	Gate,
	Up,
	Down,
}
LoRA_Targets :: bit_set[LoRA_Target]

LORA_DEFAULT_TARGETS :: LoRA_Targets{.Q, .K, .V, .O}

LoRA_Config :: struct {
	rank:    int,
	alpha:   f32,
	targets: LoRA_Targets,
}

Layer :: struct {
	input_norm_weight:            ml.Tensor, // [hidden_size]
	post_attention_norm_weight:   ml.Tensor, // [hidden_size]
	pre_feedforward_norm_weight:  ml.Tensor, // [hidden_size]
	post_feedforward_norm_weight: ml.Tensor, // [hidden_size]

	q_proj_weight: ml.Tensor, // [num_attention_heads * head_dim, hidden_size]
	q_norm_weight: ml.Tensor, // [head_dim] pre-baked with sqrt(head_dim) for scaling=1.0
	k_proj_weight: ml.Tensor, // [num_kv_heads * head_dim, hidden_size] (omitted for shared layers)
	k_norm_weight: ml.Tensor, // [head_dim] (omitted for shared layers)
	v_proj_weight: ml.Tensor, // [num_kv_heads * head_dim, hidden_size] (omitted for shared layers)
	o_proj_weight: ml.Tensor, // [hidden_size, num_attention_heads * head_dim]

	gate_proj_weight: ml.Tensor, // [intermediate_size, hidden_size]
	up_proj_weight:   ml.Tensor, // [intermediate_size, hidden_size]
	down_proj_weight: ml.Tensor, // [hidden_size, intermediate_size]

	per_layer_input_gate_weight:      ml.Tensor, // [hidden_size_per_layer_input, hidden_size]
	per_layer_projection_weight:      ml.Tensor, // [hidden_size, hidden_size_per_layer_input]
	post_per_layer_input_norm_weight: ml.Tensor, // [hidden_size]

	layer_scalar: ml.Tensor, // [1]

	// Optional LoRA adapters; rank == 0 means the adapter is unused.
	q_lora:    lora.Adapter,
	k_lora:    lora.Adapter,
	v_lora:    lora.Adapter,
	o_lora:    lora.Adapter,
	gate_lora: lora.Adapter,
	up_lora:   lora.Adapter,
	down_lora: lora.Adapter,
}

Gemma :: struct {
	config: Config,
	dtype:  ml.Data_Type,

	embed_tokens_weight: ml.Tensor, // [vocab_size, hidden_size]
	output_norm_weight:  ml.Tensor, // [hidden_size]
	lm_head_weight:      ml.Tensor, // tied to embed_tokens_weight

	layers: []Layer,

	embed_tokens_per_layer_bytes:      []byte,
	embed_tokens_per_layer_row_bytes:  int,       // bytes per vocab row
	per_layer_model_projection_weight: ml.Tensor, // [num_hidden_layers * ple_dim, hidden_size]
	per_layer_projection_norm_weight:  ml.Tensor, // [hidden_size_per_layer_input]

	v_norm_ones_sliding: ml.Tensor, // [head_dim_sliding]
	v_norm_ones_full:    ml.Tensor, // [head_dim_full]

	embed_scale:         ml.Tensor, // sqrt(hidden_size)
	ple_token_scale:     ml.Tensor, // sqrt(ple_dim)
	ple_ctx_scale:       ml.Tensor, // 1 / sqrt(hidden_size)
	ple_combine_scale:   ml.Tensor, // 1 / sqrt(2)
	softcap_inv:         ml.Tensor, // 1 / final_logit_softcapping (only if > 0)
	softcap:             ml.Tensor, // final_logit_softcapping     (only if > 0)

	params:      ml.Registry,
	train_flags: ml.Parameter_Flags,
}

@(require_results)
make :: proc(config: Config, dtype: ml.Data_Type = .F32, for_training: bool = false, lora_cfg: Maybe(LoRA_Config) = nil, allocator := context.allocator) -> (model: Gemma) {
	context.allocator = allocator

	model.config = config
	model.dtype  = dtype
	model.layers = builtin.make([]Layer, config.num_hidden_layers)

	// Const scalars stay .Data-only: ml.mul backward skips the b-side
	// when there is no gradient buffer, which avoids both the bf16 CAS
	// path AND any wasted writes to a frozen scalar.
	const_buffers := ml.Buffer_Set{.Data}

	// Trainable params get the {Data, Gradient} set;
	// anything else (inference, or "frozen base under LoRA") is .Data only.
	// Linear backward checks for a missing gradient buffer and skips the
	// dW GEMM, so frozen weights pay no backward cost.
	trainable_buffers := ml.DEFAULT_PARAMETER_BUFFERS
	frozen_buffers    := ml.Buffer_Set{.Data}

	use_lora := lora_cfg != nil
	base_buffers := ml.Buffer_Set{.Data}
	switch {
	case !for_training:
		base_buffers = frozen_buffers
	case use_lora:
		base_buffers = frozen_buffers // QLoRA: only adapters train
	case:
		base_buffers      = trainable_buffers
		model.train_flags = {.Train, .Checkpoint}
	}

	make_w :: proc(dtype: ml.Data_Type, shape: []int, buffers: ml.Buffer_Set) -> ml.Tensor {
		return ml.alloc(dtype, shape, persistent=true, buffers=buffers)
	}

	model.embed_tokens_weight = make_w(dtype, {config.vocab_size, config.hidden_size}, base_buffers)
	model.output_norm_weight  = make_w(dtype, {config.hidden_size}, base_buffers)
	if config.tie_word_embeddings {
		model.lm_head_weight = model.embed_tokens_weight
	} else {
		model.lm_head_weight = make_w(dtype, {config.vocab_size, config.hidden_size}, base_buffers)
	}

	ple_total := config.num_hidden_layers * config.hidden_size_per_layer_input
	dtype_bytes := dtype == .F32 ? 4 : 2
	model.embed_tokens_per_layer_row_bytes  = ple_total * dtype_bytes
	model.embed_tokens_per_layer_bytes      = builtin.make([]byte, config.vocab_size * model.embed_tokens_per_layer_row_bytes)
	model.per_layer_model_projection_weight = make_w(dtype, {ple_total, config.hidden_size}, base_buffers)
	model.per_layer_projection_norm_weight  = make_w(dtype, {config.hidden_size_per_layer_input}, base_buffers)

	// v_norm_ones is a literal constant (filled with 1.0), no gradient ever needed.
	model.v_norm_ones_sliding = make_w(dtype, {config.head_dim_sliding}, const_buffers)
	model.v_norm_ones_full    = make_w(dtype, {config.head_dim_full}, const_buffers)
	ml.fill_value(model.v_norm_ones_sliding, 1)
	ml.fill_value(model.v_norm_ones_full,    1)

	for &layer, layer_idx in model.layers {
		head_dim := config_head_dim(config, layer_idx)
		q_size   := config.num_attention_heads * head_dim
		kv_size  := config.num_key_value_heads * head_dim

		layer.input_norm_weight             = make_w(dtype, {config.hidden_size}, base_buffers)
		layer.post_attention_norm_weight    = make_w(dtype, {config.hidden_size}, base_buffers)
		layer.pre_feedforward_norm_weight   = make_w(dtype, {config.hidden_size}, base_buffers)
		layer.post_feedforward_norm_weight  = make_w(dtype, {config.hidden_size}, base_buffers)

		layer.q_proj_weight = make_w(dtype, {q_size, config.hidden_size}, base_buffers)
		layer.q_norm_weight = make_w(dtype, {head_dim}, base_buffers)
		layer.o_proj_weight = make_w(dtype, {config.hidden_size, q_size}, base_buffers)

		if !is_kv_shared_layer(config, layer_idx) {
			layer.k_proj_weight = make_w(dtype, {kv_size, config.hidden_size}, base_buffers)
			layer.k_norm_weight = make_w(dtype, {head_dim}, base_buffers)
			layer.v_proj_weight = make_w(dtype, {kv_size, config.hidden_size}, base_buffers)
		}

		layer.gate_proj_weight = make_w(dtype, {config.intermediate_size, config.hidden_size}, base_buffers)
		layer.up_proj_weight   = make_w(dtype, {config.intermediate_size, config.hidden_size}, base_buffers)
		layer.down_proj_weight = make_w(dtype, {config.hidden_size, config.intermediate_size}, base_buffers)

		layer.per_layer_input_gate_weight      = make_w(dtype, {config.hidden_size_per_layer_input, config.hidden_size}, base_buffers)
		layer.per_layer_projection_weight      = make_w(dtype, {config.hidden_size, config.hidden_size_per_layer_input}, base_buffers)
		layer.post_per_layer_input_norm_weight = make_w(dtype, {config.hidden_size}, base_buffers)
		layer.layer_scalar                     = make_w(dtype, {1}, base_buffers)

		if cfg, ok := lora_cfg.?; ok {
			if .Q in cfg.targets {
				layer.q_lora = lora.make(config.hidden_size, q_size, cfg.rank, cfg.alpha, dtype)
			}
			if !is_kv_shared_layer(config, layer_idx) {
				if .K in cfg.targets {
					layer.k_lora = lora.make(config.hidden_size, kv_size, cfg.rank, cfg.alpha, dtype)
				}
				if .V in cfg.targets {
					layer.v_lora = lora.make(config.hidden_size, kv_size, cfg.rank, cfg.alpha, dtype)
				}
			}
			if .O in cfg.targets {
				layer.o_lora = lora.make(q_size, config.hidden_size, cfg.rank, cfg.alpha, dtype)
			}
			if .Gate in cfg.targets {
				layer.gate_lora = lora.make(config.hidden_size, config.intermediate_size, cfg.rank, cfg.alpha, dtype)
			}
			if .Up in cfg.targets {
				layer.up_lora = lora.make(config.hidden_size, config.intermediate_size, cfg.rank, cfg.alpha, dtype)
			}
			if .Down in cfg.targets {
				layer.down_lora = lora.make(config.intermediate_size, config.hidden_size, cfg.rank, cfg.alpha, dtype)
			}
		}
	}

	model.embed_scale       = ml.scalar(dtype, math.sqrt(f32(config.hidden_size)), persistent=true)
	model.ple_token_scale   = ml.scalar(dtype, math.sqrt(f32(config.hidden_size_per_layer_input)), persistent=true)
	model.ple_ctx_scale     = ml.scalar(dtype, 1.0 / math.sqrt(f32(config.hidden_size)), persistent=true)
	model.ple_combine_scale = ml.scalar(dtype, 1.0 / math.sqrt(f32(2)), persistent=true)
	if config.final_logit_softcapping > 0 {
		model.softcap_inv = ml.scalar(dtype, 1.0 / config.final_logit_softcapping, persistent=true)
		model.softcap     = ml.scalar(dtype, config.final_logit_softcapping, persistent=true)
	}

	reserve(&model.params.parameters, 16 * config.num_hidden_layers + 12)
	_register_parameters(&model)

	return
}

_register_parameters :: proc(model: ^Gemma) {
	cfg            := model.config
	residual_scale := f32(0.02 / math.sqrt(f32(2 * cfg.num_hidden_layers)))

	reg :: proc(model: ^Gemma, prefix, name: string, tensor: ml.Tensor, init: ml.Init) {
		used_init: ml.Init = ml.Init_None{}
		if .Train in model.train_flags {
			used_init = init
		}
		ml.parameter_register(&model.params, prefix, name, tensor, init=used_init, flags=model.train_flags)
	}

	reg(model, "", "model.language_model.embed_tokens.weight",            model.embed_tokens_weight,               ml.Init_Normal{mean=0, std=0.02})
	reg(model, "", "model.language_model.norm.weight",                    model.output_norm_weight,                ml.Init_Value{value=1})
	if !cfg.tie_word_embeddings {
		reg(model, "", "lm_head.weight", model.lm_head_weight, ml.Init_Normal{mean=0, std=0.02})
	}
	reg(model, "", "model.language_model.per_layer_model_projection.weight", model.per_layer_model_projection_weight, ml.Init_Normal{mean=0, std=0.02})
	reg(model, "", "model.language_model.per_layer_projection_norm.weight",  model.per_layer_projection_norm_weight,  ml.Init_Value{value=1})

	for &layer, layer_idx in model.layers {
		prefix := fmt.tprintf("model.language_model.layers.%v", layer_idx)

		reg(model, prefix, "input_layernorm.weight",            layer.input_norm_weight,            ml.Init_Value{value=1})
		reg(model, prefix, "post_attention_layernorm.weight",   layer.post_attention_norm_weight,   ml.Init_Value{value=1})
		reg(model, prefix, "pre_feedforward_layernorm.weight",  layer.pre_feedforward_norm_weight,  ml.Init_Value{value=1})
		reg(model, prefix, "post_feedforward_layernorm.weight", layer.post_feedforward_norm_weight, ml.Init_Value{value=1})

		reg(model, prefix, "self_attn.q_proj.weight", layer.q_proj_weight, ml.Init_Normal{mean=0, std=0.02})
		reg(model, prefix, "self_attn.q_norm.weight", layer.q_norm_weight, ml.Init_Value{value=1})
		reg(model, prefix, "self_attn.o_proj.weight", layer.o_proj_weight, ml.Init_Normal{mean=0, std=residual_scale})

		if !is_kv_shared_layer(cfg, layer_idx) {
			reg(model, prefix, "self_attn.k_proj.weight", layer.k_proj_weight, ml.Init_Normal{mean=0, std=0.02})
			reg(model, prefix, "self_attn.k_norm.weight", layer.k_norm_weight, ml.Init_Value{value=1})
			reg(model, prefix, "self_attn.v_proj.weight", layer.v_proj_weight, ml.Init_Normal{mean=0, std=0.02})
		}

		reg(model, prefix, "mlp.gate_proj.weight", layer.gate_proj_weight, ml.Init_Normal{mean=0, std=0.02})
		reg(model, prefix, "mlp.up_proj.weight",   layer.up_proj_weight,   ml.Init_Normal{mean=0, std=0.02})
		reg(model, prefix, "mlp.down_proj.weight", layer.down_proj_weight, ml.Init_Normal{mean=0, std=residual_scale})

		reg(model, prefix, "per_layer_input_gate.weight",      layer.per_layer_input_gate_weight,      ml.Init_Normal{mean=0, std=0.02})
		reg(model, prefix, "per_layer_projection.weight",      layer.per_layer_projection_weight,      ml.Init_Normal{mean=0, std=0.02})
		reg(model, prefix, "post_per_layer_input_norm.weight", layer.post_per_layer_input_norm_weight, ml.Init_Value{value=1})
		reg(model, prefix, "layer_scalar",                     layer.layer_scalar,                     ml.Init_Value{value=1})
	}
}

destroy :: proc(model: Gemma) {
	model := model
	ml.registry_destroy(&model.params)

	ml.destroy(model.v_norm_ones_sliding)
	ml.destroy(model.v_norm_ones_full)
	ml.destroy(model.embed_scale)
	ml.destroy(model.ple_token_scale)
	ml.destroy(model.ple_ctx_scale)
	ml.destroy(model.ple_combine_scale)
	ml.destroy(model.softcap_inv)
	ml.destroy(model.softcap)

	delete(model.embed_tokens_per_layer_bytes)

	_destroy_lora :: proc(adapter: lora.Adapter) {
		if adapter.rank > 0 {
			lora.destroy(adapter)
		}
	}
	for layer in model.layers {
		_destroy_lora(layer.q_lora)
		_destroy_lora(layer.k_lora)
		_destroy_lora(layer.v_lora)
		_destroy_lora(layer.o_lora)
		_destroy_lora(layer.gate_lora)
		_destroy_lora(layer.up_lora)
		_destroy_lora(layer.down_lora)
	}
	delete(model.layers)
}

randomize :: proc(model: Gemma) {
	model := model
	ml.registry_randomize(&model.params)

	_fill_per_layer_bytes_normal(model, 0.02)

	for layer in model.layers {
		if layer.q_lora.rank    > 0 { lora.randomize(layer.q_lora) }
		if layer.k_lora.rank    > 0 { lora.randomize(layer.k_lora) }
		if layer.v_lora.rank    > 0 { lora.randomize(layer.v_lora) }
		if layer.o_lora.rank    > 0 { lora.randomize(layer.o_lora) }
		if layer.gate_lora.rank > 0 { lora.randomize(layer.gate_lora) }
		if layer.up_lora.rank   > 0 { lora.randomize(layer.up_lora) }
		if layer.down_lora.rank > 0 { lora.randomize(layer.down_lora) }
	}
}

_fill_per_layer_bytes_normal :: proc(model: Gemma, std: f32) {
	cfg     := model.config
	count   := cfg.vocab_size * cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
	bytes   := model.embed_tokens_per_layer_bytes
	switch model.dtype {
	case .F32:
		f := ([^]f32)(raw_data(bytes))[:count]
		for i in 0 ..< count {
			f[i] = rand.float32_normal(0, std)
		}
	case .Bf16:
		f := ([^]ml.Bf16)(raw_data(bytes))[:count]
		for i in 0 ..< count {
			f[i] = ml.bf16_from_f32(rand.float32_normal(0, std))
		}
	case .Q4_K, .Q6_K:
		fmt.panicf("unsupported dtype %v", model.dtype)
	}
}

update :: proc(opt: ^ml.Optimizer, model: Gemma) {
	model := model
	ml.registry_update(opt, &model.params)

	for layer in model.layers {
		if layer.q_lora.rank    > 0 { lora.update(opt, layer.q_lora) }
		if layer.k_lora.rank    > 0 { lora.update(opt, layer.k_lora) }
		if layer.v_lora.rank    > 0 { lora.update(opt, layer.v_lora) }
		if layer.o_lora.rank    > 0 { lora.update(opt, layer.o_lora) }
		if layer.gate_lora.rank > 0 { lora.update(opt, layer.gate_lora) }
		if layer.up_lora.rank   > 0 { lora.update(opt, layer.up_lora) }
		if layer.down_lora.rank > 0 { lora.update(opt, layer.down_lora) }
	}
}

parameters :: proc(model: Gemma, dst: ^ml.Registry) {
	model := model
	ml.registry_gather(dst, &model.params)

	for layer, layer_idx in model.layers {
		attn_prefix := fmt.tprintf("model.language_model.layers.%v.self_attn", layer_idx)
		mlp_prefix  := fmt.tprintf("model.language_model.layers.%v.mlp",       layer_idx)
		if layer.q_lora.rank    > 0 { lora.parameters(layer.q_lora,    fmt.tprintf("%v.q_proj",    attn_prefix), dst) }
		if layer.k_lora.rank    > 0 { lora.parameters(layer.k_lora,    fmt.tprintf("%v.k_proj",    attn_prefix), dst) }
		if layer.v_lora.rank    > 0 { lora.parameters(layer.v_lora,    fmt.tprintf("%v.v_proj",    attn_prefix), dst) }
		if layer.o_lora.rank    > 0 { lora.parameters(layer.o_lora,    fmt.tprintf("%v.o_proj",    attn_prefix), dst) }
		if layer.gate_lora.rank > 0 { lora.parameters(layer.gate_lora, fmt.tprintf("%v.gate_proj", mlp_prefix),  dst) }
		if layer.up_lora.rank   > 0 { lora.parameters(layer.up_lora,   fmt.tprintf("%v.up_proj",   mlp_prefix),  dst) }
		if layer.down_lora.rank > 0 { lora.parameters(layer.down_lora, fmt.tprintf("%v.down_proj", mlp_prefix),  dst) }
	}
}

@(require_results)
_linear :: proc(input, weight: ml.Tensor) -> ml.Tensor {
	#partial switch weight.type {
	case .Q4_K: return ml.linear_q4_k(input, weight)
	case .Q6_K: return ml.linear_q6_k(input, weight)
	}
	return ml.linear(input, weight)
}

_gate_up_geglu :: proc(input, w_gate, w_up: ml.Tensor) -> ml.Tensor {
	if w_gate.type == .Q4_K && w_up.type == .Q4_K {
		return ml.linear_q4_k_gate_up_geglu(input, w_gate, w_up)
	}
	gate := _linear(input, w_gate)
	up   := _linear(input, w_up)
	return ml.gelu_mul(gate, up)
}

@(require_results)
_per_layer_inputs :: proc(model: Gemma, tokens: []int, inputs_embeds: ml.Tensor) -> ml.Tensor {
	cfg         := model.config
	ple_dim     := cfg.hidden_size_per_layer_input
	token_count := builtin.len(tokens)
	ple_total   := cfg.num_hidden_layers * ple_dim

	row_bytes := model.embed_tokens_per_layer_row_bytes
	lookup_buf := builtin.make([]byte, token_count * row_bytes, context.temp_allocator)
	for tok, t in tokens {
		src := model.embed_tokens_per_layer_bytes[tok * row_bytes : (tok + 1) * row_bytes]
		copy(lookup_buf[t * row_bytes : (t + 1) * row_bytes], src)
	}
	// `zeros` allocates the gradient buffer too (unless the pass was cleared
	// with training=false), which the training backward pass needs.
	token_identity := ml.zeros(model.dtype, {token_count, ple_total})
	ml.set_data_bytes(token_identity, lookup_buf)
	token_identity = ml.mul(token_identity, model.ple_token_scale)

	ctx_proj := _linear(inputs_embeds, model.per_layer_model_projection_weight)
	ctx_proj  = ml.mul(ctx_proj, model.ple_ctx_scale)

	flat_shape := []int{token_count * cfg.num_hidden_layers, ple_dim}
	ctx_proj    = ml.reshape(ctx_proj, flat_shape)
	ctx_proj    = ml.rmsnorm(ctx_proj, model.per_layer_projection_norm_weight, cfg.rms_norm_eps)
	ctx_proj    = ml.reshape(ctx_proj, []int{token_count, ple_total})

	combined := ml.add(ctx_proj, token_identity)
	combined  = ml.mul(combined, model.ple_combine_scale)
	return combined
}

Cache :: ml.Kv_Cache

@(require_results)
cache_make :: proc(model: Gemma, t_max: int, allocator := context.allocator) -> (cache: Cache) {
	cfg := model.config
	cache.t_max  = t_max
	cache.length = 0
	cache.layers = builtin.make([]ml.Kv_Layer_Cache, cfg.num_hidden_layers, allocator)

	for i in 0 ..< cfg.num_hidden_layers {
		if is_kv_shared_layer(cfg, i) {
			continue
		}
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
	ml.kv_cache_destroy(cache)
}

cache_reset :: proc(cache: ^Cache) {
	ml.kv_cache_reset(cache)
}

@(require_results)
forward :: proc(model: Gemma, tokens: []int) -> (logits: ml.Tensor) {
	logits, _ = _forward(model, nil, tokens)
	return
}

@(require_results)
forward_with_hidden :: proc(model: Gemma, tokens: []int) -> (logits, final_hidden: ml.Tensor) {
	return _forward(model, nil, tokens)
}

@(require_results)
forward_cached :: proc(model: Gemma, cache: ^Cache, new_tokens: []int, loc := #caller_location) -> (logits: ml.Tensor) {
	logits, _ = _forward(model, cache, new_tokens, loc=loc)
	return
}

@(require_results)
_forward :: proc(model: Gemma, cache: ^Cache, tokens: []int, loc := #caller_location) -> (logits, final_hidden: ml.Tensor) {
	cfg         := model.config
	token_count := builtin.len(tokens)

	cache_position := 0
	if cache != nil {
		assert(token_count > 0, "requires at least one new token", loc=loc)
		assert(cache.length + token_count <= cache.t_max, "would overflow KV cache", loc=loc)
		assert(builtin.len(cache.layers) == cfg.num_hidden_layers, "cache layer count must match model", loc=loc)
		cache_position = cache.length
	}

	embeds := ml.select(model.embed_tokens_weight, tokens)
	embeds  = ml.mul(embeds, model.embed_scale)
	inputs_embeds := embeds

	per_layer_inputs := _per_layer_inputs(model, tokens, inputs_embeds)
	ple_dim := cfg.hidden_size_per_layer_input

	residual := embeds

	shared_keys   := builtin.make(map[int]ml.Tensor, allocator=context.temp_allocator)
	shared_values := builtin.make(map[int]ml.Tensor, allocator=context.temp_allocator)

	for layer, layer_idx in model.layers {
		head_dim      := config_head_dim(cfg, layer_idx)
		rope_base     := config_rope_base(cfg, layer_idx)
		rope_fraction := config_rope_fraction(cfg, layer_idx)
		is_sliding    := cfg.layer_types[layer_idx] == .Sliding
		window        := cfg.sliding_window if is_sliding else 0

		hidden := ml.rmsnorm(residual, layer.input_norm_weight, cfg.rms_norm_eps)

		q := _linear(hidden, layer.q_proj_weight)
		if layer.q_lora.rank > 0 {
			q = lora.apply(hidden, q, layer.q_lora)
		}
		q = ml.rmsnorm_rope(q, layer.q_norm_weight, cfg.num_attention_heads, cfg.rms_norm_eps, rope_base, cache_position, rope_fraction)

		k, v: ml.Tensor
		if is_kv_shared_layer(cfg, layer_idx) {
			source := kv_source_layer_idx(cfg, layer_idx)
			k = shared_keys  [source]
			v = shared_values[source]
		} else {
			k = _linear(hidden, layer.k_proj_weight)
			if layer.k_lora.rank > 0 {
				k = lora.apply(hidden, k, layer.k_lora)
			}
			if cache != nil {
				k_cache := cache.layers[layer_idx].k
				k = ml.rmsnorm_rope_write_cache(k, layer.k_norm_weight, cfg.num_key_value_heads, cfg.rms_norm_eps, rope_base, cache_position, rope_fraction, k_cache, k_cache.shape[0])
			} else {
				k = ml.rmsnorm_rope(k, layer.k_norm_weight, cfg.num_key_value_heads, cfg.rms_norm_eps, rope_base, cache_position, rope_fraction)
			}

			v_norm_ones := model.v_norm_ones_full if head_dim == cfg.head_dim_full else model.v_norm_ones_sliding
			v = _linear(hidden, layer.v_proj_weight)
			if layer.v_lora.rank > 0 {
				v = lora.apply(hidden, v, layer.v_lora)
			}
			v = ml.per_head_rmsnorm(v, v_norm_ones, cfg.num_key_value_heads, eps=cfg.rms_norm_eps)

			shared_keys  [layer_idx] = k
			shared_values[layer_idx] = v
		}

		attn: ml.Tensor
		if cache != nil {
			cache_layer_idx := layer_idx if !is_kv_shared_layer(cfg, layer_idx) else kv_source_layer_idx(cfg, layer_idx)
			k_cache := cache.layers[cache_layer_idx].k
			v_cache := cache.layers[cache_layer_idx].v
			attn = ml.attention_with_cache(q, k, v, k_cache, v_cache, cache_position, cfg.num_attention_heads, cfg.num_key_value_heads, window)
		} else {
			attn = ml.attention(q, k, v, cfg.num_attention_heads, cfg.num_key_value_heads, true, window)
		}

		attn_pre_o := attn
		attn  = _linear(attn_pre_o, layer.o_proj_weight)
		if layer.o_lora.rank > 0 {
			attn = lora.apply(attn_pre_o, attn, layer.o_lora)
		}
		attn  = ml.rmsnorm(attn, layer.post_attention_norm_weight, cfg.rms_norm_eps)

		mlp_in: ml.Tensor
		residual, mlp_in = ml.add_rmsnorm(residual, attn, layer.pre_feedforward_norm_weight, cfg.rms_norm_eps)

		mlp_act: ml.Tensor
		if cache != nil && layer.gate_lora.rank == 0 && layer.up_lora.rank == 0 {
			mlp_act = _gate_up_geglu(mlp_in, layer.gate_proj_weight, layer.up_proj_weight)
		} else {
			gate := _linear(mlp_in, layer.gate_proj_weight)
			if layer.gate_lora.rank > 0 {
				gate = lora.apply(mlp_in, gate, layer.gate_lora)
			}
			up := _linear(mlp_in, layer.up_proj_weight)
			if layer.up_lora.rank > 0 {
				up = lora.apply(mlp_in, up, layer.up_lora)
			}
			mlp_act = ml.gelu_mul(gate, up)
		}
		mlp := _linear(mlp_act, layer.down_proj_weight)
		if layer.down_lora.rank > 0 {
			mlp = lora.apply(mlp_act, mlp, layer.down_lora)
		}
		mlp  = ml.rmsnorm(mlp, layer.post_feedforward_norm_weight, cfg.rms_norm_eps)

		residual = ml.add(residual, mlp)

		ple_input := ml.slice_trailing(per_layer_inputs, layer_idx * ple_dim, (layer_idx + 1) * ple_dim)
		ple       := _linear(residual, layer.per_layer_input_gate_weight)
		ple        = ml.gelu_mul(ple, ple_input)
		ple        = _linear(ple, layer.per_layer_projection_weight)
		ple        = ml.rmsnorm(ple, layer.post_per_layer_input_norm_weight, cfg.rms_norm_eps)
		residual   = ml.add(residual, ple)

		residual = ml.mul(residual, layer.layer_scalar)
	}

	final_hidden = ml.rmsnorm(residual, model.output_norm_weight, cfg.rms_norm_eps)
	logits = _linear(final_hidden, model.lm_head_weight)
	if cfg.final_logit_softcapping > 0 {
		logits = ml.mul(logits, model.softcap_inv)
		logits = ml.tanh(logits)
		logits = ml.mul(logits, model.softcap)
	}

	if cache != nil {
		if logits.type != .F32 {
			logits = ml.cast_to(logits, .F32)
		}
		cache.length += token_count
	} else {
		if model.dtype != .F32 {
			final_hidden = ml.cast_to(final_hidden, .F32)
			logits       = ml.cast_to(logits,       .F32)
		}
	}

	return
}
