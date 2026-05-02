package machine_learning_network_gemma

import "base:builtin"
import "core:fmt"
import "core:math"
import "core:slice"

import ml   "../../"
import "../../loaders/gguf"

// Load a GGUF Gemma 4 model (e.g. Ollama's Q4_K_M-quantized E4B blob) into
// a `Gemma` already allocated by `make(config, dtype=.Bf16)`. The loader
// swaps Bf16 placeholder projection tensors out for `.Q4_K` / `.Q6_K`
// tensors and copies the GGUF block bytes verbatim. Norms come in as F32
// in GGUF but we convert them down to the model dtype (Bf16) at load.
//
// Differences from `load_safetensors`:
//
//   - GGUF stores tensor shapes in ggml's reversed (column-major-fastest)
//     order. Bytes are identical; we just verify shape after reversal.
//
//   - ggml's RoPE consumes the interleaved-pair `(x_{2i}, x_{2i+1})` form
//     directly, which is also what our network expects. No row permutation
//     of q_proj / k_proj is needed (the safetensors path needed it because
//     HF stored `[first_half | second_half]`).
//
//   - q_norm.weight in GGUF is the raw trained value (~1.0), with no
//     `sqrt(head_dim)` baked in. Our `ml.attention` always applies
//     `1/sqrt(head_size)` internally; we absorb it by pre-multiplying
//     q_norm by `sqrt(head_dim)` here, same as the safetensors path.
//
//   - The Q6_K `token_embd` is dequantized into a Bf16 buffer at load so
//     the existing select-row embedding path keeps working unchanged. The
//     5.6 GB Bf16 `per_layer_token_embd` is copied verbatim into the
//     existing host-side byte buffer.
//
// GGUF tensor name → Gemma field map (per-layer prefix `blk.<i>.`):
//
//   token_embd.weight                 → embed_tokens_weight        (Q6_K  → Bf16 dequant)
//   per_layer_token_embd.weight       → embed_tokens_per_layer_bytes (BF16 host)
//   per_layer_model_proj.weight       → per_layer_model_projection_weight (Q4_K)
//   per_layer_proj_norm.weight        → per_layer_projection_norm_weight  (F32 → Bf16)
//   output_norm.weight                → output_norm_weight                (F32 → Bf16)
//   blk.<i>.attn_norm.weight          → input_norm_weight
//   blk.<i>.attn_q.weight             → q_proj_weight                     (Q4_K)
//   blk.<i>.attn_q_norm.weight        → q_norm_weight (× sqrt(head_dim))
//   blk.<i>.attn_k.weight             → k_proj_weight                     (Q4_K)
//   blk.<i>.attn_k_norm.weight        → k_norm_weight
//   blk.<i>.attn_v.weight             → v_proj_weight                     (Q6_K)
//   blk.<i>.attn_output.weight        → o_proj_weight                     (Q4_K)
//   blk.<i>.post_attention_norm.weight→ post_attention_norm_weight
//   blk.<i>.ffn_norm.weight           → pre_feedforward_norm_weight
//   blk.<i>.post_ffw_norm.weight      → post_feedforward_norm_weight
//   blk.<i>.post_norm.weight          → post_per_layer_input_norm_weight
//   blk.<i>.ffn_gate.weight           → gate_proj_weight                  (Q4_K)
//   blk.<i>.ffn_up.weight             → up_proj_weight                    (Q4_K)
//   blk.<i>.ffn_down.weight           → down_proj_weight                  (Q6_K)
//   blk.<i>.inp_gate.weight           → per_layer_input_gate_weight       (Q4_K)
//   blk.<i>.proj.weight               → per_layer_projection_weight       (Q4_K)
//   blk.<i>.layer_output_scale.weight → layer_scalar
@(require_results)
load_gguf :: proc(model: ^Gemma, path: string) -> bool {
	loader, load_ok := gguf.load(path)
	if !load_ok do return false
	defer gguf.destroy(loader)

	cfg := model.config

	// Globals.
	if !_load_dequant_to_bf16(loader, model.embed_tokens_weight, "token_embd.weight")              do return false
	if !_load_norm_f32_to_dtype(loader, model.output_norm_weight, "output_norm.weight", 1.0)        do return false
	if !_load_per_layer_token_embd(loader, model^)                                                  do return false
	if !_load_quant_passthrough(loader, &model.per_layer_model_projection_weight, "per_layer_model_proj.weight") do return false
	if !_load_norm_f32_to_dtype(loader, model.per_layer_projection_norm_weight, "per_layer_proj_norm.weight", 1.0) do return false

	for &layer, layer_idx in model.layers {
		head_dim   := config_head_dim(cfg, layer_idx)
		q_norm_scale := math.sqrt(f32(head_dim))
		prefix     := fmt.tprintf("blk.%v", layer_idx)

		ok := _load_norm_f32_to_dtype(loader, layer.input_norm_weight,                fmt.tprintf("%v.attn_norm.weight",            prefix), 1.0) &&
		      _load_norm_f32_to_dtype(loader, layer.post_attention_norm_weight,       fmt.tprintf("%v.post_attention_norm.weight",  prefix), 1.0) &&
		      _load_norm_f32_to_dtype(loader, layer.pre_feedforward_norm_weight,      fmt.tprintf("%v.ffn_norm.weight",             prefix), 1.0) &&
		      _load_norm_f32_to_dtype(loader, layer.post_feedforward_norm_weight,     fmt.tprintf("%v.post_ffw_norm.weight",        prefix), 1.0) &&
		      _load_norm_f32_to_dtype(loader, layer.post_per_layer_input_norm_weight, fmt.tprintf("%v.post_norm.weight",            prefix), 1.0) &&
		      _load_norm_f32_to_dtype(loader, layer.q_norm_weight,                    fmt.tprintf("%v.attn_q_norm.weight",          prefix), q_norm_scale) &&
		      _load_norm_f32_to_dtype(loader, layer.layer_scalar,                     fmt.tprintf("%v.layer_output_scale.weight",   prefix), 1.0) &&
		      _load_rope_permuted_q(loader, &layer.q_proj_weight,               fmt.tprintf("%v.attn_q.weight",               prefix), cfg.num_attention_heads, head_dim) &&
		      _load_quant_passthrough(loader, &layer.o_proj_weight,                   fmt.tprintf("%v.attn_output.weight",          prefix)) &&
		      _load_quant_passthrough(loader, &layer.gate_proj_weight,                fmt.tprintf("%v.ffn_gate.weight",             prefix)) &&
		      _load_quant_passthrough(loader, &layer.up_proj_weight,                  fmt.tprintf("%v.ffn_up.weight",               prefix)) &&
		      _load_quant_passthrough(loader, &layer.down_proj_weight,                fmt.tprintf("%v.ffn_down.weight",             prefix)) &&
		      _load_quant_passthrough(loader, &layer.per_layer_input_gate_weight,     fmt.tprintf("%v.inp_gate.weight",             prefix)) &&
		      _load_quant_passthrough(loader, &layer.per_layer_projection_weight,     fmt.tprintf("%v.proj.weight",                 prefix))
		if !ok do return false

		if !is_kv_shared_layer(cfg, layer_idx) {
			ok2 := _load_rope_permuted_q(loader, &layer.k_proj_weight,       fmt.tprintf("%v.attn_k.weight", prefix), cfg.num_key_value_heads, head_dim) &&
			       _load_quant_passthrough(loader, &layer.v_proj_weight,           fmt.tprintf("%v.attn_v.weight",      prefix)) &&
			       _load_norm_f32_to_dtype(loader, layer.k_norm_weight,            fmt.tprintf("%v.attn_k_norm.weight", prefix), 1.0)
			if !ok2 do return false
		}
	}

	// Gemma 4 E4B has tied embeddings — `lm_head_weight` already aliases
	// `embed_tokens_weight`, which we just populated. Nothing to do.
	if !cfg.tie_word_embeddings {
		fmt.eprintfln("gemma.load_gguf: untied lm_head not present in this GGUF; not implemented")
		return false
	}

	return true
}

// Read a GGUF Q4_K or Q6_K weight tensor and replace the Bf16 placeholder
// in `target_ptr` with a freshly allocated tensor of matching dtype, then
// copy the GGUF bytes verbatim. F32 / Bf16 source dtypes are also accepted
// for sanity (would convert via the norm path normally).
_load_quant_passthrough :: proc(loader: gguf.Loader, target_ptr: ^ml.Tensor, name: string) -> bool {
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_gguf: missing tensor %q", name)
		return false
	}

	target := target_ptr^
	shape_buf := target.shape
	target_shape := shape_buf[:target.rank]
	if !_shape_matches_reversed(info.shape, target_shape) {
		fmt.eprintfln("gemma.load_gguf: %q shape %v doesn't match target shape (reversed) %v", name, info.shape, target_shape)
		return false
	}

	new_dtype: ml.Data_Type
	#partial switch info.type {
	case .Q4_K: new_dtype = .Q4_K
	case .Q6_K: new_dtype = .Q6_K
	case:
		fmt.eprintfln("gemma.load_gguf: %q expected Q4_K or Q6_K, got %v", name, info.type)
		return false
	}

	bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok do return false

	// Replace the Bf16 placeholder with a typed tensor. The shape stays
	// the same; only `type` and the underlying byte buffer size change.
	ml.destroy(target)
	new_t := ml.alloc(new_dtype, target_shape, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(new_t, bytes)
	target_ptr^ = new_t
	return true
}

// q_proj / k_proj weights need a row permutation on load: GGUF stores them
// in HF "proportional" / split-half form (`[first_half | second_half]` per
// head), but ggml's neox-style RoPE — which is what our `ml.rope` op does —
// expects interleaved-pair form. The convert_hf_to_gguf.py for Gemma 4 has
// a comment noting it intentionally doesn't permute and uses ROPE_FREQS to
// disable rotation on the unrotated dims; we instead permute on load.
//
// Q4_K (and Q6_K) quantize along the inner (input_size) axis into 256-element
// super-blocks; the permutation reorders OUTPUT ROWS, leaving each row's
// super-blocks untouched. So we can do this as a pure byte-level row shuffle
// without dequantizing — preserving the Q4_K / Q6_K compression for q_proj
// and k_proj (~500 MB win on E4B vs an f32-dequant-and-bf16-encode path).
_load_rope_permuted_q :: proc(loader: gguf.Loader, target_ptr: ^ml.Tensor, name: string, head_count, head_size: int) -> bool {
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_gguf: missing tensor %q", name)
		return false
	}

	target := target_ptr^
	shape_buf := target.shape
	target_shape := shape_buf[:target.rank]
	if !_shape_matches_reversed(info.shape, target_shape) {
		fmt.eprintfln("gemma.load_gguf: %q shape %v doesn't match target shape %v", name, info.shape, target_shape)
		return false
	}
	if target.rank != 2 || target_shape[0] != head_count * head_size {
		fmt.eprintfln("gemma.load_gguf: %q expected [%v, embed], got %v", name, head_count * head_size, target_shape)
		return false
	}

	new_dtype: ml.Data_Type
	bytes_per_block: int
	#partial switch info.type {
	case .Q4_K: new_dtype = .Q4_K; bytes_per_block = ml.Q4_K_BLOCK_BYTES
	case .Q6_K: new_dtype = .Q6_K; bytes_per_block = ml.Q6_K_BLOCK_BYTES
	case:
		fmt.eprintfln("gemma.load_gguf: %q unsupported source dtype %v for row-permuted load", name, info.type)
		return false
	}

	src_bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok do return false

	embedding_size := target_shape[1]
	half_size      := head_size / 2
	if embedding_size % ml.K_QUANT_BLOCK_SIZE != 0 {
		fmt.eprintfln("gemma.load_gguf: %q embedding_size=%v not a multiple of 256", name, embedding_size)
		return false
	}
	row_bytes := (embedding_size / ml.K_QUANT_BLOCK_SIZE) * bytes_per_block
	total_bytes := target_shape[0] * row_bytes
	if builtin.len(src_bytes) != total_bytes {
		fmt.eprintfln("gemma.load_gguf: %q byte count %v != expected %v", name, builtin.len(src_bytes), total_bytes)
		return false
	}

	// Per head, src row `i` → dst row `2i`, src row `half_size + i` → dst row `2i+1`.
	dst_bytes := builtin.make([]byte, total_bytes, context.temp_allocator)
	for h in 0 ..< head_count {
		head_offset := h * head_size * row_bytes
		for i in 0 ..< half_size {
			even_dst := head_offset + (2 * i + 0)     * row_bytes
			odd_dst  := head_offset + (2 * i + 1)     * row_bytes
			even_src := head_offset + (i)             * row_bytes
			odd_src  := head_offset + (half_size + i) * row_bytes
			builtin.copy(dst_bytes[even_dst:even_dst + row_bytes], src_bytes[even_src:even_src + row_bytes])
			builtin.copy(dst_bytes[odd_dst :odd_dst  + row_bytes], src_bytes[odd_src :odd_src  + row_bytes])
		}
	}

	ml.destroy(target)
	new_t := ml.alloc(new_dtype, target_shape, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(new_t, dst_bytes)
	target_ptr^ = new_t
	return true
}

// Read a 1-D (or 1-element) F32 tensor from GGUF, multiply by `extra_scale`,
// and write into the model tensor at the model's dtype.
_load_norm_f32_to_dtype :: proc(loader: gguf.Loader, target: ml.Tensor, name: string, extra_scale: f32) -> bool {
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_gguf: missing tensor %q", name)
		return false
	}
	if info.type != .F32 {
		fmt.eprintfln("gemma.load_gguf: %q expected F32 norm, got %v", name, info.type)
		return false
	}
	shape_buf := target.shape
	target_shape := shape_buf[:target.rank]
	if !_shape_matches_reversed(info.shape, target_shape) {
		fmt.eprintfln("gemma.load_gguf: %q shape %v doesn't match target shape %v", name, info.shape, target_shape)
		return false
	}

	bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok do return false

	count := ml.len(target)
	src   := slice.from_ptr((^f32)(raw_data(bytes)), count)

	#partial switch target.type {
	case .F32:
		if extra_scale == 1.0 {
			ml.set_data_bytes(target, bytes)
		} else {
			scaled := builtin.make([]f32, count, context.temp_allocator)
			for v, i in src do scaled[i] = v * extra_scale
			ml.set_data(target, scaled)
		}
	case .Bf16:
		bytes_out := builtin.make([]byte, count * 2, context.temp_allocator)
		bf := ([^]ml.Bf16)(raw_data(bytes_out))
		for v, i in src do bf[i] = ml.bf16_from_f32(v * extra_scale)
		ml.set_data_bytes(target, bytes_out)
	case:
		fmt.eprintfln("gemma.load_gguf: %q unsupported target dtype %v", name, target.type)
		return false
	}
	return true
}

// Decode a Q6_K tensor into the model's Bf16 (or F32) embedding buffer.
// Used for `embed_tokens_weight`, which is Q6_K in the Q4_K_M GGUF.
_load_dequant_to_bf16 :: proc(loader: gguf.Loader, target: ml.Tensor, name: string) -> bool {
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_gguf: missing tensor %q", name)
		return false
	}
	shape_buf := target.shape
	target_shape := shape_buf[:target.rank]
	if !_shape_matches_reversed(info.shape, target_shape) {
		fmt.eprintfln("gemma.load_gguf: %q shape %v doesn't match target shape %v", name, info.shape, target_shape)
		return false
	}

	bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok do return false

	count := ml.len(target)
	floats := builtin.make([]f32, count, context.temp_allocator)

	#partial switch info.type {
	case .Q6_K: gguf.dequantize_q6_k(bytes, floats)
	case .Q4_K: gguf.dequantize_q4_k(bytes, floats)
	case:
		fmt.eprintfln("gemma.load_gguf: %q expected Q4_K or Q6_K source for dequant load, got %v", name, info.type)
		return false
	}

	#partial switch target.type {
	case .F32:
		ml.set_data(target, floats)
	case .Bf16:
		bytes_out := builtin.make([]byte, count * 2, context.temp_allocator)
		bf := ([^]ml.Bf16)(raw_data(bytes_out))
		for v, i in floats do bf[i] = ml.bf16_from_f32(v)
		ml.set_data_bytes(target, bytes_out)
	case:
		fmt.eprintfln("gemma.load_gguf: %q unsupported embed target dtype %v", name, target.type)
		return false
	}
	return true
}

// `embed_tokens_per_layer` is too big for a single Vulkan buffer. The
// model holds the raw bytes host-side and looks up rows per forward.
_load_per_layer_token_embd :: proc(loader: gguf.Loader, model: Gemma) -> bool {
	name := "per_layer_token_embd.weight"
	info, info_ok := gguf.get_info(loader, name)
	if !info_ok {
		fmt.eprintfln("gemma.load_gguf: missing %q", name)
		return false
	}
	cfg := model.config
	expected_unreversed := []int{cfg.vocab_size, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input}
	if !_shape_matches_reversed(info.shape, expected_unreversed) {
		fmt.eprintfln("gemma.load_gguf: %q shape %v != expected (reversed of %v)", name, info.shape, expected_unreversed)
		return false
	}

	bytes, bytes_ok := gguf.get_bytes(loader, name)
	if !bytes_ok do return false

	count := cfg.vocab_size * cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
	#partial switch info.type {
	case .BF16:
		if model.dtype != .Bf16 {
			// Pre-cast to model dtype; the only realistic non-Bf16 case is F32.
			fmt.eprintfln("gemma.load_gguf: %q BF16 source needs Bf16 model dtype (got %v)", name, model.dtype)
			return false
		}
		if builtin.len(bytes) != count * 2 {
			fmt.eprintfln("gemma.load_gguf: %q BF16 byte count %v != expected %v", name, builtin.len(bytes), count * 2)
			return false
		}
		builtin.copy(model.embed_tokens_per_layer_bytes, bytes)
	case .F32:
		if model.dtype == .F32 {
			builtin.copy(model.embed_tokens_per_layer_bytes, bytes)
		} else {
			src := slice.from_ptr((^f32)(raw_data(bytes)), count)
			dst := ([^]ml.Bf16)(raw_data(model.embed_tokens_per_layer_bytes))
			for v, i in src do dst[i] = ml.bf16_from_f32(v)
		}
	case:
		fmt.eprintfln("gemma.load_gguf: %q unsupported source dtype %v", name, info.type)
		return false
	}
	return true
}

// GGUF stores shape with `ne[0]` (fastest-varying) first; our row-major
// shapes have outer dim first. Compare reversed.
_shape_matches_reversed :: proc(gguf_shape, target_shape: []int) -> bool {
	if builtin.len(gguf_shape) != builtin.len(target_shape) do return false
	n := builtin.len(gguf_shape)
	for i in 0 ..< n {
		if gguf_shape[i] != target_shape[n - 1 - i] do return false
	}
	return true
}
