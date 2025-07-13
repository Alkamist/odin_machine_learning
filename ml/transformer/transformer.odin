package machine_learning_transformer

import "base:builtin"
import "core:math"
import ml "../"

Layer :: struct {
	norm0_weight:    ml.Parameter, // [embedding_size]
	qkv_weight:      ml.Parameter, // [embedding_size, 3 * embedding_size]
	proj_weight:     ml.Parameter, // [embedding_size, embedding_size]
	norm1_weight:    ml.Parameter, // [embedding_size]
	mlp_up_weight:   ml.Parameter, // [embedding_size, 4 * embedding_size]
	mlp_down_weight: ml.Parameter, // [4 * embedding_size, embedding_size]
}

Transformer :: struct {
	head_count:      int,
	embedding_size:  int,
	vocabulary_size: int,

	token_embeddings: ml.Parameter, // [vocabulary_size, embedding_size]

	layers: []Layer,

	norm_weight:   ml.Parameter, // [embedding_size]
	output_weight: ml.Parameter, // [embedding_size, vocabulary_size]
}

make :: proc(layer_count, head_count, embedding_size, vocabulary_size: int, allocator := context.allocator) -> (transformer: Transformer) {
	transformer.head_count      = head_count
	transformer.embedding_size  = embedding_size
	transformer.vocabulary_size = vocabulary_size

	transformer.layers = builtin.make([]Layer, layer_count, allocator=allocator)

	transformer.token_embeddings = ml.make(vocabulary_size * embedding_size, allocator=allocator)

	for &layer in transformer.layers {
		layer.norm0_weight = ml.make(embedding_size,                      allocator=allocator)
		layer.qkv_weight   = ml.make(embedding_size * 3 * embedding_size, allocator=allocator)
		layer.proj_weight  = ml.make(embedding_size * embedding_size,     allocator=allocator)
		layer.norm1_weight = ml.make(embedding_size,                      allocator=allocator)

		hidden_size := 4 * embedding_size
		layer.mlp_up_weight   = ml.make(embedding_size * hidden_size,    allocator=allocator)
		layer.mlp_down_weight = ml.make(hidden_size    * embedding_size, allocator=allocator)
	}

	transformer.norm_weight   = ml.make(embedding_size,                   allocator=allocator)
	transformer.output_weight = ml.make(embedding_size * vocabulary_size, allocator=allocator)

	randomize(transformer)

	return
}

destroy :: proc(transformer: Transformer) {
	ml.destroy(transformer.token_embeddings)

	for layer in transformer.layers {
		ml.destroy(layer.norm0_weight)
		ml.destroy(layer.qkv_weight)
		ml.destroy(layer.proj_weight)
		ml.destroy(layer.norm1_weight)
		ml.destroy(layer.mlp_up_weight)
		ml.destroy(layer.mlp_down_weight)
	}

	ml.destroy(transformer.norm_weight)
	ml.destroy(transformer.output_weight)

	delete(transformer.layers)
}

copy :: proc(dst, src: Transformer) {
	ml.copy(dst.token_embeddings, src.token_embeddings)

	for i in 0 ..< len(dst.layers) {
		ml.copy(dst.layers[i].norm0_weight,    src.layers[i].norm0_weight)
		ml.copy(dst.layers[i].qkv_weight,      src.layers[i].qkv_weight)
		ml.copy(dst.layers[i].proj_weight,     src.layers[i].proj_weight)
		ml.copy(dst.layers[i].norm1_weight,    src.layers[i].norm1_weight)
		ml.copy(dst.layers[i].mlp_up_weight,   src.layers[i].mlp_up_weight)
		ml.copy(dst.layers[i].mlp_down_weight, src.layers[i].mlp_down_weight)
	}

	ml.copy(dst.norm_weight,   src.norm_weight)
	ml.copy(dst.output_weight, src.output_weight)
}

randomize :: proc(transformer: Transformer) {
	layer_count    := len(transformer.layers)
	embedding_size := transformer.embedding_size

	ml.fill_normal(transformer.token_embeddings, 0, 0.02)

	for &layer in transformer.layers {
		ml.fill_value(layer.norm0_weight, 1)
		ml.fill_normal(layer.qkv_weight, 0, 0.02)
		ml.fill_normal(layer.proj_weight, 0, 0.02 / math.sqrt(f32(2 * layer_count)))
		ml.fill_value(layer.norm1_weight, 1)

		hidden_size := 4 * embedding_size
		ml.he_initialization(layer.mlp_up_weight,   embedding_size)
		ml.he_initialization(layer.mlp_down_weight, hidden_size)
	}

	ml.fill_value(transformer.norm_weight, 1)
	ml.fill_normal(transformer.output_weight, 0, 0.02)
}

@(require_results)
forward :: proc(transformer: Transformer, tokens: []int) -> (output: ml.Array) {
	token_count := len(tokens)

	// Token embeddings
	output = ml.select(transformer.token_embeddings, tokens, transformer.embedding_size) // [token_count, embedding_size]

	// Start the residual connection
	residual := output // [token_count, embedding_size]

	for layer in transformer.layers {
		// Layernorm
		norm_output := ml.layernorm(residual, layer.norm0_weight, token_count) // [token_count, embedding_size]

		// Linear projection for Q, K, V
		qkv := ml.linear(norm_output, layer.qkv_weight, token_count) // [token_count, 3 * embedding_size]

		// Deinterleave Q, K, and V into separate arrays
		q := ml.deinterleave(qkv, 0, 3) // [token_count, embedding_size]
		k := ml.deinterleave(qkv, 1, 3) // [token_count, embedding_size] 
		v := ml.deinterleave(qkv, 2, 3) // [token_count, embedding_size]

		// Apply rope to Q and K
		q = ml.rope(q, token_count, transformer.head_count) // [token_count, embedding_size]
		k = ml.rope(k, token_count, transformer.head_count) // [token_count, embedding_size]

		// Interleave back into qkv
		qkv = ml.interleave(q, k, v)

		// Self-attention with causal masking
		attn_output := ml.attention(qkv, token_count, transformer.head_count) // [token_count, embedding_size]
		attn_output  = ml.linear(attn_output, layer.proj_weight, token_count) // [token_count, embedding_size]

		// Residual connection
		residual = ml.add(residual, attn_output) // [token_count, embedding_size]

		// Layernorm
		norm_output = ml.layernorm(residual, layer.norm1_weight, token_count) // [token_count, embedding_size]

		// MLP: up projection -> activation -> down projection
		mlp_output := ml.linear(norm_output, layer.mlp_up_weight, token_count)  // [token_count, 4 * embedding_size]
		mlp_output  = ml.gelu(mlp_output)                                       // [token_count, 4 * embedding_size]
		mlp_output  = ml.linear(mlp_output, layer.mlp_down_weight, token_count) // [token_count, embedding_size]

		// Residual connection
		residual = ml.add(residual, mlp_output) // [token_count, embedding_size]
	}

	// Layernorm
	output = ml.layernorm(residual, transformer.norm_weight, token_count) // [token_count, embedding_size]

	// Project to vocabulary size for logits
	output = ml.linear(output, transformer.output_weight, token_count) // [token_count, vocabulary_size]

	return
}

update :: proc(opt: ml.Optimizer, transformer: Transformer) {
	ml.update(opt, transformer.token_embeddings)

	for layer in transformer.layers {
		ml.update(opt, layer.norm0_weight)
		ml.update(opt, layer.qkv_weight)
		ml.update(opt, layer.proj_weight)
		ml.update(opt, layer.norm1_weight)
		ml.update(opt, layer.mlp_up_weight)
		ml.update(opt, layer.mlp_down_weight)
	}

	ml.update(opt, transformer.norm_weight)
	ml.update(opt, transformer.output_weight)
}