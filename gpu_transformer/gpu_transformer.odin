// GPU mirror of the CPU transformer package: Layer/Transformer hold GPU
// tensors instead of ml.Parameters, and forward walks the same op sequence
// using gpu.* kernels.
//
// Activations are tracked in an `Activations` accumulator so the caller can
// free everything allocated during a forward pass with a single call. No
// pooling yet — each forward call allocates fresh tensors and `destroy`s
// them; works fine while we're verifying correctness, will need a per-shape
// reuse pool for inference perf.
package machine_learning_gpu_transformer

import "base:builtin"
import "core:fmt"
import "core:math"
import ml  "../"
import gpu "../gpu"
import tfm "../transformer"

// Each weight has paired gradient + Adam moment tensors of identical
// shape. Forward uses `*_weight`; backward uses `*_weight_grad`; update
// reads/writes all four. Parallel fields rather than a wrapper struct
// to keep forward call sites uncluttered.
Layer :: struct {
	norm0_weight:         gpu.GpuTensor, // [embedding_size]
	norm0_weight_grad:    gpu.GpuTensor,
	norm0_weight_m:       gpu.GpuTensor,
	norm0_weight_v:       gpu.GpuTensor,
	qkv_weight:           gpu.GpuTensor, // [3*embedding_size, embedding_size]
	qkv_weight_grad:      gpu.GpuTensor,
	qkv_weight_m:         gpu.GpuTensor,
	qkv_weight_v:         gpu.GpuTensor,
	proj_weight:          gpu.GpuTensor, // [embedding_size, embedding_size]
	proj_weight_grad:     gpu.GpuTensor,
	proj_weight_m:        gpu.GpuTensor,
	proj_weight_v:        gpu.GpuTensor,
	norm1_weight:         gpu.GpuTensor, // [embedding_size]
	norm1_weight_grad:    gpu.GpuTensor,
	norm1_weight_m:       gpu.GpuTensor,
	norm1_weight_v:       gpu.GpuTensor,
	mlp_up_weight:        gpu.GpuTensor, // [4*embedding_size, embedding_size]
	mlp_up_weight_grad:   gpu.GpuTensor,
	mlp_up_weight_m:      gpu.GpuTensor,
	mlp_up_weight_v:      gpu.GpuTensor,
	mlp_down_weight:      gpu.GpuTensor, // [embedding_size, 4*embedding_size]
	mlp_down_weight_grad: gpu.GpuTensor,
	mlp_down_weight_m:    gpu.GpuTensor,
	mlp_down_weight_v:    gpu.GpuTensor,
}

Transformer :: struct {
	head_count:      int,
	embedding_size:  int,
	vocabulary_size: int,

	token_embeddings:      gpu.GpuTensor, // [vocab, embedding_size]
	token_embeddings_grad: gpu.GpuTensor,
	token_embeddings_m:    gpu.GpuTensor,
	token_embeddings_v:    gpu.GpuTensor,
	layers:                []Layer,
	norm_weight:           gpu.GpuTensor, // [embedding_size]
	norm_weight_grad:      gpu.GpuTensor,
	norm_weight_m:         gpu.GpuTensor,
	norm_weight_v:         gpu.GpuTensor,
	output_weight:         gpu.GpuTensor, // [vocab, embedding_size]
	output_weight_grad:    gpu.GpuTensor,
	output_weight_m:       gpu.GpuTensor,
	output_weight_v:       gpu.GpuTensor,
}

// Per-shape activation pool. forward acquires tensors by index; the
// allocation sequence is deterministic for a given (model config, token
// count), so once the pool has grown to the steady-state size every
// subsequent forward reuses tensors with zero Vulkan alloc traffic.
//
// Tensors stay allocated for the lifetime of the pool. Caller is
// responsible for not changing token_count between forwards if they want
// to keep reusing — a different shape sequence will trip the shape assert
// in _act. Different shapes need different pools (or destroy + recreate).
// `data[i]` is the i-th allocated activation; `grad[i]` is its gradient
// buffer of identical shape. Both grow lazily in lockstep on the first
// forward pass and are reused on subsequent calls.
Activations :: struct {
	data: [dynamic]gpu.GpuTensor,
	grad: [dynamic]gpu.GpuTensor,
	next: int,
}

destroy_activations :: proc(a: ^Activations) {
	for t in a.data { gpu.destroy_tensor(t) }
	for t in a.grad { gpu.destroy_tensor(t) }
	delete(a.data); a.data = nil
	delete(a.grad); a.grad = nil
	a.next = 0
}

make :: proc(layer_count, head_count, embedding_size, vocabulary_size: int) -> (t: Transformer) {
	t.head_count      = head_count
	t.embedding_size  = embedding_size
	t.vocabulary_size = vocabulary_size

	t.layers = builtin.make([]Layer, layer_count)

	t.token_embeddings      = gpu.alloc(vocabulary_size, embedding_size)
	t.token_embeddings_grad = gpu.alloc(vocabulary_size, embedding_size)
	t.token_embeddings_m    = gpu.alloc(vocabulary_size, embedding_size)
	t.token_embeddings_v    = gpu.alloc(vocabulary_size, embedding_size)

	for &layer in t.layers {
		layer.norm0_weight      = gpu.alloc(embedding_size)
		layer.norm0_weight_grad = gpu.alloc(embedding_size)
		layer.norm0_weight_m    = gpu.alloc(embedding_size)
		layer.norm0_weight_v    = gpu.alloc(embedding_size)
		layer.qkv_weight        = gpu.alloc(3 * embedding_size, embedding_size)
		layer.qkv_weight_grad   = gpu.alloc(3 * embedding_size, embedding_size)
		layer.qkv_weight_m      = gpu.alloc(3 * embedding_size, embedding_size)
		layer.qkv_weight_v      = gpu.alloc(3 * embedding_size, embedding_size)
		layer.proj_weight       = gpu.alloc(embedding_size,     embedding_size)
		layer.proj_weight_grad  = gpu.alloc(embedding_size,     embedding_size)
		layer.proj_weight_m     = gpu.alloc(embedding_size,     embedding_size)
		layer.proj_weight_v     = gpu.alloc(embedding_size,     embedding_size)
		layer.norm1_weight      = gpu.alloc(embedding_size)
		layer.norm1_weight_grad = gpu.alloc(embedding_size)
		layer.norm1_weight_m    = gpu.alloc(embedding_size)
		layer.norm1_weight_v    = gpu.alloc(embedding_size)

		hidden_size := 4 * embedding_size
		layer.mlp_up_weight        = gpu.alloc(hidden_size,    embedding_size)
		layer.mlp_up_weight_grad   = gpu.alloc(hidden_size,    embedding_size)
		layer.mlp_up_weight_m      = gpu.alloc(hidden_size,    embedding_size)
		layer.mlp_up_weight_v      = gpu.alloc(hidden_size,    embedding_size)
		layer.mlp_down_weight      = gpu.alloc(embedding_size, hidden_size)
		layer.mlp_down_weight_grad = gpu.alloc(embedding_size, hidden_size)
		layer.mlp_down_weight_m    = gpu.alloc(embedding_size, hidden_size)
		layer.mlp_down_weight_v    = gpu.alloc(embedding_size, hidden_size)
	}

	t.norm_weight        = gpu.alloc(embedding_size)
	t.norm_weight_grad   = gpu.alloc(embedding_size)
	t.norm_weight_m      = gpu.alloc(embedding_size)
	t.norm_weight_v      = gpu.alloc(embedding_size)
	t.output_weight      = gpu.alloc(vocabulary_size, embedding_size)
	t.output_weight_grad = gpu.alloc(vocabulary_size, embedding_size)
	t.output_weight_m    = gpu.alloc(vocabulary_size, embedding_size)
	t.output_weight_v    = gpu.alloc(vocabulary_size, embedding_size)

	// Vulkan doesn't zero device memory at alloc — clear grads + moments
	// so the first training step doesn't accumulate into garbage.
	_clear_training_state(t)
	return
}

_clear_training_state :: proc(t: Transformer) {
	gpu.begin_batch()
	defer gpu.end_batch()

	gpu.zero(t.token_embeddings_grad); gpu.zero(t.token_embeddings_m); gpu.zero(t.token_embeddings_v)
	for layer in t.layers {
		gpu.zero(layer.norm0_weight_grad);    gpu.zero(layer.norm0_weight_m);    gpu.zero(layer.norm0_weight_v)
		gpu.zero(layer.qkv_weight_grad);      gpu.zero(layer.qkv_weight_m);      gpu.zero(layer.qkv_weight_v)
		gpu.zero(layer.proj_weight_grad);     gpu.zero(layer.proj_weight_m);     gpu.zero(layer.proj_weight_v)
		gpu.zero(layer.norm1_weight_grad);    gpu.zero(layer.norm1_weight_m);    gpu.zero(layer.norm1_weight_v)
		gpu.zero(layer.mlp_up_weight_grad);   gpu.zero(layer.mlp_up_weight_m);   gpu.zero(layer.mlp_up_weight_v)
		gpu.zero(layer.mlp_down_weight_grad); gpu.zero(layer.mlp_down_weight_m); gpu.zero(layer.mlp_down_weight_v)
	}
	gpu.zero(t.norm_weight_grad);   gpu.zero(t.norm_weight_m);   gpu.zero(t.norm_weight_v)
	gpu.zero(t.output_weight_grad); gpu.zero(t.output_weight_m); gpu.zero(t.output_weight_v)
}

destroy :: proc(t: Transformer) {
	gpu.destroy_tensor(t.token_embeddings);      gpu.destroy_tensor(t.token_embeddings_grad)
	gpu.destroy_tensor(t.token_embeddings_m);    gpu.destroy_tensor(t.token_embeddings_v)
	for layer in t.layers {
		gpu.destroy_tensor(layer.norm0_weight);    gpu.destroy_tensor(layer.norm0_weight_grad)
		gpu.destroy_tensor(layer.norm0_weight_m);  gpu.destroy_tensor(layer.norm0_weight_v)
		gpu.destroy_tensor(layer.qkv_weight);      gpu.destroy_tensor(layer.qkv_weight_grad)
		gpu.destroy_tensor(layer.qkv_weight_m);    gpu.destroy_tensor(layer.qkv_weight_v)
		gpu.destroy_tensor(layer.proj_weight);     gpu.destroy_tensor(layer.proj_weight_grad)
		gpu.destroy_tensor(layer.proj_weight_m);   gpu.destroy_tensor(layer.proj_weight_v)
		gpu.destroy_tensor(layer.norm1_weight);    gpu.destroy_tensor(layer.norm1_weight_grad)
		gpu.destroy_tensor(layer.norm1_weight_m);  gpu.destroy_tensor(layer.norm1_weight_v)
		gpu.destroy_tensor(layer.mlp_up_weight);   gpu.destroy_tensor(layer.mlp_up_weight_grad)
		gpu.destroy_tensor(layer.mlp_up_weight_m); gpu.destroy_tensor(layer.mlp_up_weight_v)
		gpu.destroy_tensor(layer.mlp_down_weight); gpu.destroy_tensor(layer.mlp_down_weight_grad)
		gpu.destroy_tensor(layer.mlp_down_weight_m); gpu.destroy_tensor(layer.mlp_down_weight_v)
	}
	gpu.destroy_tensor(t.norm_weight);   gpu.destroy_tensor(t.norm_weight_grad)
	gpu.destroy_tensor(t.norm_weight_m); gpu.destroy_tensor(t.norm_weight_v)
	gpu.destroy_tensor(t.output_weight);   gpu.destroy_tensor(t.output_weight_grad)
	gpu.destroy_tensor(t.output_weight_m); gpu.destroy_tensor(t.output_weight_v)
	delete(t.layers)
}

// Copy weights from a CPU transformer to the matching GPU transformer.
// Shapes must already match (same layer_count / head_count / embed / vocab).
upload :: proc(dst: Transformer, src: tfm.Transformer, loc := #caller_location) {
	fmt.assertf(len(dst.layers) == len(src.layers),
		"upload: layer count mismatch dst=%v src=%v", len(dst.layers), len(src.layers), loc=loc)
	fmt.assertf(dst.embedding_size == src.embedding_size && dst.head_count == src.head_count && dst.vocabulary_size == src.vocabulary_size,
		"upload: model dim mismatch", loc=loc)

	gpu.upload(src.token_embeddings.data, dst.token_embeddings)
	for i in 0 ..< len(dst.layers) {
		gpu.upload(src.layers[i].norm0_weight.data,    dst.layers[i].norm0_weight)
		gpu.upload(src.layers[i].qkv_weight.data,      dst.layers[i].qkv_weight)
		gpu.upload(src.layers[i].proj_weight.data,     dst.layers[i].proj_weight)
		gpu.upload(src.layers[i].norm1_weight.data,    dst.layers[i].norm1_weight)
		gpu.upload(src.layers[i].mlp_up_weight.data,   dst.layers[i].mlp_up_weight)
		gpu.upload(src.layers[i].mlp_down_weight.data, dst.layers[i].mlp_down_weight)
	}
	gpu.upload(src.norm_weight.data,   dst.norm_weight)
	gpu.upload(src.output_weight.data, dst.output_weight)
}

// Forward pass mirroring tfm.forward. `acts` collects every activation
// tensor; caller frees them all via destroy_activations. The returned
// tensor is one of the entries in `acts.tensors` — don't free it separately.
@(require_results)
forward :: proc(model: Transformer, tokens: []int, acts: ^Activations) -> gpu.GpuTensor {
	n           := len(tokens)
	embed       := model.embedding_size
	head_count  := model.head_count
	head_size   := embed / head_count
	hidden_size := 4 * embed
	vocab       := model.vocabulary_size

	acts.next = 0

	gpu.begin_batch()
	defer gpu.end_batch()

	// Embedding lookup: [n, embed]
	output := _act(acts, n, embed)
	gpu.select(model.token_embeddings, tokens, output, embed)

	residual := output

	for layer in model.layers {
		// --- Attention sub-block. ---
		norm0 := _act(acts, n, embed)
		gpu.layernorm(residual, layer.norm0_weight, norm0, n, embed)

		qkv := _act(acts, n, 3 * embed)
		gpu.linear(norm0, layer.qkv_weight, qkv, n, embed, 3 * embed)

		q := _act(acts, n, embed)
		k := _act(acts, n, embed)
		v := _act(acts, n, embed)
		gpu.slice_trailing(qkv, q, n, 3 * embed, 0,         embed)
		gpu.slice_trailing(qkv, k, n, 3 * embed, embed,     2 * embed)
		gpu.slice_trailing(qkv, v, n, 3 * embed, 2 * embed, 3 * embed)

		q_rot := _act(acts, n, embed)
		k_rot := _act(acts, n, embed)
		gpu.rope(q, q_rot, n, head_count, head_size)
		gpu.rope(k, k_rot, n, head_count, head_size)

		qkv2 := _act(acts, n, 3 * embed)
		gpu.concat3(q_rot, k_rot, v, qkv2, n, embed, embed, embed)

		attn := _act(acts, n, embed)
		gpu.attention(qkv2, attn, n, head_count, head_size)

		attn_proj := _act(acts, n, embed)
		gpu.linear(attn, layer.proj_weight, attn_proj, n, embed, embed)

		residual_attn := _act(acts, n, embed)
		gpu.add(residual, attn_proj, residual_attn)
		residual = residual_attn

		// --- MLP sub-block. ---
		norm1 := _act(acts, n, embed)
		gpu.layernorm(residual, layer.norm1_weight, norm1, n, embed)

		mlp_up := _act(acts, n, hidden_size)
		gpu.linear(norm1, layer.mlp_up_weight, mlp_up, n, embed, hidden_size)

		mlp_act := _act(acts, n, hidden_size)
		gpu.gelu(mlp_up, mlp_act)

		mlp_down := _act(acts, n, embed)
		gpu.linear(mlp_act, layer.mlp_down_weight, mlp_down, n, hidden_size, embed)

		residual_mlp := _act(acts, n, embed)
		gpu.add(residual, mlp_down, residual_mlp)
		residual = residual_mlp
	}

	final_norm := _act(acts, n, embed)
	gpu.layernorm(residual, model.norm_weight, final_norm, n, embed)

	final_logits := _act(acts, n, vocab)
	gpu.linear(final_norm, model.output_weight, final_logits, n, embed, vocab)

	return final_logits
}

// Adam state, mirroring `ml.Optimizer`. Caller stores one of these and
// passes the same instance to `update` each step. `iteration` is bumped
// inside `update` so the bias corrections advance.
Optimizer :: struct {
	iteration:     u64,
	learning_rate: f32,
	beta1:         f32,
	beta2:         f32,
	epsilon:       f32,
	weight_decay:  f32,
}

// In-place Adam update of every model parameter, plus zeroing of all
// gradient buffers. Drop-in for `tfm.update + ml.optimize(period=1)`.
update :: proc(opt: ^Optimizer, model: Transformer, loc := #caller_location) {
	if opt.learning_rate == 0 { opt.learning_rate = 0.001 }
	if opt.beta1         == 0 { opt.beta1         = 0.9   }
	if opt.beta2         == 0 { opt.beta2         = 0.999 }
	if opt.epsilon       == 0 { opt.epsilon       = 1e-8  }

	opt.iteration += 1
	bc1 := 1 - math.pow(opt.beta1, f32(opt.iteration))
	bc2 := 1 - math.pow(opt.beta2, f32(opt.iteration))

	gpu.begin_batch()
	defer gpu.end_batch()

	step :: proc(opt: ^Optimizer, bc1, bc2: f32, x, g, m, v: gpu.GpuTensor) {
		gpu.adam_step(x, g, m, v, opt.learning_rate, opt.beta1, opt.beta2, opt.epsilon, opt.weight_decay, bc1, bc2)
	}

	step(opt, bc1, bc2, model.token_embeddings, model.token_embeddings_grad, model.token_embeddings_m, model.token_embeddings_v)
	for layer in model.layers {
		step(opt, bc1, bc2, layer.norm0_weight,    layer.norm0_weight_grad,    layer.norm0_weight_m,    layer.norm0_weight_v)
		step(opt, bc1, bc2, layer.qkv_weight,      layer.qkv_weight_grad,      layer.qkv_weight_m,      layer.qkv_weight_v)
		step(opt, bc1, bc2, layer.proj_weight,     layer.proj_weight_grad,     layer.proj_weight_m,     layer.proj_weight_v)
		step(opt, bc1, bc2, layer.norm1_weight,    layer.norm1_weight_grad,    layer.norm1_weight_m,    layer.norm1_weight_v)
		step(opt, bc1, bc2, layer.mlp_up_weight,   layer.mlp_up_weight_grad,   layer.mlp_up_weight_m,   layer.mlp_up_weight_v)
		step(opt, bc1, bc2, layer.mlp_down_weight, layer.mlp_down_weight_grad, layer.mlp_down_weight_m, layer.mlp_down_weight_v)
	}
	step(opt, bc1, bc2, model.norm_weight,   model.norm_weight_grad,   model.norm_weight_m,   model.norm_weight_v)
	step(opt, bc1, bc2, model.output_weight, model.output_weight_grad, model.output_weight_m, model.output_weight_v)
}

// Zero every parameter gradient. Call once per training step before any
// backward call accumulates into them. Activations grads are zeroed
// inside `backward` as part of its own setup.
//
// Note: `update` already zeroes grads as a side effect of the Adam kernel.
// `zero_grad` is only needed if you want to reset without stepping (e.g.
// gradient accumulation across micro-batches).
zero_grad :: proc(model: Transformer) {
	gpu.begin_batch()
	defer gpu.end_batch()

	gpu.zero(model.token_embeddings_grad)
	for layer in model.layers {
		gpu.zero(layer.norm0_weight_grad)
		gpu.zero(layer.qkv_weight_grad)
		gpu.zero(layer.proj_weight_grad)
		gpu.zero(layer.norm1_weight_grad)
		gpu.zero(layer.mlp_up_weight_grad)
		gpu.zero(layer.mlp_down_weight_grad)
	}
	gpu.zero(model.norm_weight_grad)
	gpu.zero(model.output_weight_grad)
}

// Per-layer activation handles, captured by re-walking the forward
// allocation sequence in `backward` so we can refer to each tensor by
// name during the reverse sweep.
Layer_Acts :: struct {
	input_residual: gpu.GpuTensor, // residual flowing IN to this layer
	norm0:          gpu.GpuTensor,
	qkv:            gpu.GpuTensor,
	q, k, v:        gpu.GpuTensor,
	q_rot, k_rot:   gpu.GpuTensor,
	qkv2:           gpu.GpuTensor,
	attn:           gpu.GpuTensor,
	attn_proj:      gpu.GpuTensor,
	residual_attn:  gpu.GpuTensor,
	norm1:          gpu.GpuTensor,
	mlp_up:         gpu.GpuTensor,
	mlp_act:        gpu.GpuTensor,
	mlp_down:       gpu.GpuTensor,
	residual_mlp:   gpu.GpuTensor,
}

// Backward pass. Assumes a `forward` call with the same (model, tokens,
// acts) just ran — the activation pool already holds every intermediate.
// Writes accumulated gradients into `model.*_grad` and returns the
// scalar mean cross-entropy loss.
//
// Caller is responsible for `zero_grad(model)` before the step (or before
// a sequence of accumulate-without-update micro-batch calls).
backward :: proc(model: Transformer, tokens: []int, targets: []int, acts: ^Activations, loc := #caller_location) -> f32 {
	fmt.assertf(len(targets) == len(tokens),
		"backward: targets %v != tokens %v", len(targets), len(tokens), loc=loc)

	n           := len(tokens)
	embed       := model.embedding_size
	head_count  := model.head_count
	head_size   := embed / head_count
	hidden_size := 4 * embed
	vocab       := model.vocabulary_size

	// Re-walk forward _act sequence to bind names. Pool is already populated
	// from the prior forward(), so _act here just looks up existing tensors.
	acts.next = 0
	output := _act(acts, n, embed)

	layer_acts := builtin.make([]Layer_Acts, len(model.layers), context.temp_allocator)
	residual := output
	for li in 0 ..< len(model.layers) {
		la := &layer_acts[li]
		la.input_residual = residual
		la.norm0          = _act(acts, n, embed)
		la.qkv            = _act(acts, n, 3 * embed)
		la.q              = _act(acts, n, embed)
		la.k              = _act(acts, n, embed)
		la.v              = _act(acts, n, embed)
		la.q_rot          = _act(acts, n, embed)
		la.k_rot          = _act(acts, n, embed)
		la.qkv2           = _act(acts, n, 3 * embed)
		la.attn           = _act(acts, n, embed)
		la.attn_proj      = _act(acts, n, embed)
		la.residual_attn  = _act(acts, n, embed)
		la.norm1          = _act(acts, n, embed)
		la.mlp_up         = _act(acts, n, hidden_size)
		la.mlp_act        = _act(acts, n, hidden_size)
		la.mlp_down       = _act(acts, n, embed)
		la.residual_mlp   = _act(acts, n, embed)
		residual = la.residual_mlp
	}
	final_norm   := _act(acts, n, embed)
	final_logits := _act(acts, n, vocab)

	output_g       := _act_grad(acts, output)
	final_norm_g   := _act_grad(acts, final_norm)
	final_logits_g := _act_grad(acts, final_logits)

	// Per-row loss buffer. Lives beyond end_batch (downloaded after).
	loss_buf := gpu.alloc(n)
	defer gpu.destroy_tensor(loss_buf)

	gpu.begin_batch()

	// Zero all activation grads — backward kernels accumulate.
	for g in acts.grad { gpu.zero(g) }

	gpu.cross_entropy_grad(final_logits, final_logits_g, loss_buf, targets, n, vocab)

	// Final linear: dfinal_norm += output_w^T @ dfinal_logits; output_w_grad += ...
	gpu.linear_back(final_norm, model.output_weight, final_logits_g,
		final_norm_g, model.output_weight_grad,
		n, embed, vocab)

	// Final layernorm: input is the last layer's residual_mlp.
	last := &layer_acts[len(layer_acts) - 1]
	last_resid_g := _act_grad(acts, last.residual_mlp)
	gpu.layernorm_back(last.residual_mlp, model.norm_weight, final_norm_g,
		last_resid_g, model.norm_weight_grad,
		n, embed)

	// Layers in reverse.
	for li := len(model.layers) - 1; li >= 0; li -= 1 {
		layer := &model.layers[li]
		la    := &layer_acts[li]

		norm0_g         := _act_grad(acts, la.norm0)
		qkv_g           := _act_grad(acts, la.qkv)
		q_g             := _act_grad(acts, la.q)
		k_g             := _act_grad(acts, la.k)
		v_g             := _act_grad(acts, la.v)
		q_rot_g         := _act_grad(acts, la.q_rot)
		k_rot_g         := _act_grad(acts, la.k_rot)
		qkv2_g          := _act_grad(acts, la.qkv2)
		attn_g          := _act_grad(acts, la.attn)
		attn_proj_g     := _act_grad(acts, la.attn_proj)
		residual_attn_g := _act_grad(acts, la.residual_attn)
		norm1_g         := _act_grad(acts, la.norm1)
		mlp_up_g        := _act_grad(acts, la.mlp_up)
		mlp_act_g       := _act_grad(acts, la.mlp_act)
		mlp_down_g      := _act_grad(acts, la.mlp_down)
		residual_mlp_g  := _act_grad(acts, la.residual_mlp)
		input_resid_g   := _act_grad(acts, la.input_residual)

		// MLP sub-block reverse.
		// residual_mlp = add(residual_attn, mlp_down)
		gpu.add_back(residual_attn_g, mlp_down_g, residual_mlp_g)
		// mlp_down = linear(mlp_act, mlp_down_weight)
		gpu.linear_back(la.mlp_act, layer.mlp_down_weight, mlp_down_g,
			mlp_act_g, layer.mlp_down_weight_grad,
			n, hidden_size, embed)
		// mlp_act = gelu(mlp_up)
		gpu.gelu_back(la.mlp_up, mlp_up_g, mlp_act_g)
		// mlp_up = linear(norm1, mlp_up_weight)
		gpu.linear_back(la.norm1, layer.mlp_up_weight, mlp_up_g,
			norm1_g, layer.mlp_up_weight_grad,
			n, embed, hidden_size)
		// norm1 = layernorm(residual_attn, norm1_weight)
		gpu.layernorm_back(la.residual_attn, layer.norm1_weight, norm1_g,
			residual_attn_g, layer.norm1_weight_grad,
			n, embed)

		// Attention sub-block reverse.
		// residual_attn = add(input_residual, attn_proj)
		gpu.add_back(input_resid_g, attn_proj_g, residual_attn_g)
		// attn_proj = linear(attn, proj_weight)
		gpu.linear_back(la.attn, layer.proj_weight, attn_proj_g,
			attn_g, layer.proj_weight_grad,
			n, embed, embed)
		// attn = attention(qkv2, ...)
		gpu.attention_back(la.qkv2, attn_g, qkv2_g, n, head_count, head_size)
		// qkv2 = concat3(q_rot, k_rot, v)
		gpu.concat3_back(q_rot_g, k_rot_g, v_g, qkv2_g, n, embed, embed, embed)
		// q_rot = rope(q); k_rot = rope(k)
		gpu.rope_back(q_g, q_rot_g, n, head_count, head_size)
		gpu.rope_back(k_g, k_rot_g, n, head_count, head_size)
		// q,k,v = slice_trailing(qkv, ...)
		gpu.slice_trailing_back(qkv_g, q_g, n, 3 * embed, 0,         embed)
		gpu.slice_trailing_back(qkv_g, k_g, n, 3 * embed, embed,     2 * embed)
		gpu.slice_trailing_back(qkv_g, v_g, n, 3 * embed, 2 * embed, 3 * embed)
		// qkv = linear(norm0, qkv_weight)
		gpu.linear_back(la.norm0, layer.qkv_weight, qkv_g,
			norm0_g, layer.qkv_weight_grad,
			n, embed, 3 * embed)
		// norm0 = layernorm(input_residual, norm0_weight)
		gpu.layernorm_back(la.input_residual, layer.norm0_weight, norm0_g,
			input_resid_g, layer.norm0_weight_grad,
			n, embed)
	}

	// output = select(token_embeddings, tokens). output_g has accumulated
	// from layer 0's input_residual contributions.
	gpu.select_back(tokens, output_g, model.token_embeddings_grad, vocab, embed)

	gpu.end_batch()

	// Reduce per-row losses to a scalar on the host. n is small (~64).
	losses := builtin.make([]f32, n, context.temp_allocator)
	gpu.download(loss_buf, losses)
	sum: f32 = 0
	for v in losses { sum += v }
	return sum / f32(n)
}

// --- Internal ---

_act :: proc(a: ^Activations, shape: ..int, loc := #caller_location) -> gpu.GpuTensor {
	if a.next >= len(a.data) {
		append(&a.data, gpu.alloc(..shape, loc=loc))
		append(&a.grad, gpu.alloc(..shape, loc=loc))
	}
	t := a.data[a.next]

	// Pool reuse only works if the same call site asks for the same shape on
	// every forward. Catch the misuse early — the alternative is silently
	// reading/writing past the buffer when shapes drift.
	count := 1
	for d in shape { count *= d }
	fmt.assertf(t.count == count,
		"_act: shape mismatch at index %v — pool has count %v, caller asked for %v",
		a.next, t.count, count, loc=loc)

	a.next += 1
	return t
}

// Look up the gradient buffer paired with a data tensor returned from
// _act. Linear search by buffer handle is fine — pool sizes are small
// (~50 entries) and backward visits each activation O(1) times.
_act_grad :: proc(a: ^Activations, t: gpu.GpuTensor, loc := #caller_location) -> gpu.GpuTensor {
	for d, i in a.data {
		if d.buffer == t.buffer {
			return a.grad[i]
		}
	}
	fmt.panicf("_act_grad: tensor not found in activation pool", loc=loc)
}
