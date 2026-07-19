package lora

import "core:mem"

import ml "../../"

// Low-Rank Adaptation: y = base + scale * B @ (A @ x), where A is small
// and B is small (rank << in_features and rank << out_features).
//
// Forward cost: two skinny matmuls instead of one big one. Backward
// gradient flows only into A and B - the base weight stays frozen, which
// is what unlocks QLoRA (base in 4-bit, adapters in bf16).
Adapter :: struct {
	a:     ml.Tensor, // [rank, in_features]
	b:     ml.Tensor, // [out_features, rank]
	scale: ml.Tensor, // scalar, alpha / rank, baked once at make time

	rank:         int,
	in_features:  int,
	out_features: int,

	params: ml.Registry,
}

@(require_results)
make :: proc(in_features, out_features, rank: int, alpha: f32, dtype: ml.Data_Type = .Bf16, loc := #caller_location) -> (adapter: Adapter) {
	adapter.rank         = rank
	adapter.in_features  = in_features
	adapter.out_features = out_features

	adapter.a = ml.parameter_make(&adapter.params, "", "lora_A.weight", dtype, {rank, in_features}, init=ml.Init_Normal{mean=0, std=0.02})
	adapter.b = ml.parameter_make(&adapter.params, "", "lora_B.weight", dtype, {out_features, rank}, init=ml.Init_Value{value=0})

	adapter.scale = ml.scalar(dtype, alpha / f32(rank), persistent=true, loc=loc)

	return
}

destroy :: proc(adapter: Adapter) {
	adapter := adapter
	ml.registry_destroy(&adapter.params)
	ml.destroy(adapter.scale)
}

// Standard QLoRA init: A ~ N(0, sigma) so the input through A is non-zero,
// B = 0 so the adapter contribution starts at zero. The model behaves
// identically to the frozen base at step 0; LoRA learns from there.
randomize :: proc(adapter: Adapter, sigma: f32 = 0.02) {
	ml.fill_normal(adapter.a, 0, sigma)
	ml.fill_value (adapter.b, 0)
}

// Augment a base linear output with the adapter contribution.
// `base_output` is whatever the frozen base linear produced; this returns
// `base_output + scale * B @ (A @ input)`.
@(require_results)
apply :: proc(input, base_output: ml.Tensor, adapter: Adapter) -> ml.Tensor {
	a_out  := ml.linear(input, adapter.a)  // [tokens, rank]
	b_out  := ml.linear(a_out, adapter.b)  // [tokens, out_features]
	scaled := ml.mul(b_out, adapter.scale)
	return ml.add(base_output, scaled)
}

update :: proc(opt: ^ml.Optimizer, adapter: Adapter) {
	adapter := adapter
	ml.registry_update(opt, &adapter.params)
}

// Element count for parameter accounting / progress reporting.
@(require_results)
parameter_count :: proc(adapter: Adapter) -> int {
	adapter := adapter
	return ml.registry_element_count(&adapter.params)
}

// Enumerates the trainable adapter tensors under PEFT-style names
// (`{prefix}.lora_A.weight`, `{prefix}.lora_B.weight`) so saved adapters
// interoperate with the wider ecosystem. `scale` is a frozen constant
// recomputed from alpha/rank at make time, so it is not a saved parameter.
parameters :: proc(adapter: Adapter, prefix: string, dst: ^ml.Registry) {
	adapter := adapter
	ml.registry_gather(dst, &adapter.params, prefix=prefix)
}
