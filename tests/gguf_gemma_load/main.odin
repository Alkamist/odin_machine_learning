package gguf_gemma_load

import "core:fmt"
import "core:mem"
import "core:os"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import gemma "../../networks/gemma"

// Smoke test for the GGUF Gemma loader. Builds a Gemma 4 E4B model on the
// CPU backend, loads the Ollama Q4_K_M GGUF, and asserts a handful of
// invariants. Doesn't run forward — that's the next step.

main :: proc() {
	if len(os.args) < 2 {
		fmt.eprintfln("usage: %v <gguf_path>", os.args[0])
		os.exit(2)
	}
	path := os.args[1]

	any_failed := false
	check :: proc(cond: bool, msg: string, any_failed: ^bool) {
		if cond {
			fmt.printfln("OK   %v", msg)
		} else {
			fmt.printfln("FAIL %v", msg)
			any_failed^ = true
		}
	}

	ctx := cpu.context_create(64 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	cfg := gemma.make_e4b_config()
	defer gemma.config_destroy(cfg)

	t_make := time.tick_now()
	model  := gemma.make(cfg, .Bf16)
	defer gemma.destroy(model)
	fmt.printfln("make: %.1f s", time.duration_seconds(time.tick_since(t_make)))

	t_load := time.tick_now()
	ok := gemma.load_gguf(&model, path)
	check(ok, "gemma.load_gguf returned ok", &any_failed)
	fmt.printfln("load: %.1f s", time.duration_seconds(time.tick_since(t_load)))

	if !ok {
		os.exit(1)
	}

	// Spot-check tensor types after the load.
	check(model.embed_tokens_weight.type == .Bf16,
		"embed_tokens_weight is Bf16 (dequantized from Q6_K)", &any_failed)
	check(model.output_norm_weight.type == .Bf16,
		"output_norm_weight is Bf16", &any_failed)
	check(model.per_layer_model_projection_weight.type == .Q4_K,
		"per_layer_model_projection_weight is Q4_K", &any_failed)

	// Spot-check a few layers.
	for layer_idx in ([?]int{0, 5, 21, 41}) {
		layer := model.layers[layer_idx]
		// Q4_K_M mixes Q4_K and Q6_K per tensor and per layer; just check that
		// every projection is *one of* the two k-quant types.
		_is_kquant :: proc(t: ml.Data_Type) -> bool { return t == .Q4_K || t == .Q6_K }

		check(_is_kquant(layer.q_proj_weight.type),
			fmt.tprintf("layer %v q_proj is k-quant (got %v)", layer_idx, layer.q_proj_weight.type), &any_failed)
		check(_is_kquant(layer.o_proj_weight.type),
			fmt.tprintf("layer %v o_proj is k-quant (got %v)", layer_idx, layer.o_proj_weight.type), &any_failed)
		check(_is_kquant(layer.gate_proj_weight.type),
			fmt.tprintf("layer %v gate_proj is k-quant (got %v)", layer_idx, layer.gate_proj_weight.type), &any_failed)
		check(_is_kquant(layer.up_proj_weight.type),
			fmt.tprintf("layer %v up_proj is k-quant (got %v)", layer_idx, layer.up_proj_weight.type), &any_failed)
		check(_is_kquant(layer.down_proj_weight.type),
			fmt.tprintf("layer %v down_proj is k-quant (got %v)", layer_idx, layer.down_proj_weight.type), &any_failed)

		if !gemma.is_kv_shared_layer(cfg, layer_idx) {
			check(_is_kquant(layer.k_proj_weight.type),
				fmt.tprintf("layer %v k_proj is k-quant (got %v)", layer_idx, layer.k_proj_weight.type), &any_failed)
			check(_is_kquant(layer.v_proj_weight.type),
				fmt.tprintf("layer %v v_proj is k-quant (got %v)", layer_idx, layer.v_proj_weight.type), &any_failed)
		}

		// Norms should be Bf16 (converted from F32 at load).
		check(layer.input_norm_weight.type == .Bf16,
			fmt.tprintf("layer %v input_norm is Bf16", layer_idx), &any_failed)

		// q_norm should have sqrt(head_dim) baked in: stored values ~ 1.0 in
		// GGUF, so loaded values should be in the [3, 40] range (sliding
		// head_dim=256 → sqrt=16; full head_dim=512 → sqrt≈22; allow some
		// slack since trained values aren't exactly 1).
		buf: [1]ml.Bf16
		ml.get_data_bytes(layer.q_norm_weight, mem.slice_to_bytes(buf[:]))
		v := ml.bf16_to_f32(buf[0])
		check(v > 3.0 && v < 40.0,
			fmt.tprintf("layer %v q_norm[0] in [3, 40]: got %v", layer_idx, v),
			&any_failed)
	}

	if any_failed do os.exit(1)
	fmt.println("ok")
}
