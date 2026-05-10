package gguf_gemma_forward

import "core:fmt"
import "core:math"
import "core:os"
import "core:slice"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import gemma "../../networks/gemma"

// Minimal end-to-end smoke for the GGUF loader + Q4_K/Q6_K linear ops.
// Loads the Gemma 4 E4B Q4_K_M model on the CPU backend, runs a single
// forward_cached call on the BOS token, and reports:
//   - whether logits are finite + bounded
//   - the top-5 next-token IDs (compare manually against llama.cpp's
//     output for the same prompt to catch gross divergences)

main :: proc() {
	if len(os.args) < 2 {
		fmt.eprintfln("usage: %v <gguf_path>", os.args[0])
		os.exit(2)
	}
	path := os.args[1]

	ctx := cpu.context_create(256 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)
	ml.clear({.No_Gradients})

	cfg := gemma.make_e4b_config()
	defer gemma.config_destroy(cfg)

	t_make := time.tick_now()
	model  := gemma.make(cfg, .Bf16)
	defer gemma.destroy(model)
	fmt.printfln("make: %.1f s", time.duration_seconds(time.tick_since(t_make)))

	t_load := time.tick_now()
	if !gemma.load_gguf(&model, path) {
		os.exit(1)
	}
	fmt.printfln("load: %.1f s", time.duration_seconds(time.tick_since(t_load)))

	cache := gemma.cache_make(model, 64)
	defer gemma.cache_destroy(cache)

	// BOS token id from the GGUF KV: tokenizer.ggml.bos_token_id = 2.
	bos := 2

	t_fwd := time.tick_now()
	logits := gemma.forward_cached(model, &cache, []int{bos})
	fmt.printfln("forward(1 token): %.1f s", time.duration_seconds(time.tick_since(t_fwd)))

	logit_count := ml.len(logits)
	floats      := make([]f32, logit_count)
	defer delete(floats)
	if logits.type == .F32 {
		ml.get_data(logits, floats)
	} else if logits.type == .Bf16 {
		bf := make([]ml.Bf16, logit_count)
		defer delete(bf)
		ml.get_data_bytes(logits, slice.bytes_from_ptr(raw_data(bf), logit_count * 2))
		for v, i in bf {
			floats[i] = ml.bf16_to_f32(v)
		}
	} else {
		fmt.eprintfln("unexpected logits dtype %v", logits.type)
		os.exit(1)
	}

	// Final-position logits (last vocab_size floats).
	vocab := cfg.vocab_size
	final := floats[len(floats) - vocab:]

	nan_count, inf_count: int
	max_abs: f32
	for v in final {
		if math.is_nan(v) { nan_count += 1 } else if math.is_inf(v) { inf_count += 1 } else {
			a := v
			if a < 0 {
				a = -a
			}
			if a > max_abs {
				max_abs = a
			}
		}
	}
	fmt.printfln("logits: vocab=%v finite_max_abs=%.4f nan=%v inf=%v", vocab, max_abs, nan_count, inf_count)
	if nan_count != 0 || inf_count != 0 {
		fmt.eprintfln("FAIL: non-finite logits")
		os.exit(1)
	}
	// Softcap caps logits at ~30 in our config, so anything beyond that is suspect.
	if max_abs > 100 {
		fmt.eprintfln("FAIL: logit magnitude too large (%v)", max_abs)
		os.exit(1)
	}

	// Top-5.
	Pair :: struct { id: int, score: f32 }
	pairs := make([]Pair, vocab)
	defer delete(pairs)
	for v, i in final {
		pairs[i] = {id = i, score = v}
	}
	slice.sort_by(pairs, proc(a, b: Pair) -> bool { return a.score > b.score })

	fmt.println("top-5 next-token ids after BOS:")
	for i in 0 ..< 5 {
		fmt.printfln("  rank %v: id=%v score=%.3f", i, pairs[i].id, pairs[i].score)
	}
	fmt.println("ok")
}
