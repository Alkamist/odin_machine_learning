package smollm_smoke

import "core:fmt"
import "core:time"

import ml    "../.."
import cpu   "../../backends/cpu"
import llama "../../networks/llama"

main :: proc() {
	cpu.set_thread_count(8)
	ctx := cpu.context_create(1024 * 1024 * 64)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	fmt.println("Allocating full SmolLM2-135M (30 layers, ~135M params).")

	cfg := llama.SMOLLM2_135M_CONFIG

	model := llama.make(cfg)
	defer llama.destroy(model)

	// Single forward pass on a tiny prompt to confirm every op accepts the
	// SmolLM2 dimensions (head_size=64 is at the GPU shaders' upper limit;
	// CPU is unconstrained).
	tokens := []int{0, 1, 2, 3, 4, 5, 6, 7}

	t0 := time.tick_now()
	logits := llama.forward(model, tokens)
	logits_buf := make([]f32, ml.len(logits))
	defer delete(logits_buf)
	ml.get_data(logits, logits_buf)
	dt := time.tick_since(t0)

	fmt.printfln(
		"Forward OK in %.1f ms. logits shape=[%v, %v], first 4 = %v",
		f64(time.duration_milliseconds(dt)),
		logits.shape[0], logits.shape[1],
		logits_buf[:4],
	)
}
