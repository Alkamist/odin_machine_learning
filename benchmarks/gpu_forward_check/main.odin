// End-to-end check: train a CPU transformer for a few steps, upload its
// weights to a GPU mirror, run forward on both with the same tokens, and
// compare logits. Tight match (within ~1e-4 absolute, ~1e-3 relative) means
// the full forward pipeline is correct on the GPU.
//
// Build: odin build examples/gpu_forward_check -o:speed -no-bounds-check -microarch:native -out:examples/gpu_forward_check/gpu_forward_check.exe
package gpu_forward_check

import "core:fmt"
import "core:math"
import "core:math/rand"
import ml  "../.."
import tfm "../../transformer"
import gpu "../../gpu"
import gtfm "../../gpu_transformer"

// Match benchmark/main.odin so checksums are comparable.
LAYERS          :: 4
HEADS           :: 4
EMBEDDING_SIZE  :: 128
VOCABULARY      :: 256
SEQUENCE_LENGTH :: 64
SEED            :: 0xC0FFEE

main :: proc() {
	ml.init(256 * 1024 * 1024)
	ml.set_thread_count(1)

	gpu.init()
	defer gpu.destroy()

	// --- CPU model ---
	rand.reset(SEED)
	cpu_model := tfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer tfm.destroy(cpu_model)

	tokens := make([]int, SEQUENCE_LENGTH); defer delete(tokens)
	targets := make([]int, SEQUENCE_LENGTH); defer delete(targets)
	for i in 0 ..< SEQUENCE_LENGTH {
		tokens[i]  = i % VOCABULARY
		targets[i] = (i + 1) % VOCABULARY
	}

	// Train a few steps so the weights are non-trivial. Avoids the case
	// where freshly-initialized layernorm gain=1, weights~N(0, 0.02) might
	// be too easy and hide bugs that only show after training has shifted
	// distributions.
	TRAIN_STEPS :: 5
	fmt.printfln("Training CPU model for %v steps...", TRAIN_STEPS)
	for step in 0 ..< TRAIN_STEPS {
		ml.clear()
		logits := tfm.forward(cpu_model, tokens)
		ce     := ml.cross_entropy(logits, targets)
		loss   := ml.mean(ce)
		ml.backward()

		opt: ml.Optimizer
		if ml.optimize(&opt, period = 1) {
			tfm.update(opt, cpu_model)
		}
		fmt.printfln("  step %v: loss = %.6f", step, loss.data[0])
	}

	// --- GPU mirror ---
	gpu_model := gtfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer gtfm.destroy(gpu_model)

	gtfm.upload(gpu_model, cpu_model)

	// --- Compare forward outputs ---
	ml.clear()
	cpu_logits := tfm.forward(cpu_model, tokens)

	acts: gtfm.Activations
	defer gtfm.destroy_activations(&acts)
	gpu_logits := gtfm.forward(gpu_model, tokens, &acts)

	out := make([]f32, SEQUENCE_LENGTH * VOCABULARY); defer delete(out)
	gpu.download(gpu_logits, out)

	max_abs: f32
	max_rel: f32
	max_abs_idx: int
	for i in 0 ..< SEQUENCE_LENGTH * VOCABULARY {
		d := math.abs(out[i] - cpu_logits.data[i])
		if d > max_abs {
			max_abs = d
			max_abs_idx = i
		}
		denom := math.max(math.abs(cpu_logits.data[i]), 1e-3)
		r := d / denom
		if r > max_rel { max_rel = r }
	}

	cpu_sum: f32
	gpu_sum: f32
	for i in 0 ..< SEQUENCE_LENGTH * VOCABULARY {
		cpu_sum += cpu_logits.data[i]
		gpu_sum += out[i]
	}

	fmt.println()
	fmt.printfln("logits comparison (%v elements):", SEQUENCE_LENGTH * VOCABULARY)
	fmt.printfln("  cpu sum: %.6f", cpu_sum)
	fmt.printfln("  gpu sum: %.6f", gpu_sum)
	fmt.printfln("  max_abs: %.3e (at index %v: cpu=%.6f gpu=%.6f)",
		max_abs, max_abs_idx, cpu_logits.data[max_abs_idx], out[max_abs_idx])
	fmt.printfln("  max_rel: %.3e", max_rel)
}
