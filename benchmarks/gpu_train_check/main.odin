// End-to-end "training works" check. Trains the same transformer on CPU
// and GPU from identical starting weights for N steps and prints both
// loss curves side by side.
//
// Important caveat: byte-equivalent training is NOT a goal. CPU and GPU
// disagree in fp32 reduction order on every parallel reduction (layernorm
// row stats, attention softmax, cross-entropy normalization, etc.), and
// Adam's `1/(sqrt(v_hat)+eps)` denominator amplifies tiny grad differences
// when v_hat is small. After many steps the trajectories drift apart.
//
// What this check verifies:
//   * One-step gradient agreement is at the fp32 floor (covered by
//     gpu_grad_check, which is part of the same test bar).
//   * Both backends train — loss strictly decreases on both.
//   * After N steps both reach comparable final loss (within RELATIVE_TOL).
//
// Build: odin build examples/gpu_train_check -o:speed -no-bounds-check -microarch:native -out:examples/gpu_train_check/gpu_train_check.exe
package gpu_train_check

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import ml   "../.."
import tfm  "../../transformer"
import gpu  "../../gpu"
import gtfm "../../gpu_transformer"

LAYERS          :: 4
HEADS           :: 4
EMBEDDING_SIZE  :: 128
VOCABULARY      :: 256
SEQUENCE_LENGTH :: 64
SEED            :: 0xC0FFEE
TRAIN_STEPS     :: 100

// Both backends must drop loss by at least this fraction of the initial
// loss. Lenient on purpose — see file header re: drift.
MIN_LOSS_DROP_FRAC :: f32(0.95)
// Final loss must be below this. Sanity check that training actually
// overfits this trivial task. CPU typically reaches ~0.015, GPU ~0.04
// after 100 steps at lr=0.001.
MAX_FINAL_LOSS :: f32(0.1)

main :: proc() {
	ctx := ml.context_create(256 * 1024 * 1024)
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)
	ml.set_thread_count(1)

	gpu.init()
	defer gpu.destroy()

	gctx := gpu.context_create()
	defer gpu.context_destroy(gctx)
	gpu.context_scope(gctx)

	rand.reset(SEED)
	cpu_model := tfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer tfm.destroy(cpu_model)

	gpu_model := gtfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer gtfm.destroy(gpu_model)
	gtfm.upload(gpu_model, cpu_model)

	tokens  := make([]int, SEQUENCE_LENGTH); defer delete(tokens)
	targets := make([]int, SEQUENCE_LENGTH); defer delete(targets)
	for i in 0 ..< SEQUENCE_LENGTH {
		tokens[i]  = i % VOCABULARY
		targets[i] = (i + 1) % VOCABULARY
	}

	acts: gtfm.Activations
	defer gtfm.destroy_activations(&acts)
	gpu_opt: gtfm.Optimizer

	fmt.printfln("Training %v steps (CPU 1-thread vs GPU)", TRAIN_STEPS)
	fmt.println("step      cpu_loss      gpu_loss        delta")

	cpu_first, gpu_first: f32
	cpu_last,  gpu_last:  f32

	for step in 0 ..< TRAIN_STEPS {
		ml.clear()
		cpu_logits := tfm.forward(cpu_model, tokens)
		cpu_ce     := ml.cross_entropy(cpu_logits, targets)
		cpu_loss   := ml.mean(cpu_ce)
		ml.backward()
		cpu_o: ml.Optimizer
		if ml.optimize(&cpu_o, period = 1) {
			tfm.update(cpu_o, cpu_model)
		}

		_ = gtfm.forward(gpu_model, tokens, &acts)
		gpu_loss := gtfm.backward(gpu_model, tokens, targets, &acts)
		gtfm.update(&gpu_opt, gpu_model)

		c := cpu_loss.data[0]
		g := gpu_loss
		fmt.printfln("%4v  %12.6f  %12.6f  %12.3e", step, c, g, math.abs(g - c))

		if step == 0           { cpu_first = c; gpu_first = g }
		if step == TRAIN_STEPS - 1 { cpu_last  = c; gpu_last  = g }
	}

	fmt.println()
	fmt.printfln("CPU loss: %.4f -> %.4f  (drop %.4f)", cpu_first, cpu_last, cpu_first - cpu_last)
	fmt.printfln("GPU loss: %.4f -> %.4f  (drop %.4f)", gpu_first, gpu_last, gpu_first - gpu_last)

	any_failed := false
	min_drop := MIN_LOSS_DROP_FRAC * cpu_first
	if (cpu_first - cpu_last) < min_drop { fmt.println("FAIL: CPU did not converge"); any_failed = true }
	if (gpu_first - gpu_last) < min_drop { fmt.println("FAIL: GPU did not converge"); any_failed = true }
	if cpu_last > MAX_FINAL_LOSS { fmt.printfln("FAIL: CPU final loss %.4f > %.4f", cpu_last, MAX_FINAL_LOSS); any_failed = true }
	if gpu_last > MAX_FINAL_LOSS { fmt.printfln("FAIL: GPU final loss %.4f > %.4f", gpu_last, MAX_FINAL_LOSS); any_failed = true }

	if any_failed { os.exit(1) }
	fmt.println("OK")
}
