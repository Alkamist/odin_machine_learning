// End-to-end gradient check: train a CPU transformer for a few steps, upload
// to a GPU mirror, run one forward+backward on each with identical tokens
// + targets, then compare every parameter gradient. Tight match (within
// ~1e-3 absolute) means the full backward pipeline is correct on the GPU.
//
// Build: odin build examples/gpu_grad_check -o:speed -no-bounds-check -microarch:native -out:examples/gpu_grad_check/gpu_grad_check.exe
package gpu_grad_check

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

ABS_TOL :: f32(2e-3)

main :: proc() {
	ml.init(256 * 1024 * 1024)
	ml.set_thread_count(1)

	gpu.init()
	defer gpu.destroy()

	rand.reset(SEED)
	cpu_model := tfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer tfm.destroy(cpu_model)

	tokens  := make([]int, SEQUENCE_LENGTH); defer delete(tokens)
	targets := make([]int, SEQUENCE_LENGTH); defer delete(targets)
	for i in 0 ..< SEQUENCE_LENGTH {
		tokens[i]  = i % VOCABULARY
		targets[i] = (i + 1) % VOCABULARY
	}

	TRAIN_STEPS :: 3
	fmt.printfln("Training CPU model for %v steps to non-trivial weights...", TRAIN_STEPS)
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

	// CPU forward + backward, capture gradients.
	ml.clear()
	cpu_logits := tfm.forward(cpu_model, tokens)
	cpu_ce     := ml.cross_entropy(cpu_logits, targets)
	cpu_loss   := ml.mean(cpu_ce)
	ml.backward()

	// GPU mirror.
	gpu_model := gtfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer gtfm.destroy(gpu_model)
	gtfm.upload(gpu_model, cpu_model)

	acts: gtfm.Activations
	defer gtfm.destroy_activations(&acts)

	gtfm.zero_grad(gpu_model)
	_ = gtfm.forward(gpu_model, tokens, &acts)
	gpu_loss := gtfm.backward(gpu_model, tokens, targets, &acts)

	fmt.println()
	fmt.printfln("loss: cpu=%.6f gpu=%.6f  delta=%.3e",
		cpu_loss.data[0], gpu_loss, math.abs(gpu_loss - cpu_loss.data[0]))

	any_failed := false
	any_failed |= !cmp("token_embeddings", cpu_model.token_embeddings.gradient, gpu_model.token_embeddings_grad)
	for i in 0 ..< LAYERS {
		any_failed |= !cmp_layer(i, cpu_model.layers[i], gpu_model.layers[i])
	}
	any_failed |= !cmp("norm_weight",   cpu_model.norm_weight.gradient,   gpu_model.norm_weight_grad)
	any_failed |= !cmp("output_weight", cpu_model.output_weight.gradient, gpu_model.output_weight_grad)

	if any_failed {
		fmt.println("FAIL")
		os.exit(1)
	}
	fmt.println("OK")
}

cmp_layer :: proc(li: int, c: tfm.Layer, g: gtfm.Layer) -> bool {
	ok := true
	ok &= cmp(fmt.tprintf("L%v.norm0_weight",    li), c.norm0_weight.gradient,    g.norm0_weight_grad)
	ok &= cmp(fmt.tprintf("L%v.qkv_weight",      li), c.qkv_weight.gradient,      g.qkv_weight_grad)
	ok &= cmp(fmt.tprintf("L%v.proj_weight",     li), c.proj_weight.gradient,     g.proj_weight_grad)
	ok &= cmp(fmt.tprintf("L%v.norm1_weight",    li), c.norm1_weight.gradient,    g.norm1_weight_grad)
	ok &= cmp(fmt.tprintf("L%v.mlp_up_weight",   li), c.mlp_up_weight.gradient,   g.mlp_up_weight_grad)
	ok &= cmp(fmt.tprintf("L%v.mlp_down_weight", li), c.mlp_down_weight.gradient, g.mlp_down_weight_grad)
	return ok
}

cmp :: proc(name: string, cpu_grad: []f32, gpu_t: gpu.GpuTensor) -> bool {
	got := make([]f32, gpu_t.count); defer delete(got)
	gpu.download(gpu_t, got)

	max_abs: f32
	max_idx: int
	for i in 0 ..< len(got) {
		d := math.abs(got[i] - cpu_grad[i])
		if d > max_abs { max_abs = d; max_idx = i }
	}
	if max_abs > ABS_TOL {
		fmt.printfln("  %-30s FAIL  max_abs=%.3e at %v (cpu=%.6f gpu=%.6f)",
			name, max_abs, max_idx, cpu_grad[max_idx], got[max_idx])
		return false
	}
	fmt.printfln("  %-30s OK    max_abs=%.3e", name, max_abs)
	return true
}
