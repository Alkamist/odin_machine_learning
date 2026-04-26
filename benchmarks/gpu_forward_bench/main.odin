// End-to-end transformer forward bench: CPU `tfm.forward` vs GPU
// `gtfm.forward`, same weights, same tokens, same shape as the canonical
// benchmark (4L, 4H, 128e, 256 vocab, 64-token sequence).
//
// IMPORTANT: each GPU forward currently allocates ~50 fresh activation
// tensors (one per intermediate op result). That bakes in alloc + free
// Vulkan API overhead on top of the per-dispatch submit overhead. Both
// will improve with: (a) command-buffer batching, (b) an activation pool.
// The number this bench reports is the floor with neither in place.
//
// Build: odin build examples/gpu_forward_bench -o:speed -no-bounds-check -microarch:native -out:examples/gpu_forward_bench/gpu_forward_bench.exe
package gpu_forward_bench

import "core:fmt"
import "core:time"
import ml   "../.."
import tfm  "../../transformer"
import gpu  "../../gpu"
import gtfm "../../gpu_transformer"

LAYERS         :: 4
HEADS          :: 4
EMBEDDING_SIZE :: 128
VOCABULARY     :: 256
SEQUENCE       :: 64

WARMUP     :: 5
ITERATIONS :: 50

main :: proc() {
	ml.init(256 * 1024 * 1024)
	gpu.init()
	defer gpu.destroy()

	cpu_model := tfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer tfm.destroy(cpu_model)

	gpu_model := gtfm.make(LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY)
	defer gtfm.destroy(gpu_model)
	gtfm.upload(gpu_model, cpu_model)

	tokens := make([]int, SEQUENCE); defer delete(tokens)
	for i in 0 ..< SEQUENCE { tokens[i] = i % VOCABULARY }

	fmt.printfln("config: L=%v H=%v E=%v V=%v T=%v  (warmup=%v iters=%v)",
		LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY, SEQUENCE, WARMUP, ITERATIONS)
	fmt.println()

	bench_cpu(cpu_model, tokens, 1)
	bench_cpu(cpu_model, tokens, 4)
	bench_cpu(cpu_model, tokens, 24)
	bench_gpu(gpu_model, tokens)
}

bench_cpu :: proc(model: tfm.Transformer, tokens: []int, threads: int) {
	ml.set_thread_count(threads)

	for _ in 0 ..< WARMUP {
		ml.clear()
		_ = tfm.forward(model, tokens)
	}

	min_ms, total_ms: f64 = 1e18, 0
	for _ in 0 ..< ITERATIONS {
		ml.clear()
		t0 := time.tick_now()
		_ = tfm.forward(model, tokens)
		dt := f64(time.tick_since(t0)) / 1_000_000.0
		if dt < min_ms { min_ms = dt }
		total_ms += dt
	}
	mean_ms := total_ms / f64(ITERATIONS)
	fmt.printfln("cpu  (threads=%2v):  min=%7.3f ms   mean=%7.3f ms", threads, min_ms, mean_ms)
}

bench_gpu :: proc(model: gtfm.Transformer, tokens: []int) {
	acts: gtfm.Activations
	defer gtfm.destroy_activations(&acts)

	for _ in 0 ..< WARMUP {
		_ = gtfm.forward(model, tokens, &acts)
	}

	min_ms, total_ms: f64 = 1e18, 0
	for _ in 0 ..< ITERATIONS {
		t0 := time.tick_now()
		_ = gtfm.forward(model, tokens, &acts)
		dt := f64(time.tick_since(t0)) / 1_000_000.0
		if dt < min_ms { min_ms = dt }
		total_ms += dt
	}
	mean_ms := total_ms / f64(ITERATIONS)
	fmt.printfln("gpu  (3090 Ti):     min=%7.3f ms   mean=%7.3f ms", min_ms, mean_ms)
}
