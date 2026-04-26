// Full training-step benchmark: forward + backward + Adam update on
// CPU (1, 4, 24 threads) vs GPU. Same shape as the canonical benches.
//
// Build: odin build examples/gpu_train_bench -o:speed -no-bounds-check -microarch:native -out:examples/gpu_train_bench/gpu_train_bench.exe
package gpu_train_bench

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

	tokens  := make([]int, SEQUENCE); defer delete(tokens)
	targets := make([]int, SEQUENCE); defer delete(targets)
	for i in 0 ..< SEQUENCE {
		tokens[i]  = i % VOCABULARY
		targets[i] = (i + 1) % VOCABULARY
	}

	fmt.printfln("config: L=%v H=%v E=%v V=%v T=%v  (warmup=%v iters=%v)",
		LAYERS, HEADS, EMBEDDING_SIZE, VOCABULARY, SEQUENCE, WARMUP, ITERATIONS)
	fmt.println()

	bench_cpu(cpu_model, tokens, targets, 1)
	bench_cpu(cpu_model, tokens, targets, 4)
	bench_cpu(cpu_model, tokens, targets, 24)
	bench_gpu(gpu_model, tokens, targets)
}

bench_cpu :: proc(model: tfm.Transformer, tokens, targets: []int, threads: int) {
	ml.set_thread_count(threads)

	for _ in 0 ..< WARMUP {
		ml.clear()
		logits := tfm.forward(model, tokens)
		ce     := ml.cross_entropy(logits, targets)
		_       = ml.mean(ce)
		ml.backward()
		opt: ml.Optimizer
		if ml.optimize(&opt, period = 1) {
			tfm.update(opt, model)
		}
	}

	min_ms, total_ms: f64 = 1e18, 0
	for _ in 0 ..< ITERATIONS {
		ml.clear()
		t0 := time.tick_now()
		logits := tfm.forward(model, tokens)
		ce     := ml.cross_entropy(logits, targets)
		_       = ml.mean(ce)
		ml.backward()
		opt: ml.Optimizer
		if ml.optimize(&opt, period = 1) {
			tfm.update(opt, model)
		}
		dt := f64(time.tick_since(t0)) / 1_000_000.0
		if dt < min_ms { min_ms = dt }
		total_ms += dt
	}
	mean_ms := total_ms / f64(ITERATIONS)
	fmt.printfln("cpu  (threads=%2v):  min=%7.3f ms   mean=%7.3f ms", threads, min_ms, mean_ms)
}

bench_gpu :: proc(model: gtfm.Transformer, tokens, targets: []int) {
	acts: gtfm.Activations
	defer gtfm.destroy_activations(&acts)
	opt: gtfm.Optimizer

	for _ in 0 ..< WARMUP {
		_ = gtfm.forward(model, tokens, &acts)
		_ = gtfm.backward(model, tokens, targets, &acts)
		gtfm.update(&opt, model)
	}

	min_ms, total_ms: f64 = 1e18, 0
	for _ in 0 ..< ITERATIONS {
		t0 := time.tick_now()
		_ = gtfm.forward(model, tokens, &acts)
		_ = gtfm.backward(model, tokens, targets, &acts)
		gtfm.update(&opt, model)
		dt := f64(time.tick_since(t0)) / 1_000_000.0
		if dt < min_ms { min_ms = dt }
		total_ms += dt
	}
	mean_ms := total_ms / f64(ITERATIONS)
	fmt.printfln("gpu  (3090 Ti):     min=%7.3f ms   mean=%7.3f ms", min_ms, mean_ms)
}
