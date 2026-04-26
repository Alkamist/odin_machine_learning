// Benchmarks for the machine_learning library.
//
// Each benchmark warms up, then runs ITERATIONS timed iterations and reports
// min/mean time in milliseconds plus a numerical checksum. Compare runs
// before/after a change to confirm the change is faster AND that the checksum
// is unchanged (so correctness wasn't broken).
//
// Build: odin build benchmark -o:speed -no-bounds-check
// Run:   benchmark.exe

package machine_learning_benchmark

import "core:fmt"
import "core:time"
import "core:os"
import "core:math"
import "core:math/rand"
import ml "../../"
import tfm "../../transformer"

// Fixed seed so checksums are reproducible across runs (and across versions).
SEED :: 0xC0FFEE

// Edit these to change how the benchmark behaves.
THREAD_COUNT :: 0 // 0 means os.get_processor_core_count()
WARMUP       :: 2
ITERATIONS   :: 20

// Transformer end-to-end config (mirrors examples/text_generation_transformer scale).
E2E_LAYERS          :: 4
E2E_HEADS           :: 4
E2E_EMBEDDING_SIZE  :: 128
E2E_VOCABULARY      :: 256
E2E_SEQUENCE_LENGTH :: 64

Result :: struct {
	name:     string,
	min_ms:   f64,
	mean_ms:  f64,
	checksum: f32,
}

main :: proc() {
	thread_count := THREAD_COUNT
	if thread_count <= 0 {
		thread_count = os.get_processor_core_count()
	}

	// 256 MB arena. Plenty for the configs below.
	ctx := ml.context_create(256 * 1024 * 1024)
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)

	fmt.printfln("threads=%v warmup=%v iterations=%v", thread_count, WARMUP, ITERATIONS)
	fmt.println("================================================================")

	// Run once before timing: verifies multi-step training trajectory from a
	// fresh seeded model. Per-iteration losses + final param checksum should
	// match across versions (within ~1 ULP on ST). Catches backward/update
	// bugs that the single-step forward checksums in the timed benches miss.
	ml.set_thread_count(1)
	verify_training_trajectory()
	fmt.println()

	// Sweep thread counts so threading overhead vs SIMD work-per-thread is
	// visible. Single-threaded numbers are the algorithmic ground truth;
	// higher counts show parallel scaling.
	run_all(1)
	if thread_count >= 4 {
		fmt.println()
		run_all(4)
	}
	if thread_count >= 8 {
		fmt.println()
		run_all(8)
	}
	if thread_count > 8 {
		fmt.println()
		run_all(thread_count)
	}
}

run_all :: proc(thread_count: int) {
	ml.set_thread_count(thread_count)
	fmt.printfln("--- thread_count = %v ---", thread_count)
	fmt.printfln("%-44s %10s %10s %16s", "benchmark", "min ms", "mean ms", "checksum")

	report(bench_linear_inference_fwd())
	report(bench_linear_inference())
	report(bench_linear_training_fwd())
	report(bench_linear_training())
	report(bench_attention())
	report(bench_layernorm())
	report(bench_softmax())
	report(bench_gelu())
	report(bench_adam_update())
	report(bench_transformer_step())
}

report :: proc(r: Result) {
	min_s  := fmt.tprintf("%.3f", r.min_ms)
	mean_s := fmt.tprintf("%.3f", r.mean_ms)
	csum_s := fmt.tprintf("%.6f", r.checksum)
	fmt.printfln("%-44s %10s %10s %16s", r.name, min_s, mean_s, csum_s)
}

// Time the given proc over WARMUP+ITERATIONS calls. The proc must do its own
// ml.clear() at the top so each iteration starts from a fresh arena/op buffer.
time_iters :: proc(name: string, run: proc() -> f32) -> Result {
	for _ in 0 ..< WARMUP {
		_ = run()
	}

	min_ns:  i64 = max(i64)
	total_ns: i64 = 0
	checksum: f32

	for i in 0 ..< ITERATIONS {
		start := time.tick_now()
		c := run()
		dt := i64(time.tick_since(start))
		if dt < min_ns {
			min_ns = dt
		}
		total_ns += dt
		if i == 0 {
			checksum = c
		}
	}

	return Result{
		name     = name,
		min_ms   = f64(min_ns)         / 1_000_000.0,
		mean_ms  = f64(total_ns)       / 1_000_000.0 / f64(ITERATIONS),
		checksum = checksum,
	}
}

sum :: proc(t: ml.Tensor) -> f32 {
	s: f32
	for v in ml.data(t) {
		s += v
	}
	return s
}

sum_grad :: proc(t: ml.Tensor) -> f32 {
	s: f32
	for v in ml.gradient(t) {
		s += v
	}
	return s
}

// --- Micro-benchmarks ---

// Small-batch linear: count=1 means parallelize-over-count gives no parallelism.
// This is the workload that exposes the "parallelize over output rows instead"
// optimization most clearly.
bench_linear_inference_fwd :: proc() -> Result {
	INPUT  :: 512
	OUTPUT :: 2048

	rand.reset(SEED)
	w := ml.make(OUTPUT, INPUT)
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(INPUT)
		ml.fill_value(x, 0.01)
		y := ml.linear(x, w_state)
		return sum(y)
	}

	state_w = w
	return time_iters("linear forward only        (count=1, 512x2048)", run)
}

bench_linear_inference :: proc() -> Result {
	INPUT  :: 512
	OUTPUT :: 2048

	rand.reset(SEED)
	w := ml.make(OUTPUT, INPUT)
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(INPUT)
		ml.fill_value(x, 0.01)
		y := ml.linear(x, w_state)
		ml.backward()
		return sum(y) + sum_grad(x) + sum_grad(w_state)
	}

	state_w = w
	return time_iters("linear forward+backward    (count=1, 512x2048)", run)
}

// Larger-batch linear, training-like: count is the token dimension.
bench_linear_training_fwd :: proc() -> Result {
	COUNT  :: 64
	INPUT  :: 128
	OUTPUT :: 512

	rand.reset(SEED)
	w := ml.make(OUTPUT, INPUT)
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(COUNT, INPUT)
		ml.fill_value(x, 0.01)
		y := ml.linear(x, w_state)
		return sum(y)
	}

	state_w = w
	return time_iters("linear forward only        (count=64, 128x512)", run)
}

bench_linear_training :: proc() -> Result {
	COUNT  :: 64
	INPUT  :: 128
	OUTPUT :: 512

	rand.reset(SEED)
	w := ml.make(OUTPUT, INPUT)
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(COUNT, INPUT)
		ml.fill_value(x, 0.01)
		y := ml.linear(x, w_state)
		ml.backward()
		return sum(y) + sum_grad(x) + sum_grad(w_state)
	}

	state_w = w
	return time_iters("linear forward+backward    (count=64, 128x512)", run)
}

bench_attention :: proc() -> Result {
	TOKENS :: 64
	HEADS  :: 4
	EMBED  :: 128

	run :: proc() -> f32 {
		ml.clear()
		// Input shape for attention is stacked QKV [tokens, 3 * embed].
		qkv := ml.zeros(TOKENS, 3 * EMBED)
		ml.fill_value(qkv, 0.01)
		y := ml.attention(qkv, HEADS)
		ml.backward()
		return sum(y) + sum_grad(qkv)
	}

	return time_iters("attention forward+backward (64t, 4h, 128e)", run)
}

bench_layernorm :: proc() -> Result {
	COUNT :: 64
	SIZE  :: 128

	w := ml.make(SIZE)
	defer ml.destroy(w)
	ml.fill_value(w, 1)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(COUNT, SIZE)
		ml.fill_value(x, 0.01)
		y := ml.layernorm(x, w_state)
		ml.backward()
		return sum(y) + sum_grad(x) + sum_grad(w_state)
	}

	state_w = w
	return time_iters("layernorm forward+backward (64x128)", run)
}

bench_softmax :: proc() -> Result {
	COUNT :: 64
	SIZE  :: 256

	run :: proc() -> f32 {
		ml.clear()
		x := ml.zeros(COUNT, SIZE)
		ml.fill_value(x, 0.01)
		y := ml.softmax(x)
		ml.backward()
		return sum(y) + sum_grad(x)
	}

	return time_iters("softmax forward+backward (64x256)", run)
}

bench_gelu :: proc() -> Result {
	N :: 64 * 512

	run :: proc() -> f32 {
		ml.clear()
		x := ml.zeros(N)
		ml.fill_value(x, 0.01)
		y := ml.gelu(x)
		ml.backward()
		return sum(y) + sum_grad(x)
	}

	return time_iters("gelu forward+backward (32768)", run)
}

bench_adam_update :: proc() -> Result {
	N :: 128 * 512

	rand.reset(SEED)
	p := ml.make(N)
	defer ml.destroy(p)
	ml.fill_normal(p, 0, 0.02)
	for i in 0 ..< N {
		ml.gradient(p)[i] = 0.001
	}

	state_w = p

	run :: proc() -> f32 {
		w_state := state_w
		// Refill gradient since update() zeroes it.
		for i in 0 ..< N {
			ml.gradient(w_state)[i] = 0.001
		}
		opt: ml.Optimizer
		if ml.optimize(&opt, period = 1) {
			ml.update(opt, w_state)
		}
		s: f32
		for v in ml.data(w_state) {
			s += v
		}
		return s
	}

	return time_iters("adam update (65536 params)", run)
}

bench_transformer_step :: proc() -> Result {
	rand.reset(SEED)
	model := tfm.make(E2E_LAYERS, E2E_HEADS, E2E_EMBEDDING_SIZE, E2E_VOCABULARY)
	defer tfm.destroy(model)

	tokens := make([]int, E2E_SEQUENCE_LENGTH)
	defer delete(tokens)
	targets := make([]int, E2E_SEQUENCE_LENGTH)
	defer delete(targets)
	for i in 0 ..< E2E_SEQUENCE_LENGTH {
		tokens[i]  = i % E2E_VOCABULARY
		targets[i] = (i + 1) % E2E_VOCABULARY
	}

	state_model  = model
	state_tokens = tokens
	state_target = targets

	run :: proc() -> f32 {
		m := state_model
		t := state_tokens
		tg := state_target

		ml.clear()
		logits := tfm.forward(m, t)
		loss   := ml.cross_entropy(logits, tg)
		_ = ml.mean(loss)
		ml.backward()

		opt: ml.Optimizer
		if ml.optimize(&opt, period = 1) {
			tfm.update(opt, m)
		}

		// Checksum: use a deterministic value from the forward pass.
		data := ml.data(logits)
		s: f32
		for v in data[:math.min(64, len(data))] {
			s += v
		}
		return s
	}

	return time_iters("transformer training step (4L 4H 128e 64t)", run)
}

// Multi-step training trajectory check. Builds a fresh seeded transformer,
// runs TRAJECTORY_STEPS optimizer steps, and prints the loss after each step
// plus a final parameter checksum. Loss should decrease monotonically (or
// near-so) and the per-step values + final checksum should match across
// versions on a single thread (within ~1 ULP). This catches backward/update
// bugs that the iteration-0 forward checksums in the timed benches miss.
TRAJECTORY_STEPS :: 10

verify_training_trajectory :: proc() {
	rand.reset(SEED)
	model := tfm.make(E2E_LAYERS, E2E_HEADS, E2E_EMBEDDING_SIZE, E2E_VOCABULARY)
	defer tfm.destroy(model)

	tokens := make([]int, E2E_SEQUENCE_LENGTH)
	defer delete(tokens)
	targets := make([]int, E2E_SEQUENCE_LENGTH)
	defer delete(targets)
	for i in 0 ..< E2E_SEQUENCE_LENGTH {
		tokens[i]  = i % E2E_VOCABULARY
		targets[i] = (i + 1) % E2E_VOCABULARY
	}

	fmt.println("--- training trajectory (single-threaded, fresh model) ---")
	fmt.printfln("%-10s %16s", "step", "loss")

	for step in 0 ..< TRAJECTORY_STEPS {
		ml.clear()
		logits := tfm.forward(model, tokens)
		ce     := ml.cross_entropy(logits, targets)
		loss   := ml.mean(ce)
		ml.backward()

		opt: ml.Optimizer
		if ml.optimize(&opt, period = 1) {
			tfm.update(opt, model)
		}

		step_s := fmt.tprintf("%v", step)
		loss_s := fmt.tprintf("%.6f", ml.data(loss)[0])
		fmt.printfln("%-10s %16s", step_s, loss_s)
	}

	// Final parameter checksum: forward once more on the trained model and
	// sum the logits. Sensitive to any drift in the trained weights.
	ml.clear()
	final_logits := tfm.forward(model, tokens)
	checksum: f32
	for v in ml.data(final_logits) {
		checksum += v
	}
	csum_s := fmt.tprintf("%.6f", checksum)
	fmt.printfln("final logits checksum: %16s", csum_s)
}

// Module-local globals used to smuggle setup state into the closure-less
// `proc()` literals required by time_iters. This keeps the timing harness's
// signature simple at the cost of a couple of file-scope vars.
@(private="file") state_w:      ml.Tensor
@(private="file") state_model:  tfm.Transformer
@(private="file") state_tokens: []int
@(private="file") state_target: []int
