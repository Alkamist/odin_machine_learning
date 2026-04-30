package machine_learning_benchmark

import "core:fmt"
import "core:time"
import "core:os"
import "core:math"
import "core:math/rand"

import ml  "../.."
import cpu "../../backends/cpu"
import tfm "../../networks/transformer"

SEED :: 0xC0FFEE

THREAD_COUNT :: 0
WARMUP       :: 2
ITERATIONS   :: 20

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

	ctx := cpu.context_create(256 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	fmt.printfln("threads=%v warmup=%v iterations=%v", thread_count, WARMUP, ITERATIONS)
	fmt.println("================================================================")

	cpu.set_thread_count(1)
	verify_training_trajectory()
	fmt.println()

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
	cpu.set_thread_count(thread_count)
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

time_iters :: proc(name: string, run: proc() -> f32) -> Result {
	for _ in 0 ..< WARMUP {
		_ = run()
	}

	min_ns:   i64 = max(i64)
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
		min_ms   = f64(min_ns)   / 1_000_000.0,
		mean_ms  = f64(total_ns) / 1_000_000.0 / f64(ITERATIONS),
		checksum = checksum,
	}
}

sum :: proc(t: ml.Tensor) -> f32 {
	s: f32
	for v in cpu.data(t) {
		s += v
	}
	return s
}

sum_grad :: proc(t: ml.Tensor) -> f32 {
	s: f32
	for v in cpu.gradient(t) {
		s += v
	}
	return s
}

bench_linear_inference_fwd :: proc() -> Result {
	INPUT  :: 512
	OUTPUT :: 2048

	rand.reset(SEED)
	w := ml.make(.F32, {OUTPUT, INPUT})
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(.F32, {INPUT})
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
	w := ml.make(.F32, {OUTPUT, INPUT})
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(.F32, {INPUT})
		ml.fill_value(x, 0.01)
		y := ml.linear(x, w_state)
		ml.backward()
		return sum(y) + sum_grad(x) + sum_grad(w_state)
	}

	state_w = w
	return time_iters("linear forward+backward    (count=1, 512x2048)", run)
}

bench_linear_training_fwd :: proc() -> Result {
	COUNT  :: 64
	INPUT  :: 128
	OUTPUT :: 512

	rand.reset(SEED)
	w := ml.make(.F32, {OUTPUT, INPUT})
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(.F32, {COUNT, INPUT})
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
	w := ml.make(.F32, {OUTPUT, INPUT})
	defer ml.destroy(w)
	ml.fill_normal(w, 0, 0.02)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(.F32, {COUNT, INPUT})
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
		qkv := ml.zeros(.F32, {TOKENS, 3 * EMBED})
		ml.fill_value(qkv, 0.01)
		q := ml.slice_trailing(qkv, 0,         EMBED)
		k := ml.slice_trailing(qkv, EMBED,     2 * EMBED)
		v := ml.slice_trailing(qkv, 2 * EMBED, 3 * EMBED)
		y := ml.attention(q, k, v, HEADS)
		ml.backward()
		return sum(y) + sum_grad(qkv)
	}

	return time_iters("attention forward+backward (64t, 4h, 128e)", run)
}

bench_layernorm :: proc() -> Result {
	COUNT :: 64
	SIZE  :: 128

	w := ml.make(.F32, {SIZE})
	defer ml.destroy(w)
	ml.fill_value(w, 1)

	run :: proc() -> f32 {
		w_state := state_w
		ml.clear()
		x := ml.zeros(.F32, {COUNT, SIZE})
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
		x := ml.zeros(.F32, {COUNT, SIZE})
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
		x := ml.zeros(.F32, {N})
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
	p := ml.make(.F32, {N})
	defer ml.destroy(p)
	ml.fill_normal(p, 0, 0.02)
	for i in 0 ..< N {
		cpu.gradient(p)[i] = 0.001
	}

	state_w = p

	run :: proc() -> f32 {
		w_state := state_w
		for i in 0 ..< N {
			cpu.gradient(w_state)[i] = 0.001
		}
		opt: ml.Optimizer
		if ml.optimize(&opt, period = 1) {
			ml.update(opt, w_state)
		}
		s: f32
		for v in cpu.data(w_state) {
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
	state_opt    = {}

	run :: proc() -> f32 {
		m  := state_model
		t  := state_tokens
		tg := state_target

		ml.clear()
		logits := tfm.forward(m, t)
		loss   := ml.cross_entropy(logits, tg)
		_       = ml.mean(loss)
		ml.backward()

		if ml.optimize(&state_opt, period = 1) {
			tfm.update(state_opt, m)
		}

		data := cpu.data(logits)
		s: f32
		for v in data[:math.min(64, len(data))] {
			s += v
		}
		return s
	}

	return time_iters("transformer training step (4L 4H 128e 64t)", run)
}

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

	opt: ml.Optimizer
	for step in 0 ..< TRAJECTORY_STEPS {
		ml.clear()
		logits := tfm.forward(model, tokens)
		ce     := ml.cross_entropy(logits, targets)
		loss   := ml.mean(ce)
		ml.backward()

		if ml.optimize(&opt, period = 1) {
			tfm.update(opt, model)
		}

		step_s := fmt.tprintf("%v", step)
		loss_s := fmt.tprintf("%.6f", cpu.data(loss)[0])
		fmt.printfln("%-10s %16s", step_s, loss_s)
	}

	ml.clear()
	final_logits := tfm.forward(model, tokens)
	checksum: f32
	for v in cpu.data(final_logits) {
		checksum += v
	}
	csum_s := fmt.tprintf("%.6f", checksum)
	fmt.printfln("final logits checksum: %16s", csum_s)
}

@(private="file") state_w:      ml.Tensor
@(private="file") state_model:  tfm.Transformer
@(private="file") state_tokens: []int
@(private="file") state_target: []int
@(private="file") state_opt:    ml.Optimizer
