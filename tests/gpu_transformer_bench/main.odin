package gpu_transformer_bench

import "core:fmt"
import "core:time"

import ml  "../.."
import cpu "../../backends/cpu"
import gpu "../../backends/gpu"
import tfm "../../networks/transformer"

SMALL :: Arch{layers = 4,  heads = 4, embed = 128, vocab = 256, seq = 64 }
LARGE :: Arch{layers = 12, heads = 8, embed = 512, vocab = 256, seq = 256}

WARMUP_STEPS :: 5
TIMED_STEPS  :: 30

Arch :: struct {
	layers, heads, embed, vocab, seq: int,
}

main :: proc() {
	archs := []Arch{SMALL, LARGE}
	for arch in archs {
		fmt.printfln("=== Architecture L=%v H=%v E=%v V=%v T=%v  (%v warmup + %v timed) ===",
			arch.layers, arch.heads, arch.embed, arch.vocab, arch.seq,
			WARMUP_STEPS, TIMED_STEPS)
		bench_arch(arch)
		fmt.println()
	}
}

bench_arch :: proc(arch: Arch) {
	tokens := make([]int, arch.seq)
	defer delete(tokens)
	for i in 0 ..< arch.seq do tokens[i] = (i * 7 + 3) % arch.vocab

	{
		cpu.set_thread_count(24)

		ctx := cpu.context_create(2 * 1024 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		model := tfm.make(arch.layers, arch.heads, arch.embed, arch.vocab)
		defer tfm.destroy(model)

		fwd_ns, fb_ns, step_ns := bench(model, tokens, false)

		fmt.printfln("CPU forward:           %.3f ms/step  (%.1f tokens/sec)",
			f64(fwd_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fwd_ns))
		fmt.printfln("CPU forward+backward:  %.3f ms/step  (%.1f tokens/sec)",
			f64(fb_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fb_ns))
		fmt.printfln("CPU full step (+Adam): %.3f ms/step  (%.1f tokens/sec)",
			f64(step_ns) / 1e6, f64(arch.seq) * 1e9 / f64(step_ns))
	}

	{
		ctx := gpu.context_create()
		defer gpu.context_destroy(ctx)
		ml.context_scope(ctx)

		model := tfm.make(arch.layers, arch.heads, arch.embed, arch.vocab)
		defer tfm.destroy(model)

		fwd_ns, fb_ns, step_ns := bench(model, tokens, true)

		fmt.printfln("GPU forward:           %.3f ms/step  (%.1f tokens/sec)",
			f64(fwd_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fwd_ns))
		fmt.printfln("GPU forward+backward:  %.3f ms/step  (%.1f tokens/sec)",
			f64(fb_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fb_ns))
		fmt.printfln("GPU full step (+Adam): %.3f ms/step  (%.1f tokens/sec)",
			f64(step_ns) / 1e6, f64(arch.seq) * 1e9 / f64(step_ns))
	}
}

// flush_if_gpu drains any pending GPU work before stopping the clock.
flush_if_gpu :: proc(is_gpu: bool) {
	if is_gpu {
		gpu.flush()
	}
}

bench :: proc(model: tfm.Transformer, tokens: []int, is_gpu: bool) -> (fwd_ns, fb_ns, step_ns: i64) {
	for _ in 0 ..< WARMUP_STEPS do step_forward(model, tokens)
	flush_if_gpu(is_gpu)
	t0 := time.tick_now()
	for _ in 0 ..< TIMED_STEPS do step_forward(model, tokens)
	flush_if_gpu(is_gpu)
	fwd_ns = i64(time.tick_since(t0)) / TIMED_STEPS

	for _ in 0 ..< WARMUP_STEPS do step_forward_backward(model, tokens)
	flush_if_gpu(is_gpu)
	t1 := time.tick_now()
	for _ in 0 ..< TIMED_STEPS do step_forward_backward(model, tokens)
	flush_if_gpu(is_gpu)
	fb_ns = i64(time.tick_since(t1)) / TIMED_STEPS

	opt: ml.Optimizer
	for _ in 0 ..< WARMUP_STEPS do step_full(model, tokens, &opt)
	flush_if_gpu(is_gpu)
	t2 := time.tick_now()
	for _ in 0 ..< TIMED_STEPS do step_full(model, tokens, &opt)
	flush_if_gpu(is_gpu)
	step_ns = i64(time.tick_since(t2)) / TIMED_STEPS
	return
}

step_forward :: proc(model: tfm.Transformer, tokens: []int) {
	ml.clear()
	_ = tfm.forward(model, tokens)
}

step_forward_backward :: proc(model: tfm.Transformer, tokens: []int) {
	ml.clear()
	_ = tfm.forward(model, tokens)
	ml.backward()
}

step_full :: proc(model: tfm.Transformer, tokens: []int, opt: ^ml.Optimizer) {
	ml.clear()
	_ = tfm.forward(model, tokens)
	ml.backward()
	if ml.optimize(opt, period = 1, learning_rate = 0.001) {
		tfm.update(opt^, model)
	}
}
