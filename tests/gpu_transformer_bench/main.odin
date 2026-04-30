package gpu_transformer_bench

import "core:fmt"
import "core:time"

import ml  "../.."
import cpu "../../backends/cpu"
import gpu "../../backends/gpu"
import tfm "../../networks/transformer"

import "base:builtin"

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

		fmt.printfln("GPU F32  forward:           %.3f ms/step  (%.1f tokens/sec)",
			f64(fwd_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fwd_ns))
		fmt.printfln("GPU F32  forward+backward:  %.3f ms/step  (%.1f tokens/sec)",
			f64(fb_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fb_ns))
		fmt.printfln("GPU F32  full step (+Adam): %.3f ms/step  (%.1f tokens/sec)",
			f64(step_ns) / 1e6, f64(arch.seq) * 1e9 / f64(step_ns))

		fwd_b_ns, fb_b_ns, step_b_ns := bench_bf16(arch, tokens)

		fmt.printfln("GPU Bf16 forward:           %.3f ms/step  (%.1f tokens/sec)",
			f64(fwd_b_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fwd_b_ns))
		fmt.printfln("GPU Bf16 forward+backward:  %.3f ms/step  (%.1f tokens/sec)",
			f64(fb_b_ns) / 1e6, f64(arch.seq) * 1e9 / f64(fb_b_ns))
		fmt.printfln("GPU Bf16 full step (+Adam): %.3f ms/step  (%.1f tokens/sec)",
			f64(step_b_ns) / 1e6, f64(arch.seq) * 1e9 / f64(step_b_ns))
	}
}

Bf16_Layer :: struct {
	norm0_weight, qkv_weight, proj_weight, norm1_weight, mlp_up_weight, mlp_down_weight: ml.Tensor,
}

Bf16_Model :: struct {
	head_count, embedding_size: int,
	token_embeddings:           ml.Tensor,
	layers:                     []Bf16_Layer,
	norm_weight, output_weight: ml.Tensor,
}

bf16_make :: proc(arch: Arch) -> (m: Bf16_Model) {
	m.head_count     = arch.heads
	m.embedding_size = arch.embed

	m.token_embeddings = ml.make(.F32, {arch.vocab, arch.embed})
	ml.fill_normal(m.token_embeddings, 0, 0.02)

	m.layers = builtin.make([]Bf16_Layer, arch.layers)
	hidden  := 4 * arch.embed
	for &layer in m.layers {
		layer.norm0_weight    = ml.make(.F32, {arch.embed})
		layer.qkv_weight      = ml.make(.F32, {3 * arch.embed, arch.embed})
		layer.proj_weight     = ml.make(.F32, {arch.embed,     arch.embed})
		layer.norm1_weight    = ml.make(.F32, {arch.embed})
		layer.mlp_up_weight   = ml.make(.F32, {hidden,    arch.embed})
		layer.mlp_down_weight = ml.make(.F32, {arch.embed, hidden})
		ml.fill_value(layer.norm0_weight, 1)
		ml.fill_value(layer.norm1_weight, 1)
		ml.fill_normal(layer.qkv_weight,  0, 0.02)
		ml.fill_normal(layer.proj_weight, 0, 0.02)
		ml.he_initialization(layer.mlp_up_weight,   arch.embed)
		ml.he_initialization(layer.mlp_down_weight, hidden)
	}

	m.norm_weight   = ml.make(.F32, {arch.embed})
	m.output_weight = ml.make(.F32, {arch.vocab, arch.embed})
	ml.fill_value(m.norm_weight, 1)
	ml.fill_normal(m.output_weight, 0, 0.02)
	return
}

bf16_destroy :: proc(m: Bf16_Model) {
	ml.destroy(m.token_embeddings)
	for layer in m.layers {
		ml.destroy(layer.norm0_weight)
		ml.destroy(layer.qkv_weight)
		ml.destroy(layer.proj_weight)
		ml.destroy(layer.norm1_weight)
		ml.destroy(layer.mlp_up_weight)
		ml.destroy(layer.mlp_down_weight)
	}
	ml.destroy(m.norm_weight)
	ml.destroy(m.output_weight)
	delete(m.layers)
}

bf16_forward :: proc(m: Bf16_Model, tokens: []int) -> ml.Tensor {
	residual := ml.cast_to(ml.select(m.token_embeddings, tokens), .Bf16)
	embed := m.embedding_size
	for layer in m.layers {
		n0_w  := ml.cast_to(layer.norm0_weight,    .Bf16)
		qkv_w := ml.cast_to(layer.qkv_weight,      .Bf16)
		pr_w  := ml.cast_to(layer.proj_weight,     .Bf16)
		n1_w  := ml.cast_to(layer.norm1_weight,    .Bf16)
		up_w  := ml.cast_to(layer.mlp_up_weight,   .Bf16)
		dn_w  := ml.cast_to(layer.mlp_down_weight, .Bf16)

		normed := ml.layernorm(residual, n0_w)
		qkv    := ml.linear(normed, qkv_w)

		q := ml.slice_trailing(qkv, 0,         embed)
		k := ml.slice_trailing(qkv, embed,     2 * embed)
		v := ml.slice_trailing(qkv, 2 * embed, 3 * embed)
		q  = ml.rope(q, m.head_count)
		k  = ml.rope(k, m.head_count)
		qkv_concat := ml.concat(q, k, v)

		attn_out := ml.attention(qkv_concat, m.head_count)
		attn_out  = ml.linear(attn_out, pr_w)
		residual  = ml.add(residual, attn_out)

		normed_mlp := ml.layernorm(residual, n1_w)
		mlp_out    := ml.linear(normed_mlp, up_w)
		mlp_out     = ml.gelu(mlp_out)
		mlp_out     = ml.linear(mlp_out, dn_w)
		residual    = ml.add(residual, mlp_out)
	}
	nm_w  := ml.cast_to(m.norm_weight,   .Bf16)
	out_w := ml.cast_to(m.output_weight, .Bf16)
	out_bf  := ml.layernorm(residual, nm_w)
	logits  := ml.linear(out_bf, out_w)
	return ml.cast_to(logits, .F32)
}

bf16_update :: proc(opt: ml.Optimizer, m: Bf16_Model) {
	ml.update(opt, m.token_embeddings)
	for layer in m.layers {
		ml.update(opt, layer.norm0_weight)
		ml.update(opt, layer.qkv_weight)
		ml.update(opt, layer.proj_weight)
		ml.update(opt, layer.norm1_weight)
		ml.update(opt, layer.mlp_up_weight)
		ml.update(opt, layer.mlp_down_weight)
	}
	ml.update(opt, m.norm_weight)
	ml.update(opt, m.output_weight)
}

bench_bf16 :: proc(arch: Arch, tokens: []int) -> (fwd_ns, fb_ns, step_ns: i64) {
	m := bf16_make(arch)
	defer bf16_destroy(m)

	step_fwd := proc(m: Bf16_Model, tokens: []int) {
		ml.clear()
		_ = bf16_forward(m, tokens)
	}
	step_fwd_bwd := proc(m: Bf16_Model, tokens: []int) {
		ml.clear()
		logits := bf16_forward(m, tokens)
		_       = ml.mean(logits)
		ml.backward()
	}

	for _ in 0 ..< WARMUP_STEPS do step_fwd(m, tokens)
	gpu.flush()
	t0 := time.tick_now()
	for _ in 0 ..< TIMED_STEPS do step_fwd(m, tokens)
	gpu.flush()
	fwd_ns = i64(time.tick_since(t0)) / TIMED_STEPS

	for _ in 0 ..< WARMUP_STEPS do step_fwd_bwd(m, tokens)
	gpu.flush()
	t1 := time.tick_now()
	for _ in 0 ..< TIMED_STEPS do step_fwd_bwd(m, tokens)
	gpu.flush()
	fb_ns = i64(time.tick_since(t1)) / TIMED_STEPS

	opt: ml.Optimizer
	step_full_b := proc(m: Bf16_Model, tokens: []int, opt: ^ml.Optimizer) {
		ml.clear()
		logits := bf16_forward(m, tokens)
		_       = ml.mean(logits)
		ml.backward()
		if ml.optimize(opt, period = 1, learning_rate = 0.001) {
			bf16_update(opt^, m)
		}
	}
	for _ in 0 ..< WARMUP_STEPS do step_full_b(m, tokens, &opt)
	gpu.flush()
	t2 := time.tick_now()
	for _ in 0 ..< TIMED_STEPS do step_full_b(m, tokens, &opt)
	gpu.flush()
	step_ns = i64(time.tick_since(t2)) / TIMED_STEPS
	return
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
