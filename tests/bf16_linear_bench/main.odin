package bf16_linear_bench

import "core:fmt"
import "core:time"

import ml  "../.."
import gpu "../../backends/vulkan"

WARMUP_DISPATCHES :: 10
TIMED_DISPATCHES  :: 50

Shape :: struct {
	label:   string,
	m, k, n: int,
}

main :: proc() {
	shapes := []Shape{
		{"tiny    M=64  K=64   N=128 ", 64,   64,   128},
		{"medium  M=512 K=768  N=768 ", 512,  768,  768},
		{"big     M=512 K=2048 N=2048", 512,  2048, 2048},
		{"huge    M=2048 K=2048 N=2048", 2048, 2048, 2048},
	}

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	for shape in shapes {
		bench_shape(shape)
	}
}

bench_shape :: proc(shape: Shape) {
	x := ml.make(.Bf16, {shape.m, shape.k})
	w := ml.make(.Bf16, {shape.n, shape.k})
	ml.fill_normal(x, 0, 1)
	ml.fill_normal(w, 0, 1)

	fwd_ns := time_op(shape, proc(x, w: ml.Tensor) {
		ml.clear()
		_ = ml.linear(x, w)
	}, x, w)

	fwd_bwd_ns := time_op(shape, proc(x, w: ml.Tensor) {
		ml.clear()
		y    := ml.linear(x, w)
		y_f  := ml.cast_to(y, .F32)
		_     = ml.mean(y_f)
		ml.backward()
	}, x, w)

	bwd_ns := fwd_bwd_ns - fwd_ns

	flops := f64(shape.m) * f64(shape.k) * f64(shape.n) * 2.0
	fwd_tflops := flops / f64(fwd_ns) / 1e3
	// Backward is 2 GEMMs (dx + dw) of the same shape.
	bwd_tflops := 2.0 * flops / f64(bwd_ns) / 1e3

	fmt.printfln("%v  fwd %.3f ms (%.2f TFLOPS)  bwd %.3f ms (%.2f TFLOPS)",
		shape.label,
		f64(fwd_ns) / 1e6, fwd_tflops,
		f64(bwd_ns) / 1e6, bwd_tflops)
}

time_op :: proc(shape: Shape, op: proc(x, w: ml.Tensor), x, w: ml.Tensor) -> i64 {
	for _ in 0 ..< WARMUP_DISPATCHES { op(x, w) }
	gpu.flush()
	t0 := time.tick_now()
	for _ in 0 ..< TIMED_DISPATCHES { op(x, w) }
	gpu.flush()
	return i64(time.tick_since(t0)) / TIMED_DISPATCHES
}
