package gpu_alloc_bench

import "base:builtin"

import "core:fmt"
import "core:time"

import ml  "../.."
import gpu "../../backends/gpu"

ITERS :: 1000

main :: proc() {
	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	fmt.println("=== GPU activation alloc cost (zeros + clear cycle, no compute) ===")
	fmt.println()

	bench("zeros [1, 2048]   (Bf16, .Data + .Gradient)", {}, proc() {
		_ = ml.zeros(.Bf16, {1, 2048})
	})

	bench("zeros [1, 2048]   inference mode (.Data only)", {.No_Gradients}, proc() {
		_ = ml.zeros(.Bf16, {1, 2048})
	})
	bench("zeros [1, 10240]  inference mode (.Data only)", {.No_Gradients}, proc() {
		_ = ml.zeros(.Bf16, {1, 10240})
	})

	bench("zeros [1, 2560]   (Bf16, .Data + .Gradient)", {}, proc() {
		_ = ml.zeros(.Bf16, {1, 2560})
	})

	bench("zeros [1, 10240]  (Bf16, .Data + .Gradient)", {}, proc() {
		_ = ml.zeros(.Bf16, {1, 10240})
	})

	bench("alloc data-only [1, 2048] (Bf16, .Data)", {}, proc() {
		_ = ml.alloc(.Bf16, {1, 2048}, persistent=false, buffers=ml.Buffer_Set{.Data})
	})

	bench("alloc data-only [1, 2560] (Bf16, .Data)", {}, proc() {
		_ = ml.alloc(.Bf16, {1, 2560}, persistent=false, buffers=ml.Buffer_Set{.Data})
	})
}

bench :: proc(label: string, flags: ml.Clear_Flags, fn: proc()) {
	for _ in 0 ..< 50 {
		ml.clear(flags)
		fn()
	}

	t0 := time.tick_now()
	for _ in 0 ..< ITERS {
		ml.clear(flags)
		fn()
	}
	per_op_ns := i64(time.tick_since(t0)) / ITERS
	fmt.printfln("  %v   %.3f ms/op", label, f64(per_op_ns) / 1e6)
}
