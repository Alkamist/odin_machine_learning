// gpu.linear vs ml.linear at the same shapes the CPU benchmark exercises.
//
// IMPORTANT: each gpu.linear call here is a full one-shot submit:
//   alloc descriptor set, alloc cmd buffer, record, submit, wait_idle, free.
// That overhead is fixed-cost per call (microseconds-scale) and dominates at
// small problem sizes. The number we get tells us the floor for "single
// op, blocking" usage. Real training will batch many ops per submit, which
// will change the picture — but we need that infrastructure first.
//
// Build: odin build examples/gpu_bench -o:speed -no-bounds-check -microarch:native -out:examples/gpu_bench/gpu_bench.exe
package gpu_bench

import "core:fmt"
import "core:math/rand"
import "core:time"
import ml "../.."
import "../../gpu"

WARMUP     :: 5
ITERATIONS :: 50

main :: proc() {
	ml.init(256 * 1024 * 1024)
	ml.set_thread_count(1)

	gpu.init()
	defer gpu.destroy()

	fmt.println("each row: ms per call (min, mean over", ITERATIONS, "iters)")
	fmt.printfln("%-28s %12s %12s %12s %12s %10s", "shape", "cpu_st_min", "cpu_st_mean", "gpu_min", "gpu_mean", "speedup")

	bench(1,  512, 2048)  // matches CPU bench_linear_inference_fwd
	bench(64, 128, 512)   // matches CPU bench_linear_training_fwd
	bench(64, 512, 2048)  // bigger
	bench(256, 1024, 1024)
	bench(1024, 1024, 1024)
}

bench :: proc(count, input_size, output_size: int) {
	rand.reset(0xC0FFEE)
	ml.clear()

	x_t := ml.zeros(count, input_size)
	w_p := ml.make(output_size, input_size)
	defer ml.destroy(w_p)
	for i in 0 ..< count * input_size {
		x_t.data[i] = rand.float32_range(-1, 1)
	}
	for i in 0 ..< output_size * input_size {
		w_p.data[i] = rand.float32_range(-1, 1) * 0.02
	}

	// CPU timing. ml.linear builds an op into the global op buffer + does the
	// forward pass eagerly inside parallelize, so each call here is one full
	// forward matmul. We re-clear between iterations so the op buffer doesn't
	// grow without bound.
	cpu_min, cpu_total: f64 = 1e18, 0
	for _ in 0 ..< WARMUP {
		ml.clear()
		x := ml.zeros(count, input_size); copy(x.data, x_t.data)
		_ = ml.linear(x, w_p)
	}
	for _ in 0 ..< ITERATIONS {
		ml.clear()
		x := ml.zeros(count, input_size); copy(x.data, x_t.data)

		t0 := time.tick_now()
		_ = ml.linear(x, w_p)
		dt := f64(time.tick_since(t0)) / 1_000_000.0
		if dt < cpu_min { cpu_min = dt }
		cpu_total += dt
	}
	cpu_mean := cpu_total / f64(ITERATIONS)

	// GPU timing. Allocate buffers + upload weights/inputs once outside the
	// loop. Each timed call is one gpu.linear which submits + wait_idles.
	x_g := gpu.alloc(count, input_size);          defer gpu.destroy_tensor(x_g)
	w_g := gpu.alloc(output_size, input_size);    defer gpu.destroy_tensor(w_g)
	y_g := gpu.alloc(count, output_size);         defer gpu.destroy_tensor(y_g)
	gpu.upload(x_t.data, x_g)
	gpu.upload(w_p.data, w_g)

	for _ in 0 ..< WARMUP {
		gpu.linear(x_g, w_g, y_g, count, input_size, output_size)
	}
	gpu_min, gpu_total: f64 = 1e18, 0
	for _ in 0 ..< ITERATIONS {
		t0 := time.tick_now()
		gpu.linear(x_g, w_g, y_g, count, input_size, output_size)
		dt := f64(time.tick_since(t0)) / 1_000_000.0
		if dt < gpu_min { gpu_min = dt }
		gpu_total += dt
	}
	gpu_mean := gpu_total / f64(ITERATIONS)

	shape   := fmt.tprintf("count=%v in=%v out=%v", count, input_size, output_size)
	cmin_s  := fmt.tprintf("%.3f", cpu_min)
	cmean_s := fmt.tprintf("%.3f", cpu_mean)
	gmin_s  := fmt.tprintf("%.3f", gpu_min)
	gmean_s := fmt.tprintf("%.3f", gpu_mean)
	speed_s := fmt.tprintf("%.2fx", cpu_min / gpu_min)
	fmt.printfln("%-28s %12s %12s %12s %12s %10s", shape, cmin_s, cmean_s, gmin_s, gmean_s, speed_s)
}
