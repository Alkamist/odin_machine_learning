package cuda_graph_smoke

// Smoke test for the CUDA graph capture/replay API. Captures one
// `add` op into a graph, replays it many times, verifies output, and
// reports per-replay vs per-direct-launch latency.

import "base:builtin"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:time"

import ml  "../.."
import cu  "../../backends/cuda"

bset :: proc(t: ml.Tensor, kind: ml.Buffer_Kind, src: []byte, loc := #caller_location) {
	t.backend.buffer_set(t.buffers[kind], src, loc)
}
bget :: proc(t: ml.Tensor, kind: ml.Buffer_Kind, dst: []byte, loc := #caller_location) {
	t.backend.buffer_get(t.buffers[kind], dst, loc)
}

main :: proc() {
	fail :: proc(msg: string) -> ! {
		fmt.eprintln("FAIL:", msg)
		os.exit(1)
	}

	ctx := cu.context_create()
	ml.context_begin(ctx)
	defer { ml.context_end(); cu.context_destroy(ctx) }

	N :: 1 << 16  // small enough that launch overhead matters
	x_host := builtin.make([]f32, N); defer delete(x_host)
	y_host := builtin.make([]f32, N); defer delete(y_host)
	for i in 0..<N {
		x_host[i] = f32(i)
		y_host[i] = f32(2 * i)
	}

	a := ml.alloc(.F32, []int{N}, persistent=true, buffers={.Data})
	b := ml.alloc(.F32, []int{N}, persistent=true, buffers={.Data})
	bset(a, .Data, mem.slice_to_bytes(x_host))
	bset(b, .Data, mem.slice_to_bytes(y_host))

	// Warmup: run add once normally so the activation pool has a slot for the
	// output and the kernel module is compiled before we begin capture.
	_ = ml.add(a, b)
	ml.clear()

	// Capture.
	cu.begin_graph_capture()
	out := ml.add(a, b)
	cu.end_graph_capture()
	fmt.printfln("captured 1 add op into graph")

	// Replay loop.
	REPS :: 1000
	t0 := time.tick_now()
	for _ in 0..<REPS {
		cu.replay_graph()
	}
	// Drain.
	out_bytes := builtin.make([]byte, N * size_of(f32)); defer delete(out_bytes)
	bget(out, .Data, out_bytes)
	dur_ms := f64(time.duration_milliseconds(time.tick_since(t0)))
	per_replay_us := dur_ms * 1000.0 / REPS
	fmt.printfln("replay  x %d: %.3f ms total (%.2f us/replay)", REPS, dur_ms, per_replay_us)

	// Direct-launch comparison.
	ml.clear()
	t1 := time.tick_now()
	for _ in 0..<REPS {
		_ = ml.add(a, b)
		ml.clear()
	}
	out2_bytes := builtin.make([]byte, N * size_of(f32)); defer delete(out2_bytes)
	out2 := ml.add(a, b)
	bget(out2, .Data, out2_bytes)
	dur2_ms := f64(time.duration_milliseconds(time.tick_since(t1)))
	per_launch_us := dur2_ms * 1000.0 / REPS
	fmt.printfln("direct  x %d: %.3f ms total (%.2f us/launch)", REPS, dur2_ms, per_launch_us)

	speedup := per_launch_us / per_replay_us
	fmt.printfln("graph speedup: %.2fx", speedup)

	// Verify output of the captured replay.
	out_f := mem.slice_data_cast([]f32, out_bytes)
	for i in 0..<N {
		expected := f32(3 * i)
		if out_f[i] != expected {
			fail(fmt.tprintf("output mismatch at %d: got %v want %v", i, out_f[i], expected))
		}
	}
	fmt.printfln("OK  replayed graph produced correct output for %d elements", N)
}
