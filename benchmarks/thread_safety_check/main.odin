// Verifies that multiple host threads, each with their own ml.Context, can
// concurrently run ops that fan out through the shared worker pool.
//
// All threads run an identical deterministic workload (linear forward+backward)
// and must produce a checksum byte-equal to a reference run on the main
// thread. If contexts leak into each other or the pool isn't properly
// serialized, results diverge or the run crashes.
package thread_safety_check

import "core:fmt"
import "core:os"
import "core:thread"
import ml "../.."

WORKLOAD_BATCH :: 64
WORKLOAD_IN   :: 128
WORKLOAD_OUT  :: 256
THREAD_COUNT  :: 4

run_workload :: proc() -> (data_sum, grad_sum: f32) {
	ctx := ml.context_create(16 * 1024 * 1024)
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)

	weight, _ := ml.make(WORKLOAD_OUT, WORKLOAD_IN)
	defer ml.destroy(weight)

	for i in 0 ..< ml.len(weight) {
		ml.data(weight)[i] = f32(i % 257) * 1e-3 - 0.05
	}

	input := ml.zeros(WORKLOAD_BATCH, WORKLOAD_IN)
	for i in 0 ..< ml.len(input) {
		ml.data(input)[i] = f32((i * 13 + 7) % 199) * 1e-3
	}

	output := ml.linear(input, weight)
	ml.backward()

	for v in ml.data(output)    do data_sum += v
	for g in ml.gradient(weight) do grad_sum += g
	return
}

Result :: struct {
	data_sum: f32,
	grad_sum: f32,
	id:       int,
}

worker :: proc(t: ^thread.Thread) {
	r := cast(^Result)t.data
	r.data_sum, r.grad_sum = run_workload()
}

main :: proc() {
	ml.set_thread_count(os.get_processor_core_count())
	defer ml.set_thread_count(1)

	ref_data, ref_grad := run_workload()
	fmt.printfln("reference: data_sum=%.6f  grad_sum=%.6f", ref_data, ref_grad)

	threads: [THREAD_COUNT]^thread.Thread
	results: [THREAD_COUNT]Result

	for i in 0 ..< THREAD_COUNT {
		results[i].id = i
		threads[i] = thread.create(worker)
		threads[i].data = &results[i]
		thread.start(threads[i])
	}

	for t in threads do thread.join(t)
	for t in threads do thread.destroy(t)

	// Forward (data_sum) must be bit-exact: every thread's parallelize fan-out
	// runs identical work and there's no cross-worker write contention.
	// Backward (grad_sum) is allowed to drift because linear_backward has
	// multiple workers accumulating into the same w_grad slice — pre-existing
	// non-determinism, same as running thread_count > 1 on the main thread
	// alone. We just check it stays in the right ballpark.
	GRAD_TOL :: f32(0.01) // 1% relative — generous, since drift is in low bits

	all_ok := true
	for r in results {
		fwd_ok := r.data_sum == ref_data
		grad_rel := abs(r.grad_sum - ref_grad) / abs(ref_grad)
		grad_ok := grad_rel < GRAD_TOL

		status := "OK"
		if !fwd_ok       do status = "FWD MISMATCH"
		else if !grad_ok do status = "GRAD DRIFT"

		fmt.printfln("thread %v: data_sum=%.6f  grad_sum=%.6f  rel=%.2e  %v",
			r.id, r.data_sum, r.grad_sum, grad_rel, status)

		if !fwd_ok || !grad_ok do all_ok = false
	}

	if all_ok {
		fmt.println("OK: pool stays correct under concurrent host threads")
	} else {
		fmt.println("FAIL")
		os.exit(1)
	}
}
