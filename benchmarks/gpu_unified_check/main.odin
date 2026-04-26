// End-to-end correctness check for the GPU backend's plug-in to ml.
//
//   Phase 1: alloc / clear plumbing — gpu.backend() routes ml.zeros and
//            ml.clear correctly, Tensors get GPU storage, allocations
//            track and release.
//
//   Phase 2: ml.add forward + backward — same workload run through both
//            backends produces matching outputs and matching gradients.
//
// Op kernels migrate one at a time; this test grows a phase-3 / phase-N
// section as more ops port over.
package gpu_unified_check

import "core:fmt"
import "core:os"
import "core:math"
import ml "../.."
import "../../gpu"

main :: proc() {
	gpu.init()
	defer gpu.destroy()

	gctx := gpu.context_create()
	defer gpu.context_destroy(gctx)
	gpu.context_scope(gctx)

	any_failed := false
	check :: proc(cond: bool, msg: string, any_failed: ^bool) {
		if cond {
			fmt.printfln("OK   %v", msg)
		} else {
			fmt.printfln("FAIL %v", msg)
			any_failed^ = true
		}
	}

	// --- Phase 1: alloc / clear plumbing ---
	{
		ctx := ml.context_create(16 * 1024 * 1024, gpu.backend())
		defer ml.context_destroy(ctx)
		ml.context_scope(ctx)

		a := ml.zeros(128, 128)
		b := ml.zeros(64, 32)

		check(a.backend == gpu.backend(),       "phase1: tensor a's backend is GPU",                 &any_failed)
		check(a.storage != nil,                 "phase1: tensor a has GPU storage",                  &any_failed)
		check(len(a.data) == 0,                 "phase1: tensor a's CPU data slice is empty",        &any_failed)
		check(len(a.gradient) == 0,             "phase1: tensor a's CPU gradient slice is empty",    &any_failed)

		storage_a := cast(^gpu.Gpu_Storage)a.storage
		storage_b := cast(^gpu.Gpu_Storage)b.storage
		check(storage_a.count == 128 * 128,     "phase1: storage_a count matches",                   &any_failed)
		check(storage_a.buffer != 0,            "phase1: storage_a.buffer valid",                    &any_failed)
		check(storage_a.grad_buffer != 0,       "phase1: storage_a.grad_buffer valid",               &any_failed)
		check(storage_b.buffer != storage_a.buffer, "phase1: a/b have distinct data buffers",        &any_failed)

		check(len(gctx.allocations) == 2,       "phase1: tracked allocations before clear",          &any_failed)
		ml.clear()
		check(len(gctx.allocations) == 0,       "phase1: allocations released by ml.clear",          &any_failed)
	}

	// --- Phase 2: ml.add forward + backward, CPU vs GPU ---
	{
		N :: 1024
		a_data := [N]f32{}
		b_data := [N]f32{}
		for i in 0 ..< N {
			a_data[i] = f32(i % 41) * 0.01 - 0.2
			b_data[i] = f32((i * 7) % 53) * 0.005 + 0.1
		}

		// CPU reference.
		cpu_out:    [N]f32
		cpu_grad_a: [N]f32
		cpu_grad_b: [N]f32
		{
			ctx := ml.context_create(1 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.tensor(a_data[:])
			b := ml.tensor(b_data[:])
			c := ml.add(a, b)
			ml.backward()

			copy(cpu_out[:],    c.data)
			copy(cpu_grad_a[:], a.gradient)
			copy(cpu_grad_b[:], b.gradient)
		}

		// GPU run.
		gpu_out:    [N]f32
		gpu_grad_a: [N]f32
		gpu_grad_b: [N]f32
		{
			ctx := ml.context_create(16 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros(N)
			b := ml.zeros(N)
			gpu.upload_tensor(a, a_data[:])
			gpu.upload_tensor(b, b_data[:])

			c := ml.add(a, b)
			ml.backward()

			gpu.download_tensor          (c, gpu_out[:])
			gpu.download_tensor_gradient (a, gpu_grad_a[:])
			gpu.download_tensor_gradient (b, gpu_grad_b[:])
		}

		// Compare. add is bit-exact across backends (no reduction order
		// issue), so we expect zero error, but allow a tiny tolerance for
		// any compiler-level FMA differences.
		TOL :: f32(1e-7)
		max_data_err: f32
		max_da_err:   f32
		max_db_err:   f32
		for i in 0 ..< N {
			max_data_err = math.max(max_data_err, math.abs(cpu_out[i]    - gpu_out[i]))
			max_da_err   = math.max(max_da_err,   math.abs(cpu_grad_a[i] - gpu_grad_a[i]))
			max_db_err   = math.max(max_db_err,   math.abs(cpu_grad_b[i] - gpu_grad_b[i]))
		}

		fmt.printfln("phase2: max abs error  data=%v  da=%v  db=%v", max_data_err, max_da_err, max_db_err)
		check(max_data_err <= TOL, "phase2: ml.add output matches CPU within tolerance",   &any_failed)
		check(max_da_err   <= TOL, "phase2: ml.add backward grad_a matches CPU",           &any_failed)
		check(max_db_err   <= TOL, "phase2: ml.add backward grad_b matches CPU",           &any_failed)
	}

	// --- Phase 3: ml.linear forward + backward, CPU vs GPU ---
	{
		COUNT       :: 32
		INPUT_SIZE  :: 64
		OUTPUT_SIZE :: 48

		x_data := [COUNT * INPUT_SIZE]f32{}
		w_data := [OUTPUT_SIZE * INPUT_SIZE]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32(i % 37) * 0.01 - 0.18
		for i in 0 ..< len(w_data) do w_data[i] = f32((i * 11) % 53) * 0.005 - 0.13

		cpu_y:  [COUNT * OUTPUT_SIZE]f32
		cpu_dx: [COUNT * INPUT_SIZE]f32
		cpu_dw: [OUTPUT_SIZE * INPUT_SIZE]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(COUNT, INPUT_SIZE)
			w := ml.zeros(OUTPUT_SIZE, INPUT_SIZE)
			copy(x.data, x_data[:])
			copy(w.data, w_data[:])

			y := ml.linear(x, w)
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dx[:], x.gradient)
			copy(cpu_dw[:], w.gradient)
		}

		gpu_y:  [COUNT * OUTPUT_SIZE]f32
		gpu_dx: [COUNT * INPUT_SIZE]f32
		gpu_dw: [OUTPUT_SIZE * INPUT_SIZE]f32
		{
			ctx := ml.context_create(16 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(COUNT, INPUT_SIZE)
			w := ml.zeros(OUTPUT_SIZE, INPUT_SIZE)
			gpu.upload_tensor(x, x_data[:])
			gpu.upload_tensor(w, w_data[:])

			y := ml.linear(x, w)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
			gpu.download_tensor_gradient (w, gpu_dw[:])
		}

		// linear has reductions across input_size and count, so accept fp32
		// reduction-order drift (same bar as gpu_grad_check: ~1e-5 abs at
		// these shapes).
		TOL :: f32(1e-4)
		max_y_err, max_dx_err, max_dw_err: f32
		for i in 0 ..< len(cpu_y)  do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx_err = math.max(max_dx_err, math.abs(cpu_dx[i] - gpu_dx[i]))
		for i in 0 ..< len(cpu_dw) do max_dw_err = math.max(max_dw_err, math.abs(cpu_dw[i] - gpu_dw[i]))

		fmt.printfln("phase3: max abs error  y=%v  dx=%v  dw=%v", max_y_err, max_dx_err, max_dw_err)
		check(max_y_err  <= TOL, "phase3: ml.linear output matches CPU within tolerance", &any_failed)
		check(max_dx_err <= TOL, "phase3: ml.linear backward dx matches CPU",             &any_failed)
		check(max_dw_err <= TOL, "phase3: ml.linear backward dw matches CPU",             &any_failed)
	}

	// --- Phase 4: ml.gelu forward + backward, CPU vs GPU ---
	{
		N :: 4096
		x_data := [N]f32{}
		for i in 0 ..< N do x_data[i] = f32(i % 73) * 0.03 - 1.1

		cpu_y:  [N]f32
		cpu_dx: [N]f32
		{
			ctx := ml.context_create(1 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.tensor(x_data[:])
			y := ml.gelu(x)
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dx[:], x.gradient)
		}

		gpu_y:  [N]f32
		gpu_dx: [N]f32
		{
			ctx := ml.context_create(8 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(N)
			gpu.upload_tensor(x, x_data[:])

			y := ml.gelu(x)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		// gelu math is elementwise; the only divergence comes from CPU vs.
		// GPU implementations of tanh / cosh. CPU uses libm; GPU uses
		// GLSL's tanh. Tight but not bit-exact.
		TOL :: f32(2e-6)
		max_y_err, max_dx_err: f32
		for i in 0 ..< N do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< N do max_dx_err = math.max(max_dx_err, math.abs(cpu_dx[i] - gpu_dx[i]))

		fmt.printfln("phase4: max abs error  y=%v  dx=%v", max_y_err, max_dx_err)
		check(max_y_err  <= TOL, "phase4: ml.gelu output matches CPU within tolerance", &any_failed)
		check(max_dx_err <= TOL, "phase4: ml.gelu backward dx matches CPU",             &any_failed)
	}

	// --- Phase 5: ml.select ---
	{
		VOCAB :: 256
		SIZE  :: 64
		N     :: 32

		table_data := [VOCAB * SIZE]f32{}
		for i in 0 ..< len(table_data) do table_data[i] = f32(i % 97) * 0.013 - 0.6
		indices := [N]int{}
		for i in 0 ..< N do indices[i] = (i * 17 + 5) % VOCAB

		cpu_y:  [N * SIZE]f32
		cpu_dt: [VOCAB * SIZE]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			table := ml.zeros(VOCAB, SIZE)
			copy(table.data, table_data[:])

			y := ml.select(table, indices[:])
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dt[:], table.gradient)
		}

		gpu_y:  [N * SIZE]f32
		gpu_dt: [VOCAB * SIZE]f32
		{
			ctx := ml.context_create(16 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			table := ml.zeros(VOCAB, SIZE)
			gpu.upload_tensor(table, table_data[:])

			y := ml.select(table, indices[:])
			ml.backward()

			gpu.download_tensor          (y,     gpu_y[:])
			gpu.download_tensor_gradient (table, gpu_dt[:])
		}

		TOL :: f32(1e-7)
		max_y_err, max_dt_err: f32
		for i in 0 ..< len(cpu_y)  do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dt) do max_dt_err = math.max(max_dt_err, math.abs(cpu_dt[i] - gpu_dt[i]))
		fmt.printfln("phase5: max abs error  y=%v  dtable=%v", max_y_err, max_dt_err)
		check(max_y_err  <= TOL, "phase5: ml.select output matches CPU",          &any_failed)
		check(max_dt_err <= TOL, "phase5: ml.select backward dtable matches CPU", &any_failed)
	}

	// --- Phase 6: ml.rope ---
	{
		TOKENS    :: 8
		HEADS     :: 4
		HEAD_SIZE :: 16

		x_data := [TOKENS * HEADS * HEAD_SIZE]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32(i % 89) * 0.011 - 0.5

		cpu_y:  [TOKENS * HEADS * HEAD_SIZE]f32
		cpu_dx: [TOKENS * HEADS * HEAD_SIZE]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(TOKENS, HEADS * HEAD_SIZE)
			copy(x.data, x_data[:])

			y := ml.rope(x, HEADS)
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dx[:], x.gradient)
		}

		gpu_y:  [TOKENS * HEADS * HEAD_SIZE]f32
		gpu_dx: [TOKENS * HEADS * HEAD_SIZE]f32
		{
			ctx := ml.context_create(8 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(TOKENS, HEADS * HEAD_SIZE)
			gpu.upload_tensor(x, x_data[:])

			y := ml.rope(x, HEADS)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		TOL :: f32(2e-6)
		max_y_err, max_dx_err: f32
		for i in 0 ..< len(cpu_y)  do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx_err = math.max(max_dx_err, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase6: max abs error  y=%v  dx=%v", max_y_err, max_dx_err)
		check(max_y_err  <= TOL, "phase6: ml.rope output matches CPU",      &any_failed)
		check(max_dx_err <= TOL, "phase6: ml.rope backward dx matches CPU", &any_failed)
	}

	// --- Phase 7: ml.slice_trailing ---
	{
		ROWS     :: 16
		TRAILING :: 96
		START    :: 32
		END      :: 80

		x_data := [ROWS * TRAILING]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32(i % 71) * 0.017 - 0.4

		cpu_y:  [ROWS * (END - START)]f32
		cpu_dx: [ROWS * TRAILING]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(ROWS, TRAILING)
			copy(x.data, x_data[:])

			y := ml.slice_trailing(x, START, END)
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dx[:], x.gradient)
		}

		gpu_y:  [ROWS * (END - START)]f32
		gpu_dx: [ROWS * TRAILING]f32
		{
			ctx := ml.context_create(8 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(ROWS, TRAILING)
			gpu.upload_tensor(x, x_data[:])

			y := ml.slice_trailing(x, START, END)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		TOL :: f32(1e-7)
		max_y_err, max_dx_err: f32
		for i in 0 ..< len(cpu_y)  do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx_err = math.max(max_dx_err, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase7: max abs error  y=%v  dx=%v", max_y_err, max_dx_err)
		check(max_y_err  <= TOL, "phase7: ml.slice_trailing output matches CPU",      &any_failed)
		check(max_dx_err <= TOL, "phase7: ml.slice_trailing backward dx matches CPU", &any_failed)
	}

	// --- Phase 8: ml.concat (3 inputs only on GPU) ---
	{
		ROWS :: 8
		T_A  :: 16
		T_B  :: 24
		T_C  :: 32

		a_data := [ROWS * T_A]f32{}
		b_data := [ROWS * T_B]f32{}
		c_data := [ROWS * T_C]f32{}
		for i in 0 ..< len(a_data) do a_data[i] = f32(i % 31) * 0.02 - 0.3
		for i in 0 ..< len(b_data) do b_data[i] = f32((i * 5) % 41) * 0.015 + 0.1
		for i in 0 ..< len(c_data) do c_data[i] = f32((i * 11) % 53) * 0.01 - 0.2

		out_size :: ROWS * (T_A + T_B + T_C)
		cpu_y:  [out_size]f32
		cpu_da: [ROWS * T_A]f32
		cpu_db: [ROWS * T_B]f32
		cpu_dc: [ROWS * T_C]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros(ROWS, T_A)
			b := ml.zeros(ROWS, T_B)
			c := ml.zeros(ROWS, T_C)
			copy(a.data, a_data[:])
			copy(b.data, b_data[:])
			copy(c.data, c_data[:])

			y := ml.concat(a, b, c)
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_da[:], a.gradient)
			copy(cpu_db[:], b.gradient)
			copy(cpu_dc[:], c.gradient)
		}

		gpu_y:  [out_size]f32
		gpu_da: [ROWS * T_A]f32
		gpu_db: [ROWS * T_B]f32
		gpu_dc: [ROWS * T_C]f32
		{
			ctx := ml.context_create(8 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros(ROWS, T_A)
			b := ml.zeros(ROWS, T_B)
			c := ml.zeros(ROWS, T_C)
			gpu.upload_tensor(a, a_data[:])
			gpu.upload_tensor(b, b_data[:])
			gpu.upload_tensor(c, c_data[:])

			y := ml.concat(a, b, c)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (a, gpu_da[:])
			gpu.download_tensor_gradient (b, gpu_db[:])
			gpu.download_tensor_gradient (c, gpu_dc[:])
		}

		TOL :: f32(1e-7)
		max_y_err, max_da_err, max_db_err, max_dc_err: f32
		for i in 0 ..< len(cpu_y)  do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_da) do max_da_err = math.max(max_da_err, math.abs(cpu_da[i] - gpu_da[i]))
		for i in 0 ..< len(cpu_db) do max_db_err = math.max(max_db_err, math.abs(cpu_db[i] - gpu_db[i]))
		for i in 0 ..< len(cpu_dc) do max_dc_err = math.max(max_dc_err, math.abs(cpu_dc[i] - gpu_dc[i]))
		fmt.printfln("phase8: max abs error  y=%v  da=%v  db=%v  dc=%v", max_y_err, max_da_err, max_db_err, max_dc_err)
		check(max_y_err  <= TOL, "phase8: ml.concat output matches CPU",   &any_failed)
		check(max_da_err <= TOL, "phase8: ml.concat backward da matches CPU", &any_failed)
		check(max_db_err <= TOL, "phase8: ml.concat backward db matches CPU", &any_failed)
		check(max_dc_err <= TOL, "phase8: ml.concat backward dc matches CPU", &any_failed)
	}

	// --- Phase 9: ml.softmax ---
	{
		COUNT :: 8
		SIZE  :: 64

		x_data := [COUNT * SIZE]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 7) % 53) * 0.05 - 1.2

		cpu_y:  [COUNT * SIZE]f32
		cpu_dx: [COUNT * SIZE]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(COUNT, SIZE)
			copy(x.data, x_data[:])

			y := ml.softmax(x)
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dx[:], x.gradient)
		}

		gpu_y:  [COUNT * SIZE]f32
		gpu_dx: [COUNT * SIZE]f32
		{
			ctx := ml.context_create(8 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(COUNT, SIZE)
			gpu.upload_tensor(x, x_data[:])

			y := ml.softmax(x)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		TOL :: f32(1e-6)
		max_y_err, max_dx_err: f32
		for i in 0 ..< len(cpu_y)  do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx_err = math.max(max_dx_err, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase9: max abs error  y=%v  dx=%v", max_y_err, max_dx_err)
		check(max_y_err  <= TOL, "phase9: ml.softmax output matches CPU",      &any_failed)
		check(max_dx_err <= TOL, "phase9: ml.softmax backward dx matches CPU", &any_failed)
	}

	// --- Phase 10: ml.permute ---
	{
		D0 :: 4
		D1 :: 6
		D2 :: 8
		x_data := [D0 * D1 * D2]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 13 + 5) % 67) * 0.02 - 0.4

		cpu_y:  [D0 * D1 * D2]f32
		cpu_dx: [D0 * D1 * D2]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(D0, D1, D2)
			copy(x.data, x_data[:])

			y := ml.permute(x, {1, 0, 2})
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dx[:], x.gradient)
		}

		gpu_y:  [D0 * D1 * D2]f32
		gpu_dx: [D0 * D1 * D2]f32
		{
			ctx := ml.context_create(8 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(D0, D1, D2)
			gpu.upload_tensor(x, x_data[:])

			y := ml.permute(x, {1, 0, 2})
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		TOL :: f32(1e-7)
		max_y_err, max_dx_err: f32
		for i in 0 ..< len(cpu_y)  do max_y_err  = math.max(max_y_err,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx_err = math.max(max_dx_err, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase10: max abs error  y=%v  dx=%v", max_y_err, max_dx_err)
		check(max_y_err  <= TOL, "phase10: ml.permute output matches CPU",      &any_failed)
		check(max_dx_err <= TOL, "phase10: ml.permute backward dx matches CPU", &any_failed)
	}

	// --- Phase 11: ml.causal_mask ---
	{
		H :: 4
		T :: 16
		x_data := [H * T * T]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 17) % 91) * 0.04 - 1.5

		cpu_y:  [H * T * T]f32
		cpu_dx: [H * T * T]f32
		{
			ctx := ml.context_create(2 * 1024 * 1024)
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(H, T, T)
			copy(x.data, x_data[:])

			y := ml.causal_mask(x)
			ml.backward()

			copy(cpu_y[:],  y.data)
			copy(cpu_dx[:], x.gradient)
		}

		gpu_y:  [H * T * T]f32
		gpu_dx: [H * T * T]f32
		{
			ctx := ml.context_create(8 * 1024 * 1024, gpu.backend())
			defer ml.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros(H, T, T)
			gpu.upload_tensor(x, x_data[:])

			y := ml.causal_mask(x)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		TOL :: f32(1e-7)
		// Compare; -inf positions are equal to themselves, finite positions
		// must match. (We can't subtract -inf - -inf cleanly, so check.)
		max_y_err, max_dx_err: f32
		for i in 0 ..< len(cpu_y) {
			c := cpu_y[i]
			g := gpu_y[i]
			if c == g do continue
			max_y_err = math.max(max_y_err, math.abs(c - g))
		}
		for i in 0 ..< len(cpu_dx) do max_dx_err = math.max(max_dx_err, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase11: max abs error  y=%v  dx=%v", max_y_err, max_dx_err)
		check(max_y_err  <= TOL, "phase11: ml.causal_mask output matches CPU",      &any_failed)
		check(max_dx_err <= TOL, "phase11: ml.causal_mask backward dx matches CPU", &any_failed)
	}

	if any_failed {
		os.exit(1)
	}
	fmt.println("OK: gpu backend integration is healthy")
}
