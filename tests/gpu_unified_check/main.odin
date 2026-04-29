package gpu_unified_check

import "core:fmt"
import "core:os"
import "core:math"

import ml  "../.."
import cpu "../../backends/cpu"
import gpu "../../backends/gpu"

main :: proc() {
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
		ctx := gpu.context_create()
		defer gpu.context_destroy(ctx)
		ml.context_scope(ctx)

		a := ml.zeros({128, 128})
		b := ml.zeros({64, 32})

		check(a.backend == &ctx.backend,       "phase1: tensor a's backend is GPU",      &any_failed)
		check(ml.len(a) == 128 * 128,          "phase1: tensor a element count matches", &any_failed)

		data_a := transmute(gpu.Gpu_Buffer)a.buffers[.Data]
		grad_a := transmute(gpu.Gpu_Buffer)a.buffers[.Gradient]
		data_b := transmute(gpu.Gpu_Buffer)b.buffers[.Data]
		check(data_a.buffer != 0,              "phase1: tensor a data buffer valid",     &any_failed)
		check(grad_a.buffer != 0,              "phase1: tensor a gradient buffer valid", &any_failed)
		check(data_b.buffer != data_a.buffer,  "phase1: a/b have distinct data buffers", &any_failed)

		// Each tensor allocates 2 buffers (Data + Gradient), so 4 total before clear.
		gctx := cast(^gpu.Context)ctx
		check(len(gctx.activations) == 4,      "phase1: tracked activations before clear", &any_failed)
		ml.clear()
		check(len(gctx.activations) == 0,      "phase1: activations released by ml.clear", &any_failed)
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
			ctx := cpu.context_create(1 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.tensor(a_data[:])
			b := ml.tensor(b_data[:])
			c := ml.add(a, b)
			ml.backward()

			copy(cpu_out[:],    cpu.data(c))
			copy(cpu_grad_a[:], cpu.gradient(a))
			copy(cpu_grad_b[:], cpu.gradient(b))
		}

		// GPU run.
		gpu_out:    [N]f32
		gpu_grad_a: [N]f32
		gpu_grad_b: [N]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros({N})
			b := ml.zeros({N})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, INPUT_SIZE})
			w := ml.zeros({OUTPUT_SIZE, INPUT_SIZE})
			ml.set_data(x, x_data[:])
			ml.set_data(w, w_data[:])

			y := ml.linear(x, w)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
			copy(cpu_dw[:], cpu.gradient(w))
		}

		gpu_y:  [COUNT * OUTPUT_SIZE]f32
		gpu_dx: [COUNT * INPUT_SIZE]f32
		gpu_dw: [OUTPUT_SIZE * INPUT_SIZE]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, INPUT_SIZE})
			w := ml.zeros({OUTPUT_SIZE, INPUT_SIZE})
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
			ctx := cpu.context_create(1 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.tensor(x_data[:])
			y := ml.gelu(x)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}

		gpu_y:  [N]f32
		gpu_dx: [N]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({N})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			table := ml.zeros({VOCAB, SIZE})
			ml.set_data(table, table_data[:])

			y := ml.select(table, indices[:])
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dt[:], cpu.gradient(table))
		}

		gpu_y:  [N * SIZE]f32
		gpu_dt: [VOCAB * SIZE]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			table := ml.zeros({VOCAB, SIZE})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({TOKENS, HEADS * HEAD_SIZE})
			ml.set_data(x, x_data[:])

			y := ml.rope(x, HEADS)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}

		gpu_y:  [TOKENS * HEADS * HEAD_SIZE]f32
		gpu_dx: [TOKENS * HEADS * HEAD_SIZE]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({TOKENS, HEADS * HEAD_SIZE})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({ROWS, TRAILING})
			ml.set_data(x, x_data[:])

			y := ml.slice_trailing(x, START, END)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}

		gpu_y:  [ROWS * (END - START)]f32
		gpu_dx: [ROWS * TRAILING]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({ROWS, TRAILING})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros({ROWS, T_A})
			b := ml.zeros({ROWS, T_B})
			c := ml.zeros({ROWS, T_C})
			ml.set_data(a, a_data[:])
			ml.set_data(b, b_data[:])
			ml.set_data(c, c_data[:])

			y := ml.concat(a, b, c)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_da[:], cpu.gradient(a))
			copy(cpu_db[:], cpu.gradient(b))
			copy(cpu_dc[:], cpu.gradient(c))
		}

		gpu_y:  [out_size]f32
		gpu_da: [ROWS * T_A]f32
		gpu_db: [ROWS * T_B]f32
		gpu_dc: [ROWS * T_C]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros({ROWS, T_A})
			b := ml.zeros({ROWS, T_B})
			c := ml.zeros({ROWS, T_C})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, SIZE})
			ml.set_data(x, x_data[:])

			y := ml.softmax(x)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}

		gpu_y:  [COUNT * SIZE]f32
		gpu_dx: [COUNT * SIZE]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, SIZE})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({D0, D1, D2})
			ml.set_data(x, x_data[:])

			y := ml.permute(x, {1, 0, 2})
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}

		gpu_y:  [D0 * D1 * D2]f32
		gpu_dx: [D0 * D1 * D2]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({D0, D1, D2})
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
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({H, T, T})
			ml.set_data(x, x_data[:])

			y := ml.causal_mask(x)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}

		gpu_y:  [H * T * T]f32
		gpu_dx: [H * T * T]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({H, T, T})
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

	// --- Phase 12: ml.mul with broadcast ---
	// Two sub-cases:
	//   12a) same-shape (stride==1): elementwise multiply.
	//   12b) broadcast (stride>1):   b is broadcast across the leading axis.
	{
		run_mul :: proc(a_data, b_data: []f32, cpu_out, cpu_da, cpu_db, gpu_out, gpu_da, gpu_db: []f32) {
			{
				ctx := cpu.context_create(2 * 1024 * 1024)
				defer cpu.context_destroy(ctx)
				ml.context_scope(ctx)

				a := ml.tensor(a_data)
				b := ml.tensor(b_data)
				c := ml.mul(a, b)
				ml.backward()

				copy(cpu_out, cpu.data(c))
				copy(cpu_da,  cpu.gradient(a))
				copy(cpu_db,  cpu.gradient(b))
			}
			{
				ctx := gpu.context_create()
				defer gpu.context_destroy(ctx)
				ml.context_scope(ctx)

				a := ml.zeros({len(a_data)})
				b := ml.zeros({len(b_data)})
				gpu.upload_tensor(a, a_data)
				gpu.upload_tensor(b, b_data)

				c := ml.mul(a, b)
				ml.backward()

				gpu.download_tensor          (c, gpu_out)
				gpu.download_tensor_gradient (a, gpu_da)
				gpu.download_tensor_gradient (b, gpu_db)
			}
		}

		// 12a: same-shape elementwise.
		{
			N :: 1024
			a_data := [N]f32{}
			b_data := [N]f32{}
			for i in 0 ..< N {
				a_data[i] = f32(i % 41) * 0.01 - 0.2
				b_data[i] = f32((i * 7) % 53) * 0.005 + 0.1
			}
			cpu_out, cpu_da, cpu_db: [N]f32
			gpu_out, gpu_da, gpu_db: [N]f32
			run_mul(a_data[:], b_data[:], cpu_out[:], cpu_da[:], cpu_db[:], gpu_out[:], gpu_da[:], gpu_db[:])

			TOL :: f32(1e-7)
			max_y, max_da, max_db: f32
			for i in 0 ..< N {
				max_y  = math.max(max_y,  math.abs(cpu_out[i] - gpu_out[i]))
				max_da = math.max(max_da, math.abs(cpu_da [i] - gpu_da [i]))
				max_db = math.max(max_db, math.abs(cpu_db [i] - gpu_db [i]))
			}
			fmt.printfln("phase12a: max abs error  y=%v  da=%v  db=%v", max_y, max_da, max_db)
			check(max_y  <= TOL, "phase12a: ml.mul (same-shape) output matches CPU",       &any_failed)
			check(max_da <= TOL, "phase12a: ml.mul (same-shape) backward da matches CPU",  &any_failed)
			check(max_db <= TOL, "phase12a: ml.mul (same-shape) backward db matches CPU",  &any_failed)
		}

		// 12b: broadcast (stride > 1). a length = STRIDE * N_B; b length = N_B.
		{
			STRIDE :: 32
			N_B    :: 48
			N      :: STRIDE * N_B
			a_data := [N]f32{}
			b_data := [N_B]f32{}
			for i in 0 ..< N   do a_data[i] = f32((i * 13) % 67) * 0.013 - 0.3
			for i in 0 ..< N_B do b_data[i] = f32((i * 5)  % 29) * 0.04  + 0.05
			cpu_out, gpu_out: [N]f32
			cpu_da,  gpu_da:  [N]f32
			cpu_db,  gpu_db:  [N_B]f32
			run_mul(a_data[:], b_data[:], cpu_out[:], cpu_da[:], cpu_db[:], gpu_out[:], gpu_da[:], gpu_db[:])

			// b.grad reduces over STRIDE elements, so allow a touch of fp32
			// reduction-order drift. a.grad is per-element (no reduction).
			TOL_Y  :: f32(1e-7)
			TOL_DA :: f32(1e-7)
			TOL_DB :: f32(1e-5)
			max_y, max_da, max_db: f32
			for i in 0 ..< N   {
				max_y  = math.max(max_y,  math.abs(cpu_out[i] - gpu_out[i]))
				max_da = math.max(max_da, math.abs(cpu_da [i] - gpu_da [i]))
			}
			for i in 0 ..< N_B do max_db = math.max(max_db, math.abs(cpu_db[i] - gpu_db[i]))
			fmt.printfln("phase12b: max abs error  y=%v  da=%v  db=%v", max_y, max_da, max_db)
			check(max_y  <= TOL_Y,  "phase12b: ml.mul (broadcast) output matches CPU",      &any_failed)
			check(max_da <= TOL_DA, "phase12b: ml.mul (broadcast) backward da matches CPU", &any_failed)
			check(max_db <= TOL_DB, "phase12b: ml.mul (broadcast) backward db matches CPU", &any_failed)
		}
	}

	// --- Phase 13: ml.batched_matmul ---
	{
		B :: 4
		M :: 12
		K :: 20
		N :: 16

		a_data := [B * M * K]f32{}
		b_data := [B * K * N]f32{}
		for i in 0 ..< len(a_data) do a_data[i] = f32((i * 13) % 67) * 0.011 - 0.3
		for i in 0 ..< len(b_data) do b_data[i] = f32((i * 7)  % 53) * 0.013 - 0.25

		cpu_y:  [B * M * N]f32
		cpu_da: [B * M * K]f32
		cpu_db: [B * K * N]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros({B, M, K})
			b := ml.zeros({B, K, N})
			ml.set_data(a, a_data[:])
			ml.set_data(b, b_data[:])

			y := ml.batched_matmul(a, b)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_da[:], cpu.gradient(a))
			copy(cpu_db[:], cpu.gradient(b))
		}

		gpu_y:  [B * M * N]f32
		gpu_da: [B * M * K]f32
		gpu_db: [B * K * N]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			a := ml.zeros({B, M, K})
			b := ml.zeros({B, K, N})
			gpu.upload_tensor(a, a_data[:])
			gpu.upload_tensor(b, b_data[:])

			y := ml.batched_matmul(a, b)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (a, gpu_da[:])
			gpu.download_tensor_gradient (b, gpu_db[:])
		}

		// Reductions over k / j / i — fp32 reduction-order drift. Same bar
		// as `linear`.
		TOL :: f32(1e-4)
		max_y, max_da, max_db: f32
		for i in 0 ..< len(cpu_y)  do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_da) do max_da = math.max(max_da, math.abs(cpu_da[i] - gpu_da[i]))
		for i in 0 ..< len(cpu_db) do max_db = math.max(max_db, math.abs(cpu_db[i] - gpu_db[i]))
		fmt.printfln("phase13: max abs error  y=%v  da=%v  db=%v", max_y, max_da, max_db)
		check(max_y  <= TOL, "phase13: ml.batched_matmul output matches CPU",       &any_failed)
		check(max_da <= TOL, "phase13: ml.batched_matmul backward da matches CPU",  &any_failed)
		check(max_db <= TOL, "phase13: ml.batched_matmul backward db matches CPU",  &any_failed)
	}

	// --- Phase 14: end-to-end ml.attention ---
	// Decomposes into slice_trailing → reshape → permute → batched_matmul →
	// mul (broadcast scalar) → causal_mask → softmax → batched_matmul →
	// permute → reshape. Exercises every op that's been ported plus the
	// set_data hook (used by `scalar(1/sqrt(D))`).
	{
		TOKENS    :: 16
		HEADS     :: 4
		HEAD_SIZE :: 8
		EMBED     :: HEADS * HEAD_SIZE
		IN_SIZE   :: 3 * EMBED

		x_data := [TOKENS * IN_SIZE]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 13 + 5) % 71) * 0.013 - 0.45

		cpu_y:  [TOKENS * EMBED]f32
		cpu_dx: [TOKENS * IN_SIZE]f32
		{
			ctx := cpu.context_create(4 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.tensor(x_data[:])
			x = ml.reshape(x, {TOKENS, IN_SIZE})

			y := ml.attention(x, HEADS)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}

		gpu_y:  [TOKENS * EMBED]f32
		gpu_dx: [TOKENS * IN_SIZE]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({TOKENS, IN_SIZE})
			gpu.upload_tensor(x, x_data[:])

			y := ml.attention(x, HEADS)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		// attention chains many reductions (two batched_matmuls + softmax
		// + the input-grad sum across the slice_trailing concat boundaries).
		// fp32 reduction-order drift is real here; same bar as `linear`.
		TOL :: f32(1e-4)
		max_y, max_dx: f32
		for i in 0 ..< len(cpu_y)  do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase14: max abs error  y=%v  dx=%v", max_y, max_dx)
		check(max_y  <= TOL, "phase14: ml.attention output matches CPU",      &any_failed)
		check(max_dx <= TOL, "phase14: ml.attention backward dx matches CPU", &any_failed)
	}

	// --- Phase 15: ml.layernorm ---
	{
		COUNT :: 16
		SIZE  :: 64

		x_data := [COUNT * SIZE]f32{}
		w_data := [SIZE]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 11) % 73) * 0.027 - 0.9
		for i in 0 ..< len(w_data) do w_data[i] = f32((i * 5)  % 19) * 0.05  + 0.5

		cpu_y:  [COUNT * SIZE]f32
		cpu_dx: [COUNT * SIZE]f32
		cpu_dw: [SIZE]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, SIZE})
			w := ml.zeros({SIZE})
			ml.set_data(x, x_data[:])
			ml.set_data(w, w_data[:])

			y := ml.layernorm(x, w)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
			copy(cpu_dw[:], cpu.gradient(w))
		}

		gpu_y:  [COUNT * SIZE]f32
		gpu_dx: [COUNT * SIZE]f32
		gpu_dw: [SIZE]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, SIZE})
			w := ml.zeros({SIZE})
			gpu.upload_tensor(x, x_data[:])
			gpu.upload_tensor(w, w_data[:])

			y := ml.layernorm(x, w)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
			gpu.download_tensor_gradient (w, gpu_dw[:])
		}

		// Mean / variance / dnorm reductions per row plus the dW reduction
		// across rows — fp32 reduction-order drift.
		TOL :: f32(1e-4)
		max_y, max_dx, max_dw: f32
		for i in 0 ..< len(cpu_y)  do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
		for i in 0 ..< len(cpu_dw) do max_dw = math.max(max_dw, math.abs(cpu_dw[i] - gpu_dw[i]))
		fmt.printfln("phase15: max abs error  y=%v  dx=%v  dw=%v", max_y, max_dx, max_dw)
		check(max_y  <= TOL, "phase15: ml.layernorm output matches CPU",       &any_failed)
		check(max_dx <= TOL, "phase15: ml.layernorm backward dx matches CPU",  &any_failed)
		check(max_dw <= TOL, "phase15: ml.layernorm backward dw matches CPU",  &any_failed)
	}

	// --- Phase 16: ml.cross_entropy ---
	{
		COUNT      :: 16
		CLASS_SIZE :: 64

		x_data := [COUNT * CLASS_SIZE]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 11) % 89) * 0.04 - 1.6
		targets := [COUNT]int{}
		for i in 0 ..< COUNT do targets[i] = (i * 13 + 7) % CLASS_SIZE

		cpu_loss: [COUNT]f32
		cpu_dx:   [COUNT * CLASS_SIZE]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, CLASS_SIZE})
			ml.set_data(x, x_data[:])

			loss := ml.cross_entropy(x, targets[:])
			ml.backward()

			copy(cpu_loss[:], cpu.data(loss))
			copy(cpu_dx[:],   cpu.gradient(x))
		}

		gpu_loss: [COUNT]f32
		gpu_dx:   [COUNT * CLASS_SIZE]f32
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, CLASS_SIZE})
			gpu.upload_tensor(x, x_data[:])

			loss := ml.cross_entropy(x, targets[:])
			ml.backward()

			gpu.download_tensor          (loss, gpu_loss[:])
			gpu.download_tensor_gradient (x,    gpu_dx[:])
		}

		// Per-row max + sum reductions; tolerate fp32 reduction-order drift.
		TOL :: f32(1e-5)
		max_loss, max_dx: f32
		for i in 0 ..< COUNT             do max_loss = math.max(max_loss, math.abs(cpu_loss[i] - gpu_loss[i]))
		for i in 0 ..< COUNT * CLASS_SIZE do max_dx  = math.max(max_dx,   math.abs(cpu_dx[i]   - gpu_dx[i]))
		fmt.printfln("phase16: max abs error  loss=%v  dx=%v", max_loss, max_dx)
		check(max_loss <= TOL, "phase16: ml.cross_entropy output matches CPU",      &any_failed)
		check(max_dx   <= TOL, "phase16: ml.cross_entropy backward dx matches CPU", &any_failed)
	}

	// --- Phase 17: ml.mean ---
	{
		COUNT :: 16
		SIZE  :: 64
		x_data := [COUNT * SIZE]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 13) % 71) * 0.025 - 0.7

		cpu_y, gpu_y:   [COUNT]f32
		cpu_dx, gpu_dx: [COUNT * SIZE]f32

		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, SIZE})
			ml.set_data(x, x_data[:])

			y := ml.mean(x)
			ml.backward()

			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)

			x := ml.zeros({COUNT, SIZE})
			gpu.upload_tensor(x, x_data[:])

			y := ml.mean(x)
			ml.backward()

			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}

		TOL :: f32(1e-6)
		max_y, max_dx: f32
		for i in 0 ..< COUNT       do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< COUNT*SIZE  do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase17: max abs error  y=%v  dx=%v", max_y, max_dx)
		check(max_y  <= TOL, "phase17: ml.mean output matches CPU",      &any_failed)
		check(max_dx <= TOL, "phase17: ml.mean backward dx matches CPU", &any_failed)
	}

	// --- Phase 18: parameter training loop (CPU vs GPU) ---
	// 3 Adam steps on a tiny linear -> gelu -> linear -> mean(cross_entropy)
	// network. After each step we read both backends' parameter values and
	// confirm they agree at the fp32 reduction floor.
	{
		IN_DIM   :: 16
		HIDDEN   :: 24
		OUT_DIM  :: 8
		BATCH    :: 4
		STEPS    :: 3

		x_data := [BATCH * IN_DIM]f32{}
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 7) % 41) * 0.03 - 0.4
		w0_init := [HIDDEN * IN_DIM]f32{}
		for i in 0 ..< len(w0_init) do w0_init[i] = f32((i * 11) % 53) * 0.02 - 0.3
		w1_init := [OUT_DIM * HIDDEN]f32{}
		for i in 0 ..< len(w1_init) do w1_init[i] = f32((i * 17) % 47) * 0.025 - 0.4
		targets := [BATCH]int{}
		for i in 0 ..< BATCH do targets[i] = (i * 3 + 1) % OUT_DIM

		run :: proc(use_gpu: bool, x_data: []f32, w0_init, w1_init: []f32, targets: []int,
			out_w0, out_w1: []f32) {
			arena_size := 16 * 1024 * 1024
			ctx := use_gpu ? gpu.context_create() : cpu.context_create(arena_size)
			defer { if use_gpu { gpu.context_destroy(ctx) } else { cpu.context_destroy(ctx) } }
			ml.context_scope(ctx)

			// Persistent parameters survive ml.clear; x lives in the per-step
			// arena and is re-seeded each iteration.
			w0, _ := ml.make({HIDDEN, IN_DIM})
			defer ml.destroy(w0)
			w1, _ := ml.make({OUT_DIM, HIDDEN})
			defer ml.destroy(w1)

			ml.set_data(w0, w0_init)
			ml.set_data(w1, w1_init)

			opt: ml.Optimizer
			for step in 0 ..< STEPS {
				ml.clear()

				x := ml.zeros({BATCH, IN_DIM})
				ml.set_data(x, x_data)

				h := ml.linear(x, w0)
				h  = ml.gelu(h)
				y := ml.linear(h, w1)
				loss := ml.cross_entropy(y, targets)
				loss  = ml.mean(loss)

				ml.backward()
				if ml.optimize(&opt, period=1, learning_rate=0.01) {
					ml.update(opt, w0)
					ml.update(opt, w1)
				}
			}

			ml.get_data(w0, out_w0)
			ml.get_data(w1, out_w1)
		}

		cpu_w0, gpu_w0: [HIDDEN * IN_DIM]f32
		cpu_w1, gpu_w1: [OUT_DIM * HIDDEN]f32

		run(false, x_data[:], w0_init[:], w1_init[:], targets[:], cpu_w0[:], cpu_w1[:])
		run(true,  x_data[:], w0_init[:], w1_init[:], targets[:], gpu_w0[:], gpu_w1[:])

		// Adam amplifies tiny grad-order differences via 1/(sqrt(v)+eps),
		// so accept fp32 floor across a few steps.
		TOL :: f32(1e-4)
		max_w0, max_w1: f32
		for i in 0 ..< len(cpu_w0) do max_w0 = math.max(max_w0, math.abs(cpu_w0[i] - gpu_w0[i]))
		for i in 0 ..< len(cpu_w1) do max_w1 = math.max(max_w1, math.abs(cpu_w1[i] - gpu_w1[i]))
		fmt.printfln("phase18: max abs error after %v Adam steps  w0=%v  w1=%v", STEPS, max_w0, max_w1)
		check(max_w0 <= TOL, "phase18: GPU w0 matches CPU after training", &any_failed)
		check(max_w1 <= TOL, "phase18: GPU w1 matches CPU after training", &any_failed)
	}

	// --- Phase 19: elementwise activations (relu, sigmoid, silu, tanh, exp, clamp) ---
	{
		N :: 4096
		x_data: [N]f32
		for i in 0 ..< N do x_data[i] = f32((i * 13) % 91) * 0.04 - 1.8

		check_unary :: proc(name: string, x_data: []f32, fn: proc(t: ml.Tensor) -> ml.Tensor, tol: f32, any_failed: ^bool) {
			n := len(x_data)
			cpu_y  := make([]f32, n, context.temp_allocator)
			cpu_dx := make([]f32, n, context.temp_allocator)
			gpu_y  := make([]f32, n, context.temp_allocator)
			gpu_dx := make([]f32, n, context.temp_allocator)

			{
				ctx := cpu.context_create(2 * 1024 * 1024)
				defer cpu.context_destroy(ctx)
				ml.context_scope(ctx)
				x := ml.tensor(x_data)
				y := fn(x)
				ml.backward()
				copy(cpu_y,  cpu.data(y))
				copy(cpu_dx, cpu.gradient(x))
			}
			{
				ctx := gpu.context_create()
				defer gpu.context_destroy(ctx)
				ml.context_scope(ctx)
				x := ml.zeros({n})
				gpu.upload_tensor(x, x_data)
				y := fn(x)
				ml.backward()
				gpu.download_tensor          (y, gpu_y)
				gpu.download_tensor_gradient (x, gpu_dx)
			}

			max_y, max_dx: f32
			for i in 0 ..< n do max_y  = math.max(max_y,  math.abs(cpu_y [i] - gpu_y [i]))
			for i in 0 ..< n do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
			fmt.printfln("phase19/%v: y=%v  dx=%v", name, max_y, max_dx)
			check(max_y  <= tol, fmt.tprintf("phase19: ml.%v output matches CPU",      name), any_failed)
			check(max_dx <= tol, fmt.tprintf("phase19: ml.%v backward dx matches CPU", name), any_failed)
		}

		// Wrappers that match the simple `(Tensor) -> Tensor` shape (the ml.* procs take a default loc parameter, which makes the type incompatible with our proc-pointer field).
		relu_w    :: proc(t: ml.Tensor) -> ml.Tensor { return ml.relu(t)    }
		sigmoid_w :: proc(t: ml.Tensor) -> ml.Tensor { return ml.sigmoid(t) }
		silu_w    :: proc(t: ml.Tensor) -> ml.Tensor { return ml.silu(t)    }
		tanh_w    :: proc(t: ml.Tensor) -> ml.Tensor { return ml.tanh(t)    }
		exp_w     :: proc(t: ml.Tensor) -> ml.Tensor { return ml.exp(t)     }
		clamp_w   :: proc(t: ml.Tensor) -> ml.Tensor { return ml.clamp(t, -0.5, 0.5) }

		check_unary("relu",    x_data[:], relu_w,    1e-7, &any_failed)
		check_unary("sigmoid", x_data[:], sigmoid_w, 2e-6, &any_failed)
		check_unary("silu",    x_data[:], silu_w,    2e-6, &any_failed)
		check_unary("tanh",    x_data[:], tanh_w,    2e-6, &any_failed)
		check_unary("exp",     x_data[:], exp_w,     2e-5, &any_failed) // exp blows up the abs error
		check_unary("clamp",   x_data[:], clamp_w,   1e-7, &any_failed)
	}

	// --- Phase 20: same-shape min / max ---
	{
		N :: 1024
		a_data, b_data: [N]f32
		for i in 0 ..< N {
			a_data[i] = f32((i * 7)  % 53) * 0.04 - 1.0
			b_data[i] = f32((i * 11) % 47) * 0.04 - 1.0
		}

		check_binary :: proc(name: string, a_data, b_data: []f32, fn: proc(a, b: ml.Tensor) -> ml.Tensor, any_failed: ^bool) {
			n := len(a_data)
			cpu_y, cpu_da, cpu_db := make([]f32, n, context.temp_allocator), make([]f32, n, context.temp_allocator), make([]f32, n, context.temp_allocator)
			gpu_y, gpu_da, gpu_db := make([]f32, n, context.temp_allocator), make([]f32, n, context.temp_allocator), make([]f32, n, context.temp_allocator)

			{
				ctx := cpu.context_create(2 * 1024 * 1024)
				defer cpu.context_destroy(ctx)
				ml.context_scope(ctx)
				a := ml.tensor(a_data)
				b := ml.tensor(b_data)
				y := fn(a, b)
				ml.backward()
				copy(cpu_y,  cpu.data(y))
				copy(cpu_da, cpu.gradient(a))
				copy(cpu_db, cpu.gradient(b))
			}
			{
				ctx := gpu.context_create()
				defer gpu.context_destroy(ctx)
				ml.context_scope(ctx)
				a := ml.zeros({n}); b := ml.zeros({n})
				gpu.upload_tensor(a, a_data)
				gpu.upload_tensor(b, b_data)
				y := fn(a, b)
				ml.backward()
				gpu.download_tensor          (y, gpu_y)
				gpu.download_tensor_gradient (a, gpu_da)
				gpu.download_tensor_gradient (b, gpu_db)
			}

			TOL :: f32(1e-7)
			max_y, max_da, max_db: f32
			for i in 0 ..< n {
				max_y  = math.max(max_y,  math.abs(cpu_y [i] - gpu_y [i]))
				max_da = math.max(max_da, math.abs(cpu_da[i] - gpu_da[i]))
				max_db = math.max(max_db, math.abs(cpu_db[i] - gpu_db[i]))
			}
			fmt.printfln("phase20/%v: y=%v da=%v db=%v", name, max_y, max_da, max_db)
			check(max_y  <= TOL, fmt.tprintf("phase20: ml.%v output matches CPU",      name), any_failed)
			check(max_da <= TOL, fmt.tprintf("phase20: ml.%v backward da matches CPU", name), any_failed)
			check(max_db <= TOL, fmt.tprintf("phase20: ml.%v backward db matches CPU", name), any_failed)
		}

		min_w :: proc(a, b: ml.Tensor) -> ml.Tensor { return ml.min(a, b) }
		max_w :: proc(a, b: ml.Tensor) -> ml.Tensor { return ml.max(a, b) }
		check_binary("min", a_data[:], b_data[:], min_w, &any_failed)
		check_binary("max", a_data[:], b_data[:], max_w, &any_failed)
	}

	// --- Phase 21: broadcast sub / div ---
	{
		STRIDE :: 32
		N_B    :: 48
		N      :: STRIDE * N_B
		a_data: [N]f32
		b_data: [N_B]f32
		for i in 0 ..< N   do a_data[i] = f32((i * 13) % 67) * 0.013 - 0.3
		for i in 0 ..< N_B do b_data[i] = f32((i * 5)  % 29) * 0.04  + 0.2 // keep b > 0 so div is well-defined

		check_broadcast :: proc(name: string, a_data, b_data: []f32, fn: proc(a, b: ml.Tensor) -> ml.Tensor, tol_db: f32, any_failed: ^bool) {
			n := len(a_data); n_b := len(b_data)
			cpu_y,  gpu_y  := make([]f32, n,   context.temp_allocator), make([]f32, n,   context.temp_allocator)
			cpu_da, gpu_da := make([]f32, n,   context.temp_allocator), make([]f32, n,   context.temp_allocator)
			cpu_db, gpu_db := make([]f32, n_b, context.temp_allocator), make([]f32, n_b, context.temp_allocator)

			{
				ctx := cpu.context_create(2 * 1024 * 1024)
				defer cpu.context_destroy(ctx)
				ml.context_scope(ctx)
				a := ml.tensor(a_data); b := ml.tensor(b_data)
				y := fn(a, b)
				ml.backward()
				copy(cpu_y,  cpu.data(y))
				copy(cpu_da, cpu.gradient(a))
				copy(cpu_db, cpu.gradient(b))
			}
			{
				ctx := gpu.context_create()
				defer gpu.context_destroy(ctx)
				ml.context_scope(ctx)
				a := ml.zeros({n}); b := ml.zeros({n_b})
				gpu.upload_tensor(a, a_data); gpu.upload_tensor(b, b_data)
				y := fn(a, b)
				ml.backward()
				gpu.download_tensor          (y, gpu_y)
				gpu.download_tensor_gradient (a, gpu_da)
				gpu.download_tensor_gradient (b, gpu_db)
			}

			TOL_Y_DA :: f32(1e-6)
			max_y, max_da, max_db: f32
			for i in 0 ..< n {
				max_y  = math.max(max_y,  math.abs(cpu_y [i] - gpu_y [i]))
				max_da = math.max(max_da, math.abs(cpu_da[i] - gpu_da[i]))
			}
			for i in 0 ..< n_b do max_db = math.max(max_db, math.abs(cpu_db[i] - gpu_db[i]))
			fmt.printfln("phase21/%v: y=%v da=%v db=%v", name, max_y, max_da, max_db)
			check(max_y  <= TOL_Y_DA, fmt.tprintf("phase21: ml.%v output matches CPU",      name), any_failed)
			check(max_da <= TOL_Y_DA, fmt.tprintf("phase21: ml.%v backward da matches CPU", name), any_failed)
			check(max_db <= tol_db,   fmt.tprintf("phase21: ml.%v backward db matches CPU", name), any_failed)
		}

		add_w :: proc(a, b: ml.Tensor) -> ml.Tensor { return ml.add(a, b) }
		sub_w :: proc(a, b: ml.Tensor) -> ml.Tensor { return ml.sub(a, b) }
		div_w :: proc(a, b: ml.Tensor) -> ml.Tensor { return ml.div(a, b) }
		check_broadcast("add", a_data[:], b_data[:], add_w, 1e-5, &any_failed)
		check_broadcast("sub", a_data[:], b_data[:], sub_w, 1e-5, &any_failed)
		check_broadcast("div", a_data[:], b_data[:], div_w, 1e-3, &any_failed) // db has 1/b^2 which can amplify error
	}

	// --- Phase 22: ml.transpose ---
	{
		ROWS :: 16
		COLS :: 24
		x_data: [ROWS * COLS]f32
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 11) % 67) * 0.02 - 0.5

		cpu_y, gpu_y:   [COLS * ROWS]f32
		cpu_dx, gpu_dx: [ROWS * COLS]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)
			x := ml.zeros({ROWS, COLS}); ml.set_data(x, x_data[:])
			y := ml.transpose(x)
			ml.backward()
			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)
			x := ml.zeros({ROWS, COLS}); gpu.upload_tensor(x, x_data[:])
			y := ml.transpose(x)
			ml.backward()
			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}
		max_y, max_dx: f32
		for i in 0 ..< len(cpu_y)  do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase22: ml.transpose y=%v dx=%v", max_y, max_dx)
		check(max_y  <= 1e-7, "phase22: ml.transpose output matches CPU",      &any_failed)
		check(max_dx <= 1e-7, "phase22: ml.transpose backward dx matches CPU", &any_failed)
	}

	// --- Phase 23: ml.slice ---
	{
		N :: 256
		START :: 32
		END   :: 192
		x_data: [N]f32
		for i in 0 ..< N do x_data[i] = f32((i * 17) % 73) * 0.017 - 0.4

		cpu_y, gpu_y:   [END - START]f32
		cpu_dx, gpu_dx: [N]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)
			x := ml.tensor(x_data[:])
			y := ml.slice(x, START, END)
			ml.backward()
			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)
			x := ml.zeros({N}); gpu.upload_tensor(x, x_data[:])
			y := ml.slice(x, START, END)
			ml.backward()
			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}
		max_y, max_dx: f32
		for i in 0 ..< len(cpu_y)  do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase23: ml.slice y=%v dx=%v", max_y, max_dx)
		check(max_y  <= 1e-7, "phase23: ml.slice output matches CPU",      &any_failed)
		check(max_dx <= 1e-7, "phase23: ml.slice backward dx matches CPU", &any_failed)
	}

	// --- Phase 24: ml.log_softmax ---
	{
		COUNT :: 8
		SIZE  :: 64
		x_data: [COUNT * SIZE]f32
		for i in 0 ..< len(x_data) do x_data[i] = f32((i * 7) % 53) * 0.05 - 1.2

		cpu_y, gpu_y:   [COUNT * SIZE]f32
		cpu_dx, gpu_dx: [COUNT * SIZE]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)
			x := ml.zeros({COUNT, SIZE}); ml.set_data(x, x_data[:])
			y := ml.log_softmax(x)
			ml.backward()
			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(x))
		}
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)
			x := ml.zeros({COUNT, SIZE}); gpu.upload_tensor(x, x_data[:])
			y := ml.log_softmax(x)
			ml.backward()
			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (x, gpu_dx[:])
		}
		max_y, max_dx: f32
		for i in 0 ..< len(cpu_y)  do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase24: ml.log_softmax y=%v dx=%v", max_y, max_dx)
		check(max_y  <= 2e-6, "phase24: ml.log_softmax output matches CPU",      &any_failed)
		check(max_dx <= 2e-6, "phase24: ml.log_softmax backward dx matches CPU", &any_failed)
	}

	// --- Phase 25: ml.entropy ---
	// Build a valid probability distribution by softmaxing random logits, then
	// detach via tensor() so entropy receives genuine probabilities.
	{
		COUNT :: 8
		SIZE  :: 32
		// Generate probabilities via a deterministic softmax of fake logits.
		raw: [COUNT * SIZE]f32
		for i in 0 ..< len(raw) do raw[i] = f32((i * 13) % 41) * 0.05 - 0.7
		probs: [COUNT * SIZE]f32
		for r in 0 ..< COUNT {
			max_v := f32(-1e30)
			for c in 0 ..< SIZE do max_v = math.max(max_v, raw[r * SIZE + c])
			sum: f32
			for c in 0 ..< SIZE {
				probs[r * SIZE + c] = math.exp(raw[r * SIZE + c] - max_v)
				sum += probs[r * SIZE + c]
			}
			for c in 0 ..< SIZE do probs[r * SIZE + c] /= sum
		}

		cpu_y, gpu_y:   [COUNT]f32
		cpu_dx, gpu_dx: [COUNT * SIZE]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)
			p := ml.zeros({COUNT, SIZE}); ml.set_data(p, probs[:])
			y := ml.entropy(p)
			ml.backward()
			copy(cpu_y[:],  cpu.data(y))
			copy(cpu_dx[:], cpu.gradient(p))
		}
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)
			p := ml.zeros({COUNT, SIZE}); gpu.upload_tensor(p, probs[:])
			y := ml.entropy(p)
			ml.backward()
			gpu.download_tensor          (y, gpu_y[:])
			gpu.download_tensor_gradient (p, gpu_dx[:])
		}
		max_y, max_dx: f32
		for i in 0 ..< len(cpu_y)  do max_y  = math.max(max_y,  math.abs(cpu_y[i]  - gpu_y[i]))
		for i in 0 ..< len(cpu_dx) do max_dx = math.max(max_dx, math.abs(cpu_dx[i] - gpu_dx[i]))
		fmt.printfln("phase25: ml.entropy y=%v dx=%v", max_y, max_dx)
		check(max_y  <= 2e-6, "phase25: ml.entropy output matches CPU",      &any_failed)
		check(max_dx <= 2e-6, "phase25: ml.entropy backward dx matches CPU", &any_failed)
	}

	// --- Phase 26: ml.mean_squared_error ---
	{
		COUNT :: 8
		SIZE  :: 32
		pred_data:   [COUNT * SIZE]f32
		target_data: [COUNT * SIZE]f32
		for i in 0 ..< len(pred_data)   do pred_data[i]   = f32((i * 13) % 41) * 0.03 - 0.4
		for i in 0 ..< len(target_data) do target_data[i] = f32((i * 7)  % 31) * 0.04 - 0.5

		cpu_y, gpu_y:     [COUNT]f32
		cpu_dpred, gpu_dpred: [COUNT * SIZE]f32
		{
			ctx := cpu.context_create(2 * 1024 * 1024)
			defer cpu.context_destroy(ctx)
			ml.context_scope(ctx)
			pred   := ml.zeros({COUNT, SIZE}); ml.set_data(pred, pred_data[:])
			target := ml.zeros({COUNT, SIZE}); ml.set_data(target, target_data[:])
			y := ml.mean_squared_error(pred, target)
			ml.backward()
			copy(cpu_y[:],     cpu.data(y))
			copy(cpu_dpred[:], cpu.gradient(pred))
		}
		{
			ctx := gpu.context_create()
			defer gpu.context_destroy(ctx)
			ml.context_scope(ctx)
			pred   := ml.zeros({COUNT, SIZE}); gpu.upload_tensor(pred,   pred_data[:])
			target := ml.zeros({COUNT, SIZE}); gpu.upload_tensor(target, target_data[:])
			y := ml.mean_squared_error(pred, target)
			ml.backward()
			gpu.download_tensor          (y,    gpu_y[:])
			gpu.download_tensor_gradient (pred, gpu_dpred[:])
		}
		max_y, max_dpred: f32
		for i in 0 ..< len(cpu_y)     do max_y     = math.max(max_y,     math.abs(cpu_y    [i] - gpu_y    [i]))
		for i in 0 ..< len(cpu_dpred) do max_dpred = math.max(max_dpred, math.abs(cpu_dpred[i] - gpu_dpred[i]))
		fmt.printfln("phase26: ml.mean_squared_error y=%v dpred=%v", max_y, max_dpred)
		check(max_y     <= 1e-6, "phase26: ml.mean_squared_error output matches CPU",         &any_failed)
		check(max_dpred <= 1e-6, "phase26: ml.mean_squared_error backward dpred matches CPU", &any_failed)
	}

	if any_failed {
		os.exit(1)
	}
	fmt.println("OK: gpu backend integration is healthy")
}
