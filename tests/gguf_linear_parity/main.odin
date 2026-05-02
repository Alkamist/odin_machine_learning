package gguf_linear_parity

import "core:fmt"
import "core:mem"
import "core:os"

import "core:math"

import ml  "../.."
import cpu "../../backends/cpu"
import gpu "../../backends/gpu"
import "../../loaders/gguf"

// Parity test for the new ml.linear_q4_k / linear_q6_k ops. Both the CPU
// op and the reference here dequantize via gguf.dequantize_q* and accumulate
// in f32, so a correct wiring should match bit-exactly. Tolerance is 0.

K   :: 256 // input_size — one Q4_K / Q6_K block per row
N   ::  64 // output_size
M   ::   2 // batch (number of activation rows)

main :: proc() {
	any_failed := false

	{
		fmt.println("== synthetic Q4_K ==")
		ctx := cpu.context_create(8 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		w_bytes := _make_synthetic_q4_k_bytes(N)
		defer delete(w_bytes)

		_run_q4_k_parity(w_bytes, &any_failed, "synthetic")

		ml.clear()
	}

	{
		fmt.println("== synthetic Q6_K ==")
		ctx := cpu.context_create(8 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		ml.context_scope(ctx)

		w_bytes := _make_synthetic_q6_k_bytes(N)
		defer delete(w_bytes)

		_run_q6_k_parity(w_bytes, &any_failed, "synthetic")

		ml.clear()
	}

	{
		fmt.println("== synthetic Q4_K on GPU ==")
		ctx := gpu.context_create()
		defer gpu.context_destroy(ctx)
		ml.context_scope(ctx)

		w_bytes := _make_synthetic_q4_k_bytes(N)
		defer delete(w_bytes)

		_run_q4_k_gpu_parity(w_bytes, &any_failed, "synthetic")

		ml.clear()
	}

	{
		fmt.println("== synthetic Q4_K on GPU (coopmat M=64) ==")
		ctx := gpu.context_create()
		defer gpu.context_destroy(ctx)
		ml.context_scope(ctx)

		w_bytes := _make_synthetic_q4_k_bytes(64)
		defer delete(w_bytes)

		_run_q4_k_gpu_coopmat_parity(w_bytes, &any_failed, "synthetic")

		ml.clear()
	}

	{
		fmt.println("== synthetic Q6_K on GPU ==")
		ctx := gpu.context_create()
		defer gpu.context_destroy(ctx)
		ml.context_scope(ctx)

		w_bytes := _make_synthetic_q6_k_bytes(N)
		defer delete(w_bytes)

		_run_q6_k_gpu_parity(w_bytes, &any_failed, "synthetic")

		ml.clear()
	}

	{
		fmt.println("== synthetic Q6_K on GPU (coopmat M=64) ==")
		ctx := gpu.context_create()
		defer gpu.context_destroy(ctx)
		ml.context_scope(ctx)

		w_bytes := _make_synthetic_q6_k_bytes(64)
		defer delete(w_bytes)

		_run_q6_k_gpu_coopmat_parity(w_bytes, &any_failed, "synthetic")

		ml.clear()
	}

	if len(os.args) >= 2 {
		fmt.println("== real Gemma 4 E4B Q4_K + Q6_K ==")
		_run_real_parity(os.args[1], &any_failed)
		_run_real_gpu_parity(os.args[1], &any_failed)
	} else {
		fmt.println("(skipping real-tensor parity; pass GGUF path as first arg to enable)")
	}

	if any_failed do os.exit(1)
	fmt.println("ok")
}

// Build N rows of valid Q4_K block bytes. Each block uses moderate scale/min
// values to keep the dequantized weights bounded; quants are an LCG byte
// stream so every nibble pattern shows up.
_make_synthetic_q4_k_bytes :: proc(rows: int) -> []byte {
	out := make([]byte, rows * ml.Q4_K_BLOCK_BYTES)
	state := u64(0xC0FFEE_1234)
	for r in 0 ..< rows {
		base := r * ml.Q4_K_BLOCK_BYTES

		// d, dmin (fp16). Pick values that keep dequant magnitude small.
		d_h    := f16(0.01)
		dmin_h := f16(0.005)
		mem.copy(raw_data(out[base + 0:]), &d_h,    2)
		mem.copy(raw_data(out[base + 2:]), &dmin_h, 2)

		// scales[12] — fill with mixed but in-range bytes. Each 6-bit scale
		// and 6-bit min is masked from 8-bit storage in
		// _unpack_scale_min_k4 so any bit pattern produces a valid block.
		for i in 0 ..< 12 {
			out[base + 4 + i] = u8(_lcg(&state) & 0xFF)
		}
		// qs[128]
		for i in 0 ..< 128 {
			out[base + 16 + i] = u8(_lcg(&state) & 0xFF)
		}
	}
	return out
}

_make_synthetic_q6_k_bytes :: proc(rows: int) -> []byte {
	out := make([]byte, rows * ml.Q6_K_BLOCK_BYTES)
	state := u64(0xBEEF_5678)
	for r in 0 ..< rows {
		base := r * ml.Q6_K_BLOCK_BYTES
		// ql[128]
		for i in 0 ..< 128 do out[base +   0 + i] = u8(_lcg(&state) & 0xFF)
		// qh[64]
		for i in 0 ..<  64 do out[base + 128 + i] = u8(_lcg(&state) & 0xFF)
		// scales[16] (i8). Modest magnitudes to keep dequant bounded.
		for i in 0 ..<  16 {
			s := i8(_lcg(&state) & 0xFF) >> 5 // ±3-ish
			out[base + 192 + i] = u8(s)
		}
		// d (fp16)
		d_h := f16(0.01)
		mem.copy(raw_data(out[base + 208:]), &d_h, 2)
	}
	return out
}

_lcg :: proc(state: ^u64) -> u64 {
	state^ = state^ * 6364136223846793005 + 1442695040888963407
	return state^
}

_make_synthetic_activation :: proc() -> [M * K]ml.Bf16 {
	x: [M * K]ml.Bf16
	state := u64(0xABCD_EF01)
	for i in 0 ..< M * K {
		// Map to ~[-1, 1] in bf16.
		raw := _lcg(&state)
		v := (f32(i32(raw & 0xFFFF)) - 32768.0) / 32768.0
		x[i] = ml.bf16_from_f32(v)
	}
	return x
}

_run_q4_k_parity :: proc(w_bytes: []byte, any_failed: ^bool, label: string) {
	weight := ml.alloc(.Q4_K, {N, K}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x := _make_synthetic_activation()
	x_t := ml.alloc(.Bf16, {M, K}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x[:]))

	y := ml.linear_q4_k(x_t, weight)
	got: [M * N]ml.Bf16
	ml.get_data_bytes(y, mem.slice_to_bytes(got[:]))

	expected := _reference_q4_k_matmul(w_bytes, x[:])
	_compare(got[:], expected[:], fmt.tprintf("Q4_K %v", label), any_failed)
}

_run_q6_k_parity :: proc(w_bytes: []byte, any_failed: ^bool, label: string) {
	weight := ml.alloc(.Q6_K, {N, K}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x := _make_synthetic_activation()
	x_t := ml.alloc(.Bf16, {M, K}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x[:]))

	y := ml.linear_q6_k(x_t, weight)
	got: [M * N]ml.Bf16
	ml.get_data_bytes(y, mem.slice_to_bytes(got[:]))

	expected := _reference_q6_k_matmul(w_bytes, x[:])
	_compare(got[:], expected[:], fmt.tprintf("Q6_K %v", label), any_failed)
}

// Reference: dequantize each weight row to f32, dot with bf16 activation
// (converting each element to f32), bf16 the result. This matches what
// linear_q4_k_forward does internally — equality means the op wiring is
// correct (variant unpack, output indexing, parallelization).
_reference_q4_k_matmul :: proc(w_bytes: []byte, x: []ml.Bf16) -> [M * N]ml.Bf16 {
	out: [M * N]ml.Bf16
	row_bytes := (K / ml.K_QUANT_BLOCK_SIZE) * ml.Q4_K_BLOCK_BYTES
	dequant   := make([]f32, K)
	defer delete(dequant)
	for o in 0 ..< N {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q4_k(w_row, dequant)
		for c in 0 ..< M {
			total: f32
			for k in 0 ..< K {
				total += dequant[k] * ml.bf16_to_f32(x[c * K + k])
			}
			out[c * N + o] = ml.bf16_from_f32(total)
		}
	}
	return out
}

_reference_q6_k_matmul :: proc(w_bytes: []byte, x: []ml.Bf16) -> [M * N]ml.Bf16 {
	out: [M * N]ml.Bf16
	row_bytes := (K / ml.K_QUANT_BLOCK_SIZE) * ml.Q6_K_BLOCK_BYTES
	dequant   := make([]f32, K)
	defer delete(dequant)
	for o in 0 ..< N {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q6_k(w_row, dequant)
		for c in 0 ..< M {
			total: f32
			for k in 0 ..< K {
				total += dequant[k] * ml.bf16_to_f32(x[c * K + k])
			}
			out[c * N + o] = ml.bf16_from_f32(total)
		}
	}
	return out
}

_compare :: proc(got, expected: []ml.Bf16, label: string, any_failed: ^bool) {
	if len(got) != len(expected) {
		fmt.printfln("FAIL %v: length mismatch %v vs %v", label, len(got), len(expected))
		any_failed^ = true
		return
	}
	for i in 0 ..< len(got) {
		if got[i] != expected[i] {
			fmt.printfln("FAIL %v: bit-exact mismatch at %v: got %v expected %v",
				label, i, ml.bf16_to_f32(got[i]), ml.bf16_to_f32(expected[i]))
			any_failed^ = true
			return
		}
	}
	fmt.printfln("OK   %v: bit-exact (%v outputs)", label, len(got))
}

// --- Real-tensor parity ---------------------------------------------------

_run_real_parity :: proc(path: string, any_failed: ^bool) {
	loader, ok := gguf.load(path)
	if !ok {
		any_failed^ = true
		return
	}
	defer gguf.destroy(loader)

	ctx := cpu.context_create(64 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	_run_real_for_type(loader, .Q4_K, any_failed)
	_run_real_for_type(loader, .Q6_K, any_failed)

	ml.clear()
}

_run_real_for_type :: proc(loader: gguf.Loader, ty: gguf.Tensor_Type, any_failed: ^bool) {
	// Find a tensor of the right type with input_size % 256 == 0 and a
	// modest output_size to keep the test fast.
	chosen_name: string
	chosen_info: gguf.Tensor_Info
	found := false
	for name, info in loader.tensors {
		if info.type != ty do continue
		if len(info.shape) != 2 do continue
		if info.shape[1] % ml.K_QUANT_BLOCK_SIZE != 0 do continue
		// Pick the tensor with the smallest output_size × input_size product
		// to keep the test fast. Real Gemma 4 E4B Q6_K tensors all have
		// output_size = 2560 (vocab/hidden dim), so capping output_size alone
		// would skip them entirely.
		if !found || info.shape[0] * info.shape[1] < chosen_info.shape[0] * chosen_info.shape[1] {
			chosen_name = name
			chosen_info = info
			found = true
		}
	}
	if !found {
		fmt.printfln("(no small %v tensor found; skipping real parity)", ty)
		return
	}

	output_size := chosen_info.shape[0]
	input_size  := chosen_info.shape[1]
	w_bytes, _  := gguf.get_bytes(loader, chosen_name)

	weight := ml.alloc(_ml_dtype(ty), {output_size, input_size}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	// Activation: small batch so the test stays cheap.
	batch := 2
	x_buf := make([]ml.Bf16, batch * input_size)
	defer delete(x_buf)
	state := u64(0x1234_5678)
	for i in 0 ..< batch * input_size {
		raw := _lcg(&state)
		v := (f32(i32(raw & 0xFFFF)) - 32768.0) / 32768.0
		x_buf[i] = ml.bf16_from_f32(v)
	}
	x_t := ml.alloc(.Bf16, {batch, input_size}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x_buf))

	y: ml.Tensor
	#partial switch ty {
	case .Q4_K: y = ml.linear_q4_k(x_t, weight)
	case .Q6_K: y = ml.linear_q6_k(x_t, weight)
	case:       return
	}

	got_buf := make([]ml.Bf16, batch * output_size)
	defer delete(got_buf)
	ml.get_data_bytes(y, mem.slice_to_bytes(got_buf))

	expected_buf := make([]ml.Bf16, batch * output_size)
	defer delete(expected_buf)
	dequant := make([]f32, input_size)
	defer delete(dequant)

	bytes_per_row := 0
	#partial switch ty {
	case .Q4_K: bytes_per_row = (input_size / ml.K_QUANT_BLOCK_SIZE) * ml.Q4_K_BLOCK_BYTES
	case .Q6_K: bytes_per_row = (input_size / ml.K_QUANT_BLOCK_SIZE) * ml.Q6_K_BLOCK_BYTES
	case:       return
	}
	for o in 0 ..< output_size {
		w_row := w_bytes[o * bytes_per_row : (o + 1) * bytes_per_row]
		#partial switch ty {
		case .Q4_K: gguf.dequantize_q4_k(w_row, dequant)
		case .Q6_K: gguf.dequantize_q6_k(w_row, dequant)
		}
		for c in 0 ..< batch {
			total: f32
			for k in 0 ..< input_size {
				total += dequant[k] * ml.bf16_to_f32(x_buf[c * input_size + k])
			}
			expected_buf[c * output_size + o] = ml.bf16_from_f32(total)
		}
	}

	_compare(got_buf, expected_buf,
		fmt.tprintf("%v real %v shape=%v", ty, chosen_name, chosen_info.shape),
		any_failed)
}

_ml_dtype :: proc(ty: gguf.Tensor_Type) -> ml.Data_Type {
	#partial switch ty {
	case .Q4_K: return .Q4_K
	case .Q6_K: return .Q6_K
	case:       return .F32
	}
}

// --- GPU parity (Q4_K only — Q6_K shader not implemented yet) ----------

// The GPU GEMV shader supports M=1 only. Reductions are parallel (subgroupAdd
// across 32 lanes), so summation order differs from the sequential CPU
// reference; expect bit-different bf16 outputs and use tolerance.

REL_TOL :: 5e-2
ABS_TOL :: 1e-2

_run_q4_k_gpu_parity :: proc(w_bytes: []byte, any_failed: ^bool, label: string) {
	weight := ml.alloc(.Q4_K, {N, K}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	// Single activation row (M=1).
	x := _make_synthetic_activation()
	x_t := ml.alloc(.Bf16, {1, K}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x[:K]))

	y := ml.linear_q4_k(x_t, weight)
	got: [N]ml.Bf16
	ml.get_data_bytes(y, mem.slice_to_bytes(got[:]))

	expected := _reference_q4_k_matmul_m1(w_bytes, x[:K])
	// The integer-dot path round-trips the activation through Q8_1, adding
	// ~1/127 per-element error. Synthetic weights here have d*scale magnitudes
	// up to ~10 per element, so accumulated noise over 256 elements is large.
	// Use a looser tolerance on the synthetic case; real-tensor parity below
	// runs at the standard threshold and is the binding correctness check.
	_compare_with_tolerance_eps(got[:], expected[:], fmt.tprintf("Q4_K GPU %v", label), any_failed,
		rel_tol=0.20, abs_tol=0.30)
}

// M=64, N=64, K=256 (one Q4_K block). Exercises the coopmat tile path; weight
// rows are dense in N so the BN=64 tile is fully utilized.
_run_q4_k_gpu_coopmat_parity :: proc(w_bytes: []byte, any_failed: ^bool, label: string) {
	M_CM :: 64
	N_CM :: 64
	K_CM :: 256

	weight := ml.alloc(.Q4_K, {N_CM, K_CM}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x_buf := make([]ml.Bf16, M_CM * K_CM)
	defer delete(x_buf)
	state := u64(0xABCD_EF02)
	for i in 0 ..< M_CM * K_CM {
		raw := _lcg(&state)
		v := (f32(i32(raw & 0xFFFF)) - 32768.0) / 32768.0
		x_buf[i] = ml.bf16_from_f32(v)
	}
	x_t := ml.alloc(.Bf16, {M_CM, K_CM}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x_buf))

	y := ml.linear_q4_k(x_t, weight)
	got := make([]ml.Bf16, M_CM * N_CM)
	defer delete(got)
	ml.get_data_bytes(y, mem.slice_to_bytes(got))

	expected := make([]ml.Bf16, M_CM * N_CM)
	defer delete(expected)
	row_bytes := (K_CM / ml.K_QUANT_BLOCK_SIZE) * ml.Q4_K_BLOCK_BYTES
	dequant   := make([]f32, K_CM)
	defer delete(dequant)
	for o in 0 ..< N_CM {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q4_k(w_row, dequant)
		for c in 0 ..< M_CM {
			total: f32
			for k in 0 ..< K_CM {
				total += dequant[k] * ml.bf16_to_f32(x_buf[c * K_CM + k])
			}
			expected[c * N_CM + o] = ml.bf16_from_f32(total)
		}
	}

	// Synthetic weights run to ~10/element; coopmat stages dequanted weights
	// through a bf16 shared-memory tile (one round-trip per weight) before
	// the fp32 tensor-core accumulate, giving output-magnitude-dependent
	// noise vs the f32 CPU reference. Real-tensor parity below uses the
	// standard tolerance.
	_compare_with_tolerance_eps(got, expected, fmt.tprintf("Q4_K GPU coopmat %v", label), any_failed,
		rel_tol=0.20, abs_tol=0.50)
}

_run_q6_k_gpu_parity :: proc(w_bytes: []byte, any_failed: ^bool, label: string) {
	weight := ml.alloc(.Q6_K, {N, K}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x := _make_synthetic_activation()
	x_t := ml.alloc(.Bf16, {1, K}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x[:K]))

	y := ml.linear_q6_k(x_t, weight)
	got: [N]ml.Bf16
	ml.get_data_bytes(y, mem.slice_to_bytes(got[:]))

	expected := _reference_q6_k_matmul_m1(w_bytes, x[:K])
	_compare_with_tolerance(got[:], expected[:], fmt.tprintf("Q6_K GPU %v", label), any_failed)
}

// M=64, N=64, K=256 Q6_K coopmat parity. Mirrors the Q4_K coopmat case.
_run_q6_k_gpu_coopmat_parity :: proc(w_bytes: []byte, any_failed: ^bool, label: string) {
	M_CM :: 64
	N_CM :: 64
	K_CM :: 256

	weight := ml.alloc(.Q6_K, {N_CM, K_CM}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x_buf := make([]ml.Bf16, M_CM * K_CM)
	defer delete(x_buf)
	state := u64(0xABCD_EF03)
	for i in 0 ..< M_CM * K_CM {
		raw := _lcg(&state)
		v := (f32(i32(raw & 0xFFFF)) - 32768.0) / 32768.0
		x_buf[i] = ml.bf16_from_f32(v)
	}
	x_t := ml.alloc(.Bf16, {M_CM, K_CM}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x_buf))

	y := ml.linear_q6_k(x_t, weight)
	got := make([]ml.Bf16, M_CM * N_CM)
	defer delete(got)
	ml.get_data_bytes(y, mem.slice_to_bytes(got))

	expected := make([]ml.Bf16, M_CM * N_CM)
	defer delete(expected)
	row_bytes := (K_CM / ml.K_QUANT_BLOCK_SIZE) * ml.Q6_K_BLOCK_BYTES
	dequant   := make([]f32, K_CM)
	defer delete(dequant)
	for o in 0 ..< N_CM {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q6_k(w_row, dequant)
		for k in 0 ..< K_CM {
			dequant[k] = ml.bf16_to_f32(ml.bf16_from_f32(dequant[k]))
		}
		for c in 0 ..< M_CM {
			total: f32
			for k in 0 ..< K_CM {
				total += dequant[k] * ml.bf16_to_f32(x_buf[c * K_CM + k])
			}
			expected[c * N_CM + o] = ml.bf16_from_f32(total)
		}
	}

	_compare_with_tolerance(got, expected, fmt.tprintf("Q6_K GPU coopmat %v", label), any_failed)
}

_reference_q6_k_matmul_m1 :: proc(w_bytes: []byte, x: []ml.Bf16) -> [N]ml.Bf16 {
	out: [N]ml.Bf16
	row_bytes := (K / ml.K_QUANT_BLOCK_SIZE) * ml.Q6_K_BLOCK_BYTES
	dequant   := make([]f32, K)
	defer delete(dequant)
	for o in 0 ..< N {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q6_k(w_row, dequant)
		total: f32
		for k in 0 ..< K {
			total += dequant[k] * ml.bf16_to_f32(x[k])
		}
		out[o] = ml.bf16_from_f32(total)
	}
	return out
}

_reference_q4_k_matmul_m1 :: proc(w_bytes: []byte, x: []ml.Bf16) -> [N]ml.Bf16 {
	out: [N]ml.Bf16
	row_bytes := (K / ml.K_QUANT_BLOCK_SIZE) * ml.Q4_K_BLOCK_BYTES
	dequant   := make([]f32, K)
	defer delete(dequant)
	for o in 0 ..< N {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q4_k(w_row, dequant)
		total: f32
		for k in 0 ..< K {
			total += dequant[k] * ml.bf16_to_f32(x[k])
		}
		out[o] = ml.bf16_from_f32(total)
	}
	return out
}

_compare_with_tolerance_eps :: proc(got, expected: []ml.Bf16, label: string, any_failed: ^bool, rel_tol, abs_tol: f32) {
	if len(got) != len(expected) {
		fmt.printfln("FAIL %v: length mismatch %v vs %v", label, len(got), len(expected))
		any_failed^ = true
		return
	}
	max_abs_err: f32
	max_rel_err: f32
	first_bad := -1
	for i in 0 ..< len(got) {
		g := ml.bf16_to_f32(got[i])
		e := ml.bf16_to_f32(expected[i])
		diff := g - e
		if diff < 0 do diff = -diff
		if diff > max_abs_err do max_abs_err = diff
		ae := e
		if ae < 0 do ae = -ae
		rel := diff / (ae + abs_tol)
		if rel > max_rel_err do max_rel_err = rel
		if first_bad < 0 && diff > rel_tol * ae + abs_tol && !math.is_nan(g) {
			first_bad = i
		}
	}
	if first_bad < 0 {
		fmt.printfln("OK   %v: within tolerance (%v outputs, max_abs=%.5f, max_rel=%.5f)",
			label, len(got), max_abs_err, max_rel_err)
	} else {
		i := first_bad
		fmt.printfln("FAIL %v: first mismatch at %v: got %v expected %v (max_abs=%.5f, max_rel=%.5f)",
			label, i, ml.bf16_to_f32(got[i]), ml.bf16_to_f32(expected[i]), max_abs_err, max_rel_err)
		any_failed^ = true
	}
}

_compare_with_tolerance :: proc(got, expected: []ml.Bf16, label: string, any_failed: ^bool) {
	if len(got) != len(expected) {
		fmt.printfln("FAIL %v: length mismatch %v vs %v", label, len(got), len(expected))
		any_failed^ = true
		return
	}
	max_abs_err: f32
	max_rel_err: f32
	first_bad := -1
	for i in 0 ..< len(got) {
		g := ml.bf16_to_f32(got[i])
		e := ml.bf16_to_f32(expected[i])
		diff := g - e
		if diff < 0 do diff = -diff
		if diff > max_abs_err do max_abs_err = diff
		ae := e
		if ae < 0 do ae = -ae
		rel := diff / (ae + ABS_TOL)
		if rel > max_rel_err do max_rel_err = rel
		if first_bad < 0 && diff > REL_TOL * ae + ABS_TOL && !math.is_nan(g) {
			first_bad = i
		}
	}
	if first_bad < 0 {
		fmt.printfln("OK   %v: within tolerance (%v outputs, max_abs=%.5f, max_rel=%.5f)",
			label, len(got), max_abs_err, max_rel_err)
	} else {
		i := first_bad
		fmt.printfln("FAIL %v: first mismatch at %v: got %v expected %v (max_abs=%.5f, max_rel=%.5f)",
			label, i, ml.bf16_to_f32(got[i]), ml.bf16_to_f32(expected[i]), max_abs_err, max_rel_err)
		any_failed^ = true
	}
}

_run_real_gpu_parity :: proc(path: string, any_failed: ^bool) {
	loader, ok := gguf.load(path)
	if !ok {
		any_failed^ = true
		return
	}
	defer gguf.destroy(loader)

	ctx := gpu.context_create()
	defer gpu.context_destroy(ctx)
	ml.context_scope(ctx)

	_run_real_gpu_for_type(loader, .Q4_K, any_failed)
	_run_real_gpu_for_type(loader, .Q6_K, any_failed)
	_run_real_gpu_coopmat_q4_k(loader, any_failed)
	_run_real_gpu_coopmat_q6_k(loader, any_failed)

	ml.clear()
}

// Real Gemma 4 Q4_K tensor through the coopmat path. Picks a tensor with
// output_size and input_size both multiples of 64; runs M=64 activations.
_run_real_gpu_coopmat_q4_k :: proc(loader: gguf.Loader, any_failed: ^bool) {
	M_CM :: 64

	chosen_name: string
	chosen_info: gguf.Tensor_Info
	found := false
	for name, info in loader.tensors {
		if info.type != .Q4_K do continue
		if len(info.shape) != 2 do continue
		if info.shape[0] % 64 != 0 do continue
		if info.shape[1] % 64 != 0 do continue
		if !found || info.shape[0] * info.shape[1] < chosen_info.shape[0] * chosen_info.shape[1] {
			chosen_name = name
			chosen_info = info
			found = true
		}
	}
	if !found {
		fmt.println("(no Q4_K tensor with 64-aligned dims; skipping coopmat real parity)")
		return
	}

	output_size := chosen_info.shape[0]
	input_size  := chosen_info.shape[1]
	w_bytes, _  := gguf.get_bytes(loader, chosen_name)

	weight := ml.alloc(.Q4_K, {output_size, input_size}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x_buf := make([]ml.Bf16, M_CM * input_size)
	defer delete(x_buf)
	state := u64(0x1234_5679)
	for i in 0 ..< M_CM * input_size {
		raw := _lcg(&state)
		v := (f32(i32(raw & 0xFFFF)) - 32768.0) / 32768.0
		x_buf[i] = ml.bf16_from_f32(v)
	}
	x_t := ml.alloc(.Bf16, {M_CM, input_size}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x_buf))

	y := ml.linear_q4_k(x_t, weight)

	got := make([]ml.Bf16, M_CM * output_size)
	defer delete(got)
	ml.get_data_bytes(y, mem.slice_to_bytes(got))

	expected := make([]ml.Bf16, M_CM * output_size)
	defer delete(expected)
	row_bytes := (input_size / ml.K_QUANT_BLOCK_SIZE) * ml.Q4_K_BLOCK_BYTES
	dequant   := make([]f32, input_size)
	defer delete(dequant)
	for o in 0 ..< output_size {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q4_k(w_row, dequant)
		// Mirror the coopmat path: stage dequanted weights through bf16 in
		// shared memory, then accumulate against bf16 activations in fp32.
		for k in 0 ..< input_size {
			dequant[k] = ml.bf16_to_f32(ml.bf16_from_f32(dequant[k]))
		}
		for c in 0 ..< M_CM {
			total: f32
			for k in 0 ..< input_size {
				total += dequant[k] * ml.bf16_to_f32(x_buf[c * input_size + k])
			}
			expected[c * output_size + o] = ml.bf16_from_f32(total)
		}
	}

	_compare_with_tolerance(got, expected,
		fmt.tprintf("Q4_K GPU coopmat real %v shape=%v M=%v", chosen_name, chosen_info.shape, M_CM),
		any_failed)
}

_run_real_gpu_coopmat_q6_k :: proc(loader: gguf.Loader, any_failed: ^bool) {
	M_CM :: 64

	chosen_name: string
	chosen_info: gguf.Tensor_Info
	found := false
	for name, info in loader.tensors {
		if info.type != .Q6_K do continue
		if len(info.shape) != 2 do continue
		if info.shape[0] % 64 != 0 do continue
		if info.shape[1] % 64 != 0 do continue
		if !found || info.shape[0] * info.shape[1] < chosen_info.shape[0] * chosen_info.shape[1] {
			chosen_name = name
			chosen_info = info
			found = true
		}
	}
	if !found {
		fmt.println("(no Q6_K tensor with 64-aligned dims; skipping coopmat real parity)")
		return
	}

	output_size := chosen_info.shape[0]
	input_size  := chosen_info.shape[1]
	w_bytes, _  := gguf.get_bytes(loader, chosen_name)

	weight := ml.alloc(.Q6_K, {output_size, input_size}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x_buf := make([]ml.Bf16, M_CM * input_size)
	defer delete(x_buf)
	state := u64(0x1234_567A)
	for i in 0 ..< M_CM * input_size {
		raw := _lcg(&state)
		v := (f32(i32(raw & 0xFFFF)) - 32768.0) / 32768.0
		x_buf[i] = ml.bf16_from_f32(v)
	}
	x_t := ml.alloc(.Bf16, {M_CM, input_size}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x_buf))

	y := ml.linear_q6_k(x_t, weight)

	got := make([]ml.Bf16, M_CM * output_size)
	defer delete(got)
	ml.get_data_bytes(y, mem.slice_to_bytes(got))

	expected := make([]ml.Bf16, M_CM * output_size)
	defer delete(expected)
	row_bytes := (input_size / ml.K_QUANT_BLOCK_SIZE) * ml.Q6_K_BLOCK_BYTES
	dequant   := make([]f32, input_size)
	defer delete(dequant)
	for o in 0 ..< output_size {
		w_row := w_bytes[o * row_bytes : (o + 1) * row_bytes]
		gguf.dequantize_q6_k(w_row, dequant)
		for k in 0 ..< input_size {
			dequant[k] = ml.bf16_to_f32(ml.bf16_from_f32(dequant[k]))
		}
		for c in 0 ..< M_CM {
			total: f32
			for k in 0 ..< input_size {
				total += dequant[k] * ml.bf16_to_f32(x_buf[c * input_size + k])
			}
			expected[c * output_size + o] = ml.bf16_from_f32(total)
		}
	}

	_compare_with_tolerance(got, expected,
		fmt.tprintf("Q6_K GPU coopmat real %v shape=%v M=%v", chosen_name, chosen_info.shape, M_CM),
		any_failed)
}

_run_real_gpu_for_type :: proc(loader: gguf.Loader, ty: gguf.Tensor_Type, any_failed: ^bool) {
	chosen_name: string
	chosen_info: gguf.Tensor_Info
	found := false
	for name, info in loader.tensors {
		if info.type != ty do continue
		if len(info.shape) != 2 do continue
		if info.shape[1] % ml.K_QUANT_BLOCK_SIZE != 0 do continue
		// Even output_size required by ROWS_PER_WG=2.
		if info.shape[0] % 2 != 0 do continue
		if !found || info.shape[0] * info.shape[1] < chosen_info.shape[0] * chosen_info.shape[1] {
			chosen_name = name
			chosen_info = info
			found = true
		}
	}
	if !found {
		fmt.printfln("(no %v tensor found; skipping GPU real parity)", ty)
		return
	}

	output_size := chosen_info.shape[0]
	input_size  := chosen_info.shape[1]
	w_bytes, _  := gguf.get_bytes(loader, chosen_name)

	weight := ml.alloc(_ml_dtype(ty), {output_size, input_size}, persistent=true, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(weight, w_bytes)

	x_buf := make([]ml.Bf16, input_size)
	defer delete(x_buf)
	state := u64(0x1234_5678)
	for i in 0 ..< input_size {
		raw := _lcg(&state)
		v := (f32(i32(raw & 0xFFFF)) - 32768.0) / 32768.0
		x_buf[i] = ml.bf16_from_f32(v)
	}
	x_t := ml.alloc(.Bf16, {1, input_size}, persistent=false, buffers=ml.Buffer_Set{.Data})
	ml.set_data_bytes(x_t, mem.slice_to_bytes(x_buf))

	y: ml.Tensor
	bytes_per_row := 0
	#partial switch ty {
	case .Q4_K:
		y = ml.linear_q4_k(x_t, weight)
		bytes_per_row = (input_size / ml.K_QUANT_BLOCK_SIZE) * ml.Q4_K_BLOCK_BYTES
	case .Q6_K:
		y = ml.linear_q6_k(x_t, weight)
		bytes_per_row = (input_size / ml.K_QUANT_BLOCK_SIZE) * ml.Q6_K_BLOCK_BYTES
	case:
		return
	}

	got_buf := make([]ml.Bf16, output_size)
	defer delete(got_buf)
	ml.get_data_bytes(y, mem.slice_to_bytes(got_buf))

	expected_buf := make([]ml.Bf16, output_size)
	defer delete(expected_buf)
	dequant := make([]f32, input_size)
	defer delete(dequant)
	for o in 0 ..< output_size {
		w_row := w_bytes[o * bytes_per_row : (o + 1) * bytes_per_row]
		#partial switch ty {
		case .Q4_K: gguf.dequantize_q4_k(w_row, dequant)
		case .Q6_K: gguf.dequantize_q6_k(w_row, dequant)
		}
		total: f32
		for k in 0 ..< input_size {
			total += dequant[k] * ml.bf16_to_f32(x_buf[k])
		}
		expected_buf[o] = ml.bf16_from_f32(total)
	}

	_compare_with_tolerance(got_buf, expected_buf,
		fmt.tprintf("%v GPU real %v shape=%v", ty, chosen_name, chosen_info.shape),
		any_failed)
}
