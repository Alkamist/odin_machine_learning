package gpu_linear_q4

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

import ml  "../.."
import cpu "../../backends/cpu"
import gpu "../../backends/gpu"

OUT_SIZE :: 128
IN_SIZE  :: 256
COUNT    :: 1

run :: proc(use_gpu: bool, weight_bf, x_bf: []ml.Bf16, out_bf: []ml.Bf16) {
	ctx_cpu: ^ml.Context
	ctx_gpu: ^ml.Context
	if use_gpu {
		ctx_gpu = gpu.context_create()
	} else {
		ctx_cpu = cpu.context_create(64 * 1024 * 1024)
	}
	defer if use_gpu { gpu.context_destroy(ctx_gpu) } else { cpu.context_destroy(ctx_cpu) }
	ml.context_scope(use_gpu ? ctx_gpu : ctx_cpu)

	w := ml.zeros(.Bf16, {OUT_SIZE, IN_SIZE})
	x := ml.zeros(.Bf16, {COUNT, IN_SIZE})
	ml.set_data_bytes(w, mem.slice_to_bytes(weight_bf))
	ml.set_data_bytes(x, mem.slice_to_bytes(x_bf))

	w_q, w_s := ml.quantize_int4(w)
	y := ml.linear_q4(x, w_q, w_s)

	ml.get_data_bytes(y, mem.slice_to_bytes(out_bf))
}

main :: proc() {
	weight_bf := make([]ml.Bf16, OUT_SIZE * IN_SIZE)
	x_bf      := make([]ml.Bf16, COUNT * IN_SIZE)
	defer delete(weight_bf)
	defer delete(x_bf)

	for i in 0 ..< len(weight_bf) {
		v := f32((i * 13) % 191) * 0.013 - 1.2
		weight_bf[i] = ml.bf16_from_f32(v)
	}
	for i in 0 ..< len(x_bf) {
		v := f32((i * 7) % 53) * 0.04 - 1.0
		x_bf[i] = ml.bf16_from_f32(v)
	}

	cpu_out := make([]ml.Bf16, COUNT * OUT_SIZE)
	gpu_out := make([]ml.Bf16, COUNT * OUT_SIZE)
	defer delete(cpu_out)
	defer delete(gpu_out)

	run(false, weight_bf, x_bf, cpu_out)
	run(true,  weight_bf, x_bf, gpu_out)

	max_abs:    f32
	max_rel:    f32
	mismatches: int
	for i in 0 ..< len(cpu_out) {
		a := ml.bf16_to_f32(cpu_out[i])
		b := ml.bf16_to_f32(gpu_out[i])
		d := math.abs(a - b)
		if d > max_abs do max_abs = d
		denom := math.max(math.abs(a), 1e-3)
		if d / denom > max_rel do max_rel = d / denom
		if d > 0.05 do mismatches += 1
	}

	fmt.printfln("linear_q4 GPU vs CPU:  max_abs=%.5f  max_rel=%.5f  mismatches(>0.05)=%v / %v",
		max_abs, max_rel, mismatches, len(cpu_out))

	// Bf16 has ~3 decimal digits of precision; per-output sum of ~256 terms with
	// q4 grid step ~max/7 means accumulated rounding is dominated by bf16 storage,
	// not by the int4 quantization itself. Allow generous abs tolerance.
	if mismatches > 0 {
		fmt.println("FAIL")
		os.exit(1)
	}
	fmt.println("OK")
}
