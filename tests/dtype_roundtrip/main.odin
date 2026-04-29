package dtype_roundtrip

import "core:fmt"
import "core:math"
import "core:os"

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

	run_backend :: proc(label: string, ctx: ^ml.Context, any_failed: ^bool) {
		ml.context_scope(ctx)

		// F32 round-trip — sanity, since the slice header semantics changed.
		{
			t := ml.zeros(.F32, {4})
			src := [4]f32{1.5, -2.25, 3.75, 0.125}
			ml.set_data(t, src[:])
			dst: [4]f32
			ml.get_data(t, dst[:])
			ok := dst == src
			check(ok, fmt.tprintf("%v: F32 round-trip", label), any_failed)
		}

		// Bf16 round-trip — bit-exact under the truncate-with-RNE conversion
		// for values whose mantissa fits in 7 bits (clean halves work).
		{
			t := ml.zeros(.Bf16, {4})
			values := [4]f32{1, -2, 3.5, -0.25}
			encoded: [4]ml.Bf16
			for v, i in values { encoded[i] = ml.bf16_from_f32(v) }
			ml.set_data_bytes(t, ml._slice_to_bytes(encoded[:]))

			out: [4]ml.Bf16
			ml.get_data_bytes(t, ml._slice_to_bytes(out[:]))
			ok := out == encoded
			check(ok, fmt.tprintf("%v: Bf16 byte round-trip", label), any_failed)

			decoded_ok := true
			for v, i in values {
				if ml.bf16_to_f32(out[i]) != v { decoded_ok = false }
			}
			check(decoded_ok, fmt.tprintf("%v: Bf16 decode matches source", label), any_failed)
		}

		// F16 round-trip — same idea, native f16 type.
		{
			t := ml.zeros(.F16, {4})
			encoded := [4]f16{1, -2, 3.5, -0.25}
			ml.set_data_bytes(t, ml._slice_to_bytes(encoded[:]))

			out: [4]f16
			ml.get_data_bytes(t, ml._slice_to_bytes(out[:]))
			ok := out == encoded
			check(ok, fmt.tprintf("%v: F16 byte round-trip", label), any_failed)
		}

		// cast_to F32 -> Bf16 -> F32 round-trip with values that survive
		// truncation exactly (powers of 2, halves).
		{
			x_src := [9]f32{0, 1, -1, 2, -2, 3.5, -0.25, 64, -0.0625}
			x := ml.tensor(x_src[:])
			y := ml.cast_to(x, .Bf16)
			z := ml.cast_to(y, .F32)

			recovered: [9]f32
			ml.get_data(z, recovered[:])
			ok := true
			for v, i in x_src {
				if recovered[i] != v { ok = false }
			}
			check(ok, fmt.tprintf("%v: cast_to F32->Bf16->F32 exact for clean values", label), any_failed)
		}

		// cast_to backward — F32 -> Bf16 cast. backward() seeds the final
		// output's gradient with ones; cast backward should expand those
		// bf16 ones into f32 ones and accumulate into x.grad. Loss tensor
		// must currently be F32, so cast back at the end.
		{
			n :: 4
			x_src := [n]f32{1, 2, 3, 4}
			x  := ml.tensor(x_src[:])
			y  := ml.cast_to(x, .Bf16)
			z  := ml.cast_to(y, .F32)
			_   = z
			ml.backward()

			got: [n]f32
			ml.get_gradient(x, got[:])
			ok := true
			for i in 0 ..< n {
				if math.abs(got[i] - 1) > 1e-6 { ok = false }
			}
			check(ok, fmt.tprintf("%v: cast_to backward propagates ones through bf16", label), any_failed)
		}

		ml.clear()
	}

	{
		ctx := cpu.context_create(1 * 1024 * 1024)
		defer cpu.context_destroy(ctx)
		run_backend("cpu", ctx, &any_failed)
	}
	{
		ctx := gpu.context_create()
		defer gpu.context_destroy(ctx)
		run_backend("gpu", ctx, &any_failed)
	}

	if any_failed {
		os.exit(1)
	}
	fmt.println("all dtype round-trip checks passed")
}
