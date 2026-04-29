package dtype_roundtrip

import "core:fmt"
import "core:math"
import "core:mem"
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
			ml.set_data_bytes(t, mem.slice_to_bytes(encoded[:]))

			out: [4]ml.Bf16
			ml.get_data_bytes(t, mem.slice_to_bytes(out[:]))
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
			ml.set_data_bytes(t, mem.slice_to_bytes(encoded[:]))

			out: [4]f16
			ml.get_data_bytes(t, mem.slice_to_bytes(out[:]))
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

		// Bf16 add forward + backward — broadcast add of clean values that
		// round-trip exactly through bf16. cast_to .F32 at the end so backward
		// can seed an F32 gradient.
		{
			N   :: 8
			N_B :: 4

			x_src := [N]f32{1, 2, 3, 4, 5, 6, 7, 8}
			b_src := [N_B]f32{10, 20, 30, 40}

			x_f32 := ml.tensor(x_src[:])
			b_f32 := ml.tensor(b_src[:])
			x     := ml.cast_to(x_f32, .Bf16)
			b     := ml.cast_to(b_f32, .Bf16)

			y_bf := ml.add(x, b)
			y    := ml.cast_to(y_bf, .F32)

			got_y: [N]f32
			ml.get_data(y, got_y[:])

			expected_y := [N]f32{11, 22, 33, 44, 15, 26, 37, 48}
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 add forward (broadcast) matches", label), any_failed)

			ml.backward()

			got_dx: [N]f32
			got_db: [N_B]f32
			ml.get_gradient(x_f32, got_dx[:])
			ml.get_gradient(b_f32, got_db[:])

			dx_ok := true
			for v in got_dx {
				if v != 1 { dx_ok = false }
			}
			db_ok := true
			for v in got_db {
				if v != f32(N / N_B) { db_ok = false }
			}
			check(dx_ok, fmt.tprintf("%v: Bf16 add backward dx == ones", label), any_failed)
			check(db_ok, fmt.tprintf("%v: Bf16 add backward db == stride", label), any_failed)
		}

		// Bf16 linear forward — clean values that round-trip through bf16
		// exactly so the result is bit-identical to the f32 reference (and
		// thus identical across the cpu and gpu paths).
		{
			M :: 4
			K :: 8
			N :: 6

			x_src: [M * K]f32
			for i in 0 ..< M * K {
				x_src[i] = f32((i % 5) - 2)
			}
			w_src: [N * K]f32
			for i in 0 ..< N * K {
				w_src[i] = f32(((i + 3) % 7) - 3) * 0.5
			}

			x_f32 := ml.tensor(x_src[:])
			w_f32 := ml.tensor(w_src[:])
			x     := ml.cast_to(ml.reshape(x_f32, {M, K}), .Bf16)
			w     := ml.cast_to(ml.reshape(w_f32, {N, K}), .Bf16)

			y_bf := ml.linear(x, w)
			y    := ml.cast_to(y_bf, .F32)

			got: [M * N]f32
			ml.get_data(y, got[:])

			expected: [M * N]f32
			for i in 0 ..< M {
				for o in 0 ..< N {
					sum: f32
					for k in 0 ..< K {
						sum += x_src[i * K + k] * w_src[o * K + k]
					}
					expected[i * N + o] = sum
				}
			}

			ok := true
			for i in 0 ..< M * N {
				if got[i] != expected[i] { ok = false }
			}
			check(ok, fmt.tprintf("%v: Bf16 linear forward matches f32 reference", label), any_failed)
		}

		// Bf16 linear forward across tile boundaries. M=64 covers 2x TILE_M=32,
		// N=128 covers 2x TILE_N=64, and K=64 covers 4x TILE_K=16. Values are
		// in {-1, 0, 1} so each multiply is exact, every partial sum stays
		// integer in [-K, K], and the final result is bit-exact in bf16.
		{
			M :: 64
			K :: 64
			N :: 128

			x_src: [M * K]f32
			for i in 0 ..< M * K {
				x_src[i] = f32((i % 3) - 1)
			}
			w_src: [N * K]f32
			for i in 0 ..< N * K {
				w_src[i] = f32(((i + 1) % 3) - 1)
			}

			x_f32 := ml.tensor(x_src[:])
			w_f32 := ml.tensor(w_src[:])
			x     := ml.cast_to(ml.reshape(x_f32, {M, K}), .Bf16)
			w     := ml.cast_to(ml.reshape(w_f32, {N, K}), .Bf16)

			y_bf := ml.linear(x, w)
			y    := ml.cast_to(y_bf, .F32)

			got: [M * N]f32
			ml.get_data(y, got[:])

			expected: [M * N]f32
			for i in 0 ..< M {
				for o in 0 ..< N {
					sum: f32
					for k in 0 ..< K {
						sum += x_src[i * K + k] * w_src[o * K + k]
					}
					expected[i * N + o] = sum
				}
			}

			ok := true
			for i in 0 ..< M * N {
				if got[i] != expected[i] { ok = false }
			}
			check(ok, fmt.tprintf("%v: Bf16 linear forward (multi-tile) matches f32 reference", label), any_failed)
		}

		// Bf16 linear backward — same {-1, 0, 1} value trick so the f32 reference
		// gradients are bit-exact in bf16. backward() seeds dy = ones, so:
		//   dx[c, k] = sum_o w[o, k]
		//   dw[o, k] = sum_c x[c, k]
		// Both sums are bounded by ±count or ±output_size respectively; with
		// count=N=output_size=16 they fit in [-16, 16], representable exactly.
		{
			M :: 16
			K :: 16
			N :: 16

			x_src: [M * K]f32
			for i in 0 ..< M * K {
				x_src[i] = f32((i % 3) - 1)
			}
			w_src: [N * K]f32
			for i in 0 ..< N * K {
				w_src[i] = f32(((i + 1) % 3) - 1)
			}

			x_f32 := ml.tensor(x_src[:])
			w_f32 := ml.tensor(w_src[:])
			x     := ml.cast_to(ml.reshape(x_f32, {M, K}), .Bf16)
			w     := ml.cast_to(ml.reshape(w_f32, {N, K}), .Bf16)

			y_bf := ml.linear(x, w)
			y    := ml.cast_to(y_bf, .F32)
			_ = y

			ml.backward()

			expected_dx: [M * K]f32
			for c in 0 ..< M {
				for k in 0 ..< K {
					sum: f32
					for o in 0 ..< N {
						sum += w_src[o * K + k]
					}
					expected_dx[c * K + k] = sum
				}
			}
			expected_dw: [N * K]f32
			for o in 0 ..< N {
				for k in 0 ..< K {
					sum: f32
					for c in 0 ..< M {
						sum += x_src[c * K + k]
					}
					expected_dw[o * K + k] = sum
				}
			}

			got_dx: [M * K]f32
			got_dw: [N * K]f32
			ml.get_gradient(x_f32, got_dx[:])
			ml.get_gradient(w_f32, got_dw[:])

			dx_ok := true
			for i in 0 ..< M * K {
				if got_dx[i] != expected_dx[i] { dx_ok = false }
			}
			dw_ok := true
			for i in 0 ..< N * K {
				if got_dw[i] != expected_dw[i] { dw_ok = false }
			}
			check(dx_ok, fmt.tprintf("%v: Bf16 linear backward dx matches f32 reference", label), any_failed)
			check(dw_ok, fmt.tprintf("%v: Bf16 linear backward dw matches f32 reference", label), any_failed)
		}

		// Bf16 batched_matmul forward + backward. Same {-1, 0, 1} trick: each
		// reduction is bounded so the bf16 result is bit-exact.
		{
			BATCH :: 2
			M     :: 4
			K     :: 8
			N     :: 6

			a_src: [BATCH * M * K]f32
			for i in 0 ..< BATCH * M * K {
				a_src[i] = f32((i % 3) - 1)
			}
			b_src: [BATCH * K * N]f32
			for i in 0 ..< BATCH * K * N {
				b_src[i] = f32(((i + 1) % 3) - 1)
			}

			a_f32 := ml.tensor(a_src[:])
			b_f32 := ml.tensor(b_src[:])
			a     := ml.cast_to(ml.reshape(a_f32, {BATCH, M, K}), .Bf16)
			b     := ml.cast_to(ml.reshape(b_f32, {BATCH, K, N}), .Bf16)

			c_bf := ml.batched_matmul(a, b)
			c    := ml.cast_to(c_bf, .F32)

			got_c: [BATCH * M * N]f32
			ml.get_data(c, got_c[:])

			expected_c: [BATCH * M * N]f32
			for bi in 0 ..< BATCH {
				for i in 0 ..< M {
					for j in 0 ..< N {
						sum: f32
						for kk in 0 ..< K {
							sum += a_src[bi * M * K + i * K + kk] *
							       b_src[bi * K * N + kk * N + j]
						}
						expected_c[bi * M * N + i * N + j] = sum
					}
				}
			}
			fwd_ok := true
			for i in 0 ..< BATCH * M * N {
				if got_c[i] != expected_c[i] { fwd_ok = false }
			}
			check(fwd_ok, fmt.tprintf("%v: Bf16 batched_matmul forward matches f32 reference", label), any_failed)

			ml.backward()

			expected_da: [BATCH * M * K]f32
			for bi in 0 ..< BATCH {
				for i in 0 ..< M {
					for kk in 0 ..< K {
						sum: f32
						for j in 0 ..< N {
							sum += b_src[bi * K * N + kk * N + j]
						}
						expected_da[bi * M * K + i * K + kk] = sum
					}
				}
			}
			expected_db: [BATCH * K * N]f32
			for bi in 0 ..< BATCH {
				for kk in 0 ..< K {
					for j in 0 ..< N {
						sum: f32
						for i in 0 ..< M {
							sum += a_src[bi * M * K + i * K + kk]
						}
						expected_db[bi * K * N + kk * N + j] = sum
					}
				}
			}

			got_da: [BATCH * M * K]f32
			got_db: [BATCH * K * N]f32
			ml.get_gradient(a_f32, got_da[:])
			ml.get_gradient(b_f32, got_db[:])

			da_ok := true
			for i in 0 ..< BATCH * M * K {
				if got_da[i] != expected_da[i] { da_ok = false }
			}
			db_ok := true
			for i in 0 ..< BATCH * K * N {
				if got_db[i] != expected_db[i] { db_ok = false }
			}
			check(da_ok, fmt.tprintf("%v: Bf16 batched_matmul backward da matches f32 reference", label), any_failed)
			check(db_ok, fmt.tprintf("%v: Bf16 batched_matmul backward db matches f32 reference", label), any_failed)
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
