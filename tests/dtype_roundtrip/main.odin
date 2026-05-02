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

		// Bf16 batched_matmul across the coopmat tile boundary. M=K=N=64 is the
		// smallest shape for which all three workgroup tiles (BM/BN=64, BK=16)
		// are exercised in both forward and backward. Same {-1, 0, 1} value
		// trick keeps every partial sum exact in bf16.
		{
			BATCH :: 2
			M     :: 64
			K     :: 64
			N     :: 64

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
						for k in 0 ..< K {
							sum += a_src[bi * M * K + i * K + k] * b_src[bi * K * N + k * N + j]
						}
						expected_c[bi * M * N + i * N + j] = sum
					}
				}
			}

			fwd_ok := true
			for i in 0 ..< BATCH * M * N {
				if got_c[i] != expected_c[i] { fwd_ok = false }
			}
			check(fwd_ok, fmt.tprintf("%v: Bf16 batched_matmul (coopmat-tile) forward matches f32 reference", label), any_failed)

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
			check(da_ok, fmt.tprintf("%v: Bf16 batched_matmul (coopmat-tile) backward da matches f32 reference", label), any_failed)
			check(db_ok, fmt.tprintf("%v: Bf16 batched_matmul (coopmat-tile) backward db matches f32 reference", label), any_failed)
		}

		// Bf16 sub forward + backward. Clean integer/half values that round-trip
		// exactly through bf16 so the result matches the f32 reference bit-for-bit.
		{
			N   :: 8
			N_B :: 4

			a_src := [N]f32{1, 2, 3, 4, -1, -2, -3, -4}
			b_src := [N_B]f32{1, 2, 0.5, -1}

			a_f32 := ml.tensor(a_src[:])
			b_f32 := ml.tensor(b_src[:])
			a     := ml.cast_to(a_f32, .Bf16)
			b     := ml.cast_to(b_f32, .Bf16)

			y_bf := ml.sub(a, b)
			y    := ml.cast_to(y_bf, .F32)

			got_y: [N]f32
			ml.get_data(y, got_y[:])
			expected_y: [N]f32
			for i in 0 ..< N { expected_y[i] = a_src[i] - b_src[i % N_B] }
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 sub forward matches f32 reference", label), any_failed)

			ml.backward()

			got_da: [N]f32
			got_db: [N_B]f32
			ml.get_gradient(a_f32, got_da[:])
			ml.get_gradient(b_f32, got_db[:])

			da_ok := true
			for v in got_da { if v != 1 { da_ok = false } }
			db_ok := true
			for v in got_db { if v != -f32(N / N_B) { db_ok = false } }
			check(da_ok, fmt.tprintf("%v: Bf16 sub backward da == ones", label), any_failed)
			check(db_ok, fmt.tprintf("%v: Bf16 sub backward db == -stride", label), any_failed)
		}

		// Bf16 mul forward + backward.
		{
			N   :: 8
			N_B :: 4

			a_src := [N]f32{1, 2, 3, 4, -1, -2, -3, -4}
			b_src := [N_B]f32{2, 0.5, 1, -2}

			a_f32 := ml.tensor(a_src[:])
			b_f32 := ml.tensor(b_src[:])
			a     := ml.cast_to(a_f32, .Bf16)
			b     := ml.cast_to(b_f32, .Bf16)

			y_bf := ml.mul(a, b)
			y    := ml.cast_to(y_bf, .F32)

			got_y: [N]f32
			ml.get_data(y, got_y[:])
			expected_y: [N]f32
			for i in 0 ..< N { expected_y[i] = a_src[i] * b_src[i % N_B] }
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 mul forward matches f32 reference", label), any_failed)

			ml.backward()

			got_da: [N]f32
			got_db: [N_B]f32
			ml.get_gradient(a_f32, got_da[:])
			ml.get_gradient(b_f32, got_db[:])

			expected_da: [N]f32
			for i in 0 ..< N { expected_da[i] = b_src[i % N_B] }
			expected_db: [N_B]f32
			for j in 0 ..< N_B {
				sum: f32
				for i in 0 ..< N {
					if i % N_B == j { sum += a_src[i] }
				}
				expected_db[j] = sum
			}

			da_ok := got_da == expected_da
			db_ok := got_db == expected_db
			check(da_ok, fmt.tprintf("%v: Bf16 mul backward da matches f32 reference", label), any_failed)
			check(db_ok, fmt.tprintf("%v: Bf16 mul backward db matches f32 reference", label), any_failed)
		}

		// Bf16 div forward + backward. b values avoid zero. a values and b values
		// chosen so each a/b ratio and gradient term is bf16-clean.
		{
			N   :: 8
			N_B :: 4

			a_src := [N]f32{2, 4, 6, 8, -2, -4, -6, -8}
			b_src := [N_B]f32{2, 4, 1, -2}

			a_f32 := ml.tensor(a_src[:])
			b_f32 := ml.tensor(b_src[:])
			a     := ml.cast_to(a_f32, .Bf16)
			b     := ml.cast_to(b_f32, .Bf16)

			y_bf := ml.div(a, b)
			y    := ml.cast_to(y_bf, .F32)

			got_y: [N]f32
			ml.get_data(y, got_y[:])
			expected_y: [N]f32
			for i in 0 ..< N { expected_y[i] = a_src[i] / b_src[i % N_B] }
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 div forward matches f32 reference", label), any_failed)

			ml.backward()

			got_da: [N]f32
			got_db: [N_B]f32
			ml.get_gradient(a_f32, got_da[:])
			ml.get_gradient(b_f32, got_db[:])

			expected_da: [N]f32
			for i in 0 ..< N { expected_da[i] = 1.0 / b_src[i % N_B] }
			expected_db: [N_B]f32
			for j in 0 ..< N_B {
				sum: f32
				bj_sq := b_src[j] * b_src[j]
				for i in 0 ..< N {
					if i % N_B == j { sum += -a_src[i] / bj_sq }
				}
				expected_db[j] = sum
			}

			da_ok := got_da == expected_da
			db_ok := got_db == expected_db
			check(da_ok, fmt.tprintf("%v: Bf16 div backward da matches f32 reference", label), any_failed)
			check(db_ok, fmt.tprintf("%v: Bf16 div backward db matches f32 reference", label), any_failed)
		}

		// Bf16 unary activations — relu/sigmoid/silu/tanh/gelu/exp. Compare
		// against the F32 reference run on the same (clean-bf16) input. exp/tanh/
		// sigmoid/silu/gelu are nonlinear so we tolerate small relative error;
		// relu's output and gradient are bf16-clean for these inputs.
		{
			N :: 8
			x_src := [N]f32{-2, -1, -0.5, 0, 0.5, 1, 1.5, 2}

			run_unary :: proc(name: string, label: string, tol: f32,
			                  fwd: proc(x: ml.Tensor) -> ml.Tensor,
			                  x_src: []f32, any_failed: ^bool) {
				n := len(x_src)

				ref_y  := make([]f32, n); defer delete(ref_y)
				ref_dx := make([]f32, n); defer delete(ref_dx)

				{
					x := ml.tensor(x_src)
					y := fwd(x)
					ml.get_data(y, ref_y)
					ml.backward()
					ml.get_gradient(x, ref_dx)
					ml.clear()
				}

				x_f32 := ml.tensor(x_src)
				x_bf  := ml.cast_to(x_f32, .Bf16)
				y_bf  := fwd(x_bf)
				y     := ml.cast_to(y_bf, .F32)

				got_y := make([]f32, n); defer delete(got_y)
				ml.get_data(y, got_y)

				fwd_ok := true
				for i in 0 ..< n {
					if math.abs(got_y[i] - ref_y[i]) > tol { fwd_ok = false }
				}
				check(fwd_ok, fmt.tprintf("%v: Bf16 %v forward matches f32 reference", label, name), any_failed)

				ml.backward()

				got_dx := make([]f32, n); defer delete(got_dx)
				ml.get_gradient(x_f32, got_dx)

				bwd_ok := true
				for i in 0 ..< n {
					if math.abs(got_dx[i] - ref_dx[i]) > tol { bwd_ok = false }
				}
				check(bwd_ok, fmt.tprintf("%v: Bf16 %v backward dx matches f32 reference", label, name), any_failed)

				ml.clear()
			}

			run_unary("relu",    label, 0.0,  proc(x: ml.Tensor) -> ml.Tensor { return ml.relu(x) },    x_src[:], any_failed)
			run_unary("sigmoid", label, 0.05, proc(x: ml.Tensor) -> ml.Tensor { return ml.sigmoid(x) }, x_src[:], any_failed)
			run_unary("silu",    label, 0.05, proc(x: ml.Tensor) -> ml.Tensor { return ml.silu(x) },    x_src[:], any_failed)
			run_unary("tanh",    label, 0.05, proc(x: ml.Tensor) -> ml.Tensor { return ml.tanh(x) },    x_src[:], any_failed)
			run_unary("gelu",    label, 0.05, proc(x: ml.Tensor) -> ml.Tensor { return ml.gelu(x) },    x_src[:], any_failed)
			run_unary("exp",     label, 0.2,  proc(x: ml.Tensor) -> ml.Tensor { return ml.exp(x) },     x_src[:], any_failed)
		}

		// Bf16 softmax / log_softmax / layernorm / entropy — compared against
		// f32 reference on the same input. Tolerance covers the bf16 round-off
		// in the reductions and stored outputs.
		{
			run_softmax_like :: proc(name: string, label: string, tol: f32,
			                         fwd: proc(x: ml.Tensor) -> ml.Tensor, any_failed: ^bool) {
				count :: 4
				size  :: 8
				x_src: [count * size]f32
				for i in 0 ..< count * size { x_src[i] = f32(((i + 1) % 7) - 3) * 0.5 }

				ref_y, ref_dx: [count * size]f32
				{
					x := ml.tensor(x_src[:])
					x  = ml.reshape(x, {count, size})
					y := fwd(x)
					ml.get_data(y, ref_y[:])
					ml.backward()
					ml.get_gradient(x, ref_dx[:])
					ml.clear()
				}

				x_f32 := ml.tensor(x_src[:])
				x_f32  = ml.reshape(x_f32, {count, size})
				x_bf  := ml.cast_to(x_f32, .Bf16)
				y_bf  := fwd(x_bf)
				y     := ml.cast_to(y_bf, .F32)

				got_y: [count * size]f32
				ml.get_data(y, got_y[:])
				fwd_ok := true
				for i in 0 ..< count * size {
					if math.abs(got_y[i] - ref_y[i]) > tol { fwd_ok = false }
				}
				check(fwd_ok, fmt.tprintf("%v: Bf16 %v forward matches f32 reference", label, name), any_failed)

				ml.backward()
				got_dx: [count * size]f32
				ml.get_gradient(x_f32, got_dx[:])
				bwd_ok := true
				for i in 0 ..< count * size {
					if math.abs(got_dx[i] - ref_dx[i]) > tol { bwd_ok = false }
				}
				check(bwd_ok, fmt.tprintf("%v: Bf16 %v backward dx matches f32 reference", label, name), any_failed)

				ml.clear()
			}

			run_softmax_like("softmax",     label, 0.05, proc(x: ml.Tensor) -> ml.Tensor { return ml.softmax(x) },     any_failed)
			run_softmax_like("log_softmax", label, 0.1,  proc(x: ml.Tensor) -> ml.Tensor { return ml.log_softmax(x) }, any_failed)

			// entropy needs probabilities (non-negative, sum to ~1). Build via
			// softmax of clean f32 input.
			{
				count :: 4
				size  :: 8
				x_src: [count * size]f32
				for i in 0 ..< count * size { x_src[i] = f32(((i + 1) % 5) - 2) * 0.5 }

				ref_y, ref_dp: [count * size]f32
				ref_y_out: [count]f32
				_ = ref_y
				{
					x := ml.tensor(x_src[:])
					x  = ml.reshape(x, {count, size})
					p := ml.softmax(x)
					y := ml.entropy(p)
					ml.get_data(y, ref_y_out[:])
					ml.backward()
					ml.get_gradient(x, ref_dp[:])
					ml.clear()
				}

				x_f32 := ml.tensor(x_src[:])
				x_f32  = ml.reshape(x_f32, {count, size})
				x_bf  := ml.cast_to(x_f32, .Bf16)
				p_bf  := ml.softmax(x_bf)
				y_bf  := ml.entropy(p_bf)
				y     := ml.cast_to(y_bf, .F32)

				got_y_out: [count]f32
				ml.get_data(y, got_y_out[:])
				fwd_ok := true
				for i in 0 ..< count {
					if math.abs(got_y_out[i] - ref_y_out[i]) > 0.05 { fwd_ok = false }
				}
				check(fwd_ok, fmt.tprintf("%v: Bf16 entropy forward matches f32 reference", label), any_failed)

				ml.backward()
				got_dp: [count * size]f32
				ml.get_gradient(x_f32, got_dp[:])
				bwd_ok := true
				for i in 0 ..< count * size {
					if math.abs(got_dp[i] - ref_dp[i]) > 0.1 { bwd_ok = false }
				}
				check(bwd_ok, fmt.tprintf("%v: Bf16 entropy backward dp matches f32 reference", label), any_failed)
			}

			// Layernorm: needs an extra weight tensor.
			{
				count :: 4
				size  :: 8
				x_src: [count * size]f32
				w_src: [size]f32
				for i in 0 ..< count * size { x_src[i] = f32(((i + 1) % 7) - 3) * 0.5 }
				for i in 0 ..< size { w_src[i] = f32((i % 3) - 1) }

				ref_y:  [count * size]f32
				ref_dx: [count * size]f32
				ref_dw: [size]f32
				{
					x := ml.tensor(x_src[:])
					x  = ml.reshape(x, {count, size})
					w := ml.tensor(w_src[:])
					y := ml.layernorm(x, w)
					ml.get_data(y, ref_y[:])
					ml.backward()
					ml.get_gradient(x, ref_dx[:])
					ml.get_gradient(w, ref_dw[:])
					ml.clear()
				}

				x_f32 := ml.tensor(x_src[:])
				x_f32  = ml.reshape(x_f32, {count, size})
				w_f32 := ml.tensor(w_src[:])
				x_bf  := ml.cast_to(x_f32, .Bf16)
				w_bf  := ml.cast_to(w_f32, .Bf16)
				y_bf  := ml.layernorm(x_bf, w_bf)
				y     := ml.cast_to(y_bf, .F32)

				got_y: [count * size]f32
				ml.get_data(y, got_y[:])
				fwd_ok := true
				for i in 0 ..< count * size {
					if math.abs(got_y[i] - ref_y[i]) > 0.05 { fwd_ok = false }
				}
				check(fwd_ok, fmt.tprintf("%v: Bf16 layernorm forward matches f32 reference", label), any_failed)

				ml.backward()
				got_dx: [count * size]f32
				got_dw: [size]f32
				ml.get_gradient(x_f32, got_dx[:])
				ml.get_gradient(w_f32, got_dw[:])
				dx_ok := true
				for i in 0 ..< count * size {
					if math.abs(got_dx[i] - ref_dx[i]) > 0.1 { dx_ok = false }
				}
				dw_ok := true
				for i in 0 ..< size {
					if math.abs(got_dw[i] - ref_dw[i]) > 0.1 { dw_ok = false }
				}
				check(dx_ok, fmt.tprintf("%v: Bf16 layernorm backward dx matches f32 reference", label), any_failed)
				check(dw_ok, fmt.tprintf("%v: Bf16 layernorm backward dw matches f32 reference", label), any_failed)
			}

			// Rmsnorm: same shape as layernorm, no mean term.
			{
				count :: 4
				size  :: 8
				x_src: [count * size]f32
				w_src: [size]f32
				for i in 0 ..< count * size { x_src[i] = f32(((i + 1) % 7) - 3) * 0.5 }
				for i in 0 ..< size { w_src[i] = f32((i % 3) - 1) }

				ref_y:  [count * size]f32
				ref_dx: [count * size]f32
				ref_dw: [size]f32
				{
					x := ml.tensor(x_src[:])
					x  = ml.reshape(x, {count, size})
					w := ml.tensor(w_src[:])
					y := ml.rmsnorm(x, w)
					ml.get_data(y, ref_y[:])
					ml.backward()
					ml.get_gradient(x, ref_dx[:])
					ml.get_gradient(w, ref_dw[:])
					ml.clear()
				}

				x_f32 := ml.tensor(x_src[:])
				x_f32  = ml.reshape(x_f32, {count, size})
				w_f32 := ml.tensor(w_src[:])
				x_bf  := ml.cast_to(x_f32, .Bf16)
				w_bf  := ml.cast_to(w_f32, .Bf16)
				y_bf  := ml.rmsnorm(x_bf, w_bf)
				y     := ml.cast_to(y_bf, .F32)

				got_y: [count * size]f32
				ml.get_data(y, got_y[:])
				fwd_ok := true
				for i in 0 ..< count * size {
					if math.abs(got_y[i] - ref_y[i]) > 0.05 { fwd_ok = false }
				}
				check(fwd_ok, fmt.tprintf("%v: Bf16 rmsnorm forward matches f32 reference", label), any_failed)

				ml.backward()
				got_dx: [count * size]f32
				got_dw: [size]f32
				ml.get_gradient(x_f32, got_dx[:])
				ml.get_gradient(w_f32, got_dw[:])
				dx_ok := true
				for i in 0 ..< count * size {
					if math.abs(got_dx[i] - ref_dx[i]) > 0.1 { dx_ok = false }
				}
				dw_ok := true
				for i in 0 ..< size {
					if math.abs(got_dw[i] - ref_dw[i]) > 0.1 { dw_ok = false }
				}
				check(dx_ok, fmt.tprintf("%v: Bf16 rmsnorm backward dx matches f32 reference", label), any_failed)
				check(dw_ok, fmt.tprintf("%v: Bf16 rmsnorm backward dw matches f32 reference", label), any_failed)
			}
		}

		// Bf16 attention forward + backward — compared against an F32 attention
		// run on the same (clean-bf16) input. softmax/exp make exact bit-match
		// impossible, so we tolerate small relative error.
		{
			T :: 4
			HEADS :: 2
			D :: 4
			E :: HEADS * D

			input_src: [T * 3 * E]f32
			for i in 0 ..< T * 3 * E {
				input_src[i] = f32(((i + 1) % 5) - 2) * 0.5
			}

			// F32 reference forward + backward.
			ref_out: [T * E]f32
			ref_dx:  [T * 3 * E]f32
			{
				x := ml.tensor(input_src[:])
				x  = ml.reshape(x, {T, 3 * E})
				q := ml.slice_trailing(x, 0,     E)
				k := ml.slice_trailing(x, E,     2 * E)
				v := ml.slice_trailing(x, 2 * E, 3 * E)
				y := ml.attention(q, k, v, HEADS, causal=true)
				ml.get_data(y, ref_out[:])
				ml.backward()
				ml.get_gradient(x, ref_dx[:])
				ml.clear()
			}

			// Bf16 forward + backward.
			x_f32 := ml.tensor(input_src[:])
			x_f32  = ml.reshape(x_f32, {T, 3 * E})
			x_bf  := ml.cast_to(x_f32, .Bf16)
			q_bf  := ml.slice_trailing(x_bf, 0,     E)
			k_bf  := ml.slice_trailing(x_bf, E,     2 * E)
			v_bf  := ml.slice_trailing(x_bf, 2 * E, 3 * E)
			y_bf  := ml.attention(q_bf, k_bf, v_bf, HEADS, causal=true)
			y     := ml.cast_to(y_bf, .F32)

			got_y: [T * E]f32
			ml.get_data(y, got_y[:])

			tol :: f32(5e-2)
			fwd_ok := true
			for i in 0 ..< T * E {
				if math.abs(got_y[i] - ref_out[i]) > tol { fwd_ok = false }
			}
			check(fwd_ok, fmt.tprintf("%v: Bf16 attention forward matches f32 reference", label), any_failed)

			ml.backward()

			got_dx: [T * 3 * E]f32
			ml.get_gradient(x_f32, got_dx[:])

			bwd_ok := true
			for i in 0 ..< T * 3 * E {
				if math.abs(got_dx[i] - ref_dx[i]) > tol { bwd_ok = false }
			}
			check(bwd_ok, fmt.tprintf("%v: Bf16 attention backward dx matches f32 reference", label), any_failed)
		}

		// Bf16 attention forward at coopmat shape: head_size=16 (the smallest
		// shape the coopmat shader accepts), T=32 covers 2 query-blocks at
		// BR=16, exercising the WG.y > 1 path. Backward still falls back to
		// the SIMT bf16 shaders (coopmat backward is not yet implemented).
		{
			T     :: 32
			HEADS :: 2
			D     :: 16
			E     :: HEADS * D

			input_src: [T * 3 * E]f32
			for i in 0 ..< T * 3 * E {
				input_src[i] = f32(((i + 1) % 5) - 2) * 0.25
			}

			ref_out: [T * E]f32
			{
				x := ml.tensor(input_src[:])
				x  = ml.reshape(x, {T, 3 * E})
				q := ml.slice_trailing(x, 0,     E)
				k := ml.slice_trailing(x, E,     2 * E)
				v := ml.slice_trailing(x, 2 * E, 3 * E)
				y := ml.attention(q, k, v, HEADS, causal=true)
				ml.get_data(y, ref_out[:])
			}

			x_f32 := ml.tensor(input_src[:])
			x_f32  = ml.reshape(x_f32, {T, 3 * E})
			x_bf  := ml.cast_to(x_f32, .Bf16)
			q_bf  := ml.slice_trailing(x_bf, 0,     E)
			k_bf  := ml.slice_trailing(x_bf, E,     2 * E)
			v_bf  := ml.slice_trailing(x_bf, 2 * E, 3 * E)
			y_bf  := ml.attention(q_bf, k_bf, v_bf, HEADS, causal=true)
			y     := ml.cast_to(y_bf, .F32)

			got_y: [T * E]f32
			ml.get_data(y, got_y[:])

			tol :: f32(5e-2)
			fwd_ok := true
			for i in 0 ..< T * E {
				if math.abs(got_y[i] - ref_out[i]) > tol { fwd_ok = false }
			}
			check(fwd_ok, fmt.tprintf("%v: Bf16 attention (coopmat-shape) forward matches f32 reference", label), any_failed)
		}

		// Bf16 GQA attention: 4 q-heads, 2 kv-heads (group_size=2).
		{
			T     :: 8
			N_Q   :: 4
			N_KV  :: 2
			D     :: 4
			Q_E   :: N_Q  * D
			KV_E  :: N_KV * D

			q_src: [T * Q_E]f32
			k_src: [T * KV_E]f32
			v_src: [T * KV_E]f32
			for i in 0 ..< T * Q_E  { q_src[i] = f32(((i + 1) % 5) - 2) * 0.5 }
			for i in 0 ..< T * KV_E { k_src[i] = f32(((i + 3) % 5) - 2) * 0.5 }
			for i in 0 ..< T * KV_E { v_src[i] = f32(((i + 2) % 5) - 2) * 0.5 }

			ref_out: [T * Q_E]f32
			ref_dq:  [T * Q_E]f32
			ref_dk:  [T * KV_E]f32
			ref_dv:  [T * KV_E]f32
			{
				q := ml.reshape(ml.tensor(q_src[:]), {T, Q_E})
				k := ml.reshape(ml.tensor(k_src[:]), {T, KV_E})
				v := ml.reshape(ml.tensor(v_src[:]), {T, KV_E})
				y := ml.attention(q, k, v, N_Q, N_KV, causal=true)
				ml.get_data(y, ref_out[:])
				ml.backward()
				ml.get_gradient(q, ref_dq[:])
				ml.get_gradient(k, ref_dk[:])
				ml.get_gradient(v, ref_dv[:])
				ml.clear()
			}

			q_f32 := ml.reshape(ml.tensor(q_src[:]), {T, Q_E})
			k_f32 := ml.reshape(ml.tensor(k_src[:]), {T, KV_E})
			v_f32 := ml.reshape(ml.tensor(v_src[:]), {T, KV_E})
			q_bf  := ml.cast_to(q_f32, .Bf16)
			k_bf  := ml.cast_to(k_f32, .Bf16)
			v_bf  := ml.cast_to(v_f32, .Bf16)
			y_bf  := ml.attention(q_bf, k_bf, v_bf, N_Q, N_KV, causal=true)
			y     := ml.cast_to(y_bf, .F32)

			got_out: [T * Q_E]f32
			ml.get_data(y, got_out[:])

			tol :: f32(5e-2)
			fwd_ok := true
			for i in 0 ..< T * Q_E {
				if math.abs(got_out[i] - ref_out[i]) > tol { fwd_ok = false }
			}
			check(fwd_ok, fmt.tprintf("%v: Bf16 GQA attention forward matches f32 reference", label), any_failed)

			ml.backward()
			got_dq: [T * Q_E]f32
			got_dk: [T * KV_E]f32
			got_dv: [T * KV_E]f32
			ml.get_gradient(q_f32, got_dq[:])
			ml.get_gradient(k_f32, got_dk[:])
			ml.get_gradient(v_f32, got_dv[:])

			dq_ok := true
			for i in 0 ..< T * Q_E  { if math.abs(got_dq[i] - ref_dq[i]) > tol { dq_ok = false } }
			dk_ok := true
			for i in 0 ..< T * KV_E { if math.abs(got_dk[i] - ref_dk[i]) > tol { dk_ok = false } }
			dv_ok := true
			for i in 0 ..< T * KV_E { if math.abs(got_dv[i] - ref_dv[i]) > tol { dv_ok = false } }
			check(dq_ok, fmt.tprintf("%v: Bf16 GQA attention backward dq matches f32 reference", label), any_failed)
			check(dk_ok, fmt.tprintf("%v: Bf16 GQA attention backward dk matches f32 reference", label), any_failed)
			check(dv_ok, fmt.tprintf("%v: Bf16 GQA attention backward dv matches f32 reference", label), any_failed)
		}
		ml.clear()

		// attention_with_cache: prefill then per-token decode through the
		// cache should match a full forward pass. Covers both F32 and Bf16
		// on whichever backend is being exercised.
		for cache_type in ([?]ml.Data_Type{.F32, .Bf16}) {
			T_TOTAL  :: 6
			T_PREFIX :: 3
			N_Q      :: 4
			N_KV     :: 2
			D        :: 4
			Q_E      :: N_Q  * D
			KV_E     :: N_KV * D

			q_src: [T_TOTAL * Q_E]f32
			k_src: [T_TOTAL * KV_E]f32
			v_src: [T_TOTAL * KV_E]f32
			for i in 0 ..< T_TOTAL * Q_E  { q_src[i] = f32(((i + 1) % 5) - 2) * 0.5 }
			for i in 0 ..< T_TOTAL * KV_E { k_src[i] = f32(((i + 3) % 5) - 2) * 0.5 }
			for i in 0 ..< T_TOTAL * KV_E { v_src[i] = f32(((i + 2) % 5) - 2) * 0.5 }

			ref_out: [T_TOTAL * Q_E]f32
			{
				q := ml.reshape(ml.tensor(q_src[:]), {T_TOTAL, Q_E})
				k := ml.reshape(ml.tensor(k_src[:]), {T_TOTAL, KV_E})
				v := ml.reshape(ml.tensor(v_src[:]), {T_TOTAL, KV_E})
				if cache_type == .Bf16 {
					q = ml.cast_to(q, .Bf16)
					k = ml.cast_to(k, .Bf16)
					v = ml.cast_to(v, .Bf16)
				}
				y := ml.attention(q, k, v, N_Q, N_KV, causal=true)
				if cache_type == .Bf16 {
					y = ml.cast_to(y, .F32)
				}
				ml.get_data(y, ref_out[:])
				ml.clear()
			}

			k_cache := ml.alloc(cache_type, {T_TOTAL, KV_E}, persistent=true, buffers={.Data})
			v_cache := ml.alloc(cache_type, {T_TOTAL, KV_E}, persistent=true, buffers={.Data})

			got_out: [T_TOTAL * Q_E]f32

			{
				q := ml.reshape(ml.tensor(q_src[: T_PREFIX * Q_E]),  {T_PREFIX, Q_E})
				k := ml.reshape(ml.tensor(k_src[: T_PREFIX * KV_E]), {T_PREFIX, KV_E})
				v := ml.reshape(ml.tensor(v_src[: T_PREFIX * KV_E]), {T_PREFIX, KV_E})
				if cache_type == .Bf16 {
					q = ml.cast_to(q, .Bf16)
					k = ml.cast_to(k, .Bf16)
					v = ml.cast_to(v, .Bf16)
				}
				y := ml.attention_with_cache(q, k, v, k_cache, v_cache, 0, N_Q, N_KV)
				if cache_type == .Bf16 {
					y = ml.cast_to(y, .F32)
				}
				ml.get_data(y, got_out[: T_PREFIX * Q_E])
				ml.clear()
			}

			for step in 0 ..< (T_TOTAL - T_PREFIX) {
				pos    := T_PREFIX + step
				q_step: [Q_E]f32
				k_step: [KV_E]f32
				v_step: [KV_E]f32
				copy(q_step[:], q_src[pos * Q_E  : (pos + 1) * Q_E ])
				copy(k_step[:], k_src[pos * KV_E : (pos + 1) * KV_E])
				copy(v_step[:], v_src[pos * KV_E : (pos + 1) * KV_E])

				q := ml.reshape(ml.tensor(q_step[:]), {1, Q_E})
				k := ml.reshape(ml.tensor(k_step[:]), {1, KV_E})
				v := ml.reshape(ml.tensor(v_step[:]), {1, KV_E})
				if cache_type == .Bf16 {
					q = ml.cast_to(q, .Bf16)
					k = ml.cast_to(k, .Bf16)
					v = ml.cast_to(v, .Bf16)
				}
				y := ml.attention_with_cache(q, k, v, k_cache, v_cache, pos, N_Q, N_KV)
				if cache_type == .Bf16 {
					y = ml.cast_to(y, .F32)
				}
				ml.get_data(y, got_out[pos * Q_E : (pos + 1) * Q_E])
				ml.clear()
			}

			tol := cache_type == .Bf16 ? f32(5e-2) : f32(1e-4)
			ok := true
			for i in 0 ..< T_TOTAL * Q_E {
				if math.abs(got_out[i] - ref_out[i]) > tol { ok = false }
			}
			check(ok, fmt.tprintf("%v: %v attention_with_cache prefill+decode matches full attention", label, cache_type), any_failed)

			ml.destroy(k_cache)
			ml.destroy(v_cache)
		}

		for window_type in ([?]ml.Data_Type{.F32, .Bf16}) {
			T      :: 8
			N_Q    :: 4
			N_KV   :: 2
			D      :: 4
			Q_E    :: N_Q  * D
			KV_E   :: N_KV * D
			WINDOW :: 3

			q_src: [T * Q_E]f32
			k_src: [T * KV_E]f32
			v_src: [T * KV_E]f32
			for i in 0 ..< T * Q_E  { q_src[i] = f32(((i + 1) % 7) - 3) * 0.3 }
			for i in 0 ..< T * KV_E { k_src[i] = f32(((i + 4) % 7) - 3) * 0.3 }
			for i in 0 ..< T * KV_E { v_src[i] = f32(((i + 2) % 7) - 3) * 0.3 }

			ref_out: [T * Q_E]f32
			{
				q := ml.reshape(ml.tensor(q_src[:]), {T, Q_E})
				k := ml.reshape(ml.tensor(k_src[:]), {T, KV_E})
				v := ml.reshape(ml.tensor(v_src[:]), {T, KV_E})
				if window_type == .Bf16 {
					q = ml.cast_to(q, .Bf16)
					k = ml.cast_to(k, .Bf16)
					v = ml.cast_to(v, .Bf16)
				}
				y := ml.attention(q, k, v, N_Q, N_KV, causal=true, window=WINDOW)
				if window_type == .Bf16 {
					y = ml.cast_to(y, .F32)
				}
				ml.get_data(y, ref_out[:])
				ml.clear()
			}

			k_cache := ml.alloc(window_type, {T, KV_E}, persistent=true, buffers={.Data})
			v_cache := ml.alloc(window_type, {T, KV_E}, persistent=true, buffers={.Data})

			got_out: [T * Q_E]f32
			for pos in 0 ..< T {
				q_step: [Q_E]f32
				k_step: [KV_E]f32
				v_step: [KV_E]f32
				copy(q_step[:], q_src[pos * Q_E  : (pos + 1) * Q_E ])
				copy(k_step[:], k_src[pos * KV_E : (pos + 1) * KV_E])
				copy(v_step[:], v_src[pos * KV_E : (pos + 1) * KV_E])

				q := ml.reshape(ml.tensor(q_step[:]), {1, Q_E})
				k := ml.reshape(ml.tensor(k_step[:]), {1, KV_E})
				v := ml.reshape(ml.tensor(v_step[:]), {1, KV_E})
				if window_type == .Bf16 {
					q = ml.cast_to(q, .Bf16)
					k = ml.cast_to(k, .Bf16)
					v = ml.cast_to(v, .Bf16)
				}
				y := ml.attention_with_cache(q, k, v, k_cache, v_cache, pos, N_Q, N_KV, window=WINDOW)
				if window_type == .Bf16 {
					y = ml.cast_to(y, .F32)
				}
				ml.get_data(y, got_out[pos * Q_E : (pos + 1) * Q_E])
				ml.clear()
			}

			tol := window_type == .Bf16 ? f32(5e-2) : f32(1e-4)
			ok := true
			for i in 0 ..< T * Q_E {
				if math.abs(got_out[i] - ref_out[i]) > tol { ok = false }
			}
			check(ok, fmt.tprintf("%v: %v attention sliding-window cache matches non-cache reference", label, window_type), any_failed)

			ml.destroy(k_cache)
			ml.destroy(v_cache)
		}

		// attention_with_cache ring-buffer: same sliding-window scenario but
		// the cache is allocated at exactly `WINDOW` rows so writes wrap
		// repeatedly. Exercises the t_capacity modulo path on both backends.
		for ring_type in ([?]ml.Data_Type{.F32, .Bf16}) {
			T      :: 8
			N_Q    :: 4
			N_KV   :: 2
			D      :: 4
			Q_E    :: N_Q  * D
			KV_E   :: N_KV * D
			WINDOW :: 3

			q_src: [T * Q_E]f32
			k_src: [T * KV_E]f32
			v_src: [T * KV_E]f32
			for i in 0 ..< T * Q_E  { q_src[i] = f32(((i + 1) % 7) - 3) * 0.3 }
			for i in 0 ..< T * KV_E { k_src[i] = f32(((i + 4) % 7) - 3) * 0.3 }
			for i in 0 ..< T * KV_E { v_src[i] = f32(((i + 2) % 7) - 3) * 0.3 }

			ref_out: [T * Q_E]f32
			{
				q := ml.reshape(ml.tensor(q_src[:]), {T, Q_E})
				k := ml.reshape(ml.tensor(k_src[:]), {T, KV_E})
				v := ml.reshape(ml.tensor(v_src[:]), {T, KV_E})
				if ring_type == .Bf16 {
					q = ml.cast_to(q, .Bf16)
					k = ml.cast_to(k, .Bf16)
					v = ml.cast_to(v, .Bf16)
				}
				y := ml.attention(q, k, v, N_Q, N_KV, causal=true, window=WINDOW)
				if ring_type == .Bf16 {
					y = ml.cast_to(y, .F32)
				}
				ml.get_data(y, ref_out[:])
				ml.clear()
			}

			k_cache := ml.alloc(ring_type, {WINDOW, KV_E}, persistent=true, buffers={.Data})
			v_cache := ml.alloc(ring_type, {WINDOW, KV_E}, persistent=true, buffers={.Data})

			got_out: [T * Q_E]f32
			for pos in 0 ..< T {
				q_step: [Q_E]f32
				k_step: [KV_E]f32
				v_step: [KV_E]f32
				copy(q_step[:], q_src[pos * Q_E  : (pos + 1) * Q_E ])
				copy(k_step[:], k_src[pos * KV_E : (pos + 1) * KV_E])
				copy(v_step[:], v_src[pos * KV_E : (pos + 1) * KV_E])

				q := ml.reshape(ml.tensor(q_step[:]), {1, Q_E})
				k := ml.reshape(ml.tensor(k_step[:]), {1, KV_E})
				v := ml.reshape(ml.tensor(v_step[:]), {1, KV_E})
				if ring_type == .Bf16 {
					q = ml.cast_to(q, .Bf16)
					k = ml.cast_to(k, .Bf16)
					v = ml.cast_to(v, .Bf16)
				}
				y := ml.attention_with_cache(q, k, v, k_cache, v_cache, pos, N_Q, N_KV, window=WINDOW)
				if ring_type == .Bf16 {
					y = ml.cast_to(y, .F32)
				}
				ml.get_data(y, got_out[pos * Q_E : (pos + 1) * Q_E])
				ml.clear()
			}

			tol := ring_type == .Bf16 ? f32(5e-2) : f32(1e-4)
			ok := true
			for i in 0 ..< T * Q_E {
				if math.abs(got_out[i] - ref_out[i]) > tol { ok = false }
			}
			check(ok, fmt.tprintf("%v: %v attention sliding-window ring cache (capacity=window) matches reference", label, ring_type), any_failed)

			ml.destroy(k_cache)
			ml.destroy(v_cache)
		}

		// Bf16 mean forward + backward.
		{
			N    :: 12
			SIZE :: 4
			x_src: [N]f32
			for i in 0 ..< N {
				x_src[i] = f32(i)
			}
			x_f32 := ml.tensor(x_src[:])
			x_r   := ml.reshape(x_f32, {N / SIZE, SIZE})
			x_bf  := ml.cast_to(x_r, .Bf16)
			y_bf  := ml.mean(x_bf)
			y     := ml.cast_to(y_bf, .F32)

			got_y: [N / SIZE]f32
			ml.get_data(y, got_y[:])
			expected_y := [N / SIZE]f32{
				(0 + 1 + 2 + 3) / 4.0,
				(4 + 5 + 6 + 7) / 4.0,
				(8 + 9 + 10 + 11) / 4.0,
			}
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 mean forward matches", label), any_failed)

			ml.backward()
			got_dx: [N]f32
			ml.get_gradient(x_f32, got_dx[:])
			dx_ok := true
			for v in got_dx {
				if math.abs(v - 1.0 / f32(SIZE)) > 1e-3 { dx_ok = false }
			}
			check(dx_ok, fmt.tprintf("%v: Bf16 mean backward dx == 1/size", label), any_failed)
		}
		ml.clear()

		// Bf16 slice forward + backward.
		{
			N    :: 8
			S    :: 2
			E    :: 6
			x_src := [N]f32{1, 2, 3, 4, 5, 6, 7, 8}
			x_f32 := ml.tensor(x_src[:])
			x_bf  := ml.cast_to(x_f32, .Bf16)
			y_bf  := ml.slice(x_bf, S, E)
			y     := ml.cast_to(y_bf, .F32)

			got_y: [E - S]f32
			ml.get_data(y, got_y[:])
			expected_y := [E - S]f32{3, 4, 5, 6}
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 slice forward matches", label), any_failed)

			ml.backward()
			got_dx: [N]f32
			ml.get_gradient(x_f32, got_dx[:])
			expected_dx := [N]f32{0, 0, 1, 1, 1, 1, 0, 0}
			dx_ok := got_dx == expected_dx
			check(dx_ok, fmt.tprintf("%v: Bf16 slice backward dx matches", label), any_failed)
		}
		ml.clear()

		// Bf16 concat forward + backward (rank-2, 3 inputs).
		{
			LEAD :: 2
			TA   :: 2
			TB   :: 4
			TC   :: 2
			a_src := [LEAD * TA]f32{1, 2, 5, 6}
			b_src := [LEAD * TB]f32{3, 4, 5, 6, 7, 8, 9, 10}
			c_src := [LEAD * TC]f32{11, 12, 13, 14}

			a_f32 := ml.reshape(ml.tensor(a_src[:]), {LEAD, TA})
			b_f32 := ml.reshape(ml.tensor(b_src[:]), {LEAD, TB})
			c_f32 := ml.reshape(ml.tensor(c_src[:]), {LEAD, TC})
			a_bf  := ml.cast_to(a_f32, .Bf16)
			b_bf  := ml.cast_to(b_f32, .Bf16)
			c_bf  := ml.cast_to(c_f32, .Bf16)
			y_bf  := ml.concat(a_bf, b_bf, c_bf)
			y     := ml.cast_to(y_bf, .F32)

			OUT_T :: TA + TB + TC
			got_y: [LEAD * OUT_T]f32
			ml.get_data(y, got_y[:])
			expected_y := [LEAD * OUT_T]f32{
				1, 2, 3, 4, 5, 6, 11, 12,
				5, 6, 7, 8, 9, 10, 13, 14,
			}
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 concat3 forward matches", label), any_failed)

			ml.backward()
			got_da: [LEAD * TA]f32
			got_db: [LEAD * TB]f32
			got_dc: [LEAD * TC]f32
			ml.get_gradient(a_f32, got_da[:])
			ml.get_gradient(b_f32, got_db[:])
			ml.get_gradient(c_f32, got_dc[:])
			ok := true
			for v in got_da { if v != 1 { ok = false } }
			for v in got_db { if v != 1 { ok = false } }
			for v in got_dc { if v != 1 { ok = false } }
			check(ok, fmt.tprintf("%v: Bf16 concat3 backward dx == ones", label), any_failed)
		}
		ml.clear()

		// Bf16 rope forward + backward.
		{
			TC :: 4
			HC :: 2
			HS :: 4
			N  :: TC * HC * HS

			x_src: [N]f32
			for i in 0 ..< N {
				x_src[i] = f32((i % 5) - 2) * 0.5
			}

			rope_ref :: proc(x: [N]f32, base: f32) -> [N]f32 {
				out: [N]f32
				for t in 0 ..< TC {
					for h in 0 ..< HC {
						head_off := t * HC * HS + h * HS
						for i in 0 ..< HS / 2 {
							theta := f32(t) / math.pow(base, f32(i * 2) / f32(HS))
							cv := math.cos(theta); sv := math.sin(theta)
							xv := x[head_off + i * 2]
							yv := x[head_off + i * 2 + 1]
							out[head_off + i * 2]     = xv * cv - yv * sv
							out[head_off + i * 2 + 1] = xv * sv + yv * cv
						}
					}
				}
				return out
			}

			BASE :: f32(10000)
			ref_y := rope_ref(x_src, BASE)

			x_f32 := ml.tensor(x_src[:])
			x_r   := ml.reshape(x_f32, {TC, HC * HS})
			x_bf  := ml.cast_to(x_r, .Bf16)
			y_bf  := ml.rope(x_bf, HC, BASE)
			y     := ml.cast_to(y_bf, .F32)

			got_y: [N]f32
			ml.get_data(y, got_y[:])
			tol :: f32(5e-2)
			fwd_ok := true
			for i in 0 ..< N {
				if math.abs(got_y[i] - ref_y[i]) > tol { fwd_ok = false }
			}
			check(fwd_ok, fmt.tprintf("%v: Bf16 rope forward matches f32 reference", label), any_failed)

			ml.backward()
			got_dx: [N]f32
			ml.get_gradient(x_f32, got_dx[:])
			// dy = ones; the bwd rotation maps (1,1) → (cos+sin, -sin+cos).
			ref_dx: [N]f32
			for t in 0 ..< TC {
				for h in 0 ..< HC {
					head_off := t * HC * HS + h * HS
					for i in 0 ..< HS / 2 {
						theta := f32(t) / math.pow(BASE, f32(i * 2) / f32(HS))
						cv := math.cos(theta); sv := math.sin(theta)
						ref_dx[head_off + i * 2]     =  cv + sv
						ref_dx[head_off + i * 2 + 1] = -sv + cv
					}
				}
			}
			bwd_ok := true
			for i in 0 ..< N {
				if math.abs(got_dx[i] - ref_dx[i]) > tol { bwd_ok = false }
			}
			check(bwd_ok, fmt.tprintf("%v: Bf16 rope backward matches f32 reference", label), any_failed)
		}
		ml.clear()

		// Bf16 permute forward + backward.
		{
			D0 :: 2
			D1 :: 3
			D2 :: 4
			N  :: D0 * D1 * D2
			x_src: [N]f32
			for i in 0 ..< N {
				x_src[i] = f32(i)
			}

			x_f32 := ml.tensor(x_src[:])
			x_r   := ml.reshape(x_f32, {D0, D1, D2})
			x_bf  := ml.cast_to(x_r, .Bf16)
			y_bf  := ml.permute(x_bf, [3]int{1, 0, 2})
			y     := ml.cast_to(y_bf, .F32)

			got_y: [N]f32
			ml.get_data(y, got_y[:])
			expected_y: [N]f32
			for i0 in 0 ..< D1 {
				for i1 in 0 ..< D0 {
					for i2 in 0 ..< D2 {
						src_idx := i1 * D1 * D2 + i0 * D2 + i2
						dst_idx := i0 * D0 * D2 + i1 * D2 + i2
						expected_y[dst_idx] = x_src[src_idx]
					}
				}
			}
			fwd_ok := got_y == expected_y
			check(fwd_ok, fmt.tprintf("%v: Bf16 permute forward matches", label), any_failed)

			ml.backward()
			got_dx: [N]f32
			ml.get_gradient(x_f32, got_dx[:])
			ok := true
			for v in got_dx { if v != 1 { ok = false } }
			check(ok, fmt.tprintf("%v: Bf16 permute backward dx == ones", label), any_failed)
		}
		ml.clear()

		// Bf16 causal_mask forward + backward.
		{
			T  :: 4
			N  :: T * T
			x_src: [N]f32
			for i in 0 ..< N {
				x_src[i] = f32(i + 1)
			}

			x_f32 := ml.tensor(x_src[:])
			x_r   := ml.reshape(x_f32, {T, T})
			x_bf  := ml.cast_to(x_r, .Bf16)
			y_bf  := ml.causal_mask(x_bf)
			// Cast back to F32 for an F32 loss; -inf bf16 stays -inf in F32.
			y     := ml.cast_to(y_bf, .F32)

			got_y: [N]f32
			ml.get_data(y, got_y[:])
			ok := true
			for t1 in 0 ..< T {
				for t2 in 0 ..< T {
					idx := t1 * T + t2
					if t2 <= t1 {
						if got_y[idx] != x_src[idx] { ok = false }
					} else {
						if !math.is_inf(got_y[idx], -1) { ok = false }
					}
				}
			}
			check(ok, fmt.tprintf("%v: Bf16 causal_mask forward matches", label), any_failed)

			// backward seeds dy = ones, but unmasked positions hold -inf and
			// cast_to backward propagates ones regardless of forward value, so
			// the mask backward should add 1 only to lower-triangular dx.
			ml.backward()
			got_dx: [N]f32
			ml.get_gradient(x_f32, got_dx[:])
			bwd_ok := true
			for t1 in 0 ..< T {
				for t2 in 0 ..< T {
					idx := t1 * T + t2
					expected: f32 = (t2 <= t1) ? 1 : 0
					if got_dx[idx] != expected { bwd_ok = false }
				}
			}
			check(bwd_ok, fmt.tprintf("%v: Bf16 causal_mask backward dx matches", label), any_failed)
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
