// Per-kernel correctness check for backward shaders. For each backward op
// we exercise, compute the reference result in pure Odin and compare to a
// GPU run on the same random inputs.
//
// Build: odin build examples/gpu_back_check -o:speed -no-bounds-check -microarch:native -out:examples/gpu_back_check/gpu_back_check.exe
package gpu_back_check

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import ml  "../.."
import gpu "../../gpu"

PASS_TOL :: 1e-4

main :: proc() {
	ctx := ml.context_create(64 * 1024 * 1024)
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)
	gpu.init()
	defer gpu.destroy()

	gctx := gpu.context_create()
	defer gpu.context_destroy(gctx)
	gpu.context_scope(gctx)

	any_failed := false
	any_failed |= !check_zero()
	any_failed |= !check_add_back()
	any_failed |= !check_gelu_back()
	any_failed |= !check_slice_trailing_back()
	any_failed |= !check_concat3_back()
	any_failed |= !check_rope_back()
	any_failed |= !check_linear_back()
	any_failed |= !check_layernorm_back()
	any_failed |= !check_select_back()
	any_failed |= !check_attention_back()
	any_failed |= !check_cross_entropy_grad()

	if any_failed {
		fmt.println("FAIL")
		os.exit(1)
	}
	fmt.println("OK")
}

check_zero :: proc() -> bool {
	N :: 1000
	t := gpu.alloc(N); defer gpu.destroy_tensor(t)

	src := make([]f32, N); defer delete(src)
	for i in 0 ..< N { src[i] = rand.float32_range(-1, 1) }
	gpu.upload(src, t)

	gpu.begin_batch()
	gpu.zero(t)
	gpu.end_batch()

	got := make([]f32, N); defer delete(got)
	gpu.download(t, got)
	for v, i in got {
		if v != 0 {
			fmt.printfln("zero: FAIL at %v: got %v", i, v)
			return false
		}
	}
	fmt.println("zero: OK")
	return true
}

check_add_back :: proc() -> bool {
	N :: 1024
	dy   := make([]f32, N); defer delete(dy)
	da_a := make([]f32, N); defer delete(da_a)
	da_b := make([]f32, N); defer delete(da_b)
	for i in 0 ..< N {
		dy[i]   = rand.float32_range(-1, 1)
		da_a[i] = rand.float32_range(-1, 1) // pre-existing accumulation
		da_b[i] = rand.float32_range(-1, 1)
	}

	g_dy   := gpu.alloc(N); defer gpu.destroy_tensor(g_dy)
	g_da_a := gpu.alloc(N); defer gpu.destroy_tensor(g_da_a)
	g_da_b := gpu.alloc(N); defer gpu.destroy_tensor(g_da_b)
	gpu.upload(dy, g_dy)
	gpu.upload(da_a, g_da_a)
	gpu.upload(da_b, g_da_b)

	gpu.begin_batch()
	gpu.add_back(g_da_a, g_da_b, g_dy)
	gpu.end_batch()

	got_a := make([]f32, N); defer delete(got_a)
	got_b := make([]f32, N); defer delete(got_b)
	gpu.download(g_da_a, got_a)
	gpu.download(g_da_b, got_b)

	for i in 0 ..< N {
		exp_a := da_a[i] + dy[i]
		exp_b := da_b[i] + dy[i]
		if abs(got_a[i] - exp_a) > PASS_TOL || abs(got_b[i] - exp_b) > PASS_TOL {
			fmt.printfln("add_back: FAIL at %v: got_a=%v exp_a=%v got_b=%v exp_b=%v",
				i, got_a[i], exp_a, got_b[i], exp_b)
			return false
		}
	}
	fmt.println("add_back: OK")
	return true
}

check_gelu_back :: proc() -> bool {
	N :: 4096
	GELU_SCALE :: f32(0.7978845608028654)

	x  := make([]f32, N); defer delete(x)
	dy := make([]f32, N); defer delete(dy)
	dx := make([]f32, N); defer delete(dx)
	for i in 0 ..< N {
		x[i]  = rand.float32_range(-3, 3)
		dy[i] = rand.float32_range(-1, 1)
		dx[i] = rand.float32_range(-1, 1)
	}

	g_x  := gpu.alloc(N); defer gpu.destroy_tensor(g_x)
	g_dx := gpu.alloc(N); defer gpu.destroy_tensor(g_dx)
	g_dy := gpu.alloc(N); defer gpu.destroy_tensor(g_dy)
	gpu.upload(x, g_x)
	gpu.upload(dx, g_dx)
	gpu.upload(dy, g_dy)

	gpu.begin_batch()
	gpu.gelu_back(g_x, g_dx, g_dy)
	gpu.end_batch()

	got := make([]f32, N); defer delete(got)
	gpu.download(g_dx, got)

	max_err: f32 = 0
	for i in 0 ..< N {
		v       := x[i]
		cube    := f32(0.044715) * v * v * v
		t_arg   := GELU_SCALE * (v + cube)
		t_out   := math.tanh(t_arg)
		c_out   := math.cosh(t_arg)
		sech2   := 1.0 / (c_out * c_out)
		local_g := 0.5 * (1.0 + t_out) + v * 0.5 * sech2 * GELU_SCALE * (1.0 + 3.0 * f32(0.044715) * v * v)
		exp     := dx[i] + local_g * dy[i]
		err     := abs(got[i] - exp)
		if err > max_err { max_err = err }
	}
	if max_err > PASS_TOL {
		fmt.printfln("gelu_back: FAIL  max_err=%v", max_err)
		return false
	}
	fmt.printfln("gelu_back: OK  max_err=%.3e", max_err)
	return true
}

check_slice_trailing_back :: proc() -> bool {
	LEADING  :: 32
	TRAILING :: 96
	START    :: 32
	END      :: 64
	NEW      :: END - START

	dx := make([]f32, LEADING * TRAILING); defer delete(dx)
	dy := make([]f32, LEADING * NEW);      defer delete(dy)
	for i in 0 ..< len(dx) { dx[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dy) { dy[i] = rand.float32_range(-1, 1) }

	g_dx := gpu.alloc(LEADING, TRAILING); defer gpu.destroy_tensor(g_dx)
	g_dy := gpu.alloc(LEADING, NEW);      defer gpu.destroy_tensor(g_dy)
	gpu.upload(dx, g_dx)
	gpu.upload(dy, g_dy)

	gpu.begin_batch()
	gpu.slice_trailing_back(g_dx, g_dy, LEADING, TRAILING, START, END)
	gpu.end_batch()

	got := make([]f32, LEADING * TRAILING); defer delete(got)
	gpu.download(g_dx, got)

	for r in 0 ..< LEADING {
		for c in 0 ..< TRAILING {
			exp := dx[r * TRAILING + c]
			if c >= START && c < END {
				exp += dy[r * NEW + (c - START)]
			}
			if abs(got[r * TRAILING + c] - exp) > PASS_TOL {
				fmt.printfln("slice_trailing_back: FAIL at (%v,%v) got=%v exp=%v", r, c, got[r*TRAILING+c], exp)
				return false
			}
		}
	}
	fmt.println("slice_trailing_back: OK")
	return true
}

check_concat3_back :: proc() -> bool {
	LEADING :: 16
	T_A     :: 8
	T_B     :: 12
	T_C     :: 4
	TOTAL   :: T_A + T_B + T_C

	da := make([]f32, LEADING * T_A);   defer delete(da)
	db := make([]f32, LEADING * T_B);   defer delete(db)
	dc := make([]f32, LEADING * T_C);   defer delete(dc)
	dy := make([]f32, LEADING * TOTAL); defer delete(dy)
	for i in 0 ..< len(da) { da[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(db) { db[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dc) { dc[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dy) { dy[i] = rand.float32_range(-1, 1) }

	g_da := gpu.alloc(LEADING, T_A);   defer gpu.destroy_tensor(g_da)
	g_db := gpu.alloc(LEADING, T_B);   defer gpu.destroy_tensor(g_db)
	g_dc := gpu.alloc(LEADING, T_C);   defer gpu.destroy_tensor(g_dc)
	g_dy := gpu.alloc(LEADING, TOTAL); defer gpu.destroy_tensor(g_dy)
	gpu.upload(da, g_da); gpu.upload(db, g_db); gpu.upload(dc, g_dc); gpu.upload(dy, g_dy)

	gpu.begin_batch()
	gpu.concat3_back(g_da, g_db, g_dc, g_dy, LEADING, T_A, T_B, T_C)
	gpu.end_batch()

	got_a := make([]f32, LEADING * T_A); defer delete(got_a)
	got_b := make([]f32, LEADING * T_B); defer delete(got_b)
	got_c := make([]f32, LEADING * T_C); defer delete(got_c)
	gpu.download(g_da, got_a); gpu.download(g_db, got_b); gpu.download(g_dc, got_c)

	for r in 0 ..< LEADING {
		for i in 0 ..< T_A {
			exp := da[r*T_A + i] + dy[r*TOTAL + i]
			if abs(got_a[r*T_A + i] - exp) > PASS_TOL { fmt.println("concat3_back: FAIL a"); return false }
		}
		for i in 0 ..< T_B {
			exp := db[r*T_B + i] + dy[r*TOTAL + T_A + i]
			if abs(got_b[r*T_B + i] - exp) > PASS_TOL { fmt.println("concat3_back: FAIL b"); return false }
		}
		for i in 0 ..< T_C {
			exp := dc[r*T_C + i] + dy[r*TOTAL + T_A + T_B + i]
			if abs(got_c[r*T_C + i] - exp) > PASS_TOL { fmt.println("concat3_back: FAIL c"); return false }
		}
	}
	fmt.println("concat3_back: OK")
	return true
}

check_linear_back :: proc() -> bool {
	COUNT  :: 32
	IN     :: 48
	OUT    :: 64

	x  := make([]f32, COUNT * IN);  defer delete(x)
	w  := make([]f32, OUT * IN);    defer delete(w)
	dy := make([]f32, COUNT * OUT); defer delete(dy)
	dx := make([]f32, COUNT * IN);  defer delete(dx)
	dw := make([]f32, OUT * IN);    defer delete(dw)
	for i in 0 ..< len(x)  { x[i]  = rand.float32_range(-1, 1) }
	for i in 0 ..< len(w)  { w[i]  = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dy) { dy[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dx) { dx[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dw) { dw[i] = rand.float32_range(-1, 1) }

	g_x  := gpu.alloc(COUNT, IN);  defer gpu.destroy_tensor(g_x)
	g_w  := gpu.alloc(OUT, IN);    defer gpu.destroy_tensor(g_w)
	g_dy := gpu.alloc(COUNT, OUT); defer gpu.destroy_tensor(g_dy)
	g_dx := gpu.alloc(COUNT, IN);  defer gpu.destroy_tensor(g_dx)
	g_dw := gpu.alloc(OUT, IN);    defer gpu.destroy_tensor(g_dw)
	gpu.upload(x, g_x); gpu.upload(w, g_w); gpu.upload(dy, g_dy)
	gpu.upload(dx, g_dx); gpu.upload(dw, g_dw)

	gpu.begin_batch()
	gpu.linear_back(g_x, g_w, g_dy, g_dx, g_dw, COUNT, IN, OUT)
	gpu.end_batch()

	got_dx := make([]f32, COUNT * IN); defer delete(got_dx)
	got_dw := make([]f32, OUT * IN);   defer delete(got_dw)
	gpu.download(g_dx, got_dx); gpu.download(g_dw, got_dw)

	exp_dx := make([]f32, COUNT * IN); defer delete(exp_dx)
	exp_dw := make([]f32, OUT * IN);   defer delete(exp_dw)
	copy(exp_dx, dx); copy(exp_dw, dw)
	for c in 0 ..< COUNT {
		for o in 0 ..< OUT {
			d := dy[c*OUT + o]
			for k in 0 ..< IN {
				exp_dx[c*IN + k] += w[o*IN + k] * d
				exp_dw[o*IN + k] += x[c*IN + k] * d
			}
		}
	}

	max_dx, max_dw: f32 = 0, 0
	for i in 0 ..< len(exp_dx) {
		e := abs(got_dx[i] - exp_dx[i])
		if e > max_dx { max_dx = e }
	}
	for i in 0 ..< len(exp_dw) {
		e := abs(got_dw[i] - exp_dw[i])
		if e > max_dw { max_dw = e }
	}
	if max_dx > 1e-3 || max_dw > 1e-3 {
		fmt.printfln("linear_back: FAIL  max_dx=%v max_dw=%v", max_dx, max_dw)
		return false
	}
	fmt.printfln("linear_back: OK  max_dx=%.3e max_dw=%.3e", max_dx, max_dw)
	return true
}

check_layernorm_back :: proc() -> bool {
	COUNT :: 32
	SIZE  :: 128
	EPS   :: f32(1e-5)

	x  := make([]f32, COUNT * SIZE); defer delete(x)
	w  := make([]f32, SIZE);         defer delete(w)
	dy := make([]f32, COUNT * SIZE); defer delete(dy)
	dx := make([]f32, COUNT * SIZE); defer delete(dx)
	dw := make([]f32, SIZE);         defer delete(dw)
	for i in 0 ..< len(x)  { x[i]  = rand.float32_range(-2, 2) }
	for i in 0 ..< len(w)  { w[i]  = rand.float32_range(0.5, 1.5) }
	for i in 0 ..< len(dy) { dy[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dx) { dx[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dw) { dw[i] = rand.float32_range(-1, 1) }

	g_x  := gpu.alloc(COUNT, SIZE); defer gpu.destroy_tensor(g_x)
	g_w  := gpu.alloc(SIZE);        defer gpu.destroy_tensor(g_w)
	g_dy := gpu.alloc(COUNT, SIZE); defer gpu.destroy_tensor(g_dy)
	g_dx := gpu.alloc(COUNT, SIZE); defer gpu.destroy_tensor(g_dx)
	g_dw := gpu.alloc(SIZE);        defer gpu.destroy_tensor(g_dw)
	gpu.upload(x, g_x); gpu.upload(w, g_w); gpu.upload(dy, g_dy)
	gpu.upload(dx, g_dx); gpu.upload(dw, g_dw)

	gpu.begin_batch()
	gpu.layernorm_back(g_x, g_w, g_dy, g_dx, g_dw, COUNT, SIZE)
	gpu.end_batch()

	got_dx := make([]f32, COUNT * SIZE); defer delete(got_dx)
	got_dw := make([]f32, SIZE);         defer delete(got_dw)
	gpu.download(g_dx, got_dx); gpu.download(g_dw, got_dw)

	// CPU reference, mirroring ml.layernorm_backward.
	exp_dx := make([]f32, COUNT * SIZE); defer delete(exp_dx)
	exp_dw := make([]f32, SIZE);         defer delete(exp_dw)
	copy(exp_dx, dx); copy(exp_dw, dw)
	for c in 0 ..< COUNT {
		off := c * SIZE
		m: f32 = 0
		for i in 0 ..< SIZE { m += x[off + i] }
		m /= f32(SIZE)
		v: f32 = 0
		for i in 0 ..< SIZE {
			d := x[off + i] - m
			v += d * d
		}
		v /= f32(SIZE)
		s := 1.0 / math.sqrt(v + EPS)

		dn_mean, dn_norm_mean: f32 = 0, 0
		for i in 0 ..< SIZE {
			norm  := (x[off + i] - m) * s
			dnorm := w[i] * dy[off + i]
			dn_mean      += dnorm
			dn_norm_mean += dnorm * norm
		}
		dn_mean      /= f32(SIZE)
		dn_norm_mean /= f32(SIZE)

		for i in 0 ..< SIZE {
			norm  := (x[off + i] - m) * s
			dnorm := w[i] * dy[off + i]
			exp_dw[i] += norm * dy[off + i]
			g := (dnorm - dn_mean - norm * dn_norm_mean) * s
			exp_dx[off + i] += g
		}
	}

	max_dx, max_dw: f32 = 0, 0
	for i in 0 ..< len(exp_dx) {
		e := abs(got_dx[i] - exp_dx[i])
		if e > max_dx { max_dx = e }
	}
	for i in 0 ..< len(exp_dw) {
		e := abs(got_dw[i] - exp_dw[i])
		if e > max_dw { max_dw = e }
	}
	if max_dx > 1e-3 || max_dw > 1e-3 {
		fmt.printfln("layernorm_back: FAIL  max_dx=%v max_dw=%v", max_dx, max_dw)
		return false
	}
	fmt.printfln("layernorm_back: OK  max_dx=%.3e max_dw=%.3e", max_dx, max_dw)
	return true
}

check_select_back :: proc() -> bool {
	VOCAB :: 64
	N     :: 32
	SIZE  :: 48

	indices := make([]int, N); defer delete(indices)
	for i in 0 ..< N { indices[i] = int(rand.uint32() % VOCAB) }

	dy := make([]f32, N * SIZE);     defer delete(dy)
	dt := make([]f32, VOCAB * SIZE); defer delete(dt)
	for i in 0 ..< len(dy) { dy[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< len(dt) { dt[i] = rand.float32_range(-1, 1) }

	g_dy := gpu.alloc(N, SIZE);     defer gpu.destroy_tensor(g_dy)
	g_dt := gpu.alloc(VOCAB, SIZE); defer gpu.destroy_tensor(g_dt)
	gpu.upload(dy, g_dy); gpu.upload(dt, g_dt)

	gpu.begin_batch()
	gpu.select_back(indices, g_dy, g_dt, VOCAB, SIZE)
	gpu.end_batch()

	got := make([]f32, VOCAB * SIZE); defer delete(got)
	gpu.download(g_dt, got)

	exp := make([]f32, VOCAB * SIZE); defer delete(exp)
	copy(exp, dt)
	for i in 0 ..< N {
		v := indices[i]
		for j in 0 ..< SIZE {
			exp[v*SIZE + j] += dy[i*SIZE + j]
		}
	}

	max_e: f32 = 0
	for i in 0 ..< len(exp) {
		e := abs(got[i] - exp[i])
		if e > max_e { max_e = e }
	}
	if max_e > 1e-4 {
		fmt.printfln("select_back: FAIL  max_err=%v", max_e)
		return false
	}
	fmt.printfln("select_back: OK  max_err=%.3e", max_e)
	return true
}

check_cross_entropy_grad :: proc() -> bool {
	N      :: 32
	VOCAB  :: 64

	logits := make([]f32, N * VOCAB); defer delete(logits)
	for i in 0 ..< len(logits) { logits[i] = rand.float32_range(-3, 3) }
	targets := make([]int, N); defer delete(targets)
	for i in 0 ..< N { targets[i] = int(rand.uint32() % VOCAB) }

	g_logits := gpu.alloc(N, VOCAB); defer gpu.destroy_tensor(g_logits)
	g_dx     := gpu.alloc(N, VOCAB); defer gpu.destroy_tensor(g_dx)
	g_loss   := gpu.alloc(N);        defer gpu.destroy_tensor(g_loss)
	gpu.upload(logits, g_logits)

	gpu.begin_batch()
	gpu.cross_entropy_grad(g_logits, g_dx, g_loss, targets, N, VOCAB)
	gpu.end_batch()

	got_dx   := make([]f32, N * VOCAB); defer delete(got_dx)
	got_loss := make([]f32, N);         defer delete(got_loss)
	gpu.download(g_dx, got_dx)
	gpu.download(g_loss, got_loss)

	max_dx, max_loss: f32 = 0, 0
	for c in 0 ..< N {
		off := c * VOCAB
		mx := f32(-1e30)
		for j in 0 ..< VOCAB { if logits[off + j] > mx { mx = logits[off + j] } }
		sum: f32 = 0
		for j in 0 ..< VOCAB { sum += math.exp(logits[off + j] - mx) }
		exp_loss := mx + math.ln(sum) - logits[off + targets[c]]
		e := abs(got_loss[c] - exp_loss)
		if e > max_loss { max_loss = e }

		for j in 0 ..< VOCAB {
			prob := math.exp(logits[off + j] - mx) / sum
			ind  := f32(0)
			if j == targets[c] { ind = 1 }
			exp_dx := (prob - ind) / f32(N)
			ed := abs(got_dx[off + j] - exp_dx)
			if ed > max_dx { max_dx = ed }
		}
	}
	if max_dx > 1e-5 || max_loss > 1e-4 {
		fmt.printfln("cross_entropy_grad: FAIL  max_dx=%v max_loss=%v", max_dx, max_loss)
		return false
	}
	fmt.printfln("cross_entropy_grad: OK  max_dx=%.3e max_loss=%.3e", max_dx, max_loss)
	return true
}

check_attention_back :: proc() -> bool {
	TOKENS    :: 16
	HEADS     :: 4
	HEAD_SIZE :: 16
	EMBED     :: HEADS * HEAD_SIZE
	N_QKV     :: TOKENS * 3 * EMBED
	N_OUT     :: TOKENS * EMBED

	// CPU reference: build a parameter qkv, run ml.attention, ml.backward.
	// ml.backward seeds the final op's output gradient with all-ones, which
	// gives us a known seed dy to feed the GPU kernel for comparison.
	ml.clear()
	qkv_param, _ := ml.make(TOKENS, 3 * EMBED)
	for i in 0 ..< N_QKV { qkv_param.data[i] = rand.float32_range(-1, 1) }

	y := ml.attention(qkv_param, HEADS)
	ml.backward()
	cpu_grad := qkv_param.gradient

	// GPU run with dy = all ones and d_qkv accumulator starting at 0.
	dy_ones := make([]f32, N_OUT); defer delete(dy_ones)
	for i in 0 ..< N_OUT { dy_ones[i] = 1 }

	g_qkv   := gpu.alloc(TOKENS, 3 * EMBED); defer gpu.destroy_tensor(g_qkv)
	g_dy    := gpu.alloc(TOKENS, EMBED);     defer gpu.destroy_tensor(g_dy)
	g_d_qkv := gpu.alloc(TOKENS, 3 * EMBED); defer gpu.destroy_tensor(g_d_qkv)

	gpu.upload(qkv_param.data, g_qkv)
	gpu.upload(dy_ones, g_dy)

	gpu.begin_batch()
	gpu.zero(g_d_qkv)
	gpu.attention_back(g_qkv, g_dy, g_d_qkv, TOKENS, HEADS, HEAD_SIZE)
	gpu.end_batch()

	got := make([]f32, N_QKV); defer delete(got)
	gpu.download(g_d_qkv, got)

	max_e: f32 = 0
	for i in 0 ..< N_QKV {
		e := abs(got[i] - cpu_grad[i])
		if e > max_e { max_e = e }
	}
	if max_e > 1e-3 {
		fmt.printfln("attention_back: FAIL  max_err=%v", max_e)
		return false
	}
	fmt.printfln("attention_back: OK  max_err=%.3e", max_e)
	return true
}

check_rope_back :: proc() -> bool {
	TOKENS    :: 16
	HEADS     :: 4
	HEAD_SIZE :: 32
	BASE      :: f32(10000)
	N         :: TOKENS * HEADS * HEAD_SIZE

	dx := make([]f32, N); defer delete(dx)
	dy := make([]f32, N); defer delete(dy)
	for i in 0 ..< N {
		dx[i] = rand.float32_range(-1, 1)
		dy[i] = rand.float32_range(-1, 1)
	}

	g_dx := gpu.alloc(TOKENS, HEADS * HEAD_SIZE); defer gpu.destroy_tensor(g_dx)
	g_dy := gpu.alloc(TOKENS, HEADS * HEAD_SIZE); defer gpu.destroy_tensor(g_dy)
	gpu.upload(dx, g_dx)
	gpu.upload(dy, g_dy)

	gpu.begin_batch()
	gpu.rope_back(g_dx, g_dy, TOKENS, HEADS, HEAD_SIZE, BASE)
	gpu.end_batch()

	got := make([]f32, N); defer delete(got)
	gpu.download(g_dx, got)

	half := HEAD_SIZE / 2
	max_err: f32 = 0
	for pos in 0 ..< TOKENS {
		for h in 0 ..< HEADS {
			head_off := pos * HEADS * HEAD_SIZE + h * HEAD_SIZE
			for i in 0 ..< half {
				exponent := f32(i * 2) / f32(HEAD_SIZE)
				theta    := f32(pos) / math.pow(BASE, exponent)
				c_v      := math.cos(theta)
				s_v      := math.sin(theta)
				lo := head_off + i*2
				hi := lo + 1
				gx := dy[lo]
				gy := dy[hi]
				exp_lo := dx[lo] +  gx*c_v + gy*s_v
				exp_hi := dx[hi] + -gx*s_v + gy*c_v
				err := max(abs(got[lo] - exp_lo), abs(got[hi] - exp_hi))
				if err > max_err { max_err = err }
			}
		}
	}
	if max_err > PASS_TOL {
		fmt.printfln("rope_back: FAIL  max_err=%v", max_err)
		return false
	}
	fmt.printfln("rope_back: OK  max_err=%.3e", max_err)
	return true
}
