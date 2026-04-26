// Smoke test for the Vulkan compute backend. Verifies:
//   1. Device init.
//   2. Upload/download round-trip (staging copy path).
//   3. A real compute kernel (elementwise add) — buffers + pipeline +
//      descriptor set + dispatch all wired up correctly.
//
// Build: odin build examples/gpu_hello -out:examples/gpu_hello/gpu_hello.exe
package gpu_hello

import "core:fmt"
import "core:math"
import "core:math/rand"
import ml "../.."
import "../../gpu"

main :: proc() {
	gpu.init()
	defer gpu.destroy()

	fmt.printfln("device: %v", gpu.device_name())

	// 1+2: round-trip.
	N :: 1024
	src: [N]f32
	for i in 0 ..< N {
		src[i] = f32(i) * 0.5 - 17.0
	}

	t := gpu.alloc(N)
	defer gpu.destroy_tensor(t)
	gpu.upload(src[:], t)

	dst: [N]f32
	gpu.download(t, dst[:])

	for i in 0 ..< N {
		if dst[i] != src[i] {
			fmt.printfln("FAIL round-trip at %v: got %v want %v", i, dst[i], src[i])
			return
		}
	}
	fmt.println("upload/download round-trip ok")

	// 3: gpu.add — verify against CPU reference.
	a_cpu, b_cpu, out_cpu, out_gpu: [N]f32
	for i in 0 ..< N {
		a_cpu[i] = f32(i) * 0.1
		b_cpu[i] = f32(N - i) * 0.2
		out_cpu[i] = a_cpu[i] + b_cpu[i]
	}

	a_gpu := gpu.alloc(N); defer gpu.destroy_tensor(a_gpu)
	b_gpu := gpu.alloc(N); defer gpu.destroy_tensor(b_gpu)
	o_gpu := gpu.alloc(N); defer gpu.destroy_tensor(o_gpu)

	gpu.upload(a_cpu[:], a_gpu)
	gpu.upload(b_cpu[:], b_gpu)
	gpu.add(a_gpu, b_gpu, o_gpu)
	gpu.download(o_gpu, out_gpu[:])

	for i in 0 ..< N {
		if out_gpu[i] != out_cpu[i] {
			fmt.printfln("FAIL add at %v: got %v want %v", i, out_gpu[i], out_cpu[i])
			return
		}
	}
	fmt.println("gpu.add matches CPU reference")

	// 4: gpu.linear vs ml.linear.
	ml.init(64 * 1024 * 1024)
	ml.set_thread_count(1)
	test_linear(8, 32, 16)
	test_linear(64, 128, 512)
	test_linear(1, 512, 2048)

	test_gelu(64 * 512)
	test_layernorm(64, 128)
	test_layernorm(8, 1024)
	test_softmax(64, 256)
	test_softmax(4, 1024)

	test_select(256, 128, 64)
	test_deinterleave(64, 384, 0, 3)
	test_deinterleave(64, 384, 1, 3)
	test_interleave3(64, 128)
	test_rope(64, 4, 32)

	test_slice_trailing(64, 384, 0, 128)
	test_slice_trailing(64, 384, 128, 256)
	test_slice_trailing(64, 384, 256, 384)
	test_concat3(64, 128, 128, 128)

	test_attention(8,  2, 4)
	test_attention(64, 4, 32)
	test_attention(128, 8, 16)
}

test_linear :: proc(count, input_size, output_size: int) {
	rand.reset(0xC0FFEE)
	ml.clear()

	x_t := ml.zeros(count, input_size)
	w_p := ml.make(output_size, input_size)
	defer ml.destroy(w_p)
	ml.fill_normal(w_p, 0, 0.02)
	for i in 0 ..< count * input_size {
		x_t.data[i] = rand.float32_range(-1, 1)
	}

	y_t := ml.linear(x_t, w_p)

	x_g := gpu.alloc(count, input_size);          defer gpu.destroy_tensor(x_g)
	w_g := gpu.alloc(output_size, input_size);    defer gpu.destroy_tensor(w_g)
	y_g := gpu.alloc(count, output_size);         defer gpu.destroy_tensor(y_g)
	gpu.upload(x_t.data, x_g)
	gpu.upload(w_p.data, w_g)
	gpu.linear(x_g, w_g, y_g, count, input_size, output_size)

	y_gpu := make([]f32, count * output_size)
	defer delete(y_gpu)
	gpu.download(y_g, y_gpu)

	max_abs: f32
	max_rel: f32
	for i in 0 ..< count * output_size {
		d := math.abs(y_gpu[i] - y_t.data[i])
		if d > max_abs { max_abs = d }
		denom := math.max(math.abs(y_t.data[i]), 1e-6)
		if d / denom > max_rel { max_rel = d / denom }
	}
	fmt.printfln("linear (count=%v in=%v out=%v): max_abs=%.3e max_rel=%.3e",
		count, input_size, output_size, max_abs, max_rel)
}

test_gelu :: proc(n: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	x := ml.zeros(n)
	for i in 0 ..< n { x.data[i] = rand.float32_range(-3, 3) }
	y := ml.gelu(x)

	xg := gpu.alloc(n); defer gpu.destroy_tensor(xg)
	yg := gpu.alloc(n); defer gpu.destroy_tensor(yg)
	gpu.upload(x.data, xg)
	gpu.gelu(xg, yg)
	out := make([]f32, n); defer delete(out)
	gpu.download(yg, out)

	max_abs: f32
	for i in 0 ..< n {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("gelu (n=%v): max_abs=%.3e", n, max_abs)
}

test_layernorm :: proc(count, size: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	x := ml.zeros(count, size)
	w := ml.make(size); defer ml.destroy(w)
	for i in 0 ..< count*size { x.data[i] = rand.float32_range(-2, 2) }
	for i in 0 ..< size       { w.data[i] = rand.float32_range(0.5, 1.5) }
	y := ml.layernorm(x, w)

	xg := gpu.alloc(count, size); defer gpu.destroy_tensor(xg)
	wg := gpu.alloc(size);        defer gpu.destroy_tensor(wg)
	yg := gpu.alloc(count, size); defer gpu.destroy_tensor(yg)
	gpu.upload(x.data, xg)
	gpu.upload(w.data, wg)
	gpu.layernorm(xg, wg, yg, count, size)
	out := make([]f32, count*size); defer delete(out)
	gpu.download(yg, out)

	max_abs: f32
	for i in 0 ..< count*size {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("layernorm (count=%v size=%v): max_abs=%.3e", count, size, max_abs)
}

test_softmax :: proc(count, size: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	x := ml.zeros(count, size)
	for i in 0 ..< count*size { x.data[i] = rand.float32_range(-5, 5) }
	y := ml.softmax(x)

	xg := gpu.alloc(count, size); defer gpu.destroy_tensor(xg)
	yg := gpu.alloc(count, size); defer gpu.destroy_tensor(yg)
	gpu.upload(x.data, xg)
	gpu.softmax(xg, yg, count, size)
	out := make([]f32, count*size); defer delete(out)
	gpu.download(yg, out)

	max_abs: f32
	for i in 0 ..< count*size {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("softmax (count=%v size=%v): max_abs=%.3e", count, size, max_abs)
}

test_select :: proc(vocab, size, n_idx: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	tab := ml.zeros(vocab, size)
	for i in 0 ..< vocab*size { tab.data[i] = rand.float32_range(-1, 1) }
	idx := make([]int, n_idx); defer delete(idx)
	for i in 0 ..< n_idx { idx[i] = int(rand.uint32()) % vocab }

	y := ml.select(tab, idx)

	tg := gpu.alloc(vocab, size); defer gpu.destroy_tensor(tg)
	og := gpu.alloc(n_idx, size); defer gpu.destroy_tensor(og)
	gpu.upload(tab.data, tg)
	gpu.select(tg, idx, og, size)
	out := make([]f32, n_idx*size); defer delete(out)
	gpu.download(og, out)

	max_abs: f32
	for i in 0 ..< n_idx*size {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("select (vocab=%v size=%v n=%v): max_abs=%.3e", vocab, size, n_idx, max_abs)
}

test_deinterleave :: proc(rows, trailing, col, ncol: int) {
	rand.reset(u64(0xC0FFEE + col))
	ml.clear()
	x := ml.zeros(rows, trailing)
	for i in 0 ..< rows*trailing { x.data[i] = rand.float32_range(-1, 1) }
	y := ml.deinterleave(x, col, ncol)

	xg := gpu.alloc(rows, trailing);          defer gpu.destroy_tensor(xg)
	yg := gpu.alloc(rows, trailing / ncol);   defer gpu.destroy_tensor(yg)
	gpu.upload(x.data, xg)
	gpu.deinterleave(xg, yg, col, ncol)
	out := make([]f32, rows * trailing / ncol); defer delete(out)
	gpu.download(yg, out)

	max_abs: f32
	for i in 0 ..< rows * trailing / ncol {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("deinterleave (rows=%v trailing=%v col=%v/%v): max_abs=%.3e",
		rows, trailing, col, ncol, max_abs)
}

test_interleave3 :: proc(rows, trailing: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	a := ml.zeros(rows, trailing)
	b := ml.zeros(rows, trailing)
	c := ml.zeros(rows, trailing)
	for i in 0 ..< rows*trailing {
		a.data[i] = rand.float32_range(-1, 1)
		b.data[i] = rand.float32_range(-1, 1)
		c.data[i] = rand.float32_range(-1, 1)
	}
	y := ml.interleave(a, b, c)

	ag := gpu.alloc(rows, trailing);     defer gpu.destroy_tensor(ag)
	bg := gpu.alloc(rows, trailing);     defer gpu.destroy_tensor(bg)
	cg := gpu.alloc(rows, trailing);     defer gpu.destroy_tensor(cg)
	og := gpu.alloc(rows, trailing * 3); defer gpu.destroy_tensor(og)
	gpu.upload(a.data, ag)
	gpu.upload(b.data, bg)
	gpu.upload(c.data, cg)
	gpu.interleave3(ag, bg, cg, og)
	out := make([]f32, rows * trailing * 3); defer delete(out)
	gpu.download(og, out)

	max_abs: f32
	for i in 0 ..< rows * trailing * 3 {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("interleave3 (rows=%v trailing=%v): max_abs=%.3e", rows, trailing, max_abs)
}

test_rope :: proc(tokens, heads, head_size: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	x := ml.zeros(tokens, heads * head_size)
	for i in 0 ..< tokens * heads * head_size { x.data[i] = rand.float32_range(-1, 1) }
	y := ml.rope(x, heads)

	xg := gpu.alloc(tokens, heads * head_size); defer gpu.destroy_tensor(xg)
	yg := gpu.alloc(tokens, heads * head_size); defer gpu.destroy_tensor(yg)
	gpu.upload(x.data, xg)
	gpu.rope(xg, yg, tokens, heads, head_size)
	out := make([]f32, tokens * heads * head_size); defer delete(out)
	gpu.download(yg, out)

	max_abs: f32
	for i in 0 ..< tokens * heads * head_size {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("rope (tokens=%v heads=%v head_size=%v): max_abs=%.3e",
		tokens, heads, head_size, max_abs)
}

test_slice_trailing :: proc(leading, trailing, start, end: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	x := ml.zeros(leading, trailing)
	for i in 0 ..< leading*trailing { x.data[i] = rand.float32_range(-1, 1) }
	y := ml.slice_trailing(x, start, end)

	new_trailing := end - start
	xg := gpu.alloc(leading, trailing);     defer gpu.destroy_tensor(xg)
	yg := gpu.alloc(leading, new_trailing); defer gpu.destroy_tensor(yg)
	gpu.upload(x.data, xg)
	gpu.slice_trailing(xg, yg, leading, trailing, start, end)
	out := make([]f32, leading * new_trailing); defer delete(out)
	gpu.download(yg, out)

	max_abs: f32
	for i in 0 ..< leading*new_trailing {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("slice_trailing (leading=%v trailing=%v %v:%v): max_abs=%.3e",
		leading, trailing, start, end, max_abs)
}

test_concat3 :: proc(leading, t_a, t_b, t_c: int) {
	rand.reset(0xC0FFEE)
	ml.clear()
	a := ml.zeros(leading, t_a)
	b := ml.zeros(leading, t_b)
	c := ml.zeros(leading, t_c)
	for i in 0 ..< leading*t_a { a.data[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< leading*t_b { b.data[i] = rand.float32_range(-1, 1) }
	for i in 0 ..< leading*t_c { c.data[i] = rand.float32_range(-1, 1) }
	y := ml.concat(a, b, c)

	ag := gpu.alloc(leading, t_a); defer gpu.destroy_tensor(ag)
	bg := gpu.alloc(leading, t_b); defer gpu.destroy_tensor(bg)
	cg := gpu.alloc(leading, t_c); defer gpu.destroy_tensor(cg)
	og := gpu.alloc(leading, t_a + t_b + t_c); defer gpu.destroy_tensor(og)
	gpu.upload(a.data, ag); gpu.upload(b.data, bg); gpu.upload(c.data, cg)
	gpu.concat3(ag, bg, cg, og, leading, t_a, t_b, t_c)
	out := make([]f32, leading * (t_a+t_b+t_c)); defer delete(out)
	gpu.download(og, out)

	max_abs: f32
	for i in 0 ..< leading*(t_a+t_b+t_c) {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("concat3 (leading=%v t=%v+%v+%v): max_abs=%.3e",
		leading, t_a, t_b, t_c, max_abs)
}

test_attention :: proc(tokens, heads, head_size: int) {
	embed := heads * head_size
	rand.reset(0xC0FFEE)
	ml.clear()
	qkv := ml.zeros(tokens, 3 * embed)
	for i in 0 ..< tokens * 3 * embed { qkv.data[i] = rand.float32_range(-1, 1) }
	y := ml.attention(qkv, heads)

	xg := gpu.alloc(tokens, 3 * embed); defer gpu.destroy_tensor(xg)
	yg := gpu.alloc(tokens, embed);     defer gpu.destroy_tensor(yg)
	gpu.upload(qkv.data, xg)
	gpu.attention(xg, yg, tokens, heads, head_size)
	out := make([]f32, tokens * embed); defer delete(out)
	gpu.download(yg, out)

	max_abs: f32
	for i in 0 ..< tokens * embed {
		d := math.abs(out[i] - y.data[i])
		if d > max_abs { max_abs = d }
	}
	fmt.printfln("attention (tokens=%v heads=%v head_size=%v): max_abs=%.3e",
		tokens, heads, head_size, max_abs)
}
