// Public ops over GpuTensor. Each op lazy-creates its compute pipeline on
// first call and caches it as a file-local pointer. SPIR-V is embedded at
// compile time via #load so the binary doesn't depend on the shader files
// at runtime.
package gpu

import "core:fmt"
import "core:math"
import vk "vendor:vulkan"

// add: out = a + b, elementwise. Shapes must match.
add :: proc(a, b, out: GpuTensor, loc := #caller_location) {
	fmt.assertf(a.count == b.count && a.count == out.count,
		"add: shape mismatch a=%v b=%v out=%v", a.count, b.count, out.count, loc=loc)

	if _add_pipeline == nil {
		_add_pipeline = _make_pipeline(ADD_SPIRV, 3, size_of(Add_Params))
	}

	params := Add_Params{ n = u32(a.count) }
	_dispatch(_add_pipeline, _buffers(a, b, out), &params, _div_up(a.count, ADD_LOCAL_SIZE), loc=loc)
}

// linear: y = x · W^T (matches CPU `ml.linear`).
//   x: [count, input_size]
//   w: [output_size, input_size]
//   y: [count, output_size]
linear :: proc(x, w, y: GpuTensor, count, input_size, output_size: int, loc := #caller_location) {
	fmt.assertf(x.count == count * input_size,
		"linear: x size %v != count*input_size = %v*%v", x.count, count, input_size, loc=loc)
	fmt.assertf(w.count == output_size * input_size,
		"linear: w size %v != output_size*input_size = %v*%v", w.count, output_size, input_size, loc=loc)
	fmt.assertf(y.count == count * output_size,
		"linear: y size %v != count*output_size = %v*%v", y.count, count, output_size, loc=loc)

	if _linear_pipeline == nil {
		_linear_pipeline = _make_pipeline(LINEAR_SPIRV, 3, size_of(Linear_Params))
	}

	params := Linear_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}
	_dispatch(
		_linear_pipeline,
		_buffers(x, w, y),
		&params,
		_div_up(count,       LINEAR_LOCAL_X),
		_div_up(output_size, LINEAR_LOCAL_Y),
		1,
		loc = loc,
	)
}

// gelu: y = gelu(x), elementwise (tanh approx).
gelu :: proc(x, y: GpuTensor, loc := #caller_location) {
	fmt.assertf(x.count == y.count, "gelu: shape mismatch x=%v y=%v", x.count, y.count, loc=loc)

	if _gelu_pipeline == nil {
		_gelu_pipeline = _make_pipeline(GELU_SPIRV, 2, size_of(Gelu_Params))
	}
	params := Gelu_Params{ n = u32(x.count) }
	_dispatch(_gelu_pipeline, _buffers(x, y), &params, _div_up(x.count, GELU_LOCAL_SIZE), loc=loc)
}

// layernorm: y[i,j] = ((x[i,j] - mean(x[i])) / std(x[i])) * w[j], per row.
//   x: [count, size]
//   w: [size]
//   y: [count, size]
layernorm :: proc(x, w, y: GpuTensor, count, size: int, loc := #caller_location) {
	fmt.assertf(x.count == count * size, "layernorm: x size %v != %v*%v", x.count, count, size, loc=loc)
	fmt.assertf(w.count == size,          "layernorm: w size %v != %v",   w.count, size, loc=loc)
	fmt.assertf(y.count == count * size, "layernorm: y size %v != %v*%v", y.count, count, size, loc=loc)

	if _layernorm_pipeline == nil {
		_layernorm_pipeline = _make_pipeline(LAYERNORM_SPIRV, 3, size_of(Layernorm_Params))
	}
	params := Layernorm_Params{ count = u32(count), size = u32(size) }
	_dispatch(_layernorm_pipeline, _buffers(x, w, y), &params, u32(count), loc=loc)
}

// softmax: per-row softmax along the trailing dim.
//   x: [count, size]
//   y: [count, size]
softmax :: proc(x, y: GpuTensor, count, size: int, loc := #caller_location) {
	fmt.assertf(x.count == count * size, "softmax: x size %v != %v*%v", x.count, count, size, loc=loc)
	fmt.assertf(y.count == count * size, "softmax: y size %v != %v*%v", y.count, count, size, loc=loc)

	if _softmax_pipeline == nil {
		_softmax_pipeline = _make_pipeline(SOFTMAX_SPIRV, 2, size_of(Softmax_Params))
	}
	params := Softmax_Params{ count = u32(count), size = u32(size) }
	_dispatch(_softmax_pipeline, _buffers(x, y), &params, u32(count), loc=loc)
}

// select: out[i, j] = table[indices[i], j]
//   table:   [vocab, size]
//   indices: []int  (uploaded as u32 to a transient host-visible buffer)
//   out:     [len(indices), size]
select :: proc(table: GpuTensor, indices: []int, out: GpuTensor, size: int, loc := #caller_location) {
	n := len(indices)
	fmt.assertf(out.count == n * size, "select: out size %v != %v*%v", out.count, n, size, loc=loc)

	// Indices buffer: HOST_VISIBLE so we can write directly. They're tiny
	// (token count, ~64) so PCIe-mapped reads from the shader are fine vs
	// running another staging copy.
	idx_size := vk.DeviceSize(n * size_of(u32))
	idx_buf, idx_mem := _create_buffer(idx_size, {.STORAGE_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT}, loc)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, idx_mem, 0, idx_size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(select indices) failed: %v", res, loc=loc)
	arr := ([^]u32)(mapped)
	for v, i in indices { arr[i] = u32(v) }
	vk.UnmapMemory(_gpu.device, idx_mem)

	if _select_pipeline == nil {
		_select_pipeline = _make_pipeline(SELECT_SPIRV, 3, size_of(Select_Params))
	}
	params := Select_Params{ n_indices = u32(n), size = u32(size) }
	bufs := []vk.Buffer{ table.buffer, idx_buf, out.buffer }
	_dispatch(_select_pipeline, bufs, &params, _div_up(size, 256), u32(n), 1, loc=loc)

	// In a batch, GPU hasn't run yet — defer destruction to end_batch.
	// Outside a batch, _dispatch already waited, so destroy now.
	_queue_destroy_buffer(idx_buf, idx_mem)
}

// rope: rotary position embedding, matches CPU `ml.rope`.
//   input/out: [token_count, head_count * head_size]
rope :: proc(input, out: GpuTensor, token_count, head_count, head_size: int, base: f32 = 10000, loc := #caller_location) {
	fmt.assertf(head_size % 2 == 0, "rope: head_size %v must be even", head_size, loc=loc)
	expected := token_count * head_count * head_size
	fmt.assertf(input.count == expected, "rope: input %v != expected %v", input.count, expected, loc=loc)
	fmt.assertf(out.count   == expected, "rope: out %v != expected %v",   out.count,   expected, loc=loc)

	if _rope_pipeline == nil {
		_rope_pipeline = _make_pipeline(ROPE_SPIRV, 2, size_of(Rope_Params))
	}
	params := Rope_Params{
		token_count = u32(token_count),
		head_count  = u32(head_count),
		head_size   = u32(head_size),
		base        = base,
	}
	total_pairs := token_count * head_count * (head_size / 2)
	_dispatch(_rope_pipeline, _buffers(input, out), &params, _div_up(total_pairs, 256), loc=loc)
}

// slice_trailing: out[r, i] = input[r, start + i] for i in [0, end - start).
// Input/out shapes: [leading, trailing] -> [leading, end - start].
slice_trailing :: proc(input, out: GpuTensor, leading, trailing, start, end: int, loc := #caller_location) {
	new_trailing := end - start
	fmt.assertf(start >= 0 && end <= trailing && start <= end,
		"slice_trailing: bounds %v:%v out of [0,%v]", start, end, trailing, loc=loc)
	fmt.assertf(input.count == leading * trailing,
		"slice_trailing: input %v != %v*%v", input.count, leading, trailing, loc=loc)
	fmt.assertf(out.count == leading * new_trailing,
		"slice_trailing: out %v != %v*%v", out.count, leading, new_trailing, loc=loc)

	if _slice_trailing_pipeline == nil {
		_slice_trailing_pipeline = _make_pipeline(SLICE_TRAILING_SPIRV, 2, size_of(Slice_Trailing_Params))
	}
	params := Slice_Trailing_Params{
		leading      = u32(leading),
		trailing     = u32(trailing),
		new_trailing = u32(new_trailing),
		start        = u32(start),
	}
	_dispatch(_slice_trailing_pipeline, _buffers(input, out), &params, _div_up(leading * new_trailing, 256), loc=loc)
}

// concat3: trailing-dim concat of three same-leading-shape tensors.
//   a, b, c: [leading, t_a], [leading, t_b], [leading, t_c]
//   out:     [leading, t_a + t_b + t_c]
concat3 :: proc(a, b, c, out: GpuTensor, leading, t_a, t_b, t_c: int, loc := #caller_location) {
	fmt.assertf(a.count == leading * t_a, "concat3: a %v != %v*%v", a.count, leading, t_a, loc=loc)
	fmt.assertf(b.count == leading * t_b, "concat3: b %v != %v*%v", b.count, leading, t_b, loc=loc)
	fmt.assertf(c.count == leading * t_c, "concat3: c %v != %v*%v", c.count, leading, t_c, loc=loc)
	fmt.assertf(out.count == leading * (t_a + t_b + t_c),
		"concat3: out %v != %v*%v", out.count, leading, t_a + t_b + t_c, loc=loc)

	if _concat3_pipeline == nil {
		_concat3_pipeline = _make_pipeline(CONCAT3_SPIRV, 4, size_of(Concat3_Params))
	}
	params := Concat3_Params{
		leading = u32(leading),
		t_a     = u32(t_a),
		t_b     = u32(t_b),
		t_c     = u32(t_c),
	}
	total := leading * (t_a + t_b + t_c)
	_dispatch(_concat3_pipeline, _buffers(a, b, c, out), &params, _div_up(total, 256), loc=loc)
}

// attention: multi-head causal scaled-dot-product attention.
//   qkv: [token_count, 3 * embed]   layout [Q | K | V] per row
//   out: [token_count, embed]
attention :: proc(qkv, out: GpuTensor, token_count, head_count, head_size: int, causal := true, loc := #caller_location) {
	embed       := head_count * head_size
	input_size  := 3 * embed
	expected_in := token_count * input_size
	expected_o  := token_count * embed
	fmt.assertf(qkv.count == expected_in, "attention: qkv %v != expected %v", qkv.count, expected_in, loc=loc)
	fmt.assertf(out.count == expected_o,  "attention: out %v != expected %v", out.count, expected_o, loc=loc)

	if _attention_pipeline == nil {
		_attention_pipeline = _make_pipeline(ATTENTION_SPIRV, 2, size_of(Attention_Params))
	}

	scale := f32(1) / math.sqrt(f32(head_size))
	params := Attention_Params{
		token_count = u32(token_count),
		head_count  = u32(head_count),
		head_size   = u32(head_size),
		scale       = scale,
		causal      = u32(1 if causal else 0),
	}
	_dispatch(_attention_pipeline, _buffers(qkv, out), &params, u32(token_count), u32(head_count), 1, loc=loc)
}

// --- Backward ops ---
//
// Convention matches CPU autograd: every backward kernel **accumulates**
// into its destination gradient buffer (`dx += local_grad * dy`). Callers
// are responsible for zeroing input-gradient buffers (via gpu.zero) before
// the first backward op of a step.

// zero: write 0 into every element of `t`.
zero :: proc(t: GpuTensor, loc := #caller_location) {
	if _zero_pipeline == nil {
		_zero_pipeline = _make_pipeline(ZERO_SPIRV, 1, size_of(Zero_Params))
	}
	params := Zero_Params{ n = u32(t.count) }
	_dispatch(_zero_pipeline, _buffers(t), &params, _div_up(t.count, 256), loc=loc)
}

// add_back: backward of `add` with same-shape inputs.
//   da_a += dy
//   da_b += dy
add_back :: proc(da_a, da_b, dy: GpuTensor, loc := #caller_location) {
	fmt.assertf(da_a.count == da_b.count && da_a.count == dy.count,
		"add_back: shape mismatch da_a=%v da_b=%v dy=%v", da_a.count, da_b.count, dy.count, loc=loc)

	if _add_back_pipeline == nil {
		_add_back_pipeline = _make_pipeline(ADD_BACK_SPIRV, 3, size_of(Add_Back_Params))
	}
	params := Add_Back_Params{ n = u32(dy.count) }
	_dispatch(_add_back_pipeline, _buffers(da_a, da_b, dy), &params, _div_up(dy.count, 256), loc=loc)
}

// gelu_back: dx += gelu_grad(x) * dy.
gelu_back :: proc(x, dx, dy: GpuTensor, loc := #caller_location) {
	fmt.assertf(x.count == dx.count && x.count == dy.count,
		"gelu_back: shape mismatch x=%v dx=%v dy=%v", x.count, dx.count, dy.count, loc=loc)

	if _gelu_back_pipeline == nil {
		_gelu_back_pipeline = _make_pipeline(GELU_BACK_SPIRV, 3, size_of(Gelu_Back_Params))
	}
	params := Gelu_Back_Params{ n = u32(x.count) }
	_dispatch(_gelu_back_pipeline, _buffers(x, dx, dy), &params, _div_up(x.count, 256), loc=loc)
}

// slice_trailing_back: scatter dy into the [start, end) trailing slice of dx.
slice_trailing_back :: proc(dx, dy: GpuTensor, leading, trailing, start, end: int, loc := #caller_location) {
	new_trailing := end - start
	fmt.assertf(start >= 0 && end <= trailing && start <= end,
		"slice_trailing_back: bounds %v:%v out of [0,%v]", start, end, trailing, loc=loc)
	fmt.assertf(dx.count == leading * trailing,
		"slice_trailing_back: dx %v != %v*%v", dx.count, leading, trailing, loc=loc)
	fmt.assertf(dy.count == leading * new_trailing,
		"slice_trailing_back: dy %v != %v*%v", dy.count, leading, new_trailing, loc=loc)

	if _slice_trailing_back_pipeline == nil {
		_slice_trailing_back_pipeline = _make_pipeline(SLICE_TRAILING_BACK_SPIRV, 2, size_of(Slice_Trailing_Back_Params))
	}
	params := Slice_Trailing_Back_Params{
		leading      = u32(leading),
		trailing     = u32(trailing),
		new_trailing = u32(new_trailing),
		start        = u32(start),
	}
	_dispatch(_slice_trailing_back_pipeline, _buffers(dx, dy), &params, _div_up(leading * new_trailing, 256), loc=loc)
}

// concat3_back: split dy into da, db, dc trailing-slab gradients.
concat3_back :: proc(da, db, dc, dy: GpuTensor, leading, t_a, t_b, t_c: int, loc := #caller_location) {
	fmt.assertf(da.count == leading * t_a, "concat3_back: da %v != %v*%v", da.count, leading, t_a, loc=loc)
	fmt.assertf(db.count == leading * t_b, "concat3_back: db %v != %v*%v", db.count, leading, t_b, loc=loc)
	fmt.assertf(dc.count == leading * t_c, "concat3_back: dc %v != %v*%v", dc.count, leading, t_c, loc=loc)
	fmt.assertf(dy.count == leading * (t_a + t_b + t_c),
		"concat3_back: dy %v != %v*%v", dy.count, leading, t_a + t_b + t_c, loc=loc)

	if _concat3_back_pipeline == nil {
		_concat3_back_pipeline = _make_pipeline(CONCAT3_BACK_SPIRV, 4, size_of(Concat3_Back_Params))
	}
	params := Concat3_Back_Params{
		leading = u32(leading),
		t_a     = u32(t_a),
		t_b     = u32(t_b),
		t_c     = u32(t_c),
	}
	total := leading * (t_a + t_b + t_c)
	_dispatch(_concat3_back_pipeline, _buffers(da, db, dc, dy), &params, _div_up(total, 256), loc=loc)
}

// linear_back: backward of `linear`, splitting input-grad and weight-grad
// into two kernels (different reduction axes, neither needs atomics).
//   dx[c, k] += sum_o W[o, k] * dy[c, o]
//   dW[o, k] += sum_c x[c, k] * dy[c, o]
linear_back :: proc(x, w, dy, dx, dw: GpuTensor, count, input_size, output_size: int, loc := #caller_location) {
	fmt.assertf(x.count == count * input_size,    "linear_back: x %v != %v*%v",  x.count, count, input_size, loc=loc)
	fmt.assertf(w.count == output_size * input_size, "linear_back: w %v != %v*%v", w.count, output_size, input_size, loc=loc)
	fmt.assertf(dy.count == count * output_size,  "linear_back: dy %v != %v*%v", dy.count, count, output_size, loc=loc)
	fmt.assertf(dx.count == count * input_size,   "linear_back: dx %v != %v*%v", dx.count, count, input_size, loc=loc)
	fmt.assertf(dw.count == output_size * input_size, "linear_back: dw %v != %v*%v", dw.count, output_size, input_size, loc=loc)

	if _linear_back_input_pipeline == nil {
		_linear_back_input_pipeline = _make_pipeline(LINEAR_BACK_INPUT_SPIRV, 3, size_of(Linear_Back_Params))
	}
	if _linear_back_weight_pipeline == nil {
		_linear_back_weight_pipeline = _make_pipeline(LINEAR_BACK_WEIGHT_SPIRV, 3, size_of(Linear_Back_Params))
	}
	params := Linear_Back_Params{
		count       = u32(count),
		input_size  = u32(input_size),
		output_size = u32(output_size),
	}
	_dispatch(
		_linear_back_input_pipeline,
		_buffers(dy, w, dx),
		&params,
		_div_up(count,      16),
		_div_up(input_size, 16),
		1, loc = loc,
	)
	_dispatch(
		_linear_back_weight_pipeline,
		_buffers(x, dy, dw),
		&params,
		_div_up(output_size, 16),
		_div_up(input_size,  16),
		1, loc = loc,
	)
}

// layernorm_back: backward of `layernorm`. Recomputes per-row mean/rstd
// into transient buffers (forward doesn't save them), then runs separate
// input-grad and weight-grad kernels that share the stats.
layernorm_back :: proc(x, weight, dy, dx, dweight: GpuTensor, count, size: int, loc := #caller_location) {
	fmt.assertf(x.count       == count * size, "layernorm_back: x %v != %v*%v", x.count, count, size, loc=loc)
	fmt.assertf(weight.count  == size,         "layernorm_back: weight %v != %v", weight.count, size, loc=loc)
	fmt.assertf(dy.count      == count * size, "layernorm_back: dy %v != %v*%v", dy.count, count, size, loc=loc)
	fmt.assertf(dx.count      == count * size, "layernorm_back: dx %v != %v*%v", dx.count, count, size, loc=loc)
	fmt.assertf(dweight.count == size,         "layernorm_back: dweight %v != %v", dweight.count, size, loc=loc)

	if _layernorm_stats_pipeline == nil {
		_layernorm_stats_pipeline = _make_pipeline(LAYERNORM_STATS_SPIRV, 3, size_of(Layernorm_Stats_Params))
	}
	if _layernorm_back_input_pipeline == nil {
		_layernorm_back_input_pipeline = _make_pipeline(LAYERNORM_BACK_INPUT_SPIRV, 6, size_of(Layernorm_Back_Params))
	}
	if _layernorm_back_weight_pipeline == nil {
		_layernorm_back_weight_pipeline = _make_pipeline(LAYERNORM_BACK_WEIGHT_SPIRV, 5, size_of(Layernorm_Back_Params))
	}

	// Transient mean/rstd buffers — small (one f32 per row), cleaned up by
	// the batch (or immediately if no batch is active).
	stat_size := vk.DeviceSize(count * size_of(f32))
	mean_buf, mean_mem := _create_buffer(stat_size, {.STORAGE_BUFFER}, {.DEVICE_LOCAL}, loc)
	rstd_buf, rstd_mem := _create_buffer(stat_size, {.STORAGE_BUFFER}, {.DEVICE_LOCAL}, loc)

	stats_params := Layernorm_Stats_Params{ count = u32(count), size = u32(size) }
	stats_bufs   := []vk.Buffer{ x.buffer, mean_buf, rstd_buf }
	_dispatch(_layernorm_stats_pipeline, stats_bufs, &stats_params, u32(count), 1, 1, loc=loc)

	back_params := Layernorm_Back_Params{ count = u32(count), size = u32(size) }
	in_bufs  := []vk.Buffer{ x.buffer, weight.buffer, dy.buffer, mean_buf, rstd_buf, dx.buffer }
	_dispatch(_layernorm_back_input_pipeline, in_bufs, &back_params, u32(count), 1, 1, loc=loc)

	w_bufs := []vk.Buffer{ x.buffer, dy.buffer, mean_buf, rstd_buf, dweight.buffer }
	_dispatch(_layernorm_back_weight_pipeline, w_bufs, &back_params, _div_up(size, 256), 1, 1, loc=loc)

	_queue_destroy_buffer(mean_buf, mean_mem)
	_queue_destroy_buffer(rstd_buf, rstd_mem)
}

// select_back: scatter-add the embedding gradient into the table.
//   dtable[indices[i], j] += dy[i, j]
// Thread per (vocab_id, j); avoids f32 atomics by looping over indices.
select_back :: proc(indices: []int, dy, dtable: GpuTensor, vocab, size: int, loc := #caller_location) {
	n := len(indices)
	fmt.assertf(dy.count     == n * size,     "select_back: dy %v != %v*%v",     dy.count, n, size, loc=loc)
	fmt.assertf(dtable.count == vocab * size, "select_back: dtable %v != %v*%v", dtable.count, vocab, size, loc=loc)

	idx_size := vk.DeviceSize(n * size_of(u32))
	idx_buf, idx_mem := _create_buffer(idx_size, {.STORAGE_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT}, loc)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, idx_mem, 0, idx_size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(select_back indices) failed: %v", res, loc=loc)
	arr := ([^]u32)(mapped)
	for v, i in indices { arr[i] = u32(v) }
	vk.UnmapMemory(_gpu.device, idx_mem)

	if _select_back_pipeline == nil {
		_select_back_pipeline = _make_pipeline(SELECT_BACK_SPIRV, 3, size_of(Select_Back_Params))
	}
	params := Select_Back_Params{
		vocab     = u32(vocab),
		n_indices = u32(n),
		size      = u32(size),
	}
	bufs := []vk.Buffer{ idx_buf, dy.buffer, dtable.buffer }
	_dispatch(_select_back_pipeline, bufs, &params, _div_up(vocab, 16), _div_up(size, 16), 1, loc=loc)

	_queue_destroy_buffer(idx_buf, idx_mem)
}

// attention_back: backward of multi-head causal attention.
//
// Six dispatches, all atomic-free:
//   0. Recompute post = softmax(scale * Q @ K^T) into a transient buffer.
//   1. post_grad[t,h,t2] = dot(V[t2,h], dout[t,h]).
//   2. dV[t2,h,d] += sum_{t>=t2} post[t,h,t2] * dout[t,h,d].
//   3. pre_grad[t,h,t3] = post[t,h,t3] * (post_grad[t,h,t3] - <post,post_grad>_row).
//   4. dQ[t,h,d] += scale * sum_{t2<=t} pre_grad[t,h,t2] * K[t2,h,d].
//   5. dK[t2,h,d] += scale * sum_{t>=t2} pre_grad[t,h,t2] * Q[t,h,d].
//
// Transient post / post_grad / pre_grad buffers are
// [token_count, head_count, token_count] floats (~64KB at our sizes) and
// are queued for batch cleanup.
attention_back :: proc(qkv, dy, d_qkv: GpuTensor, token_count, head_count, head_size: int, causal := true, loc := #caller_location) {
	embed       := head_count * head_size
	input_size  := 3 * embed
	expected_in := token_count * input_size
	expected_o  := token_count * embed
	fmt.assertf(qkv.count   == expected_in, "attention_back: qkv %v != expected %v",   qkv.count,   expected_in, loc=loc)
	fmt.assertf(dy.count    == expected_o,  "attention_back: dy %v != expected %v",    dy.count,    expected_o,  loc=loc)
	fmt.assertf(d_qkv.count == expected_in, "attention_back: d_qkv %v != expected %v", d_qkv.count, expected_in, loc=loc)

	if _attention_back_post_pipeline      == nil { _attention_back_post_pipeline      = _make_pipeline(ATTENTION_BACK_POST_SPIRV,      2, size_of(Attention_Back_Post_Params)) }
	if _attention_back_post_grad_pipeline == nil { _attention_back_post_grad_pipeline = _make_pipeline(ATTENTION_BACK_POST_GRAD_SPIRV, 3, size_of(Attention_Back_Params_No_Scale)) }
	if _attention_back_dv_pipeline        == nil { _attention_back_dv_pipeline        = _make_pipeline(ATTENTION_BACK_DV_SPIRV,        3, size_of(Attention_Back_Params_No_Scale)) }
	if _attention_back_pre_grad_pipeline  == nil { _attention_back_pre_grad_pipeline  = _make_pipeline(ATTENTION_BACK_PRE_GRAD_SPIRV,  3, size_of(Attention_Back_Pre_Grad_Params)) }
	if _attention_back_dq_pipeline        == nil { _attention_back_dq_pipeline        = _make_pipeline(ATTENTION_BACK_DQ_SPIRV,        3, size_of(Attention_Back_Post_Params)) }
	if _attention_back_dk_pipeline        == nil { _attention_back_dk_pipeline        = _make_pipeline(ATTENTION_BACK_DK_SPIRV,        3, size_of(Attention_Back_Post_Params)) }

	scale  := f32(1) / math.sqrt(f32(head_size))
	caus_u := u32(1 if causal else 0)

	// Transient [T, H, T] buffers.
	scores_size := vk.DeviceSize(token_count * head_count * token_count * size_of(f32))
	post_buf,  post_mem  := _create_buffer(scores_size, {.STORAGE_BUFFER}, {.DEVICE_LOCAL}, loc)
	pgrad_buf, pgrad_mem := _create_buffer(scores_size, {.STORAGE_BUFFER}, {.DEVICE_LOCAL}, loc)
	preg_buf,  preg_mem  := _create_buffer(scores_size, {.STORAGE_BUFFER}, {.DEVICE_LOCAL}, loc)

	post_params := Attention_Back_Post_Params{
		token_count = u32(token_count),
		head_count  = u32(head_count),
		head_size   = u32(head_size),
		scale       = scale,
		causal      = caus_u,
	}
	{
		bufs := []vk.Buffer{ qkv.buffer, post_buf }
		_dispatch(_attention_back_post_pipeline, bufs, &post_params, u32(token_count), u32(head_count), 1, loc=loc)
	}

	ng_params := Attention_Back_Params_No_Scale{
		token_count = u32(token_count),
		head_count  = u32(head_count),
		head_size   = u32(head_size),
		causal      = caus_u,
	}
	{
		// post_grad: thread per (t, h, t2).
		bufs := []vk.Buffer{ qkv.buffer, dy.buffer, pgrad_buf }
		_dispatch(_attention_back_post_grad_pipeline, bufs, &ng_params,
			_div_up(token_count, 32), _div_up(head_count, 4), u32(token_count), loc=loc)
	}
	{
		// dV: thread per (t2, h, d).
		bufs := []vk.Buffer{ post_buf, dy.buffer, d_qkv.buffer }
		_dispatch(_attention_back_dv_pipeline, bufs, &ng_params,
			_div_up(token_count, 8), _div_up(head_count, 4), _div_up(head_size, 8), loc=loc)
	}

	pg_params := Attention_Back_Pre_Grad_Params{
		token_count = u32(token_count),
		head_count  = u32(head_count),
		causal      = caus_u,
	}
	{
		// pre_grad: workgroup per (t, h).
		bufs := []vk.Buffer{ post_buf, pgrad_buf, preg_buf }
		_dispatch(_attention_back_pre_grad_pipeline, bufs, &pg_params, u32(token_count), u32(head_count), 1, loc=loc)
	}
	{
		// dQ: thread per (t, h, d).
		bufs := []vk.Buffer{ qkv.buffer, preg_buf, d_qkv.buffer }
		_dispatch(_attention_back_dq_pipeline, bufs, &post_params,
			_div_up(token_count, 8), _div_up(head_count, 4), _div_up(head_size, 8), loc=loc)
	}
	{
		// dK: thread per (t2, h, d).
		bufs := []vk.Buffer{ qkv.buffer, preg_buf, d_qkv.buffer }
		_dispatch(_attention_back_dk_pipeline, bufs, &post_params,
			_div_up(token_count, 8), _div_up(head_count, 4), _div_up(head_size, 8), loc=loc)
	}

	_queue_destroy_buffer(post_buf,  post_mem)
	_queue_destroy_buffer(pgrad_buf, pgrad_mem)
	_queue_destroy_buffer(preg_buf,  preg_mem)
}

// adam_step: in-place Adam(W) parameter update, matching CPU `ml.update`.
//   m = beta1*m + (1-beta1)*g
//   v = beta2*v + (1-beta2)*g^2
//   x = x*(1 - lr*wd) - lr * (m/bc1) / (sqrt(v/bc2) + eps)
//   g = 0
// Caller computes bias corrections (bc1, bc2) once per step.
adam_step :: proc(x, grad, m, v: GpuTensor, lr, beta1, beta2, eps, wd, bc1, bc2: f32, loc := #caller_location) {
	fmt.assertf(grad.count == x.count && m.count == x.count && v.count == x.count,
		"adam_step: shape mismatch x=%v grad=%v m=%v v=%v", x.count, grad.count, m.count, v.count, loc=loc)

	if _adam_step_pipeline == nil {
		_adam_step_pipeline = _make_pipeline(ADAM_STEP_SPIRV, 4, size_of(Adam_Params))
	}
	params := Adam_Params{
		n     = u32(x.count),
		lr    = lr,
		beta1 = beta1,
		beta2 = beta2,
		eps   = eps,
		wd    = wd,
		bc1   = bc1,
		bc2   = bc2,
	}
	_dispatch(_adam_step_pipeline, _buffers(x, grad, m, v), &params, _div_up(x.count, 256), loc=loc)
}

// cross_entropy_grad: writes dlogits = (softmax(logits) - one_hot(targets))/N
// (assigned, not accumulated — head of the backward sweep) and the
// per-row loss into `loss_row`. Caller takes the mean of loss_row to get
// the scalar loss; small enough that doing it on the host is cheaper than
// another reduction dispatch.
//
// `targets` is uploaded to a transient host-visible u32 buffer, queued
// for batch cleanup just like `select`.
cross_entropy_grad :: proc(logits, dlogits, loss_row: GpuTensor, targets: []int, count, class_size: int, loc := #caller_location) {
	fmt.assertf(logits.count   == count * class_size, "ce_grad: logits %v != %v*%v",   logits.count,   count, class_size, loc=loc)
	fmt.assertf(dlogits.count  == count * class_size, "ce_grad: dlogits %v != %v*%v",  dlogits.count,  count, class_size, loc=loc)
	fmt.assertf(loss_row.count == count,              "ce_grad: loss_row %v != %v",    loss_row.count, count, loc=loc)
	fmt.assertf(len(targets)   == count,              "ce_grad: targets len %v != %v", len(targets),   count, loc=loc)

	// Targets buffer.
	tgt_size := vk.DeviceSize(count * size_of(u32))
	tgt_buf, tgt_mem := _create_buffer(tgt_size, {.STORAGE_BUFFER}, {.HOST_VISIBLE, .HOST_COHERENT}, loc)
	mapped: rawptr
	res := vk.MapMemory(_gpu.device, tgt_mem, 0, tgt_size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(ce_grad targets) failed: %v", res, loc=loc)
	arr := ([^]u32)(mapped)
	for v, i in targets {
		fmt.assertf(v >= 0 && v < class_size, "ce_grad: target %v out of [0, %v)", v, class_size, loc=loc)
		arr[i] = u32(v)
	}
	vk.UnmapMemory(_gpu.device, tgt_mem)

	if _cross_entropy_grad_pipeline == nil {
		_cross_entropy_grad_pipeline = _make_pipeline(CROSS_ENTROPY_GRAD_SPIRV, 4, size_of(Cross_Entropy_Grad_Params))
	}
	params := Cross_Entropy_Grad_Params{ count = u32(count), class_size = u32(class_size) }
	bufs   := []vk.Buffer{ logits.buffer, tgt_buf, dlogits.buffer, loss_row.buffer }
	_dispatch(_cross_entropy_grad_pipeline, bufs, &params, u32(count), 1, 1, loc=loc)

	_queue_destroy_buffer(tgt_buf, tgt_mem)
}

// rope_back: dx += rope^T(dy). Same sin/cos schedule as the forward kernel.
rope_back :: proc(dx, dy: GpuTensor, token_count, head_count, head_size: int, base: f32 = 10000, loc := #caller_location) {
	fmt.assertf(head_size % 2 == 0, "rope_back: head_size %v must be even", head_size, loc=loc)
	expected := token_count * head_count * head_size
	fmt.assertf(dx.count == expected, "rope_back: dx %v != expected %v", dx.count, expected, loc=loc)
	fmt.assertf(dy.count == expected, "rope_back: dy %v != expected %v", dy.count, expected, loc=loc)

	if _rope_back_pipeline == nil {
		_rope_back_pipeline = _make_pipeline(ROPE_BACK_SPIRV, 2, size_of(Rope_Back_Params))
	}
	params := Rope_Back_Params{
		token_count = u32(token_count),
		head_count  = u32(head_count),
		head_size   = u32(head_size),
		base        = base,
	}
	total_pairs := token_count * head_count * (head_size / 2)
	_dispatch(_rope_back_pipeline, _buffers(dx, dy), &params, _div_up(total_pairs, 256), loc=loc)
}

// --- Internal ---

ADD_SPIRV :: #load("shaders/add.spv", []u8)
ADD_LOCAL_SIZE :: 256

Add_Params :: struct {
	n: u32,
}

_add_pipeline: ^Pipeline

LINEAR_SPIRV :: #load("shaders/linear.spv", []u8)
// Tile sizes — must match #defines in linear.comp.
LINEAR_LOCAL_X :: 64  // TILE_M (output rows per workgroup)
LINEAR_LOCAL_Y :: 64  // TILE_N (output cols per workgroup)

Linear_Params :: struct {
	count:       u32,
	input_size:  u32,
	output_size: u32,
}

_linear_pipeline: ^Pipeline

GELU_SPIRV      :: #load("shaders/gelu.spv", []u8)
GELU_LOCAL_SIZE :: 256
Gelu_Params :: struct { n: u32 }
_gelu_pipeline: ^Pipeline

LAYERNORM_SPIRV :: #load("shaders/layernorm.spv", []u8)
Layernorm_Params :: struct { count, size: u32 }
_layernorm_pipeline: ^Pipeline

SOFTMAX_SPIRV   :: #load("shaders/softmax.spv", []u8)
Softmax_Params :: struct { count, size: u32 }
_softmax_pipeline: ^Pipeline

SOFTMAX_BACK_SPIRV :: #load("shaders/softmax_back.spv", []u8)
Softmax_Back_Params :: struct { count, size: u32 }
_softmax_back_pipeline: ^Pipeline

PERMUTE_SPIRV :: #load("shaders/permute.spv", []u8)
PERMUTE_BACK_SPIRV :: #load("shaders/permute_back.spv", []u8)
Permute_Params :: struct {
	out_d0, out_d1, out_d2: u32,
	in_d1,  in_d2:          u32,
	axes_0, axes_1, axes_2: u32,
}
_permute_pipeline:      ^Pipeline
_permute_back_pipeline: ^Pipeline

CAUSAL_MASK_SPIRV      :: #load("shaders/causal_mask.spv", []u8)
CAUSAL_MASK_BACK_SPIRV :: #load("shaders/causal_mask_back.spv", []u8)
Causal_Mask_Params :: struct { total, T: u32 }
_causal_mask_pipeline:      ^Pipeline
_causal_mask_back_pipeline: ^Pipeline

SELECT_SPIRV :: #load("shaders/select.spv", []u8)
Select_Params :: struct { n_indices, size: u32 }
_select_pipeline: ^Pipeline

ROPE_SPIRV :: #load("shaders/rope.spv", []u8)
Rope_Params :: struct {
	token_count, head_count, head_size: u32,
	base: f32,
}
_rope_pipeline: ^Pipeline

SLICE_TRAILING_SPIRV :: #load("shaders/slice_trailing.spv", []u8)
Slice_Trailing_Params :: struct { leading, trailing, new_trailing, start: u32 }
_slice_trailing_pipeline: ^Pipeline

CONCAT3_SPIRV :: #load("shaders/concat3.spv", []u8)
Concat3_Params :: struct { leading, t_a, t_b, t_c: u32 }
_concat3_pipeline: ^Pipeline

ATTENTION_SPIRV :: #load("shaders/attention.spv", []u8)
Attention_Params :: struct {
	token_count, head_count, head_size: u32,
	scale: f32,
	causal: u32,
}
_attention_pipeline: ^Pipeline

ZERO_SPIRV :: #load("shaders/zero.spv", []u8)
Zero_Params :: struct { n: u32 }
_zero_pipeline: ^Pipeline

ADD_BACK_SPIRV :: #load("shaders/add_back.spv", []u8)
Add_Back_Params :: struct { n: u32 }
_add_back_pipeline: ^Pipeline

GELU_BACK_SPIRV :: #load("shaders/gelu_back.spv", []u8)
Gelu_Back_Params :: struct { n: u32 }
_gelu_back_pipeline: ^Pipeline

SLICE_TRAILING_BACK_SPIRV :: #load("shaders/slice_trailing_back.spv", []u8)
Slice_Trailing_Back_Params :: struct { leading, trailing, new_trailing, start: u32 }
_slice_trailing_back_pipeline: ^Pipeline

CONCAT3_BACK_SPIRV :: #load("shaders/concat3_back.spv", []u8)
Concat3_Back_Params :: struct { leading, t_a, t_b, t_c: u32 }
_concat3_back_pipeline: ^Pipeline

ROPE_BACK_SPIRV :: #load("shaders/rope_back.spv", []u8)
Rope_Back_Params :: struct {
	token_count, head_count, head_size: u32,
	base: f32,
}
_rope_back_pipeline: ^Pipeline

LINEAR_BACK_INPUT_SPIRV  :: #load("shaders/linear_back_input.spv",  []u8)
LINEAR_BACK_WEIGHT_SPIRV :: #load("shaders/linear_back_weight.spv", []u8)
Linear_Back_Params :: struct { count, input_size, output_size: u32 }
_linear_back_input_pipeline:  ^Pipeline
_linear_back_weight_pipeline: ^Pipeline

LAYERNORM_STATS_SPIRV       :: #load("shaders/layernorm_stats.spv",       []u8)
LAYERNORM_BACK_INPUT_SPIRV  :: #load("shaders/layernorm_back_input.spv",  []u8)
LAYERNORM_BACK_WEIGHT_SPIRV :: #load("shaders/layernorm_back_weight.spv", []u8)
Layernorm_Stats_Params :: struct { count, size: u32 }
Layernorm_Back_Params  :: struct { count, size: u32 }
_layernorm_stats_pipeline:        ^Pipeline
_layernorm_back_input_pipeline:   ^Pipeline
_layernorm_back_weight_pipeline:  ^Pipeline

SELECT_BACK_SPIRV :: #load("shaders/select_back.spv", []u8)
Select_Back_Params :: struct { vocab, n_indices, size: u32 }
_select_back_pipeline: ^Pipeline

ATTENTION_BACK_POST_SPIRV      :: #load("shaders/attention_back_post.spv",      []u8)
ATTENTION_BACK_POST_GRAD_SPIRV :: #load("shaders/attention_back_post_grad.spv", []u8)
ATTENTION_BACK_DV_SPIRV        :: #load("shaders/attention_back_dv.spv",        []u8)
ATTENTION_BACK_PRE_GRAD_SPIRV  :: #load("shaders/attention_back_pre_grad.spv",  []u8)
ATTENTION_BACK_DQ_SPIRV        :: #load("shaders/attention_back_dq.spv",        []u8)
ATTENTION_BACK_DK_SPIRV        :: #load("shaders/attention_back_dk.spv",        []u8)
Attention_Back_Post_Params :: struct {
	token_count, head_count, head_size: u32,
	scale: f32,
	causal: u32,
}
Attention_Back_Params_No_Scale :: struct {
	token_count, head_count, head_size, causal: u32,
}
Attention_Back_Pre_Grad_Params :: struct {
	token_count, head_count, causal: u32,
}
_attention_back_post_pipeline:      ^Pipeline
_attention_back_post_grad_pipeline: ^Pipeline
_attention_back_dv_pipeline:        ^Pipeline
_attention_back_pre_grad_pipeline:  ^Pipeline
_attention_back_dq_pipeline:        ^Pipeline
_attention_back_dk_pipeline:        ^Pipeline

CROSS_ENTROPY_GRAD_SPIRV :: #load("shaders/cross_entropy_grad.spv", []u8)
Cross_Entropy_Grad_Params :: struct { count, class_size: u32 }
_cross_entropy_grad_pipeline: ^Pipeline

ADAM_STEP_SPIRV :: #load("shaders/opt_step_adam.spv", []u8)
Adam_Params :: struct {
	n: u32,
	lr, beta1, beta2, eps, wd, bc1, bc2: f32,
}
_adam_step_pipeline: ^Pipeline
