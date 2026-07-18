package machine_learning

import "base:builtin"

import "core:mem"

dequantize_q4_k :: proc(src: []byte, dst: []f32) {
	assert(builtin.len(dst) % K_QUANT_BLOCK_SIZE == 0, "count must be a multiple of 256")
	assert(builtin.len(src) == (builtin.len(dst) / K_QUANT_BLOCK_SIZE) * Q4_K_BLOCK_BYTES, "src byte count mismatch")

	num_blocks := builtin.len(dst) / K_QUANT_BLOCK_SIZE

	for b in 0 ..< num_blocks {
		base := b * Q4_K_BLOCK_BYTES

		d_h:    f16
		dmin_h: f16
		mem.copy(&d_h,    raw_data(src[base + 0:]), 2)
		mem.copy(&dmin_h, raw_data(src[base + 2:]), 2)
		d    := f32(d_h)
		dmin := f32(dmin_h)

		scales_packed := src[base + 4 : base + 16]
		quants        := src[base + 16 : base + 144]

		out := dst[b * K_QUANT_BLOCK_SIZE : (b + 1) * K_QUANT_BLOCK_SIZE]

		is := 0
		for j := 0; j < K_QUANT_BLOCK_SIZE; j += 64 {
			sc1, m1 := _unpack_scale_min_k4(is + 0, scales_packed)
			sc2, m2 := _unpack_scale_min_k4(is + 1, scales_packed)
			d1, n1 := d * f32(sc1), dmin * f32(m1)
			d2, n2 := d * f32(sc2), dmin * f32(m2)

			q := quants[(j / 2) : (j / 2) + 32]
			for l in 0 ..< 32 {
				out[j +      l] = d1 * f32(q[l] & 0x0F) - n1
				out[j + 32 + l] = d2 * f32(q[l] >>   4) - n2
			}
			is += 2
		}
	}
}

dequantize_q6_k :: proc(src: []byte, dst: []f32) {
	assert(builtin.len(dst) % K_QUANT_BLOCK_SIZE == 0, "count must be a multiple of 256")
	assert(builtin.len(src) == (builtin.len(dst) / K_QUANT_BLOCK_SIZE) * Q6_K_BLOCK_BYTES, "src byte count mismatch")

	num_blocks := builtin.len(dst) / K_QUANT_BLOCK_SIZE

	for b in 0 ..< num_blocks {
		base := b * Q6_K_BLOCK_BYTES

		ql := src[base +   0 : base + 128]
		qh := src[base + 128 : base + 192]

		sc_signed := transmute([^]i8)(raw_data(src[base + 192:]))

		d_h: f16
		mem.copy(&d_h, raw_data(src[base + 208:]), 2)
		d := f32(d_h)

		out := dst[b * K_QUANT_BLOCK_SIZE : (b + 1) * K_QUANT_BLOCK_SIZE]

		for half in 0 ..< 2 {
			ql_h := ql[half * 64 :]
			qh_h := qh[half * 32 :]
			sc_h := sc_signed[half * 8 :]
			y    := out[half * 128 :]

			for l in 0 ..< 32 {
				is := l / 16
				q1 := i32((ql_h[l +  0] & 0x0F) | u8(((qh_h[l] >> 0) & 3) << 4)) - 32
				q2 := i32((ql_h[l + 32] & 0x0F) | u8(((qh_h[l] >> 2) & 3) << 4)) - 32
				q3 := i32((ql_h[l +  0]  >> 4) | u8(((qh_h[l] >> 4) & 3) << 4)) - 32
				q4 := i32((ql_h[l + 32]  >> 4) | u8(((qh_h[l] >> 6) & 3) << 4)) - 32

				y[l +  0] = d * f32(sc_h[is + 0]) * f32(q1)
				y[l + 32] = d * f32(sc_h[is + 2]) * f32(q2)
				y[l + 64] = d * f32(sc_h[is + 4]) * f32(q3)
				y[l + 96] = d * f32(sc_h[is + 6]) * f32(q4)
			}
		}
	}
}

@(require_results)
_unpack_scale_min_k4 :: #force_inline proc "contextless" (j: int, q: []byte) -> (scale, min: u8) {
	if j < 4 {
		scale = q[j]     & 0x3F
		min   = q[j + 4] & 0x3F
	} else {
		scale = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
		min   = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4)
	}
	return
}
