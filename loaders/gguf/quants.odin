package gguf

import "core:mem"

// Block layouts and dequant formulas mirror
// `dequantize_row_q4_K` / `dequantize_row_q6_K` in
// llama.cpp/ggml/src/ggml-quants.c. The on-disk byte layout is also fixed
// there; we read it via `mem.copy` to avoid alignment assumptions.

QK_K       :: 256 // super-block size (k-quants)
Q4_K_BYTES :: 144 // 2 + 2 + 12 + 128 = sizeof(block_q4_K)
Q6_K_BYTES :: 210 // 128 + 64 + 16 + 2 = sizeof(block_q6_K)

// Dequantize Q4_K weights to f32. `src` must be exactly `count/256 * 144`
// bytes; `count` must be a multiple of 256. Output `dst` length == count.
dequantize_q4_k :: proc(src: []byte, dst: []f32) {
	assert(len(dst) % QK_K == 0,           "dequantize_q4_k: count must be a multiple of 256")
	assert(len(src) == (len(dst) / QK_K) * Q4_K_BYTES, "dequantize_q4_k: src byte count mismatch")

	num_blocks := len(dst) / QK_K

	for b in 0 ..< num_blocks {
		base := b * Q4_K_BYTES

		d_h:    f16
		dmin_h: f16
		mem.copy(&d_h,    raw_data(src[base + 0:]), 2)
		mem.copy(&dmin_h, raw_data(src[base + 2:]), 2)
		d    := f32(d_h)
		dmin := f32(dmin_h)

		scales_packed := src[base + 4 : base + 16] // 12 bytes, 8 sub-blocks × (6-bit scale + 6-bit min)
		quants        := src[base + 16 : base + 144] // 128 bytes, 256 nibbles

		out := dst[b * QK_K : (b + 1) * QK_K]

		// Two sub-blocks per iteration; unpack the 6-bit scale/min pair via
		// `_unpack_scale_min_k4`, then dequant 32 low nibbles (sc1,m1)
		// followed by 32 high nibbles (sc2,m2) of the same 32 quant bytes.
		is := 0
		for j := 0; j < QK_K; j += 64 {
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

// Dequantize Q6_K weights to f32. `src` length must be `count/256 * 210`,
// `count` a multiple of 256, `dst` length == count.
dequantize_q6_k :: proc(src: []byte, dst: []f32) {
	assert(len(dst) % QK_K == 0,           "dequantize_q6_k: count must be a multiple of 256")
	assert(len(src) == (len(dst) / QK_K) * Q6_K_BYTES, "dequantize_q6_k: src byte count mismatch")

	num_blocks := len(dst) / QK_K

	for b in 0 ..< num_blocks {
		base := b * Q6_K_BYTES

		ql := src[base +   0 : base + 128] // low 4 bits of each 6-bit quant
		qh := src[base + 128 : base + 192] // upper 2 bits of each 6-bit quant
		// `scales` is 16 signed i8 values (one per 16 elements).
		sc_signed := transmute([^]i8)(raw_data(src[base + 192:]))

		d_h: f16
		mem.copy(&d_h, raw_data(src[base + 208:]), 2)
		d := f32(d_h)

		out := dst[b * QK_K : (b + 1) * QK_K]

		// Process two 128-element halves (matches the upstream loop).
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

// Unpack the j-th sub-block's 6-bit scale and 6-bit min from the 12-byte
// packed `scales` field of a Q4_K (or Q5_K) block. j ∈ [0, 8).
//
// Layout (mirrors `get_scale_min_k4` in ggml-quants.c):
//   For j < 4:  scale = q[j]   & 0x3F                   ; min = q[j+4] & 0x3F
//   For j ≥ 4:  scale = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
//               min   = (q[j+4] >>  4)  | ((q[j  ] >> 6) << 4)
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