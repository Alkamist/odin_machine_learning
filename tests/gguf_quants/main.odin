package gguf_quants

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"

import "../../loaders/gguf"

main :: proc() {
	any_failed := false
	check(_test_q4k_constructed_block(), "Q4_K constructed-block dequant matches hand-computed", &any_failed)
	check(_test_q6k_constructed_block(), "Q6_K constructed-block dequant matches hand-computed", &any_failed)

	if len(os.args) >= 2 {
		check(_test_real_tensors(os.args[1]), "Q4_K/Q6_K dequant on real Gemma 4 E4B tensors is finite + bounded", &any_failed)
	} else {
		fmt.println("(skipping real-tensor smoke test; pass GGUF path as first arg to enable)")
	}

	if any_failed {
		os.exit(1)
	}
	fmt.println("ok")
}

check :: proc(ok: bool, name: string, any_failed: ^bool) {
	if ok {
		fmt.printfln("  PASS  %v", name)
	} else {
		fmt.printfln("  FAIL  %v", name)
		any_failed^ = true
	}
}

// Build a single Q4_K block where dequant should produce a known sequence,
// then dequantize and compare bit-for-bit.
//
// Block layout: d (fp16) | dmin (fp16) | scales[12] | qs[128]
//
// Choose: d = 1.0, dmin = 0.0
//         all eight 6-bit scales = 1, all eight 6-bit mins = 0
//         qs nibbles repeat [0..15]
// Then weights[i] = 1 * 1 * (qs_nib[i]) - 0 * 0 = qs_nib[i] ∈ {0..15}.
_test_q4k_constructed_block :: proc() -> bool {
	src: [gguf.Q4_K_BYTES]byte

	// d = 1.0, dmin = 0.0 (fp16)
	d_h    := f16(1.0)
	dmin_h := f16(0.0)
	mem.copy(raw_data(src[0:]), &d_h,    2)
	mem.copy(raw_data(src[2:]), &dmin_h, 2)

	// All scales = 1 (low 6 bits), all mins = 0.
	// Layout: q[0..3] hold scales 0..3 in low 6 bits; q[4..7] hold mins 0..3.
	// q[8..11] each hold high-2-bits of scale[i] in bits 0..5 + high-2-bits of
	// min[i] in bits 0..5 — but for sc≤63 and m≤63, q[8..11] = 0.
	src[4 + 0] = 1
	src[4 + 1] = 1
	src[4 + 2] = 1
	src[4 + 3] = 1
	// q[4..7] (mins 0..3) already zero from `[..]byte` init.
	// For j ∈ [4, 8): scale[j] = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
	//                min[j]   = (q[j+4] >> 4)   | ((q[j-0] >> 6) << 4)
	// We want scale = 1, min = 0 → q[j+4] = 0x01 (low 4 bits = 1, high 4 = 0).
	src[4 + 8 + 0] = 0x01
	src[4 + 8 + 1] = 0x01
	src[4 + 8 + 2] = 0x01
	src[4 + 8 + 3] = 0x01

	// qs[0..127]: nibbles cycle 0..15. Two nibbles per byte (low first).
	for i in 0 ..< 128 {
		lo := u8((2 * i + 0) & 0xF)
		hi := u8((2 * i + 1) & 0xF)
		src[16 + i] = lo | (hi << 4)
	}

	// Hand-compute expected output. The dequant unpacks per sub-block of 32:
	// for j = 0,32,64,...: take 32 low-nibbles of qs[j/2:j/2+32], then 32
	// high-nibbles of the same window. So index in `out`:
	//   out[j +      l] = qs_lo[ (j/2) + l ]   for l in 0..32
	//   out[j + 32 + l] = qs_hi[ (j/2) + l ]   for l in 0..32
	expected: [gguf.QK_K]f32
	for j := 0; j < gguf.QK_K; j += 64 {
		for l in 0 ..< 32 {
			lo_byte := src[16 + (j / 2) + l]
			expected[j      + l] = f32(lo_byte & 0x0F)
			expected[j + 32 + l] = f32(lo_byte >>   4)
		}
	}

	got: [gguf.QK_K]f32
	gguf.dequantize_q4_k(src[:], got[:])
	for i in 0 ..< gguf.QK_K {
		if got[i] != expected[i] {
			fmt.printfln("    Q4_K mismatch at %v: got %v expected %v", i, got[i], expected[i])
			return false
		}
	}
	return true
}

// Q6_K: 256 weights → ql[128] (low 4 bits) | qh[64] (upper 2 bits) | scales[16] (i8) | d (fp16).
// d = 1.0, all 16 scales = 1, q[i] reconstructed = 32 (so weight = 1 * 1 * (32 - 32) = 0).
//
// Easy-to-verify variant: set the 6-bit quant to (i % 64) and verify
// weight[i] = scale[i/16] * (q[i] - 32) = (q[i] - 32).
_test_q6k_constructed_block :: proc() -> bool {
	src: [gguf.Q6_K_BYTES]byte

	// scales: 16 i8 values = 1
	for i in 0 ..< 16 {
		src[192 + i] = 1
	}
	// d = 1.0
	d_h := f16(1.0)
	mem.copy(raw_data(src[208:]), &d_h, 2)

	// Layout for the 256 quants comes from the dequant:
	//   q1 reconstructed for l in 0..32 takes from ql[l + 0] low + qh[l] bits 0..1
	//   q2: ql[l + 32] low  + qh[l] bits 2..3
	//   q3: ql[l +  0] high + qh[l] bits 4..5
	//   q4: ql[l + 32] high + qh[l] bits 6..7
	//   y[l],  y[l+32], y[l+64], y[l+96]   for the first 128-element half;
	//   then ql += 64, qh += 32, sc += 8 for the second half.
	//
	// Set every 6-bit quant to a value `v(i)` where i is the destination index.
	// We'll choose v(i) = (i % 64) — 6 bits — so weight(i) = (i % 64) - 32.
	put_q :: proc(src: []byte, half, idx_in_half: int, v: u8) {
		// idx_in_half ∈ [0, 128) selects which of (q1,q2,q3,q4, l=0..31)
		// this is. Mapping per the loop:
		//   y[l       ]  ← q1 ← ql[l +  0] low  + qh[l] bits 0..1   (idx_in_half =  0..31)
		//   y[l +  32]  ← q2 ← ql[l + 32] low  + qh[l] bits 2..3   (idx_in_half = 32..63)
		//   y[l +  64]  ← q3 ← ql[l +  0] high + qh[l] bits 4..5   (idx_in_half = 64..95)
		//   y[l +  96]  ← q4 ← ql[l + 32] high + qh[l] bits 6..7   (idx_in_half = 96..127)
		base_ql := half * 64 // start of ql for this half
		base_qh := 128 + half * 32 // start of qh for this half
		l       := idx_in_half % 32

		lo := v & 0x0F
		hi := (v >> 4) & 0x03

		ql_pos: int
		qh_shift: u8
		switch idx_in_half / 32 {
		case 0: ql_pos = base_ql + l       ; qh_shift = 0
		case 1: ql_pos = base_ql + l +  32 ; qh_shift = 2
		case 2: ql_pos = base_ql + l       ; qh_shift = 4
		case 3: ql_pos = base_ql + l +  32 ; qh_shift = 6
		}

		if idx_in_half / 32 == 0 || idx_in_half / 32 == 1 {
			// Low nibble destination.
			src[ql_pos] = (src[ql_pos] & 0xF0) | lo
		} else {
			// High nibble destination.
			src[ql_pos] = (src[ql_pos] & 0x0F) | (lo << 4)
		}
		src[base_qh + l] |= (hi << qh_shift)
	}

	// Encode the 256 quants. Output index:
	//   half ∈ {0,1}, idx_in_half maps to y[half*128 + dest_l] where:
	//     idx_in_half  0..31 → l +   0
	//     idx_in_half 32..63 → l +  32
	//     idx_in_half 64..95 → l +  64
	//     idx_in_half 96..127 → l + 96
	// We want the value at output index `i` to be (i % 64).
	for half in 0 ..< 2 {
		for idx_in_half in 0 ..< 128 {
			l        := idx_in_half % 32
			quadrant := idx_in_half / 32
			out_idx  := half * 128 + l + quadrant * 32
			v        := u8(out_idx % 64)
			put_q(src[:], half, idx_in_half, v)
		}
	}

	got: [gguf.QK_K]f32
	gguf.dequantize_q6_k(src[:], got[:])
	for i in 0 ..< gguf.QK_K {
		expected := f32(i32(i % 64) - 32)
		if got[i] != expected {
			fmt.printfln("    Q6_K mismatch at %v: got %v expected %v", i, got[i], expected)
			return false
		}
	}
	return true
}

// Smoke test on real model weights. Loads the GGUF, dequantizes the first
// Q4_K and Q6_K tensor we find, and asserts: all values finite, abs(value)
// bounded. This catches gross errors (NaN explosions, off-by-N indexing
// bugs) without requiring an external reference.
_test_real_tensors :: proc(path: string) -> bool {
	loader, ok := gguf.load(path)
	if !ok do return false
	defer gguf.destroy(loader)

	q4k_ok := _check_first_of_type(loader, .Q4_K)
	q6k_ok := _check_first_of_type(loader, .Q6_K)
	return q4k_ok && q6k_ok
}

_check_first_of_type :: proc(loader: gguf.Loader, ty: gguf.Tensor_Type) -> bool {
	for name, info in loader.tensors {
		if info.type != ty do continue
		bytes, bytes_ok := gguf.get_bytes(loader, name)
		if !bytes_ok do return false

		// Element count = product of shape dims.
		count := 1
		for d in info.shape do count *= d

		// Dequantize at most the first ~4096 weights (16 Q4_K blocks or
		// Q6_K blocks) so the test stays fast.
		max_count := count
		if max_count > 4096 do max_count = 4096
		max_count -= max_count % gguf.QK_K
		if max_count == 0 do max_count = gguf.QK_K

		bytes_per_block := 0
		#partial switch ty {
		case .Q4_K: bytes_per_block = gguf.Q4_K_BYTES
		case .Q6_K: bytes_per_block = gguf.Q6_K_BYTES
		case:       return false
		}
		num_blocks := max_count / gguf.QK_K
		src        := bytes[: num_blocks * bytes_per_block]
		dst        := make([]f32, max_count)
		defer delete(dst)

		#partial switch ty {
		case .Q4_K: gguf.dequantize_q4_k(src, dst)
		case .Q6_K: gguf.dequantize_q6_k(src, dst)
		}

		nan_count, inf_count: int
		max_abs: f32
		for v in dst {
			if math.is_nan(v) {
				nan_count += 1
			} else if math.is_inf(v) {
				inf_count += 1
			} else {
				a := v
				if a < 0 do a = -a
				if a > max_abs do max_abs = a
			}
		}
		fmt.printfln("    %v %v shape=%v max_abs=%.4f", ty, name, info.shape, max_abs)
		if nan_count != 0 || inf_count != 0 {
			fmt.printfln("    %v: %v NaN, %v Inf in dequantized output", name, nan_count, inf_count)
			return false
		}
		if max_abs > 100.0 {
			fmt.printfln("    %v: max_abs=%.3f looks unrealistically large", name, max_abs)
			return false
		}
		return true
	}
	fmt.printfln("    no %v tensor found in model", ty)
	return false
}
