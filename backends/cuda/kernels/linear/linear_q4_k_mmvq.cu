// Q4_K weight x Q8_1 activation GEMV (M=1 decode path). Mirrors ggml's mmvq.
//
//   block_q4_K (144 bytes / 256 elements):
//     half  d, dmin                     (4 bytes)
//     u8    scales_mins[12]             (12 bytes packed 6-bit scales+mins)
//     u8    qs[128]                     (128 bytes 4-bit nibbles)
//   block_q8_1 (36 bytes / 32 elements):
//     half  d, s                        (s == d * sum(qs[i]))
//     i8    qs[32]                      (packed 4-per-uint)
//
// Algorithm per Q4_K block:
//   sum_e w(e)*x(e) = sum_{sub=0..7}   d * scale_s * d_q8 * dp4a(q4, q8)
//                                    - dmin * min_s * s_q8
#include <cuda_fp16.h>

__device__ __forceinline__ void unpack_scale_min(int sub,
                                                 unsigned int sw0,
                                                 unsigned int sw1,
                                                 unsigned int sw2,
                                                 int& scale,
                                                 int& min_v) {
	if (sub < 4) {
		int shift = sub * 8;
		scale = (int)((sw0 >> shift) & 0x3Fu);
		min_v = (int)((sw1 >> shift) & 0x3Fu);
	} else {
		int shift = (sub - 4) * 8;
		unsigned int b_low = (sw2 >> shift) & 0xFFu;
		unsigned int b_sc  = (sw0 >> shift) & 0xFFu;
		unsigned int b_mn  = (sw1 >> shift) & 0xFFu;
		scale = (int)((b_low & 0x0Fu) | ((b_sc >> 6) << 4));
		min_v = (int)((b_low >>  4)  | ((b_mn >> 6) << 4));
	}
}

__device__ __forceinline__ unsigned int bf16_from_f32(float v) {
	unsigned int bits = __float_as_uint(v);
	if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;  // NaN
	unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
	return (rounded >> 16) & 0xffffu;
}

#define ROWS_PER_WG 2
#define BLOCK_UINTS 36   // 144 / 4

extern "C" __global__
void linear_q4_k_mmvq(const unsigned int* __restrict__ x,   // q8_1 stream
                      const unsigned int* __restrict__ w,   // q4_k stream
                      unsigned int*       __restrict__ y,   // bf16 packed pairs
                      int M, int K, int N) {
	int n_base = blockIdx.x * ROWS_PER_WG;
	int tid    = threadIdx.x;

	int num_blocks = K / 256;

	float acc[ROWS_PER_WG];
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) acc[r] = 0.0f;

	for (int b = 0; b < num_blocks; ++b) {
		int q8_block_base = b * 8;

		#pragma unroll
		for (int r = 0; r < ROWS_PER_WG; ++r) {
			int n = n_base + r;
			if (n >= N) continue;

			int block_off = (n * num_blocks + b) * BLOCK_UINTS;

			unsigned int dm = __ldg(&w[block_off]);
			float d    = __half2float(__ushort_as_half((unsigned short)(dm & 0xffff)));
			float dmin = __half2float(__ushort_as_half((unsigned short)((dm >> 16) & 0xffff)));
			unsigned int sw0 = __ldg(&w[block_off + 1]);
			unsigned int sw1 = __ldg(&w[block_off + 2]);
			unsigned int sw2 = __ldg(&w[block_off + 3]);

			float row_acc = 0.0f;

			#pragma unroll
			for (int chunk_pair = 0; chunk_pair < 2; ++chunk_pair) {
				int chunk = tid + chunk_pair * 32;
				int sub   = chunk >> 3;
				int inner = chunk & 7;

				unsigned int nib_word = __ldg(&w[block_off + 4 + (sub >> 1) * 8 + inner]);
				int shift     = (sub & 1) * 4;
				int q4_packed = (int)((nib_word >> shift) & 0x0F0F0F0Fu);

				int q8_off    = (q8_block_base + sub) * 9;
				int q8_packed = (int)__ldg(&x[q8_off + 1 + inner]);

				int int_dot = __dp4a(q4_packed, q8_packed, 0);

				unsigned int q8_ds = __ldg(&x[q8_off]);
				float q8_d = __half2float(__ushort_as_half((unsigned short)(q8_ds & 0xffff)));
				float q8_s = __half2float(__ushort_as_half((unsigned short)((q8_ds >> 16) & 0xffff)));

				int scale, min_v;
				unpack_scale_min(sub, sw0, sw1, sw2, scale, min_v);

				row_acc += d * (float)scale * q8_d * (float)int_dot;

				if (inner == 0) {
					row_acc -= dmin * (float)min_v * q8_s;
				}
			}

			acc[r] += row_acc;
		}
	}

	// Warp reduction.
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) {
		#pragma unroll
		for (int off = 16; off > 0; off >>= 1) {
			acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
		}
	}

	if (tid == 0) {
		#pragma unroll
		for (int pair = 0; pair < ROWS_PER_WG / 2; ++pair) {
			int n_lo = n_base + 2 * pair;
			if (n_lo >= N) break;
			int n_hi = n_lo + 1;

			unsigned int lo = bf16_from_f32(acc[2 * pair]);
			unsigned int hi = (n_hi < N) ? bf16_from_f32(acc[2 * pair + 1]) : 0u;
			y[n_lo >> 1] = (hi << 16) | lo;
		}
	}
}
