// Q6_K x bf16 GEMV (M=1 decode path). Mirrors ggml's Q6_K dequantize +
// dot-product layout. No Q8_1 step — we read x as bf16 directly.
//
//   block_q6_K (210 bytes / 256 elements):
//     ql      bytes   0..127   (low 4 bits of each 6-bit quant, 2 nibbles/byte)
//     qh      bytes 128..191   (upper 2 bits of each 6-bit quant, 4/byte)
//     scales  bytes 192..207   (16 i8 sub-block scales, one per 16 weights)
//     d       bytes 208..209   (fp16 super-block scale)
//
// Per-element dequant: w(e) = d * scales[e/16] * (ql_nibble | qh_2bits<<4 - 32)
//
// Block layout: 4 warps x 32 lanes = 128 threads. Each block produces
// ROWS_PER_WG=2 output rows; the 4 warps cooperate on K (each warp handles
// 1/NWARPS of the K-blocks), then a tree reduction folds across warps.
#include <cuda_fp16.h>

__device__ __forceinline__ unsigned int q6k_load_byte(const unsigned int* w, int byte_offset) {
	unsigned int u = w[byte_offset >> 2];
	return (u >> ((byte_offset & 3) * 8)) & 0xFFu;
}

__device__ __forceinline__ int q6k_load_i8(const unsigned int* w, int byte_offset) {
	unsigned int b = q6k_load_byte(w, byte_offset);
	int s = (int)b;
	if ((b & 0x80u) != 0u) s -= 256;
	return s;
}

__device__ __forceinline__ unsigned int bf16_from_f32_q6k(float v) {
	unsigned int bits = __float_as_uint(v);
	if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
	unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
	return (rounded >> 16) & 0xffffu;
}

#define NWARPS       4
#define WARP_SIZE   32
#define ROWS_PER_WG  2
#define BLOCK_K    256
#define BLOCK_BYTES 210

extern "C" __global__
__launch_bounds__(NWARPS * WARP_SIZE, 1)
void linear_q6_k_gemv(const unsigned int* __restrict__ x,   // bf16 packed pairs
                      const unsigned int* __restrict__ w,   // q6_k byte stream
                      unsigned int*       __restrict__ y,   // bf16 packed pairs
                      int M, int K, int N) {
	int n_base = blockIdx.x * ROWS_PER_WG;
	int wid    = threadIdx.y;
	int lane   = threadIdx.x;

	int num_blocks = K / BLOCK_K;

	float acc[ROWS_PER_WG];
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) acc[r] = 0.0f;

	for (int b = wid; b < num_blocks; b += NWARPS) {
		// Load the bf16 input chunk for this K-block once and share across
		// the row loop. Each lane covers 8 elements (= 2 halves x 4 quadrants);
		// hoisting these out of the row+quadrant inner loops removes the
		// redundant per-row x reads.
		float xv[2][4];
		#pragma unroll
		for (int h = 0; h < 2; ++h) {
			#pragma unroll
			for (int quadrant = 0; quadrant < 4; ++quadrant) {
				int e = h * 128 + quadrant * 32 + lane;
				int k = b * BLOCK_K + e;
				unsigned int pkx = __ldg(&x[k >> 1]);
				unsigned short half_bits = (k & 1) ? (unsigned short)(pkx >> 16)
				                                   : (unsigned short)(pkx & 0xffff);
				unsigned int as_u32 = (unsigned int)half_bits << 16;
				xv[h][quadrant] = __int_as_float((int)as_u32);
			}
		}

		#pragma unroll
		for (int r = 0; r < ROWS_PER_WG; ++r) {
			int n = n_base + r;
			if (n >= N) continue;

			int block_off = (n * num_blocks + b) * BLOCK_BYTES;

			unsigned int d_bits = q6k_load_byte(w, block_off + 208) | (q6k_load_byte(w, block_off + 209) << 8);
			float d = __half2float(__ushort_as_half((unsigned short)d_bits));

			#pragma unroll
			for (int h = 0; h < 2; ++h) {
				unsigned int ql_lo = q6k_load_byte(w, block_off + h * 64 +      lane);
				unsigned int ql_hi = q6k_load_byte(w, block_off + h * 64 + 32 + lane);
				unsigned int qh    = q6k_load_byte(w, block_off + 128 + h * 32 + lane);

				int sb_base = h * 8 + (lane >> 4);
				int sc0 = q6k_load_i8(w, block_off + 192 + sb_base + 0);
				int sc1 = q6k_load_i8(w, block_off + 192 + sb_base + 2);
				int sc2 = q6k_load_i8(w, block_off + 192 + sb_base + 4);
				int sc3 = q6k_load_i8(w, block_off + 192 + sb_base + 6);

				#pragma unroll
				for (int quadrant = 0; quadrant < 4; ++quadrant) {
					unsigned int ql_byte   = ((quadrant & 1) == 0) ? ql_lo : ql_hi;
					unsigned int ql_nibble = (ql_byte >> ((quadrant >> 1) * 4)) & 0x0Fu;
					unsigned int qh_bits   = (qh      >>  (quadrant       * 2)) & 0x03u;
					int q = (int)(ql_nibble | (qh_bits << 4)) - 32;

					int scale_i8 = (quadrant == 0) ? sc0
					             : (quadrant == 1) ? sc1
					             : (quadrant == 2) ? sc2
					             :                   sc3;

					acc[r] += d * (float)scale_i8 * (float)q * xv[h][quadrant];
				}
			}
		}
	}

	// Warp reduction: collapse each warp to lane 0.
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) {
		#pragma unroll
		for (int off = 16; off > 0; off >>= 1) {
			acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
		}
	}

	// Cross-warp reduction via shared memory.
	__shared__ float warp_sums[NWARPS][ROWS_PER_WG];
	if (lane == 0) {
		#pragma unroll
		for (int r = 0; r < ROWS_PER_WG; ++r) warp_sums[wid][r] = acc[r];
	}
	__syncthreads();

	if (wid != 0 || lane != 0) return;

	float final_acc[ROWS_PER_WG];
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) {
		float s = 0.0f;
		#pragma unroll
		for (int w_idx = 0; w_idx < NWARPS; ++w_idx) s += warp_sums[w_idx][r];
		final_acc[r] = s;
	}

	int n_lo = n_base;
	if (n_lo >= N) return;
	int n_hi = n_lo + 1;

	unsigned int lo = bf16_from_f32_q6k(final_acc[0]);
	unsigned int hi = (n_hi < N) ? bf16_from_f32_q6k(final_acc[1]) : 0u;
	y[n_lo >> 1] = (hi << 16) | lo;
}
