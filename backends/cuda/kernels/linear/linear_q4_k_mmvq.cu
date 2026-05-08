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
//
// Block layout: 4 warps x 32 lanes = 128 threads. Each block produces
// ROWS_PER_WG=2 output rows; the 4 warps cooperate on K (each warp handles
// 1/NWARPS of the K-blocks), then a tree reduction folds across warps.
// Versus the previous 1-warp/2-row layout this 4xs SM warp count and lifts
// the per-row K parallelism from 32 lanes to 128.
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
#define NWARPS      4
#define WARP_SIZE   32
#define BLOCK_UINTS 36   // 144 / 4

extern "C" __global__
__launch_bounds__(NWARPS * WARP_SIZE, 1)
void linear_q4_k_mmvq(const unsigned int* __restrict__ x,   // q8_1 stream
                      const unsigned int* __restrict__ w,   // q4_k stream
                      unsigned int*       __restrict__ y,   // bf16 packed pairs
                      int M, int K, int N) {
	int n_base = blockIdx.x * ROWS_PER_WG;
	int wid    = threadIdx.y;        // 0..NWARPS-1
	int lane   = threadIdx.x;        // 0..31

	int num_blocks = K / 256;

	float acc[ROWS_PER_WG];
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) acc[r] = 0.0f;

	int t_sub_0 = lane >> 3;
	int t_inner = lane & 7;
	int t_sub_1 = t_sub_0 + 4;

	// Each warp strides through K with stride NWARPS, processing 1/NWARPS of
	// the Q4_K blocks. Inside a warp the 32 lanes split the block's 8 sub-blocks
	// x 8 inner ints exactly as in the 1-warp version.
	for (int b = wid; b < num_blocks; b += NWARPS) {
		int q8_block_base = b * 8;

		int          q8_packed_0 = (int)__ldg(&x[(q8_block_base + t_sub_0) * 9 + 1 + t_inner]);
		int          q8_packed_1 = (int)__ldg(&x[(q8_block_base + t_sub_1) * 9 + 1 + t_inner]);
		unsigned int q8_ds_0 = __ldg(&x[(q8_block_base + t_sub_0) * 9]);
		unsigned int q8_ds_1 = __ldg(&x[(q8_block_base + t_sub_1) * 9]);
		float q8_d_0 = __half2float(__ushort_as_half((unsigned short)(q8_ds_0 & 0xffff)));
		float q8_s_0 = __half2float(__ushort_as_half((unsigned short)((q8_ds_0 >> 16) & 0xffff)));
		float q8_d_1 = __half2float(__ushort_as_half((unsigned short)(q8_ds_1 & 0xffff)));
		float q8_s_1 = __half2float(__ushort_as_half((unsigned short)((q8_ds_1 >> 16) & 0xffff)));

		#pragma unroll
		for (int r = 0; r < ROWS_PER_WG; ++r) {
			int n = n_base + r;
			if (n >= N) continue;

			int block_off = (n * num_blocks + b) * BLOCK_UINTS;

			uint4 hdr = __ldg(reinterpret_cast<const uint4*>(&w[block_off]));
			unsigned int dm = hdr.x;
			float d    = __half2float(__ushort_as_half((unsigned short)(dm & 0xffff)));
			float dmin = __half2float(__ushort_as_half((unsigned short)((dm >> 16) & 0xffff)));
			unsigned int sw0 = hdr.y;
			unsigned int sw1 = hdr.z;
			unsigned int sw2 = hdr.w;

			float row_acc = 0.0f;

			{
				unsigned int nib_word = __ldg(&w[block_off + 4 + (t_sub_0 >> 1) * 8 + t_inner]);
				int shift     = (t_sub_0 & 1) * 4;
				int q4_packed = (int)((nib_word >> shift) & 0x0F0F0F0Fu);
				int int_dot   = __dp4a(q4_packed, q8_packed_0, 0);

				int scale, min_v;
				unpack_scale_min(t_sub_0, sw0, sw1, sw2, scale, min_v);

				row_acc += d * (float)scale * q8_d_0 * (float)int_dot;
				if (t_inner == 0) row_acc -= dmin * (float)min_v * q8_s_0;
			}
			{
				unsigned int nib_word = __ldg(&w[block_off + 4 + (t_sub_1 >> 1) * 8 + t_inner]);
				int shift     = (t_sub_1 & 1) * 4;
				int q4_packed = (int)((nib_word >> shift) & 0x0F0F0F0Fu);
				int int_dot   = __dp4a(q4_packed, q8_packed_1, 0);

				int scale, min_v;
				unpack_scale_min(t_sub_1, sw0, sw1, sw2, scale, min_v);

				row_acc += d * (float)scale * q8_d_1 * (float)int_dot;
				if (t_inner == 0) row_acc -= dmin * (float)min_v * q8_s_1;
			}

			acc[r] += row_acc;
		}
	}

	// Warp-level reduction: each warp folds its 32 lanes into lane 0.
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) {
		#pragma unroll
		for (int off = 16; off > 0; off >>= 1) {
			acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
		}
	}

	// Cross-warp reduction via shared memory: each warp's lane-0 contributes
	// its sum, warp 0 finalizes.
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
		for (int w = 0; w < NWARPS; ++w) s += warp_sums[w][r];
		final_acc[r] = s;
	}

	int n_lo = n_base;
	if (n_lo >= N) return;
	int n_hi = n_lo + 1;

	unsigned int lo = bf16_from_f32(final_acc[0]);
	unsigned int hi = (n_hi < N) ? bf16_from_f32(final_acc[1]) : 0u;
	y[n_lo >> 1] = (hi << 16) | lo;
}
