// Q4_K weight x Q8_1 activation GEMV (M=1 decode path). Mirrors the inner
// per-thread work pattern of ggml's `mul_mat_vec_q` / `vec_dot_q4_K_q8_1`:
//
//   block_q4_K (144 bytes / 256 elements):
//     half  d, dmin                     (4 bytes)
//     u8    scales_mins[12]             (12 bytes packed 6-bit scales+mins)
//     u8    qs[128]                     (32 ints, low+high nibbles per byte)
//   block_q8_1 (36 bytes / 32 elements):
//     half  d, s                        (s == d * sum(qs[i]))
//     i8    qs[32]                      (8 ints, packed 4-per-uint)
//
// Block layout: 4 warps x 32 lanes = 128 threads. Each block produces
// ROWS_PER_WG=2 output rows. The 128 threads are striped across K-blocks
// at granularity `blocks_per_iter = 2 * NWARPS * WARP_SIZE / QI = 8`,
// so 16 threads cover each K-block per outer iter (2 K-blocks per warp
// per iter, vs the prior 1 K-block per warp). Each thread handles 16
// weights via 4 dp4a calls — two q4 ints loaded once, expanded into
// low+high nibble passes against four q8_1 ints. This matches ggml's
// register-tiled mmvq body.
//
// q/k/v across the same `hidden` activation share the q8_1 buffer via
// the q8_1 cache in ops.odin; downstream consumers read packed bf16
// pairs (`y[n_lo >> 1] = (hi << 16) | lo`).
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

#define ROWS_PER_WG       2
#define NWARPS            4
#define WARP_SIZE         32
#define VDR               2     // dp4a pairs per thread per K-block (== ggml's VDR_Q4_K_Q8_1_MMVQ)
#define QI                32    // ints in a Q4_K block (qs section)
#define BLOCK_UINTS       36    // total uints per Q4_K block (header + qs)
#define THREADS_PER_BLOCK (NWARPS * WARP_SIZE)
#define BLOCKS_PER_ITER   (VDR * NWARPS * WARP_SIZE / QI)  // = 8

extern "C" __global__
__launch_bounds__(THREADS_PER_BLOCK, 1)
void linear_q4_k_mmvq(const unsigned int* __restrict__ x,   // q8_1 stream
                      const unsigned int* __restrict__ w,   // q4_k stream
                      unsigned int*       __restrict__ y,   // bf16 packed pairs
                      int M, int K, int N) {
	int n_base = blockIdx.x * ROWS_PER_WG;
	int wid    = threadIdx.y;
	int lane   = threadIdx.x;
	int tid    = wid * WARP_SIZE + lane;

	int blocks_per_row = K / 256;

	// Each thread covers 16 weights of one (K-block, sub-pair) combo. Threads
	// are split into groups of QI/VDR=16: lanes within a group share the same
	// kbx_start and walk the K dimension via blocks_per_iter strides.
	int kqs               = VDR * (tid % (QI / VDR));    // 0,2,...,30
	int bq8_offset        = 2 * ((kqs / 2) / 4);          // 0, 2, 4, or 6 — selects sub-block pair
	int idx_in_subblock   = (kqs / 2) % 4;                 // 0..3 — which int inside the sub-block

	float acc[ROWS_PER_WG];
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) acc[r] = 0.0f;

	for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
		// Load q8_1 once for this K-block — shared across both output rows.
		int q8_base = (kbx * 8 + bq8_offset) * 9;
		unsigned int ds_0 = __ldg(&x[q8_base]);
		unsigned int ds_1 = __ldg(&x[q8_base + 9]);
		float d8_0  = __half2float(__ushort_as_half((unsigned short)(ds_0 & 0xffffu)));
		float d8_1  = __half2float(__ushort_as_half((unsigned short)(ds_1 & 0xffffu)));
		float q8s_0 = __half2float(__ushort_as_half((unsigned short)((ds_0 >> 16) & 0xffffu)));
		float q8s_1 = __half2float(__ushort_as_half((unsigned short)((ds_1 >> 16) & 0xffffu)));

		int u[4];
		u[0] = (int)__ldg(&x[q8_base + 1 + idx_in_subblock]);
		u[1] = (int)__ldg(&x[q8_base + 1 + idx_in_subblock + 4]);
		u[2] = (int)__ldg(&x[q8_base + 9 + 1 + idx_in_subblock]);
		u[3] = (int)__ldg(&x[q8_base + 9 + 1 + idx_in_subblock + 4]);

		#pragma unroll
		for (int r = 0; r < ROWS_PER_WG; ++r) {
			int n = n_base + r;
			if (n >= N) continue;

			int block_off = (n * blocks_per_row + kbx) * BLOCK_UINTS;

			uint4 hdr = __ldg(reinterpret_cast<const uint4*>(&w[block_off]));
			unsigned int dm = hdr.x;
			float d    = __half2float(__ushort_as_half((unsigned short)(dm & 0xffffu)));
			float dmin = __half2float(__ushort_as_half((unsigned short)((dm >> 16) & 0xffffu)));
			unsigned int sw0 = hdr.y;
			unsigned int sw1 = hdr.z;
			unsigned int sw2 = hdr.w;

			int sc0, m0, sc1, m1;
			unpack_scale_min(bq8_offset,     sw0, sw1, sw2, sc0, m0);
			unpack_scale_min(bq8_offset + 1, sw0, sw1, sw2, sc1, m1);

			// Load v[0], v[1] — each holds 8 nibbles (= 8 weights) when expanded
			// across the i={0,1} low/high nibble pass.
			int v[2];
			v[0] = (int)__ldg(&w[block_off + 4 + 4 * bq8_offset + idx_in_subblock]);
			v[1] = (int)__ldg(&w[block_off + 4 + 4 * bq8_offset + idx_in_subblock + 4]);

			// 4 dp4a calls per K-block per row, covering 16 weights.
			float sumf_d = 0.0f;
			#pragma unroll
			for (int i = 0; i < 2; ++i) {
				int v0i  = (v[0] >> (4 * i)) & 0x0F0F0F0Fu;
				int v1i  = (v[1] >> (4 * i)) & 0x0F0F0F0Fu;
				int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
				int sc   = (i == 0) ? sc0 : sc1;
				float d8 = (i == 0) ? d8_0 : d8_1;
				sumf_d += d8 * (float)sc * (float)dot1;
			}

			// Min term uses precomputed q8_1 sums (q8s = d * sum(qs)). Only one
			// thread per (sub-block-pair, K-block) credits the contribution to
			// avoid quadruple-counting across the 4 threads sharing bq8_offset.
			float min_contrib = 0.0f;
			if (idx_in_subblock == 0) {
				min_contrib = q8s_0 * (float)m0 + q8s_1 * (float)m1;
			}

			acc[r] += d * sumf_d - dmin * min_contrib;
		}
	}

	// Intra-warp reduction.
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
