// Faithful port of ggml's `mul_mat_vec_q` specialized for:
//   type = Q4_K, ncols_dst = 1, rows_per_cuda_block = 1, no fusion, no MoE.
//
// Reference: ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q template) and
// ggml/src/ggml-cuda/vecdotq.cuh (vec_dot_q4_K_q8_1, vec_dot_q4_K_q8_1_impl_vmmq).
//
// Layouts match ggml bit-for-bit:
//   block_q4_K (144 bytes / 36 uints):
//     uint  0      : d (low 16) | dmin (high 16)            — half2 packed
//     uints 1..3   : scales[12 bytes] (6 packed uint16s)
//     uints 4..35  : qs[128 bytes = 32 ints]
//   block_q8_1 (36 bytes / 9 uints):
//     uint  0      : d (low 16) | s (high 16)               — half2 packed
//     uints 1..8   : qs[32 bytes = 8 ints]
//
// Output: float per row. The downstream `pack_f32_to_bf16_pairs` kernel
// converts to the bf16-packed-pair layout the rest of the pipeline expects.
// We don't pack inline because doing so forced ROWS_PER_WG=2 in the previous
// version, which deviates from ggml's empirically-chosen rows_per_cuda_block=1
// for ncols_dst=1 on Ampere.
#include <cuda_fp16.h>

#define QK4_K            256
#define QI4_K_QS         32   // ints in qs section of block_q4_K
#define BQ4_K_UINTS      36   // total uints in block_q4_K (header + qs)
#define QK8_1            32
#define QI8_1            8    // ints in qs section of block_q8_1
#define BQ8_1_UINTS      9
#define QR4_K            2
#define VDR_Q4_K_Q8_1    2

#define NWARPS           4
#define WARP_SIZE        32
#define ROWS_PER_BLOCK   1
#define BLOCKS_PER_ITER  (VDR_Q4_K_Q8_1 * NWARPS * WARP_SIZE / QI4_K_QS)  // = 8

// vec_dot_q4_K_q8_1_impl_vmmq from ggml/vecdotq.cuh, transcribed.
__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl(
    const int * __restrict__ v, const int * __restrict__ u,
    const unsigned char * __restrict__ sc, const unsigned char * __restrict__ m,
    unsigned int dm_packed, const float * __restrict__ d8) {

	float sumf_d = 0.0f;
	float sumf_m = 0.0f;

	#pragma unroll
	for (int i = 0; i < QR4_K; ++i) {
		const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
		const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

		const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
		const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));

		sumf_d += d8[i] * (dot1 * sc[i]);
		sumf_m += d8[i] * (dot2 * m[i]);
	}

	float d    = __half2float(__ushort_as_half((unsigned short)(dm_packed & 0xffffu)));
	float dmin = __half2float(__ushort_as_half((unsigned short)((dm_packed >> 16) & 0xffffu)));
	return d * sumf_d - dmin * sumf_m;
}

// vec_dot_q4_K_q8_1 from ggml/vecdotq.cuh, transcribed for our raw-uint layout.
__device__ __forceinline__ float vec_dot_q4_K_q8_1_call(
    const unsigned int * __restrict__ vx_row, const unsigned int * __restrict__ vy_kby,
    int kbx, int iqs) {

	const unsigned int * bq4_K = vx_row + kbx * BQ4_K_UINTS;

	int   v[2];
	int   u[2 * QR4_K];
	float d8[QR4_K];

	const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));   // 0, 2, 4, or 6
	const int idx        = (iqs / 2) % 4;                         // 0..3

	const unsigned int * q4 = bq4_K + 4 + 4 * bq8_offset + idx;
	v[0] = (int)__ldg(q4 + 0);
	v[1] = (int)__ldg(q4 + 4);

	// Scales/mins from bq4_K->scales (12 bytes = uints 1..3, 6 uint16s).
	const unsigned short * scales = (const unsigned short *)(bq4_K + 1);
	unsigned short aux[2];
	const int j = bq8_offset / 2;
	if (j < 2) {
		aux[0] = scales[j + 0] & 0x3f3f;
		aux[1] = scales[j + 2] & 0x3f3f;
	} else {
		aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
		aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
	}
	const unsigned char * sc = (const unsigned char *)aux;
	const unsigned char * m  = sc + 2;

	#pragma unroll
	for (int i = 0; i < QR4_K; ++i) {
		const unsigned int * bq8i = vy_kby + (bq8_offset + i) * BQ8_1_UINTS;
		unsigned int ds = __ldg(bq8i + 0);
		d8[i] = __half2float(__ushort_as_half((unsigned short)(ds & 0xffffu)));

		const unsigned int * q8 = bq8i + 1 + idx;
		u[2 * i + 0] = (int)__ldg(q8 + 0);
		u[2 * i + 1] = (int)__ldg(q8 + 4);
	}

	return vec_dot_q4_K_q8_1_impl(v, u, sc, m, __ldg(bq4_K + 0), d8);
}

extern "C" __global__
__launch_bounds__(NWARPS * WARP_SIZE, 1)
void linear_q4_k_mmvq(
    const unsigned int * __restrict__ vy,    // q8_1 input row (one Q8_1 stream)
    const unsigned int * __restrict__ vx,    // q4_k weights, row-major [N, K]
    float *              __restrict__ dst,    // fp32 output [N]
    int M, int K, int N) {

	const int tid  = WARP_SIZE * threadIdx.y + threadIdx.x;
	const int row0 = ROWS_PER_BLOCK * blockIdx.x;
	const int blocks_per_row = K / QK4_K;

	// Q8_1 input is one row per call. ggml's outer kernel offsets `y` by sample
	// /channel for batched MoE; we don't have any of that, so vy is the row.
	const unsigned int * vy_row = vy;

	float tmp = 0.0f;

	for (int kbx = tid / (QI4_K_QS / VDR_Q4_K_Q8_1); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
		const int kby = kbx * (QK4_K / QK8_1);  // = kbx * 8
		const int kqs = VDR_Q4_K_Q8_1 * (tid % (QI4_K_QS / VDR_Q4_K_Q8_1));

		const unsigned int * vx_row = vx + (size_t)row0 * blocks_per_row * BQ4_K_UINTS;
		const unsigned int * vy_kby = vy_row + kby * BQ8_1_UINTS;
		tmp += vec_dot_q4_K_q8_1_call(vx_row, vy_kby, kbx, kqs);
	}

	// ggml's reduction: warps 1+ stash partial sums to shared, then warp 0
	// pulls them in and warp-reduces. With ROWS_PER_BLOCK=1 the indexing
	// collapses to `tmp_shared[wid-1][lane]`.
	__shared__ float tmp_shared[NWARPS - 1][WARP_SIZE];
	if (threadIdx.y > 0) {
		tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
	}
	__syncthreads();
	if (threadIdx.y > 0) return;

	#pragma unroll
	for (int l = 0; l < NWARPS - 1; ++l) {
		tmp += tmp_shared[l][threadIdx.x];
	}
	#pragma unroll
	for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
		tmp += __shfl_xor_sync(0xffffffffu, tmp, off);
	}

	if (threadIdx.x == 0 && row0 < N) {
		dst[row0] = tmp;
	}
}
