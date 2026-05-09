// Faithful port of ggml's `mul_mat_vec_q` specialized for:
//   type = Q6_K, ncols_dst = 1, rows_per_cuda_block = 1, no fusion, no MoE.
//
// Reference: ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q template) and
// ggml/src/ggml-cuda/vecdotq.cuh (vec_dot_q6_K_q8_1, vec_dot_q6_K_q8_1_impl_mmvq).
//
// Layouts match ggml bit-for-bit:
//   block_q6_K (210 bytes / 256 elements):
//     bytes   0..127   ql (low 4 bits of each 6-bit quant)
//     bytes 128..191   qh (upper 2 bits)
//     bytes 192..207   scales (16 i8 sub-block scales)
//     bytes 208..209   d (fp16 super-block scale)
//   block_q8_1 (36 bytes / 32 weights / 9 uints): same as Q4_K mmvq.
//
// Output: float per row.
#include <cuda_fp16.h>

#define QK6_K            256
#define BQ6_K_BYTES      210
#define QI6_K            32   // ints per Q6_K block (block byte size / 4 in ggml's layout)
#define QR6_K            2    // QK6_K / (4*QI6_K) inverse: 2 q6_K halves per Q8_1 block
#define VDR_Q6_K_Q8_1    1

#define QK8_1            32
#define QI8_1            8
#define BQ8_1_UINTS      9

#define NWARPS           4
#define WARP_SIZE        32
#define ROWS_PER_BLOCK   1
// blocks_per_iter = vdr * nwarps * warp_size / qi  =  1 * 4 * 32 / 32  = 4
#define BLOCKS_PER_ITER  (VDR_Q6_K_Q8_1 * NWARPS * WARP_SIZE / QI6_K)

// 2-byte-aligned 32-bit read. Block boundaries land on a 2-byte (not 4-byte)
// boundary because BQ6_K_BYTES = 210, so reads of ql/qh straddling block i>=1
// must use 16-bit loads.
__device__ __forceinline__ int q6k_get_int_b2(const unsigned int* w, int byte_offset) {
	const unsigned short* w16 = (const unsigned short*)w;
	int idx = byte_offset >> 1;
	unsigned int lo = (unsigned int)w16[idx + 0];
	unsigned int hi = (unsigned int)w16[idx + 1];
	return (int)(lo | (hi << 16));
}

__device__ __forceinline__ int q6k_load_i8(const unsigned int* w, int byte_offset) {
	const signed char* w8 = (const signed char*)w;
	return (int)w8[byte_offset];
}

// vec_dot_q6_K_q8_1_impl_mmvq from ggml/vecdotq.cuh, transcribed.
__device__ __forceinline__ float vec_dot_q6_K_q8_1_impl(
    int vl, int vh, const int* __restrict__ u, const signed char* __restrict__ scales,
    float d, const float* __restrict__ d8) {

	float sumf = 0.0f;

	#pragma unroll
	for (int i = 0; i < QR6_K; ++i) {
		const int sc = scales[4 * i];

		const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
		const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
		const int vi  = __vsubss4((vil | vih), 0x20202020);  // vi = (vil | vih) - 32

		sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
	}

	return d * sumf;
}

// vec_dot_q6_K_q8_1 from ggml/vecdotq.cuh, transcribed for our raw-uint layout.
__device__ __forceinline__ float vec_dot_q6_K_q8_1_call(
    const unsigned int* __restrict__ vx, const unsigned int* __restrict__ vy_kby,
    int kbx, int iqs) {

	// Q6_K block byte base.
	const int block_off = kbx * BQ6_K_BYTES;

	// vl: 32 bits of ql at iqs * 4 bytes. ql lives at bytes [0..127].
	const int vl = q6k_get_int_b2(vx, block_off + iqs * 4);

	// vh: 32 bits of qh at byte (QI6_K/4)*(iqs/(QI6_K/2)) + iqs%(QI6_K/4).
	// qh lives at bytes [128..191]. vh_shift packs the right 2-bit pair.
	const int vh_shift   = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
	const int vh_byte    = 128 + 4 * ((QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4));
	const int vh         = q6k_get_int_b2(vx, block_off + vh_byte) >> vh_shift;

	// scales: 16 int8s at bytes [192..207]. Per-call we touch the 2 entries
	// at scale_offset and scale_offset+4 (read by 4*i in impl).
	const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
	const signed char* scales_base = (const signed char*)vx + block_off + 192 + scale_offset;

	// d: fp16 super-block scale at bytes [208..209].
	const unsigned short* d_ptr = (const unsigned short*)((const char*)vx + block_off + 208);
	const float d = __half2float(__ushort_as_half(*d_ptr));

	// Q8_1 reads: 2 sub-blocks per call. bq8_offset selects which two of the
	// 8 Q8_1 blocks per Q6_K block. iqs%8 = which i32 within their qs[8].
	const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);

	int   u[QR6_K];
	float d8[QR6_K];
	#pragma unroll
	for (int i = 0; i < QR6_K; ++i) {
		const unsigned int* bq8i = vy_kby + (bq8_offset + 2 * i) * BQ8_1_UINTS;
		unsigned int ds = __ldg(bq8i + 0);
		d8[i] = __half2float(__ushort_as_half((unsigned short)(ds & 0xffffu)));

		const unsigned int* q8 = bq8i + 1 + (iqs % QI8_1);
		u[i] = (int)__ldg(q8);
	}

	return vec_dot_q6_K_q8_1_impl(vl, vh, u, scales_base, d, d8);
}

extern "C" __global__
__launch_bounds__(NWARPS * WARP_SIZE, 1)
void linear_q6_k_mmvq(
    const unsigned int* __restrict__ vy,    // q8_1 input row (one Q8_1 stream)
    const unsigned int* __restrict__ vx,    // q6_k weights, row-major [N, K]
    float*              __restrict__ dst,    // fp32 output [N]
    int M, int K, int N) {

	const int tid  = WARP_SIZE * threadIdx.y + threadIdx.x;
	const int row0 = ROWS_PER_BLOCK * blockIdx.x;
	const int blocks_per_row = K / QK6_K;

	const unsigned int* vy_row = vy;

	// Each Q6_K weight row is `blocks_per_row * BQ6_K_BYTES` bytes.
	// Treat as `unsigned int*` for `q6k_get_int_b2`'s 2-byte-aligned reads.
	const unsigned int* vx_row = (const unsigned int*)((const char*)vx + (size_t)row0 * blocks_per_row * BQ6_K_BYTES);

	float tmp = 0.0f;

	for (int kbx = tid / (QI6_K / VDR_Q6_K_Q8_1); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
		const int kby = kbx * (QK6_K / QK8_1);            // = kbx * 8
		const int kqs = VDR_Q6_K_Q8_1 * (tid % (QI6_K / VDR_Q6_K_Q8_1));

		const unsigned int* vy_kby = vy_row + kby * BQ8_1_UINTS;
		tmp += vec_dot_q6_K_q8_1_call(vx_row, vy_kby, kbx, kqs);
	}

	// Reduction: warps 1+ stash to shared, warp 0 reduces and writes.
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
