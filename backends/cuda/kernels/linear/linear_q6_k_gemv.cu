// Q6_K x bf16 GEMV (M=1 decode path). Outer structure mirrors ggml's
// `mul_mat_vec_q` (rows_per_cuda_block=1, fp32 output, blocks_per_iter
// striding); inner dot product still reads bf16 input directly rather than
// going through the q8_1 + dp4a pipeline that ggml uses for Q6_K. Keeping
// bf16 here avoids adding a quantize_q8_1 dispatch in front and the q8_1
// cache that goes with it; revisit if Q6_K starts dominating GPU time.
//
//   block_q6_K (210 bytes / 256 elements):
//     ql      bytes   0..127   (low 4 bits of each 6-bit quant)
//     qh      bytes 128..191   (upper 2 bits)
//     scales  bytes 192..207   (16 i8 sub-block scales, one per 16 weights)
//     d       bytes 208..209   (fp16 super-block scale)
//
// Per-element dequant: w(e) = d * scales[e/16] * (ql_nibble | qh_2bits<<4 - 32).
//
// Block layout: 4 warps × 32 lanes = 128 threads, one block per output row.
// Output is fp32; downstream pack_f32_to_bf16_pairs converts to packed bf16.
#include <cuda_fp16.h>

#define QK6_K            256
#define BLOCK_BYTES      210

#define NWARPS           4
#define WARP_SIZE        32
#define ROWS_PER_BLOCK   1
// blocks_per_iter for Q6_K: vdr * nwarps * warp_size / qi.
// Q6_K has qi = 32, vdr = 1, so = 4.
#define BLOCKS_PER_ITER  4

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

extern "C" __global__
__launch_bounds__(NWARPS * WARP_SIZE, 1)
void linear_q6_k_gemv(const unsigned int* __restrict__ x,   // bf16 packed pairs
                      const unsigned int* __restrict__ w,   // q6_k byte stream
                      float*              __restrict__ dst,  // fp32 output [N]
                      int M, int K, int N) {

	const int tid  = WARP_SIZE * threadIdx.y + threadIdx.x;
	const int row0 = ROWS_PER_BLOCK * blockIdx.x;
	const int blocks_per_row = K / QK6_K;

	float tmp = 0.0f;

	// Each warp covers a stride of K-blocks; 32 lanes within a warp cooperate
	// across the K-block's sub-blocks. With blocks_per_iter=4, four K-blocks
	// are processed per outer iter (one per warp).
	for (int kbx = threadIdx.y; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
		// Per-block, the 32 lanes process the 256 weights as 8 sub-blocks
		// (h × 2 halves) × 4 quadrants × 32 lanes/quadrant. Lane indexes the
		// quadrant via lane%32; halves indexed by `h` loop; quadrants
		// distributed across lanes via lane>>4.

		const int block_off = (row0 * blocks_per_row + kbx) * BLOCK_BYTES;

		const unsigned int d_bits = q6k_load_byte(w, block_off + 208) | (q6k_load_byte(w, block_off + 209) << 8);
		const float d = __half2float(__ushort_as_half((unsigned short)d_bits));

		// Hoist the bf16 input chunk for this K-block once.
		float xv[2][4];
		const int lane = threadIdx.x;
		#pragma unroll
		for (int h = 0; h < 2; ++h) {
			#pragma unroll
			for (int quadrant = 0; quadrant < 4; ++quadrant) {
				const int e = h * 128 + quadrant * 32 + lane;
				const int k = kbx * QK6_K + e;
				const unsigned int pkx = __ldg(&x[k >> 1]);
				const unsigned short half_bits = (k & 1) ? (unsigned short)(pkx >> 16)
				                                         : (unsigned short)(pkx & 0xffff);
				const unsigned int as_u32 = (unsigned int)half_bits << 16;
				xv[h][quadrant] = __int_as_float((int)as_u32);
			}
		}

		#pragma unroll
		for (int h = 0; h < 2; ++h) {
			const unsigned int ql_lo = q6k_load_byte(w, block_off + h * 64 +      lane);
			const unsigned int ql_hi = q6k_load_byte(w, block_off + h * 64 + 32 + lane);
			const unsigned int qh    = q6k_load_byte(w, block_off + 128 + h * 32 + lane);

			const int sb_base = h * 8 + (lane >> 4);
			const int sc0 = q6k_load_i8(w, block_off + 192 + sb_base + 0);
			const int sc1 = q6k_load_i8(w, block_off + 192 + sb_base + 2);
			const int sc2 = q6k_load_i8(w, block_off + 192 + sb_base + 4);
			const int sc3 = q6k_load_i8(w, block_off + 192 + sb_base + 6);

			#pragma unroll
			for (int quadrant = 0; quadrant < 4; ++quadrant) {
				const unsigned int ql_byte   = ((quadrant & 1) == 0) ? ql_lo : ql_hi;
				const unsigned int ql_nibble = (ql_byte >> ((quadrant >> 1) * 4)) & 0x0Fu;
				const unsigned int qh_bits   = (qh      >>  (quadrant       * 2)) & 0x03u;
				const int q = (int)(ql_nibble | (qh_bits << 4)) - 32;

				const int scale_i8 = (quadrant == 0) ? sc0
				                   : (quadrant == 1) ? sc1
				                   : (quadrant == 2) ? sc2
				                   :                   sc3;

				tmp += d * (float)scale_i8 * (float)q * xv[h][quadrant];
			}
		}
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
