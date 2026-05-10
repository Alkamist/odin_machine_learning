// Dequantize Q6_K weights to bf16. Mirrors loaders/gguf/quants.odin
// (`dequantize_q6_k`) and ggml's `dequantize_row_q6_K`.
//
// Block layout (210 bytes for 256 elements):
//   bytes   0-127 : ql  (low 4 bits of each 6-bit quant)
//   bytes 128-191 : qh  (upper 2 bits of each 6-bit quant, packed 4-per-byte)
//   bytes 192-207 : 16 signed i8 scales
//   bytes 208-209 : d (f16 super-scale)
//
// One block per super-block, 256 threads, one element per thread.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define Q6_K_BLOCK_BYTES 210
#define QK_K             256

extern "C" __global__
void dequantize_q6_k_to_bf16(const unsigned char* __restrict__ src,
                              unsigned int*         __restrict__ dst_packed,
                              int total_elements) {
	int block = blockIdx.x;
	int tid   = threadIdx.x;

	int block_in_offset  = block * Q6_K_BLOCK_BYTES;
	int block_out_offset = block * QK_K;

	__shared__ unsigned char shared_block[Q6_K_BLOCK_BYTES];
	if (tid < Q6_K_BLOCK_BYTES) {
		shared_block[tid] = src[block_in_offset + tid];
	}
	__syncthreads();

	if (block_out_offset + tid >= total_elements) return;

	const unsigned char* ql  = shared_block + 0;
	const unsigned char* qh  = shared_block + 128;
	const signed char*   sc  = (const signed char*)(shared_block + 192);

	__half d_h;
	*((unsigned short*)&d_h) = ((unsigned short*)(shared_block + 208))[0];
	float d = __half2float(d_h);

	int half_block  = tid / 128;
	int within      = tid % 128;
	int which_quad  = within / 32;
	int l           = within % 32;
	int is          = l / 16;

	const unsigned char* ql_h = ql + half_block * 64;
	const unsigned char* qh_h = qh + half_block * 32;
	const signed char*   sc_h = sc + half_block * 8;

	int q;
	int sc_offset;
	switch (which_quad) {
	case 0:
		q         = (int)((ql_h[l]      & 0x0Fu) | (((qh_h[l] >> 0) & 0x3u) << 4)) - 32;
		sc_offset = is + 0;
		break;
	case 1:
		q         = (int)((ql_h[l + 32] & 0x0Fu) | (((qh_h[l] >> 2) & 0x3u) << 4)) - 32;
		sc_offset = is + 2;
		break;
	case 2:
		q         = (int)((ql_h[l]      >> 4)    | (((qh_h[l] >> 4) & 0x3u) << 4)) - 32;
		sc_offset = is + 4;
		break;
	default: // 3
		q         = (int)((ql_h[l + 32] >> 4)    | (((qh_h[l] >> 6) & 0x3u) << 4)) - 32;
		sc_offset = is + 6;
		break;
	}

	float val = d * (float)sc_h[sc_offset] * (float)q;

	__shared__ unsigned short out_halves[QK_K];
	out_halves[tid] = __bfloat16_as_ushort(__float2bfloat16(val));
	__syncthreads();

	if ((tid & 1) == 0) {
		int pair_idx = (block_out_offset + tid) >> 1;
		dst_packed[pair_idx] = (unsigned int)out_halves[tid] | ((unsigned int)out_halves[tid + 1] << 16);
	}
}
