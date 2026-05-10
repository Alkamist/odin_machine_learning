// Dequantize Q4_K weights to bf16. Mirrors the CPU implementation in
// loaders/gguf/quants.odin (`dequantize_q4_k`) and the upstream
// `dequantize_row_q4_K` in ggml. One block per Q4_K super-block (256
// elements / 144 bytes), 256 threads per block, one element per thread.
//
// Block layout (144 bytes):
//   bytes  0-1   : d    (f16, super-scale)
//   bytes  2-3   : dmin (f16, super-min)
//   bytes  4-15  : 8 packed (6-bit scale, 6-bit min) pairs
//   bytes 16-143 : 128 bytes = 256 4-bit quants
//
// Output: 256 bf16 elements per block.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define Q4_K_BLOCK_BYTES 144
#define QK_K             256

__device__ __forceinline__ void unpack_scale_min_k4(int is, const unsigned char* q, unsigned char* scale, unsigned char* mn) {
	if (is < 4) {
		*scale = q[is]     & 0x3Fu;
		*mn    = q[is + 4] & 0x3Fu;
	} else {
		int j = is;
		*scale = (q[j + 4] & 0x0Fu) | (((q[j - 4] >> 6) & 0x3u) << 4);
		*mn    = (q[j + 4] >> 4)    | (((q[j    ] >> 6) & 0x3u) << 4);
	}
}

extern "C" __global__
void dequantize_q4_k_to_bf16(const unsigned char* __restrict__ src,
                              unsigned int*         __restrict__ dst_packed,
                              int total_elements) {
	int block = blockIdx.x;
	int tid   = threadIdx.x; // 0..255

	int block_in_offset  = block * Q4_K_BLOCK_BYTES;
	int block_out_offset = block * QK_K;

	__shared__ unsigned char shared_block[Q4_K_BLOCK_BYTES];
	if (tid < Q4_K_BLOCK_BYTES) {
		shared_block[tid] = src[block_in_offset + tid];
	}
	__syncthreads();

	if (block_out_offset + tid >= total_elements) return;

	__half d_h, dmin_h;
	*((unsigned short*)&d_h)    = ((unsigned short*)shared_block)[0];
	*((unsigned short*)&dmin_h) = ((unsigned short*)shared_block)[1];
	float d    = __half2float(d_h);
	float dmin = __half2float(dmin_h);

	const unsigned char* scales_packed = shared_block + 4;
	const unsigned char* quants        = shared_block + 16;

	int j_iter        = tid / 64;
	int within        = tid % 64;
	int l             = within % 32;
	int which_half    = within / 32;
	int is            = j_iter * 2 + which_half;
	int quant_byte_idx = j_iter * 32 + l;

	unsigned char scale, mn;
	unpack_scale_min_k4(is, scales_packed, &scale, &mn);

	unsigned char q = quants[quant_byte_idx];
	unsigned char nibble = (which_half == 0) ? (q & 0x0Fu) : (q >> 4);

	float val = d * (float)scale * (float)nibble - dmin * (float)mn;

	// Pack two adjacent threads' bf16 outputs into one 32-bit dst write.
	// Adjacent threads have adjacent global indices and write to the same
	// 32-bit packed dst word; cooperate via shared memory + warp shuffle
	// to avoid CAS. Since 256 threads in this block all write disjoint
	// adjacent pairs in the same 256-element output range, a simple
	// per-thread atomic-free pack works: thread t writes its half into the
	// shared output buffer, then threads with even tid write the packed
	// 32-bit word.
	__shared__ unsigned short out_halves[QK_K];
	out_halves[tid] = __bfloat16_as_ushort(__float2bfloat16(val));
	__syncthreads();

	if ((tid & 1) == 0) {
		int pair_idx = (block_out_offset + tid) >> 1;
		dst_packed[pair_idx] = (unsigned int)out_halves[tid] | ((unsigned int)out_halves[tid + 1] << 16);
	}
}
