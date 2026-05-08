// Bf16 X[K] -> Q8_1 blocks for the Q4_K mmvq path. Per-block layout matches
// llama.cpp's `block_q8_1` (36 bytes / 32 weights / 9 uints):
//   uint 0       packHalf2x16(d, s) where d = amax/127, s = d*sum(qs[i])
//   uint 1..8    qs[32] as packed-4x8 signed int8 (4 quants per uint)
// One workgroup per Q8_1 block; 32 threads/block.
#include <cuda_fp16.h>

extern "C" __global__
void quantize_q8_1_bf16(const unsigned int* __restrict__ x_packed,
                        unsigned int*       __restrict__ y,
                        int K) {
	int block_idx = blockIdx.x;
	int tid       = threadIdx.x;

	int k = block_idx * 32 + tid;
	unsigned int pkx = x_packed[k >> 1];
	unsigned short half_bits = (k & 1) ? (unsigned short)(pkx >> 16) : (unsigned short)(pkx & 0xffff);
	unsigned int as_u32 = (unsigned int)half_bits << 16;
	float xv = __int_as_float((int)as_u32);

	// Warp-wide amax via shuffle butterfly.
	float amax = fabsf(xv);
	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
	}

	float d     = amax * (1.0f / 127.0f);
	float d_inv = (d != 0.0f) ? (1.0f / d) : 0.0f;
	int   q     = __float2int_rn(xv * d_inv);

	int sum_q = q;
	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		sum_q += __shfl_xor_sync(0xffffffffu, sum_q, off);
	}

	__shared__ int s_qs[32];
	s_qs[tid] = q;
	__syncthreads();

	if (tid < 8) {
		int base = tid * 4;
		int q0 = s_qs[base + 0] & 0xff;
		int q1 = s_qs[base + 1] & 0xff;
		int q2 = s_qs[base + 2] & 0xff;
		int q3 = s_qs[base + 3] & 0xff;
		unsigned int packed =
			(unsigned int)q0
			| ((unsigned int)q1 <<  8)
			| ((unsigned int)q2 << 16)
			| ((unsigned int)q3 << 24);
		y[block_idx * 9 + 1 + tid] = packed;
	}

	if (tid == 0) {
		float s = d * (float)sum_q;
		unsigned short d_h = __half_as_ushort(__float2half(d));
		unsigned short s_h = __half_as_ushort(__float2half(s));
		y[block_idx * 9] = (unsigned int)d_h | ((unsigned int)s_h << 16);
	}
}
