// Bf16 rmsnorm forward. One block per row; size must be even.
#include <cuda_bf16.h>

#define RMS_WG 256

extern "C" __global__
void rmsnorm_bf16(const unsigned int* __restrict__ x,
                  const unsigned int* __restrict__ w,
                  unsigned int*       __restrict__ y,
                  int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int pair_count = size >> 1;
	int pair_base  = (row * size) >> 1;

	float s2 = 0.0f;
	for (int pi = tid; pi < pair_count; pi += RMS_WG) {
		unsigned int xp = x[pair_base + pi];
		float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(xp & 0xffffu)));
		float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((xp >> 16) & 0xffffu)));
		s2 += v0 * v0 + v1 * v1;
	}

	__shared__ float partial[RMS_WG];
	partial[tid] = s2;
	__syncthreads();
	#pragma unroll
	for (int stride = RMS_WG / 2; stride > 0; stride >>= 1) {
		if (tid < stride) partial[tid] += partial[tid + stride];
		__syncthreads();
	}
	float rstd = rsqrtf(partial[0] / (float)size + eps);

	for (int pi = tid; pi < pair_count; pi += RMS_WG) {
		unsigned int xp = x[pair_base + pi];
		unsigned int wp = w[pi];
		float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(xp & 0xffffu))) * rstd
		         * __bfloat162float(__ushort_as_bfloat16((unsigned short)(wp & 0xffffu)));
		float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((xp >> 16) & 0xffffu))) * rstd
		         * __bfloat162float(__ushort_as_bfloat16((unsigned short)((wp >> 16) & 0xffffu)));
		unsigned short lo = __bfloat16_as_ushort(__float2bfloat16(v0));
		unsigned short hi = __bfloat16_as_ushort(__float2bfloat16(v1));
		y[pair_base + pi] = (unsigned int)lo | ((unsigned int)hi << 16);
	}
}
