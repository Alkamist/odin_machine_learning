// Bf16 rmsnorm forward. One block per row; size must be even.
#include <cuda_bf16.h>

#define RMS_WG    256
#define RMS_NWARPS (RMS_WG / 32)

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

	// Intra-warp reduction: collapse 32 lanes into lane 0 via butterfly shuffles.
	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		s2 += __shfl_xor_sync(0xffffffffu, s2, off);
	}

	// Inter-warp: lane 0 of each warp publishes; one syncthreads; then every
	// thread sums the NWARPS-element table locally so the broadcast falls out
	// of HW shared-mem fanout (no second sync needed for the result read).
	__shared__ float warp_sums[RMS_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s2;
	__syncthreads();

	float total = 0.0f;
	#pragma unroll
	for (int i = 0; i < RMS_NWARPS; ++i) total += warp_sums[i];

	float rstd = rsqrtf(total / (float)size + eps);

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
