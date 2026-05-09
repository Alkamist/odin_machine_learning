// Fused bf16 (residual = a + b) followed by rmsnorm-with-weight on the
// residual. Writes both R = a+b (the new residual, used by downstream layer
// ops) and Y = rmsnorm(R) * W. One combined dispatch per row.
#include <cuda_bf16.h>

#define ARN_WG     256
#define ARN_NWARPS (ARN_WG / 32)

extern "C" __global__
void add_rmsnorm_bf16(const unsigned int* __restrict__ a,
                      const unsigned int* __restrict__ b,
                      const unsigned int* __restrict__ w,
                      unsigned int*       __restrict__ r,    // residual out
                      unsigned int*       __restrict__ y,    // normed out
                      int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int pair_count = size >> 1;
	int pair_base  = (row * size) >> 1;

	float s2 = 0.0f;
	for (int pi = tid; pi < pair_count; pi += ARN_WG) {
		unsigned int ap = a[pair_base + pi];
		unsigned int bp = b[pair_base + pi];
		float a0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(ap & 0xffffu)));
		float a1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((ap >> 16) & 0xffffu)));
		float b0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(bp & 0xffffu)));
		float b1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((bp >> 16) & 0xffffu)));
		__nv_bfloat16 r0 = __float2bfloat16(a0 + b0);
		__nv_bfloat16 r1 = __float2bfloat16(a1 + b1);
		unsigned short lo = __bfloat16_as_ushort(r0);
		unsigned short hi = __bfloat16_as_ushort(r1);
		r[pair_base + pi] = (unsigned int)lo | ((unsigned int)hi << 16);
		float v0 = __bfloat162float(r0);
		float v1 = __bfloat162float(r1);
		s2 += v0 * v0 + v1 * v1;
	}

	// Intra-warp reduction.
	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		s2 += __shfl_xor_sync(0xffffffffu, s2, off);
	}

	__shared__ float warp_sums[ARN_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s2;
	__syncthreads();

	float total = 0.0f;
	#pragma unroll
	for (int i = 0; i < ARN_NWARPS; ++i) total += warp_sums[i];

	float rstd = rsqrtf(total / (float)size + eps);

	for (int pi = tid; pi < pair_count; pi += ARN_WG) {
		unsigned int rp = r[pair_base + pi];
		unsigned int wp = w[pi];
		float v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(rp & 0xffffu))) * rstd
		         * __bfloat162float(__ushort_as_bfloat16((unsigned short)(wp & 0xffffu)));
		float v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((rp >> 16) & 0xffffu))) * rstd
		         * __bfloat162float(__ushort_as_bfloat16((unsigned short)((wp >> 16) & 0xffffu)));
		unsigned short lo = __bfloat16_as_ushort(__float2bfloat16(v0));
		unsigned short hi = __bfloat16_as_ushort(__float2bfloat16(v1));
		y[pair_base + pi] = (unsigned int)lo | ((unsigned int)hi << 16);
	}
}
