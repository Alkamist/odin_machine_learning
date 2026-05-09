// Fused fp32 (residual = a + b) followed by rmsnorm-with-bf16-weight on the
// residual. Writes both R = a+b (fp32, used by downstream layer ops) and
// Y = rmsnorm(R) * W (fp32, fed into next matmul's quantize_q8_1).

#include <cuda_bf16.h>

#define ARN_WG     256
#define ARN_NWARPS (ARN_WG / 32)

extern "C" __global__
void add_rmsnorm_f32(const float*        __restrict__ a,
                     const float*        __restrict__ b,
                     const unsigned int* __restrict__ w,
                     float*              __restrict__ r,    // residual out (fp32)
                     float*              __restrict__ y,    // normed out (fp32)
                     int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int pair_count = size >> 1;
	int row_base   = row * size;

	float s2 = 0.0f;
	for (int pi = tid; pi < pair_count; pi += ARN_WG) {
		float a0 = a[row_base + 2*pi + 0];
		float a1 = a[row_base + 2*pi + 1];
		float b0 = b[row_base + 2*pi + 0];
		float b1 = b[row_base + 2*pi + 1];
		float r0 = a0 + b0;
		float r1 = a1 + b1;
		r[row_base + 2*pi + 0] = r0;
		r[row_base + 2*pi + 1] = r1;
		s2 += r0 * r0 + r1 * r1;
	}

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
		float r0 = r[row_base + 2*pi + 0];
		float r1 = r[row_base + 2*pi + 1];
		unsigned int wp = w[pi];
		float w0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wp & 0xffffu)));
		float w1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((wp >> 16) & 0xffffu)));
		y[row_base + 2*pi + 0] = r0 * rstd * w0;
		y[row_base + 2*pi + 1] = r1 * rstd * w1;
	}
}
