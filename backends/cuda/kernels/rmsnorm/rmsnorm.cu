// F32 rmsnorm with bf16 weight. One block per row, 256-wide butterfly
// reduction. Activations are fp32 (the post-ggml-shape pipeline default);
// weights are bf16 packed-pairs (the model's stored dtype).
#include <cuda_bf16.h>

#define RMS_WG     256
#define RMS_NWARPS (RMS_WG / 32)

extern "C" __global__
void rmsnorm_f32(const float*        __restrict__ x,
                 const unsigned int* __restrict__ w,
                 float*              __restrict__ y,
                 int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int pair_count = size >> 1;
	int row_base   = row * size;

	float s2 = 0.0f;
	for (int pi = tid; pi < pair_count; pi += RMS_WG) {
		float v0 = x[row_base + 2*pi + 0];
		float v1 = x[row_base + 2*pi + 1];
		s2 += v0 * v0 + v1 * v1;
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		s2 += __shfl_xor_sync(0xffffffffu, s2, off);
	}

	__shared__ float warp_sums[RMS_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s2;
	__syncthreads();

	float total = 0.0f;
	#pragma unroll
	for (int i = 0; i < RMS_NWARPS; ++i) total += warp_sums[i];

	float rstd = rsqrtf(total / (float)size + eps);

	for (int pi = tid; pi < pair_count; pi += RMS_WG) {
		float v0 = x[row_base + 2*pi + 0];
		float v1 = x[row_base + 2*pi + 1];
		unsigned int wp = w[pi];
		float w0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wp & 0xffffu)));
		float w1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((wp >> 16) & 0xffffu)));
		y[row_base + 2*pi + 0] = v0 * rstd * w0;
		y[row_base + 2*pi + 1] = v1 * rstd * w1;
	}
}
