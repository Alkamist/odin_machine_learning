// F32 rmsnorm with bf16 weight. One block per row, 256-wide strided reduction.
// Activations are fp32 (the post-ggml-shape pipeline default); weights are
// bf16 packed-pairs (the model's stored dtype).
#include <cuda_bf16.h>

#define RMS_WG 256

extern "C" __global__
void rmsnorm_f32(const float*        __restrict__ x,
                 const unsigned int* __restrict__ w,
                 float*              __restrict__ y,
                 int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;
	int base = row * size;

	float s2 = 0.0f;
	for (int i = tid; i < size; i += RMS_WG) {
		float v = x[base + i];
		s2 += v * v;
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

	int pair_count = size >> 1;
	for (int pi = tid; pi < pair_count; pi += RMS_WG) {
		unsigned int wp = w[pi];
		float w0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wp & 0xffffu)));
		float w1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((wp >> 16) & 0xffffu)));
		y[base + 2*pi + 0] = x[base + 2*pi + 0] * rstd * w0;
		y[base + 2*pi + 1] = x[base + 2*pi + 1] * rstd * w1;
	}
}
