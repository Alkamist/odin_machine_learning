#include "broadcast.cuh"
// Fused y = gelu(a) * b for fp32 inputs and output.

#define GELU_SCALE 0.7978845608028654f

__device__ __forceinline__ float gelu_tanh(float v) {
	float cube = 0.044715f * v * v * v;
	return 0.5f * v * (1.0f + tanhf(GELU_SCALE * (v + cube)));
}

extern "C" __global__
void gelu_mul_f32(const float* __restrict__ a,
                  const float* __restrict__ b,
                  float*       __restrict__ c,
                  int n, int n_b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	int j = bc_b_index(i, n_b);
	c[i] = gelu_tanh(a[i]) * b[j];
}
