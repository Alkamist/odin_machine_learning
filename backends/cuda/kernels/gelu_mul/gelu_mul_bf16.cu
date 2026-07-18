// Fused y = gelu(a) * b for bf16. Eliminates the gelu output round-trip
// through memory in MLP gate paths (e.g. Gemma's `mul(gelu(gate), up)`).
#include <cuda_bf16.h>
#include "broadcast.cuh"

#define GELU_SCALE 0.7978845608028654f

__device__ __forceinline__ float gelu_tanh(float v) {
	float cube = 0.044715f * v * v * v;
	return 0.5f * v * (1.0f + tanhf(GELU_SCALE * (v + cube)));
}

extern "C" __global__
void gelu_mul_bf16(const unsigned int* __restrict__ a,
                   const unsigned int* __restrict__ b,
                   unsigned int*       __restrict__ c,
                   int n, int n_b, int pair_count) {
	int pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= pair_count) return;

	unsigned int ap = a[pair];
	unsigned short a0 = (unsigned short)(ap & 0xffffu);
	unsigned short a1 = (unsigned short)((ap >> 16) & 0xffffu);

	int i0 = 2 * pair;
	int i1 = i0 + 1;
	unsigned short c0 = 0, c1 = 0;

	if (i0 < n) {
		int j = bc_b_index(i0, n_b);
		unsigned short b0 = (unsigned short)((b[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
		float a_f = __bfloat162float(__ushort_as_bfloat16(a0));
		float b_f = __bfloat162float(__ushort_as_bfloat16(b0));
		c0 = __bfloat16_as_ushort(__float2bfloat16(gelu_tanh(a_f) * b_f));
	}
	if (i1 < n) {
		int j = bc_b_index(i1, n_b);
		unsigned short b1 = (unsigned short)((b[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
		float a_f = __bfloat162float(__ushort_as_bfloat16(a1));
		float b_f = __bfloat162float(__ushort_as_bfloat16(b1));
		c1 = __bfloat16_as_ushort(__float2bfloat16(gelu_tanh(a_f) * b_f));
	}

	c[pair] = (unsigned int)c0 | ((unsigned int)c1 << 16);
}
