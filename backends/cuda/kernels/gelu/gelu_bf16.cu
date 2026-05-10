// GELU activation, tanh approximation, bf16 packed pairs. Mirrors gelu_f32.cu.
#include <cuda_bf16.h>

extern "C" __global__
void gelu_bf16(const unsigned int* __restrict__ x,
               unsigned int*       __restrict__ y,
               int n, int pair_count) {
	int pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= pair_count) return;

	unsigned int xp = x[pair];

	int i0 = 2 * pair;
	int i1 = i0 + 1;

	float v0   = __bfloat162float(__ushort_as_bfloat16((unsigned short)(xp & 0xffffu)));
	float c0   = 0.044715f * v0 * v0 * v0;
	float out0 = 0.5f * v0 * (1.0f + tanhf(0.7978845608028654f * (v0 + c0)));

	unsigned short out_lo = __bfloat16_as_ushort(__float2bfloat16(out0));
	unsigned short out_hi = 0;
	if (i1 < n) {
		float v1   = __bfloat162float(__ushort_as_bfloat16((unsigned short)((xp >> 16) & 0xffffu)));
		float c1   = 0.044715f * v1 * v1 * v1;
		float out1 = 0.5f * v1 * (1.0f + tanhf(0.7978845608028654f * (v1 + c1)));
		out_hi     = __bfloat16_as_ushort(__float2bfloat16(out1));
	}
	y[pair] = (unsigned int)out_lo | ((unsigned int)out_hi << 16);
}
