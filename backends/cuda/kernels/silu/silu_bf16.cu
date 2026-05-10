// silu(x) = x * sigmoid(x), bf16 packed pairs. Mirrors silu_f32.cu.
#include <cuda_bf16.h>

extern "C" __global__
void silu_bf16(const unsigned int* __restrict__ x,
               unsigned int*       __restrict__ y,
               int n, int pair_count) {
	int pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= pair_count) return;

	unsigned int xp = x[pair];

	int i0 = 2 * pair;
	int i1 = i0 + 1;

	float v0   = __bfloat162float(__ushort_as_bfloat16((unsigned short)(xp & 0xffffu)));
	float s0   = 1.0f / (1.0f + expf(-v0));
	float out0 = v0 * s0;

	unsigned short out_lo = __bfloat16_as_ushort(__float2bfloat16(out0));
	unsigned short out_hi = 0;
	if (i1 < n) {
		float v1   = __bfloat162float(__ushort_as_bfloat16((unsigned short)((xp >> 16) & 0xffffu)));
		float s1   = 1.0f / (1.0f + expf(-v1));
		float out1 = v1 * s1;
		out_hi     = __bfloat16_as_ushort(__float2bfloat16(out1));
	}
	y[pair] = (unsigned int)out_lo | ((unsigned int)out_hi << 16);
}
