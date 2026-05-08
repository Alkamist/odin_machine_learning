#include <cuda_bf16.h>

extern "C" __global__
void tanh_bf16(const unsigned int* __restrict__ x,
               unsigned int*       __restrict__ y,
               int n, int pair_count) {
	int pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= pair_count) return;
	unsigned int xp = x[pair];
	int i0 = 2 * pair, i1 = i0 + 1;
	unsigned short c0 = 0, c1 = 0;
	if (i0 < n) {
		float v = __bfloat162float(__ushort_as_bfloat16((unsigned short)(xp & 0xffffu)));
		c0 = __bfloat16_as_ushort(__float2bfloat16(tanhf(v)));
	}
	if (i1 < n) {
		float v = __bfloat162float(__ushort_as_bfloat16((unsigned short)((xp >> 16) & 0xffffu)));
		c1 = __bfloat16_as_ushort(__float2bfloat16(tanhf(v)));
	}
	y[pair] = (unsigned int)c0 | ((unsigned int)c1 << 16);
}
