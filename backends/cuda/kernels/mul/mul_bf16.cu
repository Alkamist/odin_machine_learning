// Bf16 broadcast mul. Native __hmul on Ampere.
#include <cuda_bf16.h>
#include "broadcast.cuh"

extern "C" __global__
void mul_bf16(const unsigned int* __restrict__ a,
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
		c0 = __bfloat16_as_ushort(__hmul(__ushort_as_bfloat16(a0), __ushort_as_bfloat16(b0)));
	}
	if (i1 < n) {
		int j = bc_b_index(i1, n_b);
		unsigned short b1 = (unsigned short)((b[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
		c1 = __bfloat16_as_ushort(__hmul(__ushort_as_bfloat16(a1), __ushort_as_bfloat16(b1)));
	}

	c[pair] = (unsigned int)c0 | ((unsigned int)c1 << 16);
}
