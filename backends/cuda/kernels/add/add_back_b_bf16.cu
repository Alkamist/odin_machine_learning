// da_b[j] += sum_i dy[i*n_b + j], bf16 packed-pair output. Walks the
// broadcast slab serially, accumulating in fp32 for stability.
#include <cuda_bf16.h>
#include "broadcast.cuh"

extern "C" __global__
void add_back_b_bf16(const unsigned int* __restrict__ dy,
                     unsigned int*       __restrict__ da_b,
                     int n_b, int stride, int pair_count) {
	int pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= pair_count) return;

	int j0 = 2 * pair;
	int j1 = j0 + 1;

	float acc0 = 0.0f, acc1 = 0.0f;
	for (int i = 0; i < stride; ++i) {
		int idx0 = bc_tile_index(i, j0, n_b);
		int idx1 = bc_tile_index(i, j1, n_b);
		unsigned int dy0_pack = dy[idx0 >> 1];
		unsigned short dy0 = (unsigned short)((dy0_pack >> ((idx0 & 1) * 16)) & 0xffffu);
		acc0 += __bfloat162float(__ushort_as_bfloat16(dy0));
		if (j1 < n_b) {
			unsigned int dy1_pack = dy[idx1 >> 1];
			unsigned short dy1 = (unsigned short)((dy1_pack >> ((idx1 & 1) * 16)) & 0xffffu);
			acc1 += __bfloat162float(__ushort_as_bfloat16(dy1));
		}
	}

	unsigned int prev = da_b[pair];
	float prev0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(prev & 0xffffu)));
	float prev1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((prev >> 16) & 0xffffu)));
	unsigned short out0 = __bfloat16_as_ushort(__float2bfloat16(prev0 + acc0));
	unsigned short out1 = j1 < n_b
		? __bfloat16_as_ushort(__float2bfloat16(prev1 + acc1))
		: (unsigned short)((prev >> 16) & 0xffffu);
	da_b[pair] = (unsigned int)out0 | ((unsigned int)out1 << 16);
}
