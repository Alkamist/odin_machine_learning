// Cast n bf16 elements to n f32 elements. src is packed two bf16 per uint.
#include <cuda_bf16.h>

extern "C" __global__
void cast_bf16_to_f32(const unsigned int* __restrict__ src,
                      float*              __restrict__ dst,
                      int n, int pair_count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= pair_count) return;
	unsigned int packed = src[i];
	int i0 = 2 * i, i1 = i0 + 1;
	if (i0 < n) {
		dst[i0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xffffu)));
	}
	if (i1 < n) {
		dst[i1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)((packed >> 16) & 0xffffu)));
	}
}
