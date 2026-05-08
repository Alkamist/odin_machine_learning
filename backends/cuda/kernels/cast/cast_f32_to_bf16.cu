// Cast n f32 elements to n bf16 elements. dst is packed two bf16 per uint.
#include <cuda_bf16.h>

extern "C" __global__
void cast_f32_to_bf16(const float* __restrict__ src,
                      unsigned int* __restrict__ dst,
                      int n, int pair_count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= pair_count) return;
	int i0 = 2 * i, i1 = i0 + 1;
	unsigned short lo = __bfloat16_as_ushort(__float2bfloat16(src[i0]));
	unsigned short hi = (i1 < n) ? __bfloat16_as_ushort(__float2bfloat16(src[i1])) : 0;
	dst[i] = (unsigned int)lo | ((unsigned int)hi << 16);
}
