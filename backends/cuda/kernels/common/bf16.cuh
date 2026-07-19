#pragma once
#include <cuda_bf16.h>

__device__ __forceinline__ float ld_bf16(const unsigned short* p, int i) {
	return __bfloat162float(__ushort_as_bfloat16(p[i]));
}

__device__ __forceinline__ void st_bf16(unsigned short* p, int i, float v) {
	p[i] = __bfloat16_as_ushort(__float2bfloat16(v));
}
