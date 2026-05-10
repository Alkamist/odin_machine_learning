// Mul backward, a-side, bf16 data + F32 grads.
// da[o] += dy[o] * b[o % n_b]. One thread per output element.
#include <cuda_bf16.h>

extern "C" __global__
void mul_back_a_bf16(const unsigned int* __restrict__ b,
                     const float*        __restrict__ dy,
                     float*              __restrict__ da,
                     int n_a, int n_b) {
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	if (o >= n_a) return;
	int j = o % n_b;
	unsigned short bb = (unsigned short)((b[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
	float bv = __bfloat162float(__ushort_as_bfloat16(bb));
	da[o] += dy[o] * bv;
}
