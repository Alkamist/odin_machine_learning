// Mul backward, b-side, bf16 data + F32 grads.
// db[j] += sum over i of dy[i] * a[i] for i where i % n_b == j.
// b broadcasts over leading dims; many a-elements may target the same b
// cell, so native f32 atomicAdd handles the contention.
#include <cuda_bf16.h>
#include "broadcast.cuh"

extern "C" __global__
void mul_back_b_bf16(const unsigned int* __restrict__ a,
                     const float*        __restrict__ dy,
                     float*              __restrict__ db,
                     int n_a, int n_b) {
	int o = blockIdx.x * blockDim.x + threadIdx.x;
	if (o >= n_a) return;
	unsigned short ab = (unsigned short)((a[o >> 1] >> ((o & 1) * 16)) & 0xffffu);
	float av = __bfloat162float(__ushort_as_bfloat16(ab));
	int j = bc_b_index(o, n_b);
	atomicAdd(&db[j], dy[o] * av);
}
