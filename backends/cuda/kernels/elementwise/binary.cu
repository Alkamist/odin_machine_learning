#include "broadcast.cuh"
extern "C" __global__
void OP_NAME(const float* __restrict__ a,
             const float* __restrict__ b,
             float*       __restrict__ c,
             int n, int n_b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float av = a[i];
	float bv = b[bc_b_index(i, n_b)];
	c[i] = (OP_EXPR);
}
