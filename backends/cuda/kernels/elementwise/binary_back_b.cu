#include "broadcast.cuh"
extern "C" __global__
void OP_NAME(const float* __restrict__ a,
             const float* __restrict__ b,
             const float* __restrict__ dy,
             float*       __restrict__ db,
             int n_b, int stride) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n_b) return;
	float bv = b[j];
	(void)bv;
	float acc = 0.0f;
	for (int i = 0; i < stride; ++i) {
		int o = bc_tile_index(i, j, n_b);
		float av = a[o];
		(void)av;
		acc += dy[o] * (DB_EXPR);
	}
	db[j] += acc;
}
