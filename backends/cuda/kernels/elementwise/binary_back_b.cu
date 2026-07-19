#include "broadcast.cuh"
#ifdef DTYPE_BF16
#include "bf16.cuh"
#define DATA_T unsigned short
#define LOAD_EW(p, i) ld_bf16(p, i)
#else
#define DATA_T float
#define LOAD_EW(p, i) (p[i])
#endif

extern "C" __global__
void OP_NAME(const DATA_T* __restrict__ a,
             const DATA_T* __restrict__ b,
             const float*  __restrict__ dy,
             float*        __restrict__ db,
             int n_b, int stride) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n_b) return;
	float bv = LOAD_EW(b, j);
	(void)bv;
	float acc = 0.0f;
	for (int i = 0; i < stride; ++i) {
		int o = bc_tile_index(i, j, n_b);
		float av = LOAD_EW(a, o);
		(void)av;
		acc += dy[o] * (DB_EXPR);
	}
	db[j] += acc;
}
