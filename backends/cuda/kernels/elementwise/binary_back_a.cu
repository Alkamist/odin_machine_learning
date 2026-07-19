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
             float*        __restrict__ da,
             int n, int n_b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float av = LOAD_EW(a, i);
	float bv = LOAD_EW(b, bc_b_index(i, n_b));
	(void)av;
	(void)bv;
	da[i] += dy[i] * (DA_EXPR);
}
