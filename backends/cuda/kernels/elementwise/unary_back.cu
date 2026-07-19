#ifdef DTYPE_BF16
#include "bf16.cuh"
#define DATA_T unsigned short
#define LOAD_EW(p, i) ld_bf16(p, i)
#else
#define DATA_T float
#define LOAD_EW(p, i) (p[i])
#endif

extern "C" __global__
void OP_NAME(const DATA_T* __restrict__ x,
             const DATA_T* __restrict__ y,
             const float*  __restrict__ dy,
             float*        __restrict__ dx,
             int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float xv = LOAD_EW(x, i);
	float yv = LOAD_EW(y, i);
	(void)xv;
	(void)yv;
	dx[i] += dy[i] * (OP_DERIV);
}
