extern "C" __global__
void OP_NAME(const float* __restrict__ x,
             const float* __restrict__ y,
             const float* __restrict__ dy,
             float*       __restrict__ dx,
             int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float xv = x[i];
	float yv = y[i];
	(void)xv;
	(void)yv;
	dx[i] += dy[i] * (OP_DERIV);
}
