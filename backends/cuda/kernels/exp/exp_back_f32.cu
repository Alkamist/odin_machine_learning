// exp backward: dx[i] += dy[i] * y[i] where y = exp(x). Uses the cached
// forward output, since d/dx exp(x) = exp(x) = y.

extern "C" __global__
void exp_back_f32(const float* __restrict__ y,
                  const float* __restrict__ dy,
                  float*       __restrict__ dx,
                  int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	dx[i] += dy[i] * y[i];
}
