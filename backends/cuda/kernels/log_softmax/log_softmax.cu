extern "C" __global__
void log_softmax_f32(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows) return;
	const float* xr = x + (size_t)r * cols;
	float* yr = y + (size_t)r * cols;

	float m = xr[0];
	for (int i = 1; i < cols; ++i) m = fmaxf(m, xr[i]);

	float lse = 0.0f;
	for (int i = 0; i < cols; ++i) lse += expf(xr[i] - m);
	lse = logf(lse) + m;

	for (int i = 0; i < cols; ++i) yr[i] = xr[i] - lse;
}
