extern "C" __global__
void max_reduce_f32(const float* __restrict__ x, float* __restrict__ y, int count, int size) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= count) return;
	const float* xr = x + (size_t)r * size;
	float m = xr[0];
	for (int i = 1; i < size; ++i) {
		if (xr[i] > m) m = xr[i];
	}
	y[r] = m;
}
