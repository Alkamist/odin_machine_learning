extern "C" __global__
void mean_f32(const float* __restrict__ x, float* __restrict__ y, int count, int size) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= count) return;
	const float* xr = x + (size_t)r * size;
	float s = 0.0f;
	for (int i = 0; i < size; ++i) s += xr[i];
	y[r] = s / (float)size;
}
