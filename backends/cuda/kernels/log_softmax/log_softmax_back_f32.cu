extern "C" __global__
void log_softmax_back_f32(const float* __restrict__ y,
                          const float* __restrict__ dy,
                          float*       __restrict__ dx,
                          int rows, int cols) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows) return;
	const float* yr  = y  + (size_t)r * cols;
	const float* dyr = dy + (size_t)r * cols;
	float* dxr = dx + (size_t)r * cols;

	float gsum = 0.0f;
	for (int i = 0; i < cols; ++i) gsum += dyr[i];
	for (int i = 0; i < cols; ++i) dxr[i] += dyr[i] - expf(yr[i]) * gsum;
}
