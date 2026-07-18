// softmax backward from the cached output y: dx_i += y_i * (dy_i - sum_j y_j dy_j).
extern "C" __global__
void softmax_back_f32(const float* __restrict__ y,
                      const float* __restrict__ dy,
                      float*       __restrict__ dx,
                      int rows, int cols) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows) return;
	const float* yr  = y  + (size_t)r * cols;
	const float* dyr = dy + (size_t)r * cols;
	float* dxr = dx + (size_t)r * cols;

	float dot = 0.0f;
	for (int i = 0; i < cols; i++) dot += yr[i] * dyr[i];
	for (int i = 0; i < cols; i++) dxr[i] += yr[i] * (dyr[i] - dot);
}
