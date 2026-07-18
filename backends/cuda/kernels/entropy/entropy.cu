// Shannon entropy of each row of probabilities: h[r] = -sum_i p_i log p_i, over the last dimension.
// One thread per row. p_i = 0 contributes 0 (the limit of p log p).
extern "C" __global__
void entropy_f32(const float* __restrict__ p, float* __restrict__ h, int rows, int cols) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows) return;
	const float* pr = p + (size_t)r * cols;

	float e = 0.0f;
	for (int i = 0; i < cols; i++) {
		float v = pr[i];
		if (v > 0.0f) e -= v * logf(v);
	}
	h[r] = e;
}
