// Row-wise softmax over the last dimension. One thread per row; vocabularies here are small (a few
// hundred at most), so a serial pass per row is fine. Shifted by the row max for numerical safety.
extern "C" __global__
void softmax_f32(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows) return;
	const float* xr = x + (size_t)r * cols;
	float* yr = y + (size_t)r * cols;

	float m = xr[0];
	for (int i = 1; i < cols; i++) m = fmaxf(m, xr[i]);

	float s = 0.0f;
	for (int i = 0; i < cols; i++) { float e = expf(xr[i] - m); yr[i] = e; s += e; }

	float inv = 1.0f / s;
	for (int i = 0; i < cols; i++) yr[i] *= inv;
}
