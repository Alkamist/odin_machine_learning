// entropy backward: dH/dp_i = -(log p_i + 1). `dy[r]` is the gradient flowing into row r's scalar
// entropy. Guarded at tiny p: there d/dp is unbounded, but the downstream softmax backward multiplies
// by p_i and the true contribution vanishes, so emitting 0 there is both finite and correct.
extern "C" __global__
void entropy_back_f32(const float* __restrict__ p,
                      const float* __restrict__ dy,
                      float*       __restrict__ dp,
                      int rows, int cols) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows) return;
	const float* pr = p + (size_t)r * cols;
	float* dpr = dp + (size_t)r * cols;
	float g = dy[r];

	for (int i = 0; i < cols; i++) {
		float v = pr[i];
		dpr[i] += (v > 1e-12f) ? (-g * (logf(v) + 1.0f)) : 0.0f;
	}
}
