// tanh backward: dx[i] += dy[i] * (1 - y[i]^2) where y = tanh(x).
// Uses the cached forward output rather than recomputing tanh from x.

extern "C" __global__
void tanh_back_f32(const float* __restrict__ y,
                   const float* __restrict__ dy,
                   float*       __restrict__ dx,
                   int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	float v = y[i];
	dx[i] += dy[i] * (1.0f - v * v);
}
