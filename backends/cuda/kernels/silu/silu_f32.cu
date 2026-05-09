// silu(x) = x * sigmoid(x). One thread per element.

extern "C" __global__
void silu_f32(const float* __restrict__ x, float* __restrict__ y, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	float v = x[i];
	float s = 1.0f / (1.0f + expf(-v));
	y[i] = v * s;
}
