// out = clamp(x, lo, hi), elementwise.
extern "C" __global__
void clamp_f32(const float* __restrict__ x, float* __restrict__ y, float lo, float hi, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float v = x[i];
	y[i] = v < lo ? lo : (v > hi ? hi : v);
}
