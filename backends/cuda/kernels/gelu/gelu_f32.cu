// GELU activation, tanh approximation. Mirrors backends/cpu/cpu.odin:gelu_forward.
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

extern "C" __global__
void gelu_f32(const float* __restrict__ x, float* __restrict__ y, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float v    = x[i];
	float cube = 0.044715f * v * v * v;
	y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + cube)));
}
