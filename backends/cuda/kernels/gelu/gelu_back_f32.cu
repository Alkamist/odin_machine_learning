// GELU backward (tanh approximation). Mirrors backends/cpu/cpu.odin:gelu_backward.
//   y     = 0.5 * x * (1 + tanh(t)),  t = sqrt(2/pi) * (x + 0.044715 * x^3)
//   dy/dx = 0.5 * (1 + tanh(t)) + 0.5 * x * sech^2(t) * sqrt(2/pi) * (1 + 3*0.044715*x^2)

extern "C" __global__
void gelu_back_f32(const float* __restrict__ x,
                   const float* __restrict__ dy,
                   float*       __restrict__ dx,
                   int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float v     = x[i];
	float cube  = 0.044715f * v * v * v;
	float t_arg = 0.7978845608028654f * (v + cube);
	float t     = tanhf(t_arg);
	float sech2 = 1.0f - t * t;
	float deriv = 0.5f * (1.0f + t) + 0.5f * v * sech2 * 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v * v);
	dx[i] += dy[i] * deriv;
}
