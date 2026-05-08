extern "C" __global__
void tanh_f32(const float* __restrict__ x, float* __restrict__ y, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = tanhf(x[i]);
}
