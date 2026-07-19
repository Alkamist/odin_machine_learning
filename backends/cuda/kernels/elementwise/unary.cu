extern "C" __global__
void OP_NAME(const float* __restrict__ x, float* __restrict__ y, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float v = x[i];
	y[i] = (OP_EXPR);
}
