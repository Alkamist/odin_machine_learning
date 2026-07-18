extern "C" __global__
void scale_f32(float* __restrict__ g, int n, float s) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		g[i] *= s;
	}
}
