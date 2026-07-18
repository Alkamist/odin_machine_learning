extern "C" __global__
void sq_sum_f32(const float* __restrict__ g, double* __restrict__ acc, int n) {
	__shared__ double sdata[256];
	int tid = threadIdx.x;

	double local = 0.0;
	for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
		double x = (double)g[i];
		local += x * x;
	}

	sdata[tid] = local;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(acc, sdata[0]);
	}
}
