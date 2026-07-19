extern "C" __global__
void select_back(const float* __restrict__ dy,
                 const int*   __restrict__ indices,
                 float*       __restrict__ dw,
                 int n_idx, int row_size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total = n_idx * row_size;
	if (idx >= total) return;
	int i = idx / row_size;
	int d = idx - i * row_size;
	int target = indices[i];
	atomicAdd(&dw[target * row_size + d], dy[idx]);
}
