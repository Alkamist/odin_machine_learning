// Select backward. Forward: y[i, :] = w[indices[i], :].
// Backward: dw[indices[i], :] += dy[i, :].
// Multiple inputs may reference the same row, so atomicAdd into dw.

extern "C" __global__
void select_back_f32(const float* __restrict__ dy,        // [n_idx, row_size]
                     const int*   __restrict__ indices,    // [n_idx]
                     float*       __restrict__ dw,         // [vocab, row_size]
                     int n_idx, int row_size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total = n_idx * row_size;
	if (idx >= total) return;
	int i = idx / row_size;
	int d = idx - i * row_size;
	int target = indices[i];
	atomicAdd(&dw[target * row_size + d], dy[idx]);
}
