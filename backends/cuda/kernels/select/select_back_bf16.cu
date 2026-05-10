// Select backward in bf16-data / F32-grad mode. Forward: y[i, :] = w[indices[i], :].
// Backward: dw[indices[i], :] += dy[i, :]. Native f32 atomicAdd for the
// row-shared accumulation (multiple input rows can target the same vocab row).
extern "C" __global__
void select_back_bf16(const float* __restrict__ dy,
                      const int*   __restrict__ indices,
                      float*       __restrict__ dw,
                      int n_idx, int row_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n_idx * row_size;
	if (idx >= total) return;
	int i = idx / row_size;
	int d = idx - i * row_size;
	int target = indices[i];
	atomicAdd(&dw[target * row_size + d], dy[idx]);
}
