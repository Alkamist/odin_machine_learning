// Bf16 embedding lookup. table[indices[i]] -> out[i]. Pair-wise copy of `size/2` uints; the index
// row rides grid.y, which caps at 65535, so it grid-strides to cover any n_indices the caller
// clamped grid.y below.
extern "C" __global__
void select_bf16(const unsigned int* __restrict__ table,
                 const unsigned int* __restrict__ indices,
                 unsigned int*       __restrict__ out,
                 int n_indices, int size) {
	int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pair_count = size >> 1;
	if (pair_idx >= pair_count) return;

	for (int i = blockIdx.y; i < n_indices; i += gridDim.y) {
		int row = (int)indices[i] * pair_count;
		int dst = i * pair_count;
		out[dst + pair_idx] = table[row + pair_idx];
	}
}
