// Bf16 embedding lookup. table[indices[i]] -> out[i]. One workgroup per
// output row (gridDim.y == n_indices); pair-wise copy of `size/2` uints.
extern "C" __global__
void select_bf16(const unsigned int* __restrict__ table,
                 const unsigned int* __restrict__ indices,
                 unsigned int*       __restrict__ out,
                 int n_indices, int size) {
	int i        = blockIdx.y;
	int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int pair_count = size >> 1;
	if (i >= n_indices || pair_idx >= pair_count) return;

	int row = (int)indices[i] * pair_count;
	int dst = i * pair_count;
	out[dst + pair_idx] = table[row + pair_idx];
}
