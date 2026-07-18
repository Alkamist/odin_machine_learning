// F32 embedding lookup. One thread per output column; the index row rides grid.y, which caps at
// 65535, so it grid-strides to cover any n_indices the caller clamped grid.y below.
extern "C" __global__
void select_f32(const float* __restrict__ table,
                const unsigned int* __restrict__ indices,
                float*       __restrict__ out,
                int n_indices, int size) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= size) return;
	for (int i = blockIdx.y; i < n_indices; i += gridDim.y) {
		out[i * size + j] = table[(int)indices[i] * size + j];
	}
}
