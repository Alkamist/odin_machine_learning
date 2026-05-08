// F32 embedding lookup. One thread per output element.
extern "C" __global__
void select_f32(const float* __restrict__ table,
                const unsigned int* __restrict__ indices,
                float*       __restrict__ out,
                int n_indices, int size) {
	int i = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_indices || j >= size) return;
	out[i * size + j] = table[(int)indices[i] * size + j];
}
