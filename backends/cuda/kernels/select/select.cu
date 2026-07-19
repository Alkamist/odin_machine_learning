#ifdef DTYPE_BF16
#define DATA_T unsigned short
#define KERNEL_NAME select_bf16
#else
#define DATA_T float
#define KERNEL_NAME select_f32
#endif

extern "C" __global__
void KERNEL_NAME(const DATA_T*    __restrict__ table,
                 const int*       __restrict__ indices,
                 DATA_T*          __restrict__ out,
                 int n_indices, int size) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= size) return;
	for (int i = blockIdx.y; i < n_indices; i += gridDim.y) {
		out[i * size + j] = table[indices[i] * size + j];
	}
}
