// out = min(a, b), elementwise, same shape (no broadcast).
extern "C" __global__
void min_f32(const float* __restrict__ a,
             const float* __restrict__ b,
             float*       __restrict__ c,
             int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	c[i] = a[i] < b[i] ? a[i] : b[i];
}
