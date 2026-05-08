// out = a * b with b broadcast across len(a)/n_b chunks.
extern "C" __global__
void mul_f32(const float* __restrict__ a,
             const float* __restrict__ b,
             float*       __restrict__ c,
             int n, int n_b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	c[i] = a[i] * b[i % n_b];
}
