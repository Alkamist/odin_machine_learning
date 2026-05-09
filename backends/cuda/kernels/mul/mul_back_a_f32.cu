// Mul backward, a-side. For c = a * b (b broadcasts over leading dims),
// da[o] += dy[o] * b[o % n_b]. One thread per output element, no atomics.

extern "C" __global__
void mul_back_a_f32(const float* __restrict__ b,
                    const float* __restrict__ dy,
                    float*       __restrict__ da,
                    int n_a, int n_b) {
	int o = blockDim.x * blockIdx.x + threadIdx.x;
	if (o >= n_a) return;
	int j = o % n_b;
	da[o] += dy[o] * b[j];
}
