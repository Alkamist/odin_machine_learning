// da_b[j] += sum_i dy[i*n_b + j]. Thread per b element, serial reduce.
extern "C" __global__
void add_back_b_f32(const float* __restrict__ dy,
                    float*       __restrict__ da_b,
                    int n_b, int stride) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n_b) return;
	float acc = 0.0f;
	for (int i = 0; i < stride; ++i) {
		acc += dy[i * n_b + j];
	}
	da_b[j] += acc;
}
