extern "C" __global__
void mse_f32(const float* __restrict__ pred,
             const float* __restrict__ tgt,
             float*       __restrict__ out,
             int count, int sample_size) {
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= count) return;
	const float* p = pred + (size_t)s * sample_size;
	const float* t = tgt  + (size_t)s * sample_size;
	float sum = 0.0f;
	for (int i = 0; i < sample_size; ++i) {
		float d = p[i] - t[i];
		sum += d * d;
	}
	out[s] = sum / (float)sample_size;
}
