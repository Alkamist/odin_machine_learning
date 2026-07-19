#define LN_WG     256
#define LN_NWARPS (LN_WG / 32)

extern "C" __global__
void layernorm_back_f32(const float* __restrict__ x,
                        const float* __restrict__ w,
                        const float* __restrict__ mean,
                        const float* __restrict__ rstd,
                        const float* __restrict__ dy,
                        float*       __restrict__ dx,
                        float*       __restrict__ dw,
                        int count, int size, int have_dx, int have_dw) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;
	int base = row * size;

	float m  = mean[row];
	float rs = rstd[row];

	float dnorm_mean      = 0.0f;
	float dnorm_norm_mean = 0.0f;
	for (int i = tid; i < size; i += LN_WG) {
		float norm  = (x[base + i] - m) * rs;
		float dnorm = w[i] * dy[base + i];
		dnorm_mean      += dnorm;
		dnorm_norm_mean += dnorm * norm;
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		dnorm_mean      += __shfl_xor_sync(0xffffffffu, dnorm_mean,      off);
		dnorm_norm_mean += __shfl_xor_sync(0xffffffffu, dnorm_norm_mean, off);
	}

	__shared__ float warp_a[LN_NWARPS];
	__shared__ float warp_b[LN_NWARPS];
	if ((tid & 31) == 0) {
		warp_a[tid >> 5] = dnorm_mean;
		warp_b[tid >> 5] = dnorm_norm_mean;
	}
	__syncthreads();

	dnorm_mean      = 0.0f;
	dnorm_norm_mean = 0.0f;
	#pragma unroll
	for (int i = 0; i < LN_NWARPS; ++i) {
		dnorm_mean      += warp_a[i];
		dnorm_norm_mean += warp_b[i];
	}
	dnorm_mean      /= (float)size;
	dnorm_norm_mean /= (float)size;

	for (int i = tid; i < size; i += LN_WG) {
		float norm  = (x[base + i] - m) * rs;
		float dyv   = dy[base + i];
		if (have_dw) atomicAdd(&dw[i], norm * dyv);
		if (have_dx) {
			float dnorm = w[i] * dyv;
			float g = dnorm - dnorm_mean - norm * dnorm_norm_mean;
			g *= rs;
			dx[base + i] += g;
		}
	}
}
