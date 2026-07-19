#define LN_WG     256
#define LN_NWARPS (LN_WG / 32)

extern "C" __global__
void layernorm_f32(const float* __restrict__ x,
                   const float* __restrict__ w,
                   float*       __restrict__ y,
                   float*       __restrict__ mean_out,
                   float*       __restrict__ rstd_out,
                   int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;
	int base = row * size;

	__shared__ float warp_sums[LN_NWARPS];

	float s = 0.0f;
	for (int i = tid; i < size; i += LN_WG) s += x[base + i];
	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s;
	__syncthreads();
	float sum = 0.0f;
	#pragma unroll
	for (int i = 0; i < LN_NWARPS; ++i) sum += warp_sums[i];
	float mean = sum / (float)size;
	__syncthreads();

	float v = 0.0f;
	for (int i = tid; i < size; i += LN_WG) {
		float d = x[base + i] - mean;
		v += d * d;
	}
	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
	if ((tid & 31) == 0) warp_sums[tid >> 5] = v;
	__syncthreads();
	float var = 0.0f;
	#pragma unroll
	for (int i = 0; i < LN_NWARPS; ++i) var += warp_sums[i];
	var /= (float)size;

	float rstd = 1.0f / sqrtf(var + eps);
	if (tid == 0) {
		mean_out[row] = mean;
		rstd_out[row] = rstd;
	}

	for (int i = tid; i < size; i += LN_WG) {
		float n = (x[base + i] - mean) * rstd;
		y[base + i] = n * w[i];
	}
}
