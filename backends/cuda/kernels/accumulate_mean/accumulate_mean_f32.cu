#define ACC_WG     256
#define ACC_NWARPS (ACC_WG / 32)

extern "C" __global__
void accumulate_mean_f32(const float* __restrict__ src, float* __restrict__ dst, int n) {
	int tid = threadIdx.x;
	float s = 0.0f;
	for (int i = tid; i < n; i += ACC_WG) s += src[i];

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);

	__shared__ float warp_sums[ACC_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s;
	__syncthreads();

	if (tid == 0) {
		float total = 0.0f;
		#pragma unroll
		for (int i = 0; i < ACC_NWARPS; ++i) total += warp_sums[i];
		dst[0] += total / (float)n;
	}
}
