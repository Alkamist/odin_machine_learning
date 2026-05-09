// F32 rmsnorm with f32 weight. One block per row, 256-wide butterfly
// reduction. Used by training paths (Llama defaults to F32 weights).
// Stores per-row `rstd` so backward doesn't have to recompute.

#define RMS_WG     256
#define RMS_NWARPS (RMS_WG / 32)

extern "C" __global__
void rmsnorm_f32(const float* __restrict__ x,
                 const float* __restrict__ w,
                 float*       __restrict__ y,
                 float*       __restrict__ rstd_out,
                 int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int row_base = row * size;

	float s2 = 0.0f;
	for (int i = tid; i < size; i += RMS_WG) {
		float v = x[row_base + i];
		s2 += v * v;
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		s2 += __shfl_xor_sync(0xffffffffu, s2, off);
	}

	__shared__ float warp_sums[RMS_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s2;
	__syncthreads();

	float total = 0.0f;
	#pragma unroll
	for (int i = 0; i < RMS_NWARPS; ++i) total += warp_sums[i];

	float rstd = rsqrtf(total / (float)size + eps);
	if (tid == 0) rstd_out[row] = rstd;

	for (int i = tid; i < size; i += RMS_WG) {
		y[row_base + i] = x[row_base + i] * rstd * w[i];
	}
}
