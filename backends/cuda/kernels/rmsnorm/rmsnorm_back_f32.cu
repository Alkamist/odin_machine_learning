// F32 rmsnorm backward.
// Forward: y[i] = (x[i] * rstd) * w[i], rstd = 1/sqrt(mean(x^2) + eps).
// Backward (per row):
//   dnorm[i]        = w[i] * dy[i]
//   dnorm_norm_mean = mean over i of dnorm[i] * (x[i] * rstd)
//   dx[i]          += (dnorm[i] - (x[i] * rstd) * dnorm_norm_mean) * rstd
//   dw[i]          += dy[i] * (x[i] * rstd)
//
// One block per row, 256-wide reduction for the row mean. `dw` is shared
// across rows so writes use atomicAdd.

#define RMS_WG     256
#define RMS_NWARPS (RMS_WG / 32)

extern "C" __global__
void rmsnorm_back_f32(const float* __restrict__ x,
                      const float* __restrict__ w,
                      const float* __restrict__ rstd,
                      const float* __restrict__ dy,
                      float*       __restrict__ dx,
                      float*       __restrict__ dw,
                      int count, int size) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int row_base = row * size;
	float r = rstd[row];

	// Reduction over the row to compute the mean of (dnorm * norm).
	float local = 0.0f;
	for (int i = tid; i < size; i += RMS_WG) {
		float xv = x[row_base + i];
		float wv = w[i];
		float dy_v = dy[row_base + i];
		local += (wv * dy_v) * (xv * r);
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		local += __shfl_xor_sync(0xffffffffu, local, off);
	}

	__shared__ float warp_sums[RMS_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = local;
	__syncthreads();

	float total = 0.0f;
	#pragma unroll
	for (int i = 0; i < RMS_NWARPS; ++i) total += warp_sums[i];
	float mean = total / (float)size;

	for (int i = tid; i < size; i += RMS_WG) {
		float xv   = x [row_base + i];
		float wv   = w [i];
		float dy_v = dy[row_base + i];
		float norm = xv * r;
		float dnorm = wv * dy_v;
		if (dx) dx[row_base + i] += (dnorm - norm * mean) * r;
		if (dw) atomicAdd(&dw[i], dy_v * norm);
	}
}
