// rmsnorm backward, bf16 data + F32 grads.
//   y[i] = (x[i] * rstd) * w[i], rstd = 1/sqrt(mean(x^2) + eps).
//   dnorm[i]        = w[i] * dy[i]
//   dnorm_norm_mean = mean over i of dnorm[i] * (x[i] * rstd)
//   dx[i]          += (dnorm[i] - (x[i] * rstd) * dnorm_norm_mean) * rstd
//   dw[i]          += dy[i] * (x[i] * rstd)
//
// One block per row, 256-wide reduction. dx writes are unique per (row, i),
// no atomic needed. dw is shared across rows so native f32 atomicAdd handles
// the cross-block accumulation.
#include <cuda_bf16.h>

#define RMS_WG     256
#define RMS_NWARPS (RMS_WG / 32)

__device__ __forceinline__ float load_bf16_at(const unsigned int* base, int idx) {
	unsigned int p = base[idx >> 1];
	unsigned short h = (unsigned short)((p >> ((idx & 1) * 16)) & 0xffffu);
	return __bfloat162float(__ushort_as_bfloat16(h));
}

extern "C" __global__
void rmsnorm_back_bf16(const unsigned int* __restrict__ x,
                       const unsigned int* __restrict__ w,
                       const float*        __restrict__ rstd,
                       const float*        __restrict__ dy,
                       float*              __restrict__ dx,
                       float*              __restrict__ dw,
                       int count, int size) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int row_base = row * size;
	float r = rstd[row];

	float local = 0.0f;
	for (int i = tid; i < size; i += RMS_WG) {
		float xv  = load_bf16_at(x, row_base + i);
		float wv  = load_bf16_at(w, i);
		float dyv = dy[row_base + i];
		local += (wv * dyv) * (xv * r);
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
		float xv  = load_bf16_at(x, row_base + i);
		float wv  = load_bf16_at(w, i);
		float dyv = dy[row_base + i];
		float norm  = xv * r;
		float dnorm = wv * dyv;
		if (dx) dx[row_base + i] += (dnorm - norm * mean) * r;
		if (dw) atomicAdd(&dw[i], dyv * norm);
	}
}
