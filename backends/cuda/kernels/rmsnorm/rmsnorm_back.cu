#ifdef DTYPE_BF16
#include "bf16.cuh"
#define DATA_T unsigned short
#define RD(p, i) ld_bf16(p, i)
#define KERNEL_NAME rmsnorm_back_bf16
#else
#define DATA_T float
#define RD(p, i) (p[i])
#define KERNEL_NAME rmsnorm_back_f32
#endif

#define RMS_WG     256
#define RMS_NWARPS (RMS_WG / 32)

extern "C" __global__
void KERNEL_NAME(const DATA_T* __restrict__ x,
                 const DATA_T* __restrict__ w,
                 const float*  __restrict__ rstd,
                 const float*  __restrict__ dy,
                 float*        __restrict__ dx,
                 float*        __restrict__ dw,
                 int count, int size) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int row_base = row * size;
	float r = rstd[row];

	float local = 0.0f;
	for (int i = tid; i < size; i += RMS_WG) {
		float xv  = RD(x, row_base + i);
		float wv  = RD(w, i);
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
		float xv  = RD(x, row_base + i);
		float wv  = RD(w, i);
		float dyv = dy[row_base + i];
		float norm  = xv * r;
		float dnorm = wv * dyv;
		if (dx) dx[row_base + i] += (dnorm - norm * mean) * r;
		if (dw) atomicAdd(&dw[i], dyv * norm);
	}
}
