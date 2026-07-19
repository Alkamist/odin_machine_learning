#ifdef DTYPE_BF16
#include "bf16.cuh"
#define DATA_T unsigned short
#define RD(p, i) ld_bf16(p, i)
#define WR(p, i, val) st_bf16(p, i, (val))
#define KERNEL_NAME rmsnorm_bf16
#else
#define DATA_T float
#define RD(p, i) (p[i])
#define WR(p, i, val) do { (p)[i] = (val); } while (0)
#define KERNEL_NAME rmsnorm_f32
#endif

#define RMS_WG     256
#define RMS_NWARPS (RMS_WG / 32)

extern "C" __global__
void KERNEL_NAME(const DATA_T* __restrict__ x,
                 const DATA_T* __restrict__ w,
                 DATA_T*       __restrict__ y,
                 float*        __restrict__ rstd_out,
                 int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int row_base = row * size;

	float s2 = 0.0f;
	for (int i = tid; i < size; i += RMS_WG) {
		float v = RD(x, row_base + i);
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
	if (tid == 0 && rstd_out) rstd_out[row] = rstd;

	for (int i = tid; i < size; i += RMS_WG) {
		WR(y, row_base + i, RD(x, row_base + i) * rstd * RD(w, i));
	}
}
