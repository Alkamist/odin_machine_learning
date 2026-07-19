#include <cuda_bf16.h>
#ifdef DTYPE_BF16
#include "bf16.cuh"
#define DATA_T unsigned short
#define RD(p, i) ld_bf16(p, i)
#define WR(p, i, val) st_bf16(p, i, (val))
#define ROUND_STORE(v) __bfloat162float(__float2bfloat16(v))
#define KERNEL_NAME add_rmsnorm_bf16
#else
#define DATA_T float
#define RD(p, i) (p[i])
#define WR(p, i, val) do { (p)[i] = (val); } while (0)
#define ROUND_STORE(v) (v)
#define KERNEL_NAME add_rmsnorm_f32
#endif

#define ARN_WG     256
#define ARN_NWARPS (ARN_WG / 32)

extern "C" __global__
void KERNEL_NAME(const DATA_T*       __restrict__ a,
                 const DATA_T*       __restrict__ b,
                 const unsigned int* __restrict__ w,
                 DATA_T*             __restrict__ r,
                 DATA_T*             __restrict__ y,
                 int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;

	int row_base = row * size;

	float s2 = 0.0f;
	for (int pi = tid; pi < (size >> 1); pi += ARN_WG) {
		float a0 = RD(a, row_base + 2 * pi + 0);
		float a1 = RD(a, row_base + 2 * pi + 1);
		float b0 = RD(b, row_base + 2 * pi + 0);
		float b1 = RD(b, row_base + 2 * pi + 1);
		float r0 = a0 + b0;
		float r1 = a1 + b1;
		WR(r, row_base + 2 * pi + 0, r0);
		WR(r, row_base + 2 * pi + 1, r1);
		float sr0 = ROUND_STORE(r0);
		float sr1 = ROUND_STORE(r1);
		s2 += sr0 * sr0 + sr1 * sr1;
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		s2 += __shfl_xor_sync(0xffffffffu, s2, off);
	}

	__shared__ float warp_sums[ARN_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s2;
	__syncthreads();

	float total = 0.0f;
	#pragma unroll
	for (int i = 0; i < ARN_NWARPS; ++i) total += warp_sums[i];

	float rstd = rsqrtf(total / (float)size + eps);

	for (int pi = tid; pi < (size >> 1); pi += ARN_WG) {
		float r0 = RD(r, row_base + 2 * pi + 0);
		float r1 = RD(r, row_base + 2 * pi + 1);
		unsigned int wp = w[pi];
		float w0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wp & 0xffffu)));
		float w1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((wp >> 16) & 0xffffu)));
		WR(y, row_base + 2 * pi + 0, r0 * rstd * w0);
		WR(y, row_base + 2 * pi + 1, r1 * rstd * w1);
	}
}
