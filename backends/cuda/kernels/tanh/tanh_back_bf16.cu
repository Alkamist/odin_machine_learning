// tanh backward, bf16 forward output + F32 grads.
// dx[i] += dy[i] * (1 - y[i]^2). Uses cached forward output y.
#include <cuda_bf16.h>

extern "C" __global__
void tanh_back_bf16(const unsigned int* __restrict__ y,
                    const float*        __restrict__ dy,
                    float*              __restrict__ dx,
                    int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	unsigned short yb = (unsigned short)((y[i >> 1] >> ((i & 1) * 16)) & 0xffffu);
	float yv = __bfloat162float(__ushort_as_bfloat16(yb));
	dx[i] += dy[i] * (1.0f - yv * yv);
}
