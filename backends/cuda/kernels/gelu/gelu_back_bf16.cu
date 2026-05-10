// GELU backward (tanh approximation), bf16 forward input + F32 grads.
#include <cuda_bf16.h>

extern "C" __global__
void gelu_back_bf16(const unsigned int* __restrict__ x,
                    const float*        __restrict__ dy,
                    float*              __restrict__ dx,
                    int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	unsigned short xb = (unsigned short)((x[i >> 1] >> ((i & 1) * 16)) & 0xffffu);
	float v     = __bfloat162float(__ushort_as_bfloat16(xb));
	float c     = 0.044715f * v * v * v;
	float t_arg = 0.7978845608028654f * (v + c);
	float t     = tanhf(t_arg);
	float sech2 = 1.0f - t * t;
	float deriv = 0.5f * (1.0f + t) + 0.5f * v * sech2 * 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v * v);
	dx[i] += dy[i] * deriv;
}
