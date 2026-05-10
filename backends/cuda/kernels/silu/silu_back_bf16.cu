// silu backward, bf16 forward input + F32 grads.
// Reads bf16 x via packed-pair extract, reads/writes f32 dy/dx.
// One thread per element; native += into f32 dx (each thread targets unique i).
#include <cuda_bf16.h>

extern "C" __global__
void silu_back_bf16(const unsigned int* __restrict__ x,
                    const float*        __restrict__ dy,
                    float*              __restrict__ dx,
                    int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	unsigned short xb = (unsigned short)((x[i >> 1] >> ((i & 1) * 16)) & 0xffffu);
	float v = __bfloat162float(__ushort_as_bfloat16(xb));
	float s = 1.0f / (1.0f + expf(-v));
	dx[i] += dy[i] * (s + v * s * (1.0f - s));
}
