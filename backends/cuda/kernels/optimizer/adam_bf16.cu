// Mixed-precision Adam: bf16 weight, f32 gradient, f32 first/second moments.
// Reads bf16 W + f32 grad + f32 m + f32 v, computes the update in f32,
// writes back bf16 W and zeroes the f32 grad. m/v stay f32 across steps.

#include <cuda_bf16.h>

extern "C" __global__
void adam_bf16(
    __nv_bfloat16* __restrict__ d,
    float*         __restrict__ g,
    float*         __restrict__ m,
    float*         __restrict__ v,
    int n,
    float beta1, float beta2,
    float bc1, float bc2,
    float lr, float wd, float eps)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;

	float grad  = g[i];
	float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
	float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
	m[i] = m_new;
	v[i] = v_new;

	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;
	float w     = __bfloat162float(d[i]);
	w = w * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);
	d[i] = __float2bfloat16(w);
	g[i] = 0.0f;
}
