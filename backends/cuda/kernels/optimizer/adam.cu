#ifdef DTYPE_BF16
#include "bf16.cuh"
#define WEIGHT_T unsigned short
#define WLOAD(p, i) ld_bf16(p, i)
#define WSTORE(p, i, val) st_bf16(p, i, (val))
#define KERNEL_NAME adam_bf16
#else
#define WEIGHT_T float
#define WLOAD(p, i) (p[i])
#define WSTORE(p, i, val) do { (p)[i] = (val); } while (0)
#define KERNEL_NAME adam_f32
#endif

extern "C" __global__
void KERNEL_NAME(
    WEIGHT_T* __restrict__ d,
    float*    __restrict__ g,
    float*    __restrict__ m,
    float*    __restrict__ v,
    int n,
    float beta1, float beta2,
    float bc1, float bc2,
    float lr, float wd, float eps)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;

	float grad = g[i];
	float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
	float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
	m[i] = m_new;
	v[i] = v_new;

	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;
	float w = WLOAD(d, i);
	w = w * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);
	WSTORE(d, i, w);
	g[i] = 0.0f;
}
