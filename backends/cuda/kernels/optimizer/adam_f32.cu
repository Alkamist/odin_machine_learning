// Adam optimizer step. One thread per parameter element. Mirrors the CPU
// implementation in backends/cpu/cpu.odin (`update` proc): updates the
// running first/second moments, applies bias correction, computes the
// weight delta, then zeroes the gradient so the next forward starts clean.

extern "C" __global__
void adam_f32(
    float* __restrict__ d,
    float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
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
	d[i] = d[i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);
	g[i] = 0.0f;
}
