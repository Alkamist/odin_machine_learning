// Fused fp32 rmsnorm-with-bf16-weight + rope. Mirrors `rmsnorm_rope_bf16.cu`
// but reads fp32 activations and writes fp32 output. Weight is bf16
// (the model's stored dtype); the kernel converts each weight pair on use.
//
// One workgroup per (token, head). Activations are fp32, so per-element pair
// indexing is at half the stride of the bf16-packed-pair layout.
#include <cuda_bf16.h>

#define RMSROPE_WG     128
#define RMSROPE_NWARPS (RMSROPE_WG / 32)

extern "C" __global__
void rmsnorm_rope_f32(const float*        __restrict__ x,
                      const unsigned int* __restrict__ w,
                      float*              __restrict__ y,
                      int token_count, int head_count, int head_size, float eps,
                      float base, const int* __restrict__ position_offset_dev, int rotate_pair_count) {
	int wg_id = blockIdx.x;
	int tid   = threadIdx.x;
	int head  = wg_id % head_count;
	int pos   = wg_id / head_count;
	int position_offset = *position_offset_dev;

	int pair_count = head_size >> 1;
	int base_elem  = (pos * head_count + head) * head_size;

	float s2 = 0.0f;
	for (int pi = tid; pi < pair_count; pi += RMSROPE_WG) {
		float v0 = x[base_elem + 2*pi + 0];
		float v1 = x[base_elem + 2*pi + 1];
		s2 += v0 * v0 + v1 * v1;
	}

	#pragma unroll
	for (int off = 16; off > 0; off >>= 1) {
		s2 += __shfl_xor_sync(0xffffffffu, s2, off);
	}

	__shared__ float warp_sums[RMSROPE_NWARPS];
	if ((tid & 31) == 0) warp_sums[tid >> 5] = s2;
	__syncthreads();

	float total = 0.0f;
	#pragma unroll
	for (int i = 0; i < RMSROPE_NWARPS; ++i) total += warp_sums[i];

	float rstd = rsqrtf(total / (float)head_size + eps);

	float pos_f       = (float)(pos + position_offset);
	float head_size_f = (float)head_size;
	for (int pi = tid; pi < pair_count; pi += RMSROPE_WG) {
		float v0 = x[base_elem + 2*pi + 0];
		float v1 = x[base_elem + 2*pi + 1];
		unsigned int wp = w[pi];
		float w0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(wp & 0xffffu)));
		float w1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)((wp >> 16) & 0xffffu)));
		float n0 = v0 * rstd * w0;
		float n1 = v1 * rstd * w1;

		float o0, o1;
		if (pi < rotate_pair_count) {
			float exponent = (float)(pi * 2) / head_size_f;
			float theta    = pos_f / powf(base, exponent);
			float c_v, s_v;
			sincosf(theta, &s_v, &c_v);
			o0 = n0 * c_v - n1 * s_v;
			o1 = n0 * s_v + n1 * c_v;
		} else {
			o0 = n0; o1 = n1;
		}
		y[base_elem + 2*pi + 0] = o0;
		y[base_elem + 2*pi + 1] = o1;
	}
}
