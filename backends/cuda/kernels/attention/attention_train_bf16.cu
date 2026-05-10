// Bf16 attention forward for training. Port of attention_train_f32.cu:
// reads/writes bf16 for Q/K/V/out, materialises softmax_outputs as f32
// (backward needs the row probabilities), accumulates internally in fp32.
// One block per (head, q_token), 256 threads. Caps T at 2048 (smem-bound
// in backward, mirrored for symmetry).
#include <cuda_bf16.h>

#define ATT_WG 256

__device__ __forceinline__ float load_bf16(const unsigned int* buf, int elem_idx) {
	int pair  = elem_idx >> 1;
	int shift = (elem_idx & 1) * 16;
	unsigned short bits = (unsigned short)((buf[pair] >> shift) & 0xffffu);
	return __bfloat162float(__ushort_as_bfloat16(bits));
}

__device__ __forceinline__ unsigned short bf16_pack(float v) {
	return __bfloat16_as_ushort(__float2bfloat16(v));
}

extern "C" __global__
void attention_train_bf16(const unsigned int* __restrict__ q,
                          const unsigned int* __restrict__ k,
                          const unsigned int* __restrict__ v,
                          unsigned int*       __restrict__ out,
                          float*              __restrict__ sm,
                          int n_q_heads, int n_kv_heads, int head_size,
                          int token_count, int q_size, int kv_size,
                          int causal, int window) {
	int h    = blockIdx.x;
	int t_q  = blockIdx.y;
	int tid  = threadIdx.x;

	int T   = token_count;
	int D   = head_size;
	int gqa = n_q_heads / n_kv_heads;
	int kv_h = h / gqa;

	float inv_sqrt_d = rsqrtf((float)D);

	int t_k_max = (causal != 0) ? (t_q + 1) : T;
	int t_k_min = (window != 0 && t_k_max > window) ? (t_k_max - window) : 0;

	int q_offset  = t_q * q_size  + h    * D;
	int o_offset  = t_q * q_size  + h    * D;
	int sm_offset = h * T * T + t_q * T;

	__shared__ float reduction[ATT_WG];

	float local_max = -3.402823e38f;
	for (int t_k = tid; t_k < T; t_k += ATT_WG) {
		float score;
		if (t_k >= t_k_min && t_k < t_k_max) {
			int k_offset = t_k * kv_size + kv_h * D;
			float dot = 0.0f;
			for (int d = 0; d < D; ++d) {
				dot += load_bf16(q, q_offset + d) * load_bf16(k, k_offset + d);
			}
			score = dot * inv_sqrt_d;
		} else {
			score = -3.402823e38f;
		}
		sm[sm_offset + t_k] = score;
		if (score > local_max) local_max = score;
	}
	reduction[tid] = local_max;
	__syncthreads();
	for (int s = ATT_WG / 2; s > 0; s >>= 1) {
		if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid + s]);
		__syncthreads();
	}
	float row_max = reduction[0];

	float local_sum = 0.0f;
	for (int t_k = tid; t_k < T; t_k += ATT_WG) {
		float v_in = sm[sm_offset + t_k];
		float e = (v_in == -3.402823e38f) ? 0.0f : expf(v_in - row_max);
		sm[sm_offset + t_k] = e;
		local_sum += e;
	}
	reduction[tid] = local_sum;
	__syncthreads();
	for (int s = ATT_WG / 2; s > 0; s >>= 1) {
		if (tid < s) reduction[tid] += reduction[tid + s];
		__syncthreads();
	}
	float inv_sum = 1.0f / reduction[0];

	for (int t_k = tid; t_k < T; t_k += ATT_WG) {
		sm[sm_offset + t_k] *= inv_sum;
	}
	__syncthreads();

	// Output: pair-wise to avoid conflicts on shared 32-bit packed words.
	int half_d = D / 2;
	for (int dp = tid; dp < half_d; dp += ATT_WG) {
		int d0 = dp * 2;
		int d1 = d0 + 1;
		float acc0 = 0.0f, acc1 = 0.0f;
		for (int t_k = t_k_min; t_k < t_k_max; ++t_k) {
			int v_offset = t_k * kv_size + kv_h * D;
			float p = sm[sm_offset + t_k];
			acc0 += p * load_bf16(v, v_offset + d0);
			acc1 += p * load_bf16(v, v_offset + d1);
		}
		unsigned short lo = bf16_pack(acc0);
		unsigned short hi = bf16_pack(acc1);
		out[(o_offset + d0) >> 1] = (unsigned int)lo | ((unsigned int)hi << 16);
	}
}
