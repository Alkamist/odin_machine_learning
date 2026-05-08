// Bf16 flash-attention-2 forward, no KV cache. One block per (head, q_token).
// 64 threads do the per-tile online softmax: dot products, row-max,
// exp-sum, output accumulator update. Q is staged in shared memory once;
// K/V stream from global. Internal accumulators stay f32 for stability.
//
// head_size must be even. Caps at MAX_D=512 (matches the Vulkan path).
#include <cuda_bf16.h>

#define WG    64
#define BC    64
#define MAX_D 512
#define O_SLOTS ((MAX_D + WG - 1) / WG)

__device__ __forceinline__ float load_bf16(const unsigned int* buf, int elem_idx) {
	int pair = elem_idx >> 1;
	int shift = (elem_idx & 1) * 16;
	unsigned short bits = (unsigned short)((buf[pair] >> shift) & 0xffffu);
	unsigned int as_u32 = (unsigned int)bits << 16;
	return __int_as_float((int)as_u32);
}

__device__ __forceinline__ unsigned short bf16_round(float v) {
	unsigned int bits = __float_as_uint(v);
	if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
	unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
	return (unsigned short)((rounded >> 16) & 0xffffu);
}

extern "C" __global__
void attention_bf16(const unsigned int* __restrict__ q_buf,
                    const unsigned int* __restrict__ k_buf,
                    const unsigned int* __restrict__ v_buf,
                    unsigned int*       __restrict__ o,
                    float*              __restrict__ lse,
                    int n_q_heads, int n_kv_heads, int head_size,
                    int token_count, int q_size, int kv_size,
                    int causal, int window) {
	int h    = blockIdx.x;
	int t_q  = blockIdx.y;
	int tid  = threadIdx.x;

	int   D    = head_size;
	int   T    = token_count;
	int   kv_h = h * n_kv_heads / n_q_heads;
	float inv_sqrt_d = rsqrtf((float)D);

	int t_k_max = (causal != 0) ? (t_q + 1) : T;
	int t_k_min = (window != 0 && t_k_max > window) ? (t_k_max - window) : 0;
	int t_k0_start = (t_k_min / BC) * BC;

	int q_base = t_q * q_size + h * D;
	int o_base = t_q * q_size + h * D;

	__shared__ float q_shared[MAX_D];
	__shared__ float o_shared[MAX_D];
	__shared__ float score_tile[BC];
	__shared__ float partial[WG];

	for (int d = tid; d < D; d += WG) {
		q_shared[d] = load_bf16(q_buf, q_base + d);
	}
	__syncthreads();

	float o_acc[O_SLOTS];
	#pragma unroll
	for (int slot = 0; slot < O_SLOTS; ++slot) o_acc[slot] = 0.0f;

	const float NEG_INF = -3.402823e38f;
	float m_run = NEG_INF;
	float l_run = 0.0f;

	for (int t_k0 = t_k0_start; t_k0 < t_k_max; t_k0 += BC) {
		int t_k = t_k0 + tid;
		float score = NEG_INF;
		if (tid < BC && t_k < t_k_max && t_k >= t_k_min) {
			int k_base = t_k * kv_size + kv_h * D;
			float dot = 0.0f;
			for (int d = 0; d < D; ++d) {
				dot += q_shared[d] * load_bf16(k_buf, k_base + d);
			}
			score = dot * inv_sqrt_d;
		}
		if (tid < BC) score_tile[tid] = score;
		partial[tid] = score;
		__syncthreads();

		#pragma unroll
		for (int stride2 = WG / 2; stride2 > 0; stride2 >>= 1) {
			if (tid < stride2) partial[tid] = fmaxf(partial[tid], partial[tid + stride2]);
			__syncthreads();
		}
		float m_tile = partial[0];
		float m_new  = fmaxf(m_run, m_tile);

		float p_val = (tid < BC) ? expf(score_tile[tid] - m_new) : 0.0f;
		if (tid < BC) score_tile[tid] = p_val;
		partial[tid] = p_val;
		__syncthreads();
		#pragma unroll
		for (int stride2 = WG / 2; stride2 > 0; stride2 >>= 1) {
			if (tid < stride2) partial[tid] += partial[tid + stride2];
			__syncthreads();
		}
		float l_tile = partial[0];

		float alpha = (m_run == NEG_INF) ? 0.0f : expf(m_run - m_new);
		l_run = alpha * l_run + l_tile;
		m_run = m_new;

		#pragma unroll
		for (int slot = 0; slot < O_SLOTS; ++slot) {
			int d = tid + slot * WG;
			if (d < D) {
				float contrib = 0.0f;
				int j_max = min(BC, t_k_max - t_k0);
				for (int j = 0; j < j_max; ++j) {
					int v_base = (t_k0 + j) * kv_size + kv_h * D;
					contrib += score_tile[j] * load_bf16(v_buf, v_base + d);
				}
				o_acc[slot] = alpha * o_acc[slot] + contrib;
			}
		}
		__syncthreads();
	}

	float inv_l = 1.0f / l_run;
	#pragma unroll
	for (int slot = 0; slot < O_SLOTS; ++slot) {
		int d = tid + slot * WG;
		if (d < D) o_shared[d] = o_acc[slot] * inv_l;
	}
	__syncthreads();

	int pair_count = D >> 1;
	for (int pi = tid; pi < pair_count; pi += WG) {
		int d0 = 2 * pi;
		int d1 = d0 + 1;
		unsigned short lo = bf16_round(o_shared[d0]);
		unsigned short hi = bf16_round(o_shared[d1]);
		int elem_global = o_base + d0;
		o[elem_global >> 1] = ((unsigned int)hi << 16) | (unsigned int)lo;
	}

	if (tid == 0) {
		lse[h * T + t_q] = m_run + logf(l_run);
	}
}
