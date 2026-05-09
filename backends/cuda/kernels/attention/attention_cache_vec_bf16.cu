// Port of ggml's `flash_attn_ext_vec` (fattn-vec.cuh) stripped to our case:
//   - bf16 K, bf16 V (no quantized kv).
//   - ncols = 1 (single Q token per block; loop in caller for multi-token prefill).
//   - causal mask with optional sliding window (no ALiBi, no logit_softcap, no sinks).
//   - GQA via kv_h = h * n_kv_heads / n_q_heads.
//   - Output: bf16 packed pairs (one block writes its full D-vector at end).
//
// Cache layout is linear (= ggml's): slot 0 is oldest, slot cap-1 is newest.
// For full layers and for sliding layers within their first `capacity`
// tokens, slot j corresponds to seq_pos j. After a sliding layer's window
// fills (at seq_pos >= cap), the host shifts the cache back by n_rows
// before each write so the cache always holds the most recent `cap` rows
// at slots [0..cap), with the newest at slot cap-1.
//
// Per-Q-token live K range, computed unified for both layer types:
//   t_q_slot = min(cache_position + t_q, capacity - q_token_count + t_q)
//   t_k_max  = t_q_slot + 1                              (causal)
//   t_k_min  = max(0, t_q_slot + 1 - window) if window>0 (sliding mask)
//            = 0                              if window=0 (no window)
#include <cuda_bf16.h>

#ifndef D_HEAD
#define D_HEAD 256
#endif

#define D            D_HEAD
#define WARP_SIZE    32
#define NWARPS       4
#define NTHREADS     (WARP_SIZE * NWARPS)            // = 128
#define NTHREADS_KQ  8
#define NTHREADS_V   8
#define V_COLS_PER_ITER (WARP_SIZE / NTHREADS_V)     // = 4
#define BC           NTHREADS                          // = 128 K-rows per outer iter
#define D_PER_THREAD_KQ (D / NTHREADS_KQ)             // = 32 (per-thread Q slice)
#define D_PER_THREAD_V  (D / NTHREADS_V)              // = 32 (per-thread VKQ accumulators)
#define HALF_D_PER_THREAD_V (D_PER_THREAD_V / 2)      // = 16 float2 pairs

__device__ __forceinline__ float bf16_uint16_to_float(unsigned short bits) {
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
__launch_bounds__(NTHREADS, 1)
void attention_cache_vec_bf16(
    const unsigned int* __restrict__ q_buf,
    const unsigned int* __restrict__ k_buf,
    const unsigned int* __restrict__ v_buf,
    unsigned int*       __restrict__ o,
    int n_q_heads, int n_kv_heads, int head_size,
    int q_token_count, const int* __restrict__ cache_position_dev,
    int q_size, int kv_size,
    int window, int capacity) {

	const int h    = blockIdx.x;
	const int t_q  = blockIdx.y;
	const int tid  = threadIdx.y * WARP_SIZE + threadIdx.x;

	const int kv_h = h * n_kv_heads / n_q_heads;
	const int cache_position = *cache_position_dev;

	// Slot of Q[t_q]'s own K row (last row in its causal range, plus 1 = exclusive
	// upper bound). Unified formula: pre-fill it's `cache_position + t_q`; once
	// the cache is at capacity (sliding steady state), it pins to
	// `capacity - q_token_count + t_q` after the host's shift.
	const int linear_slot = cache_position + t_q;
	const int pinned_slot = capacity - q_token_count + t_q;
	const int t_q_slot    = linear_slot < pinned_slot ? linear_slot : pinned_slot;
	const int t_k_max     = t_q_slot + 1;
	const int t_k_min     = (window != 0 && t_k_max > window) ? (t_k_max - window) : 0;

	const int q_base = t_q * q_size + h * D;
	const int o_base = t_q * q_size + h * D;

	const float scale = rsqrtf((float)D);

	const int kq_group_in_warp = threadIdx.x / NTHREADS_KQ;
	const int kq_lane_in_group = threadIdx.x & (NTHREADS_KQ - 1);

	// Q load.
	float2 Q_reg[D_PER_THREAD_KQ / 2];
	{
		const int q_uint_base = (q_base + kq_lane_in_group * D_PER_THREAD_KQ) >> 1;
		#pragma unroll
		for (int i = 0; i < D_PER_THREAD_KQ / 2; ++i) {
			unsigned int packed = __ldg(&q_buf[q_uint_base + i]);
			float a = bf16_uint16_to_float((unsigned short)(packed & 0xffffu)) * scale;
			float b = bf16_uint16_to_float((unsigned short)((packed >> 16) & 0xffffu)) * scale;
			Q_reg[i] = make_float2(a, b);
		}
	}

	float2 VKQ[HALF_D_PER_THREAD_V];
	#pragma unroll
	for (int i = 0; i < HALF_D_PER_THREAD_V; ++i) VKQ[i] = make_float2(0.0f, 0.0f);

	float KQ_max = -3.402823e38f;
	float KQ_sum = 0.0f;

	__shared__ float KQ[NTHREADS];

	const int kqs_warp_base = threadIdx.y * WARP_SIZE;

	int t_k0_start = (t_k_min / BC) * BC;
	for (int t_k0 = t_k0_start; t_k0 < t_k_max; t_k0 += BC) {
		float my_KQ = -3.402823e38f;
		float KQ_max_new = KQ_max;

		#pragma unroll
		for (int i_KQ_0 = 0; i_KQ_0 < NTHREADS_KQ; ++i_KQ_0) {
			const int i_KQ = kqs_warp_base + kq_group_in_warp * NTHREADS_KQ + i_KQ_0;
			const int t_k  = t_k0 + i_KQ;

			float sum = 0.0f;
			if (t_k < t_k_max && t_k >= t_k_min) {
				const int k_elem_base = t_k * kv_size + kv_h * D
				                       + kq_lane_in_group * D_PER_THREAD_KQ;
				const int k_uint_base = k_elem_base >> 1;

				#pragma unroll
				for (int i = 0; i < D_PER_THREAD_KQ / 2; ++i) {
					unsigned int packed = __ldg(&k_buf[k_uint_base + i]);
					float a = bf16_uint16_to_float((unsigned short)(packed & 0xffffu));
					float b = bf16_uint16_to_float((unsigned short)((packed >> 16) & 0xffffu));
					sum += Q_reg[i].x * a + Q_reg[i].y * b;
				}
			}
			#pragma unroll
			for (int off = NTHREADS_KQ / 2; off > 0; off >>= 1) {
				sum += __shfl_xor_sync(0xffffffffu, sum, off, NTHREADS_KQ);
			}
			if (kq_lane_in_group == i_KQ_0) {
				my_KQ = (t_k < t_k_max && t_k >= t_k_min) ? sum : -3.402823e38f;
			}
			KQ_max_new = fmaxf(KQ_max_new, (t_k < t_k_max && t_k >= t_k_min) ? sum : -3.402823e38f);
		}

		#pragma unroll
		for (int off = NTHREADS_KQ; off < WARP_SIZE; off <<= 1) {
			KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xffffffffu, KQ_max_new, off));
		}

		const float KQ_max_scale = expf(KQ_max - KQ_max_new);
		KQ_max = KQ_max_new;

		const float p = expf(my_KQ - KQ_max);
		KQ_sum = KQ_sum * KQ_max_scale + p;
		KQ[tid] = p;

		#pragma unroll
		for (int i = 0; i < HALF_D_PER_THREAD_V; ++i) {
			VKQ[i].x *= KQ_max_scale;
			VKQ[i].y *= KQ_max_scale;
		}

		__syncthreads();

		#pragma unroll
		for (int k0 = 0; k0 < WARP_SIZE; k0 += V_COLS_PER_ITER) {
			const int v_local = kqs_warp_base + k0 + threadIdx.x / NTHREADS_V;
			const int t_v     = t_k0 + v_local;
			float p_v = KQ[v_local];

			if (t_v < t_k_max && t_v >= t_k_min) {
				const int v_elem_base = t_v * kv_size + kv_h * D
				                       + (threadIdx.x % NTHREADS_V) * D_PER_THREAD_V;
				const int v_uint_base = v_elem_base >> 1;

				#pragma unroll
				for (int i = 0; i < HALF_D_PER_THREAD_V; ++i) {
					unsigned int packed = __ldg(&v_buf[v_uint_base + i]);
					float a = bf16_uint16_to_float((unsigned short)(packed & 0xffffu));
					float b = bf16_uint16_to_float((unsigned short)((packed >> 16) & 0xffffu));
					VKQ[i].x += p_v * a;
					VKQ[i].y += p_v * b;
				}
			}
		}

		__syncthreads();
	}

	__shared__ float KQ_max_w[NWARPS];
	__shared__ float KQ_sum_w[NWARPS];

	#pragma unroll
	for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
		KQ_sum = KQ_sum + __shfl_xor_sync(0xffffffffu, KQ_sum, off);
	}

	if (threadIdx.x == 0) {
		KQ_max_w[threadIdx.y] = KQ_max;
		KQ_sum_w[threadIdx.y] = KQ_sum;
	}
	__syncthreads();

	float final_max = KQ_max_w[0];
	#pragma unroll
	for (int w = 1; w < NWARPS; ++w) final_max = fmaxf(final_max, KQ_max_w[w]);

	const float warp_to_global_scale = expf(KQ_max - final_max);
	#pragma unroll
	for (int i = 0; i < HALF_D_PER_THREAD_V; ++i) {
		VKQ[i].x *= warp_to_global_scale;
		VKQ[i].y *= warp_to_global_scale;
	}

	float final_sum = 0.0f;
	#pragma unroll
	for (int w = 0; w < NWARPS; ++w) {
		final_sum += KQ_sum_w[w] * expf(KQ_max_w[w] - final_max);
	}

	const float inv_sum = 1.0f / final_sum;

	__shared__ float VKQ_shared[NWARPS][V_COLS_PER_ITER][D];
	const int v_col_group = threadIdx.x / NTHREADS_V;
	const int v_slice     = threadIdx.x % NTHREADS_V;
	const int slice_base  = v_slice * D_PER_THREAD_V;
	#pragma unroll
	for (int i = 0; i < HALF_D_PER_THREAD_V; ++i) {
		VKQ_shared[threadIdx.y][v_col_group][slice_base + 2 * i + 0] = VKQ[i].x;
		VKQ_shared[threadIdx.y][v_col_group][slice_base + 2 * i + 1] = VKQ[i].y;
	}
	__syncthreads();

	const int total_pairs = D / 2;
	#pragma unroll
	for (int p = tid; p < total_pairs; p += NTHREADS) {
		const int d_lo = 2 * p;
		const int d_hi = d_lo + 1;

		float sum_lo = 0.0f, sum_hi = 0.0f;
		#pragma unroll
		for (int w = 0; w < NWARPS; ++w) {
			#pragma unroll
			for (int g = 0; g < V_COLS_PER_ITER; ++g) {
				sum_lo += VKQ_shared[w][g][d_lo];
				sum_hi += VKQ_shared[w][g][d_hi];
			}
		}
		sum_lo *= inv_sum;
		sum_hi *= inv_sum;

		unsigned short lo = bf16_round(sum_lo);
		unsigned short hi = bf16_round(sum_hi);
		o[(o_base + d_lo) >> 1] = (unsigned int)lo | ((unsigned int)hi << 16);
	}
}
