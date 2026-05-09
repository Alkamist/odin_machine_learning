// Port of `attention_cache_vec_bf16.cu` with fp32 Q input and fp32 output.
// K/V cache stays bf16 (the model's stored cache dtype). 16-byte LDG.E.128
// for K/V reads. Output is written as fp32, eliminating the bf16-pack
// round-trip on the attention's hot path.
//
// Cache layout, per-Q-token live K range, and shared-memory layout are
// identical to the bf16 version — only the Q load and the final dst writes
// change type.
#include <cuda_bf16.h>

#ifndef D_HEAD
#define D_HEAD 256
#endif

#define D            D_HEAD
#define WARP_SIZE    32
#define NWARPS       4
#define NTHREADS     (WARP_SIZE * NWARPS)
#define NTHREADS_KQ  8
#define NTHREADS_V   8
#define V_COLS_PER_ITER (WARP_SIZE / NTHREADS_V)
#define BC           NTHREADS
#define D_PER_THREAD_KQ (D / NTHREADS_KQ)
#define D_PER_THREAD_V  (D / NTHREADS_V)
#define HALF_D_PER_THREAD_V (D_PER_THREAD_V / 2)
#define PAIRS_PER_LDG   4
#define KQ_LDG_ITERS    (D_PER_THREAD_KQ / (2 * PAIRS_PER_LDG))
#define V_LDG_ITERS     (HALF_D_PER_THREAD_V / PAIRS_PER_LDG)

__device__ __forceinline__ float bf16_uint16_to_float(unsigned short bits) {
	unsigned int as_u32 = (unsigned int)bits << 16;
	return __int_as_float((int)as_u32);
}

extern "C" __global__
__launch_bounds__(NTHREADS, 1)
void attention_cache_vec_f32(
    const float*        __restrict__ q_buf,    // fp32 Q [tokens, n_q_heads * D]
    const unsigned int* __restrict__ k_buf,    // bf16 packed pairs (cache)
    const unsigned int* __restrict__ v_buf,    // bf16 packed pairs (cache)
    float*              __restrict__ o,         // fp32 output [tokens, n_q_heads * D]
    int n_q_heads, int n_kv_heads, int head_size,
    int q_token_count, const int* __restrict__ cache_position_dev,
    int q_size, int kv_size,
    int window, int capacity) {

	const int h    = blockIdx.x;
	const int t_q  = blockIdx.y;
	const int tid  = threadIdx.y * WARP_SIZE + threadIdx.x;

	const int kv_h = h * n_kv_heads / n_q_heads;
	const int cache_position = *cache_position_dev;

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

	// Q load: 16-byte LDG of 4 fp32 elements per LDG = float4 → 4 elements/thread/iter.
	// Total per thread = D_PER_THREAD_KQ floats.
	float2 Q_reg[D_PER_THREAD_KQ / 2];
	{
		const int q_elem_base = q_base + kq_lane_in_group * D_PER_THREAD_KQ;
		#pragma unroll
		for (int i = 0; i < D_PER_THREAD_KQ / 4; ++i) {
			float4 v = __ldg(reinterpret_cast<const float4*>(&q_buf[q_elem_base + i * 4]));
			Q_reg[i * 2 + 0] = make_float2(v.x * scale, v.y * scale);
			Q_reg[i * 2 + 1] = make_float2(v.z * scale, v.w * scale);
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
				for (int i = 0; i < KQ_LDG_ITERS; ++i) {
					uint4 packed = __ldg(reinterpret_cast<const uint4*>(&k_buf[k_uint_base + i * PAIRS_PER_LDG]));
					unsigned int p[PAIRS_PER_LDG] = { packed.x, packed.y, packed.z, packed.w };
					#pragma unroll
					for (int j = 0; j < PAIRS_PER_LDG; ++j) {
						float a = bf16_uint16_to_float((unsigned short)(p[j] & 0xffffu));
						float b = bf16_uint16_to_float((unsigned short)((p[j] >> 16) & 0xffffu));
						const int reg = i * PAIRS_PER_LDG + j;
						sum += Q_reg[reg].x * a + Q_reg[reg].y * b;
					}
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
				for (int i = 0; i < V_LDG_ITERS; ++i) {
					uint4 packed = __ldg(reinterpret_cast<const uint4*>(&v_buf[v_uint_base + i * PAIRS_PER_LDG]));
					unsigned int p_arr[PAIRS_PER_LDG] = { packed.x, packed.y, packed.z, packed.w };
					#pragma unroll
					for (int j = 0; j < PAIRS_PER_LDG; ++j) {
						float a = bf16_uint16_to_float((unsigned short)(p_arr[j] & 0xffffu));
						float b = bf16_uint16_to_float((unsigned short)((p_arr[j] >> 16) & 0xffffu));
						const int reg = i * PAIRS_PER_LDG + j;
						VKQ[reg].x += p_v * a;
						VKQ[reg].y += p_v * b;
					}
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

	#pragma unroll
	for (int d = tid; d < D; d += NTHREADS) {
		float s = 0.0f;
		#pragma unroll
		for (int w = 0; w < NWARPS; ++w) {
			#pragma unroll
			for (int g = 0; g < V_COLS_PER_ITER; ++g) {
				s += VKQ_shared[w][g][d];
			}
		}
		o[o_base + d] = s * inv_sum;
	}
}
