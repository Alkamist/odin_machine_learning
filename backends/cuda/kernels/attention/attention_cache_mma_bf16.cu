// Tensor-core flash attention against the linear K/V cache. Specialized for
// our case: bf16 K/V, q_token_count = 1 (decode), gqa_ratio = 4 (Gemma E4B).
// One CUDA block handles one (KV head, Q token) pair and computes attention
// for the gqa_ratio = NCOLS2 Q heads that share the KV head.
//
// WMMA (m16n16k16, bf16 inputs, fp32 accumulator) for KQ and VKQ matmuls.
// Q is padded from NCOLS2=4 rows to the fragment's 16 rows; rows [4..16)
// hold zeros and are computed but discarded. Per-warp work split: each
// warp owns one 16-wide stripe of N (= K rows for KQ tile, = D cols for VKQ).
//
// FA2 online-softmax algorithm. Cache layout: linear (slot 0 oldest,
// slot cap-1 newest). t_q_slot pinned once cache fills:
//   t_q_slot = min(cache_position, capacity - q_token_count + t_q)
//   t_k_max  = t_q_slot + 1                              (causal)
//   t_k_min  = max(0, t_k_max - window)  if window>0, else 0
#include <cuda_bf16.h>

// __CUDA_AMPERE_MMA__ gates the bf16 / tf32 WMMA overloads in crt/mma.hpp.
// The header sets it from `__CUDA_ARCH__ >= 800`; under NVRTC with
// `--gpu-architecture=sm_86` __CUDA_ARCH__ is 860, but we set it explicitly
// here as belt-and-suspenders.
#define __CUDA_AMPERE_MMA__ 1
#include <mma.h>

#ifndef D_HEAD
#define D_HEAD 256
#endif

#define D                   D_HEAD
#define NCOLS2              4
#define BC                  64
#define WARP_SIZE           32
#define NWARPS              4
#define NTHREADS            (WARP_SIZE * NWARPS)
#define FRAG_M              16
#define FRAG_N              16
#define FRAG_K              16
#define K_FRAGS_PER_D       (D / FRAG_K)
#define BC_FRAGS_PER_TILE   (BC / FRAG_N)
#define VKQ_FRAGS_PER_D     (D / FRAG_N)
#define VKQ_FRAGS_PER_WARP  (VKQ_FRAGS_PER_D / NWARPS)

using namespace nvcuda;

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
void attention_cache_mma_bf16(
    const unsigned int* __restrict__ q_buf,
    const unsigned int* __restrict__ k_buf,
    const unsigned int* __restrict__ v_buf,
    unsigned int*       __restrict__ o,
    int n_q_heads, int n_kv_heads, int head_size,
    int q_token_count, const int* __restrict__ cache_position_dev,
    int q_size, int kv_size,
    int window, int capacity) {

	const int kv_h = blockIdx.x;
	const int t_q  = blockIdx.y;
	const int tid  = threadIdx.y * WARP_SIZE + threadIdx.x;
	const int warp = threadIdx.y;

	const int q_h_start = kv_h * NCOLS2;
	const int cache_position = *cache_position_dev;

	const int linear_slot = cache_position + t_q;
	const int pinned_slot = capacity - q_token_count + t_q;
	const int t_q_slot    = linear_slot < pinned_slot ? linear_slot : pinned_slot;
	const int t_k_max     = t_q_slot + 1;
	const int t_k_min     = (window != 0 && t_k_max > window) ? (t_k_max - window) : 0;

	const int q_token_offset = t_q * q_size;
	const int o_token_offset = t_q * q_size;

	const float scale = rsqrtf((float)D);

	// Q_sh padded to FRAG_M rows so the fragment load reads in-bounds. Rows
	// NCOLS2..FRAG_M-1 are zeroed below.
	__shared__ __nv_bfloat16 Q_sh[FRAG_M * D];
	__shared__ __nv_bfloat16 KV_sh[BC * D];
	__shared__ float          KQ_sh[FRAG_M * BC];        // padded rows like Q
	__shared__ __nv_bfloat16  P_sh[FRAG_M * BC];          // bf16 P for VKQ matmul
	__shared__ float          frag_scratch[NWARPS][FRAG_M * FRAG_N];
	__shared__ float          row_max_block[NCOLS2 * NWARPS];
	__shared__ float          row_sum_block[NCOLS2 * NWARPS];
	__shared__ float          VKQ_out_sh[NCOLS2][D];

	// Load Q into Q_sh, padding rows NCOLS2..FRAG_M-1 with zeros. Pre-scale
	// by 1/sqrt(D).
	for (int idx = tid; idx < FRAG_M * D; idx += NTHREADS) {
		int q_row = idx / D;
		int q_col = idx % D;
		__nv_bfloat16 val;
		if (q_row < NCOLS2) {
			int q_h  = q_h_start + q_row;
			int elem = q_token_offset + q_h * D + q_col;
			unsigned int packed = q_buf[elem >> 1];
			unsigned short bits = (elem & 1) ? (unsigned short)(packed >> 16) : (unsigned short)(packed & 0xffffu);
			float v = bf16_uint16_to_float(bits) * scale;
			val = __float2bfloat16(v);
		} else {
			val = __float2bfloat16(0.0f);
		}
		Q_sh[idx] = val;
	}
	__syncthreads();

	// Per-warp VKQ accumulators along D.
	wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float> VKQ_acc[VKQ_FRAGS_PER_WARP];
	#pragma unroll
	for (int f = 0; f < VKQ_FRAGS_PER_WARP; ++f) {
		wmma::fill_fragment(VKQ_acc[f], 0.0f);
	}

	// Per-row running stats. After each iter's reductions every thread holds
	// the same value here.
	float KQ_max[NCOLS2];
	float KQ_sum[NCOLS2];
	#pragma unroll
	for (int j = 0; j < NCOLS2; ++j) {
		KQ_max[j] = -3.402823e38f;
		KQ_sum[j] = 0.0f;
	}

	int t_k0_start = (t_k_min / BC) * BC;

	for (int t_k0 = t_k0_start; t_k0 < t_k_max; t_k0 += BC) {
		// Cooperative K load. Out-of-range t_k slots get bf16 zero (their KQ
		// scores are then masked to -inf below before exp).
		for (int idx = tid; idx < BC * D; idx += NTHREADS) {
			int k_row = idx / D;
			int k_col = idx % D;
			int t_k   = t_k0 + k_row;
			__nv_bfloat16 val;
			if (t_k < t_k_max && t_k >= t_k_min) {
				int elem = t_k * kv_size + kv_h * D + k_col;
				unsigned int packed = k_buf[elem >> 1];
				unsigned short bits = (elem & 1) ? (unsigned short)(packed >> 16) : (unsigned short)(packed & 0xffffu);
				val = *reinterpret_cast<const __nv_bfloat16*>(&bits);
			} else {
				val = __float2bfloat16(0.0f);
			}
			KV_sh[idx] = val;
		}
		__syncthreads();

		// KQ matmul: KQ[FRAG_M × BC] = Q[FRAG_M × D] @ K[BC × D]^T.
		// Each warp owns one FRAG_N stripe along BC. NWARPS=4, BC=64, FRAG_N=16
		// → exactly 4 stripes covered.
		const int n_offset = warp * FRAG_N;

		wmma::fragment<wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float> kq_frag;
		wmma::fill_fragment(kq_frag, 0.0f);

		#pragma unroll
		for (int k_iter = 0; k_iter < K_FRAGS_PER_D; ++k_iter) {
			wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, __nv_bfloat16, wmma::row_major> a_frag;
			wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, __nv_bfloat16, wmma::col_major> b_frag;
			wmma::load_matrix_sync(a_frag, Q_sh + k_iter * FRAG_K, D);
			wmma::load_matrix_sync(b_frag, KV_sh + n_offset * D + k_iter * FRAG_K, D);
			wmma::mma_sync(kq_frag, a_frag, b_frag, kq_frag);
		}

		// Store this warp's KQ stripe to KQ_sh at columns [n_offset..+FRAG_N).
		// store_matrix_sync writes the full FRAG_M × FRAG_N block; KQ_sh has
		// FRAG_M rows, BC cols, so ldc = BC and pointer offset = n_offset.
		wmma::store_matrix_sync(KQ_sh + n_offset, kq_frag, BC, wmma::mem_row_major);
		__syncthreads();

		// Mask out-of-range t_k positions to -inf so they vanish under exp.
		// Also compute per-row max contribution from this thread's elements.
		float row_new_max[NCOLS2];
		#pragma unroll
		for (int j = 0; j < NCOLS2; ++j) row_new_max[j] = KQ_max[j];

		for (int idx0 = tid; idx0 < NCOLS2 * BC; idx0 += NTHREADS) {
			int row = idx0 / BC;
			int col = idx0 % BC;
			int t_k = t_k0 + col;
			float v = KQ_sh[row * BC + col];
			if (!(t_k < t_k_max && t_k >= t_k_min)) {
				v = -3.402823e38f;
				KQ_sh[row * BC + col] = v;
			}
			row_new_max[row] = fmaxf(row_new_max[row], v);
		}

		// Reduce row_new_max across the block.
		#pragma unroll
		for (int j = 0; j < NCOLS2; ++j) {
			float v = row_new_max[j];
			#pragma unroll
			for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
				v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
			}
			if (threadIdx.x == 0) row_max_block[j * NWARPS + warp] = v;
		}
		__syncthreads();

		float new_max[NCOLS2];
		#pragma unroll
		for (int j = 0; j < NCOLS2; ++j) {
			float m = row_max_block[j * NWARPS + 0];
			#pragma unroll
			for (int w = 1; w < NWARPS; ++w) m = fmaxf(m, row_max_block[j * NWARPS + w]);
			new_max[j] = m;
		}

		// scale_old: factor to apply to existing accumulators (KQ_sum and VKQ_acc).
		float scale_old[NCOLS2];
		#pragma unroll
		for (int j = 0; j < NCOLS2; ++j) {
			scale_old[j] = expf(KQ_max[j] - new_max[j]);
		}

		// Compute p = exp(KQ - new_max) into KQ_sh. Sum per row.
		float row_new_sum[NCOLS2];
		#pragma unroll
		for (int j = 0; j < NCOLS2; ++j) row_new_sum[j] = 0.0f;

		for (int idx0 = tid; idx0 < NCOLS2 * BC; idx0 += NTHREADS) {
			int row = idx0 / BC;
			int col = idx0 % BC;
			float v = KQ_sh[row * BC + col];
			float p = (v > -3.0e38f) ? expf(v - new_max[row]) : 0.0f;
			KQ_sh[row * BC + col] = p;
			row_new_sum[row] += p;
		}

		// Reduce row_new_sum across the block.
		#pragma unroll
		for (int j = 0; j < NCOLS2; ++j) {
			float v = row_new_sum[j];
			#pragma unroll
			for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
				v += __shfl_xor_sync(0xffffffffu, v, off);
			}
			if (threadIdx.x == 0) row_sum_block[j * NWARPS + warp] = v;
		}
		__syncthreads();

		#pragma unroll
		for (int j = 0; j < NCOLS2; ++j) {
			float s = 0.0f;
			#pragma unroll
			for (int w = 0; w < NWARPS; ++w) s += row_sum_block[j * NWARPS + w];
			KQ_sum[j] = KQ_sum[j] * scale_old[j] + s;
			KQ_max[j] = new_max[j];
		}

		// Scale existing VKQ_acc fragments by scale_old (per-row).
		#pragma unroll
		for (int f = 0; f < VKQ_FRAGS_PER_WARP; ++f) {
			wmma::store_matrix_sync(&frag_scratch[warp][0], VKQ_acc[f], FRAG_N, wmma::mem_row_major);
			__syncwarp();
			for (int idx = threadIdx.x; idx < FRAG_M * FRAG_N; idx += WARP_SIZE) {
				int r = idx / FRAG_N;
				if (r < NCOLS2) {
					frag_scratch[warp][idx] *= scale_old[r];
				}
			}
			__syncwarp();
			wmma::load_matrix_sync(VKQ_acc[f], &frag_scratch[warp][0], FRAG_N, wmma::mem_row_major);
		}
		__syncthreads();

		// Cooperative V load (overwrites K in shared).
		for (int idx = tid; idx < BC * D; idx += NTHREADS) {
			int v_row = idx / D;
			int v_col = idx % D;
			int t_v   = t_k0 + v_row;
			__nv_bfloat16 val;
			if (t_v < t_k_max && t_v >= t_k_min) {
				int elem = t_v * kv_size + kv_h * D + v_col;
				unsigned int packed = v_buf[elem >> 1];
				unsigned short bits = (elem & 1) ? (unsigned short)(packed >> 16) : (unsigned short)(packed & 0xffffu);
				val = *reinterpret_cast<const __nv_bfloat16*>(&bits);
			} else {
				val = __float2bfloat16(0.0f);
			}
			KV_sh[idx] = val;
		}

		// Convert KQ_sh (fp32 p) to P_sh (bf16) padded to FRAG_M rows.
		for (int idx = tid; idx < FRAG_M * BC; idx += NTHREADS) {
			int row = idx / BC;
			int col = idx % BC;
			float v = (row < NCOLS2) ? KQ_sh[row * BC + col] : 0.0f;
			P_sh[idx] = __float2bfloat16(v);
		}
		__syncthreads();

		// VKQ matmul: VKQ[FRAG_M × D] += P[FRAG_M × BC] @ V[BC × D].
		#pragma unroll
		for (int f = 0; f < VKQ_FRAGS_PER_WARP; ++f) {
			const int d_offset = (warp * VKQ_FRAGS_PER_WARP + f) * FRAG_N;

			#pragma unroll
			for (int k_iter = 0; k_iter < BC_FRAGS_PER_TILE; ++k_iter) {
				wmma::fragment<wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, __nv_bfloat16, wmma::row_major> a_frag;
				wmma::fragment<wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, __nv_bfloat16, wmma::row_major> b_frag;
				wmma::load_matrix_sync(a_frag, P_sh + k_iter * FRAG_K, BC);
				wmma::load_matrix_sync(b_frag, KV_sh + (k_iter * FRAG_K) * D + d_offset, D);
				wmma::mma_sync(VKQ_acc[f], a_frag, b_frag, VKQ_acc[f]);
			}
		}
		__syncthreads();
	}

	// Write VKQ_acc fragments to VKQ_out_sh, normalizing by KQ_sum per row.
	#pragma unroll
	for (int f = 0; f < VKQ_FRAGS_PER_WARP; ++f) {
		const int d_offset = (warp * VKQ_FRAGS_PER_WARP + f) * FRAG_N;
		wmma::store_matrix_sync(&frag_scratch[warp][0], VKQ_acc[f], FRAG_N, wmma::mem_row_major);
		__syncwarp();
		for (int idx = threadIdx.x; idx < NCOLS2 * FRAG_N; idx += WARP_SIZE) {
			int r = idx / FRAG_N;
			int c = idx % FRAG_N;
			VKQ_out_sh[r][d_offset + c] = frag_scratch[warp][r * FRAG_N + c];
		}
	}
	__syncthreads();

	// Cooperative output write. Each thread covers (NCOLS2 * D / 2) / NTHREADS
	// pairs. KQ_sum[row] is identical across all threads after the iter
	// reductions, so we use the per-thread local copy directly.
	for (int idx = tid; idx < NCOLS2 * D / 2; idx += NTHREADS) {
		int row = idx / (D / 2);
		int pi  = idx % (D / 2);
		int q_h = q_h_start + row;
		float inv = 1.0f / KQ_sum[row];
		float a = VKQ_out_sh[row][2 * pi + 0] * inv;
		float b = VKQ_out_sh[row][2 * pi + 1] * inv;
		unsigned short lo = bf16_round(a);
		unsigned short hi = bf16_round(b);
		int o_pair = (o_token_offset + q_h * D + 2 * pi) >> 1;
		o[o_pair] = (unsigned int)lo | ((unsigned int)hi << 16);
	}
}
