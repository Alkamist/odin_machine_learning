// Fused FFN front-half: y[n] = gelu_tanh(gate[n]) * up[n], where
//   gate[n] = sum_k x[k] * w_gate[n,k]      (Q4_K x Q8_1 mmvq)
//   up[n]   = sum_k x[k] * w_up  [n,k]
//
// Both matmuls share the same q8_1-quantized activation `x`, so the inner
// loop loads each q8_1 block once and uses it for both dot products. Mirrors
// ggml's `ffn_up + ffn_gate + glu` fusion in `mmvq.cu`. M=1 decode only;
// prefill (multi-token) keeps the unfused coopmat path.
#include <cuda_fp16.h>

#define GELU_SCALE 0.7978845608028654f

__device__ __forceinline__ float gelu_tanh(float v) {
	float cube = 0.044715f * v * v * v;
	return 0.5f * v * (1.0f + tanhf(GELU_SCALE * (v + cube)));
}

__device__ __forceinline__ void unpack_scale_min(int sub,
                                                 unsigned int sw0,
                                                 unsigned int sw1,
                                                 unsigned int sw2,
                                                 int& scale,
                                                 int& min_v) {
	if (sub < 4) {
		int shift = sub * 8;
		scale = (int)((sw0 >> shift) & 0x3Fu);
		min_v = (int)((sw1 >> shift) & 0x3Fu);
	} else {
		int shift = (sub - 4) * 8;
		unsigned int b_low = (sw2 >> shift) & 0xFFu;
		unsigned int b_sc  = (sw0 >> shift) & 0xFFu;
		unsigned int b_mn  = (sw1 >> shift) & 0xFFu;
		scale = (int)((b_low & 0x0Fu) | ((b_sc >> 6) << 4));
		min_v = (int)((b_low >>  4)  | ((b_mn >> 6) << 4));
	}
}

__device__ __forceinline__ unsigned int bf16_from_f32(float v) {
	unsigned int bits = __float_as_uint(v);
	if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;  // NaN
	unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
	return (rounded >> 16) & 0xffffu;
}

#define ROWS_PER_WG 2
#define NWARPS      4
#define WARP_SIZE   32
#define BLOCK_UINTS 36   // Q4_K block: 144 / 4 uints

extern "C" __global__
__launch_bounds__(NWARPS * WARP_SIZE, 1)
void linear_q4_k_gate_up_geglu_bf16(const unsigned int* __restrict__ x,        // q8_1 stream
                                    const unsigned int* __restrict__ w_gate,   // Q4_K
                                    const unsigned int* __restrict__ w_up,     // Q4_K
                                    unsigned int*       __restrict__ y,        // bf16 packed pairs
                                    int M, int K, int N) {
	int n_base = blockIdx.x * ROWS_PER_WG;
	int wid    = threadIdx.y;
	int lane   = threadIdx.x;

	int num_blocks = K / 256;

	float acc_gate[ROWS_PER_WG];
	float acc_up  [ROWS_PER_WG];
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) { acc_gate[r] = 0.0f; acc_up[r] = 0.0f; }

	int t_sub_0 = lane >> 3;
	int t_inner = lane & 7;
	int t_sub_1 = t_sub_0 + 4;

	for (int b = wid; b < num_blocks; b += NWARPS) {
		int q8_block_base = b * 8;

		int          q8_packed_0 = (int)__ldg(&x[(q8_block_base + t_sub_0) * 9 + 1 + t_inner]);
		int          q8_packed_1 = (int)__ldg(&x[(q8_block_base + t_sub_1) * 9 + 1 + t_inner]);
		unsigned int q8_ds_0 = __ldg(&x[(q8_block_base + t_sub_0) * 9]);
		unsigned int q8_ds_1 = __ldg(&x[(q8_block_base + t_sub_1) * 9]);
		float q8_d_0 = __half2float(__ushort_as_half((unsigned short)(q8_ds_0 & 0xffff)));
		float q8_s_0 = __half2float(__ushort_as_half((unsigned short)((q8_ds_0 >> 16) & 0xffff)));
		float q8_d_1 = __half2float(__ushort_as_half((unsigned short)(q8_ds_1 & 0xffff)));
		float q8_s_1 = __half2float(__ushort_as_half((unsigned short)((q8_ds_1 >> 16) & 0xffff)));

		#pragma unroll
		for (int r = 0; r < ROWS_PER_WG; ++r) {
			int n = n_base + r;
			if (n >= N) continue;

			int block_off = (n * num_blocks + b) * BLOCK_UINTS;

			// Two passes: same code shape as the single-matmul kernel, run once
			// per weight tensor. Loop body identical to linear_q4_k_mmvq; only
			// the destination accumulator differs.
			#pragma unroll
			for (int side = 0; side < 2; ++side) {
				const unsigned int * __restrict__ w = (side == 0) ? w_gate : w_up;

				uint4 hdr = __ldg(reinterpret_cast<const uint4*>(&w[block_off]));
				unsigned int dm = hdr.x;
				float d    = __half2float(__ushort_as_half((unsigned short)(dm & 0xffff)));
				float dmin = __half2float(__ushort_as_half((unsigned short)((dm >> 16) & 0xffff)));
				unsigned int sw0 = hdr.y;
				unsigned int sw1 = hdr.z;
				unsigned int sw2 = hdr.w;

				float row_acc = 0.0f;
				{
					unsigned int nib_word = __ldg(&w[block_off + 4 + (t_sub_0 >> 1) * 8 + t_inner]);
					int shift     = (t_sub_0 & 1) * 4;
					int q4_packed = (int)((nib_word >> shift) & 0x0F0F0F0Fu);
					int int_dot   = __dp4a(q4_packed, q8_packed_0, 0);

					int scale, min_v;
					unpack_scale_min(t_sub_0, sw0, sw1, sw2, scale, min_v);

					row_acc += d * (float)scale * q8_d_0 * (float)int_dot;
					if (t_inner == 0) row_acc -= dmin * (float)min_v * q8_s_0;
				}
				{
					unsigned int nib_word = __ldg(&w[block_off + 4 + (t_sub_1 >> 1) * 8 + t_inner]);
					int shift     = (t_sub_1 & 1) * 4;
					int q4_packed = (int)((nib_word >> shift) & 0x0F0F0F0Fu);
					int int_dot   = __dp4a(q4_packed, q8_packed_1, 0);

					int scale, min_v;
					unpack_scale_min(t_sub_1, sw0, sw1, sw2, scale, min_v);

					row_acc += d * (float)scale * q8_d_1 * (float)int_dot;
					if (t_inner == 0) row_acc -= dmin * (float)min_v * q8_s_1;
				}

				if (side == 0) acc_gate[r] += row_acc;
				else           acc_up  [r] += row_acc;
			}
		}
	}

	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) {
		#pragma unroll
		for (int off = 16; off > 0; off >>= 1) {
			acc_gate[r] += __shfl_xor_sync(0xffffffffu, acc_gate[r], off);
			acc_up  [r] += __shfl_xor_sync(0xffffffffu, acc_up  [r], off);
		}
	}

	__shared__ float warp_sums_gate[NWARPS][ROWS_PER_WG];
	__shared__ float warp_sums_up  [NWARPS][ROWS_PER_WG];
	if (lane == 0) {
		#pragma unroll
		for (int r = 0; r < ROWS_PER_WG; ++r) {
			warp_sums_gate[wid][r] = acc_gate[r];
			warp_sums_up  [wid][r] = acc_up  [r];
		}
	}
	__syncthreads();

	if (wid != 0 || lane != 0) return;

	float final_gate[ROWS_PER_WG];
	float final_up  [ROWS_PER_WG];
	#pragma unroll
	for (int r = 0; r < ROWS_PER_WG; ++r) {
		float sg = 0.0f, su = 0.0f;
		#pragma unroll
		for (int w = 0; w < NWARPS; ++w) {
			sg += warp_sums_gate[w][r];
			su += warp_sums_up  [w][r];
		}
		final_gate[r] = sg;
		final_up  [r] = su;
	}

	int n_lo = n_base;
	if (n_lo >= N) return;
	int n_hi = n_lo + 1;

	float v_lo = gelu_tanh(final_gate[0]) * final_up[0];
	float v_hi = (n_hi < N) ? gelu_tanh(final_gate[1]) * final_up[1] : 0.0f;
	unsigned int lo = bf16_from_f32(v_lo);
	unsigned int hi = (n_hi < N) ? bf16_from_f32(v_hi) : 0u;
	y[n_lo >> 1] = (hi << 16) | lo;
}
