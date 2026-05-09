// Fused FFN front-half over Q4_K weights: y[n] = gelu_tanh(gate[n]) * up[n]
// where gate[n] = Σ_k x[k] * w_gate[n,k] and up[n] = Σ_k x[k] * w_up[n,k].
// Both matmuls share the q8_1-quantized activation `x`. Mirrors ggml's
// `mul_mat_vec_q` with `has_fusion=true, use_gate=true, glu_op=GEGLU`
// (mmvq.cu) — same outer + inner structure as `linear_q4_k_mmvq.cu`, with
// a parallel `tmp_gate` accumulator alongside the regular `tmp` accumulator,
// and a final `gelu_tanh(gate) * up` combine before writing fp32 output.
// The downstream `pack_f32_to_bf16_pairs` kernel converts to bf16.
#include <cuda_fp16.h>

#define QK4_K            256
#define QI4_K_QS         32
#define BQ4_K_UINTS      36
#define QK8_1            32
#define QI8_1            8
#define BQ8_1_UINTS      9
#define QR4_K            2
#define VDR_Q4_K_Q8_1    2

#define NWARPS           4
#define WARP_SIZE        32
#define ROWS_PER_BLOCK   1
#define BLOCKS_PER_ITER  (VDR_Q4_K_Q8_1 * NWARPS * WARP_SIZE / QI4_K_QS)  // = 8

#define GELU_SCALE       0.7978845608028654f

__device__ __forceinline__ float gelu_tanh(float v) {
	float cube = 0.044715f * v * v * v;
	return 0.5f * v * (1.0f + tanhf(GELU_SCALE * (v + cube)));
}

__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl(
    const int * __restrict__ v, const int * __restrict__ u,
    const unsigned char * __restrict__ sc, const unsigned char * __restrict__ m,
    unsigned int dm_packed, const float * __restrict__ d8) {

	float sumf_d = 0.0f;
	float sumf_m = 0.0f;

	#pragma unroll
	for (int i = 0; i < QR4_K; ++i) {
		const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
		const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

		const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
		const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));

		sumf_d += d8[i] * (dot1 * sc[i]);
		sumf_m += d8[i] * (dot2 * m[i]);
	}

	float d    = __half2float(__ushort_as_half((unsigned short)(dm_packed & 0xffffu)));
	float dmin = __half2float(__ushort_as_half((unsigned short)((dm_packed >> 16) & 0xffffu)));
	return d * sumf_d - dmin * sumf_m;
}

// vec_dot specialized to take pre-loaded u/d8 (shared across the gate and up
// dot products since both matmuls use the same q8_1 input). Returns the dot
// for a single Q4_K weight matrix slice.
__device__ __forceinline__ float vec_dot_q4_K_with_u(
    const unsigned int * __restrict__ vx_row, int kbx, int bq8_offset, int idx,
    const int * __restrict__ u, const float * __restrict__ d8) {

	const unsigned int * bq4_K = vx_row + kbx * BQ4_K_UINTS;

	int v[2];
	const unsigned int * q4 = bq4_K + 4 + 4 * bq8_offset + idx;
	v[0] = (int)__ldg(q4 + 0);
	v[1] = (int)__ldg(q4 + 4);

	const unsigned short * scales = (const unsigned short *)(bq4_K + 1);
	unsigned short aux[2];
	const int j = bq8_offset / 2;
	if (j < 2) {
		aux[0] = scales[j + 0] & 0x3f3f;
		aux[1] = scales[j + 2] & 0x3f3f;
	} else {
		aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
		aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
	}
	const unsigned char * sc = (const unsigned char *)aux;
	const unsigned char * m  = sc + 2;

	return vec_dot_q4_K_q8_1_impl(v, u, sc, m, __ldg(bq4_K + 0), d8);
}

extern "C" __global__
__launch_bounds__(NWARPS * WARP_SIZE, 1)
void linear_q4_k_gate_up_geglu_bf16(
    const unsigned int * __restrict__ vy,         // q8_1 input row
    const unsigned int * __restrict__ vx_gate,    // q4_k gate weights, [N, K]
    const unsigned int * __restrict__ vx_up,      // q4_k up weights, [N, K]
    float *              __restrict__ dst,         // fp32 output [N]
    int M, int K, int N) {

	const int tid  = WARP_SIZE * threadIdx.y + threadIdx.x;
	const int row0 = ROWS_PER_BLOCK * blockIdx.x;
	const int blocks_per_row = K / QK4_K;

	const unsigned int * vy_row = vy;

	float tmp_gate = 0.0f;
	float tmp_up   = 0.0f;

	for (int kbx = tid / (QI4_K_QS / VDR_Q4_K_Q8_1); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
		const int kby = kbx * (QK4_K / QK8_1);
		const int kqs = VDR_Q4_K_Q8_1 * (tid % (QI4_K_QS / VDR_Q4_K_Q8_1));
		const int bq8_offset = QR4_K * ((kqs / 2) / (QI8_1 / 2));
		const int idx        = (kqs / 2) % 4;

		// Shared q8_1 loads: u[0..3] and d8[0..1] are independent of which
		// weight matrix we're dotting against.
		int   u[2 * QR4_K];
		float d8[QR4_K];
		const unsigned int * vy_kby = vy_row + kby * BQ8_1_UINTS;
		#pragma unroll
		for (int i = 0; i < QR4_K; ++i) {
			const unsigned int * bq8i = vy_kby + (bq8_offset + i) * BQ8_1_UINTS;
			unsigned int ds = __ldg(bq8i + 0);
			d8[i] = __half2float(__ushort_as_half((unsigned short)(ds & 0xffffu)));
			const unsigned int * q8 = bq8i + 1 + idx;
			u[2 * i + 0] = (int)__ldg(q8 + 0);
			u[2 * i + 1] = (int)__ldg(q8 + 4);
		}

		const unsigned int * vx_gate_row = vx_gate + (size_t)row0 * blocks_per_row * BQ4_K_UINTS;
		const unsigned int * vx_up_row   = vx_up   + (size_t)row0 * blocks_per_row * BQ4_K_UINTS;

		tmp_gate += vec_dot_q4_K_with_u(vx_gate_row, kbx, bq8_offset, idx, u, d8);
		tmp_up   += vec_dot_q4_K_with_u(vx_up_row,   kbx, bq8_offset, idx, u, d8);
	}

	__shared__ float tmp_shared_gate[NWARPS - 1][WARP_SIZE];
	__shared__ float tmp_shared_up  [NWARPS - 1][WARP_SIZE];
	if (threadIdx.y > 0) {
		tmp_shared_gate[threadIdx.y - 1][threadIdx.x] = tmp_gate;
		tmp_shared_up  [threadIdx.y - 1][threadIdx.x] = tmp_up;
	}
	__syncthreads();
	if (threadIdx.y > 0) return;

	#pragma unroll
	for (int l = 0; l < NWARPS - 1; ++l) {
		tmp_gate += tmp_shared_gate[l][threadIdx.x];
		tmp_up   += tmp_shared_up  [l][threadIdx.x];
	}
	#pragma unroll
	for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
		tmp_gate += __shfl_xor_sync(0xffffffffu, tmp_gate, off);
		tmp_up   += __shfl_xor_sync(0xffffffffu, tmp_up,   off);
	}

	if (threadIdx.x == 0 && row0 < N) {
		dst[row0] = gelu_tanh(tmp_gate) * tmp_up;
	}
}
