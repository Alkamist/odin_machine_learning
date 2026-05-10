// Bf16 attention training backward, with F32 grads.
// Reads bf16 Q/K/V (forward inputs) and f32 d_out (output gradient), softmax
// matrix sm stays f32. Writes f32 d_q/d_k/d_v via native atomicAdd. Mirrors
// attention_train_back_f32 but with bf16 data reads.
#include <cuda_bf16.h>

#define BWG 256

__device__ __forceinline__ float load_bf16_at(const unsigned int* buf, int elem_idx) {
	int pair  = elem_idx >> 1;
	int shift = (elem_idx & 1) * 16;
	unsigned short bits = (unsigned short)((buf[pair] >> shift) & 0xffffu);
	return __bfloat162float(__ushort_as_bfloat16(bits));
}

extern "C" __global__
void attention_train_back_bf16(const unsigned int* __restrict__ q,
                               const unsigned int* __restrict__ k,
                               const unsigned int* __restrict__ v,
                               const float*        __restrict__ sm,
                               const float*        __restrict__ d_out,
                               float*              __restrict__ d_q,
                               float*              __restrict__ d_k,
                               float*              __restrict__ d_v,
                               int n_q_heads, int n_kv_heads, int head_size,
                               int token_count, int q_size, int kv_size,
                               int causal, int window) {
	int kv_h     = blockIdx.x;
	int gqa      = n_q_heads / n_kv_heads;
	int q_in_g   = blockIdx.y;
	int t_q      = blockIdx.z;
	int tid      = threadIdx.x;

	int h = kv_h * gqa + q_in_g;
	int T = token_count;
	int D = head_size;

	float inv_sqrt_d = rsqrtf((float)D);

	int t_k_max = (causal != 0) ? (t_q + 1) : T;
	int t_k_min = (window != 0 && t_k_max > window) ? (t_k_max - window) : 0;

	int q_offset  = t_q * q_size + h * D;
	int o_offset  = t_q * q_size + h * D;
	int sm_offset = h * T * T + t_q * T;

	__shared__ float reduction[BWG];
	__shared__ float d_p_row[1024];

	for (int t_k = tid; t_k < T; t_k += BWG) {
		if (t_k >= t_k_min && t_k < t_k_max) {
			int v_offset = t_k * kv_size + kv_h * D;
			float dp = 0.0f;
			float p_t = sm[sm_offset + t_k];
			for (int d = 0; d < D; ++d) {
				float dout_d = d_out[o_offset + d];
				dp += dout_d * load_bf16_at(v, v_offset + d);
				atomicAdd(&d_v[v_offset + d], p_t * dout_d);
			}
			d_p_row[t_k] = dp;
		} else {
			d_p_row[t_k] = 0.0f;
		}
	}
	__syncthreads();

	float local_sum = 0.0f;
	for (int t_k = tid; t_k < T; t_k += BWG) {
		local_sum += sm[sm_offset + t_k] * d_p_row[t_k];
	}
	reduction[tid] = local_sum;
	__syncthreads();
	for (int s = BWG / 2; s > 0; s >>= 1) {
		if (tid < s) reduction[tid] += reduction[tid + s];
		__syncthreads();
	}
	float dot_dp_p = reduction[0];

	for (int t_k = tid; t_k < T; t_k += BWG) {
		float p  = sm[sm_offset + t_k];
		float dp = d_p_row[t_k];
		d_p_row[t_k] = p * (dp - dot_dp_p) * inv_sqrt_d;
	}
	__syncthreads();

	for (int d = tid; d < D; d += BWG) {
		float acc = 0.0f;
		for (int t_k = t_k_min; t_k < t_k_max; ++t_k) {
			int k_offset = t_k * kv_size + kv_h * D;
			acc += d_p_row[t_k] * load_bf16_at(k, k_offset + d);
		}
		atomicAdd(&d_q[q_offset + d], acc);
	}

	for (int t_k = tid; t_k < T; t_k += BWG) {
		if (t_k >= t_k_min && t_k < t_k_max) {
			int k_offset = t_k * kv_size + kv_h * D;
			float dl = d_p_row[t_k];
			for (int d = 0; d < D; ++d) {
				atomicAdd(&d_k[k_offset + d], dl * load_bf16_at(q, q_offset + d));
			}
		}
	}
}
