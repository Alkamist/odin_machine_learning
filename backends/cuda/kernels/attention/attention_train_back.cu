#ifdef DTYPE_BF16
#include "bf16.cuh"
#define DATA_T unsigned short
#define RD(p, i) ld_bf16(p, i)
#define KERNEL_NAME attention_train_back_bf16
#else
#define DATA_T float
#define RD(p, i) (p[i])
#define KERNEL_NAME attention_train_back_f32
#endif

#define BWG 256

extern "C" __global__
void KERNEL_NAME(const DATA_T* __restrict__ q,
                 const DATA_T* __restrict__ k,
                 const DATA_T* __restrict__ v,
                 const float*  __restrict__ sm,
                 const float*  __restrict__ d_out,
                 float*        __restrict__ d_q,
                 float*        __restrict__ d_k,
                 float*        __restrict__ d_v,
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
	__shared__ float d_p_row[2048];

	for (int t_k = tid; t_k < T; t_k += BWG) {
		if (t_k >= t_k_min && t_k < t_k_max) {
			int v_offset = t_k * kv_size + kv_h * D;
			float dp = 0.0f;
			float p_t = sm[sm_offset + t_k];
			for (int d = 0; d < D; ++d) {
				float dout_d = d_out[o_offset + d];
				dp += dout_d * RD(v, v_offset + d);
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
			acc += d_p_row[t_k] * RD(k, k_offset + d);
		}
		atomicAdd(&d_q[q_offset + d], acc);
	}

	for (int t_k = tid; t_k < T; t_k += BWG) {
		if (t_k >= t_k_min && t_k < t_k_max) {
			int k_offset = t_k * kv_size + kv_h * D;
			float dl = d_p_row[t_k];
			for (int d = 0; d < D; ++d) {
				atomicAdd(&d_k[k_offset + d], dl * RD(q, q_offset + d));
			}
		}
	}
}
