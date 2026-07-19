#ifdef DTYPE_BF16
#include "bf16.cuh"
#define DATA_T unsigned short
#define RD(p, i) ld_bf16(p, i)
#define WR(p, i, val) st_bf16(p, i, (val))
#define KERNEL_NAME attention_train_bf16
#else
#define DATA_T float
#define RD(p, i) (p[i])
#define WR(p, i, val) do { (p)[i] = (val); } while (0)
#define KERNEL_NAME attention_train_f32
#endif

#define ATT_WG 256

extern "C" __global__
void KERNEL_NAME(const DATA_T* __restrict__ q,
                 const DATA_T* __restrict__ k,
                 const DATA_T* __restrict__ v,
                 DATA_T*       __restrict__ out,
                 float*        __restrict__ sm,
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
				dot += RD(q, q_offset + d) * RD(k, k_offset + d);
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

	for (int d = tid; d < D; d += ATT_WG) {
		float acc = 0.0f;
		for (int t_k = t_k_min; t_k < t_k_max; ++t_k) {
			int v_offset = t_k * kv_size + kv_h * D;
			acc += sm[sm_offset + t_k] * RD(v, v_offset + d);
		}
		WR(out, o_offset + d, acc);
	}
}
