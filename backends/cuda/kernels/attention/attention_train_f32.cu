// F32 attention forward for training. Materialises softmax_outputs
// (saved per (head, q_token) row) so the backward pass can reuse it.
// Standard scaled-dot-product attention; not flash. One block per
// (head, q_token), 256 threads per block. Suitable for small training
// sequences (shakespeare/tinystories). For very long sequences, switch
// to a flash variant.
//
// Loss in precision vs flash is fine here: backward needs the materialised
// softmax row anyway.

#define ATT_WG 256

extern "C" __global__
void attention_train_f32(const float* __restrict__ q,        // [T, n_q_heads*D]
                         const float* __restrict__ k,        // [T, n_kv_heads*D]
                         const float* __restrict__ v,        // [T, n_kv_heads*D]
                         float*       __restrict__ out,       // [T, n_q_heads*D]
                         float*       __restrict__ sm,        // [n_q_heads, T, T]
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

	// Pass 1: scores into sm (unnormalised). Threads cooperate over t_k.
	float local_max = -3.402823e38f;
	for (int t_k = tid; t_k < T; t_k += ATT_WG) {
		float score;
		if (t_k >= t_k_min && t_k < t_k_max) {
			int k_offset = t_k * kv_size + kv_h * D;
			float dot = 0.0f;
			for (int d = 0; d < D; ++d) {
				dot += q[q_offset + d] * k[k_offset + d];
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

	// Pass 2: exp(score - max), sum.
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

	// Pass 3: output = sum_k sm[k] * V[k]. Threads cooperate over D.
	for (int d = tid; d < D; d += ATT_WG) {
		float acc = 0.0f;
		for (int t_k = t_k_min; t_k < t_k_max; ++t_k) {
			int v_offset = t_k * kv_size + kv_h * D;
			acc += sm[sm_offset + t_k] * v[v_offset + d];
		}
		out[o_offset + d] = acc;
	}
}
