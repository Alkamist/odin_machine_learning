// F32 attention backward for training. Mirrors the CPU implementation in
// `attention_backward_f32` (backends/cpu/cpu.odin). Uses the saved
// per-row softmax probabilities to compute dQ, dK, dV.
//
// Per (kv_head) block we walk all q heads in the GQA group and all q tokens.
// Inside, we accumulate into dQ, dK, dV with atomicAdds where multiple
// threads/blocks may write the same K or V element.
//
// Notes on the math (single q-row):
//   p[k]  = sm_row[k]                                       (softmax outputs)
//   d_o   = grad_output[t_q, h, :]                           ([D])
//   d_v[k]+= p[k] * d_o                                       (V gradient)
//   d_p[k] = dot(d_o, V[k])
//   dot_dp_p = sum_k p[k] * d_p[k]
//   d_logit[k] = p[k] * (d_p[k] - dot_dp_p) * inv_sqrt_d
//   d_q   += sum_k d_logit[k] * K[k]
//   d_k[k]+= d_logit[k] * Q
//
// One block per (kv_head, q_head_in_group). Threads parallelise over (t_q, k).

extern "C" __global__
void attention_train_back_f32(const float* __restrict__ q,        // [T, n_q_heads*D]
                              const float* __restrict__ k,        // [T, n_kv_heads*D]
                              const float* __restrict__ v,        // [T, n_kv_heads*D]
                              const float* __restrict__ sm,       // [n_q_heads, T, T]
                              const float* __restrict__ d_out,    // [T, n_q_heads*D]
                              float*       __restrict__ d_q,      // [T, n_q_heads*D]
                              float*       __restrict__ d_k,      // [T, n_kv_heads*D]
                              float*       __restrict__ d_v,      // [T, n_kv_heads*D]
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

	__shared__ float reduction[256];

	// First, accumulate dV[k] += p[k] * d_o, and per-k compute d_p[k] = dot(d_o, V[k]).
	// We loop over k cooperatively so each thread handles a subset.
	// d_p stored in shared so all threads can compute the row reduction.
	__shared__ float d_p_row[1024];  // assumes T <= 1024

	for (int t_k = tid; t_k < T; t_k += 256) {
		if (t_k >= t_k_min && t_k < t_k_max) {
			int v_offset = t_k * kv_size + kv_h * D;
			float dp = 0.0f;
			for (int d = 0; d < D; ++d) {
				float dout_d = d_out[o_offset + d];
				dp += dout_d * v[v_offset + d];
				atomicAdd(&d_v[v_offset + d], sm[sm_offset + t_k] * dout_d);
			}
			d_p_row[t_k] = dp;
		} else {
			d_p_row[t_k] = 0.0f;
		}
	}
	__syncthreads();

	// dot_dp_p = sum_k p[k] * d_p[k]
	float local_sum = 0.0f;
	for (int t_k = tid; t_k < T; t_k += 256) {
		local_sum += sm[sm_offset + t_k] * d_p_row[t_k];
	}
	reduction[tid] = local_sum;
	__syncthreads();
	for (int s = 128; s > 0; s >>= 1) {
		if (tid < s) reduction[tid] += reduction[tid + s];
		__syncthreads();
	}
	float dot_dp_p = reduction[0];

	// d_logit[k] = p[k] * (d_p[k] - dot_dp_p) * inv_sqrt_d
	for (int t_k = tid; t_k < T; t_k += 256) {
		float p = sm[sm_offset + t_k];
		float dp = d_p_row[t_k];
		d_p_row[t_k] = p * (dp - dot_dp_p) * inv_sqrt_d;
	}
	__syncthreads();

	// dQ += sum_k d_logit[k] * K[k]. Cooperate over D.
	for (int d = tid; d < D; d += 256) {
		float acc = 0.0f;
		for (int t_k = t_k_min; t_k < t_k_max; ++t_k) {
			int k_offset = t_k * kv_size + kv_h * D;
			acc += d_p_row[t_k] * k[k_offset + d];
		}
		atomicAdd(&d_q[q_offset + d], acc);
	}

	// dK[k] += d_logit[k] * Q. Cooperate over k.
	for (int t_k = tid; t_k < T; t_k += 256) {
		if (t_k >= t_k_min && t_k < t_k_max) {
			int k_offset = t_k * kv_size + kv_h * D;
			float dl = d_p_row[t_k];
			for (int d = 0; d < D; ++d) {
				atomicAdd(&d_k[k_offset + d], dl * q[q_offset + d]);
			}
		}
	}
}
