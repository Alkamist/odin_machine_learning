// Cross-entropy forward, per-sample softmax + NLL.
// One block per sample. Threads cooperate on the row reduction
// (max + sum-exp). Stores the per-class softmax probabilities in
// `probabilities` (used by the backward pass) and the per-sample loss
// in `output`.
//
// Loss at sample s with target t:
//   loss[s] = -input[s, t] + max + log(sum_i exp(input[s, i] - max))

extern "C" __global__
void cross_entropy_f32(const float* __restrict__ x,         // [n_samples, class_size]
                       const int*   __restrict__ targets,    // [n_samples]
                       float*       __restrict__ probs,      // [n_samples, class_size]
                       float*       __restrict__ loss,       // [n_samples]
                       int class_size) {
	int sample = blockIdx.x;
	int tid    = threadIdx.x;

	const float* row_x = x     + sample * class_size;
	float*       row_p = probs + sample * class_size;

	__shared__ float reduction[256];

	// Row max.
	float local_max = -3.402823e38f;
	for (int i = tid; i < class_size; i += blockDim.x) {
		float v = row_x[i];
		if (v > local_max) local_max = v;
	}
	reduction[tid] = local_max;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) reduction[tid] = fmaxf(reduction[tid], reduction[tid + s]);
		__syncthreads();
	}
	float row_max = reduction[0];

	// Row sum-exp.
	float local_sum = 0.0f;
	for (int i = tid; i < class_size; i += blockDim.x) {
		float e = expf(row_x[i] - row_max);
		row_p[i] = e;
		local_sum += e;
	}
	reduction[tid] = local_sum;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) reduction[tid] += reduction[tid + s];
		__syncthreads();
	}
	float row_sum = reduction[0];
	float inv_sum = 1.0f / row_sum;

	// Normalize probabilities.
	for (int i = tid; i < class_size; i += blockDim.x) {
		row_p[i] *= inv_sum;
	}

	// Per-sample loss in the first thread.
	if (tid == 0) {
		int target = targets[sample];
		loss[sample] = -row_x[target] + row_max + logf(row_sum);
	}
}
