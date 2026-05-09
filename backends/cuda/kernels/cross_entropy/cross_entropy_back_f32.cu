// Cross-entropy backward.
// dx[s, i] += (probs[s, i] - (i == target[s] ? 1 : 0)) * dy[s]
//
// One thread per (sample, class) element. The targets array is read
// once per sample so we don't need a block-level reduction.

extern "C" __global__
void cross_entropy_back_f32(const float* __restrict__ probs,    // [n_samples, class_size]
                            const int*   __restrict__ targets,   // [n_samples]
                            const float* __restrict__ dy,        // [n_samples]
                            float*       __restrict__ dx,        // [n_samples, class_size]
                            int n_samples, int class_size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total = n_samples * class_size;
	if (idx >= total) return;

	int sample = idx / class_size;
	int klass  = idx - sample * class_size;
	int target = targets[sample];

	float p     = probs[idx];
	float upstr = dy[sample];
	float t     = (klass == target) ? 1.0f : 0.0f;

	dx[idx] += (p - t) * upstr;
}
