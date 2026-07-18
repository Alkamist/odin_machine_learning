// min backward, b-side: the gradient goes to b only where b was strictly the
// minimum. Ties (a == b) go to a, matching min_back_a_f32.

extern "C" __global__
void min_back_b_f32(const float* __restrict__ a,
                    const float* __restrict__ b,
                    const float* __restrict__ dy,
                    float*       __restrict__ db,
                    int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	db[i] += (a[i] <= b[i]) ? 0.0f : dy[i];
}
