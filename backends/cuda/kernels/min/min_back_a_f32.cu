// min backward, a-side: the gradient goes to whichever input was the minimum.
// Ties (a == b) go to a.

extern "C" __global__
void min_back_a_f32(const float* __restrict__ a,
                    const float* __restrict__ b,
                    const float* __restrict__ dy,
                    float*       __restrict__ da,
                    int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	da[i] += (a[i] <= b[i]) ? dy[i] : 0.0f;
}
