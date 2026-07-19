// max_reduce backward: route dy to the argmax of each row. Ties go to the
// first (lowest-index) maximum, matching the CPU backend.
extern "C" __global__
void max_reduce_back_f32(const float* __restrict__ x, const float* __restrict__ dy,
                         float* __restrict__ dx, int count, int size) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= count) return;
	const float* xr = x + (size_t)r * size;
	int best = 0;
	float bv = xr[0];
	for (int i = 1; i < size; ++i) {
		if (xr[i] > bv) {
			bv = xr[i];
			best = i;
		}
	}
	dx[(size_t)r * size + best] += dy[r];
}
