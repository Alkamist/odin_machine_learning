// clamp backward: gradient passes through only where x was strictly inside
// [lo, hi]; at or beyond a bound the output was pinned and the local slope is 0.

extern "C" __global__
void clamp_back_f32(const float* __restrict__ x,
                    const float* __restrict__ dy,
                    float*       __restrict__ dx,
                    float lo, float hi, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	float v = x[i];
	dx[i] += (v > lo && v < hi) ? dy[i] : 0.0f;
}
