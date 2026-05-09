// silu backward: dx[i] += dy[i] * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))).
// One thread per element. Accumulates into dx (`+=`).

extern "C" __global__
void silu_back_f32(const float* __restrict__ x,
                   const float* __restrict__ dy,
                   float*       __restrict__ dx,
                   int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;
	float v = x[i];
	float s = 1.0f / (1.0f + expf(-v));
	float local = s + v * s * (1.0f - s);
	dx[i] += dy[i] * local;
}
