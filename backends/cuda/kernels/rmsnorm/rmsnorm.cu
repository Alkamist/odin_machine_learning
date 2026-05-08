// F32 rmsnorm forward. One block per row, 256-wide strided reduction.
#define RMS_WG 256

extern "C" __global__
void rmsnorm_f32(const float* __restrict__ x,
                 const float* __restrict__ w,
                 float*       __restrict__ y,
                 int count, int size, float eps) {
	int row = blockIdx.x;
	int tid = threadIdx.x;
	if (row >= count) return;
	int base = row * size;

	float s2 = 0.0f;
	for (int i = tid; i < size; i += RMS_WG) {
		float v = x[base + i];
		s2 += v * v;
	}

	__shared__ float partial[RMS_WG];
	partial[tid] = s2;
	__syncthreads();
	#pragma unroll
	for (int stride = RMS_WG / 2; stride > 0; stride >>= 1) {
		if (tid < stride) partial[tid] += partial[tid + stride];
		__syncthreads();
	}
	float rstd = rsqrtf(partial[0] / (float)size + eps);

	for (int i = tid; i < size; i += RMS_WG) {
		y[base + i] = x[base + i] * rstd * w[i];
	}
}
