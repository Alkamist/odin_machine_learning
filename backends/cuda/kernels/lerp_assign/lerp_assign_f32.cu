extern "C" __global__
void lerp_assign_f32(float* __restrict__ dst, const float* __restrict__ src, float alpha, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	dst[i] = (1.0f - alpha) * dst[i] + alpha * src[i];
}
