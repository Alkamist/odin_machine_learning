extern "C" __global__
void mean_back_f32(const float* __restrict__ dy, float* __restrict__ dx, int count, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = count * size;
	if (gid >= total) return;
	int r = gid / size;
	dx[gid] += dy[r] / (float)size;
}
