extern "C" __global__
void transpose_back_f32(const float* __restrict__ dy, float* __restrict__ dx, int rows, int cols) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = rows * cols;
	if (gid >= total) return;
	int i = gid / cols;
	int j = gid % cols;
	dx[i * cols + j] += dy[j * rows + i];
}
