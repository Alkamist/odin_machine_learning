extern "C" __global__
void transpose_f32(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = rows * cols;
	if (gid >= total) return;
	int i = gid / cols;
	int j = gid % cols;
	y[j * rows + i] = x[i * cols + j];
}
