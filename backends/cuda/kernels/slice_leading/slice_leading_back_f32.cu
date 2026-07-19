extern "C" __global__
void slice_leading_back_f32(const float* __restrict__ dy,
                           float*       __restrict__ dx,
                           int count, int offset) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= count) return;
	dx[offset + gid] += dy[gid];
}
