extern "C" __global__
void slice_trailing_back(const float* __restrict__ dy,
                         float*       __restrict__ dx,
                         int leading, int trailing, int new_trailing,
                         int start) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = leading * new_trailing;
	if (gid >= total) return;
	int r = gid / new_trailing;
	int c = gid - r * new_trailing;
	dx[r * trailing + start + c] += dy[gid];
}
