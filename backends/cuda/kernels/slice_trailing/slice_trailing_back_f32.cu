// slice_trailing forward: y[r, c] = x[r, start + c]
// backward: dx[r, start + c] += dy[r, c]
// Each output element maps to exactly one input element (one-to-one), so
// no atomics are required within a single op call. += accumulates across
// the wider input tensor's history.

extern "C" __global__
void slice_trailing_back_f32(const float* __restrict__ dy,
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
