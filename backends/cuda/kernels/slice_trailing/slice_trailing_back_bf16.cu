// slice_trailing backward in bf16-data / F32-grad mode. Identical to the
// F32 kernel: dy and dx are f32, no forward data read needed. Kept as a
// distinct file so the dispatcher path mirrors other ops; could be unified
// with slice_trailing_back_f32.cu later.
extern "C" __global__
void slice_trailing_back_bf16(const float* __restrict__ dy,
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
