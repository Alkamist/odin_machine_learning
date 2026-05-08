// F32 forward of slice_trailing. One thread per output element.
extern "C" __global__
void slice_trailing_f32(const float* __restrict__ x,
                        float*       __restrict__ y,
                        int leading, int trailing, int new_trailing,
                        int start) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = leading * new_trailing;
	if (gid >= total) return;
	int r = gid / new_trailing;
	int c = gid % new_trailing;
	y[gid] = x[r * trailing + start + c];
}
