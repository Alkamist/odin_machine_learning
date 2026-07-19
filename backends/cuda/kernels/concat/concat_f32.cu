extern "C" __global__
void concat_f32(const float* __restrict__ in, float* __restrict__ out,
                int leading, int in_trailing, int out_trailing, int dst_col) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = leading * in_trailing;
	if (gid >= total) return;
	int r = gid / in_trailing;
	int c = gid % in_trailing;
	out[r * out_trailing + dst_col + c] = in[gid];
}
