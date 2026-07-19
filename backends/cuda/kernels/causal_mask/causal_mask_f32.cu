extern "C" __global__
void causal_mask_f32(const float* __restrict__ x, float* __restrict__ y, int n_blocks, int T) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n_blocks * T * T;
	if (gid >= total) return;
	int within = gid % (T * T);
	int t1 = within / T;
	int t2 = within % T;
	y[gid] = (t2 <= t1) ? x[gid] : __int_as_float(0xff800000);
}
