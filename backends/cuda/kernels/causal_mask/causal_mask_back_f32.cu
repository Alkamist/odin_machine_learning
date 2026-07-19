extern "C" __global__
void causal_mask_back_f32(const float* __restrict__ dy, float* __restrict__ dx, int n_blocks, int T) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n_blocks * T * T;
	if (gid >= total) return;
	int within = gid % (T * T);
	int t1 = within / T;
	int t2 = within % T;
	if (t2 <= t1) dx[gid] += dy[gid];
}
