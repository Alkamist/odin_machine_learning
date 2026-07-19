extern "C" __global__
void permute_back_f32(const float* __restrict__ dy, float* __restrict__ dx,
                      int s0, int s1, int s2, int a0, int a1, int a2) {
	int ins[3] = {s0, s1, s2};
	int o0 = ins[a0];
	int o1 = ins[a1];
	int o2 = ins[a2];
	int total = o0 * o1 * o2;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= total) return;

	int i2 = gid % o2;
	int tmp = gid / o2;
	int i1 = tmp % o1;
	int i0 = tmp / o1;

	int src[3];
	src[a0] = i0;
	src[a1] = i1;
	src[a2] = i2;

	int in_strides[3] = {s1 * s2, s2, 1};
	int src_idx = src[0] * in_strides[0] + src[1] * in_strides[1] + src[2] * in_strides[2];
	dx[src_idx] += dy[gid];
}
