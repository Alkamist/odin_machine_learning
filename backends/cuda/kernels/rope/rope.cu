extern "C" __global__
void rope_f32(const float* __restrict__ x,
              float*       __restrict__ y,
              int token_count, int head_count, int head_size,
              float base, const int* __restrict__ position_offset_dev, int rotate_pair_count) {
	int gid     = blockIdx.x * blockDim.x + threadIdx.x;
	int half_hs = head_size / 2;
	int total   = token_count * head_count * half_hs;
	if (gid >= total) return;
	int position_offset = *position_offset_dev;

	int pair_idx = gid % half_hs;
	int hg       = gid / half_hs;
	int head     = hg % head_count;
	int pos      = hg / head_count;

	int head_offset = pos * head_count * head_size + head * head_size;
	int i_lo = head_offset + pair_idx * 2;
	int i_hi = i_lo + 1;

	if (pair_idx >= rotate_pair_count) {
		y[i_lo] = x[i_lo];
		y[i_hi] = x[i_hi];
		return;
	}

	float exponent = (float)(pair_idx * 2) / (float)head_size;
	float theta    = (float)(pos + position_offset) / powf(base, exponent);
	float c_v, s_v;
	sincosf(theta, &s_v, &c_v);

	float xv = x[i_lo];
	float yv = x[i_hi];
	y[i_lo] = xv * c_v - yv * s_v;
	y[i_hi] = xv * s_v + yv * c_v;
}
