extern "C" __global__
void rope_back(const float* __restrict__ dy,
               float*       __restrict__ dx,
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
		dx[i_lo] += dy[i_lo];
		dx[i_hi] += dy[i_hi];
		return;
	}

	float exponent = (float)(pair_idx * 2) / (float)head_size;
	float theta    = (float)(pos + position_offset) / powf(base, exponent);
	float c_v, s_v;
	sincosf(theta, &s_v, &c_v);

	float gx = dy[i_lo];
	float gy = dy[i_hi];
	dx[i_lo] +=  gx * c_v + gy * s_v;
	dx[i_hi] += -gx * s_v + gy * c_v;
}
