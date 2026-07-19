extern "C" __global__
void avg_pool2d_f32(const float* __restrict__ x, float* __restrict__ y,
                    int N, int H, int W, int C,
                    int KH, int KW, int SH, int SW, int OH, int OW) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * OH * OW * C;
	if (gid >= total) return;

	int t = gid;
	int ci = t % C; t /= C;
	int ox = t % OW; t /= OW;
	int oy = t % OH; t /= OH;
	int n  = t;

	int base_y = oy * SH;
	int base_x = ox * SW;
	float total_sum = 0.0f;
	for (int ky = 0; ky < KH; ++ky) {
		int iy = base_y + ky;
		for (int kx = 0; kx < KW; ++kx) {
			int ix = base_x + kx;
			total_sum += x[(((size_t)(n * H + iy) * W + ix) * C) + ci];
		}
	}
	y[gid] = total_sum / (float)(KH * KW);
}
