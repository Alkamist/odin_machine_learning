// col2im scatter: each output patch element accumulates back into the input
// position it was gathered from. Overlapping windows write the same input, so
// atomicAdd. Padded taps map outside the input and are skipped.
extern "C" __global__
void im2col_back_f32(const float* __restrict__ dy, float* __restrict__ dx,
                     int N, int H, int W, int C,
                     int KH, int KW, int SH, int SW, int PH, int PW,
                     int OH, int OW) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * OH * OW * KH * KW * C;
	if (gid >= total) return;

	int t = gid;
	int ci = t % C; t /= C;
	int kx = t % KW; t /= KW;
	int ky = t % KH; t /= KH;
	int ox = t % OW; t /= OW;
	int oy = t % OH; t /= OH;
	int n  = t;

	int iy = oy * SH - PH + ky;
	int ix = ox * SW - PW + kx;

	if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
		atomicAdd(&dx[(((size_t)(n * H + iy) * W + ix) * C) + ci], dy[gid]);
	}
}
