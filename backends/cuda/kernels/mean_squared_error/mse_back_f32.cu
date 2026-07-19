extern "C" __global__
void mse_back_f32(const float* __restrict__ pred,
                  const float* __restrict__ tgt,
                  const float* __restrict__ dy,
                  float*       __restrict__ dx,
                  int count, int sample_size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = count * sample_size;
	if (gid >= total) return;
	int s = gid / sample_size;
	float scale = 2.0f / (float)sample_size;
	dx[gid] += scale * (pred[gid] - tgt[gid]) * dy[s];
}
