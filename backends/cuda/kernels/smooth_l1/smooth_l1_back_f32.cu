extern "C" __global__
void smooth_l1_back_f32(const float* __restrict__ pred,
                        const float* __restrict__ tgt,
                        const float* __restrict__ dy,
                        float*       __restrict__ dx,
                        int count, int sample_size, float beta) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = count * sample_size;
	if (gid >= total) return;
	int s = gid / sample_size;
	float d = (pred[gid] - tgt[gid]) / beta;
	float c = d < -1.0f ? -1.0f : (d > 1.0f ? 1.0f : d);
	dx[gid] += c * (1.0f / (float)sample_size) * dy[s];
}
