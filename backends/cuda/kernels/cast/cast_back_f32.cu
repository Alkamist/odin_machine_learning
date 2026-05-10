// Cast backward, F32 grads. With gradient buffers always F32 regardless of
// the data dtype, the backward of `y = cast_to(x, target)` is simply
// dx += dy element-wise. The forward cast direction is irrelevant.
extern "C" __global__
void cast_back_f32(const float* __restrict__ dy,
                   float*       __restrict__ dx,
                   int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	dx[i] += dy[i];
}
