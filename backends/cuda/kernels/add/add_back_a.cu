// da_a[i] += dy[i]. Thread per output.
extern "C" __global__
void add_back_a_f32(const float* __restrict__ dy,
                    float*       __restrict__ da_a,
                    int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) da_a[i] += dy[i];
}
