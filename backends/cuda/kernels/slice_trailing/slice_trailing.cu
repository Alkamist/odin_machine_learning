#ifdef DTYPE_BF16
#define DATA_T unsigned short
#define KERNEL_NAME slice_trailing_bf16
#else
#define DATA_T float
#define KERNEL_NAME slice_trailing_f32
#endif

extern "C" __global__
void KERNEL_NAME(const DATA_T* __restrict__ x,
                 DATA_T*       __restrict__ y,
                 int leading, int trailing, int new_trailing,
                 int start) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int total = leading * new_trailing;
	if (gid >= total) return;
	int r = gid / new_trailing;
	int c = gid % new_trailing;
	y[gid] = x[r * trailing + start + c];
}
