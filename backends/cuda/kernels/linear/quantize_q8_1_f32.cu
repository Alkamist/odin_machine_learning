// fp32 X[K] -> Q8_1 blocks. Port of ggml's `quantize_q8_1`
// (ggml/src/ggml-cuda/quantize.cu).
//
// Block layout: CUDA_QUANTIZE_BLOCK_SIZE = 256 threads, each thread handles one
// element. With QK8_1 = 32 a warp of 32 threads exactly maps to one Q8_1
// sub-block, so a CUDA block covers 8 sub-blocks. Per-warp amax/sum via
// warp_reduce_max/sum templated on width=32.
//
// Per-block byte layout matches `block_q8_1` (36 bytes / 32 weights):
//   bytes  0..1   half d = amax/127
//   bytes  2..3   half s = d * sum(qs[i])
//   bytes  4..35  int8_t qs[32]
#include <cuda_fp16.h>

#define QK8_1                     32
#define CUDA_QUANTIZE_BLOCK_SIZE  256

template<int width>
static __device__ __forceinline__ float warp_reduce_max(float x) {
	#pragma unroll
	for (int offset = width / 2; offset > 0; offset >>= 1) {
		x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset, width));
	}
	return x;
}

template<int width>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
	#pragma unroll
	for (int offset = width / 2; offset > 0; offset >>= 1) {
		x += __shfl_xor_sync(0xffffffffu, x, offset, width);
	}
	return x;
}

extern "C" __global__
__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
void quantize_q8_1_f32(const float*  __restrict__ x,
                       unsigned int* __restrict__ y,
                       int K) {
	const int i0 = blockDim.x * blockIdx.x + threadIdx.x;

	if (i0 >= K) return;

	float xi = x[i0];

	float amax = fabsf(xi);
	float sum  = xi;
	amax = warp_reduce_max<QK8_1>(amax);
	sum  = warp_reduce_sum<QK8_1>(sum);

	const float  d = amax * (1.0f / 127.0f);
	const signed char q = (amax == 0.0f) ? (signed char)0
	                                     : (signed char)__float2int_rn(xi / d);

	const int ib  = i0 / QK8_1;
	const int iqs = i0 % QK8_1;

	signed char* y_bytes = (signed char*)y;
	y_bytes[ib * 36 + 4 + iqs] = q;

	if (iqs == 0) {
		unsigned short d_h = __half_as_ushort(__float2half(d));
		unsigned short s_h = __half_as_ushort(__float2half(d * sum));
		y[ib * 9] = (unsigned int)d_h | ((unsigned int)s_h << 16);
	}
}
