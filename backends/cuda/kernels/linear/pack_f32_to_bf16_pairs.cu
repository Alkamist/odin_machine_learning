// Convert a flat fp32 array of length N into a packed-bf16-pair uint array of
// length ceil(N/2). Pairs at uint i hold bf16(x[2i]) in the low 16 bits and
// bf16(x[2i+1]) in the high 16 bits; if 2i+1 >= N (odd N), the high half is
// zero. Pair-packed bf16 is the rest of the pipeline's interchange format
// for activations between kernels.

__device__ __forceinline__ unsigned int bf16_from_f32(float v) {
	unsigned int bits = __float_as_uint(v);
	if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;  // NaN
	unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
	return (rounded >> 16) & 0xffffu;
}

extern "C" __global__
void pack_f32_to_bf16_pairs(const float * __restrict__ x,
                            unsigned int * __restrict__ y,
                            int n) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int pair_count = (n + 1) >> 1;
	if (gid >= pair_count) return;

	int i_lo = 2 * gid;
	int i_hi = i_lo + 1;

	unsigned int lo = bf16_from_f32(x[i_lo]);
	unsigned int hi = (i_hi < n) ? bf16_from_f32(x[i_hi]) : 0u;
	y[gid] = (hi << 16) | lo;
}
