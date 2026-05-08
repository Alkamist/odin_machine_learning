// da_a += dy in bf16 packed-pair form. Native __hadd2 since pairs are
// element-aligned and there's no broadcast on this path.
#include <cuda_bf16.h>

extern "C" __global__
void add_back_a_bf16(const unsigned int* __restrict__ dy,
                     unsigned int*       __restrict__ da_a,
                     int n, int pair_count) {
	int pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= pair_count) return;

	unsigned int dyp = dy[pair];
	unsigned int dap = da_a[pair];

	__nv_bfloat162 dyv = *reinterpret_cast<__nv_bfloat162*>(&dyp);
	__nv_bfloat162 dav = *reinterpret_cast<__nv_bfloat162*>(&dap);

	int i0 = 2 * pair;
	int i1 = i0 + 1;
	if (i1 < n) {
		__nv_bfloat162 sum = __hadd2(dav, dyv);
		da_a[pair] = *reinterpret_cast<unsigned int*>(&sum);
	} else if (i0 < n) {
		// Tail: only the low half is valid; preserve the high half of da_a.
		__nv_bfloat16 lo = __hadd(__low2bfloat16(dav), __low2bfloat16(dyv));
		unsigned short lob = __bfloat16_as_ushort(lo);
		unsigned int hi = dap & 0xffff0000u;
		da_a[pair] = hi | (unsigned int)lob;
	}
}
