// Linear K/V cache write taking fp32 input, casting to bf16 at store time.
// Cache layout matches `cache_write_bf16.cu` exactly (linear seq order; new
// rows go at slot `min(*pos_dev, capacity - n_rows) + r`).
#include <cuda_bf16.h>

extern "C" __global__
void cache_write_f32(const float*        __restrict__ src,    // fp32 [n_rows, kv_size]
                     unsigned int*       __restrict__ cache,  // bf16 packed pairs [capacity, kv_size]
                     const int*          __restrict__ pos_dev,
                     int n_rows, int kv_size, int capacity) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int pairs_per_row = kv_size >> 1;
	int total_pairs   = n_rows * pairs_per_row;
	if (gid >= total_pairs) return;

	int row         = gid / pairs_per_row;
	int pair_in_row = gid - row * pairs_per_row;

	int pos       = *pos_dev;
	int slot_base = (pos < capacity - n_rows) ? pos : (capacity - n_rows);
	int dst_row   = slot_base + row;
	int dst_pair  = dst_row * pairs_per_row + pair_in_row;

	int src_base  = row * kv_size + pair_in_row * 2;
	float v0 = src[src_base + 0];
	float v1 = src[src_base + 1];
	unsigned short lo = __bfloat16_as_ushort(__float2bfloat16(v0));
	unsigned short hi = __bfloat16_as_ushort(__float2bfloat16(v1));
	cache[dst_pair] = (unsigned int)lo | ((unsigned int)hi << 16);
}
