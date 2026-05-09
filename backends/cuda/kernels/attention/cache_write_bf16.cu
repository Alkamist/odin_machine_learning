// Linear K/V cache write. Cache is laid out as `[capacity, kv_size]` in
// seq-position order — slot 0 is the oldest valid row, slot cap-1 is the
// newest. No ring-buffer modulo: ggml's MMA kernel and our future ggml
// drop-ins assume monotone K/V row order.
//
// Write slot for row r is `min(*pos_dev, capacity - n_rows) + r`. The min
// pins writes to the end of the cache once the live range fills capacity
// (sliding layers in steady state). Before that point, writes are linear-
// append at slot `*pos_dev + r` (full layers, and sliding layers in their
// first `capacity` tokens).
//
// For sliding layers in steady state, the host has already shifted cache
// contents back by `n_rows` via two cuMemcpyDtoDAsync (through a per-context
// scratch) before this kernel runs, so the write target slots `[cap-n_rows,
// cap)` are scratch and the rest of the cache holds the still-valid older
// rows.
#include <cuda_bf16.h>

extern "C" __global__
void cache_write_bf16(const unsigned int* __restrict__ src,    // [n_rows * kv_size] bf16 packed pairs
                      unsigned int*       __restrict__ cache,  // [capacity * kv_size] bf16 packed pairs
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
	int src_pair  = row * pairs_per_row + pair_in_row;

	cache[dst_pair] = src[src_pair];
}
