// K/V cache ring-buffer write. Copies a [n_rows, kv_size] bf16 source into the
// `[t_capacity, kv_size]` cache slot starting at `*pos_dev`, wrapping modulo
// `t_capacity`. Replaces the host-issued cuMemcpyDtoDAsync for the K/V write
// in attention_cache so the dst offset is computed device-side from a stable
// pointer (`pos_dev`); without this, every decode step's captured graph had
// a different memcpy dst that `cuGraphExecUpdate` had to patch.
#include <cuda_bf16.h>

extern "C" __global__
void cache_write_bf16(const unsigned int* __restrict__ src,    // [n_rows * kv_size] bf16 packed pairs
                      unsigned int*       __restrict__ cache,  // [t_capacity * kv_size] bf16 packed pairs
                      const int*          __restrict__ pos_dev,
                      int n_rows, int kv_size, int t_capacity) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int pairs_per_row = kv_size >> 1;
	int total_pairs   = n_rows * pairs_per_row;
	if (gid >= total_pairs) return;

	int row         = gid / pairs_per_row;
	int pair_in_row = gid - row * pairs_per_row;

	int pos      = *pos_dev;
	int dst_row  = (pos + row) % t_capacity;
	int dst_pair = dst_row * pairs_per_row + pair_in_row;
	int src_pair = row * pairs_per_row + pair_in_row;

	cache[dst_pair] = src[src_pair];
}
