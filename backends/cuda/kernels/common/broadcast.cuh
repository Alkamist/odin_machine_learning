#pragma once

__device__ __forceinline__ int bc_b_index(int o, int n_b) {
	return o % n_b;
}

__device__ __forceinline__ int bc_tile_index(int i, int j, int n_b) {
	return i * n_b + j;
}
