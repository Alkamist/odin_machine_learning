#include "broadcast.cuh"
// Mul backward, b-side. For c = a * b (b broadcasts over leading dims),
// db[j] += sum over i of dy[i*n_b + j] * a[i*n_b + j].
// One thread per a-element, atomicAdd into db.

extern "C" __global__
void mul_back_b_f32(const float* __restrict__ a,
                    const float* __restrict__ dy,
                    float*       __restrict__ db,
                    int n_a, int n_b) {
	int o = blockDim.x * blockIdx.x + threadIdx.x;
	if (o >= n_a) return;
	int j = bc_b_index(o, n_b);
	atomicAdd(&db[j], dy[o] * a[o]);
}
