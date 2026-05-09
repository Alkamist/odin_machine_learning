// Bf16 rope. head_size is asserted even, so each (i_lo, i_hi) pair lands on
// one packed bf16 uint slot â€” one thread per pair reads, rotates in f32,
// writes back the packed result.
#include <cuda_bf16.h>

extern "C" __global__
void rope_bf16(const unsigned int* __restrict__ x,
               unsigned int*       __restrict__ y,
               int token_count, int head_count, int head_size,
               float base, const int* __restrict__ position_offset_dev, int rotate_pair_count) {
	int gid     = blockIdx.x * blockDim.x + threadIdx.x;
	int half_hs = head_size / 2;
	int total   = token_count * head_count * half_hs;
	if (gid >= total) return;
	int position_offset = *position_offset_dev;

	int pair_idx = gid % half_hs;
	int hg       = gid / half_hs;
	int head     = hg % head_count;
	int pos      = hg / head_count;

	int head_offset = pos * head_count * head_size + head * head_size;
	int i_lo        = head_offset + pair_idx * 2;
	int pair_index  = i_lo >> 1;

	if (pair_idx >= rotate_pair_count) {
		y[pair_index] = x[pair_index];
		return;
	}

	float exponent = (float)(pair_idx * 2) / (float)head_size;
	float theta    = (float)(pos + position_offset) / powf(base, exponent);
	float c_v, s_v;
	sincosf(theta, &s_v, &c_v);

	unsigned int packed = x[pair_index];
	float xv = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xffffu)));
	float yv = __bfloat162float(__ushort_as_bfloat16((unsigned short)((packed >> 16) & 0xffffu)));
	unsigned short out_lo = __bfloat16_as_ushort(__float2bfloat16(xv * c_v - yv * s_v));
	unsigned short out_hi = __bfloat16_as_ushort(__float2bfloat16(xv * s_v + yv * c_v));
	y[pair_index] = (unsigned int)out_lo | ((unsigned int)out_hi << 16);
}
