// Element-wise bf16 → fp16 cast. Used to convert our linear K/V cache
// (stored as bf16-pair-packed unsigned ints) into the fp16 format ggml's
// flash_attn_ext_f16 MMA kernel expects. Mirrors what ggml does in their
// `launch_fattn` helper (fattn-common.cuh:965-1010) — they convert the
// entire K and V tensors to fp16 before each attention call.
//
// Layout: input is `[n_elems]` bf16 packed as `n_elems/2` unsigned ints.
// Output is `[n_elems]` fp16 packed as `n_elems/2` unsigned ints (same
// byte-level format, different value interpretation).
//
// bf16 → fp16 conversion: bf16 is sign + 8-bit exp + 7-bit mantissa.
// fp16 is sign + 5-bit exp + 10-bit mantissa. To convert: take bf16
// value as float (shift left 16), then float → fp16 (round to nearest).
// fp16 has narrower exponent range, so very large or very small bf16
// values saturate. Typical attention values are well within fp16 range.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

extern "C" __global__
void bf16_to_fp16_pairs(const unsigned int* __restrict__ src,
                        unsigned int*       __restrict__ dst,
                        int n_pairs) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= n_pairs) return;

	unsigned int packed = src[gid];

	unsigned short bf_lo = (unsigned short)(packed & 0xffffu);
	unsigned short bf_hi = (unsigned short)((packed >> 16) & 0xffffu);

	// bf16 bits → float (left-shift 16), then float → fp16.
	float fa = __int_as_float((int)((unsigned int)bf_lo << 16));
	float fb = __int_as_float((int)((unsigned int)bf_hi << 16));

	__half ha = __float2half(fa);
	__half hb = __float2half(fb);

	unsigned int out_lo = __half_as_ushort(ha);
	unsigned int out_hi = __half_as_ushort(hb);
	dst[gid] = out_lo | (out_hi << 16);
}
