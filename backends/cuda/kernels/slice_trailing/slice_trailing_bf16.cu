// Bf16 forward of slice_trailing. Layout: [leading, trailing] -> [leading,
// new_trailing], copying [start, start+new_trailing) from each row. One
// thread per output PAIR; halves of a pair come from the same row but
// possibly non-aligned input positions.
extern "C" __global__
void slice_trailing_bf16(const unsigned int* __restrict__ x,
                         unsigned int*       __restrict__ y,
                         int leading, int trailing, int new_trailing,
                         int start, int pair_count) {
	int pair = blockIdx.x * blockDim.x + threadIdx.x;
	if (pair >= pair_count) return;

	int total_out = leading * new_trailing;
	int i0 = 2 * pair;
	int i1 = i0 + 1;

	auto load_bf16 = [&](int elem_index) -> unsigned int {
		int pair_index = elem_index >> 1;
		int shift = (elem_index & 1) * 16;
		return (x[pair_index] >> shift) & 0xffffu;
	};

	unsigned int lo = 0, hi = 0;
	if (i0 < total_out) {
		int r = i0 / new_trailing;
		int c = i0 % new_trailing;
		lo = load_bf16(r * trailing + start + c);
	}
	if (i1 < total_out) {
		int r = i1 / new_trailing;
		int c = i1 % new_trailing;
		hi = load_bf16(r * trailing + start + c);
	}
	y[pair] = (hi << 16) | lo;
}
