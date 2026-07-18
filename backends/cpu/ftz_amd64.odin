#+build amd64
package cpu

import x86 "core:simd/x86"

// Denormal (subnormal) floats are handled in x86 microcode and can slow FP throughput by an
// order of magnitude. ML weights and softmax tails routinely underflow into that range, and
// the slowdown then grows as training sharpens the model. Flush-to-zero is the standard ML
// setting; it must be applied per thread (MXCSR is thread state).
//
// Denormals-are-zero (bit 6) is deliberately NOT set: it corrupts Gemma inference on this
// backend, turning replies into unrelated token soup. Isolated by masking each bit alone --
// FTZ (0x8000) alone is correct, DAZ (0x0040) alone reproduces the garbage. The mechanism is
// not understood: the bad logits are finite rather than NaN, and no op was observed to emit a
// denormal output, so something depends on denormal *inputs* in a way not yet tracked down.
@(enable_target_feature="sse")
_enable_flush_to_zero :: proc "contextless" () {
	FTZ :: u32(0x8000) // MXCSR bit 15 (flush-to-zero)
	x86._mm_setcsr(x86._mm_getcsr() | FTZ)
}
