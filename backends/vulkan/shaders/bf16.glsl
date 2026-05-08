// Bf16 packing helpers shared by every *_bf16.comp shader. Bf16 buffers are
// stored as packed uint pairs (low 16 bits = even element, high 16 bits = odd).

float bf16_expand(uint half_bits) {
    return uintBitsToFloat(half_bits << 16);
}

uint bf16_round(float x) {
    uint bits = floatBitsToUint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (rounded >> 16) & 0xffffu;
}
