1. quantize_q8_1_bf16.comp — bf16 → Q8_1 blocks (had a draft, not on disk)
2. linear_q4_k_mmvq.comp — Q4_K weight × Q8_1 activation, dotPacked4x8EXT
3. Enable VK_KHR_shader_integer_dot_product at device init (verified hw-accelerated on the 3090 Ti)
4. Plumb scratch buffer for the quantized X through linear_q4_k_forward
5. A/B vs the 37.20 tok/s baseline; expect 3-4 ms/tok savings if it works