Next round (op fusion, ~75 tok/s target):

1. Fuse rms_norm + mul (the rmsnorm-with-weight pattern, every layer hits it
   multiple times). One shader: read X, compute rsqrt+normalize, multiply by
   weight, write Y. See llama.cpp `rms_norm.comp` with `RMS_NORM_MUL_FUSION`.
2. Fuse rms_norm + mul + rope for q_norm/k_norm specifically — three dispatches
   collapse to one. See llama.cpp `pipeline_rms_norm_mul_rope_f32_f32`.
3. Residual-add fusion into next op's read (rare but cheap once 1+2 are in).

Done:
- mmvq (Q8_1 + integer dot) for Q4_K decode: 37.2 -> 43.8 tok/s (+17.8%, ~4.1 ms/tok).
