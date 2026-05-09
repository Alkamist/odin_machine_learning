// Wrapper to instantiate ggml's flash_attn_ext_f16 MMA kernel for our case:
//   DKQ = DV = 256 (Gemma E4B sliding head_dim)
//   ncols1 = 2     (Q token padding from 1 — ggml's dispatcher picks
//                   ncols1*ncols2 ∈ {8,16,32,64} from the Ampere config table;
//                   for ncols2=4 + decode (Q.ne[1]=1), the smallest is ncols=8
//                   = 2*4. The kernel skips the dummy Q via ne01.z bounds.)
//   ncols2 = 4     (gqa_ratio for Gemma E4B)
//   use_logit_softcap = false
//   V_is_K_view = false
//
// Compiled offline via nvcc to PTX, embedded in our binary, loaded via
// cuModuleLoadData. Build command (see build_ptx.ps1):
//
//   nvcc -arch=sm_86 -ptx -std=c++17 -O3 \
//        -I../../../../ggml/include -I../../../../ggml/src -I../../../../ggml/src/ggml-cuda \
//        wrapper.cu -o attention_mma_d256_ncols2_4.ptx
//
// We use offline nvcc instead of NVRTC because ggml's source pulls in
// ggml.h / ggml-impl.h / ggml-cuda.h which aren't NVRTC-friendly (they're
// host-side ggml infrastructure, not CUDA device code).
//
// Output: fp32 dst (decode path; we pack to bf16 in our launcher with the
// same `pack_f32_to_bf16_pairs` kernel we use for Q4_K mmvq).

#include "fattn-mma-f16.cuh"

// Explicit template instantiation. Forces nvcc to emit the symbol.
// We look it up by mangled name from Odin via cuModuleGetFunction.
template __global__ void flash_attn_ext_f16<256, 256, 2, 4, false, false>(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    const char * __restrict__ mask,
    const char * __restrict__ sinks,
    const int  * __restrict__ KV_max,
    float      * __restrict__ dst,
    float2     * __restrict__ dst_meta,
    const float scale,
    const float max_bias,
    const float m0,
    const float m1,
    const uint32_t n_head_log2,
    const float logit_softcap,
    const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                        const int32_t nb01, const int32_t nb02, const int32_t nb03,
    const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                        const int32_t nb11, const int32_t nb12, const int64_t nb13,
                        const int32_t nb21, const int32_t nb22, const int64_t nb23,
                        const int32_t ne31, const int32_t ne32, const int32_t ne33,
                        const int32_t nb31, const int32_t nb32, const int64_t nb33);
