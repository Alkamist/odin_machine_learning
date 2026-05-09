package machine_learning_backend_cuda

// Kernel source declarations. Each `.cu` file under `kernels/<op>/` is a
// single CUDA kernel (extern "C" __global__) compiled at startup via NVRTC,
// mirroring the per-variant `.comp` shaders in backends/vulkan/shaders/.
//
// Files are embedded as []u8 via #load and NUL-terminated at compile time
// inside `_compile_pipeline` before being passed to NVRTC.

ADD_LOCAL_SIZE :: 256

// Q4_K mmvq launch shape: one warp per workgroup, two output rows per warp.
// Mirrors the vulkan linear_q4_k_mmvq.comp tile shape.
Q4_K_MMVQ_ROWS_PER_WG :: 2
Q8_1_BLOCK_BYTES      :: 36   // 4 bytes (d, s) + 32 bytes (qs)
Q8_1_BLOCK_UINTS      :: 9    // == Q8_1_BLOCK_BYTES / 4
Q8_1_BLOCK_ELEMS      :: 32

// ----- Embedded kernel sources ----------------------------------------------

ADD_F32_SRC          :: #load("kernels/add/add.cu",                []u8)
ADD_BF16_SRC         :: #load("kernels/add/add_bf16.cu",           []u8)
ADD_BACK_A_SRC       :: #load("kernels/add/add_back_a.cu",         []u8)
ADD_BACK_B_SRC       :: #load("kernels/add/add_back_b.cu",         []u8)
ADD_BACK_A_BF16_SRC  :: #load("kernels/add/add_back_a_bf16.cu",    []u8)
ADD_BACK_B_BF16_SRC  :: #load("kernels/add/add_back_b_bf16.cu",    []u8)

MUL_F32_SRC          :: #load("kernels/mul/mul.cu",                []u8)
MUL_BF16_SRC         :: #load("kernels/mul/mul_bf16.cu",           []u8)

GELU_MUL_BF16_SRC    :: #load("kernels/gelu_mul/gelu_mul_bf16.cu", []u8)
GELU_MUL_F32_SRC     :: #load("kernels/gelu_mul/gelu_mul_f32.cu",  []u8)

TANH_F32_SRC         :: #load("kernels/tanh/tanh.cu",              []u8)
TANH_BF16_SRC        :: #load("kernels/tanh/tanh_bf16.cu",         []u8)

CAST_BF16_TO_F32_SRC :: #load("kernels/cast/cast_bf16_to_f32.cu",  []u8)
CAST_F32_TO_BF16_SRC :: #load("kernels/cast/cast_f32_to_bf16.cu",  []u8)

RMSNORM_F32_SRC      :: #load("kernels/rmsnorm/rmsnorm.cu",            []u8)
RMSNORM_BF16_SRC     :: #load("kernels/rmsnorm/rmsnorm_bf16.cu",       []u8)
ADD_RMSNORM_BF16_SRC :: #load("kernels/rmsnorm/add_rmsnorm_bf16.cu",   []u8)
ADD_RMSNORM_F32_SRC  :: #load("kernels/rmsnorm/add_rmsnorm_f32.cu",    []u8)
RMSNORM_ROPE_BF16_SRC       :: #load("kernels/rmsnorm/rmsnorm_rope_bf16.cu",       []u8)
RMSNORM_ROPE_F32_SRC        :: #load("kernels/rmsnorm/rmsnorm_rope_f32.cu",        []u8)
RMSNORM_ROPE_CACHE_BF16_SRC :: #load("kernels/rmsnorm/rmsnorm_rope_cache_bf16.cu", []u8)
RMSNORM_ROPE_CACHE_F32_SRC  :: #load("kernels/rmsnorm/rmsnorm_rope_cache_f32.cu",  []u8)

ROPE_F32_SRC         :: #load("kernels/rope/rope.cu",              []u8)
ROPE_BF16_SRC        :: #load("kernels/rope/rope_bf16.cu",         []u8)

QUANTIZE_Q8_1_BF16_SRC :: #load("kernels/linear/quantize_q8_1_bf16.cu", []u8)
QUANTIZE_Q8_1_F32_SRC  :: #load("kernels/linear/quantize_q8_1_f32.cu",  []u8)
LINEAR_Q4_K_MMVQ_SRC   :: #load("kernels/linear/linear_q4_k_mmvq.cu",   []u8)
LINEAR_Q4_K_GATE_UP_GEGLU_BF16_SRC :: #load("kernels/linear/linear_q4_k_gate_up_geglu_bf16.cu", []u8)
LINEAR_Q6_K_MMVQ_SRC   :: #load("kernels/linear/linear_q6_k_mmvq.cu",   []u8)
PACK_F32_TO_BF16_PAIRS_SRC :: #load("kernels/linear/pack_f32_to_bf16_pairs.cu", []u8)

ATTENTION_BF16_SRC           :: #load("kernels/attention/attention_bf16.cu",           []u8)
ATTENTION_CACHE_BF16_SRC     :: #load("kernels/attention/attention_cache_bf16.cu",     []u8)
ATTENTION_CACHE_VEC_BF16_SRC :: #load("kernels/attention/attention_cache_vec_bf16.cu", []u8)
ATTENTION_CACHE_VEC_F32_SRC  :: #load("kernels/attention/attention_cache_vec_f32.cu",  []u8)
ATTENTION_CACHE_MMA_BF16_SRC :: #load("kernels/attention/attention_cache_mma_bf16.cu", []u8)
CACHE_WRITE_BF16_SRC         :: #load("kernels/attention/cache_write_bf16.cu",         []u8)
CACHE_WRITE_F32_SRC          :: #load("kernels/attention/cache_write_f32.cu",          []u8)

// ggml MMA F16 kernel: compiled offline via nvcc (see
// `kernels/attention/ggml_mma/build_ptx.ps1`) and embedded as PTX. Loaded
// via cuModuleLoadData at runtime; symbol is the C++ mangled name of
// `flash_attn_ext_f16<256, 256, 2, 4, false, false>`.
GGML_MMA_D256_NCOLS2_4_PTX  :: #load("kernels/attention/ggml_mma/attention_mma_d256_ncols2_4.ptx", []u8)
GGML_MMA_D256_NCOLS2_4_NAME :: cstring("_Z18flash_attn_ext_f16ILi256ELi256ELi2ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP6float2ffffjfi5uint3iiiiiiiiiiixiixiiiiix")
BF16_TO_FP16_PAIRS_SRC      :: #load("kernels/attention/ggml_mma/bf16_to_fp16.cu", []u8)

SELECT_F32_SRC           :: #load("kernels/select/select.cu",                  []u8)
SELECT_BF16_SRC          :: #load("kernels/select/select_bf16.cu",             []u8)

SLICE_TRAILING_F32_SRC   :: #load("kernels/slice_trailing/slice_trailing.cu",      []u8)
SLICE_TRAILING_BF16_SRC  :: #load("kernels/slice_trailing/slice_trailing_bf16.cu", []u8)
