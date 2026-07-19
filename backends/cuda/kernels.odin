package machine_learning_backend_cuda

ADD_LOCAL_SIZE :: 256

BROADCAST_CUH_SRC :: #load("kernels/common/broadcast.cuh", []u8)

Q4_K_MMVQ_ROWS_PER_WG :: 2
Q8_1_BLOCK_BYTES      :: 36
Q8_1_BLOCK_UINTS      :: 9
Q8_1_BLOCK_ELEMS      :: 32

ADD_F32_SRC         :: #load("kernels/add/add.cu",             []u8)
ADD_BF16_SRC        :: #load("kernels/add/add_bf16.cu",        []u8)
ADD_BACK_A_SRC      :: #load("kernels/add/add_back_a.cu",      []u8)
ADD_BACK_B_SRC      :: #load("kernels/add/add_back_b.cu",      []u8)
ADD_BACK_A_BF16_SRC :: #load("kernels/add/add_back_a_bf16.cu", []u8)
ADD_BACK_B_BF16_SRC :: #load("kernels/add/add_back_b_bf16.cu", []u8)

MUL_F32_SRC  :: #load("kernels/mul/mul.cu",      []u8)
MUL_BF16_SRC :: #load("kernels/mul/mul_bf16.cu", []u8)

GELU_MUL_BF16_SRC :: #load("kernels/gelu_mul/gelu_mul_bf16.cu", []u8)
GELU_MUL_F32_SRC  :: #load("kernels/gelu_mul/gelu_mul_f32.cu",  []u8)

GELU_F32_SRC       :: #load("kernels/gelu/gelu_f32.cu",       []u8)
GELU_BF16_SRC      :: #load("kernels/gelu/gelu_bf16.cu",      []u8)
GELU_BACK_F32_SRC  :: #load("kernels/gelu/gelu_back_f32.cu",  []u8)
GELU_BACK_BF16_SRC :: #load("kernels/gelu/gelu_back_bf16.cu", []u8)

TANH_F32_SRC       :: #load("kernels/tanh/tanh.cu",           []u8)
TANH_BF16_SRC      :: #load("kernels/tanh/tanh_bf16.cu",      []u8)
TANH_BACK_F32_SRC  :: #load("kernels/tanh/tanh_back_f32.cu",  []u8)
TANH_BACK_BF16_SRC :: #load("kernels/tanh/tanh_back_bf16.cu", []u8)

// Elementwise ops added for the RL losses (F32 only - the losses run in F32).
EXP_F32_SRC        :: #load("kernels/exp/exp.cu",               []u8)
EXP_BACK_F32_SRC   :: #load("kernels/exp/exp_back_f32.cu",      []u8)
CLAMP_F32_SRC      :: #load("kernels/clamp/clamp.cu",           []u8)
CLAMP_BACK_F32_SRC :: #load("kernels/clamp/clamp_back_f32.cu",  []u8)

// Row-wise softmax and entropy over a head's logits (F32 only).
SOFTMAX_F32_SRC      :: #load("kernels/softmax/softmax.cu",          []u8)
SOFTMAX_BACK_F32_SRC :: #load("kernels/softmax/softmax_back_f32.cu", []u8)
ENTROPY_F32_SRC      :: #load("kernels/entropy/entropy.cu",          []u8)
ENTROPY_BACK_F32_SRC :: #load("kernels/entropy/entropy_back_f32.cu", []u8)

CAST_BF16_TO_F32_SRC :: #load("kernels/cast/cast_bf16_to_f32.cu", []u8)
CAST_F32_TO_BF16_SRC :: #load("kernels/cast/cast_f32_to_bf16.cu", []u8)
CAST_BACK_F32_SRC    :: #load("kernels/cast/cast_back_f32.cu",    []u8)

RMSNORM_F32_SRC  :: #load("kernels/rmsnorm/rmsnorm.cu",      []u8)
RMSNORM_BF16_SRC :: #load("kernels/rmsnorm/rmsnorm_bf16.cu", []u8)

ADD_RMSNORM_BF16_SRC :: #load("kernels/rmsnorm/add_rmsnorm_bf16.cu", []u8)
ADD_RMSNORM_F32_SRC  :: #load("kernels/rmsnorm/add_rmsnorm_f32.cu",  []u8)

RMSNORM_ROPE_BF16_SRC       :: #load("kernels/rmsnorm/rmsnorm_rope_bf16.cu",       []u8)
RMSNORM_ROPE_F32_SRC        :: #load("kernels/rmsnorm/rmsnorm_rope_f32.cu",        []u8)
RMSNORM_ROPE_CACHE_BF16_SRC :: #load("kernels/rmsnorm/rmsnorm_rope_cache_bf16.cu", []u8)

ROPE_F32_SRC         :: #load("kernels/rope/rope.cu",      []u8)
ROPE_BF16_SRC        :: #load("kernels/rope/rope_bf16.cu", []u8)

QUANTIZE_Q8_1_BF16_SRC :: #load("kernels/linear/quantize_q8_1_bf16.cu", []u8)

LINEAR_Q4_K_MMVQ_SRC               :: #load("kernels/linear/linear_q4_k_mmvq.cu",               []u8)
LINEAR_Q4_K_GATE_UP_GEGLU_BF16_SRC :: #load("kernels/linear/linear_q4_k_gate_up_geglu_bf16.cu", []u8)
LINEAR_Q6_K_MMVQ_SRC               :: #load("kernels/linear/linear_q6_k_mmvq.cu",               []u8)
DEQUANTIZE_Q4_K_TO_BF16_SRC        :: #load("kernels/linear/dequantize_q4_k_to_bf16.cu",        []u8)
DEQUANTIZE_Q6_K_TO_BF16_SRC        :: #load("kernels/linear/dequantize_q6_k_to_bf16.cu",        []u8)

ATTENTION_BF16_SRC           :: #load("kernels/attention/attention_bf16.cu",           []u8)
ATTENTION_CACHE_BF16_SRC     :: #load("kernels/attention/attention_cache_bf16.cu",     []u8)
ATTENTION_CACHE_VEC_BF16_SRC :: #load("kernels/attention/attention_cache_vec_bf16.cu", []u8)
CACHE_WRITE_BF16_SRC         :: #load("kernels/attention/cache_write_bf16.cu",         []u8)

SELECT_F32_SRC  :: #load("kernels/select/select.cu",      []u8)
SELECT_BF16_SRC :: #load("kernels/select/select_bf16.cu", []u8)

SLICE_TRAILING_F32_SRC       :: #load("kernels/slice_trailing/slice_trailing.cu",          []u8)
SLICE_TRAILING_BF16_SRC      :: #load("kernels/slice_trailing/slice_trailing_bf16.cu",     []u8)
SLICE_TRAILING_BACK_F32_SRC  :: #load("kernels/slice_trailing/slice_trailing_back_f32.cu",  []u8)
SLICE_TRAILING_BACK_BF16_SRC :: #load("kernels/slice_trailing/slice_trailing_back_bf16.cu", []u8)
SLICE_LEADING_BACK_F32_SRC   :: #load("kernels/slice_leading/slice_leading_back_f32.cu",    []u8)

ADAM_F32_SRC  :: #load("kernels/optimizer/adam_f32.cu",  []u8)
ADAM_BF16_SRC :: #load("kernels/optimizer/adam_bf16.cu", []u8)

SQ_SUM_F32_SRC :: #load("kernels/reduce/sq_sum_f32.cu", []u8)
SCALE_F32_SRC  :: #load("kernels/reduce/scale_f32.cu",  []u8)

SILU_F32_SRC       :: #load("kernels/silu/silu_f32.cu",       []u8)
SILU_BF16_SRC      :: #load("kernels/silu/silu_bf16.cu",      []u8)
SILU_BACK_F32_SRC  :: #load("kernels/silu/silu_back_f32.cu",  []u8)
SILU_BACK_BF16_SRC :: #load("kernels/silu/silu_back_bf16.cu", []u8)

CROSS_ENTROPY_F32_SRC      :: #load("kernels/cross_entropy/cross_entropy_f32.cu",      []u8)
CROSS_ENTROPY_BACK_F32_SRC :: #load("kernels/cross_entropy/cross_entropy_back_f32.cu", []u8)

MUL_BACK_A_F32_SRC  :: #load("kernels/mul/mul_back_a_f32.cu",  []u8)
MUL_BACK_B_F32_SRC  :: #load("kernels/mul/mul_back_b_f32.cu",  []u8)
MUL_BACK_A_BF16_SRC :: #load("kernels/mul/mul_back_a_bf16.cu", []u8)
MUL_BACK_B_BF16_SRC :: #load("kernels/mul/mul_back_b_bf16.cu", []u8)

SELECT_BACK_F32_SRC  :: #load("kernels/select/select_back_f32.cu",  []u8)
SELECT_BACK_BF16_SRC :: #load("kernels/select/select_back_bf16.cu", []u8)

ROPE_BACK_F32_SRC  :: #load("kernels/rope/rope_back_f32.cu",  []u8)
ROPE_BACK_BF16_SRC :: #load("kernels/rope/rope_back_bf16.cu", []u8)

RMSNORM_BACK_F32_SRC  :: #load("kernels/rmsnorm/rmsnorm_back_f32.cu",  []u8)
RMSNORM_BACK_BF16_SRC :: #load("kernels/rmsnorm/rmsnorm_back_bf16.cu", []u8)

ATTENTION_TRAIN_F32_SRC       :: #load("kernels/attention/attention_train_f32.cu",       []u8)
ATTENTION_TRAIN_BACK_F32_SRC  :: #load("kernels/attention/attention_train_back_f32.cu",  []u8)
ATTENTION_TRAIN_BF16_SRC      :: #load("kernels/attention/attention_train_bf16.cu",      []u8)
ATTENTION_TRAIN_BACK_BF16_SRC :: #load("kernels/attention/attention_train_back_bf16.cu", []u8)

ELEMENTWISE_UNARY_SRC         :: #load("kernels/elementwise/unary.cu",         []u8)
ELEMENTWISE_UNARY_BACK_SRC    :: #load("kernels/elementwise/unary_back.cu",    []u8)
ELEMENTWISE_BINARY_SRC        :: #load("kernels/elementwise/binary.cu",        []u8)
ELEMENTWISE_BINARY_BACK_A_SRC :: #load("kernels/elementwise/binary_back_a.cu", []u8)
ELEMENTWISE_BINARY_BACK_B_SRC :: #load("kernels/elementwise/binary_back_b.cu", []u8)

MEAN_F32_SRC      :: #load("kernels/mean/mean.cu",          []u8)
MEAN_BACK_F32_SRC :: #load("kernels/mean/mean_back_f32.cu", []u8)

SUM_F32_SRC             :: #load("kernels/reduce/sum.cu",                 []u8)
SUM_BACK_F32_SRC        :: #load("kernels/reduce/sum_back_f32.cu",        []u8)
MAX_REDUCE_F32_SRC      :: #load("kernels/reduce/max_reduce.cu",          []u8)
MAX_REDUCE_BACK_F32_SRC :: #load("kernels/reduce/max_reduce_back_f32.cu", []u8)

IM2COL_F32_SRC      :: #load("kernels/im2col/im2col_f32.cu",      []u8)
IM2COL_BACK_F32_SRC :: #load("kernels/im2col/im2col_back_f32.cu", []u8)

MAX_POOL2D_F32_SRC      :: #load("kernels/pool/max_pool2d_f32.cu",      []u8)
MAX_POOL2D_BACK_F32_SRC :: #load("kernels/pool/max_pool2d_back_f32.cu", []u8)
AVG_POOL2D_F32_SRC      :: #load("kernels/pool/avg_pool2d_f32.cu",      []u8)
AVG_POOL2D_BACK_F32_SRC :: #load("kernels/pool/avg_pool2d_back_f32.cu", []u8)

LOG_SOFTMAX_F32_SRC      :: #load("kernels/log_softmax/log_softmax.cu",          []u8)
LOG_SOFTMAX_BACK_F32_SRC :: #load("kernels/log_softmax/log_softmax_back_f32.cu", []u8)

LAYERNORM_F32_SRC      :: #load("kernels/layernorm/layernorm_f32.cu",      []u8)
LAYERNORM_BACK_F32_SRC :: #load("kernels/layernorm/layernorm_back_f32.cu", []u8)

MSE_F32_SRC      :: #load("kernels/mean_squared_error/mse_f32.cu",      []u8)
MSE_BACK_F32_SRC :: #load("kernels/mean_squared_error/mse_back_f32.cu", []u8)

SMOOTH_L1_F32_SRC      :: #load("kernels/smooth_l1/smooth_l1_f32.cu",      []u8)
SMOOTH_L1_BACK_F32_SRC :: #load("kernels/smooth_l1/smooth_l1_back_f32.cu", []u8)

TRANSPOSE_F32_SRC      :: #load("kernels/transpose/transpose_f32.cu",      []u8)
TRANSPOSE_BACK_F32_SRC :: #load("kernels/transpose/transpose_back_f32.cu", []u8)

PERMUTE_F32_SRC      :: #load("kernels/permute/permute_f32.cu",      []u8)
PERMUTE_BACK_F32_SRC :: #load("kernels/permute/permute_back_f32.cu", []u8)

CONCAT_F32_SRC      :: #load("kernels/concat/concat_f32.cu",      []u8)
CONCAT_BACK_F32_SRC :: #load("kernels/concat/concat_back_f32.cu", []u8)

CAUSAL_MASK_F32_SRC      :: #load("kernels/causal_mask/causal_mask_f32.cu",      []u8)
CAUSAL_MASK_BACK_F32_SRC :: #load("kernels/causal_mask/causal_mask_back_f32.cu", []u8)

LERP_ASSIGN_F32_SRC     :: #load("kernels/lerp_assign/lerp_assign_f32.cu",         []u8)
ACCUMULATE_MEAN_F32_SRC :: #load("kernels/accumulate_mean/accumulate_mean_f32.cu", []u8)
