package machine_learning_backend_gpu

ADD_SPIRV        :: #load("shaders/add.spv",        []u8)
ADD_BACK_A_SPIRV :: #load("shaders/add_back_a.spv", []u8)
ADD_BACK_B_SPIRV :: #load("shaders/add_back_b.spv", []u8)
ADD_LOCAL_SIZE :: 256
Add_Params        :: struct { n,   n_b:    u32 }
Add_Back_A_Params :: struct { n: u32 }
Add_Back_B_Params :: struct { n_b, stride: u32 }
_add_pipeline, _add_back_a_pipeline, _add_back_b_pipeline: ^Pipeline

LINEAR_SPIRV :: #load("shaders/linear.spv", []u8)
LINEAR_LOCAL_X :: 32   // must match TILE_M in shaders/linear.comp
LINEAR_LOCAL_Y :: 64   // must match TILE_N in shaders/linear.comp
Linear_Params :: struct { count, input_size, output_size: u32 }
_linear_pipeline: ^Pipeline

LINEAR_BACK_INPUT_SPIRV  :: #load("shaders/linear_back_input.spv",  []u8)
LINEAR_BACK_WEIGHT_SPIRV :: #load("shaders/linear_back_weight.spv", []u8)
Linear_Back_Params :: struct { count, input_size, output_size: u32 }
_linear_back_input_pipeline:  ^Pipeline
_linear_back_weight_pipeline: ^Pipeline

BATCHED_MATMUL_SPIRV             :: #load("shaders/batched_matmul.spv",             []u8)
BATCHED_MATMUL_BACK_INPUT_SPIRV  :: #load("shaders/batched_matmul_back_input.spv",  []u8)
BATCHED_MATMUL_BACK_WEIGHT_SPIRV :: #load("shaders/batched_matmul_back_weight.spv", []u8)
BATCHED_MATMUL_LOCAL_X :: 16
BATCHED_MATMUL_LOCAL_Y :: 16
Batched_Matmul_Params :: struct { batch_count, m, k, n: u32 }
_batched_matmul_pipeline:             ^Pipeline
_batched_matmul_back_input_pipeline:  ^Pipeline
_batched_matmul_back_weight_pipeline: ^Pipeline

MUL_SPIRV        :: #load("shaders/mul.spv",        []u8)
MUL_BACK_A_SPIRV :: #load("shaders/mul_back_a.spv", []u8)
MUL_BACK_B_SPIRV :: #load("shaders/mul_back_b.spv", []u8)
Mul_Params        :: struct { n,   n_b:    u32 }
Mul_Back_A_Params :: struct { n,   n_b:    u32 }
Mul_Back_B_Params :: struct { n_b, stride: u32 }
_mul_pipeline:        ^Pipeline
_mul_back_a_pipeline: ^Pipeline
_mul_back_b_pipeline: ^Pipeline

GELU_SPIRV      :: #load("shaders/gelu.spv",      []u8)
GELU_BACK_SPIRV :: #load("shaders/gelu_back.spv", []u8)
GELU_LOCAL_SIZE :: 256
Gelu_Params      :: struct { n: u32 }
Gelu_Back_Params :: struct { n: u32 }
_gelu_pipeline:      ^Pipeline
_gelu_back_pipeline: ^Pipeline

LAYERNORM_SPIRV             :: #load("shaders/layernorm.spv",             []u8)
LAYERNORM_STATS_SPIRV       :: #load("shaders/layernorm_stats.spv",       []u8)
LAYERNORM_BACK_INPUT_SPIRV  :: #load("shaders/layernorm_back_input.spv",  []u8)
LAYERNORM_BACK_WEIGHT_SPIRV :: #load("shaders/layernorm_back_weight.spv", []u8)
Layernorm_Params       :: struct { count, size: u32 }
Layernorm_Stats_Params :: struct { count, size: u32 }
Layernorm_Back_Params  :: struct { count, size: u32 }
_layernorm_pipeline:             ^Pipeline
_layernorm_stats_pipeline:       ^Pipeline
_layernorm_back_input_pipeline:  ^Pipeline
_layernorm_back_weight_pipeline: ^Pipeline

SOFTMAX_SPIRV      :: #load("shaders/softmax.spv",      []u8)
SOFTMAX_BACK_SPIRV :: #load("shaders/softmax_back.spv", []u8)
Softmax_Params      :: struct { count, size: u32 }
Softmax_Back_Params :: struct { count, size: u32 }
_softmax_pipeline:      ^Pipeline
_softmax_back_pipeline: ^Pipeline

PERMUTE_SPIRV      :: #load("shaders/permute.spv",      []u8)
PERMUTE_BACK_SPIRV :: #load("shaders/permute_back.spv", []u8)
Permute_Params :: struct {
	out_d0, out_d1, out_d2: u32,
	in_d1,  in_d2:          u32,
	axes_0, axes_1, axes_2: u32,
}
_permute_pipeline:      ^Pipeline
_permute_back_pipeline: ^Pipeline

CAUSAL_MASK_SPIRV      :: #load("shaders/causal_mask.spv",      []u8)
CAUSAL_MASK_BACK_SPIRV :: #load("shaders/causal_mask_back.spv", []u8)
Causal_Mask_Params :: struct { total, T: u32 }
_causal_mask_pipeline:      ^Pipeline
_causal_mask_back_pipeline: ^Pipeline

ATTENTION_SPIRV         :: #load("shaders/attention.spv",         []u8)
ATTENTION_BACK_D_SPIRV  :: #load("shaders/attention_back_d.spv",  []u8)
ATTENTION_BACK_KV_SPIRV :: #load("shaders/attention_back_kv.spv", []u8)
ATTENTION_BACK_Q_SPIRV  :: #load("shaders/attention_back_q.spv",  []u8)
Attention_Params :: struct {
	head_count, head_size, token_count, embed_size, causal: u32,
}
Attention_Back_D_Params :: struct {
	head_count, head_size, token_count, embed_size: u32,
}
_attention_pipeline:         ^Pipeline
_attention_back_d_pipeline:  ^Pipeline
_attention_back_kv_pipeline: ^Pipeline
_attention_back_q_pipeline:  ^Pipeline

SELECT_SPIRV      :: #load("shaders/select.spv",      []u8)
SELECT_BACK_SPIRV :: #load("shaders/select_back.spv", []u8)
Select_Params      :: struct { n_indices, size: u32 }
Select_Back_Params :: struct { vocab, n_indices, size: u32 }
_select_pipeline:      ^Pipeline
_select_back_pipeline: ^Pipeline

ROPE_SPIRV      :: #load("shaders/rope.spv",      []u8)
ROPE_BACK_SPIRV :: #load("shaders/rope_back.spv", []u8)
Rope_Params      :: struct { token_count, head_count, head_size: u32, base: f32 }
Rope_Back_Params :: struct { token_count, head_count, head_size: u32, base: f32 }
_rope_pipeline:      ^Pipeline
_rope_back_pipeline: ^Pipeline

SLICE_TRAILING_SPIRV      :: #load("shaders/slice_trailing.spv",      []u8)
SLICE_TRAILING_BACK_SPIRV :: #load("shaders/slice_trailing_back.spv", []u8)
Slice_Trailing_Params      :: struct { leading, trailing, new_trailing, start: u32 }
Slice_Trailing_Back_Params :: struct { leading, trailing, new_trailing, start: u32 }
_slice_trailing_pipeline:      ^Pipeline
_slice_trailing_back_pipeline: ^Pipeline

CONCAT3_SPIRV      :: #load("shaders/concat3.spv",      []u8)
CONCAT3_BACK_SPIRV :: #load("shaders/concat3_back.spv", []u8)
Concat3_Params      :: struct { leading, t_a, t_b, t_c: u32 }
Concat3_Back_Params :: struct { leading, t_a, t_b, t_c: u32 }
_concat3_pipeline:      ^Pipeline
_concat3_back_pipeline: ^Pipeline

MEAN_SPIRV      :: #load("shaders/mean.spv",      []u8)
MEAN_BACK_SPIRV :: #load("shaders/mean_back.spv", []u8)
Mean_Params :: struct { count, size: u32 }
_mean_pipeline:      ^Pipeline
_mean_back_pipeline: ^Pipeline

RELU_SPIRV         :: #load("shaders/relu.spv",         []u8)
RELU_BACK_SPIRV    :: #load("shaders/relu_back.spv",    []u8)
SIGMOID_SPIRV      :: #load("shaders/sigmoid.spv",      []u8)
SIGMOID_BACK_SPIRV :: #load("shaders/sigmoid_back.spv", []u8)
SILU_SPIRV         :: #load("shaders/silu.spv",         []u8)
SILU_BACK_SPIRV    :: #load("shaders/silu_back.spv",    []u8)
TANH_SPIRV         :: #load("shaders/tanh.spv",         []u8)
TANH_BACK_SPIRV    :: #load("shaders/tanh_back.spv",    []u8)
EXP_SPIRV          :: #load("shaders/exp.spv",          []u8)
EXP_BACK_SPIRV     :: #load("shaders/exp_back.spv",     []u8)
Activation_Params :: struct { n: u32 }
_relu_pipeline,    _relu_back_pipeline:    ^Pipeline
_sigmoid_pipeline, _sigmoid_back_pipeline: ^Pipeline
_silu_pipeline,    _silu_back_pipeline:    ^Pipeline
_tanh_pipeline,    _tanh_back_pipeline:    ^Pipeline
_exp_pipeline,     _exp_back_pipeline:     ^Pipeline

CLAMP_SPIRV      :: #load("shaders/clamp.spv",      []u8)
CLAMP_BACK_SPIRV :: #load("shaders/clamp_back.spv", []u8)
Clamp_Params :: struct { n: u32, min_val, max_val: f32 }
_clamp_pipeline:      ^Pipeline
_clamp_back_pipeline: ^Pipeline

MIN_SPIRV      :: #load("shaders/min.spv",      []u8)
MIN_BACK_SPIRV :: #load("shaders/min_back.spv", []u8)
MAX_SPIRV      :: #load("shaders/max.spv",      []u8)
MAX_BACK_SPIRV :: #load("shaders/max_back.spv", []u8)
MinMax_Params :: struct { n: u32 }
_min_pipeline, _min_back_pipeline: ^Pipeline
_max_pipeline, _max_back_pipeline: ^Pipeline

SUB_SPIRV        :: #load("shaders/sub.spv",        []u8)
SUB_BACK_A_SPIRV :: #load("shaders/sub_back_a.spv", []u8)
SUB_BACK_B_SPIRV :: #load("shaders/sub_back_b.spv", []u8)
DIV_SPIRV        :: #load("shaders/div.spv",        []u8)
DIV_BACK_A_SPIRV :: #load("shaders/div_back_a.spv", []u8)
DIV_BACK_B_SPIRV :: #load("shaders/div_back_b.spv", []u8)
Sub_Params        :: struct { n,   n_b:    u32 }
Sub_Back_A_Params :: struct { n: u32 }
Sub_Back_B_Params :: struct { n_b, stride: u32 }
Div_Params        :: struct { n,   n_b:    u32 }
Div_Back_A_Params :: struct { n,   n_b:    u32 }
Div_Back_B_Params :: struct { n_b, stride: u32 }
_sub_pipeline, _sub_back_a_pipeline, _sub_back_b_pipeline: ^Pipeline
_div_pipeline, _div_back_a_pipeline, _div_back_b_pipeline: ^Pipeline

TRANSPOSE_SPIRV      :: #load("shaders/transpose.spv",      []u8)
TRANSPOSE_BACK_SPIRV :: #load("shaders/transpose_back.spv", []u8)
Transpose_Params :: struct { rows, cols: u32 }
_transpose_pipeline, _transpose_back_pipeline: ^Pipeline

SLICE_SPIRV      :: #load("shaders/slice.spv",      []u8)
SLICE_BACK_SPIRV :: #load("shaders/slice_back.spv", []u8)
Slice_Params :: struct { n, start: u32 }
_slice_pipeline, _slice_back_pipeline: ^Pipeline

LOG_SOFTMAX_SPIRV      :: #load("shaders/log_softmax.spv",      []u8)
LOG_SOFTMAX_BACK_SPIRV :: #load("shaders/log_softmax_back.spv", []u8)
Log_Softmax_Params :: struct { count, size: u32 }
_log_softmax_pipeline, _log_softmax_back_pipeline: ^Pipeline

ENTROPY_SPIRV      :: #load("shaders/entropy.spv",      []u8)
ENTROPY_BACK_SPIRV :: #load("shaders/entropy_back.spv", []u8)
Entropy_Params :: struct { count, size: u32 }
_entropy_pipeline, _entropy_back_pipeline: ^Pipeline

MEAN_SQUARED_ERROR_SPIRV      :: #load("shaders/mean_squared_error.spv",      []u8)
MEAN_SQUARED_ERROR_BACK_SPIRV :: #load("shaders/mean_squared_error_back.spv", []u8)
Mean_Squared_Error_Params :: struct { count, size: u32 }
_mean_squared_error_pipeline, _mean_squared_error_back_pipeline: ^Pipeline

CROSS_ENTROPY_SPIRV      :: #load("shaders/cross_entropy.spv",      []u8)
CROSS_ENTROPY_BACK_SPIRV :: #load("shaders/cross_entropy_back.spv", []u8)
Cross_Entropy_Params :: struct { count, class_size: u32 }
_cross_entropy_pipeline:      ^Pipeline
_cross_entropy_back_pipeline: ^Pipeline

ADAM_STEP_SPIRV :: #load("shaders/opt_step_adam.spv", []u8)
Adam_Params :: struct {
	n: u32,
	lr, beta1, beta2, eps, wd, bc1, bc2: f32,
}
_adam_step_pipeline: ^Pipeline
