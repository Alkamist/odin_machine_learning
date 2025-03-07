package ggml

import "core:c"
import "core:os"

when ODIN_OS != .Windows {
	#panic("Only Windows support for this ggml binding is currently implemented, but support for other architectures is possible.")
}

CUDA_VERSION :: #config(CUDA_VERSION, "None")
CUDA_DIR     :: "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v" + CUDA_VERSION + "/lib/x64/"

when CUDA_VERSION == "None" {
	@(extra_linker_flags="/NODEFAULTLIB:libcmt")
	foreign import ggml {
		"ggml.lib",
		"ggml-base.lib",
		"ggml-cpu.lib",
		"system:advapi32.lib",
	}
} else {
	@(extra_linker_flags="/NODEFAULTLIB:libcmt")
	foreign import ggml {
		"ggml.lib",
		"ggml-base.lib",
		"ggml-cpu.lib",
		"ggml-cuda.lib",
		CUDA_DIR + "cuda.lib",
		CUDA_DIR + "cudart.lib",
		CUDA_DIR + "cublas.lib",
		"system:advapi32.lib",
	}
}

MAX_DIMS      :: 4
MAX_PARAMS    :: 2048
MAX_SRC       :: 10
MAX_N_THREADS :: 512
MAX_OP_PARAMS :: 64
MAX_NAME      :: 64

DEFAULT_GRAPH_SIZE :: 2048

Context               :: struct{}
backend               :: struct{}
backend_t             :: ^backend
backend_dev           :: struct{}
backend_dev_t         :: ^backend_dev
backend_buffer        :: struct{}
backend_buffer_t      :: ^backend_buffer
backend_buffer_type   :: struct{}
backend_buffer_type_t :: ^backend_buffer_type
backend_sched         :: struct{}
backend_sched_t       :: ^backend_sched
gallocr               :: struct{}
gallocr_t             :: ^gallocr
cgraph                :: struct{}
opt_dataset           :: struct{}
opt_dataset_t         :: ^opt_dataset
opt_context           :: struct{}
opt_context_t         :: ^opt_context
opt_result            :: struct{}
opt_result_t          :: ^opt_result

status :: enum c.int {
	ALLOC_FAILED = -2,
	FAILED = -1,
	SUCCESS = 0,
	ABORTED = 1,
}

object_type :: enum c.int {
	TENSOR,
	GRAPH,
	WORK_BUFFER,
}

type :: enum c.int {
	F32     = 0,
	F16     = 1,
	Q4_0    = 2,
	Q4_1    = 3,
	// Q4_2 = 4, support has been removed
	// Q4_3 = 5, support has been removed
	Q5_0    = 6,
	Q5_1    = 7,
	Q8_0    = 8,
	Q8_1    = 9,
	Q2_K    = 10,
	Q3_K    = 11,
	Q4_K    = 12,
	Q5_K    = 13,
	Q6_K    = 14,
	Q8_K    = 15,
	IQ2_XXS = 16,
	IQ2_XS  = 17,
	IQ3_XXS = 18,
	IQ1_S   = 19,
	IQ4_NL  = 20,
	IQ3_S   = 21,
	IQ2_S   = 22,
	IQ4_XS  = 23,
	I8      = 24,
	I16     = 25,
	I32     = 26,
	I64     = 27,
	F64     = 28,
	IQ1_M   = 29,
	BF16    = 30,
	// Q4_0_4_4 = 31, support has been removed from gguf files
	// Q4_0_4_8 = 32,
	// Q4_0_8_8 = 33,
	TQ1_0   = 34,
	TQ2_0   = 35,
	// IQ4_NL_4_4 = 36,
	// IQ4_NL_4_8 = 37,
	// IQ4_NL_8_8 = 38,
	COUNT   = 39,
}

op :: enum c.int {
	NONE = 0,

	DUP,
	ADD,
	ADD1,
	ACC,
	SUB,
	MUL,
	DIV,
	SQR,
	SQRT,
	LOG,
	SIN,
	COS,
	SUM,
	SUM_ROWS,
	MEAN,
	ARGMAX,
	COUNT_EQUAL,
	REPEAT,
	REPEAT_BACK,
	CONCAT,
	SILU_BACK,
	NORM, // normalize
	RMS_NORM,
	RMS_NORM_BACK,
	GROUP_NORM,

	MUL_MAT,
	MUL_MAT_ID,
	OUT_PROD,

	SCALE,
	SET,
	CPY,
	CONT,
	RESHAPE,
	VIEW,
	PERMUTE,
	TRANSPOSE,
	GET_ROWS,
	GET_ROWS_BACK,
	DIAG,
	DIAG_MASK_INF,
	DIAG_MASK_ZERO,
	SOFT_MAX,
	SOFT_MAX_BACK,
	ROPE,
	ROPE_BACK,
	CLAMP,
	CONV_TRANSPOSE_1D,
	IM2COL,
	IM2COL_BACK,
	CONV_TRANSPOSE_2D,
	POOL_1D,
	POOL_2D,
	POOL_2D_BACK,
	UPSCALE, // nearest interpolate
	PAD,
	PAD_REFLECT_1D,
	ARANGE,
	TIMESTEP_EMBEDDING,
	ARGSORT,
	LEAKY_RELU,

	FLASH_ATTN_EXT,
	FLASH_ATTN_BACK,
	SSM_CONV,
	SSM_SCAN,
	WIN_PART,
	WIN_UNPART,
	GET_REL_POS,
	ADD_REL_POS,
	RWKV_WKV6,
	GATED_LINEAR_ATTN,

	UNARY,

	MAP_UNARY,
	MAP_BINARY,

	MAP_CUSTOM1_F32,
	MAP_CUSTOM2_F32,
	MAP_CUSTOM3_F32,

	MAP_CUSTOM1,
	MAP_CUSTOM2,
	MAP_CUSTOM3,

	CROSS_ENTROPY_LOSS,
	CROSS_ENTROPY_LOSS_BACK,
	OPT_STEP_ADAMW,

	COUNT,
}

tensor_flag :: enum c.int32_t {
	INPUT,  // ...is an input for the GGML compute graph
	OUTPUT, // ...is an output for the GGML compute graph
	PARAM,  // ...contains trainable parameters
	LOSS,   // ...defines loss for numerical optimization (multiple loss tensors add up)
}

init_params :: struct {
	// memory pool
	mem_size:   c.size_t, // bytes
	mem_buffer: rawptr,   // if NULL, memory will be allocated internally
	no_alloc:   bool,     // don't allocate memory for the tensor data
}

object :: struct {
	offs: c.size_t,
	size: c.size_t,

	next: ^object,

	type: object_type,

	padding: [4]u8,
}

tensor :: struct {
	type: type,

	buffer: ^backend_buffer,

	ne: [MAX_DIMS]c.int64_t, // number of elements
	nb: [MAX_DIMS]c.size_t,  // stride in bytes:
							 // nb[0] = ggml_type_size(type)
							 // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
							 // nb[i] = nb[i-1] * ne[i-1]

	// compute data
	op: op,

	// op params - allocated as int32_t for alignment
	op_params: [MAX_OP_PARAMS / size_of(c.int32_t)]c.int32_t,

	flags: bit_set[tensor_flag; c.int32_t],

	src: [MAX_SRC]^tensor,

	// source tensor and offset for views
	view_src:  ^tensor,
	view_offs: c.size_t,

	data: rawptr,

	name: [MAX_NAME]c.char,

	extra: rawptr, // extra things e.g. for ggml-cuda.cu

	padding: [8]c.char,
}

opt_loss_type :: enum c.int {
	MEAN,
	SUM,
	CROSS_ENTROPY,
	MEAN_SQUARED_ERROR,
}

opt_build_type :: enum c.int {
	FORWARD,
	GRAD,
	OPT,
}

// AdamW optimizer parameters
opt_optimizer_params :: struct {
	adamw: struct {
		alpha: f32, // learning rate
		beta1: f32,
		beta2: f32,
		eps: f32,   // epsilon for numerical stability
		wd: f32,    // weight decay for AdamW, use 0.0f to disable
	}
}

// typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);
opt_get_optimizer_params :: #type proc "c" (userdata: rawptr) -> opt_optimizer_params

opt_epoch_callback :: #type proc "c" (
	train:      bool,          // true after training evaluation, false after validation evaluation
	opt_ctx:    opt_context_t,
	dataset:    opt_dataset_t,
	result:     opt_result_t , // result associated with the dataset subsection
	ibatch:     c.int64_t,     // number of batches that have been evaluated so far
	ibatch_max: c.int64_t,     // total number of batches in this dataset subsection
	t_start_us: c.int64_t,     // time at which the evaluation on the dataset subsection was started
)

opt_params :: struct {
	backend_sched: backend_sched_t, // defines which backends are used to construct the compute graphs

	ctx_compute: ^Context, // created in user code, holds non-static tensors

	// the forward graph is defined by inputs and outputs
	// those tensors and all tensors inbetween are not intended to be reusable between multiple optimization contexts
	inputs:  ^tensor,
	outputs: ^tensor,

	loss_type:  opt_loss_type,
	build_type: opt_build_type,

	opt_period: c.int32_t, // after how many gradient accumulation steps an optimizer step should be done

	get_opt_pars: opt_get_optimizer_params, // callback for calculating optimizer parameters
	get_opt_pars_ud: rawptr,                // userdata for calculating optimizer parameters
}

op_pool :: enum c.int {
	MAX,
	AVG,
	COUNT,
}

backend_dev_type_ :: enum c.int {
	// CPU device using system memory
	CPU,
	// GPU device using dedicated memory
	GPU,
	// accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
	ACCEL,
}

@(default_calling_convention="c", link_prefix="ggml_")
foreign ggml {
	time_init                           :: proc() --- // call this once at the beginning of the program
	init                                :: proc(params: init_params) -> ^Context ---
	reset                               :: proc(ctx: ^Context) ---
	free                                :: proc(ctx: ^Context) ---
	type_size                           :: proc(type: type) -> c.size_t ---
	tensor_overhead                     :: proc() -> c.size_t ---
	graph_overhead                      :: proc() -> c.size_t ---
	graph_overhead_custom               :: proc(size: c.size_t, grads: bool) -> c.size_t ---
	new_tensor                          :: proc(ctx: ^Context, type: type, n_dims: c.int, ne: [^]c.int64_t) -> ^tensor ---
	new_tensor_1d                       :: proc(ctx: ^Context, type: type, ne0: c.int64_t) -> ^tensor ---
	new_tensor_2d                       :: proc(ctx: ^Context, type: type, ne0, ne1: c.int64_t) -> ^tensor ---
	new_tensor_3d                       :: proc(ctx: ^Context, type: type, ne0, ne1, ne2: c.int64_t) -> ^tensor ---
	new_tensor_4d                       :: proc(ctx: ^Context, type: type, ne0, ne1, ne2, ne3: c.int64_t) -> ^tensor ---
	nbytes                              :: proc(tensor: ^tensor) -> c.size_t ---
	new_graph                           :: proc(ctx: ^Context) -> ^cgraph --- // size = GGML_DEFAULT_GRAPH_SIZE, grads = false
	new_graph_custom                    :: proc(ctx: ^Context, size: c.size_t, grads: bool) -> ^cgraph ---
	repeat                              :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	reshape_1d                          :: proc(ctx: ^Context, a: ^tensor, ne0: c.int64_t) -> ^tensor ---
	reshape_2d                          :: proc(ctx: ^Context, a: ^tensor, ne0, ne1: c.int64_t) -> ^tensor ---
	reshape_3d                          :: proc(ctx: ^Context, a: ^tensor, ne0, ne1, ne2: c.int64_t) -> ^tensor ---
	reshape_4d                          :: proc(ctx: ^Context, a: ^tensor, ne0, ne1, ne2, ne3: c.int64_t) -> ^tensor ---
	add                                 :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	sub                                 :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	mul                                 :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	div                                 :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	mul_mat                             :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	pow                                 :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	count_equal                         :: proc(ctx: ^Context, a, b: ^tensor) -> ^tensor ---
	scale                               :: proc(ctx: ^Context, a: ^tensor, b: f32) -> ^tensor ---
	relu                                :: proc(ctx: ^Context, a: ^tensor) -> ^tensor ---
	sigmoid                             :: proc(ctx: ^Context, a: ^tensor) -> ^tensor ---
	soft_max                            :: proc(ctx: ^Context, a: ^tensor) -> ^tensor ---
	argmax                              :: proc(ctx: ^Context, a: ^tensor) -> ^tensor ---
	cross_entropy_loss :: proc(
		ctx: ^Context,
		a: ^tensor,    // logits
		b: ^tensor     // labels
	) -> ^tensor ---
	conv_2d :: proc(
		ctx: ^Context,
		a:   ^tensor,  // convolution kernel
		b:   ^tensor,  // data
		s0:  c.int,    // stride dimension 0
		s1:  c.int,    // stride dimension 1
		p0:  c.int,    // padding dimension 0
		p1:  c.int,    // padding dimension 1
		d0:  c.int,    // dilation dimension 0
		d1:  c.int,    // dilation dimension 1
	) -> ^tensor ---
	// the result will have 2*p0 padding for the first dimension
	// and 2*p1 padding for the second dimension
	pool_2d                             :: proc(ctx: ^Context, a: ^tensor, op: op_pool, k0, k1, s0, s1: c.int, p0, p1: f32) -> ^tensor ---
	// make contiguous
	cont                                :: proc(ctx: ^Context, a: ^tensor) -> ^tensor ---
	permute                             :: proc(ctx: ^Context, a: ^tensor, axis0, axis1, axis2, axis3: c.int) -> ^tensor ---
	set_zero                            :: proc(a: ^tensor) -> ^tensor ---
	build_forward_expand                :: proc(cgraph: ^cgraph, tensor: ^tensor) ---
	build_backward_expand :: proc(
		ctx_static:  ^Context, // context for static gradients (loss + gradient accumulation)
		ctx_compute: ^Context, // context for gradient computation
		cgraph:      ^cgraph,
		accumulate:  bool,     // whether or not gradients should be accumulated, requires static allocation of tensors in ctx_static
	) ---
	graph_compute_with_ctx              :: proc(ctx: ^Context, cgraph: ^cgraph, n_thread: c.int) -> status ---
	graph_dup                           :: proc(ctx: ^Context, graph: ^cgraph) -> ^cgraph ---
	graph_reset                         :: proc(cgraph: ^cgraph) --- // set regular grads + optimizer momenta to 0, set loss grad to 1
	graph_node                          :: proc(cgraph: ^cgraph, i: c.int) -> ^tensor --- // if i < 0, returns nodes[n_nodes + i]
	graph_n_nodes                       :: proc(cgraph: ^cgraph) -> c.int ---
	graph_get_grad                      :: proc(cgraph: ^cgraph, node: ^tensor) -> ^tensor ---
	dup_tensor                          :: proc(ctx: ^Context, src: ^tensor) -> ^tensor ---
	view_tensor                         :: proc(ctx: ^Context, src: ^tensor) -> ^tensor ---
	nelements                           :: proc(tensor: ^tensor) -> c.int64_t ---
	get_f32_nd                          :: proc(tensor: ^tensor, i0, i1, i2, i3: c.int) -> f32 ---
	set_f32_nd                          :: proc(tensor: ^tensor, i0, i1, i2, i3: c.int, value: f32) ---
	get_data                            :: proc(tensor: ^tensor) -> rawptr ---
	get_data_f32                        :: proc(tensor: ^tensor) -> [^]f32 ---
	set_param                           :: proc(ctx: ^Context, tensor: ^tensor) ---
	set_input                           :: proc(tensor: ^tensor) ---
	set_output                          :: proc(tensor: ^tensor) ---
	set_loss                            :: proc(tensor: ^tensor) ---
	set_name                            :: proc(t: ^tensor, name: cstring) -> ^tensor ---
	backend_cpu_init                    :: proc() -> backend_t ---
	backend_cuda_init                   :: proc(device: c.int) -> backend_t ---
	backend_free                        :: proc(backend: backend_t) ---
	backend_alloc_ctx_tensors           :: proc(ctx: ^Context, backend: backend_t) -> ^backend_buffer ---
	backend_alloc_ctx_tensors_from_buft :: proc(ctx: ^Context, buft: backend_buffer_type_t) -> ^backend_buffer ---
	backend_tensor_set                  :: proc(tensor: ^tensor, data: rawptr, offset, size: c.size_t) ---
	backend_tensor_get                  :: proc(tensor: ^tensor, data: rawptr, offset, size: c.size_t) ---
	backend_get_default_buffer_type     :: proc(backend: backend_t) -> backend_buffer_type_t ---
	backend_is_cpu                      :: proc(backend: backend_t) -> bool ---
	backend_is_cuda                     :: proc(backend: backend_t) -> bool ---
	backend_cpu_set_n_threads           :: proc(backend_cpu: backend_t, n_threads: c.int) ---
	backend_cpu_buffer_type             :: proc() -> backend_buffer_type_t ---
	backend_graph_compute               :: proc(backend: backend_t, cgraph: ^cgraph) -> status ---
	backend_buffer_free                 :: proc(buffer: backend_buffer_t) ---
	gallocr_new                         :: proc(buft: backend_buffer_type_t) -> gallocr_t ---
	gallocr_free                        :: proc(galloc: gallocr_t) ---
	gallocr_reserve                     :: proc(galloc: gallocr_t, graph: ^cgraph) -> bool ---
	gallocr_alloc_graph                 :: proc(galloc: gallocr_t, graph: ^cgraph) -> bool ---
	backend_init_by_name                :: proc(name, params: cstring) -> backend_t ---
	backend_dev_init                    :: proc(device: backend_dev_t, params: cstring) -> backend_t ---
	backend_dev_name                    :: proc(device: backend_dev_t) -> cstring ---
	backend_dev_count                   :: proc() -> c.size_t ---
	backend_dev_get                     :: proc(index: c.size_t) -> backend_dev_t ---
	backend_dev_type                    :: proc(device: backend_dev_t) -> backend_dev_type_ ---
	backend_dev_by_name                 :: proc(name: cstring) -> backend_dev_t ---
	backend_sched_new                   :: proc(backends: [^]backend_t, bufts: [^]backend_buffer_type_t, n_backends: c.int, graph_size: c.size_t, parallel: bool) -> backend_sched_t ---
	backend_sched_free                  :: proc(sched: backend_sched_t) ---
	backend_sched_reset                 :: proc(sched: backend_sched_t) ---
	backend_sched_get_n_backends        :: proc(sched: backend_sched_t) -> c.int ---
	backend_sched_get_backend           :: proc(sched: backend_sched_t, i: c.int) -> backend_t ---
	backend_sched_alloc_graph           :: proc(sched: backend_sched_t, graph: ^cgraph) -> bool --- // returns success
	backend_sched_graph_compute         :: proc(sched: backend_sched_t, graph: ^cgraph) -> status ---
	// AdamW optimizer step
	// Paper: https://arxiv.org/pdf/1711.05101v3.pdf
	// PyTorch: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
	opt_step_adamw                      :: proc(ctx: ^Context, a, grad, m, v, adamw_params: ^tensor) -> ^tensor --- // parameters such as the learning rate
	opt_dataset_init :: proc(
		ne_datapoint: c.int64_t, // number of elements per datapoint
		ne_label:     c.int64_t, // number of elements per label
		ndata:        c.int64_t, // total number of datapoints/labels
		ndata_shard:  c.int64_t, // number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
	) -> opt_dataset_t ---
	opt_dataset_free                    :: proc(dataset: opt_dataset_t) ---
	opt_dataset_data                    :: proc(dataset: opt_dataset_t) -> ^tensor --- // shape = [ne_datapoint, ndata]
	opt_dataset_labels                  :: proc(dataset: opt_dataset_t) -> ^tensor --- // shape = [nd_label,     ndata]
	opt_default_params                  :: proc(backend_sched: backend_sched_t, ctx_compute: ^Context, inputs, outputs: ^tensor, loss_type: opt_loss_type) -> opt_params ---
	opt_get_default_optimizer_params    :: proc(userdata: rawptr) -> opt_optimizer_params ---
	opt_init                            :: proc(params: opt_params) -> opt_context_t ---
	opt_free                            :: proc(opt_ctx: opt_context_t) ---
	opt_epoch :: proc(
		opt_ctx:        opt_context_t,
		dataset:        opt_dataset_t,
		result_train:   opt_result_t,       // result to increment during training, ignored if NULL
		result_eval:    opt_result_t,       // result to increment during evaluation, ignored if NULL
		idata_split:    c.int64_t,          // data index at which to split training and evaluation
		callback_train: opt_epoch_callback,
		callback_eval:  opt_epoch_callback,
	) ---
	opt_fit :: proc(
		backend_sched:  backend_sched_t,          // backend scheduler for constructing the compute graphs
		ctx_compute:    ^Context,                 // context with temporarily allocated tensors to calculate the outputs
		inputs:         ^tensor,                  // input tensor with shape [ne_datapoint, ndata_batch]
		outputs:        ^tensor,                  // output tensor, must have shape [ne_label, ndata_batch] if labels are used
		dataset:        opt_dataset_t,            // dataset with data and optionally also labels
		loss_type:      opt_loss_type,            // loss to minimize
		get_opt_pars:   opt_get_optimizer_params, // callback to get optimizer params, userdata is pointer to epoch (of type int64_t)
		nepoch:         c.int64_t,                // how many times the dataset should be iterated over
		nbatch_logical: c.int64_t,                // datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs
		val_split:      f32,                      // fraction of the dataset to use for validation, must be in [0.0f, 1.0f)
		silent:         bool,                     // whether or not info prints to stderr should be suppressed
	) ---
	opt_dataset_get_batch :: proc(
		dataset:      opt_dataset_t,
		data_batch:   ^tensor,       // shape = [ne_datapoint, ndata_batch]
		labels_batch: ^tensor,       // shape = [ne_label,     ndata_batch]
		ibatch:        c.int64_t,
	) ---
	opt_result_init                     :: proc() -> opt_result_t ---
	opt_result_free                     :: proc(result: opt_result_t) ---
	opt_result_pred                     :: proc(result: opt_result_t, pred: ^c.int32_t) --- // writes ndata values
	opt_result_accuracy                 :: proc(result: opt_result_t, accuracy, unc: ^f64) --- // writes 1 value
}