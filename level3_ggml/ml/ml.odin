package ml

import "base:runtime"
import "base:builtin"
import "core:c"
import "core:fmt"
import "core:mem"
import "core:math"
import "core:strings"
import "ggml"

DEFAULT_GRAPH_SIZE :: ggml.DEFAULT_GRAPH_SIZE

free :: proc {
	_free_scheduler,
	_free_context,
	_free_adamw_optimizer,
}
len :: proc {
	_tensor_len,
	_dimension_len,
}
clone :: proc {
	_clone_tensor,
	_clone_graph,
}
reset :: proc {
	_reset_scheduler,
	_reset_graph,
}

Data_Type :: enum {
	F32,
	F16,
	Q4_0,
	Q4_1,
	Q5_0,
	Q5_1,
	Q8_0,
	Q8_1,
	Q2_K,
	Q3_K,
	Q4_K,
	Q5_K,
	Q6_K,
	Q8_K,
	Iq2_Xxs,
	Iq2_Xs,
	Iq3_Xxs,
	Iq1_S,
	Iq4_Nl,
	Iq3_S,
	Iq2_S,
	Iq4_Xs,
	I8,
	I16,
	I32,
	I64,
	F64,
	Iq1_M,
	Bf16,
	Tq1_0,
	Tq2_0,
}
_type_sanity_check :: proc(T: typeid, type: ggml.type) {
	switch T {
	case f16: assert(type == .F16, "Tensor type mismatch")
	case f32: assert(type == .F32, "Tensor type mismatch")
	case f64: assert(type == .F64, "Tensor type mismatch")
	case i8:  assert(type == .I8,  "Tensor type mismatch")
	case i16: assert(type == .I16, "Tensor type mismatch")
	case i32: assert(type == .I32, "Tensor type mismatch")
	case i64: assert(type == .I64, "Tensor type mismatch")
	}
}
_to_ggml_type :: proc(type: Data_Type) -> (res: ggml.type) {
	switch type {
	case .F32:     res = .F32
	case .F16:     res = .F16
	case .Q4_0:    res = .Q4_0
	case .Q4_1:    res = .Q4_1
	case .Q5_0:    res = .Q5_0
	case .Q5_1:    res = .Q5_1
	case .Q8_0:    res = .Q8_0
	case .Q8_1:    res = .Q8_1
	case .Q2_K:    res = .Q2_K
	case .Q3_K:    res = .Q3_K
	case .Q4_K:    res = .Q4_K
	case .Q5_K:    res = .Q5_K
	case .Q6_K:    res = .Q6_K
	case .Q8_K:    res = .Q8_K
	case .Iq2_Xxs: res = .IQ2_XXS
	case .Iq2_Xs:  res = .IQ2_XS
	case .Iq3_Xxs: res = .IQ3_XXS
	case .Iq1_S:   res = .IQ1_S
	case .Iq4_Nl:  res = .IQ4_NL
	case .Iq3_S:   res = .IQ3_S
	case .Iq2_S:   res = .IQ2_S
	case .Iq4_Xs:  res = .IQ4_XS
	case .I8:      res = .I8
	case .I16:     res = .I16
	case .I32:     res = .I32
	case .I64:     res = .I64
	case .F64:     res = .F64
	case .Iq1_M:   res = .IQ1_M
	case .Bf16:    res = .BF16
	case .Tq1_0:   res = .TQ1_0
	case .Tq2_0:   res = .TQ2_0
	}
	return
}

Device :: string
_get_backend :: proc(device: Device) -> (backend: ^ggml.backend) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	backend = ggml.backend_init_by_name(strings.clone_to_cstring(device, context.temp_allocator), nil)
	assert(backend != nil, fmt.tprintfln("Device %v could not be initialized.", device))
	return
}
print_devices :: proc() {
	for i in 0 ..< ggml.backend_dev_count() {
		fmt.println(ggml.backend_dev_name(ggml.backend_dev_get(i)))
	}
}

Scheduler :: struct {
	devices:  []Device,
	backends: []^ggml.backend,
	handle:   ^ggml.backend_sched,
}
@(require_results)
new_scheduler :: proc(devices: []Device, graph_size := DEFAULT_GRAPH_SIZE, parallel := false) -> (scheduler: Scheduler) {
	assert(devices[builtin.len(devices) - 1] == "CPU", "CPU must be the last device in the scheduler.")
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	scheduler.backends = make([]^ggml.backend, builtin.len(devices))
	scheduler.devices = make([]Device, builtin.len(devices))
	for i in 0 ..< builtin.len(devices) {
		scheduler.backends[i] = _get_backend(devices[i])
		scheduler.devices[i] = devices[i]
	}
	scheduler.handle = ggml.backend_sched_new(raw_data(scheduler.backends), nil, i32(builtin.len(scheduler.backends)), uint(graph_size), parallel)
	return
}
_free_scheduler :: proc(scheduler: Scheduler) {
	ggml.backend_sched_free(scheduler.handle)
	for backend in scheduler.backends {
		ggml.backend_free(backend)
	}
	delete(scheduler.backends)
	delete(scheduler.devices)
}
_reset_scheduler :: proc(scheduler: Scheduler) {
	ggml.backend_sched_reset(scheduler.handle)
}
alloc_graph :: proc(scheduler: Scheduler, graph: Graph) -> bool {
	return ggml.backend_sched_alloc_graph(scheduler.handle, graph)
}
compute :: proc(scheduler: Scheduler, graph: Graph) {
	ggml.backend_sched_graph_compute(scheduler.handle, graph)
}

Context :: struct {
	handle:        ^ggml.Context,
	backend:       ^ggml.backend,
	static_buffer: ^ggml.backend_buffer,
}
@(require_results)
new_context :: proc(size: int) -> (res: Context) {
	res.handle = ggml.init({uint(size), nil, true})
	return
}
_free_context :: proc(ctx: Context) {
	ggml.free(ctx.handle)
	ggml.backend_buffer_free(ctx.static_buffer)
}
clear :: proc(ctx: ^Context) {
	if ctx.handle != nil {
		ggml.free(ctx.handle)
		ctx.handle = nil
	}
	if ctx.backend != nil {
		ggml.backend_free(ctx.backend)
		ctx.backend = nil
	}
	if ctx.static_buffer != nil {
		ggml.backend_buffer_free(ctx.static_buffer)
		ctx.static_buffer = nil
	}
}
alloc_tensors :: proc(ctx: ^Context, device := "CPU") {
	if ctx.backend != nil {
		ggml.backend_free(ctx.backend)
	}
	if ctx.static_buffer != nil {
		ggml.backend_buffer_free(ctx.static_buffer)
	}
	ctx.backend = _get_backend(device)
	ctx.static_buffer = ggml.backend_alloc_ctx_tensors(ctx.handle, _get_backend(device))
}

Tensor_Flag :: enum {
	Input,
	Output,
	Parameter,
	Loss,
}
Tensor :: ^ggml.tensor
@(require_results)
_tensor1 :: proc(ctx: Context, type: Data_Type, len0: int) -> (res: Tensor) {
	ne := [4]i64{i64(len0), 1, 1, 1}
	return ggml.new_tensor(ctx.handle, _to_ggml_type(type), 1, &ne[0])
}
@(require_results)
_tensor2 :: proc(ctx: Context, type: Data_Type, len0, len1: int) -> (res: Tensor) {
	ne := [4]i64{i64(len0), i64(len1), 1, 1}
	return ggml.new_tensor(ctx.handle, _to_ggml_type(type), 2, &ne[0])
}
@(require_results)
_tensor3 :: proc(ctx: Context, type: Data_Type, len0, len1, len2: int) -> (res: Tensor) {
	ne := [4]i64{i64(len0), i64(len1), i64(len2), 1}
	return ggml.new_tensor(ctx.handle, _to_ggml_type(type), 3, &ne[0])
}
@(require_results)
_tensor4 :: proc(ctx: Context, type: Data_Type, len0, len1, len2, len3: int) -> (res: Tensor) {
	ne := [4]i64{i64(len0), i64(len1), i64(len2), i64(len3)}
	return ggml.new_tensor(ctx.handle, _to_ggml_type(type), 4, &ne[0])
}
tensor :: proc {
	_tensor1,
	_tensor2,
	_tensor3,
	_tensor4,
}
@(require_results)
_tensor_len :: proc(tensor: Tensor) -> int {
	return int(ggml.nelements(tensor))
}
@(require_results)
_dimension_len :: proc(tensor: Tensor, dimension: int) -> int {
	return int(tensor.ne[dimension])
}
@(require_results)
slice_data :: proc(tensor: Tensor, $T: typeid) -> (data: []T) {
	_type_sanity_check(T, tensor.type)
	return mem.slice_ptr(cast([^]T)tensor.data, len(tensor))
}
@(require_results)
get_data :: proc(tensor: Tensor, $T: typeid, allocator := context.allocator) -> (data: []T) {
	_type_sanity_check(T, tensor.type)
	data = make([]T, len(tensor), allocator)
	ggml.backend_tensor_get(tensor, raw_data(data), 0, ggml.nbytes(tensor))
	return
}
set_data :: proc(tensor: Tensor, data: []$T) {
	_type_sanity_check(T, tensor.type)
	assert(builtin.len(data) == len(tensor))
	ggml.backend_tensor_set(tensor, raw_data(data), 0, ggml.nbytes(tensor))
}
set_name :: proc(tensor: Tensor, name: cstring) {
	ggml.set_name(tensor, name)
}
flags :: proc(tensor: Tensor) -> bit_set[Tensor_Flag] {
	return transmute(bit_set[Tensor_Flag])u8(transmute(i32)tensor.flags)
}
set_flags :: proc(tensor: Tensor, flags: bit_set[Tensor_Flag]) {
	tensor.flags = transmute(bit_set[ggml.tensor_flag; c.int32_t])c.int32_t(transmute(u8)flags)
}
set_input :: proc(tensor: Tensor) {
	ggml.set_input(tensor)
}
set_output :: proc(tensor: Tensor) {
	ggml.set_output(tensor)
}
set_parameter :: proc(tensor: Tensor) {
	f := flags(tensor)
	f += {.Parameter}
	set_flags(tensor, f)
}
set_loss :: proc(tensor: Tensor) {
	ggml.set_loss(tensor)
}

@(require_results)
tensor_overhead :: proc() -> int {
	return int(ggml.tensor_overhead())
}
@(require_results)
graph_overhead :: proc(size := DEFAULT_GRAPH_SIZE, grads := false) -> int {
	return int(ggml.graph_overhead_custom(uint(size), grads))
}

@(require_results)
_clone_tensor :: proc(ctx: Context, tensor: Tensor) -> Tensor {
	return ggml.dup_tensor(ctx.handle, tensor)
}
@(require_results)
relu :: proc(ctx: Context, tensor: Tensor) -> Tensor {
	return ggml.relu(ctx.handle, tensor)
}
@(require_results)
add :: proc(ctx: Context, a, b: Tensor) -> Tensor {
	return ggml.add(ctx.handle, a, b)
}
@(require_results)
matmul :: proc(ctx: Context, a, b: Tensor) -> Tensor {
	return ggml.mul_mat(ctx.handle, a, b)
}
@(require_results)
count_equal :: proc(ctx: Context, a, b: Tensor) -> Tensor {
	return ggml.count_equal(ctx.handle, a, b)
}
@(require_results)
scale :: proc(ctx: Context, a: Tensor, b: f32) -> Tensor {
	return ggml.scale(ctx.handle, a, b)
}
@(require_results)
softmax :: proc(ctx: Context, tensor: Tensor) -> Tensor {
	return ggml.soft_max(ctx.handle, tensor)
}
@(require_results)
argmax :: proc(ctx: Context, tensor: Tensor) -> Tensor {
	return ggml.argmax(ctx.handle, tensor)
}
@(require_results)
cross_entropy_loss :: proc(ctx: Context, a, b: Tensor) -> Tensor {
	return ggml.cross_entropy_loss(ctx.handle, a, b)
}
@(require_results)
step_adamw :: proc(ctx: Context, a, grad, m, v, adamw_parameters: Tensor) -> Tensor {
	return ggml.opt_step_adamw(ctx.handle, a, grad, m, v, adamw_parameters)
}

Graph :: ^ggml.cgraph
@(require_results)
graph :: proc(ctx: Context, size := DEFAULT_GRAPH_SIZE, grads := false) -> Graph {
	return ggml.new_graph_custom(ctx.handle, c.size_t(size), grads)
}
forward :: proc(graph: Graph, tensor: Tensor) {
	ggml.build_forward_expand(graph, tensor)
}
backward :: proc(ctx_static, ctx_compute: Context, graph: Graph, accumulate: bool) {
	ggml.build_backward_expand(ctx_static.handle, ctx_compute.handle, graph, accumulate)
}
_reset_graph :: proc(graph: Graph) {
	ggml.graph_reset(graph)
}
@(require_results)
_clone_graph :: proc(ctx: Context, graph: Graph) -> Graph {
	return ggml.graph_dup(ctx.handle, graph)
}
@(require_results)
node_count :: proc(graph: Graph) -> int {
	return int(ggml.graph_n_nodes(graph))
}
@(require_results)
get_node :: proc(graph: Graph, index: int) -> Tensor {
	return ggml.graph_node(graph, c.int(index))
}
@(require_results)
get_grad :: proc(graph: Graph, node: Tensor) -> Tensor {
	return ggml.graph_get_grad(graph, node)
}

Adamw_Optimizer :: struct {
	scheduler: Scheduler,

	ctx_static:  Context,
	ctx_compute: Context,

	input:      Tensor,
	output:     Tensor,
	target:     Tensor,
	loss:       Tensor,
	prediction: Tensor,
	score:      Tensor,
	parameters: Tensor,

	graph: Graph,

	alpha:        f32,
	beta1:        f32,
	beta2:        f32,
	epsilon:      f32,
	weight_decay: f32,

	batch_len: int,
	iteration: int,
}
@(require_results)
new_adamw_optimizer :: proc(
	devices:   []Device,
	model:     rawptr,
	input:     Tensor,
	build:     proc(ctx_compute: Context, model: rawptr, input: Tensor) -> Tensor,
	batch_len: int,
) -> (opt: Adamw_Optimizer) {
	opt.alpha        = 0.001
	opt.beta1        = 0.9
	opt.beta2        = 0.999
	opt.epsilon      = 1e-8
	opt.weight_decay = 0

	opt.batch_len = batch_len
	opt.iteration = 1
	opt.scheduler = new_scheduler(devices)

	opt.ctx_compute = new_context(DEFAULT_GRAPH_SIZE * tensor_overhead() + graph_overhead())

	opt.input = input
	set_input(opt.input)

	opt.output = build(opt.ctx_compute, model, opt.input)
	set_name(opt.output, "output")
	set_output(opt.output)

	opt.graph = graph(opt.ctx_compute, grads=true)
	forward(opt.graph, opt.output)

	parameter_count := 0
	for i in 0 ..< node_count(opt.graph) {
		if .Parameter in flags(get_node(opt.graph, i)) {
			parameter_count += 1
		}
	}
	opt.ctx_static = new_context((3 * parameter_count + 9 + 1) * tensor_overhead())

	opt.target = clone(opt.ctx_static, opt.output)
	set_name(opt.target, "target")
	set_input(opt.target)

	opt.loss = cross_entropy_loss(opt.ctx_static, opt.output, opt.target)
	set_name(opt.target, "loss")
	set_output(opt.loss)
	set_loss(opt.loss)
	forward(opt.graph, opt.loss)

	opt.prediction = argmax(opt.ctx_static, opt.output)
	set_name(opt.prediction, "prediction")
	forward(opt.graph, opt.prediction)

	opt.score = count_equal(opt.ctx_static, opt.prediction, argmax(opt.ctx_static, opt.target))
	set_name(opt.score, "score")
	forward(opt.graph, opt.score)

	backward(opt.ctx_static, opt.ctx_compute, opt.graph, false)

	opt.parameters = tensor(opt.ctx_static, .F32, 7)
	set_name(opt.parameters, "adamw_parameters")
	set_input(opt.parameters)

	for i := node_count(opt.graph) - 1; i >= 0; i -= 1 {
		node := get_node(opt.graph, i)
		grad := get_grad(opt.graph, node)
		if .Parameter in flags(node) {
			m := clone(opt.ctx_static, node)
			v := clone(opt.ctx_static, node)
			opt_step := step_adamw(opt.ctx_compute, node, grad, m, v, opt.parameters)
			forward(opt.graph, opt_step)
		}
	}

	alloc_tensors(&opt.ctx_static, devices[0])
	reset(opt.graph)

	return
}
_free_adamw_optimizer :: proc(opt: Adamw_Optimizer) {
	free(opt.ctx_static)
	free(opt.ctx_compute)
	free(opt.scheduler)
}
epoch :: proc(opt: ^Adamw_Optimizer, input, target: Tensor) {
	input_batch_len  := opt.batch_len * len(input, 0)
	target_batch_len := opt.batch_len * len(target, 0)
	batch_count      := len(input, 1) / opt.batch_len
	for i in 0 ..< batch_count {
		// beta1, beta2 after applying warmup
		beta1h := 1.0 / (1.0 - math.pow(opt.beta1, f32(opt.iteration)))
		beta2h := 1.0 / (1.0 - math.pow(opt.beta2, f32(opt.iteration)))
		set_data(opt.parameters, []f32{opt.alpha, opt.beta1, opt.beta2, opt.epsilon, opt.weight_decay, beta1h, beta2h})

		set_data(opt.input,  slice_data(input,  f32)[i*input_batch_len:][:input_batch_len])
		set_data(opt.target, slice_data(target, f32)[i*target_batch_len:][:target_batch_len])
		compute(opt.scheduler, opt.graph)

		opt.iteration += 1
	}
}