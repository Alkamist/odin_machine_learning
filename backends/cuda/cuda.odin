package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:log"
import "core:sync"

import "bindings/cuda"
import "bindings/cublas"

import ml   "../.."
import pool "../activation_pool"

Gpu_Device :: struct {
	dev:       cuda.Device,
	ctx:       cuda.Context,

	cc_major, cc_minor: i32,
	sm_count:           i32,
	warp_size:          i32,
	max_threads_per_sm: i32,
	max_smem_per_block: i32,
	max_smem_optin:     i32,

	device_name: string,

	pipeline_cache: map[string]^Pipeline,
}

Context :: struct {
	using _: ml.Context,

	stream:        cuda.Stream,
	cublas_handle: cublas.Handle,

	activation_pool: pool.Pool(cuda.DevicePtr),

	persistent: [dynamic]cuda.DevicePtr,

	timing_enabled: bool,
	timing_totals:  map[^Pipeline]Timing_Stat,
	timing_pool:    [dynamic]Timing_Slot,
	timing_cursor:  int,

	graph_capturing: bool,
	graph_exec:      cuda.GraphExec,
	graph_handle:    cuda.Graph,

	auto_graph_enabled: bool,
	auto_capturing:     bool,
	auto_warmup_done:   bool,
	auto_exec:          cuda.GraphExec,

	q8_1_cache:    map[cuda.DevicePtr]cuda.DevicePtr,
	dequant_cache: map[cuda.DevicePtr]cuda.DevicePtr, // Q4_K/Q6_K weight ptr -> bf16 dequantized scratch

	k_cache_written_this_forward: map[cuda.DevicePtr]bool,
	v_cache_written_this_forward: map[cuda.DevicePtr]bool,

	position_pinned:                rawptr,
	position_dev:                   cuda.DevicePtr,
	position_written_this_forward:  bool,

	shift_scratch_dev:  cuda.DevicePtr,
	shift_scratch_size: u64,
}

_activation_pool_ops :: proc(gctx: ^Context) -> pool.Ops(cuda.DevicePtr) {
	return {user = gctx, alloc = _activation_pool_alloc, free = _activation_pool_free}
}

_activation_pool_alloc :: proc(user: rawptr, size: u64, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	ptr: cuda.DevicePtr
	cuda.check(cuda.MemAlloc(&ptr, uint(size)), loc=loc)
	return ptr
}

_activation_pool_free :: proc(user: rawptr, handle: cuda.DevicePtr, loc: runtime.Source_Code_Location) {
	cuda.check(cuda.MemFree(handle), loc=loc)
}

Timing_Stat :: struct {
	total_ns: i64,
	count:    int,
}

Timing_Slot :: struct {
	pipeline: ^Pipeline,
	start:    cuda.Event,
	end:      cuda.Event,
}

_gpu:       Gpu_Device
_gpu_mutex: sync.Mutex

@(require_results)
_gctx :: #force_inline proc(loc := #caller_location) -> ^Context {
	return cast(^Context)ml.current_context(loc=loc)
}

device_init :: proc(loc := #caller_location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)
	_device_init_locked(loc)
}

_device_init_locked :: proc(loc := #caller_location) {
	if _gpu.ctx != nil {
		return
	}

	cuda.check(cuda.Init(0))

	count: i32
	cuda.check(cuda.DeviceGetCount(&count))
	assert(count > 0, "no CUDA-capable devices found", loc=loc)

	cuda.check(cuda.DeviceGet(&_gpu.dev, 0))

	name_buf: [128]u8
	cuda.check(cuda.DeviceGetName(raw_data(name_buf[:]), i32(builtin.len(name_buf)), _gpu.dev))
	name_len := 0
	for c, i in name_buf {
		if c == 0 {
			name_len = i
			break
		}
	}
	owned := builtin.make([]u8, name_len)
	builtin.copy(owned, name_buf[:name_len])
	_gpu.device_name = builtin.string(owned)

	cuda.check(cuda.DeviceGetAttribute(&_gpu.cc_major,           .COMPUTE_CAPABILITY_MAJOR,          _gpu.dev))
	cuda.check(cuda.DeviceGetAttribute(&_gpu.cc_minor,           .COMPUTE_CAPABILITY_MINOR,          _gpu.dev))
	cuda.check(cuda.DeviceGetAttribute(&_gpu.sm_count,           .MULTIPROCESSOR_COUNT,              _gpu.dev))
	cuda.check(cuda.DeviceGetAttribute(&_gpu.warp_size,          .WARP_SIZE,                         _gpu.dev))
	cuda.check(cuda.DeviceGetAttribute(&_gpu.max_threads_per_sm, .MAX_THREADS_PER_MULTIPROCESSOR,    _gpu.dev))
	cuda.check(cuda.DeviceGetAttribute(&_gpu.max_smem_per_block, .MAX_SHARED_MEMORY_PER_BLOCK,       _gpu.dev))
	cuda.check(cuda.DeviceGetAttribute(&_gpu.max_smem_optin,     .MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, _gpu.dev))

	cuda.check(cuda.CtxCreate(&_gpu.ctx, cuda.CTX_SCHED_BLOCKING_SYNC, _gpu.dev))

	_gpu.pipeline_cache = builtin.make(map[string]^Pipeline)

	log.infof("%s  cc=%d.%d  SMs=%d  warp=%d", _gpu.device_name, _gpu.cc_major, _gpu.cc_minor, _gpu.sm_count, _gpu.warp_size)
}

device_destroy :: proc() {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	for _, p in _gpu.pipeline_cache {
		_destroy_pipeline(p)
	}
	builtin.delete(_gpu.pipeline_cache)
	_gpu.pipeline_cache = nil

	if _gpu.ctx != nil {
		cuda.CtxDestroy(_gpu.ctx)
		_gpu.ctx = nil
	}
	if _gpu.device_name != "" {
		builtin.delete(_gpu.device_name)
		_gpu.device_name = ""
	}
}

device_name :: proc() -> string {
	return _gpu.device_name
}

@(require_results)
// decode_graph arms CUDA auto-graph capture: after a warmup step, single-token
// decode replays a captured graph instead of re-issuing every kernel launch, so
// pass true for inference. Leave false for training or when using enable_timing
// (graph capture cannot coexist with cuEventRecord).
context_create :: proc(decode_graph := false, allocator := context.allocator, loc := #caller_location) -> ^ml.Context {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	_device_init_locked()

	gctx, err := builtin.new(Context, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate Context: %v", err, loc=loc)

	gctx.auto_graph_enabled = decode_graph

	cuda.check(cuda.StreamCreate(&gctx.stream, cuda.STREAM_NON_BLOCKING))
	cublas.check(cublas.Create_v2(&gctx.cublas_handle))
	cublas.check(cublas.SetStream_v2(gctx.cublas_handle, gctx.stream))

	cuda.check(cuda.MemAllocHost(&gctx.position_pinned, 4))
	cuda.check(cuda.MemAlloc(&gctx.position_dev, 4))
	cuda.check(cuda.MemsetD32(gctx.position_dev, 0, 1))
	(^i32)(gctx.position_pinned)^ = 0

	ml._context_init(gctx, {
		clear        = clear,
		forward      = forward,
		backward     = backward,
		update       = update,
		buffer_alloc = buffer_alloc,
		buffer_free  = buffer_free,
		buffer_get   = buffer_get,
		buffer_set   = buffer_set,
		buffer_copy  = buffer_copy,

		forward_ops = {
			.Add, .Mul, .Gelu_Mul, .Gelu, .Silu, .Tanh, .Cast, .Linear,
			.Linear_Q4_K, .Linear_Q4_K_Gate_Up_Geglu, .Linear_Q6_K,
			.Rmsnorm, .Add_Rmsnorm, .Rmsnorm_Rope, .Rmsnorm_Rope_Write_Cache,
			.Rope, .Attention, .Attention_Cache, .Cross_Entropy,
			.Select, .Slice_Trailing, .Exp, .Clamp, .Min, .Softmax, .Entropy,
		},
		backward_ops = {
			.Add, .Mul, .Linear, .Linear_Q4_K, .Linear_Q6_K, .Silu, .Gelu,
			.Tanh, .Select, .Slice_Trailing, .Rmsnorm, .Rope, .Attention,
			.Cross_Entropy, .Cast, .Exp, .Clamp, .Min, .Softmax, .Entropy,
		},
	}, allocator, loc)

	return gctx
}

context_destroy :: proc(ctx: ^ml.Context, allocator := context.allocator, loc := #caller_location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	gctx := cast(^Context)ctx

	if gctx.stream != nil {
		cuda.check(cuda.StreamSynchronize(gctx.stream))
	}

	pool.destroy(&gctx.activation_pool, _activation_pool_ops(gctx))

	for ptr in gctx.persistent {
		cuda.MemFree(ptr)
	}
	builtin.delete(gctx.persistent)

	for slot in gctx.timing_pool {
		cuda.EventDestroy(slot.start)
		cuda.EventDestroy(slot.end)
	}
	builtin.delete(gctx.timing_pool)
	delete(gctx.timing_totals)

	if gctx.graph_exec != nil {
		cuda.GraphExecDestroy(gctx.graph_exec)
		gctx.graph_exec = nil
	}
	if gctx.graph_handle != nil {
		cuda.GraphDestroy(gctx.graph_handle)
		gctx.graph_handle = nil
	}
	if gctx.auto_exec != nil {
		cuda.GraphExecDestroy(gctx.auto_exec)
		gctx.auto_exec = nil
	}
	delete(gctx.q8_1_cache)
	delete(gctx.dequant_cache)
	delete(gctx.k_cache_written_this_forward)
	delete(gctx.v_cache_written_this_forward)

	if gctx.position_dev != 0 {
		cuda.MemFree(gctx.position_dev)
		gctx.position_dev = 0
	}
	if gctx.position_pinned != nil {
		cuda.MemFreeHost(gctx.position_pinned)
		gctx.position_pinned = nil
	}
	if gctx.shift_scratch_dev != 0 {
		cuda.MemFree(gctx.shift_scratch_dev)
		gctx.shift_scratch_dev = 0
		gctx.shift_scratch_size = 0
	}

	if gctx.cublas_handle != nil {
		cublas.Destroy_v2(gctx.cublas_handle)
		gctx.cublas_handle = nil
	}

	if gctx.stream != nil {
		cuda.StreamDestroy(gctx.stream)
		gctx.stream = nil
	}

	ml._context_destroy(ctx, loc)
	builtin.free(gctx, allocator=allocator, loc=loc)
}

clear :: proc(loc: runtime.Source_Code_Location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	gctx := _gctx(loc)

	if gctx.auto_capturing {
		_auto_graph_finish(gctx, loc)
	}

	if gctx.stream != nil {
		cuda.check(cuda.StreamSynchronize(gctx.stream))
	}
	if gctx.timing_enabled {
		for i in 0 ..< gctx.timing_cursor {
			slot := gctx.timing_pool[i]
			ms: f32
			cuda.check(cuda.EventElapsedTime(&ms, slot.start, slot.end))
			stat := gctx.timing_totals[slot.pipeline]
			stat.total_ns += i64(ms * 1_000_000)
			stat.count    += 1
			gctx.timing_totals[slot.pipeline] = stat
		}
		gctx.timing_cursor = 0
	}

	pool.reset(&gctx.activation_pool)

	builtin.clear(&gctx.q8_1_cache)
	builtin.clear(&gctx.dequant_cache)

	builtin.clear(&gctx.k_cache_written_this_forward)
	builtin.clear(&gctx.v_cache_written_this_forward)

	gctx.position_written_this_forward = false

	if gctx.auto_graph_enabled && gctx.auto_warmup_done && !gctx.auto_capturing && !gctx.graph_capturing && !gctx.timing_enabled {
		cuda.check(cuda.StreamBeginCapture_v2(gctx.stream, .Relaxed), loc=loc)
		gctx.auto_capturing = true
	}
	if gctx.auto_graph_enabled {
		gctx.auto_warmup_done = true
	}
}

enable_decode_graph :: proc(enabled: bool, loc := #caller_location) {
	gctx := _gctx(loc)
	if enabled {
		fmt.assertf(!gctx.timing_enabled, "cannot combine with enable_timing", loc=loc)
		fmt.assertf(!gctx.graph_capturing, "explicit graph capture in progress", loc=loc)
	}
	gctx.auto_graph_enabled = enabled
}

enable_timing :: proc(enabled: bool) {
	gctx := _gctx()
	gctx.timing_enabled = enabled
}

Timing_Entry :: struct {
	name:     string,
	total_ns: i64,
	count:    int,
}

@(require_results)
timing_snapshot :: proc(allocator := context.allocator) -> []Timing_Entry {
	gctx := _gctx()
	entries := builtin.make([]Timing_Entry, builtin.len(gctx.timing_totals), allocator)
	i := 0
	for p, stat in gctx.timing_totals {
		entries[i] = {name=p.name, total_ns=stat.total_ns, count=stat.count}
		i += 1
	}
	for outer in 1 ..< builtin.len(entries) {
		j := outer
		for j > 0 && entries[j].total_ns > entries[j - 1].total_ns {
			entries[j], entries[j - 1] = entries[j - 1], entries[j]
			j -= 1
		}
	}
	return entries
}

reset_timing :: proc() {
	gctx := _gctx()
	builtin.clear(&gctx.timing_totals)
}

begin_graph_capture :: proc(loc := #caller_location) {
	gctx := _gctx(loc)
	fmt.assertf(!gctx.graph_capturing, "capture already in progress", loc=loc)
	fmt.assertf(!gctx.timing_enabled, "disable timing first (cuEventRecord is uncapturable)", loc=loc)

	cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
	cuda.check(cuda.StreamBeginCapture_v2(gctx.stream, .Relaxed), loc=loc)
	gctx.graph_capturing = true
}

end_graph_capture :: proc(loc := #caller_location) {
	gctx := _gctx(loc)
	fmt.assertf(gctx.graph_capturing, "no capture in progress", loc=loc)

	if gctx.graph_exec != nil {
		cuda.GraphExecDestroy(gctx.graph_exec)
		gctx.graph_exec = nil
	}
	if gctx.graph_handle != nil {
		cuda.GraphDestroy(gctx.graph_handle)
		gctx.graph_handle = nil
	}

	cuda.check(cuda.StreamEndCapture(gctx.stream, &gctx.graph_handle), loc=loc)
	cuda.check(cuda.GraphInstantiateWithFlags(&gctx.graph_exec, gctx.graph_handle, cuda.GRAPH_INSTANTIATE_DEFAULT), loc=loc)
	gctx.graph_capturing = false
}

replay_graph :: proc(loc := #caller_location) {
	gctx := _gctx(loc)
	fmt.assertf(gctx.graph_exec != nil, "no captured graph (call end_graph_capture first)", loc=loc)
	cuda.check(cuda.GraphLaunch(gctx.graph_exec, gctx.stream), loc=loc)
}

// Fills the variant's scratch tensors with what this backend's kernels need.
_alloc_scratch :: proc(op: ^ml.Operation, loc: runtime.Source_Code_Location) {
	#partial switch &v in op.variant {
	case ml.Attention:
		token_count := op.input.shape[0]
		if op.input.type == .Bf16 && !ml.is_training(loc=loc) {
			// The flash inference kernel keeps only per-row log-sum-exp.
			v.lse = ml.scratch(.F32, {v.n_q_heads, token_count}, loc=loc)
		} else {
			v.softmax_outputs = ml.scratch(.F32, {v.n_q_heads, token_count, token_count}, loc=loc)
		}
	case ml.Rmsnorm:
		count := ml.len(op.input) / op.input.shape[op.input.rank - 1]
		v.rstd = ml.scratch(.F32, {count}, loc=loc)
	case ml.Cross_Entropy:
		shape := op.input.shape
		v.probabilities = ml.scratch(op.input.type, shape[:op.input.rank], loc=loc)
	}
}

forward :: proc(op: ^ml.Operation, loc: runtime.Source_Code_Location) {
	_alloc_scratch(op, loc)
	_forward(op^, loc)
}
backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location)                { _backward(op, loc)   }
update   :: proc(opt: ml.Optimizer, t: ml.Tensor, loc: runtime.Source_Code_Location) { _update(opt, t, loc) }
