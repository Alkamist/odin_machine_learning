package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:sync"

import "bindings/cuda"
import "bindings/cublas"

import ml "../../"

// Per-process CUDA device state. One per backend; shared across all Contexts.
Gpu_Device :: struct {
	dev:       cuda.Device,
	ctx:       cuda.Context,

	// Device properties we cache for kernel tuning.
	cc_major, cc_minor: i32,
	sm_count:           i32,
	warp_size:          i32,
	max_threads_per_sm: i32,
	max_smem_per_block: i32,
	max_smem_optin:     i32,

	device_name:        string,

	// Pipelines are owned by the device (CUmodule + CUfunction live in the
	// driver, not the user's Context) so they can be reused across multiple
	// Contexts and torn down once at device_destroy.
	pipelines: [dynamic]^Pipeline,
}

Context :: struct {
	using _: ml.Context,

	stream:        cuda.Stream,
	cublas_handle: cublas.Handle,

	// Activation buffers (persist=false) are tracked here so `clear()` can
	// recycle them. Slots are reused by index across forward passes, mirroring
	// the vulkan backend's activation pool. On size mismatch the slot is freed
	// and re-allocated.
	activation_pool:   [dynamic]Activation_Slot,
	activation_cursor: int,

	// Persistent buffers (model weights, optimizer state) live until
	// context_destroy. We track them so we don't leak on teardown.
	persistent: [dynamic]cuda.DevicePtr,

	// Optional per-dispatch GPU timing. Populated by `enable_timing`.
	timing_enabled: bool,
	timing_totals:  map[^Pipeline]Timing_Stat,
	timing_pool:    [dynamic]Timing_Slot,
	timing_cursor:  int,
}

Activation_Slot :: struct {
	ptr:  cuda.DevicePtr,
	size: u64,
}

Timing_Stat :: struct {
	total_ns: i64,
	count:    int,
}

// Reusable pair of CUevents (start/end) for per-dispatch timing. Allocated on
// demand and recycled across batches.
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

device_init :: proc() {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)
	_device_init_locked()
}

_device_init_locked :: proc() {
	if _gpu.ctx != nil { return }

	cuda.check(cuda.Init(0))

	count: i32
	cuda.check(cuda.DeviceGetCount(&count))
	fmt.assertf(count > 0, "no CUDA-capable devices found")

	cuda.check(cuda.DeviceGet(&_gpu.dev, 0))

	name_buf: [128]u8
	cuda.check(cuda.DeviceGetName(raw_data(name_buf[:]), i32(builtin.len(name_buf)), _gpu.dev))
	name_len := 0
	for c, i in name_buf { if c == 0 { name_len = i; break } }
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

	fmt.printfln("cuda: %s  cc=%d.%d  SMs=%d  warp=%d",
		_gpu.device_name, _gpu.cc_major, _gpu.cc_minor, _gpu.sm_count, _gpu.warp_size)
}

device_destroy :: proc() {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	for p in _gpu.pipelines {
		_destroy_pipeline(p)
	}
	builtin.delete(_gpu.pipelines)
	_gpu.pipelines = nil

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
context_create :: proc(allocator := context.allocator, loc := #caller_location) -> ^ml.Context {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	_device_init_locked()

	gctx, err := builtin.new(Context, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate Context: %v", err, loc=loc)

	cuda.check(cuda.StreamCreate(&gctx.stream, cuda.STREAM_NON_BLOCKING))
	cublas.check(cublas.Create_v2(&gctx.cublas_handle))
	cublas.check(cublas.SetStream_v2(gctx.cublas_handle, gctx.stream))

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
	}, allocator, loc)

	return gctx
}

context_destroy :: proc(ctx: ^ml.Context, allocator := context.allocator, loc := #caller_location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gctx := cast(^Context)ctx

	// Drain anything still in flight before tearing down.
	if gctx.stream != nil {
		cuda.check(cuda.StreamSynchronize(gctx.stream))
	}

	for slot in gctx.activation_pool {
		cuda.MemFree(slot.ptr)
	}
	builtin.delete(gctx.activation_pool)

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
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gctx := _gctx(loc)

	// Wait for any in-flight work, fold per-dispatch timing readings into
	// totals, and rewind the timing slot cursor for the next forward pass.
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

	// Activation pool rewinds for the next forward pass. Buffers stay alive
	// and are handed back to ml.alloc in the same order they were requested.
	gctx.activation_cursor = 0
}

// Stats helpers, mirror the vulkan backend's API for parity in benchmarks.
_alloc_count:  int
_alloc_ns:     i64
_upload_count: int
_upload_ns:    i64

reset_alloc_stats :: proc() {
	_alloc_count  = 0
	_alloc_ns     = 0
	_upload_count = 0
	_upload_ns    = 0
}

alloc_stats  :: proc() -> (count: int, ns: i64) { return _alloc_count,  _alloc_ns  }
upload_stats :: proc() -> (count: int, ns: i64) { return _upload_count, _upload_ns }

// Optional GPU timing. Once enabled, every _dispatch records start/end events
// and `clear` folds the deltas into `timing_totals` keyed by Pipeline pointer.
enable_timing :: proc(enabled: bool) {
	gctx := _gctx()
	gctx.timing_enabled = enabled
}

// Op routing lives in ops.odin.
forward  :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) { _forward (op,  loc) }
backward :: proc(op: ml.Operation, loc: runtime.Source_Code_Location) { _backward(op,  loc) }
update   :: proc(opt: ml.Optimizer, t: ml.Tensor, loc: runtime.Source_Code_Location) { _update(opt, t, loc) }
