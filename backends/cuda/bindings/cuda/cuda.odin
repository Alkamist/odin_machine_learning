package cuda

// CUDA Driver API bindings.
//
// Targets the system-installed NVIDIA driver (nvcuda.dll). The import lib is
// vendored under ../lib/cuda.lib so the linker does not need CUDA_PATH on the
// user's machine; the dll itself ships with every NVIDIA driver install in
// System32.
//
// Naming follows the Odin vendor convention: procs and types mirror the C
// names with the `cu` / `CU` prefix stripped. The strip is done via
// `link_prefix="cu"` for the common case; entry points whose actual export
// name has a `_v2` (or similar) suffix are wired with `link_name` overrides.

import "core:fmt"

foreign import lib "../lib/cuda.lib"

// Opaque handles. All driver objects are pointers to internal structs.
Context  :: distinct rawptr
Module   :: distinct rawptr
Function :: distinct rawptr
Stream   :: distinct rawptr
Event    :: distinct rawptr
Graph    :: distinct rawptr
GraphExec :: distinct rawptr
GraphNode :: distinct rawptr

Device     :: distinct i32
DevicePtr  :: distinct u64

Result :: enum i32 {
	SUCCESS                              = 0,
	ERROR_INVALID_VALUE                  = 1,
	ERROR_OUT_OF_MEMORY                  = 2,
	ERROR_NOT_INITIALIZED                = 3,
	ERROR_DEINITIALIZED                  = 4,
	ERROR_PROFILER_DISABLED              = 5,
	ERROR_NO_DEVICE                      = 100,
	ERROR_INVALID_DEVICE                 = 101,
	ERROR_INVALID_IMAGE                  = 200,
	ERROR_INVALID_CONTEXT                = 201,
	ERROR_MAP_FAILED                     = 205,
	ERROR_UNMAP_FAILED                   = 206,
	ERROR_ARRAY_IS_MAPPED                = 207,
	ERROR_ALREADY_MAPPED                 = 208,
	ERROR_NO_BINARY_FOR_GPU              = 209,
	ERROR_ALREADY_ACQUIRED               = 210,
	ERROR_NOT_MAPPED                     = 211,
	ERROR_ECC_UNCORRECTABLE              = 214,
	ERROR_UNSUPPORTED_LIMIT              = 215,
	ERROR_CONTEXT_ALREADY_IN_USE         = 216,
	ERROR_PEER_ACCESS_UNSUPPORTED        = 217,
	ERROR_INVALID_PTX                    = 218,
	ERROR_INVALID_GRAPHICS_CONTEXT       = 219,
	ERROR_NVLINK_UNCORRECTABLE           = 220,
	ERROR_JIT_COMPILER_NOT_FOUND         = 221,
	ERROR_UNSUPPORTED_PTX_VERSION        = 222,
	ERROR_INVALID_SOURCE                 = 300,
	ERROR_FILE_NOT_FOUND                 = 301,
	ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
	ERROR_SHARED_OBJECT_INIT_FAILED      = 303,
	ERROR_OPERATING_SYSTEM               = 304,
	ERROR_INVALID_HANDLE                 = 400,
	ERROR_ILLEGAL_STATE                  = 401,
	ERROR_NOT_FOUND                      = 500,
	ERROR_NOT_READY                      = 600,
	ERROR_ILLEGAL_ADDRESS                = 700,
	ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,
	ERROR_LAUNCH_TIMEOUT                 = 702,
	ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
	ERROR_HARDWARE_STACK_ERROR           = 714,
	ERROR_ILLEGAL_INSTRUCTION            = 715,
	ERROR_MISALIGNED_ADDRESS             = 716,
	ERROR_INVALID_ADDRESS_SPACE          = 717,
	ERROR_INVALID_PC                     = 718,
	ERROR_LAUNCH_FAILED                  = 719,
	ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,
	ERROR_NOT_PERMITTED                  = 800,
	ERROR_NOT_SUPPORTED                  = 801,
	ERROR_SYSTEM_NOT_READY               = 802,
	ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900,
	ERROR_STREAM_CAPTURE_INVALIDATED     = 901,
	ERROR_UNKNOWN                        = 999,
}

// Subset of `CU_DEVICE_ATTRIBUTE_*` we actually use.
DeviceAttribute :: enum i32 {
	MAX_THREADS_PER_BLOCK             = 1,
	MAX_BLOCK_DIM_X                   = 2,
	MAX_BLOCK_DIM_Y                   = 3,
	MAX_BLOCK_DIM_Z                   = 4,
	MAX_GRID_DIM_X                    = 5,
	MAX_GRID_DIM_Y                    = 6,
	MAX_GRID_DIM_Z                    = 7,
	MAX_SHARED_MEMORY_PER_BLOCK       = 8,
	TOTAL_CONSTANT_MEMORY             = 9,
	WARP_SIZE                         = 10,
	MAX_PITCH                         = 11,
	MAX_REGISTERS_PER_BLOCK           = 12,
	CLOCK_RATE                        = 13,
	MULTIPROCESSOR_COUNT              = 16,
	MAX_THREADS_PER_MULTIPROCESSOR    = 39,
	COMPUTE_CAPABILITY_MAJOR          = 75,
	COMPUTE_CAPABILITY_MINOR          = 76,
	MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
	MAX_BLOCKS_PER_MULTIPROCESSOR     = 106,
}

// `CU_FUNC_ATTRIBUTE_*` (cuFuncSetAttribute / cuFuncGetAttribute).
FunctionAttribute :: enum i32 {
	MAX_THREADS_PER_BLOCK            = 0,
	SHARED_SIZE_BYTES                = 1,
	CONST_SIZE_BYTES                 = 2,
	LOCAL_SIZE_BYTES                 = 3,
	NUM_REGS                         = 4,
	PTX_VERSION                      = 5,
	BINARY_VERSION                   = 6,
	CACHE_MODE_CA                    = 7,
	MAX_DYNAMIC_SHARED_SIZE_BYTES    = 8,
	PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
}

// Context creation flags (`CU_CTX_*`).
CTX_SCHED_AUTO          :: 0x00
CTX_SCHED_SPIN          :: 0x01
CTX_SCHED_YIELD         :: 0x02
CTX_SCHED_BLOCKING_SYNC :: 0x04
CTX_MAP_HOST            :: 0x08
CTX_LMEM_RESIZE_TO_MAX  :: 0x10

// Stream creation flags (`CU_STREAM_*`).
STREAM_DEFAULT      :: 0x00
STREAM_NON_BLOCKING :: 0x01

// Stream capture mode (`CUstreamCaptureMode`). `Relaxed` is what graph
// recording usually wants â€” `Global` enforces strict cross-thread isolation.
StreamCaptureMode :: enum i32 {
	Global       = 0,
	Thread_Local = 1,
	Relaxed      = 2,
}

// `cuGraphInstantiate` / `cuGraphInstantiateWithFlags` flags.
GRAPH_INSTANTIATE_DEFAULT                    :: 0x00
GRAPH_INSTANTIATE_AUTO_FREE_ON_LAUNCH        :: 0x01
GRAPH_INSTANTIATE_UPLOAD                     :: 0x02
GRAPH_INSTANTIATE_DEVICE_LAUNCH              :: 0x04
GRAPH_INSTANTIATE_USE_NODE_PRIORITY          :: 0x08

// `cuGraphExecUpdate` outcome codes (`CUgraphExecUpdateResult`).
GraphExecUpdateResult :: enum i32 {
	SUCCESS                           = 0,
	ERROR                             = 1,
	ERROR_TOPOLOGY_CHANGED            = 2,
	ERROR_NODE_TYPE_CHANGED           = 3,
	ERROR_FUNCTION_CHANGED            = 4,
	ERROR_PARAMETERS_CHANGED          = 5,
	ERROR_NOT_SUPPORTED               = 6,
	ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
	ERROR_ATTRIBUTES_CHANGED          = 8,
}

GraphExecUpdateResultInfo :: struct {
	result:          GraphExecUpdateResult,
	error_node:      GraphNode,
	error_from_node: GraphNode,
}

// Event creation flags (`CU_EVENT_*`).
EVENT_DEFAULT        :: 0x00
EVENT_BLOCKING_SYNC  :: 0x01
EVENT_DISABLE_TIMING :: 0x02
EVENT_INTERPROCESS   :: 0x04

@(default_calling_convention="c", link_prefix="cu")
foreign lib {
	// Initialization & errors. `cuInit`, `cuDriverGetVersion`, etc.
	Init             :: proc(flags: u32) -> Result ---
	DriverGetVersion :: proc(version: ^i32) -> Result ---
	GetErrorName     :: proc(error: Result, name: ^cstring) -> Result ---
	GetErrorString   :: proc(error: Result, str:  ^cstring) -> Result ---

	// Devices.
	DeviceGet           :: proc(device: ^Device, ordinal: i32) -> Result ---
	DeviceGetCount      :: proc(count: ^i32) -> Result ---
	DeviceGetName       :: proc(name: [^]u8, len: i32, dev: Device) -> Result ---
	DeviceGetAttribute  :: proc(out: ^i32, attr: DeviceAttribute, dev: Device) -> Result ---
	@(link_name="cuDeviceTotalMem_v2")
	DeviceTotalMem      :: proc(bytes: ^uint, dev: Device) -> Result ---

	// Contexts. `_v2` is the actual export; the `cu*` macro aliases it.
	@(link_name="cuCtxCreate_v2")      CtxCreate      :: proc(ctx: ^Context, flags: u32, dev: Device) -> Result ---
	@(link_name="cuCtxDestroy_v2")     CtxDestroy     :: proc(ctx: Context) -> Result ---
	CtxSetCurrent  :: proc(ctx: Context) -> Result ---
	CtxGetCurrent  :: proc(ctx: ^Context) -> Result ---
	CtxSynchronize :: proc() -> Result ---
	@(link_name="cuCtxPushCurrent_v2") CtxPushCurrent :: proc(ctx: Context) -> Result ---
	@(link_name="cuCtxPopCurrent_v2")  CtxPopCurrent  :: proc(ctx: ^Context) -> Result ---

	// Modules.
	ModuleLoadData    :: proc(mod: ^Module, image: rawptr) -> Result ---
	ModuleLoadDataEx  :: proc(mod: ^Module, image: rawptr, num_options: u32, options: [^]i32, option_values: [^]rawptr) -> Result ---
	ModuleUnload      :: proc(mod: Module) -> Result ---
	ModuleGetFunction :: proc(fn: ^Function, mod: Module, name: cstring) -> Result ---
	@(link_name="cuModuleGetGlobal_v2")
	ModuleGetGlobal   :: proc(dptr: ^DevicePtr, bytes: ^uint, mod: Module, name: cstring) -> Result ---

	// Streams.
	StreamCreate      :: proc(stream: ^Stream, flags: u32) -> Result ---
	@(link_name="cuStreamDestroy_v2")
	StreamDestroy     :: proc(stream: Stream) -> Result ---
	StreamSynchronize :: proc(stream: Stream) -> Result ---
	StreamWaitEvent   :: proc(stream: Stream, event: Event, flags: u32) -> Result ---

	// Stream capture (CUDA graphs).
	StreamBeginCapture_v2  :: proc(stream: Stream, mode: StreamCaptureMode) -> Result ---
	StreamEndCapture       :: proc(stream: Stream, graph: ^Graph) -> Result ---
	StreamIsCapturing      :: proc(stream: Stream, capture_status: ^i32) -> Result ---

	// Graphs.
	GraphInstantiateWithFlags :: proc(exec: ^GraphExec, graph: Graph, flags: u64) -> Result ---
	GraphLaunch               :: proc(exec: GraphExec, stream: Stream) -> Result ---
	GraphExecDestroy          :: proc(exec: GraphExec) -> Result ---
	GraphDestroy              :: proc(graph: Graph) -> Result ---
	@(link_name="cuGraphExecUpdate_v2")
	GraphExecUpdate           :: proc(exec: GraphExec, graph: Graph, info: ^GraphExecUpdateResultInfo) -> Result ---

	// Events (used for timing benchmarks).
	EventCreate       :: proc(event: ^Event, flags: u32) -> Result ---
	@(link_name="cuEventDestroy_v2")
	EventDestroy      :: proc(event: Event) -> Result ---
	EventRecord       :: proc(event: Event, stream: Stream) -> Result ---
	EventSynchronize  :: proc(event: Event) -> Result ---
	EventElapsedTime  :: proc(ms: ^f32, start: Event, end: Event) -> Result ---

	// Memory: device alloc / free.
	@(link_name="cuMemAlloc_v2")     MemAlloc     :: proc(dptr: ^DevicePtr, bytes: uint) -> Result ---
	@(link_name="cuMemFree_v2")      MemFree      :: proc(dptr: DevicePtr) -> Result ---
	@(link_name="cuMemAllocHost_v2") MemAllocHost :: proc(pp: ^rawptr, bytes: uint) -> Result ---
	MemFreeHost  :: proc(p: rawptr) -> Result ---

	// Memory: copies.
	@(link_name="cuMemcpyHtoD_v2")      MemcpyHtoD       :: proc(dst: DevicePtr, src: rawptr, bytes: uint) -> Result ---
	@(link_name="cuMemcpyDtoH_v2")      MemcpyDtoH       :: proc(dst: rawptr, src: DevicePtr, bytes: uint) -> Result ---
	@(link_name="cuMemcpyDtoD_v2")      MemcpyDtoD       :: proc(dst: DevicePtr, src: DevicePtr, bytes: uint) -> Result ---
	@(link_name="cuMemcpyHtoDAsync_v2") MemcpyHtoDAsync  :: proc(dst: DevicePtr, src: rawptr, bytes: uint, stream: Stream) -> Result ---
	@(link_name="cuMemcpyDtoHAsync_v2") MemcpyDtoHAsync  :: proc(dst: rawptr, src: DevicePtr, bytes: uint, stream: Stream) -> Result ---
	@(link_name="cuMemcpyDtoDAsync_v2") MemcpyDtoDAsync  :: proc(dst: DevicePtr, src: DevicePtr, bytes: uint, stream: Stream) -> Result ---

	// Memory: zero/set.
	@(link_name="cuMemsetD8_v2")  MemsetD8       :: proc(dptr: DevicePtr, value: u8,  count: uint) -> Result ---
	@(link_name="cuMemsetD32_v2") MemsetD32      :: proc(dptr: DevicePtr, value: u32, count: uint) -> Result ---
	MemsetD8Async  :: proc(dptr: DevicePtr, value: u8,  count: uint, stream: Stream) -> Result ---
	MemsetD32Async :: proc(dptr: DevicePtr, value: u32, count: uint, stream: Stream) -> Result ---

	// Kernel launch. `extra` is typically nil; `kernel_params` is an array of
	// pointers, one per kernel argument.
	LaunchKernel :: proc(
		fn:                                    Function,
		grid_dim_x, grid_dim_y, grid_dim_z:    u32,
		block_dim_x, block_dim_y, block_dim_z: u32,
		shared_mem_bytes:                      u32,
		stream:                                Stream,
		kernel_params:                         [^]rawptr,
		extra:                                 [^]rawptr,
	) -> Result ---

	// Function attributes (needed to opt in to >48KB shared memory on Ampere).
	FuncSetAttribute :: proc(fn: Function, attr: FunctionAttribute, value: i32) -> Result ---
	FuncGetAttribute :: proc(out: ^i32, attr: FunctionAttribute, fn: Function) -> Result ---

	// Occupancy helper. Useful when tuning launch bounds against measured perf.
	OccupancyMaxActiveBlocksPerMultiprocessor :: proc(
		num_blocks: ^i32, fn: Function, block_size: i32, dynamic_shared_mem_size: uint,
	) -> Result ---
}

// Aborts with a formatted error if `r` is not SUCCESS. Mirrors how the vulkan
// backend asserts on VkResult.
check :: proc(r: Result, loc := #caller_location) {
	if r == .SUCCESS { return }
	name_cstr: cstring
	GetErrorName(r, &name_cstr)
	desc_cstr: cstring
	GetErrorString(r, &desc_cstr)
	name := name_cstr != nil ? string(name_cstr) : "?"
	desc := desc_cstr != nil ? string(desc_cstr) : "?"
	fmt.panicf("CUDA error: %s (%d): %s", name, i32(r), desc, loc=loc)
}
