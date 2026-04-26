// Vulkan compute backend.
//
// First milestone: bring up an instance, pick a compute-capable physical
// device, create a logical device with one compute queue, and a command pool.
// No buffers / shaders / dispatch yet — that lands in follow-up files.
//
// Run examples/gpu_hello to verify the device init path on this machine.
package gpu

import "core:dynlib"
import "core:fmt"
import "core:strings"
import vk "vendor:vulkan"

// Device-level state, brought up once by `init` and shared by every
// Gpu_Context on this process. Vulkan handles in here are either immutable
// post-init (instance, device, queue family) or thread-safe to use after
// creation (device, pipelines once compiled).
Gpu_Device :: struct {
	instance:           vk.Instance,
	physical_device:    vk.PhysicalDevice,
	device:             vk.Device,
	queue:              vk.Queue,
	queue_family_index: u32,
	memory_properties:  vk.PhysicalDeviceMemoryProperties,

	// Pipelines created via _make_pipeline register here so destroy() can
	// tear them down without each op needing its own cleanup hook.
	pipelines:          [dynamic]^Pipeline,

	device_name:        string, // owned, freed in destroy
	loader:             dynlib.Library,
}

// Per-thread / per-ml-Context Vulkan state. Vulkan command pools and
// descriptor pools are NOT thread-safe to use from multiple threads at once,
// so each host thread that wants to issue GPU work owns its own Gpu_Context.
// The active one is tracked via a thread-local stack analogous to ml.Context.
//
// `allocations` tracks every Gpu_Storage created by gpu_alloc on this
// context; gpu_clear_storage frees them in bulk to mirror CPU's arena
// reset.
Gpu_Context :: struct {
	command_pool:    vk.CommandPool,
	descriptor_pool: vk.DescriptorPool,
	batch:           Batch,

	allocations:     [dynamic]^Gpu_Storage,

	previous_ctx: ^Gpu_Context,
}

_gpu: Gpu_Device

@(thread_local)
_current_gpu_ctx: ^Gpu_Context

// Bring up Vulkan: load the system loader, create instance, pick a physical
// device with a compute queue, create the device. Panics on any failure —
// the rest of the library assumes a healthy GPU once init returns.
//
// Per-Context resources (command pool, descriptor pool, batch state) are
// created by `context_create`, not here.
init :: proc() {
	_load_loader()
	_create_instance()
	_pick_physical_device()
	_create_device()

	fmt.printfln("gpu: %v (queue family %v)", _gpu.device_name, _gpu.queue_family_index)
}

// Tear down everything init brought up. All Gpu_Contexts must already be
// destroyed before calling this.
destroy :: proc() {
	for p in _gpu.pipelines {
		_destroy_pipeline(p)
	}
	delete(_gpu.pipelines)
	_gpu.pipelines = nil

	if _gpu.device != nil {
		vk.DestroyDevice(_gpu.device, nil)
		_gpu.device = nil
	}
	if _gpu.instance != nil {
		vk.DestroyInstance(_gpu.instance, nil)
		_gpu.instance = nil
	}
	if _gpu.device_name != "" {
		delete(_gpu.device_name)
		_gpu.device_name = ""
	}
	if _gpu.loader != nil {
		_ = dynlib.unload_library(_gpu.loader)
		_gpu.loader = nil
	}
}

// Heap-allocate and initialize a Gpu_Context. Pair with `context_destroy`.
@(require_results)
context_create :: proc(allocator := context.allocator, loc := #caller_location) -> ^Gpu_Context {
	fmt.assertf(_gpu.device != nil, "gpu.init() must be called first", loc=loc)

	gctx, err := new(Gpu_Context, allocator=allocator, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate Gpu_Context: %v", err, loc=loc)

	_create_command_pool(gctx, loc)
	_create_descriptor_pool(gctx, loc)
	return gctx
}

// Destroy and free a Gpu_Context allocated by `context_create`. Must not be
// on the active stack when destroyed.
context_destroy :: proc(gctx: ^Gpu_Context, allocator := context.allocator, loc := #caller_location) {
	fmt.assertf(_current_gpu_ctx != gctx, "gpu.context_destroy called on the active context", loc=loc)

	for storage in gctx.allocations {
		_destroy_gpu_storage(storage)
	}
	delete(gctx.allocations)

	delete(gctx.batch.descriptor_sets)
	delete(gctx.batch.pending_buffers)
	delete(gctx.batch.pending_memories)

	if gctx.descriptor_pool != 0 {
		vk.DestroyDescriptorPool(_gpu.device, gctx.descriptor_pool, nil)
	}
	if gctx.command_pool != 0 {
		vk.DestroyCommandPool(_gpu.device, gctx.command_pool, nil)
	}
	free(gctx, allocator=allocator, loc=loc)
}

// Push a Gpu_Context onto the thread-local stack.
context_begin :: proc(gctx: ^Gpu_Context) {
	gctx.previous_ctx = _current_gpu_ctx
	_current_gpu_ctx = gctx
}

// Pop the current Gpu_Context off the stack.
context_end :: proc() {
	assert(_current_gpu_ctx != nil, "gpu.context_end called with no active context")
	_current_gpu_ctx = _current_gpu_ctx.previous_ctx
}

@(deferred_none=context_end)
context_scope :: proc(gctx: ^Gpu_Context) {
	context_begin(gctx)
}

// Read-only accessor; useful for tests/examples that want to print the picked
// device without poking at the global directly.
device_name :: proc() -> string {
	return _gpu.device_name
}

// --- Internal ---

LOADER_NAME :: "vulkan-1.dll" when ODIN_OS == .Windows else
                "libvulkan.so.1" when ODIN_OS == .Linux else
                "libvulkan.dylib"

_load_loader :: proc() {
	lib, ok := dynlib.load_library(LOADER_NAME)
	fmt.assertf(ok, "failed to load Vulkan loader %q — is the Vulkan runtime installed?", LOADER_NAME)
	_gpu.loader = lib

	get_instance_proc_addr, found := dynlib.symbol_address(lib, "vkGetInstanceProcAddr")
	fmt.assertf(found, "vkGetInstanceProcAddr not exported by %q", LOADER_NAME)

	vk.load_proc_addresses_global(get_instance_proc_addr)
	fmt.assertf(vk.CreateInstance != nil, "vk.CreateInstance still nil after loading global procs")
}

_create_instance :: proc() {
	app_info := vk.ApplicationInfo{
		sType              = .APPLICATION_INFO,
		pApplicationName   = "machine_learning",
		applicationVersion = vk.MAKE_VERSION(0, 1, 0),
		pEngineName        = "machine_learning",
		engineVersion      = vk.MAKE_VERSION(0, 1, 0),
		apiVersion         = vk.API_VERSION_1_2,
	}

	create_info := vk.InstanceCreateInfo{
		sType            = .INSTANCE_CREATE_INFO,
		pApplicationInfo = &app_info,
	}

	res := vk.CreateInstance(&create_info, nil, &_gpu.instance)
	fmt.assertf(res == .SUCCESS, "vkCreateInstance failed: %v", res)

	vk.load_proc_addresses_instance(_gpu.instance)
}

_pick_physical_device :: proc() {
	count: u32
	res := vk.EnumeratePhysicalDevices(_gpu.instance, &count, nil)
	fmt.assertf(res == .SUCCESS, "vkEnumeratePhysicalDevices count failed: %v", res)
	fmt.assertf(count > 0, "no Vulkan-capable physical devices found")

	devices := make([]vk.PhysicalDevice, count, context.temp_allocator)
	res = vk.EnumeratePhysicalDevices(_gpu.instance, &count, raw_data(devices))
	fmt.assertf(res == .SUCCESS, "vkEnumeratePhysicalDevices fetch failed: %v", res)

	// Pick the first discrete GPU with a compute queue, otherwise fall back
	// to the first device with a compute queue. Discrete is preferred because
	// integrated GPUs share memory bandwidth with the CPU and that's what
	// the existing CPU path already optimizes against.
	best:        vk.PhysicalDevice
	best_family: u32 = ~u32(0)
	best_score:  int = -1

	for pd in devices {
		family, ok := _find_compute_queue_family(pd)
		if !ok { continue }

		props: vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(pd, &props)

		score: int = 1
		if props.deviceType == .DISCRETE_GPU   { score = 100 }
		if props.deviceType == .INTEGRATED_GPU { score = 50 }

		if score > best_score {
			best        = pd
			best_family = family
			best_score  = score
		}
	}

	fmt.assertf(best_score >= 0, "no Vulkan device with a compute queue found")
	_gpu.physical_device    = best
	_gpu.queue_family_index = best_family

	props: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(best, &props)
	_gpu.device_name = strings.clone_from_cstring(cstring(raw_data(props.deviceName[:])))

	vk.GetPhysicalDeviceMemoryProperties(best, &_gpu.memory_properties)
}

_find_compute_queue_family :: proc(pd: vk.PhysicalDevice) -> (family: u32, ok: bool) {
	count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(pd, &count, nil)
	if count == 0 { return 0, false }

	families := make([]vk.QueueFamilyProperties, count, context.temp_allocator)
	vk.GetPhysicalDeviceQueueFamilyProperties(pd, &count, raw_data(families))

	for fam, i in families {
		if .COMPUTE in fam.queueFlags {
			return u32(i), true
		}
	}
	return 0, false
}

_create_device :: proc() {
	priority: f32 = 1.0
	queue_info := vk.DeviceQueueCreateInfo{
		sType            = .DEVICE_QUEUE_CREATE_INFO,
		queueFamilyIndex = _gpu.queue_family_index,
		queueCount       = 1,
		pQueuePriorities = &priority,
	}

	create_info := vk.DeviceCreateInfo{
		sType                = .DEVICE_CREATE_INFO,
		queueCreateInfoCount = 1,
		pQueueCreateInfos    = &queue_info,
	}

	res := vk.CreateDevice(_gpu.physical_device, &create_info, nil, &_gpu.device)
	fmt.assertf(res == .SUCCESS, "vkCreateDevice failed: %v", res)

	vk.load_proc_addresses_device(_gpu.device)
	vk.GetDeviceQueue(_gpu.device, _gpu.queue_family_index, 0, &_gpu.queue)
}

_create_command_pool :: proc(gctx: ^Gpu_Context, loc := #caller_location) {
	info := vk.CommandPoolCreateInfo{
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = _gpu.queue_family_index,
	}
	res := vk.CreateCommandPool(_gpu.device, &info, nil, &gctx.command_pool)
	fmt.assertf(res == .SUCCESS, "vkCreateCommandPool failed: %v", res, loc=loc)
}

// Sized for many small dispatches, FREE_DESCRIPTOR_SET so we can reclaim
// per-dispatch sets. Generous storage-buffer count: most ML kernels bind 2-4
// buffers, but matmul backwards can hit 6+. Numbers are rough; if we ever
// exhaust the pool we'll grow it.
DESCRIPTOR_POOL_MAX_SETS :: 4096
DESCRIPTOR_POOL_MAX_STORAGE :: 16384

_create_descriptor_pool :: proc(gctx: ^Gpu_Context, loc := #caller_location) {
	pool_size := vk.DescriptorPoolSize{
		type            = .STORAGE_BUFFER,
		descriptorCount = DESCRIPTOR_POOL_MAX_STORAGE,
	}
	info := vk.DescriptorPoolCreateInfo{
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		flags         = {.FREE_DESCRIPTOR_SET},
		maxSets       = DESCRIPTOR_POOL_MAX_SETS,
		poolSizeCount = 1,
		pPoolSizes    = &pool_size,
	}
	res := vk.CreateDescriptorPool(_gpu.device, &info, nil, &gctx.descriptor_pool)
	fmt.assertf(res == .SUCCESS, "vkCreateDescriptorPool failed: %v", res, loc=loc)
}
