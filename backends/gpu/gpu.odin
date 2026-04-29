package machine_learning_backend_gpu

import "base:builtin"

import "core:dynlib"
import "core:fmt"
import "core:strings"
import "core:sync"

import vk "vendor:vulkan"

import ml "../../"

Gpu_Device :: struct {
	instance:           vk.Instance,
	physical_device:    vk.PhysicalDevice,
	device:             vk.Device,
	queue:              vk.Queue,
	queue_family_index: u32,
	memory_properties:  vk.PhysicalDeviceMemoryProperties,

	coopmat_bf16:       bool,

	pipelines:          [dynamic]^Pipeline,

	device_name:        string,
	loader:             dynlib.Library,
}

// Supplemental constants from VK_KHR_shader_bfloat16, missing from the
// vendored vulkan binding. Values match the Vulkan 1.4 registry.
KHR_SHADER_BFLOAT16_EXTENSION_NAME :: "VK_KHR_shader_bfloat16"
STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR :: vk.StructureType(1000141000)
COMPONENT_TYPE_BFLOAT16_KHR :: vk.ComponentTypeKHR(1000141000)

Physical_Device_Shader_Bfloat16_Features_KHR :: struct {
	sType:                            vk.StructureType,
	pNext:                            rawptr,
	shaderBFloat16Type:               b32,
	shaderBFloat16DotProduct:         b32,
	shaderBFloat16CooperativeMatrix:  b32,
}

Context :: struct {
	using _: ml.Context,

	command_pool:    vk.CommandPool,
	descriptor_pool: vk.DescriptorPool,
	batch:           Batch,

	activations: [dynamic]Gpu_Buffer,
	pool:        map[int][dynamic]Gpu_Buffer,

	// Bytes-per-buffer side table. Backend_Buffer is exactly 16 bytes
	// (vk.Buffer + vk.DeviceMemory) with no room for the size, but
	// buffer_copy and the activation pool both need it. Map lookup beats
	// a per-call vkGetBufferMemoryRequirements.
	sizes: map[vk.Buffer]int,

	staging:           Staging,
	pending_downloads: [dynamic]Pending_Download,
}

Staging :: struct {
	buffer: vk.Buffer,
	memory: vk.DeviceMemory,
	size:   vk.DeviceSize,
	mapped: rawptr,
}

Pending_Download :: struct {
	dst:    []byte,
	offset: vk.DeviceSize,
	size:   vk.DeviceSize,
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
	if _gpu.device != nil { return }
	_load_loader()
	_create_instance()
	_pick_physical_device()
	_create_device()

	fmt.printfln("gpu: %v (queue family %v, coopmat_bf16=%v)",
		_gpu.device_name, _gpu.queue_family_index, _gpu.coopmat_bf16)
}

device_destroy :: proc() {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	for p in _gpu.pipelines {
		_destroy_pipeline(p)
	}
	builtin.delete(_gpu.pipelines)
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
		builtin.delete(_gpu.device_name)
		_gpu.device_name = ""
	}
	if _gpu.loader != nil {
		_ = dynlib.unload_library(_gpu.loader)
		_gpu.loader = nil
	}
}

device_name :: proc() -> string {
	return _gpu.device_name
}

LOADER_NAME :: "vulkan-1.dll" when ODIN_OS == .Windows else
	"libvulkan.so.1" when ODIN_OS == .Linux else
	"libvulkan.dylib"

_load_loader :: proc() {
	lib, ok := dynlib.load_library(LOADER_NAME)
	fmt.assertf(ok, "failed to load Vulkan loader %q", LOADER_NAME)
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

	devices := builtin.make([]vk.PhysicalDevice, count, context.temp_allocator)
	res = vk.EnumeratePhysicalDevices(_gpu.instance, &count, raw_data(devices))
	fmt.assertf(res == .SUCCESS, "vkEnumeratePhysicalDevices fetch failed: %v", res)

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
		if props.deviceType == .INTEGRATED_GPU { score = 50  }

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

	families := builtin.make([]vk.QueueFamilyProperties, count, context.temp_allocator)
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

	want_coopmat_bf16 := _query_coopmat_bf16_support()

	extensions := builtin.make([dynamic]cstring, 0, 2, context.temp_allocator)
	if want_coopmat_bf16 {
		append(&extensions, cstring(vk.KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
		append(&extensions, cstring(KHR_SHADER_BFLOAT16_EXTENSION_NAME))
	}

	bf16_features := Physical_Device_Shader_Bfloat16_Features_KHR{
		sType                           = STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
		shaderBFloat16Type              = true,
		shaderBFloat16CooperativeMatrix = true,
	}
	coopmat_features := vk.PhysicalDeviceCooperativeMatrixFeaturesKHR{
		sType             = .PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
		pNext             = &bf16_features,
		cooperativeMatrix = true,
	}
	v11_features := vk.PhysicalDeviceVulkan11Features{
		sType                    = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		pNext                    = &coopmat_features,
		storageBuffer16BitAccess = true,
	}
	features2 := vk.PhysicalDeviceFeatures2{
		sType = .PHYSICAL_DEVICE_FEATURES_2,
		pNext = &v11_features,
	}

	create_info := vk.DeviceCreateInfo{
		sType                = .DEVICE_CREATE_INFO,
		queueCreateInfoCount = 1,
		pQueueCreateInfos    = &queue_info,
	}
	if want_coopmat_bf16 {
		create_info.pNext                   = &features2
		create_info.enabledExtensionCount   = u32(builtin.len(extensions))
		create_info.ppEnabledExtensionNames = raw_data(extensions[:])
	}

	res := vk.CreateDevice(_gpu.physical_device, &create_info, nil, &_gpu.device)
	fmt.assertf(res == .SUCCESS, "vkCreateDevice failed: %v", res)

	vk.load_proc_addresses_device(_gpu.device)
	vk.GetDeviceQueue(_gpu.device, _gpu.queue_family_index, 0, &_gpu.queue)

	_gpu.coopmat_bf16 = want_coopmat_bf16
}

_query_coopmat_bf16_support :: proc() -> bool {
	pd := _gpu.physical_device

	ext_count: u32
	vk.EnumerateDeviceExtensionProperties(pd, nil, &ext_count, nil)
	exts := builtin.make([]vk.ExtensionProperties, ext_count, context.temp_allocator)
	vk.EnumerateDeviceExtensionProperties(pd, nil, &ext_count, raw_data(exts))

	has_coopmat_ext := false
	has_bf16_ext    := false
	for &e in exts {
		name := string(cstring(raw_data(e.extensionName[:])))
		switch name {
		case vk.KHR_COOPERATIVE_MATRIX_EXTENSION_NAME: has_coopmat_ext = true
		case KHR_SHADER_BFLOAT16_EXTENSION_NAME:       has_bf16_ext    = true
		}
	}
	if !has_coopmat_ext || !has_bf16_ext { return false }

	bf16_features := Physical_Device_Shader_Bfloat16_Features_KHR{
		sType = STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
	}
	coopmat_features := vk.PhysicalDeviceCooperativeMatrixFeaturesKHR{
		sType = .PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
		pNext = &bf16_features,
	}
	v11_features := vk.PhysicalDeviceVulkan11Features{
		sType = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
		pNext = &coopmat_features,
	}
	features2 := vk.PhysicalDeviceFeatures2{
		sType = .PHYSICAL_DEVICE_FEATURES_2,
		pNext = &v11_features,
	}
	vk.GetPhysicalDeviceFeatures2(pd, &features2)

	if !coopmat_features.cooperativeMatrix             { return false }
	if !bf16_features.shaderBFloat16Type               { return false }
	if !bf16_features.shaderBFloat16CooperativeMatrix  { return false }
	if !v11_features.storageBuffer16BitAccess          { return false }

	prop_count: u32
	vk.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &prop_count, nil)
	props := builtin.make([]vk.CooperativeMatrixPropertiesKHR, prop_count, context.temp_allocator)
	for &p in props {
		p.sType = .COOPERATIVE_MATRIX_PROPERTIES_KHR
	}
	vk.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &prop_count, raw_data(props))

	for p in props {
		if p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
		   p.AType == COMPONENT_TYPE_BFLOAT16_KHR &&
		   p.BType == COMPONENT_TYPE_BFLOAT16_KHR &&
		   p.CType == .FLOAT32 &&
		   p.ResultType == .FLOAT32 &&
		   p.scope == .SUBGROUP {
			return true
		}
	}
	return false
}

DESCRIPTOR_POOL_MAX_SETS    :: 4096
DESCRIPTOR_POOL_MAX_STORAGE :: 16384

_create_command_pool :: proc(gctx: ^Context, loc := #caller_location) {
	info := vk.CommandPoolCreateInfo{
		sType            = .COMMAND_POOL_CREATE_INFO,
		flags            = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = _gpu.queue_family_index,
	}
	res := vk.CreateCommandPool(_gpu.device, &info, nil, &gctx.command_pool)
	fmt.assertf(res == .SUCCESS, "vkCreateCommandPool failed: %v", res, loc=loc)
}

_create_descriptor_pool :: proc(gctx: ^Context, loc := #caller_location) {
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
