// GPU-resident tensors backed by device-local Vulkan buffers.
//
// Allocation strategy: every GpuTensor is DEVICE_LOCAL (lives in VRAM on a
// discrete GPU). Host transfers go through a temporary staging buffer
// (HOST_VISIBLE | HOST_COHERENT) per call. This is simple and the right
// shape long-term — once data is on the GPU, kernels read from VRAM at full
// bandwidth instead of from PCIe-mapped host memory.
//
// Per-call staging is fine for setup paths (model upload, occasional readback
// for checksums). The training hot path won't transfer between host and
// device at all, so staging-buffer pooling can wait until it shows up in a
// profile.
package gpu

import "core:fmt"
import "core:mem"
import vk "vendor:vulkan"

import ml ".."

GpuTensor :: struct {
	buffer: vk.Buffer,
	memory: vk.DeviceMemory,

	// Number of f32 elements (data only — no shadow gradient buffer; that's
	// a CPU-side concept on the autograd path and will be added separately
	// once we wire backward passes).
	count: int,

	shape: [ml.MAX_TENSOR_RANK]int,
	rank:  int,
}

// Allocate a GPU tensor with the given shape. STORAGE_BUFFER usage so compute
// shaders can read/write it; TRANSFER_SRC | TRANSFER_DST so it can participate
// in upload/download copies.
alloc :: proc(shape: ..int, loc := #caller_location) -> (t: GpuTensor) {
	fmt.assertf(_gpu.device != nil, "gpu.init() must be called first", loc=loc)
	fmt.assertf(len(shape) > 0, "GpuTensor must have at least one dimension", loc=loc)
	fmt.assertf(len(shape) <= ml.MAX_TENSOR_RANK, "GpuTensor rank exceeds MAX_TENSOR_RANK", loc=loc)

	count := 1
	for d, i in shape {
		fmt.assertf(d > 0, "GpuTensor dimension must be positive", loc=loc)
		count *= d
		t.shape[i] = d
	}
	t.rank  = len(shape)
	t.count = count

	size := vk.DeviceSize(count * size_of(f32))
	t.buffer, t.memory = _create_buffer(
		size,
		{.STORAGE_BUFFER, .TRANSFER_SRC, .TRANSFER_DST},
		{.DEVICE_LOCAL},
	)
	return
}

destroy_tensor :: proc(t: GpuTensor) {
	if t.buffer != 0 {
		vk.DestroyBuffer(_gpu.device, t.buffer, nil)
	}
	if t.memory != 0 {
		vk.FreeMemory(_gpu.device, t.memory, nil)
	}
}

// Copy `src` (CPU) into `dst` (GPU). `len(src)` must equal `dst.count`.
upload :: proc(src: []f32, dst: GpuTensor, loc := #caller_location) {
	fmt.assertf(len(src) == dst.count,
		"upload size mismatch: src=%v dst.count=%v", len(src), dst.count, loc=loc)

	size := vk.DeviceSize(dst.count * size_of(f32))
	stage_buf, stage_mem := _create_buffer(
		size,
		{.TRANSFER_SRC},
		{.HOST_VISIBLE, .HOST_COHERENT},
	)
	defer {
		vk.DestroyBuffer(_gpu.device, stage_buf, nil)
		vk.FreeMemory(_gpu.device, stage_mem, nil)
	}

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, stage_mem, 0, size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(staging upload) failed: %v", res)
	mem.copy(mapped, raw_data(src), int(size))
	vk.UnmapMemory(_gpu.device, stage_mem)

	_one_shot_copy(stage_buf, dst.buffer, size)
}

// Copy `src` (GPU) into `dst` (CPU). `len(dst)` must equal `src.count`.
download :: proc(src: GpuTensor, dst: []f32, loc := #caller_location) {
	fmt.assertf(len(dst) == src.count,
		"download size mismatch: src.count=%v dst=%v", src.count, len(dst), loc=loc)

	size := vk.DeviceSize(src.count * size_of(f32))
	stage_buf, stage_mem := _create_buffer(
		size,
		{.TRANSFER_DST},
		{.HOST_VISIBLE, .HOST_COHERENT},
	)
	defer {
		vk.DestroyBuffer(_gpu.device, stage_buf, nil)
		vk.FreeMemory(_gpu.device, stage_mem, nil)
	}

	_one_shot_copy(src.buffer, stage_buf, size)

	mapped: rawptr
	res := vk.MapMemory(_gpu.device, stage_mem, 0, size, {}, &mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(staging download) failed: %v", res)
	mem.copy(raw_data(dst), mapped, int(size))
	vk.UnmapMemory(_gpu.device, stage_mem)
}

// --- Internal ---

_create_buffer :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	mem_flags: vk.MemoryPropertyFlags,
	loc := #caller_location,
) -> (buffer: vk.Buffer, memory: vk.DeviceMemory) {
	info := vk.BufferCreateInfo{
		sType       = .BUFFER_CREATE_INFO,
		size        = size,
		usage       = usage,
		sharingMode = .EXCLUSIVE,
	}
	res := vk.CreateBuffer(_gpu.device, &info, nil, &buffer)
	fmt.assertf(res == .SUCCESS, "vkCreateBuffer failed: %v", res, loc=loc)

	reqs: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(_gpu.device, buffer, &reqs)

	alloc_info := vk.MemoryAllocateInfo{
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = reqs.size,
		memoryTypeIndex = _pick_memory_type(reqs.memoryTypeBits, mem_flags, loc),
	}
	res = vk.AllocateMemory(_gpu.device, &alloc_info, nil, &memory)
	fmt.assertf(res == .SUCCESS, "vkAllocateMemory failed: %v", res, loc=loc)

	res = vk.BindBufferMemory(_gpu.device, buffer, memory, 0)
	fmt.assertf(res == .SUCCESS, "vkBindBufferMemory failed: %v", res, loc=loc)
	return
}

_pick_memory_type :: proc(
	type_bits: u32,
	required: vk.MemoryPropertyFlags,
	loc := #caller_location,
) -> u32 {
	mp := &_gpu.memory_properties
	for i in 0 ..< mp.memoryTypeCount {
		if (type_bits & (1 << i)) == 0 { continue }
		if required <= mp.memoryTypes[i].propertyFlags {
			return i
		}
	}
	fmt.panicf("no memory type matches type_bits=0x%x required=%v", type_bits, required, loc=loc)
}

// Allocate a transient command buffer, record a single buffer-to-buffer copy,
// submit it, and wait for completion. Synchronous on purpose — this is the
// upload/download path, not a hot inner loop.
_one_shot_copy :: proc(src, dst: vk.Buffer, size: vk.DeviceSize, loc := #caller_location) {
	cmd := _begin_one_shot(loc)

	region := vk.BufferCopy{ srcOffset = 0, dstOffset = 0, size = size }
	vk.CmdCopyBuffer(cmd, src, dst, 1, &region)

	_end_one_shot(cmd, loc)
}

_begin_one_shot :: proc(loc := #caller_location) -> vk.CommandBuffer {
	alloc_info := vk.CommandBufferAllocateInfo{
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = _gpu.command_pool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}
	cmd: vk.CommandBuffer
	res := vk.AllocateCommandBuffers(_gpu.device, &alloc_info, &cmd)
	fmt.assertf(res == .SUCCESS, "vkAllocateCommandBuffers failed: %v", res, loc=loc)

	begin := vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}
	res = vk.BeginCommandBuffer(cmd, &begin)
	fmt.assertf(res == .SUCCESS, "vkBeginCommandBuffer failed: %v", res, loc=loc)
	return cmd
}

_end_one_shot :: proc(cmd: vk.CommandBuffer, loc := #caller_location) {
	cmd := cmd
	res := vk.EndCommandBuffer(cmd)
	fmt.assertf(res == .SUCCESS, "vkEndCommandBuffer failed: %v", res, loc=loc)

	submit := vk.SubmitInfo{
		sType              = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers    = &cmd,
	}
	res = vk.QueueSubmit(_gpu.queue, 1, &submit, 0)
	fmt.assertf(res == .SUCCESS, "vkQueueSubmit failed: %v", res, loc=loc)
	res = vk.QueueWaitIdle(_gpu.queue)
	fmt.assertf(res == .SUCCESS, "vkQueueWaitIdle failed: %v", res, loc=loc)

	vk.FreeCommandBuffers(_gpu.device, _gpu.command_pool, 1, &cmd)
}
