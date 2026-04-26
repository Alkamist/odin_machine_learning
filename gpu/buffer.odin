// Vulkan buffer-management helpers shared by the unified `ml.Backend`
// integration in `backend.odin`. Per-tensor storage is allocated via
// `Backend.alloc` / `persistent_alloc` / `parameter_alloc`, not here —
// this file exposes only the low-level primitives.
package gpu

import "core:fmt"
import vk "vendor:vulkan"

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
// Lazily grow the active gpu Context's persistent staging buffer to
// hold at least `min_size` bytes. The staging memory is HOST_VISIBLE +
// HOST_COHERENT and stays persistently mapped, so callers can write or
// read its contents directly via `gctx.staging.mapped`.
//
// Capacity grows by powers of two from a 64KB floor; existing contents
// are *not* preserved across a regrow, so callers must finish any
// pending use before triggering one.
_ensure_staging :: proc(min_size: vk.DeviceSize, loc := #caller_location) {
	gctx := _current_gpu_ctx
	fmt.assertf(gctx != nil, "no active gpu Context", loc=loc)

	if gctx.staging.size >= min_size {
		return
	}

	// Free old.
	if gctx.staging.buffer != 0 {
		if gctx.staging.mapped != nil {
			vk.UnmapMemory(_gpu.device, gctx.staging.memory)
			gctx.staging.mapped = nil
		}
		vk.DestroyBuffer(_gpu.device, gctx.staging.buffer, nil)
		vk.FreeMemory(_gpu.device, gctx.staging.memory, nil)
		gctx.staging.buffer = 0
		gctx.staging.memory = 0
	}

	new_size := vk.DeviceSize(64 * 1024)
	for new_size < min_size do new_size *= 2

	gctx.staging.buffer, gctx.staging.memory = _create_buffer(
		new_size,
		{.TRANSFER_SRC, .TRANSFER_DST},
		{.HOST_VISIBLE, .HOST_COHERENT},
		loc,
	)
	gctx.staging.size = new_size

	res := vk.MapMemory(_gpu.device, gctx.staging.memory, 0, new_size, {}, &gctx.staging.mapped)
	fmt.assertf(res == .SUCCESS, "vkMapMemory(staging) failed: %v", res, loc=loc)
}

// Fill a buffer with `value` (a u32 stamp — for f32 fills, pass the
// IEEE 754 bit pattern). If a batch is active, record the fill into its
// command buffer (free — no extra submit). Otherwise fall back to a
// one-shot submit + wait.
//
// Critical: when a batch is active, ALL fills must record into the batch
// CB. A one-shot submit during recording would execute on the queue
// before the batch is submitted, but the batch CB may contain a later-
// executing fill of the same buffer (e.g. an alloc-time zero-fill
// recorded before the one-shot was issued), which would clobber the
// one-shot's write when the batch finally runs.
_record_fill :: proc(buf: vk.Buffer, size: vk.DeviceSize, value: u32, loc := #caller_location) {
	gctx := _current_gpu_ctx
	if gctx != nil && gctx.batch.active {
		vk.CmdFillBuffer(gctx.batch.cmd, buf, 0, size, value)
		return
	}
	cmd := _begin_one_shot(loc)
	vk.CmdFillBuffer(cmd, buf, 0, size, value)
	_end_one_shot(cmd, loc)
}

_record_fill_zero :: #force_inline proc(buf: vk.Buffer, size: vk.DeviceSize, loc := #caller_location) {
	_record_fill(buf, size, 0, loc)
}

_one_shot_copy :: proc(src, dst: vk.Buffer, size: vk.DeviceSize, loc := #caller_location) {
	cmd := _begin_one_shot(loc)

	region := vk.BufferCopy{ srcOffset = 0, dstOffset = 0, size = size }
	vk.CmdCopyBuffer(cmd, src, dst, 1, &region)

	_end_one_shot(cmd, loc)
}

_begin_one_shot :: proc(loc := #caller_location) -> vk.CommandBuffer {
	gctx := _current_gpu_ctx
	fmt.assertf(gctx != nil, "no active gpu Context — call gpu.context_begin / context_scope", loc=loc)

	alloc_info := vk.CommandBufferAllocateInfo{
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = gctx.command_pool,
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

	vk.FreeCommandBuffers(_gpu.device, _current_gpu_ctx.command_pool, 1, &cmd)
}
