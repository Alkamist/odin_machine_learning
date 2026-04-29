package machine_learning_backend_gpu

import "core:fmt"
import "core:mem"

import vk "vendor:vulkan"

Gpu_Buffer :: struct {
	buffer: vk.Buffer,
	memory: vk.DeviceMemory,
}

#assert(size_of(Gpu_Buffer) == 16)

// f32(1.0) bit pattern for vkCmdFillBuffer's u32 stamp.
F32_ONE_BITS :: u32(0x3F800000)

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

_pick_memory_type :: proc(type_bits: u32, required: vk.MemoryPropertyFlags, loc := #caller_location) -> u32 {
	memory_properties := &_gpu.memory_properties
	for i in 0 ..< memory_properties.memoryTypeCount {
		if (type_bits & (1 << i)) == 0 { continue }
		if required <= memory_properties.memoryTypes[i].propertyFlags {
			return i
		}
	}
	fmt.panicf("no memory type matches type_bits=0x%x required=%v", type_bits, required, loc=loc)
}

// Lazily grow the active gctx's persistently-mapped staging buffer to hold
// at least min_size bytes. Existing contents are NOT preserved across regrow,
// so callers must finish any pending use before triggering one.
_ensure_staging :: proc(min_size: vk.DeviceSize, loc := #caller_location) {
	gctx := _gctx(loc)

	if gctx.staging.size >= min_size {
		return
	}

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

// Fill `buf` with `value` (u32 stamp). When a batch is active, ALL fills
// must record into the batch CB — a one-shot during recording would execute
// on the queue before later-recorded work in the batch, clobbering writes.
_record_fill :: proc(buf: vk.Buffer, size: vk.DeviceSize, value: u32, loc := #caller_location) {
	gctx := _gctx(loc)
	if gctx.batch.active {
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
	region := vk.BufferCopy{srcOffset = 0, dstOffset = 0, size = size}
	vk.CmdCopyBuffer(cmd, src, dst, 1, &region)
	_end_one_shot(cmd, loc)
}

_begin_one_shot :: proc(loc := #caller_location) -> vk.CommandBuffer {
	gctx := _gctx(loc)

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

	vk.FreeCommandBuffers(_gpu.device, _gctx(loc).command_pool, 1, &cmd)
}

// Copy host data → device buffer via the staging buffer. Synchronous on
// purpose; uploads are not on the hot path. Flushes any active batch first
// so prior queued writes to the buffer don't clobber the upload.
_upload :: proc(dst: vk.Buffer, src: []byte, loc := #caller_location) {
	if len(src) == 0 { return }

	gctx := _gctx(loc)
	if gctx.batch.active {
		end_batch(loc)
	}

	size := vk.DeviceSize(len(src))
	_ensure_staging(size, loc)

	mem.copy(gctx.staging.mapped, raw_data(src), int(size))

	cmd := _begin_one_shot(loc)
	region := vk.BufferCopy{srcOffset = 0, dstOffset = 0, size = size}
	vk.CmdCopyBuffer(cmd, gctx.staging.buffer, dst, 1, &region)
	_end_one_shot(cmd, loc)
}

// Copy device buffer → host data. Synchronous — when this returns, dst
// holds the GPU's contents. If a batch is active, fold the device →
// staging copy into it and end the batch so the deferred host memcpy
// runs (one submit total instead of two).
_download :: proc(src: vk.Buffer, dst: []byte, loc := #caller_location) {
	if len(dst) == 0 { return }

	gctx := _gctx(loc)
	size := vk.DeviceSize(len(dst))

	if gctx.batch.active {
		needed := gctx.batch.staging_offset + size
		if needed > gctx.staging.size {
			end_batch(loc)
		}
	}

	if gctx.batch.active {
		offset := gctx.batch.staging_offset
		barrier := vk.MemoryBarrier{
			sType         = .MEMORY_BARRIER,
			srcAccessMask = {.SHADER_WRITE},
			dstAccessMask = {.TRANSFER_READ},
		}
		vk.CmdPipelineBarrier(
			gctx.batch.cmd,
			{.COMPUTE_SHADER}, {.TRANSFER},
			{}, 1, &barrier, 0, nil, 0, nil,
		)
		region := vk.BufferCopy{srcOffset = 0, dstOffset = offset, size = size}
		vk.CmdCopyBuffer(gctx.batch.cmd, src, gctx.staging.buffer, 1, &region)
		gctx.batch.staging_offset += size
		append(&gctx.pending_downloads, Pending_Download{dst = dst, offset = offset, size = size})
		end_batch(loc)
		return
	}

	_ensure_staging(size, loc)

	cmd := _begin_one_shot(loc)
	region := vk.BufferCopy{srcOffset = 0, dstOffset = 0, size = size}
	vk.CmdCopyBuffer(cmd, src, gctx.staging.buffer, 1, &region)
	_end_one_shot(cmd, loc)

	mem.copy(raw_data(dst), gctx.staging.mapped, int(size))
}

_copy :: proc(dst, src: vk.Buffer, size: vk.DeviceSize, loc := #caller_location) {
	gctx := _gctx(loc)
	if gctx.batch.active {
		region := vk.BufferCopy{srcOffset = 0, dstOffset = 0, size = size}
		vk.CmdCopyBuffer(gctx.batch.cmd, src, dst, 1, &region)
		return
	}
	_one_shot_copy(src, dst, size, loc)
}
