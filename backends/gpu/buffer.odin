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

// Sub-allocate an activation (non-persistent) buffer. VkBuffer handles are
// pooled and reused across forward passes: the Nth alloc in a forward returns
// the Nth slot's buffer, with arena.used bumped to match the original binding.
// On size mismatch the slot is destroyed and recreated; on first encounter a
// new buffer is bound and appended to the pool.
_create_pooled_activation_buffer :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	mem_flags: vk.MemoryPropertyFlags,
	loc := #caller_location,
) -> (buffer: vk.Buffer, memory: vk.DeviceMemory, fresh: bool) {
	gctx := _gctx(loc)

	if gctx.activation_cursor < len(gctx.activation_pool) {
		slot := gctx.activation_pool[gctx.activation_cursor]
		if slot.size == size {
			arena := &gctx.activation_arenas[slot.arena_idx]
			aligned := (arena.used + slot.alignment - 1) & ~(slot.alignment - 1)
			if aligned == slot.offset {
				arena.used = aligned + size
				gctx.activation_cursor += 1
				return slot.buf, 0, false
			}
		}
		// Either size differs or arena.used has drifted (an earlier slot was
		// rebuilt at a different size). Every slot from here on has stale
		// offsets — destroy them all and rebuild the tail.
		for i in gctx.activation_cursor ..< len(gctx.activation_pool) {
			doomed := gctx.activation_pool[i]
			vk.DestroyBuffer(_gpu.device, doomed.buf, nil)
			delete_key(&gctx.sizes, doomed.buf)
		}
		resize(&gctx.activation_pool, gctx.activation_cursor)
	}

	new_buf, new_slot := _alloc_activation_slot(size, usage, mem_flags, loc)
	append(&gctx.activation_pool, new_slot)
	gctx.activation_cursor += 1
	return new_buf, 0, true
}

// Create a fresh VkBuffer + bind it into the next free slice of an arena
// (allocating a new arena block if the existing arenas are full).
_alloc_activation_slot :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	mem_flags: vk.MemoryPropertyFlags,
	loc := #caller_location,
) -> (buffer: vk.Buffer, slot: Activation_Slot) {
	gctx := _gctx(loc)

	info := vk.BufferCreateInfo{
		sType       = .BUFFER_CREATE_INFO,
		size        = size,
		usage       = usage,
		sharingMode = .EXCLUSIVE,
	}
	res := vk.CreateBuffer(_gpu.device, &info, nil, &buffer)
	fmt.assertf(res == .SUCCESS, "vkCreateBuffer (activation) failed: %v", res, loc=loc)

	reqs: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(_gpu.device, buffer, &reqs)
	mem_type_idx := _pick_memory_type(reqs.memoryTypeBits, mem_flags, loc)

	for &arena, idx in gctx.activation_arenas {
		if arena.mem_type_idx != mem_type_idx do continue
		aligned := (arena.used + reqs.alignment - 1) & ~(reqs.alignment - 1)
		if aligned + reqs.size <= arena.size {
			res = vk.BindBufferMemory(_gpu.device, buffer, arena.memory, aligned)
			fmt.assertf(res == .SUCCESS, "vkBindBufferMemory (activation reuse) failed: %v", res, loc=loc)
			arena.used = aligned + reqs.size
			return buffer, Activation_Slot{
				buf       = buffer,
				arena_idx = idx,
				offset    = aligned,
				size      = size,
				alignment = reqs.alignment,
			}
		}
	}

	block_size := POOL_BLOCK_SIZE
	if reqs.size > block_size do block_size = reqs.size
	alloc_info := vk.MemoryAllocateInfo{
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = block_size,
		memoryTypeIndex = mem_type_idx,
	}
	new_block_memory: vk.DeviceMemory
	res = vk.AllocateMemory(_gpu.device, &alloc_info, nil, &new_block_memory)
	fmt.assertf(res == .SUCCESS, "vkAllocateMemory (activation arena, %v MB) failed: %v",
		f64(block_size) / (1024 * 1024), res, loc=loc)

	new_arena_idx := len(gctx.activation_arenas)
	append(&gctx.activation_arenas, Pool_Block{
		memory       = new_block_memory,
		size         = block_size,
		used         = reqs.size,
		mem_type_idx = mem_type_idx,
	})

	res = vk.BindBufferMemory(_gpu.device, buffer, new_block_memory, 0)
	fmt.assertf(res == .SUCCESS, "vkBindBufferMemory (activation arena first) failed: %v", res, loc=loc)
	return buffer, Activation_Slot{
		buf       = buffer,
		arena_idx = new_arena_idx,
		offset    = 0,
		size      = size,
		alignment = reqs.alignment,
	}
}

// Sub-allocate a persistent buffer from the context's pool. Each pool block
// is one VkDeviceMemory backing many VkBuffers via offset binding, which
// avoids the per-`vkAllocateMemory` overhead NVIDIA's Windows driver reserves
// (~tens of MB minimum). The returned `Gpu_Buffer.memory == 0` to signal that
// the memory is owned by the pool, not the buffer; `_destroy_gpu_buffer`
// skips `vkFreeMemory` in that case.
_create_pooled_persistent_buffer :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	mem_flags: vk.MemoryPropertyFlags,
	loc := #caller_location,
) -> (buffer: vk.Buffer, memory: vk.DeviceMemory) {
	gctx := _gctx(loc)

	info := vk.BufferCreateInfo{
		sType       = .BUFFER_CREATE_INFO,
		size        = size,
		usage       = usage,
		sharingMode = .EXCLUSIVE,
	}
	res := vk.CreateBuffer(_gpu.device, &info, nil, &buffer)
	fmt.assertf(res == .SUCCESS, "vkCreateBuffer (pooled) failed: %v", res, loc=loc)

	reqs: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(_gpu.device, buffer, &reqs)
	mem_type_idx := _pick_memory_type(reqs.memoryTypeBits, mem_flags, loc)

	for &block in gctx.persistent_pool {
		if block.mem_type_idx != mem_type_idx do continue
		aligned := (block.used + reqs.alignment - 1) & ~(reqs.alignment - 1)
		if aligned + reqs.size <= block.size {
			res = vk.BindBufferMemory(_gpu.device, buffer, block.memory, aligned)
			fmt.assertf(res == .SUCCESS, "vkBindBufferMemory (pooled reuse) failed: %v", res, loc=loc)
			block.used = aligned + reqs.size
			return buffer, 0
		}
	}

	block_size := POOL_BLOCK_SIZE
	if reqs.size > block_size do block_size = reqs.size
	alloc_info := vk.MemoryAllocateInfo{
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = block_size,
		memoryTypeIndex = mem_type_idx,
	}
	new_block_memory: vk.DeviceMemory
	res = vk.AllocateMemory(_gpu.device, &alloc_info, nil, &new_block_memory)
	fmt.assertf(res == .SUCCESS, "vkAllocateMemory (pool block, %v MB) failed: %v",
		f64(block_size) / (1024 * 1024), res, loc=loc)

	append(&gctx.persistent_pool, Pool_Block{
		memory       = new_block_memory,
		size         = block_size,
		used         = reqs.size,
		mem_type_idx = mem_type_idx,
	})

	res = vk.BindBufferMemory(_gpu.device, buffer, new_block_memory, 0)
	fmt.assertf(res == .SUCCESS, "vkBindBufferMemory (pool block first) failed: %v", res, loc=loc)
	return buffer, 0
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