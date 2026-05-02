package machine_learning_backend_gpu

import "base:builtin"

import "core:fmt"
import "core:mem"

import vk "vendor:vulkan"

Pipeline :: struct {
	descriptor_set_layout: vk.DescriptorSetLayout,
	pipeline_layout:       vk.PipelineLayout,
	pipeline:              vk.Pipeline,
	shader_module:         vk.ShaderModule,
	num_buffers:           u32,
	push_constant_size:    u32,
}

// Batched-dispatch state. While active, _dispatch records into a single
// command buffer; nothing executes on the GPU until end_batch.
Batch :: struct {
	active:           bool,
	cmd:              vk.CommandBuffer,
	dispatch_count:   int,
	pending_buffers:  [dynamic]vk.Buffer,
	pending_memories: [dynamic]vk.DeviceMemory,
	staging_offset:   vk.DeviceSize,
}

flush :: proc(loc := #caller_location) {
	gctx := _gctx(loc)
	if gctx.batch.active {
		end_batch(loc)
	}
}

begin_batch :: proc(loc := #caller_location) {
	gctx := _gctx(loc)
	fmt.assertf(!gctx.batch.active, "begin_batch: already in a batch", loc=loc)

	alloc_info := vk.CommandBufferAllocateInfo{
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = gctx.command_pool,
		level              = .PRIMARY,
		commandBufferCount = 1,
	}
	res := vk.AllocateCommandBuffers(_gpu.device, &alloc_info, &gctx.batch.cmd)
	fmt.assertf(res == .SUCCESS, "vkAllocateCommandBuffers (batch) failed: %v", res, loc=loc)

	begin := vk.CommandBufferBeginInfo{
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}
	res = vk.BeginCommandBuffer(gctx.batch.cmd, &begin)
	fmt.assertf(res == .SUCCESS, "vkBeginCommandBuffer (batch) failed: %v", res, loc=loc)

	gctx.batch.active         = true
	gctx.batch.dispatch_count = 0
	gctx.batch.staging_offset = 0

	if gctx.timing_enabled && gctx.query_pool != 0 {
		vk.CmdResetQueryPool(gctx.batch.cmd, gctx.query_pool, 0, gctx.query_capacity)
		gctx.query_used = 0
		builtin.clear(&gctx.pending_queries)
	}
}

end_batch :: proc(loc := #caller_location) {
	gctx := _gctx(loc)
	fmt.assertf(gctx.batch.active, "end_batch: no active batch", loc=loc)

	cmd := gctx.batch.cmd
	res := vk.EndCommandBuffer(cmd)
	fmt.assertf(res == .SUCCESS, "vkEndCommandBuffer (batch) failed: %v", res, loc=loc)

	submit := vk.SubmitInfo{
		sType              = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers    = &cmd,
	}
	res = vk.QueueSubmit(_gpu.queue, 1, &submit, 0)
	fmt.assertf(res == .SUCCESS, "vkQueueSubmit (batch) failed: %v", res, loc=loc)
	res = vk.QueueWaitIdle(_gpu.queue)
	fmt.assertf(res == .SUCCESS, "vkQueueWaitIdle (batch) failed: %v", res, loc=loc)

	if len(gctx.pending_downloads) > 0 {
		base := uintptr(gctx.staging.mapped)
		for d in gctx.pending_downloads {
			src := rawptr(base + uintptr(d.offset))
			mem.copy(raw_data(d.dst), src, int(d.size))
		}
		builtin.clear(&gctx.pending_downloads)
	}

	if gctx.timing_enabled && gctx.query_pool != 0 && len(gctx.pending_queries) > 0 {
		results := builtin.make([]u64, gctx.query_used, context.temp_allocator)
		_ = vk.GetQueryPoolResults(
			_gpu.device, gctx.query_pool, 0, gctx.query_used,
			builtin.len(results) * size_of(u64),
			raw_data(results), size_of(u64),
			{._64, .WAIT},
		)
		period := f64(_gpu.timestamp_period_ns)
		for q in gctx.pending_queries {
			delta_ticks := results[q.end_idx] - results[q.start_idx]
			ns := i64(f64(delta_ticks) * period)
			stat := gctx.timing_totals[q.pipeline]
			stat.total_ns += ns
			stat.count    += 1
			gctx.timing_totals[q.pipeline] = stat
		}
		builtin.clear(&gctx.pending_queries)
	}

	vk.FreeCommandBuffers(_gpu.device, gctx.command_pool, 1, &cmd)

	for buf in gctx.batch.pending_buffers  { vk.DestroyBuffer(_gpu.device, buf, nil) }
	for m   in gctx.batch.pending_memories { vk.FreeMemory(_gpu.device, m, nil)      }

	builtin.clear(&gctx.batch.pending_buffers)
	builtin.clear(&gctx.batch.pending_memories)
	gctx.batch.active         = false
	gctx.batch.cmd            = nil
	gctx.batch.dispatch_count = 0
}

_queue_destroy_buffer :: proc(buf: vk.Buffer, m: vk.DeviceMemory) {
	gctx := _gctx()
	if gctx.batch.active {
		append(&gctx.batch.pending_buffers,  buf)
		append(&gctx.batch.pending_memories, m)
	} else {
		vk.DestroyBuffer(_gpu.device, buf, nil)
		vk.FreeMemory(_gpu.device, m, nil)
	}
}

_make_pipeline :: proc(spirv: []u8, num_buffers: u32, push_constant_size: u32, loc := #caller_location) -> ^Pipeline {
	fmt.assertf(_gpu.device != nil, "device_init must be called first", loc=loc)
	fmt.assertf(len(spirv) % 4 == 0, "SPIR-V byte length must be a multiple of 4", loc=loc)

	p := builtin.new(Pipeline)
	p.num_buffers        = num_buffers
	p.push_constant_size = push_constant_size

	module_info := vk.ShaderModuleCreateInfo{
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(spirv),
		pCode    = cast(^u32) raw_data(spirv),
	}
	res := vk.CreateShaderModule(_gpu.device, &module_info, nil, &p.shader_module)
	fmt.assertf(res == .SUCCESS, "vkCreateShaderModule failed: %v", res, loc=loc)

	bindings := builtin.make([]vk.DescriptorSetLayoutBinding, num_buffers, context.temp_allocator)
	for i in 0 ..< num_buffers {
		bindings[i] = vk.DescriptorSetLayoutBinding{
			binding         = i,
			descriptorType  = .STORAGE_BUFFER,
			descriptorCount = 1,
			stageFlags      = {.COMPUTE},
		}
	}
	dsl_info := vk.DescriptorSetLayoutCreateInfo{
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		flags        = {.PUSH_DESCRIPTOR_KHR},
		bindingCount = num_buffers,
		pBindings    = raw_data(bindings),
	}
	res = vk.CreateDescriptorSetLayout(_gpu.device, &dsl_info, nil, &p.descriptor_set_layout)
	fmt.assertf(res == .SUCCESS, "vkCreateDescriptorSetLayout failed: %v", res, loc=loc)

	pcr := vk.PushConstantRange{
		stageFlags = {.COMPUTE},
		offset     = 0,
		size       = push_constant_size,
	}
	pl_info := vk.PipelineLayoutCreateInfo{
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = 1,
		pSetLayouts            = &p.descriptor_set_layout,
		pushConstantRangeCount = 0 if push_constant_size == 0 else 1,
		pPushConstantRanges    = nil if push_constant_size == 0 else &pcr,
	}
	res = vk.CreatePipelineLayout(_gpu.device, &pl_info, nil, &p.pipeline_layout)
	fmt.assertf(res == .SUCCESS, "vkCreatePipelineLayout failed: %v", res, loc=loc)

	stage := vk.PipelineShaderStageCreateInfo{
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = {.COMPUTE},
		module = p.shader_module,
		pName  = "main",
	}
	cp_info := vk.ComputePipelineCreateInfo{
		sType  = .COMPUTE_PIPELINE_CREATE_INFO,
		stage  = stage,
		layout = p.pipeline_layout,
	}
	res = vk.CreateComputePipelines(_gpu.device, 0, 1, &cp_info, nil, &p.pipeline)
	fmt.assertf(res == .SUCCESS, "vkCreateComputePipelines failed: %v", res, loc=loc)

	append(&_gpu.pipelines, p)
	return p
}

_destroy_pipeline :: proc(p: ^Pipeline) {
	if p.pipeline != 0              { vk.DestroyPipeline(_gpu.device, p.pipeline, nil)                           }
	if p.pipeline_layout != 0       { vk.DestroyPipelineLayout(_gpu.device, p.pipeline_layout, nil)              }
	if p.descriptor_set_layout != 0 { vk.DestroyDescriptorSetLayout(_gpu.device, p.descriptor_set_layout, nil)   }
	if p.shader_module != 0         { vk.DestroyShaderModule(_gpu.device, p.shader_module, nil)                  }
	builtin.free(p)
}

_dispatch :: proc(
	p: ^Pipeline,
	buffers: []vk.Buffer,
	push_constants: rawptr,
	group_count_x: u32,
	group_count_y: u32 = 1,
	group_count_z: u32 = 1,
	loc := #caller_location,
) {
	fmt.assertf(u32(len(buffers)) == p.num_buffers,
		"dispatch: pipeline expects %v buffers, got %v", p.num_buffers, len(buffers), loc=loc)

	gctx := _gctx(loc)

	if !gctx.batch.active {
		begin_batch(loc)
	}

	timing := gctx.timing_enabled && gctx.query_pool != 0 && gctx.query_used + 2 <= gctx.query_capacity
	start_idx, end_idx: u32
	if timing {
		start_idx = gctx.query_used
		end_idx   = start_idx + 1
		gctx.query_used += 2
	}

	buf_infos := builtin.make([]vk.DescriptorBufferInfo, p.num_buffers, context.temp_allocator)
	writes    := builtin.make([]vk.WriteDescriptorSet,   p.num_buffers, context.temp_allocator)
	for i in 0 ..< p.num_buffers {
		buf_infos[i] = vk.DescriptorBufferInfo{
			buffer = buffers[i],
			offset = 0,
			range  = vk.DeviceSize(vk.WHOLE_SIZE),
		}
		writes[i] = vk.WriteDescriptorSet{
			sType           = .WRITE_DESCRIPTOR_SET,
			dstBinding      = i,
			descriptorCount = 1,
			descriptorType  = .STORAGE_BUFFER,
			pBufferInfo     = &buf_infos[i],
		}
	}

	// Insert a memory barrier before every dispatch so prior shader writes
	// and CmdFillBuffer zero-fills are visible. Per-buffer dependencies
	// aren't tracked, so a global SHADER+TRANSFER → SHADER barrier covers
	// both pipelined dispatches and alloc-time fills in the same CB.
	cmd := gctx.batch.cmd
	barrier := vk.MemoryBarrier{
		sType         = .MEMORY_BARRIER,
		srcAccessMask = {.SHADER_WRITE, .TRANSFER_WRITE},
		dstAccessMask = {.SHADER_READ, .SHADER_WRITE},
	}
	vk.CmdPipelineBarrier(
		cmd,
		{.COMPUTE_SHADER, .TRANSFER}, {.COMPUTE_SHADER},
		{},
		1, &barrier,
		0, nil,
		0, nil,
	)
	vk.CmdBindPipeline(cmd, .COMPUTE, p.pipeline)
	vk.CmdPushDescriptorSetKHR(cmd, .COMPUTE, p.pipeline_layout, 0, p.num_buffers, raw_data(writes))
	if p.push_constant_size > 0 {
		vk.CmdPushConstants(cmd, p.pipeline_layout, {.COMPUTE}, 0, p.push_constant_size, push_constants)
	}
	if timing {
		vk.CmdWriteTimestamp(cmd, {.TOP_OF_PIPE},    gctx.query_pool, start_idx)
	}
	vk.CmdDispatch(cmd, group_count_x, group_count_y, group_count_z)
	if timing {
		vk.CmdWriteTimestamp(cmd, {.BOTTOM_OF_PIPE}, gctx.query_pool, end_idx)
		append(&gctx.pending_queries, Pending_Query{pipeline = p, start_idx = start_idx, end_idx = end_idx})
	}

	gctx.batch.dispatch_count += 1
}

_div_up :: #force_inline proc(a, b: int) -> u32 {
	return u32((a + b - 1) / b)
}
