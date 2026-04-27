// Compute-pipeline plumbing.
//
// Most kernels we'll ship have the same shape: N storage buffers (the input
// and output tensors) plus a small push-constant block (sizes, strides,
// scale factors). _make_pipeline takes those two parameters and returns a
// fully-built pipeline. Ops in ops.odin lazy-init their pipeline on first
// use and cache it as a file-local var.
//
// Per-dispatch descriptor sets are allocated from the global descriptor pool
// and freed at the end of the dispatch. This keeps the design simple while
// kernel counts are small; if descriptor allocation ever shows up in a
// profile we can switch to a per-pipeline cached set.
package gpu

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
// command buffer instead of submitting each call individually. end_batch
// inserts one submit + one wait_idle for the entire batch, eliminating the
// per-dispatch drain that dominates forward-pass time.
//
// Transient resources (descriptor sets, host-visible scratch buffers like
// the select-indices buffer) can't be freed until the GPU is done with the
// batch, so they're queued here and reclaimed in end_batch.
//
// Lives on Gpu_Context; one batch in flight per host thread / context.
Batch :: struct {
	active:           bool,
	cmd:              vk.CommandBuffer,
	dispatch_count:   int,
	descriptor_sets:  [dynamic]vk.DescriptorSet,
	pending_buffers:  [dynamic]vk.Buffer,
	pending_memories: [dynamic]vk.DeviceMemory,
	staging_offset:   vk.DeviceSize, // bump-allocator inside the staging
	                                 // buffer for in-batch downloads.
}

// Open a recording batch. All subsequent _dispatch calls record into one
// command buffer; nothing executes on the GPU until end_batch.
begin_batch :: proc(loc := #caller_location) {
	fmt.assertf(_current_gpu_ctx != nil, "gpu.context_begin / context_scope must be called first", loc=loc)
	gctx := _current_gpu_ctx
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
}

// Close the batch: submit, wait, then free everything that was queued
// during recording. Synchronous on purpose — fits the existing model where
// callers expect GPU results to be ready when control returns.
end_batch :: proc(loc := #caller_location) {
	fmt.assertf(_current_gpu_ctx != nil, "gpu.context_begin / context_scope must be called first", loc=loc)
	gctx := _current_gpu_ctx
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

	// Flush any downloads that were folded into the batch: the device→staging
	// copies executed on the queue, and now that the wait has returned the
	// staging buffer's HOST_COHERENT contents are valid to memcpy out.
	if len(gctx.pending_downloads) > 0 {
		base := uintptr(gctx.staging.mapped)
		for d in gctx.pending_downloads {
			src := rawptr(base + uintptr(d.offset))
			mem.copy(raw_data(d.dst), src, int(d.size))
		}
		clear(&gctx.pending_downloads)
	}

	vk.FreeCommandBuffers(_gpu.device, gctx.command_pool, 1, &cmd)

	if len(gctx.batch.descriptor_sets) > 0 {
		vk.FreeDescriptorSets(
			_gpu.device, gctx.descriptor_pool,
			u32(len(gctx.batch.descriptor_sets)),
			raw_data(gctx.batch.descriptor_sets[:]),
		)
	}
	for buf in gctx.batch.pending_buffers  { vk.DestroyBuffer(_gpu.device, buf, nil) }
	for mem in gctx.batch.pending_memories { vk.FreeMemory(_gpu.device, mem, nil) }

	clear(&gctx.batch.descriptor_sets)
	clear(&gctx.batch.pending_buffers)
	clear(&gctx.batch.pending_memories)
	gctx.batch.active         = false
	gctx.batch.cmd            = nil
	gctx.batch.dispatch_count = 0
}

// Schedule (buf, mem) for destruction once the current batch finishes. If
// no batch is active, destroy immediately — callers in that mode have
// already waited on the GPU via the per-dispatch wait_idle.
_queue_destroy_buffer :: proc(buf: vk.Buffer, mem: vk.DeviceMemory) {
	if _current_gpu_ctx != nil && _current_gpu_ctx.batch.active {
		append(&_current_gpu_ctx.batch.pending_buffers,  buf)
		append(&_current_gpu_ctx.batch.pending_memories, mem)
	} else {
		vk.DestroyBuffer(_gpu.device, buf, nil)
		vk.FreeMemory(_gpu.device, mem, nil)
	}
}

// Build a compute pipeline from SPIR-V bytes. `num_buffers` is the count of
// `layout(set=0, binding=N) buffer ...` entries declared in the shader (all
// STORAGE_BUFFER on the COMPUTE stage). `push_constant_size` matches the
// shader's `layout(push_constant) uniform { ... };` block size in bytes;
// pass 0 if the kernel has none.
_make_pipeline :: proc(spirv: []u8, num_buffers: u32, push_constant_size: u32, loc := #caller_location) -> ^Pipeline {
	fmt.assertf(_gpu.device != nil, "gpu.init() must be called first", loc=loc)
	fmt.assertf(len(spirv) % 4 == 0, "SPIR-V byte length must be a multiple of 4", loc=loc)

	p := new(Pipeline)
	p.num_buffers        = num_buffers
	p.push_constant_size = push_constant_size

	// Shader module.
	module_info := vk.ShaderModuleCreateInfo{
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(spirv),
		pCode    = cast(^u32) raw_data(spirv),
	}
	res := vk.CreateShaderModule(_gpu.device, &module_info, nil, &p.shader_module)
	fmt.assertf(res == .SUCCESS, "vkCreateShaderModule failed: %v", res, loc=loc)

	// Descriptor set layout: one STORAGE_BUFFER per buffer, all on COMPUTE.
	bindings := make([]vk.DescriptorSetLayoutBinding, num_buffers, context.temp_allocator)
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
		bindingCount = num_buffers,
		pBindings    = raw_data(bindings),
	}
	res = vk.CreateDescriptorSetLayout(_gpu.device, &dsl_info, nil, &p.descriptor_set_layout)
	fmt.assertf(res == .SUCCESS, "vkCreateDescriptorSetLayout failed: %v", res, loc=loc)

	// Pipeline layout (descriptor set + optional push constants).
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

	// Compute pipeline.
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
	if p.pipeline != 0              { vk.DestroyPipeline(_gpu.device, p.pipeline, nil) }
	if p.pipeline_layout != 0       { vk.DestroyPipelineLayout(_gpu.device, p.pipeline_layout, nil) }
	if p.descriptor_set_layout != 0 { vk.DestroyDescriptorSetLayout(_gpu.device, p.descriptor_set_layout, nil) }
	if p.shader_module != 0         { vk.DestroyShaderModule(_gpu.device, p.shader_module, nil) }
	free(p)
}

// Run a kernel synchronously: allocate a descriptor set, bind buffers + push
// constants, dispatch `group_count_x` workgroups, wait, free the set. Caller
// supplies workgroup count along X only — that covers every elementwise /
// row-parallel kernel we'll need short-term. 2D dispatches will get their
// own helper when the first 2D kernel lands.
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

	gctx := _current_gpu_ctx
	fmt.assertf(gctx != nil, "no active gpu Context — call gpu.context_begin / context_scope", loc=loc)

	// Allocate descriptor set.
	dsl := p.descriptor_set_layout
	ds_alloc := vk.DescriptorSetAllocateInfo{
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		descriptorPool     = gctx.descriptor_pool,
		descriptorSetCount = 1,
		pSetLayouts        = &dsl,
	}
	set: vk.DescriptorSet
	res := vk.AllocateDescriptorSets(_gpu.device, &ds_alloc, &set)
	fmt.assertf(res == .SUCCESS, "vkAllocateDescriptorSets failed: %v", res, loc=loc)

	// Write buffer bindings.
	buf_infos := make([]vk.DescriptorBufferInfo, p.num_buffers, context.temp_allocator)
	writes    := make([]vk.WriteDescriptorSet,   p.num_buffers, context.temp_allocator)
	for i in 0 ..< p.num_buffers {
		buf_infos[i] = vk.DescriptorBufferInfo{
			buffer = buffers[i],
			offset = 0,
			range  = vk.DeviceSize(vk.WHOLE_SIZE),
		}
		writes[i] = vk.WriteDescriptorSet{
			sType           = .WRITE_DESCRIPTOR_SET,
			dstSet          = set,
			dstBinding      = i,
			descriptorCount = 1,
			descriptorType  = .STORAGE_BUFFER,
			pBufferInfo     = &buf_infos[i],
		}
	}
	vk.UpdateDescriptorSets(_gpu.device, p.num_buffers, raw_data(writes), 0, nil)

	// Auto-start a batch if one isn't already open. The batch is flushed
	// implicitly by `ml.clear` (via `Backend.flush`) and by `get_data` /
	// `download_tensor` before they read host memory, so most callers
	// never have to think about batches at all.
	if !gctx.batch.active {
		begin_batch()
	}

	// Record into the open batch. Insert a barrier before every dispatch
	// so prior writes are visible to this one — we don't track per-buffer
	// dependencies, so a global memory barrier
	// ({SHADER_WRITE, TRANSFER_WRITE} → {SHADER_READ, SHADER_WRITE}) on
	// COMPUTE+TRANSFER → COMPUTE makes both pipelined dispatches and
	// alloc-time CmdFillBuffer zero-fills (recorded into the same CB)
	// correct without per-op metadata.
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
	vk.CmdBindDescriptorSets(cmd, .COMPUTE, p.pipeline_layout, 0, 1, &set, 0, nil)
	if p.push_constant_size > 0 {
		vk.CmdPushConstants(cmd, p.pipeline_layout, {.COMPUTE}, 0, p.push_constant_size, push_constants)
	}
	vk.CmdDispatch(cmd, group_count_x, group_count_y, group_count_z)

	append(&gctx.batch.descriptor_sets, set)
	gctx.batch.dispatch_count += 1
}

// Convenience: ceiling-divide for picking workgroup counts.
_div_up :: proc(a, b: int) -> u32 {
	return u32((a + b - 1) / b)
}
