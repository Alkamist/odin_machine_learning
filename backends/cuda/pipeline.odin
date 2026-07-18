package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:os"
import "core:strings"

import "bindings/cuda"
import "bindings/nvrtc"

_include_arg:     cstring
_include_arg_buf: [256]u8

_resolve_include_arg :: proc() -> cstring {
	if _include_arg != nil {
		return _include_arg
	}

	candidates: [3]string
	n := 0
	include_suffix := ODIN_OS == .Windows ? "\\include" : "/include"
	if env := os.get_env("CUDA_PATH", context.temp_allocator); env != "" {
		candidates[n] = strings.concatenate({env, include_suffix}, context.temp_allocator); n += 1
	}
	when ODIN_OS == .Windows {
		candidates[n] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\include"; n += 1
		candidates[n] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\include"; n += 1
	} else {
		candidates[n] = "/usr/local/cuda/include"; n += 1
	}

	for i in 0..<n {
		path := candidates[i]
		if !os.is_dir(path) {
			continue
		}
		w := 0
		_include_arg_buf[w] = '-'; w += 1
		_include_arg_buf[w] = 'I'; w += 1
		fmt.assertf(w + builtin.len(path) + 1 <= builtin.len(_include_arg_buf),
			"CUDA include path too long: %s", path)
		builtin.copy(_include_arg_buf[w:], transmute([]u8)path)
		w += builtin.len(path)

		_include_arg_buf[w] = 0
		_include_arg = cstring(raw_data(_include_arg_buf[:]))

		return _include_arg
	}

	fmt.panicf("could not locate CUDA include directory; set CUDA_PATH or install the CUDA toolkit")
}

_src_to_temp_cstring :: proc(src: []u8) -> cstring {
	buf := builtin.make([]u8, builtin.len(src) + 1, context.temp_allocator)
	builtin.copy(buf, src)
	buf[builtin.len(src)] = 0
	return cstring(raw_data(buf))
}

Pipeline :: struct {
	module:   cuda.Module,
	function: cuda.Function,
	name:     string,

	threads_per_block: u32,
	max_dynamic_smem:  u32,
}

_compile_pipeline :: proc(
	src:           []u8,
	source_name:   cstring,
	entry:         cstring,
	extra_options: []cstring = nil,
	loc                     := #caller_location,
) -> ^Pipeline {
	key := string(source_name)
	if cached, ok := _gpu.pipeline_cache[key]; ok {
		return cached
	}

	src_cstr := _src_to_temp_cstring(src)

	prog: nvrtc.Program
	nvrtc.check(nvrtc.CreateProgram(&prog, src_cstr, source_name, 0, nil, nil), loc=loc)
	defer nvrtc.DestroyProgram(&prog)

	arch_buf: [32]u8
	arch_str := fmt.bprintf(arch_buf[:], "--gpu-architecture=sm_%d%d", _gpu.cc_major, _gpu.cc_minor)
	arch_buf[builtin.len(arch_str)] = 0

	options := builtin.make([dynamic]cstring, 0, 4 + builtin.len(extra_options), context.temp_allocator)
	builtin.append(&options, cstring(raw_data(arch_buf[:])))
	builtin.append(&options, cstring("--use_fast_math"))
	builtin.append(&options, cstring("-default-device"))
	builtin.append(&options, _resolve_include_arg())
	for o in extra_options {
		builtin.append(&options, o)
	}

	if r := nvrtc.CompileProgram(prog, i32(builtin.len(options)), raw_data(options[:])); r != .SUCCESS {
		log_size: uint
		nvrtc.check(nvrtc.GetProgramLogSize(prog, &log_size), loc=loc)
		log := builtin.make([]u8, log_size, context.temp_allocator)
		nvrtc.check(nvrtc.GetProgramLog(prog, raw_data(log)), loc=loc)
		fmt.panicf("NVRTC compile of %s failed:\n%s", source_name, builtin.string(log), loc=loc)
	}

	cubin_size: uint
	nvrtc.check(nvrtc.GetCUBINSize(prog, &cubin_size), loc=loc)
	cubin := builtin.make([]u8, cubin_size, context.temp_allocator)
	nvrtc.check(nvrtc.GetCUBIN(prog, raw_data(cubin)), loc=loc)

	pipeline, err := builtin.new(Pipeline, loc=loc)
	fmt.assertf(err == nil, "Failed to allocate Pipeline: %v", err, loc=loc)

	cuda.check(cuda.ModuleLoadData(&pipeline.module, raw_data(cubin)), loc=loc)
	cuda.check(cuda.ModuleGetFunction(&pipeline.function, pipeline.module, entry), loc=loc)
	pipeline.name = builtin.string(entry)

	_gpu.pipeline_cache[key] = pipeline

	return pipeline
}

_destroy_pipeline :: proc(p: ^Pipeline) {
	if p == nil {
		return
	}
	if p.module != nil {
		cuda.ModuleUnload(p.module)
	}
	builtin.free(p)
}

_acquire_timing_slot :: proc(gctx: ^Context, p: ^Pipeline) -> ^Timing_Slot {
	if gctx.timing_cursor < builtin.len(gctx.timing_pool) {
		slot := &gctx.timing_pool[gctx.timing_cursor]
		slot.pipeline = p
		gctx.timing_cursor += 1
		return slot
	}
	slot: Timing_Slot
	slot.pipeline = p
	cuda.check(cuda.EventCreate(&slot.start, cuda.EVENT_DEFAULT))
	cuda.check(cuda.EventCreate(&slot.end,   cuda.EVENT_DEFAULT))
	builtin.append(&gctx.timing_pool, slot)
	gctx.timing_cursor += 1
	return &gctx.timing_pool[gctx.timing_cursor - 1]
}

_dispatch :: proc(
	p:                         ^Pipeline,
	grid_x, grid_y, grid_z:    u32,
	block_x, block_y, block_z: u32,
	shared_mem_bytes:          u32,
	kernel_args:               []rawptr,
	loc                        := #caller_location,
) {
	gctx := _gctx(loc)

	if gctx.timing_enabled {
		slot := _acquire_timing_slot(gctx, p)
		cuda.check(cuda.EventRecord(slot.start, gctx.stream), loc=loc)
		cuda.check(cuda.LaunchKernel(
			p.function,
			grid_x, grid_y, grid_z,
			block_x, block_y, block_z,
			shared_mem_bytes, gctx.stream,
			raw_data(kernel_args), nil,
		), loc=loc)
		cuda.check(cuda.EventRecord(slot.end, gctx.stream), loc=loc)
	} else {
		cuda.check(cuda.LaunchKernel(
			p.function,
			grid_x, grid_y, grid_z,
			block_x, block_y, block_z,
			shared_mem_bytes, gctx.stream,
			raw_data(kernel_args), nil,
		), loc=loc)
	}
}
