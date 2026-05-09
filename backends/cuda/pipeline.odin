package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:os"
import "core:strings"

import "bindings/cuda"
import "bindings/nvrtc"

// NVRTC needs an explicit include path to find toolkit headers like
// `cuda_bf16.h` (it can't see anything in `<CUDA_PATH>\include` on its own).
// Resolved once at first compile and cached.
_include_arg:     cstring
_include_arg_buf: [256]u8

_resolve_include_arg :: proc() -> cstring {
	if _include_arg != nil { return _include_arg }

	candidates: [3]string
	n := 0
	if env := os.get_env("CUDA_PATH", context.temp_allocator); env != "" {
		candidates[n] = strings.concatenate({env, "\\include"}, context.temp_allocator); n += 1
	}
	candidates[n] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\include"; n += 1
	candidates[n] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\include"; n += 1

	for i in 0..<n {
		path := candidates[i]
		if !os.is_dir(path) { continue }
		// "-I" + path + "\0", written into the static buffer.
		w := 0
		_include_arg_buf[w] = '-'; w += 1
		_include_arg_buf[w] = 'I'; w += 1
		fmt.assertf(w + builtin.len(path) + 1 <= builtin.len(_include_arg_buf),
			"CUDA include path too long: %s", path)
		builtin.copy(_include_arg_buf[w:], transmute([]u8)path); w += builtin.len(path)
		_include_arg_buf[w] = 0
		_include_arg = cstring(raw_data(_include_arg_buf[:]))
		return _include_arg
	}

	fmt.panicf("could not locate CUDA include directory; set CUDA_PATH or install the CUDA toolkit")
}

// Converts kernel-source bytes (from #load) to a NUL-terminated cstring.
// Caches by source pointer so multiple compiles of the same file share the
// same cstring buffer. The buffers leak by design â€” kernel sources are
// embedded statics, the pointer set is small and bounded.
_kernel_cstring_cache: map[rawptr]cstring

_bytes_to_cstring :: proc(src: []u8) -> cstring {
	key := rawptr(raw_data(src))
	if c, ok := _kernel_cstring_cache[key]; ok { return c }
	buf := builtin.make([]u8, builtin.len(src) + 1)
	builtin.copy(buf, src)
	buf[builtin.len(src)] = 0
	c := cstring(raw_data(buf))
	_kernel_cstring_cache[key] = c
	return c
}

// One Pipeline wraps a single CUDA kernel: the parent module (so it can be
// unloaded) and the function pointer to launch. Pipelines are owned by the
// global Gpu_Device and live for the process lifetime, mirroring how the
// vulkan backend caches VkPipeline objects.
Pipeline :: struct {
	module:   cuda.Module,
	function: cuda.Function,
	name:     string,

	// Static per-kernel launch metadata. Filled by the caller of
	// _compile_pipeline; not enforced here.
	threads_per_block: u32,
	max_dynamic_smem:  u32,
}

// Single-source build: NVRTC-compile `src` for the device's actual compute
// capability and return a Pipeline pointing at `entry`. `src` is the raw
// bytes of a `.cu` file (typically embedded with `#load`); we NUL-terminate
// it on the temp_allocator before handing to NVRTC.
//
// Caller-supplied options are appended after the architecture flag and
// `--use_fast_math`. Ops that need strict semantics can pass `--fmad=false`
// in `extra_options` to disable fast-math.
_compile_pipeline :: proc(
	src:        []u8,
	source_name: cstring,
	entry:      cstring,
	extra_options: []cstring = nil,
	loc := #caller_location,
) -> ^Pipeline {
	src_cstr := _bytes_to_cstring(src)

	prog: nvrtc.Program
	nvrtc.check(nvrtc.CreateProgram(&prog, src_cstr, source_name, 0, nil, nil), loc=loc)
	defer nvrtc.DestroyProgram(&prog)

	arch_buf: [32]u8
	arch_str := fmt.bprintf(arch_buf[:], "--gpu-architecture=sm_%d%d", _gpu.cc_major, _gpu.cc_minor)
	arch_buf[builtin.len(arch_str)] = 0

	// Caller-supplied options can opt out of fast-math by passing `--fmad=false`
	// or similar. We detect that and skip the default `--use_fast_math` so the
	// final flag set is consistent (NVRTC honors the last conflicting flag,
	// but a kernel asking for strict semantics shouldn't have to fight defaults).
	strict := false
	for o in extra_options {
		s := string(o)
		if s == "--fmad=false" || s == "-fmad=false" { strict = true; break }
	}

	options := builtin.make([dynamic]cstring, 0, 5 + builtin.len(extra_options), context.temp_allocator)
	builtin.append(&options, cstring(raw_data(arch_buf[:])))
	if !strict {
		builtin.append(&options, cstring("--use_fast_math"))
	}
	builtin.append(&options, cstring("-default-device"))
	builtin.append(&options, _resolve_include_arg())
	for o in extra_options { builtin.append(&options, o) }

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

	builtin.append(&_gpu.pipelines, pipeline)
	return pipeline
}

_destroy_pipeline :: proc(p: ^Pipeline) {
	if p == nil { return }
	if p.module != nil {
		cuda.ModuleUnload(p.module)
	}
	builtin.free(p)
}

// Acquire/recycle a per-dispatch timing slot. Slots are reused across
// forward passes by index Ã¢â‚¬â€ the Nth dispatch in a forward gets the Nth slot
// Ã¢â‚¬â€ so we only ever allocate up to the deepest graph's dispatch count.
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

// Launch a kernel with the standard <<<grid, block, smem, stream>>> shape.
// `kernel_args` is an array of pointers, one per kernel parameter Ã¢â‚¬â€ each
// must point to memory holding the value (e.g. `&device_ptr`, `&n_arg`).
//
// Pipelines are tied to the device, not the context, so the stream + timing
// state come from the active context.
_dispatch :: proc(
	p: ^Pipeline,
	grid_x, grid_y, grid_z:    u32,
	block_x, block_y, block_z: u32,
	shared_mem_bytes:          u32,
	kernel_args:               []rawptr,
	loc := #caller_location,
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