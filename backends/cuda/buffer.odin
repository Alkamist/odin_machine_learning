package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:mem"
import "core:sync"

import "bindings/cuda"

import ml   "../.."
import pool "../activation_pool"

Gpu_Buffer :: struct {
	ptr:  cuda.DevicePtr,
	size: u64,
}
#assert(size_of(Gpu_Buffer) == 16)

buffer_alloc :: proc(byte_count: int, kind: ml.Buffer_Kind, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	if byte_count <= 0 {
		return {}
	}

	gctx := _gctx(loc)

	gb: Gpu_Buffer
	gb.size = u64(byte_count)

	if persist {
		cuda.check(cuda.MemAlloc(&gb.ptr, uint(byte_count)), loc=loc)
		builtin.append(&gctx.persistent, gb.ptr)
	} else {
		gb.ptr = _activation_alloc(gctx, u64(byte_count), loc)
	}

	if kind != .Data || persist {
		cuda.check(cuda.MemsetD8Async(gb.ptr, 0, uint(byte_count), gctx.stream), loc=loc)
	}

	return transmute(ml.Backend_Buffer)gb
}

_activation_alloc :: proc(gctx: ^Context, size: u64, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	return pool.take(&gctx.activation_pool, size, _activation_pool_ops(gctx), loc)
}

buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	gb := transmute(Gpu_Buffer)buffer
	if gb.ptr == 0 {
		return
	}

	gctx := _gctx(loc)

	if gctx.auto_capturing {
		_auto_graph_finish(gctx, loc)
	}

	for ptr, i in gctx.persistent {
		if ptr == gb.ptr {
			cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
			cuda.check(cuda.MemFree(gb.ptr), loc=loc)
			builtin.unordered_remove(&gctx.persistent, i)
			return
		}
	}
}

buffer_get :: proc(buffer: ml.Backend_Buffer, dst: []byte, loc: runtime.Source_Code_Location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	gb := transmute(Gpu_Buffer)buffer
	if gb.ptr == 0 || builtin.len(dst) == 0 {
		return
	}
	fmt.assertf(u64(builtin.len(dst)) <= gb.size, "dst (%d) larger than buffer (%d)", builtin.len(dst), gb.size, loc=loc)

	gctx := _gctx(loc)

	if gctx.auto_capturing {
		_auto_graph_finish(gctx, loc)
	}

	cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
	cuda.check(cuda.MemcpyDtoH(raw_data(dst), gb.ptr, uint(builtin.len(dst))), loc=loc)
}

buffer_set :: proc(buffer: ml.Backend_Buffer, src: []byte, loc: runtime.Source_Code_Location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	gb := transmute(Gpu_Buffer)buffer
	if gb.ptr == 0 || builtin.len(src) == 0 {
		return
	}
	fmt.assertf(u64(builtin.len(src)) <= gb.size, "src (%d) larger than buffer (%d)", builtin.len(src), gb.size, loc=loc)

	gctx := _gctx(loc)

	if gctx.auto_capturing {
		staging := _pinned_staging_take(gctx, u64(builtin.len(src)), loc)
		mem.copy(staging, raw_data(src), builtin.len(src))
		cuda.check(cuda.MemcpyHtoDAsync(gb.ptr, staging, uint(builtin.len(src)), gctx.stream), loc=loc)
		return
	}

	cuda.check(cuda.MemcpyHtoDAsync(gb.ptr, raw_data(src), uint(builtin.len(src)), gctx.stream), loc=loc)
	cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
}

buffer_copy :: proc(dst, src: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	d := transmute(Gpu_Buffer)dst
	s := transmute(Gpu_Buffer)src
	if d.ptr == 0 || s.ptr == 0 {
		return
	}

	bytes := min(d.size, s.size)
	if bytes == 0 {
		return
	}

	gctx := _gctx(loc)
	cuda.check(cuda.MemcpyDtoDAsync(d.ptr, s.ptr, uint(bytes), gctx.stream), loc=loc)
}
