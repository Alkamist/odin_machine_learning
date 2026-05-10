package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:sync"

import "bindings/cuda"

import ml "../../"

Gpu_Buffer :: struct {
	ptr:  cuda.DevicePtr,
	size: u64,
}
#assert(size_of(Gpu_Buffer) == 16)

F32_ONE_BITS :: 0x3F800000

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
	if gctx.activation_cursor < builtin.len(gctx.activation_pool) {
		slot := &gctx.activation_pool[gctx.activation_cursor]
		if slot.size == size {
			gctx.activation_cursor += 1
			return slot.ptr
		}
		cuda.check(cuda.MemFree(slot.ptr), loc=loc)
		for i in gctx.activation_cursor + 1 ..< builtin.len(gctx.activation_pool) {
			cuda.check(cuda.MemFree(gctx.activation_pool[i].ptr), loc=loc)
		}
		builtin.resize(&gctx.activation_pool, gctx.activation_cursor)
	}

	new_ptr: cuda.DevicePtr
	cuda.check(cuda.MemAlloc(&new_ptr, uint(size)), loc=loc)
	builtin.append(&gctx.activation_pool, Activation_Slot{ptr=new_ptr, size=size})
	gctx.activation_cursor += 1

	return new_ptr
}

buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	sync.lock(&_gpu_mutex)
	defer sync.unlock(&_gpu_mutex)

	gb := transmute(Gpu_Buffer)buffer
	if gb.ptr == 0 {
		return
	}

	gctx := _gctx(loc)

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
	fmt.assertf(u64(builtin.len(dst)) <= gb.size, "buffer_get: dst (%d) larger than buffer (%d)", builtin.len(dst), gb.size, loc=loc)

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
	fmt.assertf(u64(builtin.len(src)) <= gb.size, "buffer_set: src (%d) larger than buffer (%d)", builtin.len(src), gb.size, loc=loc)

	gctx := _gctx(loc)
	cuda.check(cuda.MemcpyHtoDAsync(gb.ptr, raw_data(src), uint(builtin.len(src)), gctx.stream), loc=loc)

	if !gctx.auto_capturing {
		cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
	}
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