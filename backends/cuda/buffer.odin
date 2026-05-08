package machine_learning_backend_cuda

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:sync"
import "core:time"

import "bindings/cuda"

import ml "../../"

// 16-byte device-side buffer handle, matches the size_of(Backend_Buffer) the
// frontend reserves. Stores the raw device pointer plus the logical byte
// count so buffer_copy / buffer_get can run without a side table.
//
// The persist flag is reserved for a future arena/sub-allocator; today every
// buffer is its own cuMemAlloc. CUDA's allocator is fast enough for the sizes
// we hit early, and we'd rather measure than pre-optimize. ggml-cuda only
// reaches for a pool once profiling shows allocator pressure.
Gpu_Buffer :: struct {
	ptr:  cuda.DevicePtr,  // 8
	size: u64,              // 8 (bytes)
}
#assert(size_of(Gpu_Buffer) == 16)

// f32(1.0) bit pattern. Useful for cuMemsetD32-stamped grad-of-loss buffers,
// matching the vulkan backend's F32_ONE_BITS use.
F32_ONE_BITS :: u32(0x3F800000)

buffer_alloc :: proc(byte_count: int, persist: bool, loc: runtime.Source_Code_Location) -> ml.Backend_Buffer {
	t_start := time.tick_now()
	defer {
		_alloc_count += 1
		_alloc_ns    += i64(time.tick_since(t_start))
	}
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	if byte_count <= 0 {
		return transmute(ml.Backend_Buffer)Gpu_Buffer{}
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

	// Backward kernels accumulate into gradient buffers (`+=`), so the buffer
	// must start zeroed every forward pass Ã¢â‚¬â€ mirrors the vulkan backend's
	// _record_fill_zero.
	cuda.check(cuda.MemsetD8Async(gb.ptr, 0, uint(byte_count), gctx.stream), loc=loc)

	return transmute(ml.Backend_Buffer)gb
}

// Activation pool: hand out the slot at `activation_cursor`, reallocating only
// on size mismatch. clear() rewinds the cursor; tail slots persist for reuse.
_activation_alloc :: proc(gctx: ^Context, size: u64, loc: runtime.Source_Code_Location) -> cuda.DevicePtr {
	if gctx.activation_cursor < builtin.len(gctx.activation_pool) {
		slot := &gctx.activation_pool[gctx.activation_cursor]
		if slot.size == size {
			gctx.activation_cursor += 1
			return slot.ptr
		}
		// Size mismatch: every slot from here on is stale relative to a
		// new allocation pattern. Free this one and let the trailing slots
		// also rebuild as the next allocs come in.
		cuda.check(cuda.MemFree(slot.ptr), loc=loc)
		// Drop the doomed tail entirely; subsequent allocs in this pass
		// will append fresh slots.
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

// Explicit free is a no-op for activation buffers (they live in the pool and
// are only torn down at context_destroy / size-mismatch). For persistent
// buffers we honor the request, freeing immediately.
buffer_free :: proc(buffer: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gb := transmute(Gpu_Buffer)buffer
	if gb.ptr == 0 { return }

	gctx := _gctx(loc)

	for ptr, i in gctx.persistent {
		if ptr == gb.ptr {
			cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
			cuda.check(cuda.MemFree(gb.ptr), loc=loc)
			builtin.unordered_remove(&gctx.persistent, i)
			return
		}
	}
	// Activation: silently retained by the pool.
}

buffer_get :: proc(buffer: ml.Backend_Buffer, dst: []byte, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gb := transmute(Gpu_Buffer)buffer
	if gb.ptr == 0 || builtin.len(dst) == 0 { return }
	fmt.assertf(u64(builtin.len(dst)) <= gb.size,
		"buffer_get: dst (%d) larger than buffer (%d)", builtin.len(dst), gb.size, loc=loc)

	gctx := _gctx(loc)
	cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
	cuda.check(cuda.MemcpyDtoH(raw_data(dst), gb.ptr, uint(builtin.len(dst))), loc=loc)
}

buffer_set :: proc(buffer: ml.Backend_Buffer, src: []byte, loc: runtime.Source_Code_Location) {
	t_start := time.tick_now()
	defer {
		_upload_count += 1
		_upload_ns    += i64(time.tick_since(t_start))
	}
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	gb := transmute(Gpu_Buffer)buffer
	if gb.ptr == 0 || builtin.len(src) == 0 { return }
	fmt.assertf(u64(builtin.len(src)) <= gb.size,
		"buffer_set: src (%d) larger than buffer (%d)", builtin.len(src), gb.size, loc=loc)

	gctx := _gctx(loc)
	cuda.check(cuda.MemcpyHtoDAsync(gb.ptr, raw_data(src), uint(builtin.len(src)), gctx.stream), loc=loc)
	// Synchronous semantics for the public API: caller's `src` must not be
	// reused before the copy completes. Pinning the staging would let us
	// overlap, but that's a Phase 7 perf concern.
	cuda.check(cuda.StreamSynchronize(gctx.stream), loc=loc)
}

buffer_copy :: proc(dst, src: ml.Backend_Buffer, loc: runtime.Source_Code_Location) {
	sync.mutex_lock(&_gpu_mutex)
	defer sync.mutex_unlock(&_gpu_mutex)

	d := transmute(Gpu_Buffer)dst
	s := transmute(Gpu_Buffer)src
	if d.ptr == 0 || s.ptr == 0 { return }

	bytes := min(d.size, s.size)
	if bytes == 0 { return }

	gctx := _gctx(loc)
	cuda.check(cuda.MemcpyDtoDAsync(d.ptr, s.ptr, uint(bytes), gctx.stream), loc=loc)
}
