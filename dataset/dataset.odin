package dataset

import "base:builtin"

import "core:math/rand"

Batcher :: struct {
	indices:    []int,
	batch_size: int,
	shuffle:    bool,
	drop_last:  bool,
	cursor:     int,
}

batcher_make :: proc {
	batcher_make_count,
	batcher_make_indices,
}

@(require_results)
batcher_make_count :: proc(sample_count, batch_size: int, shuffle := true, drop_last := true, allocator := context.allocator, loc := #caller_location) -> Batcher {
	assert(sample_count > 0, "batcher requires at least one sample", loc=loc)
	indices := make([]int, sample_count, allocator)
	for i in 0 ..< sample_count {
		indices[i] = i
	}
	return _batcher_from_owned(indices, batch_size, shuffle, drop_last, loc=loc)
}

@(require_results)
batcher_make_indices :: proc(indices: []int, batch_size: int, shuffle := true, drop_last := true, allocator := context.allocator, loc := #caller_location) -> Batcher {
	assert(builtin.len(indices) > 0, "batcher requires at least one sample", loc=loc)
	owned := make([]int, builtin.len(indices), allocator)
	copy(owned, indices)
	return _batcher_from_owned(owned, batch_size, shuffle, drop_last, loc=loc)
}

@(require_results)
_batcher_from_owned :: proc(indices: []int, batch_size: int, shuffle, drop_last: bool, loc := #caller_location) -> Batcher {
	assert(batch_size > 0, "batch_size must be positive", loc=loc)
	batcher := Batcher{
		indices    = indices,
		batch_size = batch_size,
		shuffle    = shuffle,
		drop_last  = drop_last,
	}
	batcher_reset(&batcher)
	return batcher
}

batcher_destroy :: proc(b: ^Batcher) {
	delete(b.indices)
	b^ = {}
}

batcher_reset :: proc(b: ^Batcher) {
	b.cursor = 0
	if b.shuffle {
		rand.shuffle(b.indices)
	}
}

@(require_results)
batcher_next :: proc(b: ^Batcher) -> (batch: []int, ok: bool) {
	remaining := builtin.len(b.indices) - b.cursor
	if remaining <= 0 {
		return
	}
	take := min(b.batch_size, remaining)
	if take < b.batch_size && b.drop_last {
		b.cursor = builtin.len(b.indices)
		return
	}
	batch = b.indices[b.cursor:][:take]
	b.cursor += take
	return batch, true
}

gather :: proc(dst: []$T, src: []T, indices: []int, stride := 1, loc := #caller_location) {
	assert(stride > 0, "gather stride must be positive", loc=loc)
	assert(builtin.len(dst) == builtin.len(indices) * stride, "gather destination length must be len(indices) * stride", loc=loc)
	for index, i in indices {
		copy(dst[i * stride:][:stride], src[index * stride:][:stride])
	}
}

@(require_results)
split :: proc(sample_count: int, validation_fraction: f32, shuffle := true, allocator := context.allocator, loc := #caller_location) -> (train, validation: []int) {
	assert(sample_count > 0, "split requires at least one sample", loc=loc)
	assert(validation_fraction >= 0 && validation_fraction < 1, "validation_fraction must be in [0, 1)", loc=loc)

	permutation := make([]int, sample_count, allocator)
	defer delete(permutation, allocator)
	for i in 0 ..< sample_count {
		permutation[i] = i
	}
	if shuffle {
		rand.shuffle(permutation)
	}

	validation_count := int(f32(sample_count) * validation_fraction)
	train      = make([]int, sample_count - validation_count, allocator)
	validation = make([]int, validation_count, allocator)
	copy(train, permutation[:builtin.len(train)])
	copy(validation, permutation[builtin.len(train):])
	return
}
