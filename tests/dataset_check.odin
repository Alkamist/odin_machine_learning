package ml_tests

import "core:math/rand"
import "core:testing"

import "../dataset"

@(test)
test_batcher_sequential :: proc(t: ^testing.T) {
	b := dataset.batcher_make(10, 3, shuffle=false)
	defer dataset.batcher_destroy(&b)

	seen: [dynamic]int
	defer delete(seen)

	batch_count := 0
	for batch in dataset.batcher_next(&b) {
		testing.expectf(t, len(batch) == 3, "drop_last batch should have 3 elements, got %v", len(batch))
		append(&seen, ..batch)
		batch_count += 1
	}
	testing.expectf(t, batch_count == 3, "10 samples at batch 3 with drop_last should give 3 batches, got %v", batch_count)
	for value, i in seen {
		testing.expectf(t, value == i, "unshuffled batches should be sequential, index %v got %v", i, value)
	}

	partial := dataset.batcher_make(10, 3, shuffle=false, drop_last=false)
	defer dataset.batcher_destroy(&partial)

	clear(&seen)
	batch_count = 0
	last_length := 0
	for batch in dataset.batcher_next(&partial) {
		append(&seen, ..batch)
		batch_count += 1
		last_length = len(batch)
	}
	testing.expectf(t, batch_count == 4, "drop_last=false should give 4 batches, got %v", batch_count)
	testing.expectf(t, last_length == 1, "final partial batch should have 1 element, got %v", last_length)
	testing.expectf(t, len(seen) == 10, "drop_last=false should visit all 10 samples, got %v", len(seen))
}

@(test)
test_batcher_shuffle :: proc(t: ^testing.T) {
	state := rand.create(u64(42))
	context.random_generator = rand.default_random_generator(&state)

	b := dataset.batcher_make(100, 10)
	defer dataset.batcher_destroy(&b)

	first_epoch: [100]int
	cursor := 0
	visited: [100]bool
	for batch in dataset.batcher_next(&b) {
		for index in batch {
			testing.expectf(t, !visited[index], "index %v visited twice in one epoch", index)
			visited[index] = true
			first_epoch[cursor] = index
			cursor += 1
		}
	}
	testing.expectf(t, cursor == 100, "shuffled epoch should visit all 100 samples, got %v", cursor)

	dataset.batcher_reset(&b)
	cursor = 0
	differs := false
	revisited: [100]bool
	for batch in dataset.batcher_next(&b) {
		for index in batch {
			testing.expectf(t, !revisited[index], "index %v visited twice after reset", index)
			revisited[index] = true
			if first_epoch[cursor] != index {
				differs = true
			}
			cursor += 1
		}
	}
	testing.expectf(t, cursor == 100, "second epoch should visit all 100 samples, got %v", cursor)
	testing.expect(t, differs, "reset with shuffle should reorder the epoch")
}

@(test)
test_batcher_indices :: proc(t: ^testing.T) {
	source := []int{5, 7, 9, 11}
	b := dataset.batcher_make(source, 2, shuffle=false)
	defer dataset.batcher_destroy(&b)

	source[0] = 999

	expected := []int{5, 7, 9, 11}
	cursor := 0
	for batch in dataset.batcher_next(&b) {
		for index in batch {
			testing.expectf(t, index == expected[cursor], "batch index %v should be %v, got %v", cursor, expected[cursor], index)
			cursor += 1
		}
	}
	testing.expectf(t, cursor == 4, "expected 4 indices, got %v", cursor)
}

@(test)
test_gather :: proc(t: ^testing.T) {
	src := []f32{0, 1, 10, 11, 20, 21, 30, 31}
	dst: [4]f32
	dataset.gather(dst[:], src, []int{2, 0}, stride=2)

	expected := []f32{20, 21, 0, 1}
	for value, i in expected {
		testing.expectf(t, dst[i] == value, "gathered element %v should be %v, got %v", i, value, dst[i])
	}

	targets := []int{100, 200, 300}
	picked: [2]int
	dataset.gather(picked[:], targets, []int{1, 2})
	testing.expect(t, picked[0] == 200 && picked[1] == 300, "stride-1 gather should pick target rows")
}

@(test)
test_split :: proc(t: ^testing.T) {
	state := rand.create(u64(7))
	context.random_generator = rand.default_random_generator(&state)

	train, validation := dataset.split(10, 0.3)
	defer delete(train)
	defer delete(validation)

	testing.expectf(t, len(train) == 7, "split(10, 0.3) train should have 7, got %v", len(train))
	testing.expectf(t, len(validation) == 3, "split(10, 0.3) validation should have 3, got %v", len(validation))

	seen: [10]bool
	for index in train {
		testing.expectf(t, !seen[index], "index %v appears twice in split", index)
		seen[index] = true
	}
	for index in validation {
		testing.expectf(t, !seen[index], "index %v appears in both train and validation", index)
		seen[index] = true
	}
	for was_seen, i in seen {
		testing.expectf(t, was_seen, "index %v missing from split", i)
	}

	all_train, empty_validation := dataset.split(5, 0, shuffle=false)
	defer delete(all_train)
	defer delete(empty_validation)

	testing.expectf(t, len(empty_validation) == 0, "fraction 0 should give empty validation, got %v", len(empty_validation))
	for value, i in all_train {
		testing.expectf(t, value == i, "unshuffled split train should be sequential, index %v got %v", i, value)
	}
}
