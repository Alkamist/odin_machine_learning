package ml_tests

import "core:math/rand"
import "core:testing"

import "../sampling"

@(test)
test_argmax :: proc(t: ^testing.T) {
	tie := []f32{1, 3, 3, 2}
	testing.expectf(t, sampling.argmax(tie) == 1, "argmax should return first max on ties, got %v", sampling.argmax(tie))

	ordered := []f32{0.1, 0.2, 9.0, 0.3}
	testing.expectf(t, sampling.argmax(ordered) == 2, "argmax should return index of max, got %v", sampling.argmax(ordered))
}

@(test)
test_sample_greedy :: proc(t: ^testing.T) {
	logits := []f32{0.1, 5.0, 0.2, 9.0, 3.0}

	zero_temp := sampling.sample(logits, {temperature=0, top_k=40, top_p=0.95})
	testing.expectf(t, zero_temp == 3, "temperature=0 should return argmax, got %v", zero_temp)

	single_k := sampling.sample(logits, {temperature=0.8, top_k=1, top_p=0.95})
	testing.expectf(t, single_k == 3, "top_k=1 should return argmax, got %v", single_k)
}

@(test)
test_sample_deterministic_with_seeded_generator :: proc(t: ^testing.T) {
	logits := []f32{0.1, 5.0, 0.2, 9.0, 0.3, 7.0, 0.05, 1.0, 0.02, 3.0}

	run :: proc(logits: []f32, ids: []int) {
		state := rand.create(u64(1234))
		context.random_generator = rand.default_random_generator(&state)
		for &id in ids {
			id = sampling.sample(logits, {temperature=0.9, top_k=5, top_p=0.9})
		}
	}

	first:  [64]int
	second: [64]int
	run(logits, first[:])
	run(logits, second[:])
	for i in 0 ..< 64 {
		testing.expect_value(t, second[i], first[i])
	}
}

@(test)
test_sample_top_k :: proc(t: ^testing.T) {
	state := rand.create(u64(42))
	context.random_generator = rand.default_random_generator(&state)
	defer free_all(context.temp_allocator)

	logits := []f32{0.1, 5.0, 0.2, 9.0, 0.3, 7.0, 0.05, 1.0, 0.02, 3.0}
	top_three := []int{1, 3, 5}

	for _ in 0 ..< 200 {
		id := sampling.sample(logits, {temperature=1, top_k=3, top_p=0})
		in_top := false
		for candidate in top_three {
			if id == candidate {
				in_top = true
				break
			}
		}
		testing.expectf(t, in_top, "top_k=3 draw %v not among the 3 largest logits", id)
	}
}

@(test)
test_sample_top_p :: proc(t: ^testing.T) {
	state := rand.create(u64(7))
	context.random_generator = rand.default_random_generator(&state)
	defer free_all(context.temp_allocator)

	logits := make([]f32, 10)
	defer delete(logits)
	logits[4] = 10

	for _ in 0 ..< 200 {
		id := sampling.sample(logits, {temperature=1, top_k=0, top_p=0.5})
		testing.expectf(t, id == 4, "top_p=0.5 with a dominant logit should always draw index 4, got %v", id)
	}
}

Fake_Call :: struct {
	tokens:     [dynamic]int,
	logits_nil: bool,
}

Fake_Eval :: struct {
	script: []int,
	step:   int,
	calls:  [dynamic]Fake_Call,
}

_fake_eval :: proc(data: rawptr, tokens: []int, logits_out: []f32) {
	fake := (^Fake_Eval)(data)
	call := Fake_Call{logits_nil=logits_out == nil}
	append(&call.tokens, ..tokens)
	append(&fake.calls, call)
	if logits_out != nil {
		for i in 0 ..< len(logits_out) {
			logits_out[i] = 0
		}
		logits_out[fake.script[fake.step]] = 10
		fake.step += 1
	}
}

_record_on_token :: proc(data: rawptr, token: int) {
	emitted := (^[dynamic]int)(data)
	append(emitted, token)
}

@(test)
test_generate :: proc(t: ^testing.T) {
	fake := Fake_Eval{script={5, 6, 7, 99}}
	defer {
		for call in fake.calls {
			delete(call.tokens)
		}
		delete(fake.calls)
	}

	prompt := []int{10, 11, 12, 13, 14}
	logits := make([]f32, 100)
	defer delete(logits)

	emitted: [dynamic]int
	defer delete(emitted)

	out_tokens: [dynamic]int
	defer delete(out_tokens)

	options := sampling.Generate_Options{
		sampler        = {temperature=0},
		max_new_tokens = 10,
		prefill_chunk  = 2,
		stop_tokens    = {99},
		on_token       = _record_on_token,
		on_token_data  = &emitted,
	}
	stats := sampling.generate(_fake_eval, &fake, prompt, logits, options, &out_tokens)

	expected := []int{5, 6, 7}
	testing.expectf(t, len(out_tokens) == len(expected), "expected %v output tokens, got %v", len(expected), len(out_tokens))
	for token, i in expected {
		testing.expectf(t, out_tokens[i] == token, "out_tokens[%v] should be %v, got %v", i, token, out_tokens[i])
	}

	testing.expectf(t, len(emitted) == len(expected), "on_token should fire once per appended token, got %v", len(emitted))
	for token, i in expected {
		testing.expectf(t, emitted[i] == token, "emitted[%v] should be %v, got %v", i, token, emitted[i])
	}

	testing.expectf(t, stats.prefill_tokens == 5, "prefill_tokens should be 5, got %v", stats.prefill_tokens)
	testing.expectf(t, stats.decode_tokens == 4, "decode_tokens should count the stop draw (4), got %v", stats.decode_tokens)

	testing.expectf(t, len(fake.calls) == 6, "expected 6 eval calls, got %v", len(fake.calls))
	prefill_lengths := []int{2, 2, 1}
	for length, i in prefill_lengths {
		testing.expectf(t, len(fake.calls[i].tokens) == length, "prefill chunk %v length should be %v, got %v", i, length, len(fake.calls[i].tokens))
	}
	testing.expect(t, fake.calls[0].logits_nil, "prefill chunk 0 should get nil logits_out")
	testing.expect(t, fake.calls[1].logits_nil, "prefill chunk 1 should get nil logits_out")
	testing.expect(t, !fake.calls[2].logits_nil, "final prefill chunk should get logits_out")
}
