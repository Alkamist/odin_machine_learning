package sampling

import "base:builtin"

import "core:math"
import "core:math/rand"
import "core:time"

Sampler :: struct {
	temperature: f32,
	top_k:       int,
	top_p:       f32,
}

argmax :: proc(logits: []f32, loc := #caller_location) -> int {
	assert(builtin.len(logits) > 0, "argmax requires a non-empty logits row", loc)
	best := 0
	for i in 1 ..< builtin.len(logits) {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

sample :: proc(logits: []f32, sampler: Sampler, loc := #caller_location) -> int {
	n := builtin.len(logits)
	assert(n > 0, "sample requires a non-empty logits row", loc)
	if sampler.temperature <= 0 || sampler.top_k == 1 {
		return argmax(logits, loc=loc)
	}

	candidate_count := sampler.top_k > 0 ? min(sampler.top_k, n) : n
	indices := make([]int, candidate_count, context.temp_allocator)

	for i in 0 ..< candidate_count {
		indices[i] = i
	}
	for i := candidate_count / 2 - 1; i >= 0; i -= 1 {
		_sift_down_min_logit(indices, logits, i, candidate_count)
	}
	for i in candidate_count ..< n {
		if logits[i] > logits[indices[0]] {
			indices[0] = i
			_sift_down_min_logit(indices, logits, 0, candidate_count)
		}
	}
	for end := candidate_count - 1; end > 0; end -= 1 {
		indices[0], indices[end] = indices[end], indices[0]
		_sift_down_min_logit(indices, logits, 0, end)
	}

	max_logit := logits[indices[0]]
	probabilities := make([]f32, candidate_count, context.temp_allocator)
	sum: f32
	for slot in 0 ..< candidate_count {
		probabilities[slot] = math.exp_f32((logits[indices[slot]] - max_logit) / sampler.temperature)
		sum += probabilities[slot]
	}
	for slot in 0 ..< candidate_count {
		probabilities[slot] /= sum
	}

	keep := candidate_count
	if sampler.top_p > 0 && sampler.top_p < 1 {
		cumulative: f32
		for slot in 0 ..< candidate_count {
			cumulative += probabilities[slot]
			if cumulative >= sampler.top_p {
				keep = slot + 1
				break
			}
		}
		new_sum: f32
		for slot in 0 ..< keep {
			new_sum += probabilities[slot]
		}
		if new_sum > 0 {
			for slot in 0 ..< keep {
				probabilities[slot] /= new_sum
			}
		}
	}

	r := rand.float32()
	cumulative: f32
	for slot in 0 ..< keep {
		cumulative += probabilities[slot]
		if r <= cumulative {
			return indices[slot]
		}
	}
	return indices[keep - 1]
}

_sift_down_min_logit :: proc(indices: []int, logits: []f32, start, n: int) {
	root := start
	for {
		child := 2 * root + 1
		if child >= n {
			return
		}
		if child + 1 < n && logits[indices[child + 1]] < logits[indices[child]] {
			child += 1
		}
		if logits[indices[root]] <= logits[indices[child]] {
			return
		}
		indices[root], indices[child] = indices[child], indices[root]
		root = child
	}
}

Eval_Proc     :: proc(data: rawptr, tokens: []int, logits_out: []f32)
On_Token_Proc :: proc(data: rawptr, token: int)

Generate_Options :: struct {
	sampler:        Sampler,
	max_new_tokens: int,
	prefill_chunk:  int,
	stop_tokens:    []int,
	on_token:       On_Token_Proc,
	on_token_data:  rawptr,
}

Generate_Stats :: struct {
	prefill_tokens:  int,
	prefill_seconds: f64,
	decode_tokens:   int,
	decode_seconds:  f64,
}

generate :: proc(eval: Eval_Proc, eval_data: rawptr, prompt: []int, logits: []f32, options: Generate_Options, out_tokens: ^[dynamic]int, loc := #caller_location) -> Generate_Stats {
	assert(eval != nil, "generate requires an eval proc", loc)
	assert(builtin.len(prompt) > 0, "generate requires a non-empty prompt", loc)
	assert(builtin.len(logits) > 0, "generate requires a non-empty logits row", loc)
	stats := Generate_Stats{prefill_tokens=builtin.len(prompt)}

	t_prefill := time.tick_now()
	if options.prefill_chunk > 0 {
		n := builtin.len(prompt)
		pos := 0
		for pos < n {
			take := min(options.prefill_chunk, n - pos)
			chunk := prompt[pos : pos + take]
			logits_out := logits if pos + take == n else nil
			eval(eval_data, chunk, logits_out)
			pos += take
		}
	} else {
		eval(eval_data, prompt, logits)
	}
	stats.prefill_seconds = time.duration_seconds(time.tick_since(t_prefill))

	t_decode := time.tick_now()
	for step in 0 ..< options.max_new_tokens {
		next_id := sample(logits, options.sampler)
		stats.decode_tokens += 1

		is_stop := false
		for stop in options.stop_tokens {
			if next_id == stop {
				is_stop = true
				break
			}
		}
		if is_stop {
			break
		}

		append(out_tokens, next_id)
		if options.on_token != nil {
			options.on_token(options.on_token_data, next_id)
		}

		if step == options.max_new_tokens - 1 {
			break
		}

		single := [1]int{next_id}
		eval(eval_data, single[:], logits)
	}
	stats.decode_seconds = time.duration_seconds(time.tick_since(t_decode))

	return stats
}
