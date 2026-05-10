// Tokenize a JSONL chat dataset (system / user / assistant turns) into the
// per-sample binary format the SFT trainer consumes.
//
// Each input row is one chat sample of the form:
//   {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
//
// Output layout (little-endian):
//   bytes 0..7  magic "RSCHAT01"
//   bytes 8..11 sample_count (i32)
//   per sample:
//     i32 total_len
//     i32 asst_start  (index into tokens where loss begins)
//     i32 asst_end    (one past the last loss token)
//     i32 * total_len token ids
//
// Loss covers the assistant content + the closing turn token. The role
// header (e.g. `<|im_start|>assistant\n`) is part of the prompt and bears
// no loss.
//
//   odin run tools/tokenize_chat -o:speed -- \
//       --input  reascript/dataset/dataset.jsonl \
//       --output reascript/dataset/reascript_gemma.bin \
//       --tokenizer gemma_data/tokenizer.json \
//       --type gemma

package main

import "base:builtin"

import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"
import "core:time"

import gemma_tok "../../tokenizers/gemma"
import gpt2_tok  "../../tokenizers/gpt2"

Tokenizer_Kind :: enum {
	Gemma,
	SmolLM,
}

GEMMA_BOS    :: "<bos>"
GEMMA_TURN_O :: "<|turn>"
GEMMA_TURN_C :: "<turn|>"

SMOLLM_IM_START :: "<|im_start|>"
SMOLLM_IM_END   :: "<|im_end|>"

main :: proc() {
	input_path:     string
	output_path:    string
	tokenizer_path: string
	kind := Tokenizer_Kind.Gemma
	limit := -1

	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--input":     i += 1; input_path     = args[i]; i += 1
		case "--output":    i += 1; output_path    = args[i]; i += 1
		case "--tokenizer": i += 1; tokenizer_path = args[i]; i += 1
		case "--type":
			i += 1
			switch args[i] {
			case "gemma":  kind = .Gemma
			case "smollm": kind = .SmolLM
			case:
				fmt.eprintfln("unknown tokenizer type: %v (expected gemma or smollm)", args[i])
				os.exit(1)
			}
			i += 1
		case "--limit":
			i += 1
			limit = _parse_int(args[i])
			i += 1
		case "--help", "-h":
			_usage()
		case:
			fmt.eprintfln("unknown argument: %v", arg)
			_usage()
		}
	}

	if input_path == "" || output_path == "" || tokenizer_path == "" {
		_usage()
	}

	fmt.printfln("Reading %v ...", input_path)
	corpus_bytes, read_err := os.read_entire_file_from_path(input_path, context.allocator)
	if read_err != nil {
		fmt.eprintfln("FAIL: could not read %v: %v", input_path, read_err)
		os.exit(1)
	}
	defer delete(corpus_bytes)

	lines: [dynamic]string
	defer delete(lines)
	{
		text := string(corpus_bytes)
		start := 0
		for j in 0 ..< builtin.len(text) {
			if text[j] == '\n' {
				line := text[start:j]
				if builtin.len(line) > 0 && line[builtin.len(line) - 1] == '\r' {
					line = line[:builtin.len(line) - 1]
				}
				if builtin.len(line) > 0 {
					append(&lines, line)
				}
				start = j + 1
			}
		}
		if start < builtin.len(text) {
			line := text[start:]
			if builtin.len(line) > 0 && line[builtin.len(line) - 1] == '\r' {
				line = line[:builtin.len(line) - 1]
			}
			if builtin.len(line) > 0 {
				append(&lines, line)
			}
		}
	}
	fmt.printfln("  %v rows", builtin.len(lines))

	if limit > 0 && builtin.len(lines) > limit {
		resize(&lines, limit)
	}

	out_bytes: [dynamic]byte
	defer delete(out_bytes)

	append(&out_bytes, ..transmute([]byte)string("RSCHAT01"))
	count_placeholder := builtin.len(out_bytes)
	append(&out_bytes, 0, 0, 0, 0)

	t_start := time.tick_now()
	written: i32
	total_tokens: int
	total_loss_tokens: int

	switch kind {
	case .Gemma:
		fmt.printfln("Loading Gemma tokenizer from %v ...", tokenizer_path)
		t, ok := gemma_tok.load(tokenizer_path)
		if !ok {
			fmt.eprintln("FAIL: could not load Gemma tokenizer")
			os.exit(1)
		}
		defer gemma_tok.destroy(t)

		bos_id := _added_id(t.added_tokens, GEMMA_BOS)
		assert(bos_id >= 0, "Gemma tokenizer missing <bos>")

		for line, idx in lines {
			sample, sample_ok := _parse_chat_row(line)
			if !sample_ok {
				fmt.eprintfln("row %v: parse failed, skipping", idx)
				continue
			}
			tokens, asst_start, asst_end := _encode_gemma(&t, sample, bos_id)
			defer delete(tokens)
			_emit_sample(&out_bytes, tokens, asst_start, asst_end)
			written += 1
			total_tokens += builtin.len(tokens)
			total_loss_tokens += asst_end - asst_start

			if idx % 64 == 0 {
				_progress(idx, builtin.len(lines), t_start)
			}
		}
	case .SmolLM:
		fmt.printfln("Loading SmolLM (GPT-2) tokenizer from %v ...", tokenizer_path)
		t, ok := gpt2_tok.load(tokenizer_path)
		if !ok {
			fmt.eprintln("FAIL: could not load SmolLM tokenizer")
			os.exit(1)
		}
		defer gpt2_tok.destroy(t)

		for line, idx in lines {
			sample, sample_ok := _parse_chat_row(line)
			if !sample_ok {
				fmt.eprintfln("row %v: parse failed, skipping", idx)
				continue
			}
			tokens, asst_start, asst_end := _encode_smollm(&t, sample)
			defer delete(tokens)
			_emit_sample(&out_bytes, tokens, asst_start, asst_end)
			written += 1
			total_tokens += builtin.len(tokens)
			total_loss_tokens += asst_end - asst_start

			if idx % 64 == 0 {
				_progress(idx, builtin.len(lines), t_start)
			}
		}
	}
	_progress(builtin.len(lines), builtin.len(lines), t_start)
	fmt.println()

	count_bytes := mem.slice_to_bytes(([]i32{written})[:])
	copy(out_bytes[count_placeholder:count_placeholder + 4], count_bytes)

	elapsed := f64(time.duration_seconds(time.tick_since(t_start)))
	fmt.printfln("  encoded %v rows / %v tokens (%v loss) in %.1f s",
		written, total_tokens, total_loss_tokens, elapsed)

	fmt.printfln("Writing %v ...", output_path)
	if err := os.write_entire_file(output_path, out_bytes[:]); err != nil {
		fmt.eprintfln("FAIL: could not write %v: %v", output_path, err)
		os.exit(1)
	}
	fmt.printfln("  wrote %v bytes", builtin.len(out_bytes))
	fmt.println("Done.")
}

Chat_Sample :: struct {
	system_text:    string,
	user_text:      string,
	assistant_text: string,
}

_parse_chat_row :: proc(line: string) -> (sample: Chat_Sample, ok: bool) {
	root, parse_err := json.parse(transmute([]byte)line, parse_integers = true)
	if parse_err != .None {
		return {}, false
	}
	defer json.destroy_value(root)

	root_obj, root_ok := root.(json.Object)
	if !root_ok {
		return {}, false
	}

	messages, messages_ok := root_obj["messages"].(json.Array)
	if !messages_ok {
		return {}, false
	}

	for entry in messages {
		entry_obj, eo_ok := entry.(json.Object)
		if !eo_ok {
			continue
		}
		role,    rs_ok := entry_obj["role"].(string)
		content, cs_ok := entry_obj["content"].(string)
		if !rs_ok || !cs_ok {
			continue
		}
		switch role {
		case "system":    sample.system_text    = strings.clone(content)
		case "user":      sample.user_text      = strings.clone(content)
		case "assistant": sample.assistant_text = strings.clone(content)
		}
	}

	if sample.user_text == "" || sample.assistant_text == "" {
		delete(sample.system_text)
		delete(sample.user_text)
		delete(sample.assistant_text)
		return {}, false
	}
	return sample, true
}

_destroy_sample :: proc(s: Chat_Sample) {
	delete(s.system_text)
	delete(s.user_text)
	delete(s.assistant_text)
}

// Gemma 4 chat template. The official template doesn't carry a system role,
// so the system content is folded into the first user turn.
//
//   <bos><|turn>user
//   {system}
//
//   {user}<turn|>
//   <|turn>model
//   {assistant}<turn|>
//   <eos>
//
// Loss covers `{assistant}<turn|>` (and the trailing newline / eos).
_encode_gemma :: proc(t: ^gemma_tok.Tokenizer, sample: Chat_Sample, bos_id: int) -> (tokens: []int, asst_start, asst_end: int) {
	defer _destroy_sample(sample)

	user_combined: string
	if sample.system_text != "" {
		user_combined = fmt.aprintf("%v\n\n%v", sample.system_text, sample.user_text)
	} else {
		user_combined = strings.clone(sample.user_text)
	}
	defer delete(user_combined)

	prompt_text := fmt.aprintf("%vuser\n%v%v\n%vmodel\n", GEMMA_TURN_O, user_combined, GEMMA_TURN_C, GEMMA_TURN_O)
	defer delete(prompt_text)

	asst_text := fmt.aprintf("%v%v\n", sample.assistant_text, GEMMA_TURN_C)
	defer delete(asst_text)

	prompt_ids := gemma_tok.encode(t, prompt_text, context.temp_allocator)
	asst_ids   := gemma_tok.encode(t, asst_text,   context.temp_allocator)

	out: [dynamic]int
	append(&out, bos_id)
	append(&out, ..prompt_ids)
	asst_start = builtin.len(out)
	append(&out, ..asst_ids)
	asst_end = builtin.len(out)

	tokens = out[:]
	return
}

// SmolLM2 / ChatML template:
//
//   <|im_start|>system
//   {system}<|im_end|>
//   <|im_start|>user
//   {user}<|im_end|>
//   <|im_start|>assistant
//   {assistant}<|im_end|>
//
// Loss covers `{assistant}<|im_end|>` plus the trailing newline.
_encode_smollm :: proc(t: ^gpt2_tok.Tokenizer, sample: Chat_Sample) -> (tokens: []int, asst_start, asst_end: int) {
	defer _destroy_sample(sample)

	im_start_id, im_start_ok := t.added_tokens[SMOLLM_IM_START]
	im_end_id,   im_end_ok   := t.added_tokens[SMOLLM_IM_END]
	assert(im_start_ok && im_end_ok, "SmolLM tokenizer missing <|im_start|> or <|im_end|>")

	out: [dynamic]int

	_emit_turn :: proc(t: ^gpt2_tok.Tokenizer, out: ^[dynamic]int, role, content: string, im_start_id, im_end_id: int) {
		append(out, im_start_id)
		header := fmt.tprintf("%v\n%v", role, content)
		header_ids := gpt2_tok.encode(t, header, context.temp_allocator)
		append(out, ..header_ids)
		append(out, im_end_id)
		newline_ids := gpt2_tok.encode(t, "\n", context.temp_allocator)
		append(out, ..newline_ids)
	}

	if sample.system_text != "" {
		_emit_turn(t, &out, "system", sample.system_text, im_start_id, im_end_id)
	}
	_emit_turn(t, &out, "user", sample.user_text, im_start_id, im_end_id)

	// Assistant header is part of the prompt; loss starts at the content.
	append(&out, im_start_id)
	header_ids := gpt2_tok.encode(t, "assistant\n", context.temp_allocator)
	append(&out, ..header_ids)

	asst_start = builtin.len(out)

	asst_ids := gpt2_tok.encode(t, sample.assistant_text, context.temp_allocator)
	append(&out, ..asst_ids)
	append(&out, im_end_id)
	newline_ids := gpt2_tok.encode(t, "\n", context.temp_allocator)
	append(&out, ..newline_ids)

	asst_end = builtin.len(out)

	tokens = out[:]
	return
}

_emit_sample :: proc(buf: ^[dynamic]byte, tokens: []int, asst_start, asst_end: int) {
	header := [3]i32{i32(builtin.len(tokens)), i32(asst_start), i32(asst_end)}
	append(buf, ..mem.slice_to_bytes(header[:]))

	body := builtin.make([]i32, builtin.len(tokens), context.temp_allocator)
	for v, idx in tokens {
		body[idx] = i32(v)
	}
	append(buf, ..mem.slice_to_bytes(body))
}

_progress :: proc(done, total: int, t_start: time.Tick) {
	elapsed := f64(time.duration_seconds(time.tick_since(t_start)))
	rate    := f64(done) / elapsed if elapsed > 0 else 0
	pct     := 100.0 * f64(done) / f64(total) if total > 0 else 0
	fmt.printf("\r  %.0f%% (%v / %v rows, %.1f s, %.0f rows/s)        ", pct, done, total, elapsed, rate)
	os.flush(os.stdout)
}

_added_id :: proc(added: map[string]int, key: string) -> int {
	if v, ok := added[key]; ok {
		return v
	}
	return -1
}

_usage :: proc() {
	fmt.eprintln("usage: tokenize_chat --input JSONL_PATH --output BIN_PATH --tokenizer TOKENIZER_JSON --type gemma|smollm [--limit N]")
	os.exit(1)
}

_parse_int :: proc(s: string) -> int {
	v: int
	negative := false
	cursor := 0
	if builtin.len(s) > 0 && s[0] == '-' {
		negative = true
		cursor = 1
	}
	for cursor < builtin.len(s) {
		c := s[cursor]
		if c < '0' || c > '9' {
			fmt.eprintfln("invalid integer: %q", s)
			os.exit(1)
		}
		v = v * 10 + int(c - '0')
		cursor += 1
	}
	return -v if negative else v
}
