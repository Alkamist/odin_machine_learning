package gemma_tokenizer_test

import "base:builtin"

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:slice"

import gemma "../../tokenizers/gemma"

DATA_DIR        :: "gemma_data"
TOKENIZER_PATH  :: DATA_DIR + "/tokenizer.json"
TOKENS_PATH     :: DATA_DIR + "/prompt_tokens.bin"
PARITY_PATH     :: DATA_DIR + "/tokenizer_parity.json"

PROMPT          :: "The capital of France is"

main :: proc() {
	tok, ok := gemma.load(TOKENIZER_PATH)
	if !ok {
		fmt.eprintln("FAIL: could not load tokenizer.json")
		os.exit(1)
	}
	defer gemma.destroy(tok)

	expected := load_token_ids(TOKENS_PATH) or_else _fatal("could not load expected token IDs")
	defer delete(expected)

	got := gemma.encode(&tok, PROMPT)
	defer delete(got)

	fmt.printfln("Prompt   = %q", PROMPT)
	fmt.printfln("Expected = %v", expected)
	fmt.printfln("Got      = %v", got)

	if !slice.equal(got, expected) {
		fmt.eprintfln("FAIL: token IDs differ")
		os.exit(1)
	}

	decoded := gemma.decode(&tok, got)
	defer delete(decoded)
	fmt.printfln("Decoded  = %q", decoded)
	if decoded != PROMPT {
		fmt.eprintfln("FAIL: decoded text %q != prompt %q", decoded, PROMPT)
		os.exit(1)
	}

	parity_bytes, parity_read_err := os.read_entire_file_from_path(PARITY_PATH, context.allocator)
	if parity_read_err != nil {
		fmt.eprintfln("FAIL: could not read %v: %v", PARITY_PATH, parity_read_err)
		os.exit(1)
	}
	defer delete(parity_bytes)

	parity_root, parity_parse_err := json.parse(parity_bytes, parse_integers = true)
	if parity_parse_err != .None {
		fmt.eprintfln("FAIL: JSON parse error in %v: %v", PARITY_PATH, parity_parse_err)
		os.exit(1)
	}
	defer json.destroy_value(parity_root)

	parity_object, parity_object_ok := parity_root.(json.Object)
	if !parity_object_ok {
		fmt.eprintln("FAIL: tokenizer_parity.json root is not an object")
		os.exit(1)
	}

	parity_count := 0
	for prompt, expected_value in parity_object {
		expected_array, expected_array_ok := expected_value.(json.Array)
		if !expected_array_ok {
			fmt.eprintfln("FAIL: parity entry %q is not an array", prompt)
			os.exit(1)
		}
		expected_ids := make([]int, len(expected_array))
		defer delete(expected_ids)
		for entry, idx in expected_array {
			id_int, id_int_ok := entry.(json.Integer)
			if !id_int_ok {
				fmt.eprintfln("FAIL: parity entry %q has non-integer id at %v", prompt, idx)
				os.exit(1)
			}
			expected_ids[idx] = int(id_int)
		}

		ids := gemma.encode(&tok, prompt)
		defer delete(ids)
		if !slice.equal(ids, expected_ids) {
			fmt.eprintfln("FAIL: HF parity mismatch on %q\n  expected = %v\n  got      = %v", prompt, expected_ids, ids)
			os.exit(1)
		}
		round_tripped := gemma.decode(&tok, ids)
		defer delete(round_tripped)
		if round_tripped != prompt {
			fmt.eprintfln("FAIL: round-trip mismatch on %q -> %v -> %q", prompt, ids, round_tripped)
			os.exit(1)
		}
		parity_count += 1
	}

	fmt.printfln("PASS (%v parity prompts)", parity_count)
}

_fatal :: proc(msg: string) -> []int {
	fmt.eprintln(msg)
	os.exit(1)
}

load_token_ids :: proc(path: string) -> ([]int, bool) {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		return nil, false
	}
	defer delete(bytes)

	count := int((^u32le)(raw_data(bytes))^)
	if 4 + count * 4 > builtin.len(bytes) {
		return nil, false
	}

	out := make([]int, count)
	for i in 0 ..< count {
		out[i] = int((^i32)(&bytes[4 + i * 4])^)
	}
	return out, true
}
