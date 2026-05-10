package gpt2_tokenizer_test

import "base:builtin"

import "core:fmt"
import "core:os"
import "core:slice"

import gpt2 "../../tokenizers/gpt2"

DATA_DIR        :: "smollm_data"
TOKENIZER_PATH  :: DATA_DIR + "/tokenizer.json"
TOKENS_PATH     :: DATA_DIR + "/prompt_tokens.bin"

PROMPT          :: "The capital of France is"

EXTRA_PROMPTS := [?]string{
	"Hello, world!",
	"   leading spaces and  internal  doubles ",
	"Numbers like 42, 3.14, and 1000000.",
	"Don't worry, it's fine — we'll see.",
	"Mix of tabs\tand\nnewlines.",
}

main :: proc() {
	tok, ok := gpt2.load(TOKENIZER_PATH)
	if !ok {
		fmt.eprintln("FAIL: could not load tokenizer.json")
		os.exit(1)
	}
	defer gpt2.destroy(tok)

	expected := load_token_ids(TOKENS_PATH) or_else _fatal("could not load expected token IDs")
	defer delete(expected)

	got := gpt2.encode(&tok, PROMPT)
	defer delete(got)

	fmt.printfln("Prompt   = %q", PROMPT)
	fmt.printfln("Expected = %v", expected)
	fmt.printfln("Got      = %v", got)

	if !slice.equal(got, expected) {
		fmt.eprintfln("FAIL: token IDs differ")
		os.exit(1)
	}

	decoded := gpt2.decode(&tok, got)
	defer delete(decoded)
	fmt.printfln("Decoded  = %q", decoded)
	if decoded != PROMPT {
		fmt.eprintfln("FAIL: decoded text %q != prompt %q", decoded, PROMPT)
		os.exit(1)
	}

	for prompt in EXTRA_PROMPTS {
		ids := gpt2.encode(&tok, prompt)
		defer delete(ids)
		round_tripped := gpt2.decode(&tok, ids)
		defer delete(round_tripped)
		if round_tripped != prompt {
			fmt.eprintfln("FAIL: round-trip mismatch on %q -> %v -> %q", prompt, ids, round_tripped)
			os.exit(1)
		}
		fmt.printfln("ok: %q -> %v tokens", prompt, builtin.len(ids))
	}

	fmt.println("PASS")
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
