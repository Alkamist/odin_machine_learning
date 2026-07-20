package ml_golden_tests

import "core:log"
import "core:os"
import "core:testing"

import gpt2_tok "../../tokenizers/gpt2"

GPT2_TOKENIZER_PATH :: #directory + "../../examples/llm_chat/data/smollm/tokenizer.json"

GPT2_TOKENIZER_GOLDENS :: []Tokenizer_Golden{
	{text="Hello, world!",                ids=[]int{19556, 28, 905, 17}},
	{text="a 12",                         ids=[]int{81, 216, 33, 34}},
	{text="123 + 456 = 579",              ids=[]int{33, 34, 35, 1232, 216, 36, 37, 38, 446, 216, 37, 39, 41}},
	{text="prices rose 3.5% in 2024",     ids=[]int{96, 12855, 8739, 216, 35, 30, 37, 21, 281, 216, 34, 32, 34, 36}},
	{text="multi  space",                 ids=[]int{20437, 216, 1898}},
	{text="line\nbreak\ttab",             ids=[]int{1311, 198, 17653, 197, 15999}},
	{text="né 日本語のテキスト",            ids=[]int{94, 2756, 17097, 241, 115, 40993, 179, 120, 248, 26453, 11100, 224, 10391, 251, 10391, 134, 11100, 226}},
	{text="emoji 🙂 test",                 ids=[]int{391, 33777, 47526, 1028}},
	{text="it's we're I'll",              ids=[]int{269, 506, 392, 2316, 339, 3060}},
	{text="Hello<|endoftext|>world",      ids=[]int{19556, 0, 6693}},
}

_gpt2_load_or_skip :: proc(t: ^testing.T) -> (tok: gpt2_tok.Tokenizer, ok: bool) {
	if !os.exists(GPT2_TOKENIZER_PATH) {
		log.warnf("skipped: smollm tokenizer asset not present at %v (run examples/llm_chat once to fetch it)", GPT2_TOKENIZER_PATH)
		return
	}
	load_err: gpt2_tok.Error
	tok, load_err = gpt2_tok.load(GPT2_TOKENIZER_PATH)
	testing.expectf(t, load_err == .None, "gpt2 tokenizer should load, got %v", load_err)
	if load_err != .None {
		return
	}
	return tok, true
}

@(test)
test_gpt2_tokenizer_goldens :: proc(t: ^testing.T) {
	tok, loaded := _gpt2_load_or_skip(t)
	if !loaded {
		return
	}
	defer gpt2_tok.destroy(tok)

	for golden in GPT2_TOKENIZER_GOLDENS {
		ids, encode_ok := gpt2_tok.encode(&tok, golden.text, context.temp_allocator)
		testing.expectf(t, encode_ok, "encode of %q should succeed", golden.text)
		testing.expectf(t, len(ids) == len(golden.ids), "%q: expected %v ids, got %v (%v vs %v)", golden.text, len(golden.ids), len(ids), golden.ids, ids)
		if len(ids) == len(golden.ids) {
			for id, i in golden.ids {
				testing.expectf(t, ids[i] == id, "%q: id %v expected %v, got %v", golden.text, i, id, ids[i])
			}
		}

		decoded := gpt2_tok.decode(&tok, ids, context.temp_allocator)
		testing.expectf(t, decoded == golden.text, "%q: decode round-trip produced %q", golden.text, decoded)
	}
}

@(test)
test_gpt2_tokenizer_special_tokens :: proc(t: ^testing.T) {
	tok, loaded := _gpt2_load_or_skip(t)
	if !loaded {
		return
	}
	defer gpt2_tok.destroy(tok)

	testing.expect(t, len(tok.added_tokens) > 0, "tokenizer should declare added tokens")

	for content, id in tok.added_tokens {
		ids, encode_ok := gpt2_tok.encode(&tok, content, context.temp_allocator)
		testing.expectf(t, encode_ok, "encode of %q should succeed", content)
		testing.expectf(t, len(ids) == 1 && ids[0] == id, "%q should encode to [%v], got %v", content, id, ids)
	}
}
