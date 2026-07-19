package ml_golden_tests

import "core:log"
import "core:os"
import "core:testing"

import ml        "../../"
import cpu       "../../backends/cpu"
import           "../../networks/mlp"
import gemma_tok "../../tokenizers/gemma"

GEMMA_TOKENIZER_PATH :: #directory + "../../examples/llm_chat/data/gemma/tokenizer.json"

Tokenizer_Golden :: struct {
	text: string,
	ids:  []int,
}

GEMMA_TOKENIZER_GOLDENS :: []Tokenizer_Golden{
	{text="Hello, world!", ids=[]int{9259, 236764, 1902, 236888}},
	{text="The quick brown fox jumps over the lazy dog.", ids=[]int{818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761}},
	{text=" leading space", ids=[]int{5830, 2557}},
	{text="trailing space ", ids=[]int{136697, 2557, 236743}},
	{text="multi  space", ids=[]int{20028, 138, 5780}},
	{text="a", ids=[]int{236746}},
	{text="123 + 456 = 579", ids=[]int{236770, 236778, 236800, 900, 236743, 236812, 236810, 236825, 578, 236743, 236810, 236832, 236819}},
	{text="CamelCaseIdentifier_snake_case", ids=[]int{114919, 9818, 19535, 236779, 37942, 236779, 4925}},
	{text="né 日本語のテキスト", ids=[]int{8504, 33375, 238582, 236945, 95830}},
	{text="emoji 🙂 test", ids=[]int{67906, 57235, 1594}},
	{text="line\nbreak\ttab", ids=[]int{1257, 107, 7284, 255968, 4823}},
}

@(test)
test_gemma_tokenizer_goldens :: proc(t: ^testing.T) {
	if !os.exists(GEMMA_TOKENIZER_PATH) {
		log.warnf("SKIPPED: gemma tokenizer asset not present at %v (run examples/llm_chat once to fetch it)", GEMMA_TOKENIZER_PATH)
		return
	}

	tok, ok := gemma_tok.load(GEMMA_TOKENIZER_PATH)
	testing.expect(t, ok, "gemma tokenizer should load")
	if !ok {
		return
	}
	defer gemma_tok.destroy(tok)

	for golden in GEMMA_TOKENIZER_GOLDENS {
		ids, encode_ok := gemma_tok.encode(&tok, golden.text, context.temp_allocator)
		testing.expectf(t, encode_ok, "encode of %q should succeed", golden.text)
		testing.expectf(t, len(ids) == len(golden.ids), "%q: expected %v ids, got %v (%v vs %v)", golden.text, len(golden.ids), len(ids), golden.ids, ids)
		if len(ids) == len(golden.ids) {
			for id, i in golden.ids {
				testing.expectf(t, ids[i] == id, "%q: id %v expected %v, got %v", golden.text, i, id, ids[i])
			}
		}

		decoded := gemma_tok.decode(&tok, ids, context.temp_allocator)
		testing.expectf(t, decoded == golden.text, "%q: decode round-trip produced %q", golden.text, decoded)
	}
}

@(test)
test_q4_k_dequant_golden :: proc(t: ^testing.T) {
	block: [ml.Q4_K_BLOCK_BYTES]u8

	block[0] = 0x00; block[1] = 0x40
	block[2] = 0x00; block[3] = 0x38

	for j in 0 ..< 4 {
		block[4 + j]  = 1
		block[8 + j]  = 1
		block[12 + j] = 1
	}

	for k in 0 ..< 128 {
		block[16 + k] = u8(k * 7 % 256)
	}

	output: [ml.K_QUANT_BLOCK_SIZE]f32
	ml.dequantize_q4_k(block[:], output[:])

	d    := f32(2.0)
	dmin := f32(0.5)
	for chunk in 0 ..< 4 {
		for i in 0 ..< 32 {
			q_byte := u8(((chunk * 32 + i) * 7) % 256)
			low  := f32(q_byte & 0xF)
			high := f32(q_byte >> 4)
			m := f32(1) if chunk < 2 else f32(0)
			expected_low  := d * 1 * low  - dmin * m
			expected_high := d * 1 * high - dmin * m
			testing.expectf(t, output[chunk * 64 + i] == expected_low, "q4_k elem %v expected %v, got %v", chunk * 64 + i, expected_low, output[chunk * 64 + i])
			testing.expectf(t, output[chunk * 64 + 32 + i] == expected_high, "q4_k elem %v expected %v, got %v", chunk * 64 + 32 + i, expected_high, output[chunk * 64 + 32 + i])
		}
	}
}

@(test)
test_q6_k_dequant_golden :: proc(t: ^testing.T) {
	block: [ml.Q6_K_BLOCK_BYTES]u8

	for k in 0 ..< 128 {
		block[k] = u8(k * 11 % 256)
	}
	for k in 0 ..< 64 {
		block[128 + k] = 0
	}
	for k in 0 ..< 16 {
		block[192 + k] = 1
	}
	block[208] = 0x00; block[209] = 0x3C

	output: [ml.K_QUANT_BLOCK_SIZE]f32
	ml.dequantize_q6_k(block[:], output[:])

	for half in 0 ..< 2 {
		ql_base := half * 64
		out_base := half * 128
		for i in 0 ..< 32 {
			ql_lo  := u8((ql_base + i) * 11 % 256)
			ql_hi  := u8((ql_base + 32 + i) * 11 % 256)
			expect := [4]f32{
				f32(int(ql_lo & 0xF)) - 32,
				f32(int(ql_hi & 0xF)) - 32,
				f32(int(ql_lo >> 4)) - 32,
				f32(int(ql_hi >> 4)) - 32,
			}
			offsets := [4]int{0, 32, 64, 96}
			for e, slot in expect {
				index := out_base + offsets[slot] + i
				testing.expectf(t, output[index] == e, "q6_k elem %v expected %v, got %v", index, e, output[index])
			}
		}
	}
}

@(test)
test_fixed_mlp_forward :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	model := mlp.make(2, 2, 1)
	defer mlp.destroy(model)

	ml.set_data(model.layers[0].weight, []f32{1, 2, 3, 4})
	ml.set_data(model.layers[0].bias,   []f32{0.5, -1})
	ml.set_data(model.layers[1].weight, []f32{1, -1})
	ml.set_data(model.layers[1].bias,   []f32{0.25})

	ml.clear()
	x      := ml.tensor([]f32{1, 1}, []int{1, 2})
	output := mlp.forward(model, x)

	result: [1]f32
	ml.get_data(output, result[:])
	testing.expectf(t, result[0] == -2.25, "fixed mlp forward expected -2.25, got %v", result[0])
}
