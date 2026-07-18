package example_chat

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:os"
import "core:time"

import ml        "../../"
import gemma     "../../networks/gemma"
import gemma_tok "../../tokenizers/gemma"

GEMMA_DATA_DIR       :: #directory + "data/gemma"
GEMMA_GGUF_PATH      :: GEMMA_DATA_DIR + "/model.gguf"
GEMMA_TOKENIZER_PATH :: GEMMA_DATA_DIR + "/tokenizer.json"

// Weights come from Ollama's registry (library/gemma4:e4b) rather than
// HuggingFace: the GGUF there is a Q4_K/Q6_K mix, which is what load_gguf
// dequantizes. The GGUFs published under ggml-org and google are Q4_0/Q8_0,
// which load_gguf rejects. The blob is addressed by its own sha256, so this
// URL is immutable.
GEMMA_ASSETS := []Asset {
	{
		url  = "https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a",
		dest = GEMMA_GGUF_PATH,
		size = 9_608_338_848,
	},
	{
		url  = "https://huggingface.co/google/gemma-4-e4b-it/resolve/main/tokenizer.json",
		dest = GEMMA_TOKENIZER_PATH,
		size = 32_169_626,
	},
}

GEMMA_EOS_ID     :: 1
END_OF_TURN_TEXT :: "<turn|>"
BOS_TEXT         :: "<bos>"

GEMMA_PREFILL_CHUNK :: 64

Gemma_Backend :: struct {
	config:         gemma.Config,
	model:          gemma.Gemma,
	cache:          gemma.Cache,
	tokenizer:      gemma_tok.Tokenizer,
	end_of_turn_id: int,
	bos_id:         int,
	first_turn:     bool,
}

@(require_results)
gemma_backend_make :: proc(gguf_path: string, t_max: int) -> (Chat_Model, bool) {
	// An explicit --gguf points at a file the user manages themselves, so only
	// fetch when we are using the default location.
	weights_path := gguf_path
	if builtin.len(weights_path) == 0 {
		if !ensure_assets(GEMMA_ASSETS, "Gemma 4 E4B") {
			return {}, false
		}
		weights_path = GEMMA_GGUF_PATH
	} else if !os.exists(weights_path) {
		fmt.eprintfln("FAIL: no such file: %v", weights_path)
		return {}, false
	}

	fmt.println("Loading tokenizer ...")
	tokenizer, tokenizer_ok := gemma_tok.load(GEMMA_TOKENIZER_PATH)
	if !tokenizer_ok {
		fmt.eprintfln("FAIL: could not load tokenizer: %v", GEMMA_TOKENIZER_PATH)
		return {}, false
	}

	fmt.printfln("Allocating Gemma 4 E4B (Q4_K/Q6_K GGUF, %v) ...", ML_BACKEND)
	config := gemma.make_e4b_config()
	model  := gemma.make(config, dtype = .Bf16)

	fmt.printfln("Loading weights from %v ...", weights_path)
	t_load := time.tick_now()
	load_ok: bool
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		load_ok = gemma.load_gguf(&model, weights_path)
	}
	if !load_ok {
		fmt.eprintfln("FAIL: could not load weights from %v", weights_path)
		gemma.destroy(model)
		gemma.config_destroy(config)
		gemma_tok.destroy(tokenizer)
		return {}, false
	}
	fmt.printfln("  loaded in %.1f s", time.duration_seconds(time.tick_since(t_load)))

	backend := new(Gemma_Backend)
	backend.config         = config
	backend.model          = model
	backend.cache          = gemma.cache_make(model, t_max)
	backend.tokenizer      = tokenizer
	backend.end_of_turn_id = tokenizer.added_tokens[END_OF_TURN_TEXT] if END_OF_TURN_TEXT in tokenizer.added_tokens else -1
	backend.bos_id         = tokenizer.added_tokens[BOS_TEXT]         if BOS_TEXT         in tokenizer.added_tokens else -1
	backend.first_turn     = true

	return Chat_Model{
		data        = backend,
		vocab_size  = config.vocab_size,
		eval        = _gemma_eval,
		encode_turn = _gemma_encode_turn,
		is_stop     = _gemma_is_stop,
		decode      = _gemma_decode,
		reset       = _gemma_reset,
		destroy     = _gemma_destroy,
	}, true
}

_gemma_eval :: proc(data: rawptr, tokens: []int, logits_out: []f32) {
	backend := (^Gemma_Backend)(data)
	n := builtin.len(tokens)
	pos := 0
	for pos < n {
		ml.clear()
		take := GEMMA_PREFILL_CHUNK
		if pos + take > n {
			take = 1
		}
		chunk := tokens[pos : pos + take]
		logits := gemma.forward_cached(backend.model, &backend.cache, chunk)
		if pos + take == n {
			_copy_last_row(logits, logits_out)
		}
		pos += take
	}
}

_gemma_encode_turn :: proc(data: rawptr, user_text: string) -> []int {
	backend := (^Gemma_Backend)(data)

	out: [dynamic]int
	out.allocator = context.temp_allocator

	if backend.first_turn && backend.bos_id >= 0 {
		append(&out, backend.bos_id)
		backend.first_turn = false
	}
	turn_text := fmt.tprintf("<|turn>user\n%v<turn|>\n<|turn>model\n", user_text)
	append(&out, ..gemma_tok.encode(&backend.tokenizer, turn_text, context.temp_allocator))

	return out[:]
}

_gemma_is_stop :: proc(data: rawptr, token: int) -> bool {
	backend := (^Gemma_Backend)(data)
	return token == GEMMA_EOS_ID || token == backend.end_of_turn_id
}

_gemma_decode :: proc(data: rawptr, tokens: []int) -> string {
	backend := (^Gemma_Backend)(data)
	return gemma_tok.decode(&backend.tokenizer, tokens, context.temp_allocator)
}

_gemma_reset :: proc(data: rawptr) {
	backend := (^Gemma_Backend)(data)
	gemma.cache_reset(&backend.cache)
	backend.first_turn = true
}

_gemma_destroy :: proc(data: rawptr) {
	backend := (^Gemma_Backend)(data)
	gemma.cache_destroy(backend.cache)
	gemma.destroy(backend.model)
	gemma.config_destroy(backend.config)
	gemma_tok.destroy(backend.tokenizer)
	free(backend)
}
