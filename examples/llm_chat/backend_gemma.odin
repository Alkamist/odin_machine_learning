package example_chat

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:log"
import "core:os"
import "core:time"

import ml        "../../"
import gemma     "../../networks/gemma"
import gemma_tok "../../tokenizers/gemma"
import           "../fetch"

GEMMA_DATA_DIR       :: #directory + "data/gemma"
GEMMA_GGUF_PATH      :: GEMMA_DATA_DIR + "/model.gguf"
GEMMA_TOKENIZER_PATH :: GEMMA_DATA_DIR + "/tokenizer.json"

GEMMA_ASSETS := []fetch.Asset{
	{
		url  = "https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a",
		dest = GEMMA_GGUF_PATH,
		size = 9_608_338_848,
	}, {
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
	stop_tokens:    [2]int,
}

@(require_results)
gemma_backend_make :: proc(gguf_path: string, t_max: int) -> (Chat_Model, bool) {
	weights_path := gguf_path
	if builtin.len(weights_path) == 0 {
		if !fetch.ensure_assets(GEMMA_ASSETS, "Gemma 4 E4B") {
			return {}, false
		}
		weights_path = GEMMA_GGUF_PATH
	} else if !os.exists(weights_path) {
		log.errorf("no such file: %v", weights_path)
		return {}, false
	}

	log.info("loading tokenizer")
	tokenizer, tokenizer_err := gemma_tok.load(GEMMA_TOKENIZER_PATH)
	if tokenizer_err != .None {
		log.errorf("could not load tokenizer %v: %v", GEMMA_TOKENIZER_PATH, tokenizer_err)
		return {}, false
	}

	log.infof("allocating Gemma 4 E4B (Q4_K/Q6_K GGUF, %v)", ML_BACKEND)
	config := gemma.make_e4b_config()
	model  := gemma.make(config, dtype = .Bf16)

	log.infof("loading weights from %v", weights_path)
	t_load := time.tick_now()
	load_ok: bool
	{
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
		load_ok = gemma.load_gguf(&model, weights_path)
	}
	if !load_ok {
		log.errorf("could not load weights from %v", weights_path)
		gemma.destroy(model)
		gemma.config_destroy(config)
		gemma_tok.destroy(tokenizer)
		return {}, false
	}
	log.infof("loaded in %.1f s", time.duration_seconds(time.tick_since(t_load)))

	backend := new(Gemma_Backend)
	backend.config         = config
	backend.model          = model
	backend.cache          = gemma.cache_make(model, t_max)
	backend.tokenizer      = tokenizer
	backend.end_of_turn_id = tokenizer.added_tokens[END_OF_TURN_TEXT] if END_OF_TURN_TEXT in tokenizer.added_tokens else -1
	backend.bos_id         = tokenizer.added_tokens[BOS_TEXT]         if BOS_TEXT         in tokenizer.added_tokens else -1
	backend.first_turn     = true
	backend.stop_tokens    = {GEMMA_EOS_ID, backend.end_of_turn_id}

	return Chat_Model{
		data          = backend,
		vocab_size    = config.vocab_size,
		prefill_chunk = GEMMA_PREFILL_CHUNK,
		stop_tokens   = backend.stop_tokens[:],
		eval          = _gemma_eval,
		encode_turn   = _gemma_encode_turn,
		decode        = _gemma_decode,
		remaining     = _gemma_remaining,
		reset         = _gemma_reset,
		destroy       = _gemma_destroy,
	}, true
}

_gemma_eval :: proc(data: rawptr, tokens: []int, logits_out: []f32) {
	backend := (^Gemma_Backend)(data)
	ml.pass()
	logits := gemma.forward_cached(backend.model, &backend.cache, tokens)
	if logits_out != nil {
		_copy_last_row(logits, logits_out)
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

_gemma_decode :: proc(data: rawptr, tokens: []int) -> string {
	backend := (^Gemma_Backend)(data)
	return gemma_tok.decode(&backend.tokenizer, tokens, context.temp_allocator)
}

_gemma_remaining :: proc(data: rawptr) -> int {
	backend := (^Gemma_Backend)(data)
	return ml.kv_cache_remaining(backend.cache)
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
