package example_chat

import "base:builtin"

import "core:fmt"
import "core:log"
import "core:os"

import ml    "../../"
import llama "../../networks/llama"
import gpt2  "../../tokenizers/gpt2"
import       "../fetch"

LLAMA_DATA_DIR       :: #directory + "data/smollm"
LLAMA_DEFAULT_MODEL  :: LLAMA_DATA_DIR + "/model_instruct.safetensors"
LLAMA_TOKENIZER_PATH :: LLAMA_DATA_DIR + "/tokenizer.json"

LLAMA_ASSETS := []fetch.Asset{
	{
		url  = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors",
		dest = LLAMA_DEFAULT_MODEL,
		size = 269_060_552,
	}, {
		url  = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json",
		dest = LLAMA_TOKENIZER_PATH,
		size = 2_104_556,
	},
}

IM_START_TEXT :: "<|im_start|>"
IM_END_TEXT   :: "<|im_end|>"
EOT_TEXT      :: "<|endoftext|>"

Llama_Backend :: struct {
	model:         llama.Llama,
	cache:         llama.Cache,
	tokenizer:     gpt2.Tokenizer,
	system_prompt: string,
	im_start_id:   int,
	im_end_id:     int,
	eot_id:        int,
	first_turn:    bool,
	stop_tokens:   [2]int,
}

@(require_results)
llama_backend_make :: proc(model_path: string, t_max: int, system_prompt: string) -> (Chat_Model, bool) {
	path := model_path
	if builtin.len(path) == 0 {
		if !fetch.ensure_assets(LLAMA_ASSETS, "SmolLM2-135M-Instruct") {
			return {}, false
		}
		path = LLAMA_DEFAULT_MODEL
	} else if !os.exists(path) {
		log.errorf("no such file: %v", path)
		return {}, false
	}

	log.info("loading tokenizer")
	tokenizer, tokenizer_err := gpt2.load(LLAMA_TOKENIZER_PATH)
	if tokenizer_err != .None {
		log.errorf("could not load tokenizer %v: %v", LLAMA_TOKENIZER_PATH, tokenizer_err)
		return {}, false
	}

	im_start_id := tokenizer.added_tokens[IM_START_TEXT] if IM_START_TEXT in tokenizer.added_tokens else -1
	im_end_id   := tokenizer.added_tokens[IM_END_TEXT]   if IM_END_TEXT   in tokenizer.added_tokens else -1
	eot_id      := tokenizer.added_tokens[EOT_TEXT]      if EOT_TEXT      in tokenizer.added_tokens else -1
	if im_start_id < 0 || im_end_id < 0 {
		log.error("tokenizer missing <|im_start|> or <|im_end|>")
		gpt2.destroy(tokenizer)
		return {}, false
	}

	log.infof("allocating SmolLM2-135M (bf16, %v)", ML_BACKEND)
	model := llama.make(llama.SMOLLM2_135M_CONFIG, dtype=.Bf16, trainable=false)

	log.infof("loading weights from %v", path)
	load_ok := llama.load_safetensors(&model, path)
	if !load_ok {
		log.errorf("could not load weights from %v", path)
		llama.destroy(model)
		gpt2.destroy(tokenizer)
		return {}, false
	}

	backend := new(Llama_Backend)
	backend.model         = model
	backend.cache         = llama.cache_make(model, t_max)
	backend.tokenizer     = tokenizer
	backend.system_prompt = system_prompt
	backend.im_start_id   = im_start_id
	backend.im_end_id     = im_end_id
	backend.eot_id        = eot_id
	backend.first_turn    = true
	backend.stop_tokens   = {im_end_id, eot_id}

	return Chat_Model{
		data          = backend,
		vocab_size    = llama.SMOLLM2_135M_CONFIG.vocab_size,
		prefill_chunk = 0,
		stop_tokens   = backend.stop_tokens[:],
		eval          = _llama_eval,
		encode_turn   = _llama_encode_turn,
		decode        = _llama_decode,
		remaining     = _llama_remaining,
		reset         = _llama_reset,
		destroy       = _llama_destroy,
	}, true
}

_llama_eval :: proc(data: rawptr, tokens: []int, logits_out: []f32) {
	backend := (^Llama_Backend)(data)
	if ml.pass() {
		logits := llama.forward_cached(backend.model, &backend.cache, tokens, logits_mode=.Last)
		if logits_out != nil {
			ml.get_data(logits, logits_out)
		}
	}
}

_llama_encode_turn :: proc(data: rawptr, user_text: string) -> []int {
	backend := (^Llama_Backend)(data)

	out: [dynamic]int
	out.allocator = context.temp_allocator

	if backend.first_turn {
		_chatml_turn(&out, backend, "system", backend.system_prompt)
		backend.first_turn = false
	}
	_chatml_turn(&out, backend, "user", user_text)
	_chatml_prefix(&out, backend, "assistant")

	return out[:]
}

_chatml_turn :: proc(out: ^[dynamic]int, backend: ^Llama_Backend, role, content: string) {
	append(out, backend.im_start_id)
	header := fmt.tprintf("%v\n%v", role, content)
	append(out, ..gpt2.encode(&backend.tokenizer, header, context.temp_allocator))
	append(out, backend.im_end_id)
	append(out, ..gpt2.encode(&backend.tokenizer, "\n", context.temp_allocator))
}

_chatml_prefix :: proc(out: ^[dynamic]int, backend: ^Llama_Backend, role: string) {
	append(out, backend.im_start_id)
	header := fmt.tprintf("%v\n", role)
	append(out, ..gpt2.encode(&backend.tokenizer, header, context.temp_allocator))
}

_llama_decode :: proc(data: rawptr, tokens: []int) -> string {
	backend := (^Llama_Backend)(data)
	return gpt2.decode(&backend.tokenizer, tokens, context.temp_allocator)
}

_llama_remaining :: proc(data: rawptr) -> int {
	backend := (^Llama_Backend)(data)
	return ml.kv_cache_remaining(backend.cache)
}

_llama_reset :: proc(data: rawptr) {
	backend := (^Llama_Backend)(data)
	ml.kv_cache_reset(&backend.cache)
	backend.first_turn = true
}

_llama_destroy :: proc(data: rawptr) {
	backend := (^Llama_Backend)(data)
	ml.kv_cache_destroy(backend.cache)
	llama.destroy(backend.model)
	gpt2.destroy(backend.tokenizer)
	free(backend)
}
