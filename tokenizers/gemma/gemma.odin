package gemma_tokenizer

// SentencePiece-style byte-fallback BPE tokenizer compatible with the
// HuggingFace `tokenizer.json` shipped with Gemma 4. Encode pipeline:
// text → replace ' ' with '▁' → split into unicode scalars → byte-fallback
// for scalars not in vocab → BPE merges → vocab IDs. Decode reverses the
// pipeline, fusing adjacent `<0xHH>` byte tokens into raw bytes before
// the final ▁→space replacement.

import "base:builtin"

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:unicode/utf8"

WHITESPACE_PIECE :: "▁" // ▁ (U+2581 LOWER ONE EIGHTH BLOCK)

Pair :: struct {
	a, b: string,
}

Tokenizer :: struct {
	vocab:         map[string]int,
	id_to_piece:   []string,
	merge_rank:    map[Pair]int,
	added_tokens:  map[string]int,
	byte_fallback: [256]int, // -1 if missing, else token id of `<0xHH>`

	_json_root:    json.Value,
}

@(require_results)
load :: proc(path: string, allocator := context.allocator) -> (tok: Tokenizer, ok: bool) {
	context.allocator = allocator

	bytes, read_err := os.read_entire_file_from_path(path, allocator)
	if read_err != nil {
		fmt.eprintfln("gemma.load: failed to read %v: %v", path, read_err)
		return {}, false
	}
	defer delete(bytes)

	root, parse_err := json.parse(bytes, parse_integers = true)
	if parse_err != .None {
		fmt.eprintfln("gemma.load: JSON parse error %v in %v", parse_err, path)
		return {}, false
	}

	root_object, root_object_ok := root.(json.Object)
	if !root_object_ok {
		fmt.eprintln("gemma.load: tokenizer.json root is not an object")
		json.destroy_value(root)
		return {}, false
	}

	model_object, model_object_ok := root_object["model"].(json.Object)
	if !model_object_ok {
		fmt.eprintln("gemma.load: missing 'model' object")
		json.destroy_value(root)
		return {}, false
	}

	vocab_object, vocab_object_ok := model_object["vocab"].(json.Object)
	if !vocab_object_ok {
		fmt.eprintln("gemma.load: missing 'model.vocab' object")
		json.destroy_value(root)
		return {}, false
	}

	merges_array, merges_array_ok := model_object["merges"].(json.Array)
	if !merges_array_ok {
		fmt.eprintln("gemma.load: missing 'model.merges' array")
		json.destroy_value(root)
		return {}, false
	}

	tok._json_root  = root
	tok.id_to_piece = make([]string, len(vocab_object))
	for piece, id_value in vocab_object {
		id_int, id_int_ok := id_value.(json.Integer)
		if !id_int_ok || int(id_int) < 0 || int(id_int) >= len(vocab_object) {
			fmt.eprintfln("gemma.load: vocab entry %q has invalid id", piece)
			destroy(tok)
			return {}, false
		}
		tok.vocab[piece] = int(id_int)
		tok.id_to_piece[int(id_int)] = piece
	}

	for merge_value, merge_index in merges_array {
		// HF Gemma writes merges as `[a, b]` arrays of two strings. Older
		// HF tokenizers serialised them as `"a b"` space-joined strings;
		// accept both for forward compatibility.
		if merge_array, is_array := merge_value.(json.Array); is_array {
			if len(merge_array) != 2 {
				fmt.eprintfln("gemma.load: merge[%v] is not a 2-element array", merge_index)
				destroy(tok)
				return {}, false
			}
			a_string, a_string_ok := merge_array[0].(string)
			b_string, b_string_ok := merge_array[1].(string)
			if !a_string_ok || !b_string_ok {
				fmt.eprintfln("gemma.load: merge[%v] contains a non-string", merge_index)
				destroy(tok)
				return {}, false
			}
			tok.merge_rank[Pair{a_string, b_string}] = merge_index
		} else if merge_string, is_string := merge_value.(string); is_string {
			space_index := strings.index_byte(merge_string, ' ')
			if space_index <= 0 || space_index >= len(merge_string) - 1 {
				fmt.eprintfln("gemma.load: merge[%v] = %q has no space separator", merge_index, merge_string)
				destroy(tok)
				return {}, false
			}
			tok.merge_rank[Pair{merge_string[:space_index], merge_string[space_index + 1:]}] = merge_index
		} else {
			fmt.eprintfln("gemma.load: merge[%v] has unsupported JSON type", merge_index)
			destroy(tok)
			return {}, false
		}
	}

	for byte_value in 0 ..< 256 do tok.byte_fallback[byte_value] = -1
	byte_token_buffer: [8]u8
	for byte_value in 0 ..< 256 {
		byte_token := fmt.bprintf(byte_token_buffer[:], "<0x%02X>", byte_value)
		if id, present := tok.vocab[byte_token]; present {
			tok.byte_fallback[byte_value] = id
		}
	}

	if added, added_present := root_object["added_tokens"]; added_present {
		added_array, _ := added.(json.Array)
		for entry in added_array {
			entry_object, entry_object_ok := entry.(json.Object)
			if !entry_object_ok do continue
			content_string, content_ok := entry_object["content"].(string)
			id_int, id_int_ok          := entry_object["id"].(json.Integer)
			if content_ok && id_int_ok do tok.added_tokens[content_string] = int(id_int)
		}
	}

	return tok, true
}

destroy :: proc(tok: Tokenizer) {
	delete(tok.vocab)
	delete(tok.id_to_piece)
	delete(tok.merge_rank)
	delete(tok.added_tokens)
	json.destroy_value(tok._json_root)
}

@(require_results)
encode :: proc(tok: ^Tokenizer, text: string, allocator := context.allocator) -> []int {
	ids: [dynamic]int
	ids.allocator = allocator

	// Pre-extract added/special tokens (e.g. `<bos>`, `<start_of_turn>`,
	// `<end_of_turn>`) as single IDs. We scan left-to-right; at each cursor
	// position we look for the longest matching added-token prefix and emit
	// its ID directly. Text in the gaps goes through normal BPE.
	cursor := 0
	for cursor < len(text) {
		match_content: string
		match_id := -1
		for content, id in tok.added_tokens {
			if cursor + len(content) > len(text) do continue
			if text[cursor:cursor + len(content)] != content do continue
			if len(content) > len(match_content) {
				match_content = content
				match_id      = id
			}
		}
		if match_id >= 0 {
			append(&ids, match_id)
			cursor += len(match_content)
			continue
		}
		next_special := len(text)
		for content, _ in tok.added_tokens {
			if len(content) == 0 do continue
			idx := strings.index(text[cursor:], content)
			if idx >= 0 && cursor + idx < next_special {
				next_special = cursor + idx
			}
		}
		_encode_text_segment(tok, text[cursor:next_special], &ids)
		cursor = next_special
	}
	return ids[:]
}

_encode_text_segment :: proc(tok: ^Tokenizer, text: string, ids: ^[dynamic]int) {
	if len(text) == 0 do return

	normalized, _ := strings.replace_all(text, " ", WHITESPACE_PIECE, context.temp_allocator)

	symbols: [dynamic]string
	symbols.allocator = context.temp_allocator

	merge_buffer: [dynamic]string
	merge_buffer.allocator = context.temp_allocator

	byte_token_buffer: [8]u8

	for offset := 0; offset < len(normalized); {
		_, rune_size := utf8.decode_rune_in_string(normalized[offset:])
		piece := normalized[offset:offset + rune_size]
		if _, present := tok.vocab[piece]; present {
			append(&symbols, piece)
		} else {
			for k in 0 ..< rune_size {
				byte_token := fmt.bprintf(byte_token_buffer[:], "<0x%02X>", normalized[offset + k])
				if id, present_byte := tok.vocab[byte_token]; present_byte {
					_ = id
					append(&symbols, strings.clone(byte_token, context.temp_allocator))
				} else {
					fmt.eprintfln("gemma.encode: byte 0x%02X has no fallback token in vocab", normalized[offset + k])
					return
				}
			}
		}
		offset += rune_size
	}

	_apply_bpe(tok, &symbols, &merge_buffer)

	for symbol in symbols {
		id, present := tok.vocab[symbol]
		if !present {
			fmt.eprintfln("gemma.encode: symbol %q not in vocab", symbol)
			return
		}
		append(ids, id)
	}
}

@(require_results)
decode :: proc(tok: ^Tokenizer, ids: []int, allocator := context.allocator) -> string {
	output: [dynamic]u8
	output.allocator = allocator

	for id in ids {
		if id < 0 || id >= len(tok.id_to_piece) do continue
		piece := tok.id_to_piece[id]
		if byte_value, ok := _parse_byte_fallback_piece(piece); ok {
			append(&output, byte_value)
			continue
		}
		for offset := 0; offset < len(piece); {
			rune_value, rune_size := utf8.decode_rune_in_string(piece[offset:])
			if rune_value == '▁' {
				append(&output, ' ')
			} else {
				rune_bytes, rune_byte_count := utf8.encode_rune(rune_value)
				for k in 0 ..< rune_byte_count do append(&output, rune_bytes[k])
			}
			offset += rune_size
		}
	}
	return string(output[:])
}

_parse_byte_fallback_piece :: proc(piece: string) -> (u8, bool) {
	if len(piece) != 6 do return 0, false
	if piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>' do return 0, false
	high, high_ok := _hex_digit_value(piece[3])
	low,  low_ok  := _hex_digit_value(piece[4])
	if !high_ok || !low_ok do return 0, false
	return high * 16 + low, true
}

_hex_digit_value :: proc(c: u8) -> (u8, bool) {
	switch c {
	case '0' ..= '9': return c - '0', true
	case 'A' ..= 'F': return c - 'A' + 10, true
	case 'a' ..= 'f': return c - 'a' + 10, true
	}
	return 0, false
}

_apply_bpe :: proc(tok: ^Tokenizer, symbols: ^[dynamic]string, merge_buffer: ^[dynamic]string) {
	if len(symbols) < 2 do return

	for {
		best_rank  := builtin.max(int)
		best_index := -1
		for i in 0 ..< len(symbols) - 1 {
			if rank, present := tok.merge_rank[Pair{symbols[i], symbols[i + 1]}]; present {
				if rank < best_rank {
					best_rank  = rank
					best_index = i
				}
			}
		}
		if best_index == -1 do break

		first_symbol  := symbols[best_index]
		second_symbol := symbols[best_index + 1]

		clear(merge_buffer)
		i := 0
		for i < len(symbols) {
			if i + 1 < len(symbols) && symbols[i] == first_symbol && symbols[i + 1] == second_symbol {
				append(merge_buffer, strings.concatenate({first_symbol, second_symbol}, context.temp_allocator))
				i += 2
			} else {
				append(merge_buffer, symbols[i])
				i += 1
			}
		}

		clear(symbols)
		append(symbols, ..merge_buffer[:])
		if len(symbols) == 1 do break
	}
}