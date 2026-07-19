package gemma_tokenizer

import "base:builtin"

import "core:encoding/json"
import "core:fmt"
import "core:log"
import "core:os"
import "core:strings"
import "core:unicode/utf8"

WHITESPACE_PIECE :: "▁"

Pair :: struct {
	a, b: string,
}

Tokenizer :: struct {
	vocab:         map[string]int,
	id_to_piece:   []string,
	merge_rank:    map[Pair]int,
	added_tokens:  map[string]int,
	byte_fallback: [256]int,

	_json_root: json.Value,
}

Error :: enum {
	None,
	Not_Found,
	Read_Failed,
	Malformed,
}

@(require_results)
load :: proc(path: string, allocator := context.allocator, loc := #caller_location) -> (tok: Tokenizer, err: Error) {
	context.allocator = allocator

	bytes, read_err := os.read_entire_file_from_path(path, allocator)
	if read_err != nil {
		if !os.exists(path) {
			log.debugf("tokenizer file not found: %v", path, location=loc)
			return {}, .Not_Found
		}
		log.errorf("failed to read %v: %v", path, read_err, location=loc)
		return {}, .Read_Failed
	}
	defer delete(bytes)

	root, parse_err := json.parse(bytes, parse_integers = true)
	if parse_err != .None {
		log.errorf("JSON parse error %v in %v", parse_err, path, location=loc)
		return {}, .Malformed
	}

	root_object, root_object_ok := root.(json.Object)
	if !root_object_ok {
		log.error("tokenizer.json root is not an object", location=loc)
		json.destroy_value(root)
		return {}, .Malformed
	}

	model_object, model_object_ok := root_object["model"].(json.Object)
	if !model_object_ok {
		log.error("missing 'model' object", location=loc)
		json.destroy_value(root)
		return {}, .Malformed
	}

	vocab_object, vocab_object_ok := model_object["vocab"].(json.Object)
	if !vocab_object_ok {
		log.error("missing 'model.vocab' object", location=loc)
		json.destroy_value(root)
		return {}, .Malformed
	}

	merges_array, merges_array_ok := model_object["merges"].(json.Array)
	if !merges_array_ok {
		log.error("missing 'model.merges' array", location=loc)
		json.destroy_value(root)
		return {}, .Malformed
	}

	tok._json_root  = root
	tok.id_to_piece = make([]string, len(vocab_object))
	for piece, id_value in vocab_object {
		id_int, id_int_ok := id_value.(json.Integer)
		if !id_int_ok || int(id_int) < 0 || int(id_int) >= len(vocab_object) {
			log.errorf("vocab entry %q has invalid id", piece, location=loc)
			destroy(tok)
			return {}, .Malformed
		}
		tok.vocab[piece] = int(id_int)
		tok.id_to_piece[int(id_int)] = piece
	}

	for merge_value, merge_index in merges_array {
		if merge_array, is_array := merge_value.(json.Array); is_array {
			if len(merge_array) != 2 {
				log.errorf("merge[%v] is not a 2-element array", merge_index, location=loc)
				destroy(tok)
				return {}, .Malformed
			}
			a_string, a_string_ok := merge_array[0].(string)
			b_string, b_string_ok := merge_array[1].(string)
			if !a_string_ok || !b_string_ok {
				log.errorf("merge[%v] contains a non-string", merge_index, location=loc)
				destroy(tok)
				return {}, .Malformed
			}
			tok.merge_rank[Pair{a_string, b_string}] = merge_index
		} else if merge_string, is_string := merge_value.(string); is_string {
			space_index := strings.index_byte(merge_string, ' ')
			if space_index <= 0 || space_index >= len(merge_string) - 1 {
				log.errorf("merge[%v] = %q has no space separator", merge_index, merge_string, location=loc)
				destroy(tok)
				return {}, .Malformed
			}
			tok.merge_rank[Pair{merge_string[:space_index], merge_string[space_index + 1:]}] = merge_index
		} else {
			log.errorf("merge[%v] has unsupported JSON type", merge_index, location=loc)
			destroy(tok)
			return {}, .Malformed
		}
	}

	for byte_value in 0 ..< 256 {
		tok.byte_fallback[byte_value] = -1
	}
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
			if !entry_object_ok {
				continue
			}
			content_string, content_ok := entry_object["content"].(string)
			id_int, id_int_ok          := entry_object["id"].(json.Integer)
			if content_ok && id_int_ok {
				tok.added_tokens[content_string] = int(id_int)
			}
		}
	}

	return tok, .None
}

destroy :: proc(tok: Tokenizer) {
	delete(tok.vocab)
	delete(tok.id_to_piece)
	delete(tok.merge_rank)
	delete(tok.added_tokens)
	json.destroy_value(tok._json_root)
}

@(require_results)
encode :: proc(tok: ^Tokenizer, text: string, allocator := context.allocator, loc := #caller_location) -> (result: []int, ok: bool) #optional_ok {
	ids: [dynamic]int
	ids.allocator = allocator

	ok = true
	cursor := 0
	for cursor < len(text) {
		match_content: string
		match_id := -1
		for content, id in tok.added_tokens {
			if cursor + len(content) > len(text) {
				continue
			}
			if text[cursor:cursor + len(content)] != content {
				continue
			}
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
		for content in tok.added_tokens {
			if len(content) == 0 {
				continue
			}
			idx := strings.index(text[cursor:], content)
			if idx >= 0 && cursor + idx < next_special {
				next_special = cursor + idx
			}
		}
		if !_encode_text_segment(tok, text[cursor:next_special], &ids, loc=loc) {
			ok = false
		}
		cursor = next_special
	}

	return ids[:], ok
}

_encode_text_segment :: proc(tok: ^Tokenizer, text: string, ids: ^[dynamic]int, loc := #caller_location) -> bool {
	if len(text) == 0 {
		return true
	}

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
					log.errorf("byte 0x%02X has no fallback token in vocab", normalized[offset + k], location=loc)
					return false
				}
			}
		}
		offset += rune_size
	}

	_apply_bpe(tok, &symbols, &merge_buffer)

	for symbol in symbols {
		id, present := tok.vocab[symbol]
		if !present {
			log.errorf("symbol %q not in vocab", symbol, location=loc)
			return false
		}
		append(ids, id)
	}
	return true
}

@(require_results)
decode :: proc(tok: ^Tokenizer, ids: []int, allocator := context.allocator) -> string {
	output: [dynamic]u8
	output.allocator = allocator

	for id in ids {
		if id < 0 || id >= len(tok.id_to_piece) {
			continue
		}
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
				for k in 0 ..< rune_byte_count {
					append(&output, rune_bytes[k])
				}
			}
			offset += rune_size
		}
	}

	return string(output[:])
}

_parse_byte_fallback_piece :: proc(piece: string) -> (u8, bool) {
	if len(piece) != 6 {
		return 0, false
	}
	if piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>' {
		return 0, false
	}

	high, high_ok := _hex_digit_value(piece[3])
	low,  low_ok  := _hex_digit_value(piece[4])

	if !high_ok || !low_ok {
		return 0, false
	}

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
	if len(symbols) < 2 {
		return
	}

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
		if best_index == -1 {
			break
		}

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

		if len(symbols) == 1 {
			break
		}
	}
}
