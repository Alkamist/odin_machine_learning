package gpt2_tokenizer

import "core:encoding/json"
import "core:log"
import "core:os"
import "core:strings"
import "core:unicode"
import "core:unicode/utf8"

Pair :: struct {
	a, b: string,
}

Tokenizer :: struct {
	vocab:           map[string]int,
	id_to_piece:     []string,
	merge_rank:      map[Pair]int,
	added_tokens:    map[string]int,
	byte_to_unicode: [256]rune,
	unicode_to_byte: map[rune]u8,

	_json_root: json.Value,
}

Error :: enum {
	None,
	Not_Found,
	Read_Failed,
	Malformed,
}

@(require_results)
load :: proc(path: string, allocator := context.allocator) -> (tok: Tokenizer, err: Error) {
	context.allocator = allocator

	bytes, read_err := os.read_entire_file_from_path(path, allocator)
	if read_err != nil {
		if !os.exists(path) {
			log.debugf("gpt2 tokenizer file not found: %v", path)
			return {}, .Not_Found
		}
		log.errorf("failed to read %v: %v", path, read_err)
		return {}, .Read_Failed
	}
	defer delete(bytes)

	root, parse_err := json.parse(bytes, parse_integers = true)
	if parse_err != .None {
		log.errorf("JSON parse error %v in %v", parse_err, path)
		return {}, .Malformed
	}

	root_object, root_object_ok := root.(json.Object)
	if !root_object_ok {
		log.error("tokenizer.json root is not an object")
		json.destroy_value(root)
		return {}, .Malformed
	}

	model_object, model_object_ok := root_object["model"].(json.Object)
	if !model_object_ok {
		log.error("missing 'model' object")
		json.destroy_value(root)
		return {}, .Malformed
	}

	vocab_object, vocab_object_ok := model_object["vocab"].(json.Object)
	if !vocab_object_ok {
		log.error("missing 'model.vocab' object")
		json.destroy_value(root)
		return {}, .Malformed
	}

	merges_array, merges_array_ok := model_object["merges"].(json.Array)
	if !merges_array_ok {
		log.error("missing 'model.merges' array")
		json.destroy_value(root)
		return {}, .Malformed
	}

	tok._json_root  = root
	tok.id_to_piece = make([]string, len(vocab_object))
	for piece, id_value in vocab_object {
		id_int, id_int_ok := id_value.(json.Integer)
		if !id_int_ok || int(id_int) < 0 || int(id_int) >= len(vocab_object) {
			log.errorf("vocab entry %q has invalid id", piece)
			destroy(tok)
			return {}, .Malformed
		}
		tok.vocab[piece] = int(id_int)
		tok.id_to_piece[int(id_int)] = piece
	}

	for merge_value, merge_index in merges_array {
		merge_string, merge_string_ok := merge_value.(string)
		if !merge_string_ok {
			log.errorf("merge[%v] is not a string", merge_index)
			destroy(tok)
			return {}, .Malformed
		}
		space_index := strings.index_byte(merge_string, ' ')
		if space_index <= 0 || space_index >= len(merge_string) - 1 {
			log.errorf("merge[%v] = %q has no space separator", merge_index, merge_string)
			destroy(tok)
			return {}, .Malformed
		}
		tok.merge_rank[Pair{merge_string[:space_index], merge_string[space_index + 1:]}] = merge_index
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

	_init_byte_unicode_maps(&tok)
	return tok, .None
}

destroy :: proc(tok: Tokenizer) {
	delete(tok.vocab)
	delete(tok.id_to_piece)
	delete(tok.merge_rank)
	delete(tok.added_tokens)
	delete(tok.unicode_to_byte)
	json.destroy_value(tok._json_root)
}

@(require_results)
encode :: proc(tok: ^Tokenizer, text: string, allocator := context.allocator) -> (result: []int, ok: bool) #optional_ok {
	pretokens: [dynamic]string
	pretokens.allocator = context.temp_allocator
	_pretokenize(text, &pretokens)

	encoded_buffer: [dynamic]u8
	encoded_buffer.allocator = context.temp_allocator

	symbols: [dynamic]string
	symbols.allocator = context.temp_allocator

	merge_buffer: [dynamic]string
	merge_buffer.allocator = context.temp_allocator

	ids: [dynamic]int
	ids.allocator = allocator

	for pretoken in pretokens {
		clear(&encoded_buffer)
		for byte_value in transmute([]u8)pretoken {
			rune_bytes, rune_size := utf8.encode_rune(tok.byte_to_unicode[byte_value])
			for k in 0 ..< rune_size {
				append(&encoded_buffer, rune_bytes[k])
			}
		}
		encoded := string(encoded_buffer[:])

		if id, present := tok.vocab[encoded]; present {
			append(&ids, id)
			continue
		}

		clear(&symbols)
		for offset := 0; offset < len(encoded); {
			_, rune_size := utf8.decode_rune_in_string(encoded[offset:])
			append(&symbols, encoded[offset:offset + rune_size])
			offset += rune_size
		}

		_apply_bpe(tok, &symbols, &merge_buffer)

		for symbol in symbols {
			id, present := tok.vocab[symbol]
			if !present {
				log.errorf("symbol %q not in vocab (pretoken=%q)", symbol, pretoken)
				return ids[:], false
			}
			append(&ids, id)
		}
	}

	return ids[:], true
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
		for offset := 0; offset < len(piece); {
			rune_value, rune_size := utf8.decode_rune_in_string(piece[offset:])
			if byte_value, present := tok.unicode_to_byte[rune_value]; present {
				append(&output, byte_value)
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

_init_byte_unicode_maps :: proc(tok: ^Tokenizer) {
	visible: [256]bool
	for byte_value in u8('!') ..= u8('~') {
		visible[byte_value] = true
	}
	for byte_value in u8(0xA1) ..= u8(0xAC) {
		visible[byte_value] = true
	}
	for byte_value in u8(0xAE) ..= u8(0xFF) {
		visible[byte_value] = true
	}

	next_extra := rune(256)
	for byte_value in 0 ..< 256 {
		if visible[byte_value] {
			tok.byte_to_unicode[byte_value] = rune(byte_value)
		} else {
			tok.byte_to_unicode[byte_value] = next_extra
			next_extra += 1
		}
	}
	for byte_value in 0 ..< 256 {
		tok.unicode_to_byte[tok.byte_to_unicode[byte_value]] = u8(byte_value)
	}
}

_pretokenize :: proc(text: string, out: ^[dynamic]string) {
	chunk_start := 0
	for offset := 0; offset < len(text); {
		rune_value, rune_size := utf8.decode_rune_in_string(text[offset:])
		if unicode.is_digit(rune_value) {
			if offset > chunk_start {
				_gpt2_split(text[chunk_start:offset], out)
			}
			append(out, text[offset:offset + rune_size])
			offset += rune_size
			chunk_start = offset
		} else {
			offset += rune_size
		}
	}
	if chunk_start < len(text) {
		_gpt2_split(text[chunk_start:], out)
	}
}

_gpt2_split :: proc(text: string, out: ^[dynamic]string) {
	offset := 0
	for offset < len(text) {
		match_length := _match_one(text, offset)
		append(out, text[offset:offset + match_length])
		offset += match_length
	}
}

_match_one :: proc(text: string, start: int) -> int {
	first_rune, first_size := utf8.decode_rune_in_string(text[start:])

	if first_rune == '\'' {
		contraction_length := _match_contraction(text, start)
		if contraction_length > 0 {
			return contraction_length
		}
	}

	if unicode.is_letter(first_rune) {
		return first_size + _consume_run(text, start + first_size, _is_letter_rune)
	}
	if unicode.is_number(first_rune) {
		return first_size + _consume_run(text, start + first_size, _is_number_rune)
	}
	if !unicode.is_space(first_rune) {
		return first_size + _consume_run(text, start + first_size, _is_other_rune)
	}

	whitespace_end := start + first_size
	for whitespace_end < len(text) {
		next_rune, next_size := utf8.decode_rune_in_string(text[whitespace_end:])
		if !unicode.is_space(next_rune) {
			break
		}
		whitespace_end += next_size
	}

	if whitespace_end == len(text) {
		return whitespace_end - start
	}

	successor_rune, _ := utf8.decode_rune_in_string(text[whitespace_end:])
	if whitespace_end - start == first_size {
		successor_size := utf8.rune_size(successor_rune)
		successor_predicate: proc(rune) -> bool = _is_other_rune
		switch {
		case unicode.is_letter(successor_rune): successor_predicate = _is_letter_rune
		case unicode.is_number(successor_rune): successor_predicate = _is_number_rune
		}
		return first_size + successor_size + _consume_run(text, whitespace_end + successor_size, successor_predicate)
	}

	_, last_whitespace_size := utf8.decode_last_rune_in_string(text[start:whitespace_end])
	return whitespace_end - start - last_whitespace_size
}

_match_contraction :: proc(text: string, start: int) -> int {
	rest := text[start:]
	contractions := [?]string{"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"}
	for contraction in contractions {
		if len(rest) >= len(contraction) && rest[:len(contraction)] == contraction {
			return len(contraction)
		}
	}
	return 0
}

_consume_run :: proc(text: string, start: int, predicate: proc(rune) -> bool) -> int {
	offset := start
	for offset < len(text) {
		rune_value, rune_size := utf8.decode_rune_in_string(text[offset:])
		if !predicate(rune_value) {
			break
		}
		offset += rune_size
	}
	return offset - start
}

_is_letter_rune :: proc(r: rune) -> bool { return unicode.is_letter(r) }
_is_number_rune :: proc(r: rune) -> bool { return unicode.is_number(r) }
_is_other_rune  :: proc(r: rune) -> bool {
	return !unicode.is_space(r) && !unicode.is_letter(r) && !unicode.is_number(r)
}

_apply_bpe :: proc(tok: ^Tokenizer, symbols: ^[dynamic]string, merge_buffer: ^[dynamic]string) {
	if len(symbols) < 2 {
		return
	}

	for {
		best_rank  := max(int)
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
