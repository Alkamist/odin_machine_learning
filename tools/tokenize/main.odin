// Tokenize a UTF-8 text file into the flat int32 binary format the
// trainers consume.
//
// Output layout:
//   bytes 0-3 : u32 little-endian count
//   bytes 4.. : count * i32 token ids
//
//   odin run tools/tokenize -o:speed -- \
//       --input  examples/data/shakespeare.txt \
//       --output examples/data/shakespeare_gemma.bin \
//       --tokenizer gemma_data/tokenizer.json
//
// `--type` selects the tokenizer (gemma or gpt2); defaults to gemma.
//
// The input is encoded in CHUNK-sized line groups so the BPE merge loop
// stays linear-ish (it's O(N^2) per encode call). Chunks split at newline
// boundaries to keep tokenization stable across edits to the corpus.

package main

import "base:builtin"

import "core:fmt"
import "core:mem"
import "core:os"
import "core:time"

import gemma_tok "../../tokenizers/gemma"
import gpt2_tok  "../../tokenizers/gpt2"

CHUNK_TARGET_BYTES :: 4 * 1024

Tokenizer_Kind :: enum {
	Gemma,
	GPT2,
}

main :: proc() {
	input_path:     string
	output_path:    string
	tokenizer_path: string
	tokenizer_type := Tokenizer_Kind.Gemma

	args := os.args[1:]
	i := 0
	for i < builtin.len(args) {
		arg := args[i]
		switch arg {
		case "--input":     i += 1; input_path     = args[i]; i += 1
		case "--output":    i += 1; output_path    = args[i]; i += 1
		case "--tokenizer": i += 1; tokenizer_path = args[i]; i += 1
		case "--type":
			i += 1
			switch args[i] {
			case "gemma": tokenizer_type = .Gemma
			case "gpt2":  tokenizer_type = .GPT2
			case:
				fmt.eprintfln("unknown tokenizer type: %v (expected gemma or gpt2)", args[i])
				os.exit(1)
			}
			i += 1
		case "--help", "-h":
			fmt.println("usage: tokenize --input TEXT_PATH --output BIN_PATH --tokenizer TOKENIZER_JSON [--type gemma|gpt2]")
			os.exit(0)
		case:
			fmt.eprintfln("unknown argument: %v", arg)
			os.exit(1)
		}
	}

	if input_path == "" || output_path == "" || tokenizer_path == "" {
		fmt.eprintln("usage: tokenize --input TEXT_PATH --output BIN_PATH --tokenizer TOKENIZER_JSON [--type gemma|gpt2]")
		os.exit(1)
	}

	fmt.printfln("Reading %v ...", input_path)
	corpus_bytes, read_err := os.read_entire_file_from_path(input_path, context.allocator)
	if read_err != nil {
		fmt.eprintfln("FAIL: could not read %v: %v", input_path, read_err)
		os.exit(1)
	}
	defer delete(corpus_bytes)
	fmt.printfln("  %v bytes", builtin.len(corpus_bytes))

	tokens: [dynamic]int
	defer delete(tokens)
	t_start := time.tick_now()

	chunks := chunk_at_newlines(string(corpus_bytes))
	defer delete(chunks)

	switch tokenizer_type {
	case .Gemma:
		fmt.printfln("Loading Gemma tokenizer from %v ...", tokenizer_path)
		t, ok := gemma_tok.load(tokenizer_path)
		if !ok {
			fmt.eprintln("FAIL: could not load Gemma tokenizer")
			os.exit(1)
		}
		defer gemma_tok.destroy(t)
		fmt.printfln("Tokenizing %v chunks ...", builtin.len(chunks))
		for chunk, i in chunks {
			ids := gemma_tok.encode(&t, chunk, context.temp_allocator)
			append(&tokens, ..ids)
			if i % 64 == 0 {
				print_progress(i, builtin.len(chunks), t_start)
			}
		}
		print_progress(builtin.len(chunks), builtin.len(chunks), t_start)
		fmt.println()
	case .GPT2:
		fmt.printfln("Loading GPT-2 tokenizer from %v ...", tokenizer_path)
		t, ok := gpt2_tok.load(tokenizer_path)
		if !ok {
			fmt.eprintln("FAIL: could not load GPT-2 tokenizer")
			os.exit(1)
		}
		defer gpt2_tok.destroy(t)
		fmt.printfln("Tokenizing %v chunks ...", builtin.len(chunks))
		for chunk, i in chunks {
			ids := gpt2_tok.encode(&t, chunk, context.temp_allocator)
			append(&tokens, ..ids)
			if i % 64 == 0 {
				print_progress(i, builtin.len(chunks), t_start)
			}
		}
		print_progress(builtin.len(chunks), builtin.len(chunks), t_start)
		fmt.println()
	}

	elapsed := f64(time.duration_seconds(time.tick_since(t_start)))
	fmt.printfln("  %v tokens in %.1f s (%.0f tok/s)",
		builtin.len(tokens), elapsed,
		f64(builtin.len(tokens)) / elapsed)

	// Pack header + tokens, write in one shot.
	count := u32le(builtin.len(tokens))
	header := mem.slice_to_bytes([]u32le{count})

	body := builtin.make([]i32, builtin.len(tokens))
	defer delete(body)
	for v, idx in tokens {
		body[idx] = i32(v)
	}
	body_bytes := mem.slice_to_bytes(body)

	buf := builtin.make([]byte, builtin.len(header) + builtin.len(body_bytes))
	defer delete(buf)
	copy(buf,                             header)
	copy(buf[builtin.len(header):],       body_bytes)

	fmt.printfln("Writing %v ...", output_path)
	if err := os.write_entire_file(output_path, buf); err != nil {
		fmt.eprintfln("FAIL: could not write %v: %v", output_path, err)
		os.exit(1)
	}
	fmt.printfln("  wrote %v bytes (%v + 4-byte header)", builtin.len(buf), builtin.len(body_bytes))
	fmt.println("Done.")
}

// Split text into chunks of up to CHUNK_TARGET_BYTES, breaking at newline
// boundaries so multi-byte runes and BPE merges aren't fragmented mid-token.
// Each chunk's trailing `\n` (if present) is kept on the chunk so the
// concatenated tokenization stays close to encoding the whole file at once.
chunk_at_newlines :: proc(text: string) -> [dynamic]string {
	out: [dynamic]string

	cursor := 0
	n := builtin.len(text)
	for cursor < n {
		end := cursor + CHUNK_TARGET_BYTES
		if end > n {
			end = n
		} else {
			// Walk back to the first newline at or before `end`.
			scan := end
			for scan > cursor && text[scan - 1] != '\n' {
				scan -= 1
			}
			if scan == cursor {
				// No newline in this window: extend forward to the next
				// newline (or EOF) so we never split mid-line.
				scan = end
				for scan < n && text[scan] != '\n' {
					scan += 1
				}
				if scan < n {
					scan += 1
				}
			}
			end = scan
		}
		append(&out, text[cursor:end])
		cursor = end
	}
	return out
}

print_progress :: proc(done, total: int, t_start: time.Tick) {
	elapsed := f64(time.duration_seconds(time.tick_since(t_start)))
	rate    := f64(done) / elapsed if elapsed > 0 else 0
	pct     := 100.0 * f64(done) / f64(total)
	fmt.printf("\r  %.0f%% (%v / %v chunks, %.1f s, %.0f chunks/s)        ", pct, done, total, elapsed, rate)
	os.flush(os.stdout)
}