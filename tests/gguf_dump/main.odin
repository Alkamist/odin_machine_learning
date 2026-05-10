package gguf_dump

import "core:fmt"
import "core:os"
import "core:slice"
import "core:strings"

import "../../loaders/gguf"

main :: proc() {
	args := os.args
	if len(args) < 2 {
		fmt.eprintfln("usage: %v <gguf_file>", args[0])
		os.exit(1)
	}
	path := args[1]

	loader, ok := gguf.load(path)
	if !ok {
		os.exit(1)
	}
	defer gguf.destroy(loader)

	fmt.printfln("file:       %v", path)
	fmt.printfln("version:    %v", loader.version)
	fmt.printfln("alignment:  %v", loader.alignment)
	fmt.printfln("data_start: %v", loader.data_start)
	fmt.printfln("kv pairs:   %v", len(loader.kv))
	fmt.printfln("tensors:    %v", len(loader.tensors))

	fmt.println()
	fmt.println("KV (sorted):")
	keys := make([dynamic]string, 0, len(loader.kv))
	defer delete(keys)
	for key, _ in loader.kv {
		append(&keys, key)
	}
	slice.sort(keys[:])
	for key in keys {
		entry := loader.kv[key]
		switch entry.type {
		case .U8, .I8, .U16, .I16, .U32, .I32, .U64, .I64, .F32, .F64, .Bool:
			fmt.printfln("  %-50s %-8v %v", key, entry.type, _format_scalar(loader, entry))
		case .String:
			s, _ := gguf.get_str(loader, key)
			s_disp := s
			if len(s_disp) > 80 {
				s_disp = strings.concatenate({s_disp[:80], "…"}, context.temp_allocator)
			}
			fmt.printfln("  %-50s %-8v %q", key, entry.type, s_disp)
		case .Array:
			fmt.printfln("  %-50s %-8v elem=%v count=%v", key, entry.type, entry.array_type, entry.array_count)
		}
	}

	fmt.println()
	fmt.println("Tensors (sorted):")
	tnames := make([dynamic]string, 0, len(loader.tensors))
	defer delete(tnames)
	for name, _ in loader.tensors {
		append(&tnames, name)
	}
	slice.sort(tnames[:])
	for name in tnames {
		info := loader.tensors[name]
		fmt.printfln("  %-60s %-6v shape=%v bytes=%v", name, info.type, info.shape, info.byte_count)
	}

	// Type histogram.
	type_counts: map[gguf.Tensor_Type]int
	defer delete(type_counts)
	for _, info in loader.tensors {
		type_counts[info.type] += 1
	}
	fmt.println()
	fmt.println("Tensor type histogram:")
	for ty, count in type_counts {
		fmt.printfln("  %-6v %v", ty, count)
	}
}

_format_scalar :: proc(loader: gguf.Loader, entry: gguf.KV) -> string {
	switch entry.type {
	case .U32:
		v, _ := gguf.get_u32(loader, entry.key)
		return fmt.tprintf("%v", v)
	case .I32:
		v, _ := gguf.get_i32(loader, entry.key)
		return fmt.tprintf("%v", v)
	case .U64:
		v, _ := gguf.get_u64(loader, entry.key)
		return fmt.tprintf("%v", v)
	case .I64:
		v, _ := gguf.get_i64(loader, entry.key)
		return fmt.tprintf("%v", v)
	case .F32:
		v, _ := gguf.get_f32(loader, entry.key)
		return fmt.tprintf("%v", v)
	case .Bool:
		v, _ := gguf.get_bool(loader, entry.key)
		return fmt.tprintf("%v", v)
	case .U8, .I8, .U16, .I16, .F64, .String, .Array:
		return "<not formatted>"
	}
	return "?"
}
