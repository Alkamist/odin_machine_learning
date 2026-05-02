package gguf

import "core:fmt"
import "core:os"
import "core:slice"

// GGUF v3 reader. The on-disk layout per llama.cpp/ggml/include/gguf.h:
//
//   magic        u32 = 'GGUF'
//   version      u32
//   tensor_count u64
//   kv_count     u64
//   for each KV pair:
//     key       (u64 length, then bytes, no NUL)
//     type      u32 (Value_Type)
//     value     (depends on type; arrays nest type+count+payload)
//   for each tensor:
//     name      (u64 length, then bytes)
//     n_dims    u32
//     dims      n_dims * u64
//     type      u32 (Tensor_Type)
//     offset    u64 (relative to data_start, aligned to general.alignment)
//   pad to alignment
//   tensor data blob

GGUF_MAGIC             :: u32(0x4655_4747) // 'GGUF' little-endian
GGUF_DEFAULT_ALIGNMENT :: 32

Value_Type :: enum u32 {
	U8     = 0,
	I8     = 1,
	U16    = 2,
	I16    = 3,
	U32    = 4,
	I32    = 5,
	F32    = 6,
	Bool   = 7,
	String = 8,
	Array  = 9,
	U64    = 10,
	I64    = 11,
	F64    = 12,
}

// ggml_type values from llama.cpp/ggml/include/ggml.h. Only the entries we
// expect to see in real model files are named; an unknown id stays as its
// raw u32 cast and `_byte_count` rejects it.
Tensor_Type :: enum u32 {
	F32  = 0,
	F16  = 1,
	Q4_0 = 2,
	Q4_1 = 3,
	Q5_0 = 6,
	Q5_1 = 7,
	Q8_0 = 8,
	Q8_1 = 9,
	Q2_K = 10,
	Q3_K = 11,
	Q4_K = 12,
	Q5_K = 13,
	Q6_K = 14,
	Q8_K = 15,
	BF16 = 30,
}

Tensor_Info :: struct {
	name:        string,
	type:        Tensor_Type,
	shape:       []int,
	data_offset: int, // byte offset relative to Loader.data_start
	byte_count:  int,
}

// Pre-parsed KV pair. For scalars/strings we stash the byte range so the
// typed accessors can decode on demand. For arrays we keep the element type
// and count so the accessor can stride into the payload.
KV :: struct {
	key:          string,
	type:         Value_Type,
	value_offset: int, // offset in Loader.bytes where the value starts
	// Array-only:
	array_type:   Value_Type,
	array_count:  int,
}

Loader :: struct {
	bytes:      []byte,
	data_start: int, // first byte of tensor data blob (post-alignment padding)
	alignment:  int,
	version:    u32,
	tensors:    map[string]Tensor_Info,
	kv:         map[string]KV,
}

@(require_results)
load :: proc(path: string, allocator := context.allocator) -> (loader: Loader, ok: bool) {
	file_bytes, read_err := os.read_entire_file_from_path(path, allocator)
	if read_err != nil {
		fmt.eprintfln("gguf.load: failed to read %v: %v", path, read_err)
		return {}, false
	}

	r := Reader{bytes=file_bytes}

	magic, magic_ok := _read_u32(&r)
	if !magic_ok || magic != GGUF_MAGIC {
		fmt.eprintfln("gguf.load: %v: bad magic 0x%08x (expected 'GGUF')", path, magic)
		delete(file_bytes)
		return {}, false
	}

	version, version_ok := _read_u32(&r)
	if !version_ok || version != 3 {
		fmt.eprintfln("gguf.load: %v: unsupported GGUF version %v (only 3 implemented)", path, version)
		delete(file_bytes)
		return {}, false
	}

	tensor_count, tensor_count_ok := _read_u64(&r)
	kv_count,     kv_count_ok     := _read_u64(&r)
	if !tensor_count_ok || !kv_count_ok {
		fmt.eprintfln("gguf.load: %v: short read in header counts", path)
		delete(file_bytes)
		return {}, false
	}

	kv: map[string]KV
	for _ in 0 ..< kv_count {
		key, key_ok := _read_str(&r, file_bytes)
		if !key_ok {
			_destroy_partial(file_bytes, kv, nil)
			return {}, false
		}

		ty_raw, ty_raw_ok := _read_u32(&r)
		if !ty_raw_ok {
			_destroy_partial(file_bytes, kv, nil)
			return {}, false
		}
		ty := Value_Type(ty_raw)

		entry := KV{key = key, type = ty, value_offset = r.offset}

		if ty == .Array {
			arr_ty_raw,  arr_ty_ok    := _read_u32(&r)
			arr_count,   arr_count_ok := _read_u64(&r)
			if !arr_ty_ok || !arr_count_ok {
				_destroy_partial(file_bytes, kv, nil)
				return {}, false
			}
			entry.array_type  = Value_Type(arr_ty_raw)
			entry.array_count = int(arr_count)
			if !_skip_array_payload(&r, entry.array_type, entry.array_count) {
				fmt.eprintfln("gguf.load: array payload overrun at key %q", key)
				_destroy_partial(file_bytes, kv, nil)
				return {}, false
			}
		} else {
			if !_skip_scalar_payload(&r, ty) {
				fmt.eprintfln("gguf.load: scalar payload overrun at key %q (type %v)", key, ty)
				_destroy_partial(file_bytes, kv, nil)
				return {}, false
			}
		}
		kv[key] = entry
	}

	tensors: map[string]Tensor_Info
	for _ in 0 ..< tensor_count {
		name, name_ok := _read_str(&r, file_bytes)
		if !name_ok {
			_destroy_partial(file_bytes, kv, tensors)
			return {}, false
		}

		n_dims, n_dims_ok := _read_u32(&r)
		if !n_dims_ok {
			_destroy_partial(file_bytes, kv, tensors)
			return {}, false
		}

		shape := make([]int, n_dims)
		for i in 0 ..< int(n_dims) {
			d, d_ok := _read_u64(&r)
			if !d_ok {
				delete(shape)
				_destroy_partial(file_bytes, kv, tensors)
				return {}, false
			}
			shape[i] = int(d)
		}

		ty_raw, ty_raw_ok := _read_u32(&r)
		offset, offset_ok := _read_u64(&r)
		if !ty_raw_ok || !offset_ok {
			delete(shape)
			_destroy_partial(file_bytes, kv, tensors)
			return {}, false
		}

		element_count := 1
		for d in shape do element_count *= d

		ty := Tensor_Type(ty_raw)
		bc, bc_ok := _byte_count(ty, element_count)
		if !bc_ok {
			fmt.eprintfln("gguf.load: tensor %q has unsupported type id %v", name, ty_raw)
			delete(shape)
			_destroy_partial(file_bytes, kv, tensors)
			return {}, false
		}

		tensors[name] = Tensor_Info{
			name        = name,
			type        = ty,
			shape       = shape,
			data_offset = int(offset),
			byte_count  = bc,
		}
	}

	alignment := GGUF_DEFAULT_ALIGNMENT
	if entry, found := kv["general.alignment"]; found && entry.type == .U32 {
		alignment = int((^u32le)(raw_data(file_bytes[entry.value_offset:]))^)
	}

	pad := (alignment - (r.offset % alignment)) % alignment
	data_start := r.offset + pad
	if data_start > len(file_bytes) {
		fmt.eprintfln("gguf.load: %v: data_start %v overruns file size %v", path, data_start, len(file_bytes))
		_destroy_partial(file_bytes, kv, tensors)
		return {}, false
	}

	loader.bytes      = file_bytes
	loader.data_start = data_start
	loader.alignment  = alignment
	loader.version    = version
	loader.tensors    = tensors
	loader.kv         = kv
	return loader, true
}

destroy :: proc(loader: Loader) {
	for _, info in loader.tensors {
		delete(info.shape)
	}
	tensors := loader.tensors
	delete(tensors)
	kv := loader.kv
	delete(kv)
	delete(loader.bytes)
}

@(require_results)
get_info :: proc(loader: Loader, name: string) -> (info: Tensor_Info, ok: bool) {
	info, ok = loader.tensors[name]
	return
}

@(require_results)
get_bytes :: proc(loader: Loader, name: string) -> ([]byte, bool) {
	info, ok := loader.tensors[name]
	if !ok do return nil, false
	start := loader.data_start + info.data_offset
	return loader.bytes[start : start + info.byte_count], true
}

@(require_results)
shapes_match :: proc(a, b: []int) -> bool {
	return slice.equal(a, b)
}

// --- KV accessors ----------------------------------------------------------

@(require_results)
get_u32 :: proc(loader: Loader, key: string) -> (u32, bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .U32 do return 0, false
	return u32((^u32le)(raw_data(loader.bytes[entry.value_offset:]))^), true
}

@(require_results)
get_i32 :: proc(loader: Loader, key: string) -> (i32, bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .I32 do return 0, false
	return i32((^i32le)(raw_data(loader.bytes[entry.value_offset:]))^), true
}

@(require_results)
get_u64 :: proc(loader: Loader, key: string) -> (u64, bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .U64 do return 0, false
	return u64((^u64le)(raw_data(loader.bytes[entry.value_offset:]))^), true
}

@(require_results)
get_i64 :: proc(loader: Loader, key: string) -> (i64, bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .I64 do return 0, false
	return i64((^i64le)(raw_data(loader.bytes[entry.value_offset:]))^), true
}

@(require_results)
get_f32 :: proc(loader: Loader, key: string) -> (f32, bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .F32 do return 0, false
	return f32((^f32le)(raw_data(loader.bytes[entry.value_offset:]))^), true
}

@(require_results)
get_bool :: proc(loader: Loader, key: string) -> (bool, bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .Bool do return false, false
	return loader.bytes[entry.value_offset] != 0, true
}

// Returns a slice into the loader's bytes; valid for the lifetime of the loader.
@(require_results)
get_str :: proc(loader: Loader, key: string) -> (string, bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .String do return "", false
	return _peek_str(loader.bytes, entry.value_offset)
}

// Read array metadata. Use the returned `payload_offset` plus elem_type
// stride to walk the elements yourself, or use a helper accessor.
@(require_results)
get_array_meta :: proc(loader: Loader, key: string) -> (elem_type: Value_Type, count: int, payload_offset: int, ok: bool) {
	entry, found := loader.kv[key]
	if !found || entry.type != .Array do return {}, 0, 0, false
	return entry.array_type, entry.array_count, entry.value_offset + 12, true
}

// Look up the i-th string in a string array. O(i) — scans from the start
// each time. Use sparingly or cache results in the caller.
@(require_results)
get_array_str :: proc(loader: Loader, key: string, index: int) -> (string, bool) {
	elem_type, count, base, ok := get_array_meta(loader, key)
	if !ok || elem_type != .String || index < 0 || index >= count do return "", false
	off := base
	for _ in 0 ..< index {
		s, s_ok := _peek_str(loader.bytes, off)
		if !s_ok do return "", false
		off += 8 + len(s)
	}
	return _peek_str(loader.bytes, off)
}

// --- internals -------------------------------------------------------------

Reader :: struct {
	bytes:  []byte,
	offset: int,
}

@(require_results)
_read_u32 :: proc(r: ^Reader) -> (u32, bool) {
	if r.offset + 4 > len(r.bytes) do return 0, false
	v := (^u32le)(raw_data(r.bytes[r.offset:]))^
	r.offset += 4
	return u32(v), true
}

@(require_results)
_read_u64 :: proc(r: ^Reader) -> (u64, bool) {
	if r.offset + 8 > len(r.bytes) do return 0, false
	v := (^u64le)(raw_data(r.bytes[r.offset:]))^
	r.offset += 8
	return u64(v), true
}

// Returns a slice that aliases the file bytes (no copy).
@(require_results)
_read_str :: proc(r: ^Reader, file_bytes: []byte) -> (string, bool) {
	n, n_ok := _read_u64(r)
	if !n_ok do return "", false
	end := r.offset + int(n)
	if end > len(r.bytes) do return "", false
	s := string(file_bytes[r.offset:end])
	r.offset = end
	return s, true
}

@(require_results)
_peek_str :: proc(bytes: []byte, off: int) -> (string, bool) {
	if off + 8 > len(bytes) do return "", false
	n := int((^u64le)(raw_data(bytes[off:]))^)
	if off + 8 + n > len(bytes) do return "", false
	return string(bytes[off + 8 : off + 8 + n]), true
}

@(require_results)
_skip_scalar_payload :: proc(r: ^Reader, ty: Value_Type) -> bool {
	bytes_to_skip := 0
	#partial switch ty {
	case .U8, .I8, .Bool:           bytes_to_skip = 1
	case .U16, .I16:                bytes_to_skip = 2
	case .U32, .I32, .F32:          bytes_to_skip = 4
	case .U64, .I64, .F64:          bytes_to_skip = 8
	case .String:
		n, n_ok := _read_u64(r)
		if !n_ok do return false
		bytes_to_skip = int(n)
	case:
		return false
	}
	if r.offset + bytes_to_skip > len(r.bytes) do return false
	r.offset += bytes_to_skip
	return true
}

@(require_results)
_skip_array_payload :: proc(r: ^Reader, elem_type: Value_Type, count: int) -> bool {
	#partial switch elem_type {
	case .U8, .I8, .Bool:
		need := count
		if r.offset + need > len(r.bytes) do return false
		r.offset += need
	case .U16, .I16:
		need := count * 2
		if r.offset + need > len(r.bytes) do return false
		r.offset += need
	case .U32, .I32, .F32:
		need := count * 4
		if r.offset + need > len(r.bytes) do return false
		r.offset += need
	case .U64, .I64, .F64:
		need := count * 8
		if r.offset + need > len(r.bytes) do return false
		r.offset += need
	case .String:
		for _ in 0 ..< count {
			n, n_ok := _read_u64(r)
			if !n_ok do return false
			if r.offset + int(n) > len(r.bytes) do return false
			r.offset += int(n)
		}
	case:
		return false
	}
	return true
}

@(require_results)
_byte_count :: proc(ty: Tensor_Type, element_count: int) -> (int, bool) {
	#partial switch ty {
	case .F32:  return element_count * 4, true
	case .F16:  return element_count * 2, true
	case .BF16: return element_count * 2, true
	case .Q4_0: return _block_bytes(element_count, 32, 18)
	case .Q4_1: return _block_bytes(element_count, 32, 20)
	case .Q5_0: return _block_bytes(element_count, 32, 22)
	case .Q5_1: return _block_bytes(element_count, 32, 24)
	case .Q8_0: return _block_bytes(element_count, 32, 34)
	case .Q8_1: return _block_bytes(element_count, 32, 40)
	case .Q2_K: return _block_bytes(element_count, 256, 84)
	case .Q3_K: return _block_bytes(element_count, 256, 110)
	case .Q4_K: return _block_bytes(element_count, 256, 144)
	case .Q5_K: return _block_bytes(element_count, 256, 176)
	case .Q6_K: return _block_bytes(element_count, 256, 210)
	case .Q8_K: return _block_bytes(element_count, 256, 292)
	}
	return 0, false
}

@(require_results)
_block_bytes :: proc(element_count, block_elements, block_bytes: int) -> (int, bool) {
	if element_count % block_elements != 0 do return 0, false
	return (element_count / block_elements) * block_bytes, true
}

_destroy_partial :: proc(file_bytes: []byte, kv: map[string]KV, tensors: map[string]Tensor_Info) {
	if tensors != nil {
		for _, info in tensors {
			delete(info.shape)
		}
		t := tensors
		delete(t)
	}
	if kv != nil {
		k := kv
		delete(k)
	}
	delete(file_bytes)
}