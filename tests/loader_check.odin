package ml_tests

import "core:log"
import "core:os"
import "core:testing"

import gguf        "../loaders/gguf"
import safetensors "../loaders/safetensors"

GGUF_TEST_PATH :: "test_loader_robustness.gguf"
ST_TEST_PATH   :: "test_loader_robustness.safetensors"

Gguf_Layout :: struct {
	version:  int,
	name_len: int,
	dim0:     int,
	type_id:  int,
}

_gguf_w_u32 :: proc(buf: ^[dynamic]u8, v: u32) {
	bytes := transmute([4]u8)u32le(v)
	append(buf, ..bytes[:])
}

_gguf_w_u64 :: proc(buf: ^[dynamic]u8, v: u64) {
	bytes := transmute([8]u8)u64le(v)
	append(buf, ..bytes[:])
}

_gguf_w_str :: proc(buf: ^[dynamic]u8, s: string) {
	_gguf_w_u64(buf, u64(len(s)))
	append(buf, s)
}

_gguf_build :: proc(buf: ^[dynamic]u8, duplicate_tensor := false) -> (layout: Gguf_Layout) {
	_gguf_w_u32(buf, gguf.GGUF_MAGIC)
	layout.version = len(buf)
	_gguf_w_u32(buf, 3)
	_gguf_w_u64(buf, duplicate_tensor ? 2 : 1)
	_gguf_w_u64(buf, 0)

	entry_count := duplicate_tensor ? 2 : 1
	for _ in 0 ..< entry_count {
		layout.name_len = len(buf)
		_gguf_w_str(buf, "w")
		_gguf_w_u32(buf, 2)
		layout.dim0 = len(buf)
		_gguf_w_u64(buf, 2)
		_gguf_w_u64(buf, 3)
		layout.type_id = len(buf)
		_gguf_w_u32(buf, u32(gguf.Tensor_Type.F32))
		_gguf_w_u64(buf, 0)
	}

	for len(buf) % gguf.GGUF_DEFAULT_ALIGNMENT != 0 {
		append(buf, 0)
	}
	for i in 0 ..< 6 {
		value := f32(i + 1)
		bytes := transmute([4]u8)value
		append(buf, ..bytes[:])
	}
	return
}

_expect_gguf_load_fails :: proc(t: ^testing.T, path: string, bytes: []u8, label: string, expected: gguf.Error, loc := #caller_location) {
	write_err := os.write_entire_file_from_bytes(path, bytes)
	testing.expectf(t, write_err == nil, "%s: writing test file failed: %v", label, write_err, loc=loc)
	real_logger := context.logger
	context.logger = log.nil_logger()
	loader, err := gguf.load(path)
	context.logger = real_logger
	testing.expectf(t, err == expected, "%s: gguf load expected %v, got %v", label, expected, err, loc=loc)
	if err == .None {
		gguf.destroy(loader)
	}
}

_expect_st_load_fails :: proc(t: ^testing.T, path: string, bytes: []u8, label: string, expected: safetensors.Error, loc := #caller_location) {
	write_err := os.write_entire_file_from_bytes(path, bytes)
	testing.expectf(t, write_err == nil, "%s: writing test file failed: %v", label, write_err, loc=loc)
	real_logger := context.logger
	context.logger = log.nil_logger()
	loader, err := safetensors.load(path)
	context.logger = real_logger
	testing.expectf(t, err == expected, "%s: safetensors load expected %v, got %v", label, expected, err, loc=loc)
	if err == .None {
		safetensors.destroy(loader)
	}
}

@(test)
test_gguf_loader_robustness :: proc(t: ^testing.T) {
	defer os.remove(GGUF_TEST_PATH)

	valid: [dynamic]u8
	defer delete(valid)
	layout := _gguf_build(&valid)

	write_err := os.write_entire_file_from_bytes(GGUF_TEST_PATH, valid[:])
	testing.expectf(t, write_err == nil, "writing valid gguf failed: %v", write_err)
	loader, err := gguf.load(GGUF_TEST_PATH)
	testing.expect_value(t, err, gguf.Error.None)
	if err == .None {
		data, data_ok := gguf.get_bytes(loader, "w")
		testing.expect(t, data_ok, "tensor w should exist")
		testing.expect_value(t, len(data), 24)
		gguf.destroy(loader)
	}

	missing, missing_err := gguf.load("does_not_exist.gguf")
	testing.expect_value(t, missing_err, gguf.Error.Not_Found)
	if missing_err == .None {
		gguf.destroy(missing)
	}

	for cut in 0 ..< len(valid) {
		_expect_gguf_load_fails(t, GGUF_TEST_PATH, valid[:cut], "gguf truncation", .Malformed)
	}

	corrupt := make([dynamic]u8, 0, len(valid))
	defer delete(corrupt)

	reset :: proc(dst: ^[dynamic]u8, src: []u8) {
		clear(dst)
		append(dst, ..src)
	}

	reset(&corrupt, valid[:])
	corrupt[0] = 'X'
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf bad magic", .Malformed)

	reset(&corrupt, valid[:])
	corrupt[layout.version] = 2
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf unsupported version", .Unsupported)

	reset(&corrupt, valid[:])
	for i in 0 ..< 8 {
		corrupt[layout.name_len + i] = 0xFF
	}
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf hostile name length", .Malformed)

	reset(&corrupt, valid[:])
	for i in 0 ..< 8 {
		corrupt[layout.dim0 + i] = 0xFF
	}
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf huge dimension", .Malformed)

	reset(&corrupt, valid[:])
	for i in 0 ..< 8 {
		corrupt[layout.dim0 + i] = 0
	}
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf zero dimension", .Malformed)

	reset(&corrupt, valid[:])
	corrupt[layout.type_id] = 99
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf unsupported tensor type", .Unsupported)

	reset(&corrupt, valid[:])
	corrupt[layout.type_id] = u8(gguf.Tensor_Type.Q4_K)
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf quant block mismatch", .Malformed)

	duplicate: [dynamic]u8
	defer delete(duplicate)
	_gguf_build(&duplicate, duplicate_tensor=true)
	_expect_gguf_load_fails(t, GGUF_TEST_PATH, duplicate[:], "gguf duplicate tensor name", .Malformed)
}

_st_build :: proc(buf: ^[dynamic]u8, header: string, data_bytes: int) {
	_gguf_w_u64(buf, u64(len(header)))
	append(buf, header)
	for i in 0 ..< data_bytes {
		append(buf, u8(i))
	}
}

@(test)
test_safetensors_loader_robustness :: proc(t: ^testing.T) {
	defer os.remove(ST_TEST_PATH)

	VALID_HEADER :: `{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}`

	valid: [dynamic]u8
	defer delete(valid)
	_st_build(&valid, VALID_HEADER, 24)

	write_err := os.write_entire_file_from_bytes(ST_TEST_PATH, valid[:])
	testing.expectf(t, write_err == nil, "writing valid safetensors failed: %v", write_err)
	loader, err := safetensors.load(ST_TEST_PATH)
	testing.expect_value(t, err, safetensors.Error.None)
	if err == .None {
		data, data_ok := safetensors.get_bytes(loader, "w")
		testing.expect(t, data_ok, "tensor w should exist")
		testing.expect_value(t, len(data), 24)
		safetensors.destroy(loader)
	}

	missing, missing_err := safetensors.load("does_not_exist.safetensors")
	testing.expect_value(t, missing_err, safetensors.Error.Not_Found)
	if missing_err == .None {
		safetensors.destroy(missing)
	}

	for cut in 0 ..< len(valid) {
		_expect_st_load_fails(t, ST_TEST_PATH, valid[:cut], "safetensors truncation", .Malformed)
	}

	corrupt: [dynamic]u8
	defer delete(corrupt)

	clear(&corrupt)
	append(&corrupt, ..valid[:])
	for i in 0 ..< 8 {
		corrupt[i] = 0xFF
	}
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors huge header length", .Malformed)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape"`, 24)
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors malformed json", .Malformed)

	clear(&corrupt)
	_st_build(&corrupt, `[1,2,3]`, 24)
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors non-object root", .Malformed)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,20]}}`, 24)
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors range vs shape mismatch", .Malformed)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[-2,3],"data_offsets":[0,24]}}`, 24)
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors negative dimension", .Malformed)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,64]}}`, 24)
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors out-of-bounds offsets", .Malformed)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[4611686018427387904,4611686018427387904],"data_offsets":[0,24]}}`, 24)
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors dimension overflow", .Malformed)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[2,3]}}`, 24)
	_expect_st_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors missing data_offsets", .Malformed)
}
