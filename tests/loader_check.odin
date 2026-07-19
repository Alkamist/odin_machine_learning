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

_expect_load_fails :: proc(t: ^testing.T, path: string, bytes: []u8, label: string, is_gguf: bool, loc := #caller_location) {
	write_err := os.write_entire_file_from_bytes(path, bytes)
	testing.expectf(t, write_err == nil, "%s: writing test file failed: %v", label, write_err, loc=loc)
	context.logger = log.nil_logger()
	if is_gguf {
		loader, ok := gguf.load(path)
		if !testing.expectf(t, !ok, "%s: gguf load should fail", label, loc=loc) {
			gguf.destroy(loader)
		}
	} else {
		loader, ok := safetensors.load(path)
		if !testing.expectf(t, !ok, "%s: safetensors load should fail", label, loc=loc) {
			safetensors.destroy(loader)
		}
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
	loader, ok := gguf.load(GGUF_TEST_PATH)
	testing.expect(t, ok, "valid gguf should load")
	if ok {
		data, data_ok := gguf.get_bytes(loader, "w")
		testing.expect(t, data_ok, "tensor w should exist")
		testing.expect_value(t, len(data), 24)
		gguf.destroy(loader)
	}

	for cut in 0 ..< len(valid) {
		_expect_load_fails(t, GGUF_TEST_PATH, valid[:cut], "gguf truncation", is_gguf=true)
	}

	corrupt := make([dynamic]u8, 0, len(valid))
	defer delete(corrupt)

	reset :: proc(dst: ^[dynamic]u8, src: []u8) {
		clear(dst)
		append(dst, ..src)
	}

	reset(&corrupt, valid[:])
	corrupt[0] = 'X'
	_expect_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf bad magic", is_gguf=true)

	reset(&corrupt, valid[:])
	corrupt[layout.version] = 2
	_expect_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf unsupported version", is_gguf=true)

	reset(&corrupt, valid[:])
	for i in 0 ..< 8 {
		corrupt[layout.name_len + i] = 0xFF
	}
	_expect_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf hostile name length", is_gguf=true)

	reset(&corrupt, valid[:])
	for i in 0 ..< 8 {
		corrupt[layout.dim0 + i] = 0xFF
	}
	_expect_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf huge dimension", is_gguf=true)

	reset(&corrupt, valid[:])
	for i in 0 ..< 8 {
		corrupt[layout.dim0 + i] = 0
	}
	_expect_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf zero dimension", is_gguf=true)

	reset(&corrupt, valid[:])
	corrupt[layout.type_id] = 99
	_expect_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf unsupported tensor type", is_gguf=true)

	reset(&corrupt, valid[:])
	corrupt[layout.type_id] = u8(gguf.Tensor_Type.Q4_K)
	_expect_load_fails(t, GGUF_TEST_PATH, corrupt[:], "gguf quant block mismatch", is_gguf=true)

	duplicate: [dynamic]u8
	defer delete(duplicate)
	_gguf_build(&duplicate, duplicate_tensor=true)
	_expect_load_fails(t, GGUF_TEST_PATH, duplicate[:], "gguf duplicate tensor name", is_gguf=true)
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
	loader, ok := safetensors.load(ST_TEST_PATH)
	testing.expect(t, ok, "valid safetensors should load")
	if ok {
		data, data_ok := safetensors.get_bytes(loader, "w")
		testing.expect(t, data_ok, "tensor w should exist")
		testing.expect_value(t, len(data), 24)
		safetensors.destroy(loader)
	}

	for cut in 0 ..< len(valid) {
		_expect_load_fails(t, ST_TEST_PATH, valid[:cut], "safetensors truncation", is_gguf=false)
	}

	corrupt: [dynamic]u8
	defer delete(corrupt)

	clear(&corrupt)
	append(&corrupt, ..valid[:])
	for i in 0 ..< 8 {
		corrupt[i] = 0xFF
	}
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors huge header length", is_gguf=false)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape"`, 24)
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors malformed json", is_gguf=false)

	clear(&corrupt)
	_st_build(&corrupt, `[1,2,3]`, 24)
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors non-object root", is_gguf=false)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,20]}}`, 24)
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors range vs shape mismatch", is_gguf=false)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[-2,3],"data_offsets":[0,24]}}`, 24)
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors negative dimension", is_gguf=false)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,64]}}`, 24)
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors out-of-bounds offsets", is_gguf=false)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[4611686018427387904,4611686018427387904],"data_offsets":[0,24]}}`, 24)
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors dimension overflow", is_gguf=false)

	clear(&corrupt)
	_st_build(&corrupt, `{"w":{"dtype":"F32","shape":[2,3]}}`, 24)
	_expect_load_fails(t, ST_TEST_PATH, corrupt[:], "safetensors missing data_offsets", is_gguf=false)
}
