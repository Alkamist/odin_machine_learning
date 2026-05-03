package main

import "base:builtin"

import "core:fmt"
import "core:mem"
import "core:os"

import ml    "../../"
import llama "../../networks/llama"

CHECKPOINT_MAGIC   :: u32le(0xA0D10AC0)
CHECKPOINT_VERSION :: u32le(1)

save_checkpoint :: proc(path: string, model: llama.Llama) -> bool {
	cfg := model.config

	tied_flag    := u32le(1) if cfg.tied_embeddings else u32le(0)
	qk_norm_flag := u32le(1) if cfg.use_qk_norm     else u32le(0)
	rope_bits    := transmute(u32)cfg.rope_base

	header := [12]u32le{
		CHECKPOINT_MAGIC,
		CHECKPOINT_VERSION,
		u32le(cfg.layer_count),
		u32le(cfg.n_q_heads),
		u32le(cfg.n_kv_heads),
		u32le(cfg.head_size),
		u32le(cfg.embedding_size),
		u32le(cfg.intermediate_size),
		u32le(cfg.vocabulary_size),
		u32le(rope_bits),
		tied_flag,
		qk_norm_flag,
	}

	total_floats := tensor_float_count(cfg)
	total_bytes  := builtin.len(header) * 4 + total_floats * 4

	blob := make([]byte, total_bytes, context.temp_allocator)
	copy(blob[:builtin.len(header) * 4], mem.slice_to_bytes(header[:]))

	cursor := builtin.len(header) * 4
	for_each_tensor(model, proc(t: ml.Tensor, user: rawptr) {
		ctx := (^Pack_Ctx)(user)
		byte_count := t.count * 4
		ml.get_data_bytes(t, ctx.blob[ctx.cursor : ctx.cursor + byte_count])
		ctx.cursor += byte_count
	}, &Pack_Ctx{blob = blob, cursor = cursor})

	if err := os.write_entire_file(path, blob); err != nil {
		fmt.eprintfln("save_checkpoint: failed to write %v: %v", path, err)
		return false
	}
	return true
}

load_checkpoint :: proc(path: string, model: llama.Llama) -> bool {
	bytes, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		fmt.eprintfln("load_checkpoint: could not read %v: %v", path, err)
		return false
	}
	defer delete(bytes)

	header_bytes := 12 * 4
	if builtin.len(bytes) < header_bytes {
		fmt.eprintfln("load_checkpoint: %v too small for header", path)
		return false
	}

	header := (^[12]u32le)(raw_data(bytes))^

	if u32le(header[0]) != CHECKPOINT_MAGIC {
		fmt.eprintfln("load_checkpoint: bad magic in %v", path)
		return false
	}
	if u32le(header[1]) != CHECKPOINT_VERSION {
		fmt.eprintfln("load_checkpoint: version %v unsupported", u32(header[1]))
		return false
	}

	cfg := model.config
	if int(header[2])  != cfg.layer_count       ||
	   int(header[3])  != cfg.n_q_heads         ||
	   int(header[4])  != cfg.n_kv_heads        ||
	   int(header[5])  != cfg.head_size         ||
	   int(header[6])  != cfg.embedding_size    ||
	   int(header[7])  != cfg.intermediate_size ||
	   int(header[8])  != cfg.vocabulary_size {
		fmt.eprintfln("load_checkpoint: config mismatch with %v", path)
		return false
	}
	tied_flag    := u32le(1) if cfg.tied_embeddings else u32le(0)
	qk_norm_flag := u32le(1) if cfg.use_qk_norm     else u32le(0)
	if header[10] != tied_flag || header[11] != qk_norm_flag {
		fmt.eprintfln("load_checkpoint: tied/qk_norm flags mismatch")
		return false
	}

	expected := header_bytes + tensor_float_count(cfg) * 4
	if builtin.len(bytes) != expected {
		fmt.eprintfln("load_checkpoint: %v has %v bytes, expected %v", path, builtin.len(bytes), expected)
		return false
	}

	cursor := header_bytes
	for_each_tensor(model, proc(t: ml.Tensor, user: rawptr) {
		ctx := (^Pack_Ctx)(user)
		byte_count := t.count * 4
		ml.set_data_bytes(t, ctx.blob[ctx.cursor : ctx.cursor + byte_count])
		ctx.cursor += byte_count
	}, &Pack_Ctx{blob = bytes, cursor = cursor})

	return true
}

Pack_Ctx :: struct {
	blob:   []byte,
	cursor: int,
}

for_each_tensor :: proc(model: llama.Llama, fn: proc(t: ml.Tensor, user: rawptr), user: rawptr) {
	fn(model.token_embeddings, user)
	for layer in model.layers {
		fn(layer.input_norm_weight,     user)
		fn(layer.q_proj_weight,         user)
		fn(layer.k_proj_weight,         user)
		fn(layer.v_proj_weight,         user)
		fn(layer.o_proj_weight,         user)
		if model.config.use_qk_norm {
			fn(layer.q_norm_weight, user)
			fn(layer.k_norm_weight, user)
		}
		fn(layer.post_attn_norm_weight, user)
		fn(layer.gate_proj_weight,      user)
		fn(layer.up_proj_weight,        user)
		fn(layer.down_proj_weight,      user)
	}
	fn(model.output_norm_weight, user)
	if !model.config.tied_embeddings {
		fn(model.lm_head_weight, user)
	}
}

tensor_float_count :: proc(c: llama.Config) -> int {
	q_size  := c.n_q_heads  * c.head_size
	kv_size := c.n_kv_heads * c.head_size

	per_layer :=
		c.embedding_size +
		q_size  * c.embedding_size +
		kv_size * c.embedding_size +
		kv_size * c.embedding_size +
		c.embedding_size * q_size +
		c.embedding_size +
		c.intermediate_size * c.embedding_size +
		c.intermediate_size * c.embedding_size +
		c.embedding_size * c.intermediate_size

	if c.use_qk_norm {
		per_layer += c.head_size * 2
	}

	total := c.vocabulary_size * c.embedding_size + per_layer * c.layer_count + c.embedding_size
	if !c.tied_embeddings {
		total += c.vocabulary_size * c.embedding_size
	}
	return total
}
