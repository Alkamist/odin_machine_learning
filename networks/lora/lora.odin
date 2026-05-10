package machine_learning_network_lora

import "base:builtin"

import "core:fmt"
import "core:mem"
import "core:os"

import ml "../../"

// Low-Rank Adaptation: y = base + scale * B @ (A @ x), where A is small
// and B is small (rank << in_features and rank << out_features).
//
// Forward cost: two skinny matmuls instead of one big one. Backward
// gradient flows only into A and B — the base weight stays frozen, which
// is what unlocks QLoRA (base in 4-bit, adapters in bf16).
Adapter :: struct {
	a:     ml.Tensor, // [rank, in_features]
	b:     ml.Tensor, // [out_features, rank]
	scale: ml.Tensor, // scalar, alpha / rank, baked once at make time

	rank:        int,
	in_features:  int,
	out_features: int,
}

@(require_results)
make :: proc(in_features, out_features, rank: int, alpha: f32, dtype: ml.Data_Type = .Bf16) -> (adapter: Adapter) {
	adapter.rank         = rank
	adapter.in_features  = in_features
	adapter.out_features = out_features

	adapter.a = ml.alloc(dtype, {rank, in_features},  persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
	adapter.b = ml.alloc(dtype, {out_features, rank}, persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)

	// scale is a frozen constant: persistent Data buffer only, no gradient.
	// ml.mul backward skips the b-side when b has no gradient buffer.
	adapter.scale = ml.alloc(dtype, {1}, persistent=true, buffers={.Data})
	switch dtype {
	case .F32:
		v := [1]f32{alpha / f32(rank)}
		ml.set_data_bytes(adapter.scale, mem.slice_to_bytes(v[:]))
	case .Bf16:
		v := [1]ml.Bf16{ml.bf16_from_f32(alpha / f32(rank))}
		ml.set_data_bytes(adapter.scale, mem.slice_to_bytes(v[:]))
	case .Q4_K, .Q6_K:
		panic("lora.make: quantized scale dtype not supported")
	}
	return
}

destroy :: proc(adapter: Adapter) {
	ml.destroy(adapter.a)
	ml.destroy(adapter.b)
	ml.destroy(adapter.scale)
}

// Standard QLoRA init: A ~ N(0, sigma) so the input through A is non-zero,
// B = 0 so the adapter contribution starts at zero. The model behaves
// identically to the frozen base at step 0; LoRA learns from there.
randomize :: proc(adapter: Adapter, sigma: f32 = 0.02) {
	ml.fill_normal(adapter.a, 0, sigma)
	ml.fill_value (adapter.b, 0)
}

// Augment a base linear output with the adapter contribution.
// `base_output` is whatever the frozen base linear produced; this returns
// `base_output + scale * B @ (A @ input)`.
@(require_results)
apply :: proc(input, base_output: ml.Tensor, adapter: Adapter) -> ml.Tensor {
	a_out  := ml.linear(input, adapter.a)  // [tokens, rank]
	b_out  := ml.linear(a_out, adapter.b)  // [tokens, out_features]
	scaled := ml.mul(b_out, adapter.scale)
	return ml.add(base_output, scaled)
}

update :: proc(opt: ml.Optimizer, adapter: Adapter) {
	ml.update(opt, adapter.a)
	ml.update(opt, adapter.b)
	// scale is a constant; no update.
}

// Element count for parameter accounting / progress reporting.
@(require_results)
parameter_count :: proc(adapter: Adapter) -> int {
	return adapter.rank * adapter.in_features + adapter.out_features * adapter.rank
}

// Adapter binary file scanned for shape probing or weight loading. Format
// (LE) — same as written by save_adapters in examples/reascript_qlora:
//   magic "LORA0001" (8 bytes)
//   layer_count (i32)
//   per layer × 7 slots {Q, K, V, O, Gate, Up, Down}:
//     i32 rank, i32 in_features, i32 out_features
//     bf16 a_bytes (rank * in_features * 2)
//     bf16 b_bytes (out_features * rank * 2)
//   rank == 0 means slot is unused; no weight bytes follow.
File :: struct {
	bytes:       []byte,
	layer_count: int,
	// One Slot_Header per (layer, slot). Slot order matches Slot below.
	slots:       []Slot_Header,
}

Slot :: enum {
	Q, K, V, O, Gate, Up, Down,
}
SLOT_COUNT :: 7

Slot_Header :: struct {
	rank:         int,
	in_features:  int,
	out_features: int,
	data_offset:  int, // offset into File.bytes where (a, b) payload begins; 0 if unused
}

@(require_results)
file_open :: proc(path: string, allocator := context.allocator) -> (file: File, ok: bool) {
	context.allocator = allocator

	bytes, err := os.read_entire_file_from_path(path, allocator)
	if err != nil {
		fmt.eprintfln("lora.file_open: could not read %v: %v", path, err)
		return {}, false
	}

	if builtin.len(bytes) < 12 || string(bytes[:8]) != "LORA0001" {
		fmt.eprintfln("lora.file_open: %v is not a LORA0001 file", path)
		delete(bytes)
		return {}, false
	}

	layer_count := int((^i32)(&bytes[8])^)
	slots := builtin.make([]Slot_Header, layer_count * SLOT_COUNT)

	cursor := 12
	for layer_idx in 0 ..< layer_count {
		for slot_idx in 0 ..< SLOT_COUNT {
			if cursor + 12 > builtin.len(bytes) {
				fmt.eprintfln("lora.file_open: truncated at layer %v slot %v header", layer_idx, slot_idx)
				delete(bytes); delete(slots)
				return {}, false
			}
			rank := int((^i32)(&bytes[cursor])^);     cursor += 4
			in_f := int((^i32)(&bytes[cursor])^);     cursor += 4
			out_f := int((^i32)(&bytes[cursor])^);    cursor += 4

			h := &slots[layer_idx * SLOT_COUNT + slot_idx]
			h.rank         = rank
			h.in_features  = in_f
			h.out_features = out_f
			if rank == 0 {
				continue
			}
			a_bytes := rank * in_f * 2
			b_bytes := out_f * rank * 2
			if cursor + a_bytes + b_bytes > builtin.len(bytes) {
				fmt.eprintfln("lora.file_open: truncated at layer %v slot %v payload", layer_idx, slot_idx)
				delete(bytes); delete(slots)
				return {}, false
			}
			h.data_offset = cursor
			cursor += a_bytes + b_bytes
		}
	}

	file.bytes       = bytes
	file.layer_count = layer_count
	file.slots       = slots
	return file, true
}

file_destroy :: proc(file: File) {
	delete(file.bytes)
	delete(file.slots)
}

@(require_results)
file_slot :: proc(file: File, layer_idx: int, slot: Slot) -> Slot_Header {
	return file.slots[layer_idx * SLOT_COUNT + int(slot)]
}

// Writes the saved A/B bytes for `slot` of `layer_idx` into the live adapter.
// The adapter must already be allocated with a matching rank/in/out.
load_into :: proc(file: File, layer_idx: int, slot: Slot, adapter: Adapter) -> bool {
	header := file_slot(file, layer_idx, slot)
	if header.rank == 0 {
		return adapter.rank == 0
	}
	if adapter.rank != header.rank ||
	   adapter.in_features  != header.in_features ||
	   adapter.out_features != header.out_features {
		fmt.eprintfln(
			"lora.load_into: layer %v slot %v shape mismatch (file rank=%v in=%v out=%v vs adapter rank=%v in=%v out=%v)",
			layer_idx, slot, header.rank, header.in_features, header.out_features,
			adapter.rank, adapter.in_features, adapter.out_features,
		)
		return false
	}

	offset := header.data_offset
	a_bytes := header.rank * header.in_features  * 2
	b_bytes := header.out_features * header.rank * 2

	ml.set_data_bytes(adapter.a, file.bytes[offset            : offset + a_bytes])
	ml.set_data_bytes(adapter.b, file.bytes[offset + a_bytes  : offset + a_bytes + b_bytes])
	return true
}
