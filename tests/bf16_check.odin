package ml_tests

import "core:math"
import "core:testing"

import ml "../"

_bf16_bits :: proc(x: f32) -> u16 {
	return u16(ml.bf16_from_f32(x))
}

_bf16_f32_from_bits :: proc(bits: u32) -> f32 {
	return transmute(f32)bits
}

@(test)
test_bf16_round_to_nearest_even :: proc(t: ^testing.T) {
	Case :: struct {
		name:     string,
		in_bits:  u32,
		want_bf:  u16,
	}

	cases := []Case{
		{name="exact_one",              in_bits=0x3f80_0000, want_bf=0x3f80},
		{name="halfway_down_to_even",   in_bits=0x3f80_8000, want_bf=0x3f80},
		{name="halfway_up_to_even",     in_bits=0x3f81_8000, want_bf=0x3f82},
		{name="round_up_over_exponent", in_bits=0x3fff_ffff, want_bf=0x4000},
		{name="neg_halfway_to_even",    in_bits=0xbf80_8000, want_bf=0xbf80},
		{name="neg_halfway_up_to_even", in_bits=0xbf81_8000, want_bf=0xbf82},
		{name="above_half_rounds_up",   in_bits=0x3f80_c000, want_bf=0x3f81},
		{name="below_half_rounds_down", in_bits=0x3f80_4000, want_bf=0x3f80},
	}

	for c in cases {
		x   := _bf16_f32_from_bits(c.in_bits)
		got := _bf16_bits(x)
		testing.expectf(t, got == c.want_bf, "bf16 rte %s: in=0x%08x got=0x%04x want=0x%04x", c.name, c.in_bits, got, c.want_bf)
	}
}

@(test)
test_bf16_nan_stays_nan :: proc(t: ^testing.T) {
	nan_inputs := []u32{0x7fc0_0000, 0xffc0_0000, 0x7f80_0001, 0x7fbf_ffff}
	for bits in nan_inputs {
		x   := _bf16_f32_from_bits(bits)
		bf  := ml.bf16_from_f32(x)
		out := ml.bf16_to_f32(bf)
		testing.expectf(t, out != out, "bf16 nan: in=0x%08x produced non-NaN out=0x%08x", bits, transmute(u32)out)
		testing.expectf(t, !math.is_inf(out), "bf16 nan: in=0x%08x collapsed to infinity", bits)
	}
}

@(test)
test_bf16_infinity_passthrough :: proc(t: ^testing.T) {
	Case :: struct {
		in_bits:  u32,
		want_bf:  u16,
		want_f32: u32,
	}
	cases := []Case{
		{in_bits=0x7f80_0000, want_bf=0x7f80, want_f32=0x7f80_0000},
		{in_bits=0xff80_0000, want_bf=0xff80, want_f32=0xff80_0000},
	}
	for c in cases {
		x   := _bf16_f32_from_bits(c.in_bits)
		bf  := ml.bf16_from_f32(x)
		out := ml.bf16_to_f32(bf)
		testing.expectf(t, u16(bf) == c.want_bf, "bf16 inf: in=0x%08x got bf=0x%04x want=0x%04x", c.in_bits, u16(bf), c.want_bf)
		testing.expectf(t, transmute(u32)out == c.want_f32, "bf16 inf: in=0x%08x round-tripped to 0x%08x want=0x%08x", c.in_bits, transmute(u32)out, c.want_f32)
		testing.expectf(t, math.is_inf(out), "bf16 inf: in=0x%08x is not infinity after round trip", c.in_bits)
	}
}

@(test)
test_bf16_to_f32_exact :: proc(t: ^testing.T) {
	Case :: struct {
		bf:   u16,
		want: u32,
	}
	cases := []Case{
		{bf=0x0000, want=0x0000_0000},
		{bf=0x3f80, want=0x3f80_0000},
		{bf=0x4000, want=0x4000_0000},
		{bf=0xbf00, want=0xbf00_0000},
		{bf=0x42c8, want=0x42c8_0000},
		{bf=0x8000, want=0x8000_0000},
	}
	for c in cases {
		out := ml.bf16_to_f32(ml.Bf16(c.bf))
		testing.expectf(t, transmute(u32)out == c.want, "bf16_to_f32: bf=0x%04x got=0x%08x want=0x%08x", c.bf, transmute(u32)out, c.want)
	}
}

@(test)
test_bf16_exact_f32_round_trip :: proc(t: ^testing.T) {
	values := []f32{0, 1, -1, 2, -0.5, 100, -256, 0.25}
	for v in values {
		round := ml.bf16_to_f32(ml.bf16_from_f32(v))
		testing.expectf(t, round == v, "bf16 round trip: value %v changed to %v", v, round)
	}
}
