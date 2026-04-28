package pytorch_parity_speed_runner

import "base:builtin"

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:time"
import "core:math/rand"

import ml  "../../.."
import cpu "../../../backend_cpu"
import mlp "../../../mlp"

SEED       :: 0xC0FFEE
WARMUP     :: 3
ITERATIONS :: 30

main :: proc() {
	thread_count := 1
	if builtin.len(os.args) >= 2 {
		parsed, ok := strconv.parse_int(os.args[1])
		if ok {
			thread_count = parsed
		}
	}

	ctx := ml.context_create(256 * 1024 * 1024, &cpu.backend)
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)

	cpu.set_thread_count(thread_count)

	rand.reset(SEED)

	bench("linear_fwd",       bench_linear_fwd)
	bench("linear_fwdbwd",    bench_linear_fwdbwd)
	bench("layernorm",        bench_layernorm)
	bench("softmax",          bench_softmax)
	bench("attention_causal", bench_attention)
	bench("mlp_step",         bench_mlp_step)
}

bench :: proc(name: string, run: proc() -> f32) {
	for _ in 0 ..< WARMUP {
		_ = run()
	}

	min_ns:   i64 = max(i64)
	total_ns: i64 = 0
	checksum: f32

	for i in 0 ..< ITERATIONS {
		start := time.tick_now()
		c := run()
		dt := i64(time.tick_since(start))
		if dt < min_ns {
			min_ns = dt
		}
		total_ns += dt
		if i == 0 {
			checksum = c
		}
	}

	min_ms  := f64(min_ns)   / 1_000_000.0
	mean_ms := f64(total_ns) / 1_000_000.0 / f64(ITERATIONS)
	fmt.printfln("%v,%.6f,%.6f,%.6f", name, min_ms, mean_ms, checksum)
}

LINEAR_BATCH  :: 64
LINEAR_INPUT  :: 512
LINEAR_OUTPUT :: 2048

LAYERNORM_BATCH :: 64
LAYERNORM_SIZE  :: 512

SOFTMAX_BATCH :: 64
SOFTMAX_SIZE  :: 1024

ATTN_TOKENS :: 256
ATTN_HEADS  :: 8
ATTN_EMBED  :: 512

MLP_BATCH :: 64
MLP_IN    :: 256
MLP_HID   :: 256
MLP_OUT   :: 64

@(thread_local) _linear_w:    ml.Tensor
@(thread_local) _layernorm_w: ml.Tensor
@(thread_local) _mlp_x:       ml.Tensor
@(thread_local) _mlp_y:       ml.Tensor
@(thread_local) _mlp_model:   mlp.Mlp

bench_linear_fwd :: proc() -> f32 {
	if _linear_w.vtable == nil {
		_linear_w = ml.make({LINEAR_OUTPUT, LINEAR_INPUT})
		ml.fill_normal(_linear_w, 0, 0.02)
	}
	w := _linear_w
	ml.clear()
	x := ml.zeros({LINEAR_BATCH, LINEAR_INPUT})
	ml.fill_value(x, 0.01)
	y := ml.linear(x, w)
	return _checksum(y)
}

bench_linear_fwdbwd :: proc() -> f32 {
	if _linear_w.vtable == nil {
		_linear_w = ml.make({LINEAR_OUTPUT, LINEAR_INPUT})
		ml.fill_normal(_linear_w, 0, 0.02)
	}
	w := _linear_w
	ml.clear()
	x := ml.zeros({LINEAR_BATCH, LINEAR_INPUT})
	ml.fill_value(x, 0.01)
	y := ml.linear(x, w)
	ml.backward()
	return _checksum(y)
}

bench_layernorm :: proc() -> f32 {
	if _layernorm_w.vtable == nil {
		_layernorm_w = ml.make({LAYERNORM_SIZE})
		ml.fill_value(_layernorm_w, 1)
	}
	w := _layernorm_w
	ml.clear()
	x := ml.zeros({LAYERNORM_BATCH, LAYERNORM_SIZE})
	ml.fill_value(x, 0.01)
	y := ml.layernorm(x, w)
	ml.backward()
	return _checksum(y)
}

bench_softmax :: proc() -> f32 {
	ml.clear()
	x := ml.zeros({SOFTMAX_BATCH, SOFTMAX_SIZE})
	ml.fill_value(x, 0.01)
	y := ml.softmax(x)
	ml.backward()
	return _checksum(y)
}

bench_attention :: proc() -> f32 {
	ml.clear()
	x := ml.zeros({ATTN_TOKENS, 3 * ATTN_EMBED})
	ml.fill_value(x, 0.01)
	y := ml.attention(x, ATTN_HEADS, causal=true)
	ml.backward()
	return _checksum(y)
}

bench_mlp_step :: proc() -> f32 {
	if _mlp_model.layers == nil {
		_mlp_model = mlp.make(MLP_IN, MLP_HID, MLP_HID, MLP_OUT)
		_mlp_x     = ml.make({MLP_BATCH, MLP_IN})
		_mlp_y     = ml.make({MLP_BATCH, MLP_OUT})
		ml.fill_value(_mlp_x, 0.01)
		ml.fill_value(_mlp_y, 0.5)
	}
	x := _mlp_x
	y := _mlp_y
	ml.clear()
	pred       := mlp.forward(_mlp_model, x)
	per_sample := ml.mean_squared_error(pred, y)
	loss       := ml.mean(per_sample)

	ml.backward()

	opt: ml.Optimizer
	if ml.optimize(&opt, period=1) {
		mlp.update(opt, _mlp_model)
	}
	return _checksum(loss)
}

_checksum :: proc(t: ml.Tensor) -> f32 {
	s: f32
	for v in cpu.data(t) {
		s += v
	}
	return s
}
