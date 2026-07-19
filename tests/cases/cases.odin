package ml_test_cases

import "core:math/rand"

import ml "../.."

MAX_INPUTS :: 8
MAX_CASES  :: 64

Prepare_Proc :: proc(inputs: [][]f32)
Run_Proc     :: proc(t: []ml.Tensor) -> ml.Tensor

Input_Spec :: struct {
	shape: [ml.MAX_TENSOR_RANK]int,
	rank:  int,
	check: bool,
}

Op_Test :: struct {
	name:        string,
	kind:        ml.Operation_Kind,
	inputs:      [MAX_INPUTS]Input_Spec,
	input_count: int,
	run:         Run_Proc,
	prepare:     Prepare_Proc,
	tol:         f64,
	seed:        u64,
}

sp :: proc(check: bool, dims: ..int) -> (s: Input_Spec) {
	s.check = check
	s.rank  = len(dims)
	for d, i in dims {
		s.shape[i] = d
	}
	return
}

_cases: [MAX_CASES]Op_Test
_count: int
_built: bool

_add :: proc(c: Op_Test) {
	_cases[_count] = c
	_count += 1
}

get :: proc() -> []Op_Test {
	if !_built {
		_build()
		_built = true
	}
	return _cases[:_count]
}

_build :: proc() {
	_add({name = "add",                kind = .Add,                input_count = 2, inputs = {0 = sp(true, 2, 3), 1 = sp(true, 3)},       run = run_add,            prepare = prep_normal,  tol = 0.01,  seed = 1})
	_add({name = "sub",                kind = .Sub,                input_count = 2, inputs = {0 = sp(true, 2, 3), 1 = sp(true, 3)},       run = run_sub,            prepare = prep_normal,  tol = 0.01,  seed = 2})
	_add({name = "mul",                kind = .Mul,                input_count = 2, inputs = {0 = sp(true, 2, 3), 1 = sp(true, 3)},       run = run_mul,            prepare = prep_normal,  tol = 0.015, seed = 3})
	_add({name = "div",                kind = .Div,                input_count = 2, inputs = {0 = sp(true, 2, 3), 1 = sp(true, 3)},       run = run_div,            prepare = prep_div,     tol = 0.02,  seed = 4})
	_add({name = "min",                kind = .Min,                input_count = 2, inputs = {0 = sp(true, 3, 4), 1 = sp(true, 3, 4)},    run = run_min,            prepare = prep_minmax,  tol = 0.01,  seed = 5})
	_add({name = "max",                kind = .Max,                input_count = 2, inputs = {0 = sp(true, 3, 4), 1 = sp(true, 3, 4)},    run = run_max,            prepare = prep_minmax,  tol = 0.01,  seed = 6})
	_add({name = "exp",                kind = .Exp,                input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_exp,            prepare = prep_exp,     tol = 0.02,  seed = 7})
	_add({name = "sqrt",               kind = .Sqrt,               input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_sqrt,           prepare = prep_sqrt,    tol = 0.02,  seed = 8})
	_add({name = "clamp",              kind = .Clamp,              input_count = 1, inputs = {0 = sp(true, 4, 5)},                    run = run_clamp,          prepare = prep_clamp,   tol = 0.01,  seed = 9})
	_add({name = "mean",               kind = .Mean,               input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_mean,           prepare = prep_normal,  tol = 0.01,  seed = 10})
	_add({name = "transpose",          kind = .Transpose,          input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_transpose,      prepare = prep_normal,  tol = 0.01,  seed = 11})
	_add({name = "select",             kind = .Select,             input_count = 1, inputs = {0 = sp(true, 4, 3)},                    run = run_select,         prepare = prep_normal,  tol = 0.01,  seed = 12})
	_add({name = "slice",              kind = .Slice,              input_count = 1, inputs = {0 = sp(true, 8)},                       run = run_slice,          prepare = prep_normal,  tol = 0.01,  seed = 13})
	_add({name = "slice_trailing",     kind = .Slice_Trailing,     input_count = 1, inputs = {0 = sp(true, 3, 5)},                    run = run_slice_trailing, prepare = prep_normal,  tol = 0.01,  seed = 14})
	_add({name = "slice_leading",      kind = .Slice_Leading,      input_count = 1, inputs = {0 = sp(true, 5, 3)},                    run = run_slice_leading,  prepare = prep_normal,  tol = 0.01,  seed = 14})
	_add({name = "concat",             kind = .Concat,             input_count = 2, inputs = {0 = sp(true, 2, 3), 1 = sp(true, 2, 2)},    run = run_concat,         prepare = prep_normal,  tol = 0.01,  seed = 15})
	_add({name = "linear",             kind = .Linear,             input_count = 2, inputs = {0 = sp(true, 3, 4), 1 = sp(true, 5, 4)},    run = run_linear,         prepare = prep_normal,  tol = 0.015, seed = 16})
	_add({name = "rope",               kind = .Rope,               input_count = 1, inputs = {0 = sp(true, 2, 8)},                    run = run_rope,           prepare = prep_normal,  tol = 0.01,  seed = 17})
	_add({name = "layernorm",          kind = .Layernorm,          input_count = 2, inputs = {0 = sp(true, 3, 4), 1 = sp(true, 4)},       run = run_layernorm,      prepare = prep_normal,  tol = 0.03,  seed = 18})
	_add({name = "rmsnorm",            kind = .Rmsnorm,            input_count = 2, inputs = {0 = sp(true, 3, 4), 1 = sp(true, 4)},       run = run_rmsnorm,        prepare = prep_normal,  tol = 0.03,  seed = 19})
	_add({name = "softmax",            kind = .Softmax,            input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_softmax,        prepare = prep_normal,  tol = 0.02,  seed = 20})
	_add({name = "entropy",            kind = .Entropy,            input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_entropy,        prepare = prep_entropy, tol = 0.02,  seed = 21})
	_add({name = "log_softmax",        kind = .Log_Softmax,        input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_log_softmax,    prepare = prep_normal,  tol = 0.02,  seed = 22})
	_add({name = "mean_squared_error", kind = .Mean_Squared_Error, input_count = 2, inputs = {0 = sp(true, 3, 4), 1 = sp(false, 3, 4)},   run = run_mse,            prepare = prep_loss,    tol = 0.02,  seed = 23})
	_add({name = "smooth_l1",          kind = .Smooth_L1,          input_count = 2, inputs = {0 = sp(true, 3, 4), 1 = sp(false, 3, 4)},   run = run_smooth_l1,      prepare = prep_loss,    tol = 0.02,  seed = 24})
	_add({name = "cross_entropy",      kind = .Cross_Entropy,      input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_cross_entropy,  prepare = prep_normal,  tol = 0.02,  seed = 25})
	_add({name = "relu",               kind = .Relu,               input_count = 1, inputs = {0 = sp(true, 4, 5)},                    run = run_relu,           prepare = prep_relu,    tol = 0.01,  seed = 26})
	_add({name = "sigmoid",            kind = .Sigmoid,            input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_sigmoid,        prepare = prep_normal,  tol = 0.02,  seed = 27})
	_add({name = "gelu",               kind = .Gelu,               input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_gelu,           prepare = prep_normal,  tol = 0.02,  seed = 28})
	_add({name = "silu",               kind = .Silu,               input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_silu,           prepare = prep_normal,  tol = 0.02,  seed = 29})
	_add({name = "tanh",               kind = .Tanh,               input_count = 1, inputs = {0 = sp(true, 3, 4)},                    run = run_tanh,           prepare = prep_normal,  tol = 0.02,  seed = 30})
	_add({name = "batched_matmul",     kind = .Batched_Matmul,     input_count = 2, inputs = {0 = sp(true, 2, 3, 4), 1 = sp(true, 2, 4, 5)}, run = run_batched_matmul, prepare = prep_normal, tol = 0.015, seed = 31})
	_add({name = "permute",            kind = .Permute,            input_count = 1, inputs = {0 = sp(true, 2, 3, 4)},                 run = run_permute,        prepare = prep_normal,  tol = 0.01,  seed = 32})
	_add({name = "causal_mask",        kind = .Causal_Mask,        input_count = 1, inputs = {0 = sp(true, 4, 4)},                    run = run_causal_mask,    prepare = prep_normal,  tol = 0.02,  seed = 33})
	_add({name = "attention",          kind = .Attention,          input_count = 3, inputs = {0 = sp(true, 3, 4), 1 = sp(true, 3, 4), 2 = sp(true, 3, 4)}, run = run_attention, prepare = prep_small, tol = 0.03, seed = 34})
}

prep_normal :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			value = rand.float32_normal(0, 0.8)
		}
	}
}

prep_small :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			value = rand.float32_normal(0, 0.4)
		}
	}
}

prep_exp :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			value = rand.float32_range(-1, 1)
		}
	}
}

prep_sqrt :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			value = rand.float32_range(0.3, 2.0)
		}
	}
}

prep_div :: proc(inputs: [][]f32) {
	for &value in inputs[0] {
		value = rand.float32_normal(0, 0.8)
	}
	for &value in inputs[1] {
		magnitude := rand.float32_range(0.5, 1.5)
		value      = rand.float32() < 0.5 ? -magnitude : magnitude
	}
}

prep_clamp :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			x := rand.float32_range(-1.5, 1.5)
			if abs(x - 0.5) < 0.12 {
				x = 0.75
			}
			if abs(x + 0.5) < 0.12 {
				x = -0.75
			}
			value = x
		}
	}
}

prep_minmax :: proc(inputs: [][]f32) {
	a := inputs[0]
	b := inputs[1]
	for i in 0 ..< len(a) {
		a[i] = rand.float32_normal(0, 0.8)
		offset := rand.float32_range(0.3, 1.0)
		b[i]   = rand.float32() < 0.5 ? a[i] - offset : a[i] + offset
	}
}

prep_entropy :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			value = rand.float32_range(0.1, 1.0)
		}
	}
}

prep_loss :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			value = rand.float32_range(-0.4, 0.4)
		}
	}
}

prep_relu :: proc(inputs: [][]f32) {
	for input in inputs {
		for &value in input {
			x := rand.float32_normal(0, 0.8)
			if abs(x) < 0.1 {
				x = x >= 0 ? 0.3 : -0.3
			}
			value = x
		}
	}
}

run_add            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.add(t[0], t[1]) }
run_sub            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.sub(t[0], t[1]) }
run_mul            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.mul(t[0], t[1]) }
run_div            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.div(t[0], t[1]) }
run_min            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.min(t[0], t[1]) }
run_max            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.max(t[0], t[1]) }
run_exp            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.exp(t[0]) }
run_sqrt           :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.sqrt(t[0]) }
run_clamp          :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.clamp(t[0], -0.5, 0.5) }
run_mean           :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.mean(t[0]) }
run_transpose      :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.transpose(t[0]) }
run_select         :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.select(t[0], {2, 0, 2, 1}) }
run_slice          :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.slice(t[0], 2, 6) }
run_slice_trailing :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.slice_trailing(t[0], 1, 4) }
run_slice_leading  :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.slice_leading(t[0], 1, 4) }
run_concat         :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.concat(t[0], t[1]) }
run_linear         :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.linear(t[0], t[1]) }
run_rope           :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.rope(t[0], 2) }
run_layernorm      :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.layernorm(t[0], t[1]) }
run_rmsnorm        :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.rmsnorm(t[0], t[1]) }
run_softmax        :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.softmax(t[0]) }
run_entropy        :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.entropy(t[0]) }
run_log_softmax    :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.log_softmax(t[0]) }
run_mse            :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.mean_squared_error(t[0], t[1]) }
run_smooth_l1      :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.smooth_l1(t[0], t[1]) }
run_cross_entropy  :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.cross_entropy(t[0], {2, 0, 3}) }
run_relu           :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.relu(t[0]) }
run_sigmoid        :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.sigmoid(t[0]) }
run_gelu           :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.gelu(t[0]) }
run_silu           :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.silu(t[0]) }
run_tanh           :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.tanh(t[0]) }
run_batched_matmul :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.batched_matmul(t[0], t[1]) }
run_permute        :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.permute(t[0], {1, 0, 2}) }
run_causal_mask    :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.exp(ml.causal_mask(t[0])) }
run_attention      :: proc(t: []ml.Tensor) -> ml.Tensor { return ml.attention(t[0], t[1], t[2], 2) }
