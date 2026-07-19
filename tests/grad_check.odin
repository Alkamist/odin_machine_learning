package ml_tests

import "core:testing"
import "core:math/rand"

import ml    "../"
import cpu   "../backends/cpu"
import cases "cases"

CTX_SIZE :: 8 * 1024 * 1024

H_REL     :: f64(1e-3)
H_MIN     :: f64(1e-3)
REL_FLOOR :: f64(1e-3)

_forward_output_count :: proc(tc: cases.Op_Test, inputs_data: [][]f32) -> int {
	ml.clear(training=true)

	tensors: [cases.MAX_INPUTS]ml.Tensor
	n := len(inputs_data)
	for i in 0 ..< n {
		shape     := tc.inputs[i].shape
		tensors[i] = ml.zeros(.F32, shape[:tc.inputs[i].rank])
		ml.set_data(tensors[i], inputs_data[i])
	}

	output := tc.run(tensors[:n])
	return ml.len(output)
}

_eval :: proc(tc: cases.Op_Test, inputs_data: [][]f32, w: []f32, do_backward: bool, grads_out: [][]f32) -> f64 {
	ml.clear(training=true)

	tensors: [cases.MAX_INPUTS]ml.Tensor
	n := len(inputs_data)
	for i in 0 ..< n {
		shape     := tc.inputs[i].shape
		tensors[i] = ml.zeros(.F32, shape[:tc.inputs[i].rank])
		ml.set_data(tensors[i], inputs_data[i])
	}

	output := tc.run(tensors[:n])
	count  := ml.len(output)

	flat_shape := [1]int{count}
	flat     := ml.reshape(output, flat_shape[:])
	weights  := ml.zeros(.F32, flat_shape[:])
	ml.set_data(weights, w[:count])
	weighted := ml.mul(flat, weights)
	loss_t   := ml.mean(weighted)

	weighted_host := make([]f32, count)
	defer delete(weighted_host)
	ml.get_data(weighted, weighted_host)

	loss: f64
	for value in weighted_host {
		loss += f64(value)
	}
	loss /= f64(count)

	if do_backward {
		ml.backward(loss_t)
		for i in 0 ..< n {
			ml.get_gradient(tensors[i], grads_out[i])
		}
	}

	return loss
}

_run_check :: proc(t: ^testing.T, tc: cases.Op_Test) {
	state := rand.create(tc.seed)
	context.random_generator = rand.default_random_generator(&state)

	n := tc.input_count

	inputs_data: [cases.MAX_INPUTS][]f32
	grads:       [cases.MAX_INPUTS][]f32
	for i in 0 ..< n {
		shape          := tc.inputs[i].shape
		element_count  := ml.shape_element_count(shape[:tc.inputs[i].rank])
		inputs_data[i]  = make([]f32, element_count)
		grads[i]        = make([]f32, element_count)
	}
	defer for i in 0 ..< n {
		delete(inputs_data[i])
		delete(grads[i])
	}

	tc.prepare(inputs_data[:n])

	ctx := cpu.context_create(CTX_SIZE)
	ml.context_begin(ctx)

	output_count := _forward_output_count(tc, inputs_data[:n])

	w := make([]f32, output_count)
	defer delete(w)
	for i in 0 ..< output_count {
		magnitude := rand.float32_range(0.5, 1.5)
		w[i]       = rand.float32() < 0.5 ? -magnitude : magnitude
	}

	_eval(tc, inputs_data[:n], w, true, grads[:n])

	for i in 0 ..< n {
		if !tc.inputs[i].check {
			continue
		}

		element_count := len(inputs_data[i])
		for e in 0 ..< element_count {
			saved := inputs_data[i][e]
			h     := max(H_MIN, H_REL * f64(abs(saved)))

			inputs_data[i][e] = saved + f32(h)
			x_plus     := inputs_data[i][e]
			loss_plus  := _eval(tc, inputs_data[:n], w, false, nil)

			inputs_data[i][e] = saved - f32(h)
			x_minus    := inputs_data[i][e]
			loss_minus := _eval(tc, inputs_data[:n], w, false, nil)

			inputs_data[i][e] = saved

			two_h    := f64(x_plus) - f64(x_minus)
			numeric  := (loss_plus - loss_minus) / two_h
			analytic := f64(grads[i][e])
			denom    := max(max(abs(analytic), abs(numeric)), REL_FLOOR)
			rel      := abs(analytic - numeric) / denom

			testing.expectf(t, rel <= tc.tol,
				"%s: input %d elem %d analytic=%.6g numeric=%.6g rel_err=%.4g (tol=%.3g)",
				tc.name, i, e, analytic, numeric, rel, tc.tol)
		}
	}

	ml.context_end()
	cpu.context_destroy(ctx)
}

@(test)
test_op_gradients :: proc(t: ^testing.T) {
	for tc in cases.get() {
		if tc.parity_only {
			continue
		}
		_run_check(t, tc)
	}
}
