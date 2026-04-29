package machine_learning_network_gru

import "base:builtin"
import "core:slice"

import ml "../../"

Gru :: struct {
	input_size:  int,
	hidden_size: int,

	state: []f32,

	update_input_weight:  ml.Tensor,
	update_hidden_weight: ml.Tensor,
	update_bias:          ml.Tensor,

	reset_input_weight:  ml.Tensor,
	reset_hidden_weight: ml.Tensor,
	reset_bias:          ml.Tensor,

	candidate_input_weight:  ml.Tensor,
	candidate_hidden_weight: ml.Tensor,
	candidate_bias:          ml.Tensor,
}

make :: proc(input_size, hidden_size: int, allocator := context.allocator) -> (gru: Gru) {
	gru.input_size  = input_size
	gru.hidden_size = hidden_size

	gru.state = builtin.make([]f32, hidden_size)

	// Update gate
	gru.update_input_weight  = ml.make(.F32, {hidden_size, input_size})
	gru.update_hidden_weight = ml.make(.F32, {hidden_size, hidden_size})
	gru.update_bias          = ml.make(.F32, {hidden_size})

	// Reset gate
	gru.reset_input_weight  = ml.make(.F32, {hidden_size, input_size})
	gru.reset_hidden_weight = ml.make(.F32, {hidden_size, hidden_size})
	gru.reset_bias          = ml.make(.F32, {hidden_size})

	// Candidate hidden state
	gru.candidate_input_weight  = ml.make(.F32, {hidden_size, input_size})
	gru.candidate_hidden_weight = ml.make(.F32, {hidden_size, hidden_size})
	gru.candidate_bias          = ml.make(.F32, {hidden_size})

	randomize(gru)

	return
}

destroy :: proc(gru: Gru) {
	ml.destroy(gru.update_input_weight)
	ml.destroy(gru.update_hidden_weight)
	ml.destroy(gru.update_bias)

	ml.destroy(gru.reset_input_weight)
	ml.destroy(gru.reset_hidden_weight)
	ml.destroy(gru.reset_bias)

	ml.destroy(gru.candidate_input_weight)
	ml.destroy(gru.candidate_hidden_weight)
	ml.destroy(gru.candidate_bias)

	delete(gru.state)
}

copy :: proc(dst, src: Gru) {
	ml.copy(dst.update_input_weight,  src.update_input_weight)
	ml.copy(dst.update_hidden_weight, src.update_hidden_weight)
	ml.copy(dst.update_bias,          src.update_bias)

	ml.copy(dst.reset_input_weight,  src.reset_input_weight)
	ml.copy(dst.reset_hidden_weight, src.reset_hidden_weight)
	ml.copy(dst.reset_bias,          src.reset_bias)

	ml.copy(dst.candidate_input_weight,  src.candidate_input_weight)
	ml.copy(dst.candidate_hidden_weight, src.candidate_hidden_weight)
	ml.copy(dst.candidate_bias,          src.candidate_bias)

	builtin.copy(dst.state, src.state)
}

randomize :: proc(gru: Gru) {
    ml.xavier_initialization(gru.update_input_weight,  gru.input_size, gru.hidden_size)
	ml.xavier_initialization(gru.update_hidden_weight, gru.hidden_size, gru.hidden_size)
	ml.fill_value(gru.update_bias, 0)

	ml.xavier_initialization(gru.reset_input_weight,  gru.input_size, gru.hidden_size)
	ml.xavier_initialization(gru.reset_hidden_weight, gru.hidden_size, gru.hidden_size)
	ml.fill_value(gru.reset_bias, 0)

	ml.xavier_initialization(gru.candidate_input_weight,  gru.input_size, gru.hidden_size)
	ml.xavier_initialization(gru.candidate_hidden_weight, gru.hidden_size, gru.hidden_size)
	ml.fill_value(gru.candidate_bias, 0)
}

reset_state :: proc(gru: Gru) {
	slice.zero(gru.state)
}

@(require_results)
forward :: proc(gru: Gru, input: ml.Tensor) -> (state: ml.Tensor) {
	hidden_size := gru.hidden_size

	state = ml.tensor(gru.state)

	// Update gate
	z        := ml.linear(input, gru.update_input_weight)
	z_hidden := ml.linear(state, gru.update_hidden_weight)
	z         = ml.add(z, z_hidden)
	z         = ml.add(z, gru.update_bias)
	z         = ml.sigmoid(z)

	// Reset gate
	r        := ml.linear(input, gru.reset_input_weight)
	r_hidden := ml.linear(state, gru.reset_hidden_weight)
	r         = ml.add(r, r_hidden)
	r         = ml.add(r, gru.reset_bias)
	r         = ml.sigmoid(r)

	// Candidate hidden state
	c        := ml.linear(input,            gru.candidate_input_weight)
	c_hidden := ml.linear(ml.mul(r, state), gru.candidate_hidden_weight)
	c         = ml.add(c, c_hidden)
	c         = ml.add(c, gru.candidate_bias)
	c         = ml.tanh(c)

	// Compute the new hidden state.
	ones := ml.zeros(.F32, {hidden_size})
	ml.fill_value(ones, 1)

	one_minus_z      := ml.sub(ones, z)
	new_hidden_state := ml.add(ml.mul(one_minus_z, state), ml.mul(z, c))

	state = new_hidden_state

	ml.get_data(state, gru.state)

	return
}

update :: proc(opt: ml.Optimizer, gru: Gru) {
	ml.update(opt, gru.update_input_weight)
	ml.update(opt, gru.update_hidden_weight)
	ml.update(opt, gru.update_bias)

	ml.update(opt, gru.reset_input_weight)
	ml.update(opt, gru.reset_hidden_weight)
	ml.update(opt, gru.reset_bias)

	ml.update(opt, gru.candidate_input_weight)
	ml.update(opt, gru.candidate_hidden_weight)
	ml.update(opt, gru.candidate_bias)
}