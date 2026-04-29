package machine_learning_network_mlp

import "base:builtin"

import ml "../../"

Layer :: struct {
	weight: ml.Tensor,
	bias:   ml.Tensor,
}

Mlp :: struct {
	layers: []Layer,
}

make :: proc(sizes: ..int, allocator := context.allocator) -> (mlp: Mlp) {
	mlp.layers = builtin.make([]Layer, len(sizes) - 1)

	for i in 0 ..< len(mlp.layers) {
		mlp.layers[i].weight = ml.make(.F32, {sizes[i + 1], sizes[i]})
		mlp.layers[i].bias   = ml.make(.F32, {sizes[i + 1]})
	}

	randomize(mlp)

	return
}

destroy :: proc(mlp: Mlp) {
	for layer in mlp.layers {
		ml.destroy(layer.weight)
		ml.destroy(layer.bias)
	}
	delete(mlp.layers)
}

copy :: proc(dst, src: Mlp) {
	for i in 0 ..< len(dst.layers) {
		ml.copy(dst.layers[i].weight, src.layers[i].weight)
		ml.copy(dst.layers[i].bias,   src.layers[i].bias)
	}
}

randomize :: proc(mlp: Mlp) {
	for i in 0 ..< len(mlp.layers) {
		input_size := mlp.layers[i].weight.shape[1]
		ml.he_initialization(mlp.layers[i].weight, input_size)
		ml.fill_value(mlp.layers[i].bias, 0)
	}
}

@(require_results)
forward :: proc(mlp: Mlp, input: ml.Tensor) -> (output: ml.Tensor) {
	output = input

	for layer, i in mlp.layers {
		output = ml.linear(output, layer.weight)
		output = ml.add(output, layer.bias)
		if i < len(mlp.layers) - 1 {
			output = ml.relu(output)
		}
	}

	return
}

update :: proc(opt: ml.Optimizer, mlp: Mlp) {
	for layer in mlp.layers {
		ml.update(opt, layer.weight)
		ml.update(opt, layer.bias)
	}
}