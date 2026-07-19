package mlp

import "base:builtin"

import "core:fmt"

import ml "../../"

Layer :: struct {
	weight: ml.Tensor,
	bias:   ml.Tensor,
}

Mlp :: struct {
	layers: []Layer,
	params: ml.Registry,
}

make :: proc(sizes: ..int, flags := ml.PARAMETER_DEFAULT_FLAGS, allocator := context.allocator) -> (mlp: Mlp) {
	context.allocator = allocator

	mlp.layers = builtin.make([]Layer, len(sizes) - 1)

	for i in 0 ..< len(mlp.layers) {
		mlp.layers[i].weight = ml.parameter_make(&mlp.params, "", fmt.tprintf("%d.weight", i), .F32, {sizes[i + 1], sizes[i]}, init=ml.Init_He{},           flags=flags)
		mlp.layers[i].bias   = ml.parameter_make(&mlp.params, "", fmt.tprintf("%d.bias",   i), .F32, {sizes[i + 1]},           init=ml.Init_Value{value=0}, flags=flags)
	}

	randomize(mlp)

	return
}

destroy :: proc(mlp: Mlp) {
	mlp := mlp
	ml.registry_destroy(&mlp.params)
	delete(mlp.layers)
}

parameters :: proc(mlp: Mlp, dst: ^ml.Registry, prefix := "") {
	mlp := mlp
	ml.registry_gather(dst, &mlp.params, prefix=prefix)
}

copy :: proc(dst, src: Mlp) {
	dst := dst
	src := src
	ml.registry_copy(&dst.params, &src.params)
}

randomize :: proc(mlp: Mlp) {
	mlp := mlp
	ml.registry_randomize(&mlp.params)
}

@(require_results)
forward :: proc(mlp: Mlp, input: ml.Tensor) -> (output: ml.Tensor) {
	output = input

	for layer, i in mlp.layers {
		output = ml.linear(output, layer.weight, bias=layer.bias)
		if i < len(mlp.layers) - 1 {
			output = ml.relu(output)
		}
	}

	return
}

update :: proc(opt: ^ml.Optimizer, mlp: Mlp) {
	mlp := mlp
	ml.registry_update(opt, &mlp.params)
}
