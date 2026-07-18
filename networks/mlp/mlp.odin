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
	params: [dynamic]ml.Parameter_Info,
}

make :: proc(sizes: ..int, allocator := context.allocator) -> (mlp: Mlp) {
	context.allocator = allocator

	mlp.layers = builtin.make([]Layer, len(sizes) - 1)

	for i in 0 ..< len(mlp.layers) {
		mlp.layers[i].weight = ml.parameter_make(&mlp.params, "", fmt.tprintf("%d.weight", i), .F32, {sizes[i + 1], sizes[i]}, init=ml.Init_He{})
		mlp.layers[i].bias   = ml.parameter_make(&mlp.params, "", fmt.tprintf("%d.bias",   i), .F32, {sizes[i + 1]}, init=ml.Init_Value{value=0})
	}

	randomize(mlp)

	return
}

destroy :: proc(mlp: Mlp) {
	mlp := mlp
	ml.registry_destroy(&mlp.params)
	delete(mlp.layers)
}

parameters :: proc(mlp: Mlp, prefix: string, list: ^[dynamic]ml.Parameter) {
	ml.registry_parameters(mlp.params[:], list, prefix=prefix)
}

copy :: proc(dst, src: Mlp) {
	ml.registry_copy(dst.params[:], src.params[:])
}

randomize :: proc(mlp: Mlp) {
	ml.registry_randomize(mlp.params[:])
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

update :: proc(opt: ^ml.Optimizer, mlp: Mlp) {
	ml.registry_update(opt, mlp.params[:])
}
